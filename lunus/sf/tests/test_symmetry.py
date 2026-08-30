"""
Validates symmetry_torch.symmetrize_sum against gemmi's Grid::symmetrize_sum()
itself -- the thing it is meant to reproduce -- plus two internal consistency
checks.

Four checks:
  1. GEMMI PARITY: torch symmetrize_sum vs gemmi symmetrize_sum on the same
     random grid, for several space groups. This is the check that matters;
     it pins the convention (which direction the operation is applied, and
     whether the identity is included) against the real implementation
     rather than against a re-reading of its source.
  2. FAST vs GATHER: the signed-permutation fast path (permute/flip/roll)
     against the general advanced-indexing path, which are entirely
     different code with no shared arithmetic.
  3. ORBIT INVARIANCE: the symmetrized grid must itself be symmetric, i.e.
     unchanged by any single operation. A sum over a group orbit has this
     property; most plausible index bugs do not.
  4. AUTOGRAD: analytic gradient of sum(w * symmetrize_sum(g)) w.r.t. g
     against finite differences, confirming the gathers/rolls carry
     gradients correctly.

Requires gemmi for check 1 (skipped with a message if unavailable); the
rest need only torch.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("cctbx", reason="cctbx not installed (needed to build the space-group ops)")

from lunus.sf.symmetry_torch import build_grid_ops, symmetrize_sum

# (space group, grid shape) pairs whose grids satisfy the divisibility /
# equal-axis conditions the operations require
CASES = [
    ("P 1", (5, 7, 9)),
    ("P 43", (4, 4, 8)),
    ("P 21 21 21", (6, 6, 6)),
    ("P 61", (6, 6, 12)),
    ("I 2 3", (8, 8, 8)),
]


def ops_from_cctbx(symbol):
    import cctbx.sgtbx as sgtbx
    sg = sgtbx.space_group_info(symbol=symbol.replace(" ", "")).group()
    rots = [np.array(o.r().as_double()).reshape(3, 3) for o in sg]
    trans = [np.array(o.t().as_double()) for o in sg]
    return rots, trans, sg.order_z()


@pytest.mark.parametrize("symbol,shape", CASES)
def test_gemmi_parity(symbol, shape):
    """torch symmetrize_sum vs gemmi's own symmetrize_sum.

    This is the check that matters: it pins the convention -- which direction
    the operation is applied, and whether the identity is included -- against
    the real implementation rather than against a re-reading of its source.
    Measured max relative error ~9e-08, i.e. float32 rounding.
    """
    gemmi = pytest.importorskip("gemmi", reason="gemmi not installed")

    rng = np.random.default_rng(0)
    data = rng.random(shape).astype(np.float32)

    g = gemmi.FloatGrid()
    g.set_size(*shape)
    g.spacegroup = gemmi.find_spacegroup_by_name(symbol)
    g.set_unit_cell(gemmi.UnitCell(10.0, 10.0, 20.0, 90, 90, 90))
    np.array(g, copy=False)[:] = data
    g.symmetrize_sum()
    ref = np.array(g, copy=True).astype(np.float64)

    rots, trans, order = ops_from_cctbx(symbol)
    ops = build_grid_ops(rots, trans, shape)
    got = symmetrize_sum(torch.tensor(data, dtype=torch.float64), ops).numpy()

    scale = max(np.max(np.abs(ref)), 1e-12)
    err = np.max(np.abs(got - ref)) / scale
    print(f"\n{symbol:12s} grid={str(shape):12s} order={order}  "
          f"n_ops(non-id)={len(ops)}  max rel err = {err:.2e}")
    assert err < 1e-6, f"{symbol}: torch symmetrize_sum differs from gemmi by {err:.2e}"


@pytest.mark.parametrize("symbol,shape", CASES)
def test_fast_path_matches_gather(symbol, shape):
    """The signed-permutation fast path (permute/flip/roll) against the general
    advanced-indexing path. Entirely different code with no shared arithmetic,
    so agreement is meaningful; they should be bit-identical.
    """
    rots, trans, _ = ops_from_cctbx(symbol)
    ops = build_grid_ops(rots, trans, shape)
    if not ops:
        pytest.skip(f"{symbol} has no non-identity ops, nothing to compare")

    grid = torch.rand(shape, dtype=torch.float64)
    worst, n_fast = 0.0, 0
    for op in ops:
        if not op.fast:
            continue
        n_fast += 1
        worst = max(worst, float((op._apply_fast(grid) - op._apply_gather(grid)).abs().max()))

    print(f"\n{symbol:12s} {n_fast}/{len(ops)} ops on the fast path, max abs diff = {worst:.2e}")
    if n_fast == 0:
        pytest.skip(f"{symbol} has no ops on the fast path")
    assert worst < 1e-12, f"{symbol}: fast path differs from gather path by {worst:.2e}"


@pytest.mark.parametrize("symbol,shape", CASES)
def test_orbit_invariance(symbol, shape):
    """A symmetrized grid must itself be symmetric -- unchanged by any single
    operation. A sum over a group orbit has this property; most plausible
    index bugs do not.
    """
    rots, trans, _ = ops_from_cctbx(symbol)
    ops = build_grid_ops(rots, trans, shape)
    if not ops:
        pytest.skip(f"{symbol} is P1, trivially invariant")

    sym = symmetrize_sum(torch.rand(shape, dtype=torch.float64), ops)
    worst = max(float((op.apply(sym) - sym).abs().max()) for op in ops)
    rel = worst / float(sym.abs().max())
    print(f"\n{symbol:12s} max rel deviation = {rel:.2e}")
    assert rel < 1e-12, f"{symbol}: symmetrized grid is not invariant ({rel:.2e})"


def test_autograd_matches_finite_difference():
    """Gradients must flow back through the symmetrization, so that an atom
    receives the contributions of all of its symmetry images. Measured 8e-09.
    """
    symbol, shape = "P 43", (4, 4, 8)
    rots, trans, _ = ops_from_cctbx(symbol)
    ops = build_grid_ops(rots, trans, shape)

    torch.manual_seed(0)
    w = torch.rand(shape, dtype=torch.float64)
    g0 = torch.rand(shape, dtype=torch.float64)

    def loss_of(g):
        return (w * symmetrize_sum(g, ops)).sum()

    g = g0.clone().requires_grad_(True)
    loss_of(g).backward()
    analytic = g.grad.clone()

    eps = 1e-6
    idx = [(0, 0, 0), (1, 2, 3), (3, 1, 5), (2, 2, 7)]
    worst = 0.0
    for p in idx:
        gp, gm = g0.clone(), g0.clone()
        gp[p] += eps
        gm[p] -= eps
        fd = (loss_of(gp) - loss_of(gm)).item() / (2 * eps)
        an = analytic[p].item()
        rel = abs(an - fd) / max(abs(fd), 1e-12)
        worst = max(worst, rel)
        print(f"  voxel {p}: autograd={an:.10f}  finite-diff={fd:.10f}  rel.err={rel:.2e}")
    print(f"  max relative gradient error = {worst:.2e}")
    assert worst < 1e-7, f"autograd disagrees with finite differences by {worst:.2e}"


RAW_SHAPES = [(300, 300, 144), (298, 302, 141), (100, 60, 60), (75, 75, 50)]


@pytest.mark.parametrize("symbol,_shape", CASES)
@pytest.mark.parametrize("raw", RAW_SHAPES)
def test_grid_adjustment(symbol, _shape, raw):
    """adjust_grid_for_symmetry must always produce a grid build_grid_ops
    accepts, never shrink an axis, and leave an already-valid grid alone.

    grid_shape_for_resolution() rounds each axis independently, so it does not
    on its own guarantee the constraints the operations impose (a 4-fold in the
    ab plane needs Nu == Nv; a 3/4 screw translation needs Nw divisible by 4).
    """
    from lunus.sf.symmetry_torch import adjust_grid_for_symmetry

    rots, trans, _ = ops_from_cctbx(symbol)
    fixed = adjust_grid_for_symmetry(raw, rots, trans)
    print(f"\n{symbol:12s} {str(raw):16s} -> {str(fixed):16s}"
          f"{'' if raw == fixed else '  (adjusted)'}")

    assert all(f >= r for f, r in zip(fixed, raw)), (
        f"{symbol}: adjustment shrank an axis, {raw} -> {fixed}"
    )
    try:
        build_grid_ops(rots, trans, fixed)
    except ValueError as e:
        pytest.fail(f"{symbol}: adjusted grid {fixed} was rejected by build_grid_ops: {e}")
    assert adjust_grid_for_symmetry(fixed, rots, trans) == fixed, (
        f"{symbol}: adjustment is not idempotent on {fixed}"
    )
