"""
Checks structure_factors_batch: N configurations in one call, with gradients
reaching all N.

The claims worth pinning are equivalence claims, because batching is an
efficiency feature that must not change any number:

  1. Batched values equal the per-configuration loop, exactly.
  2. Batched gradients equal the loop's, exactly, for EVERY member -- not
     just in aggregate. A bug that mixed up which configuration a gradient
     belonged to would leave the total unchanged.
  3. Checkpointing changes neither values nor gradients. It only decides
     whether the splat's intermediates are retained or recomputed.
  4. Every member actually receives gradient. A member silently detached
     from the graph is the failure mode that matters for guidance, and it
     would not show up in a norm over the whole batch.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from lunus.sf.cell_utils import grid_shape_for_resolution, orth_matrix
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.structure_factor_torch import (
    mean_and_diffuse,
    structure_factors_batch,
    structure_factors_one_config,
)

DTYPE = torch.float64          # tight comparisons; production runs use float32
N_CONFIGS = 4
N_ATOMS = 60
ELEMENTS = ["C", "N", "O"]


@pytest.fixture(scope="module")
def system():
    it92 = pytest.importorskip(
        "lunus.sf.elements", reason="needs gemmi or cctbx for coefficients"
    )
    try:
        coeffs = it92.it92_coefficients(ELEMENTS)
    except ImportError as e:
        pytest.skip(str(e))

    a = b = c = 30.0
    M_np = orth_matrix(a, b, c, 90.0, 90.0, 90.0)
    M = torch.tensor(M_np, dtype=DTYPE)
    cell_volume = abs(float(np.linalg.det(M_np)))
    grid_shape = grid_shape_for_resolution(a, b, c, 3.0, 1.5)

    elem_A, elem_lam, elem_offsets, elem_radius_ang, taper_width, _ = (
        build_element_kernels_torch(
            ELEMENTS, coeffs, 30.0, blur=0.0,
            grid_shape=grid_shape, orth_matrix_np=M_np,
            cutoff=1e-3, device="cpu", dtype=DTYPE,
        )
    )

    g = torch.Generator().manual_seed(0)
    element_idx = torch.randint(0, len(ELEMENTS), (N_ATOMS,), generator=g)
    base = torch.rand(N_ATOMS, 3, generator=g, dtype=DTYPE)
    coords = torch.stack(
        [base + 0.02 * torch.randn(N_ATOMS, 3, generator=g, dtype=DTYPE)
         for _ in range(N_CONFIGS)]
    )

    hh, kk, ll = torch.meshgrid(*[torch.arange(-3, 4)] * 3, indexing="ij")
    hkl = torch.stack([hh, kk, ll], dim=-1).reshape(-1, 3)
    hkl = hkl[hkl.abs().sum(dim=-1) > 0]

    return dict(
        coords=coords, element_idx=element_idx,
        occ=torch.ones(N_ATOMS, dtype=DTYPE),
        atom_A=elem_A[element_idx], atom_lam=elem_lam[element_idx],
        atom_radius_ang=elem_radius_ang[element_idx],
        elem_offsets=elem_offsets, grid_shape=grid_shape,
        M=M, cell_volume=cell_volume, hkl=hkl, taper_width=taper_width,
    )


def _common(s):
    return dict(
        element_idx=s["element_idx"], occ=s["occ"],
        atom_A=s["atom_A"], atom_lam=s["atom_lam"],
        elem_offsets=s["elem_offsets"], atom_radius_ang=s["atom_radius_ang"],
        grid_shape=s["grid_shape"], orth_matrix=s["M"],
        cell_volume=s["cell_volume"], hkl=s["hkl"],
        taper_width=s["taper_width"], blur=0.0, compile_core=False,
    )


def _loop(s, coords):
    kw = _common(s)
    return torch.stack([structure_factors_one_config(fc, **kw) for fc in coords], dim=0)


def _batch(s, coords, use_checkpoint):
    return structure_factors_batch(coords, use_checkpoint=use_checkpoint, **_common(s))


def _loss(F):
    """A stand-in for a guidance loss: depends on the ensemble observables,
    hence on every member, through both <F> and the diffuse term."""
    mean_F, diffuse = mean_and_diffuse(F)
    return diffuse.pow(2).sum() + mean_F.abs().pow(2).sum()


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_batched_values_match_the_loop(system, use_checkpoint):
    s = system
    coords = s["coords"].clone()
    ref = _loop(s, coords)
    got = _batch(s, coords, use_checkpoint)

    assert got.shape == (N_CONFIGS, s["hkl"].shape[0])
    worst = (got - ref).abs().max().item()
    print(f"\nuse_checkpoint={use_checkpoint}: max |batched - loop| = {worst:.3e}")
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_batched_gradients_match_the_loop_member_by_member(system, use_checkpoint):
    """Per-member equality, not just an aggregate norm: a bug that routed
    member i's gradient to member j would preserve the total."""
    s = system

    ref_coords = s["coords"].clone().requires_grad_(True)
    _loss(_loop(s, ref_coords)).backward()
    ref_grad = ref_coords.grad.clone()

    got_coords = s["coords"].clone().requires_grad_(True)
    _loss(_batch(s, got_coords, use_checkpoint)).backward()
    got_grad = got_coords.grad.clone()

    for i in range(N_CONFIGS):
        d = (got_grad[i] - ref_grad[i]).abs().max().item()
        print(f"\n  member {i}: max |d(grad)| = {d:.3e}, "
              f"|grad| = {ref_grad[i].norm().item():.6e}")
        torch.testing.assert_close(
            got_grad[i], ref_grad[i], rtol=0, atol=0,
            msg=f"member {i}: batched gradient differs from the loop",
        )


def test_checkpointing_is_numerically_neutral(system):
    """Retained vs recomputed must be bit-identical -- it is a memory/time
    trade, not an approximation."""
    s = system

    a_coords = s["coords"].clone().requires_grad_(True)
    _loss(_batch(s, a_coords, False)).backward()

    b_coords = s["coords"].clone().requires_grad_(True)
    _loss(_batch(s, b_coords, True)).backward()

    worst = (a_coords.grad - b_coords.grad).abs().max().item()
    print(f"\nmax |grad(retained) - grad(checkpointed)| = {worst:.3e}")
    torch.testing.assert_close(b_coords.grad, a_coords.grad, rtol=0, atol=0)


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_every_member_receives_gradient(system, use_checkpoint):
    """The failure mode guidance actually cares about: a member silently
    detached from the graph, which an aggregate norm would hide."""
    s = system
    coords = s["coords"].clone().requires_grad_(True)
    _loss(_batch(s, coords, use_checkpoint)).backward()

    assert coords.grad is not None, "no gradient reached the batch at all"
    per_member = coords.grad.reshape(N_CONFIGS, -1).norm(dim=1)
    print(f"\nper-member gradient norms: {[f'{v:.4e}' for v in per_member.tolist()]}")
    for i, norm in enumerate(per_member.tolist()):
        assert norm > 0.0, f"member {i} received no gradient"


def test_rejects_mismatched_atom_count(system):
    s = system
    bad = s["coords"][:, : N_ATOMS - 1, :].clone()
    with pytest.raises(ValueError, match="all configurations must share one atom set"):
        _batch(s, bad, False)


def test_rejects_unbatched_input(system):
    """A single (n_atoms, 3) configuration is a common mistake; it must be a
    clear error rather than being silently read as a batch of n_atoms
    one-atom configurations."""
    s = system
    with pytest.raises(ValueError, match=r"must be \(n_configs, n_atoms, 3\)"):
        _batch(s, s["coords"][0].clone(), False)


# --------------------------------------------------------------------------
# per-configuration occupancies

def _occ_system(n_atoms=40, n_configs=3, grid=(24, 24, 24), seed=0):
    """Shared setup for the occupancy tests: kernels, coordinates, hkl."""
    import numpy as np
    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.elements import IT92_COEFFS
    from lunus.sf.kernel_torch import build_element_kernels_torch

    cell = (20.0, 22.0, 24.0, 90.0, 90.0, 90.0)
    dtype = torch.float64
    M_np = orth_matrix(*cell)
    rng = np.random.default_rng(seed)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, grid, M_np, cutoff=0.01,
        device="cpu", dtype=dtype)
    idx = torch.zeros(n_atoms, dtype=torch.long)
    return dict(
        frac=torch.tensor(rng.random((n_configs, n_atoms, 3)), dtype=dtype),
        element_idx=idx,
        occ2=torch.tensor(rng.uniform(0.2, 1.0, (n_configs, n_atoms)),
                          dtype=dtype),
        kw=dict(atom_A=A[idx], atom_lam=lam[idx], elem_offsets=offsets,
                atom_radius_ang=radius[idx], grid_shape=grid,
                orth_matrix=torch.tensor(M_np, dtype=dtype),
                cell_volume=float(np.linalg.det(M_np)),
                hkl=torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                  [2, 1, 0], [3, 2, 1], [1, 1, 1]],
                                 dtype=torch.long),
                taper_width=taper_w, compile_core=False))


def test_per_configuration_occupancy_matches_calling_one_config():
    """(N, n_atoms) occ must be exactly the per-member call, not an average or
    a broadcast of row 0 -- both of which would still produce plausible
    numbers."""
    from lunus.sf.structure_factor_torch import (
        structure_factors_batch, structure_factors_one_config)

    s = _occ_system()
    got = structure_factors_batch(s["frac"], s["element_idx"], s["occ2"],
                                  **s["kw"])
    expected = torch.stack([
        structure_factors_one_config(s["frac"][i], s["element_idx"],
                                     s["occ2"][i], **s["kw"])
        for i in range(s["frac"].shape[0])])
    assert torch.equal(got, expected)


def test_shared_occupancy_is_unchanged_by_the_per_configuration_path():
    """(n_atoms,) must keep meaning what it meant before (N, n_atoms) existed."""
    from lunus.sf.structure_factor_torch import (
        structure_factors_batch, structure_factors_one_config)

    s = _occ_system()
    occ = torch.full((s["occ2"].shape[1],), 0.7, dtype=torch.float64)
    got = structure_factors_batch(s["frac"], s["element_idx"], occ, **s["kw"])
    expected = torch.stack([
        structure_factors_one_config(s["frac"][i], s["element_idx"], occ,
                                     **s["kw"])
        for i in range(s["frac"].shape[0])])
    assert torch.equal(got, expected)


def test_occupancy_gradient_matches_a_finite_difference():
    from lunus.sf.structure_factor_torch import (
        mean_and_diffuse, structure_factors_batch)

    s = _occ_system()

    def loss(o):
        F = structure_factors_batch(s["frac"], s["element_idx"], o, **s["kw"])
        return mean_and_diffuse(F)[1].sum()

    o = s["occ2"].clone().requires_grad_(True)
    loss(o).backward()

    eps, i, j = 1e-6, 2, 7
    up, down = s["occ2"].clone(), s["occ2"].clone()
    up[i, j] += eps
    down[i, j] -= eps
    fd = float((loss(up) - loss(down)) / (2 * eps))
    assert float(o.grad[i, j]) == pytest.approx(fd, rel=1e-6)


def test_occupancy_gradients_do_not_leak_between_configurations():
    """Member i's occupancies must not move member j's structure factors."""
    from lunus.sf.structure_factor_torch import structure_factors_batch

    s = _occ_system()
    o = s["occ2"].clone().requires_grad_(True)
    F = structure_factors_batch(s["frac"], s["element_idx"], o, **s["kw"])
    F[0].abs().pow(2).sum().backward()

    assert float(o.grad[0].abs().sum()) > 0.0
    assert float(o.grad[1:].abs().sum()) == 0.0


def test_checkpointing_gives_the_same_occupancy_gradient():
    from lunus.sf.structure_factor_torch import structure_factors_batch

    s = _occ_system()
    grads = []
    for use_checkpoint in (False, True):
        o = s["occ2"].clone().requires_grad_(True)
        F = structure_factors_batch(s["frac"], s["element_idx"], o,
                                    use_checkpoint=use_checkpoint, **s["kw"])
        F.abs().pow(2).sum().backward()
        grads.append(o.grad.clone())
    assert torch.allclose(grads[0], grads[1], rtol=0, atol=1e-12)


def test_occupancy_shape_is_checked():
    from lunus.sf.structure_factor_torch import structure_factors_batch

    s = _occ_system()
    n_configs, n_atoms = s["occ2"].shape
    for bad in (torch.ones(n_atoms + 1, dtype=torch.float64),
                torch.ones((n_configs + 1, n_atoms), dtype=torch.float64),
                torch.ones((n_configs, n_atoms, 1), dtype=torch.float64)):
        with pytest.raises(ValueError, match="occ"):
            structure_factors_batch(s["frac"], s["element_idx"], bad, **s["kw"])
