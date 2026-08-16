"""
The solvent mask and space-group symmetry: the order of the two operations.

docs/solvent-design.md calls this the one silent wrong answer in the design,
and until now nothing exercised it -- every other solvent test runs P1, where
symmetrize_sum is a no-op and the question does not arise.

The hazard: symmetrize_sum SUMS the density over the symmetry orbit. So

    mask(symmetrize(rho))     is a mask -- values in [0,1], solvent region of
                              the full unit cell contents        CORRECT
    symmetrize(mask(rho))     is a SUM OF MASKS -- values up to |orbit|, and
                              not a mask of anything              WRONG

Both run without error and both produce an F_mask. The difference shows up
only as a wrong answer, which is why it is pinned here.

Four checks:
  1. The pipeline matches the correct order, bit for bit.
  2. The wrong order is materially different -- and self-identifying, since
     its "mask" exceeds 1.
  3. The mask the pipeline builds is itself symmetric: invariant under every
     operation of the group. A solvent region that is not is not a solvent
     region for that space group.
  4. Symmetry and solvent compose without disturbing each other: with
     solvent=None the symmetric path is untouched.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("cctbx", reason="cctbx not installed (needed for the space group)")

from lunus.sf.cell_utils import orth_matrix
from lunus.sf.density_torch import splat_density
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.solvent_torch import (
    SolventModel, f_solvent, f_total, mask_occupancy, solvent_mask,
)
from lunus.sf.structure_factor_torch import (
    compute_fcalc, reciprocal_inv_d2, structure_factors_one_config,
)
from lunus.sf.symmetry_torch import build_grid_ops, symmetrize_sum

# Tetragonal, so a == b for the 4-fold, and Nw divisible by 4 for the 4_3
# screw translations.
SYMBOL = "P 43"
CELL = (24.0, 24.0, 32.0, 90.0, 90.0, 90.0)
GRID = (24, 24, 32)
DTYPE = torch.float64


def ops_from_cctbx(symbol):
    import cctbx.sgtbx as sgtbx
    sg = sgtbx.space_group_info(symbol=symbol.replace(" ", "")).group()
    rots = [np.array(o.r().as_double()).reshape(3, 3) for o in sg]
    trans = [np.array(o.t().as_double()) for o in sg]
    return rots, trans


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(0)
    n_atoms = 12
    M_np = orth_matrix(*CELL)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, GRID, M_np, cutoff=0.01,
        device="cpu", dtype=DTYPE,
    )
    idx = torch.zeros(n_atoms, dtype=torch.long)
    # Kept away from the cell edges so the orbit copies stay distinct.
    frac = torch.tensor(0.15 + 0.5 * rng.random((n_atoms, 3)), dtype=DTYPE)
    rots, trans = ops_from_cctbx(SYMBOL)
    grid_ops = build_grid_ops(rots, trans, GRID)
    assert grid_ops, "P 43 must contribute non-identity operations"

    s = dict(
        frac=frac, element_idx=idx, occ=torch.ones(n_atoms, dtype=DTYPE),
        atom_A=A[idx], atom_lam=lam[idx], elem_offsets=offsets,
        atom_radius_ang=radius[idx], grid_shape=GRID,
        orth=torch.tensor(M_np, dtype=DTYPE),
        cell_volume=float(np.linalg.det(M_np)), taper_width=taper_w,
        grid_ops=grid_ops,
        hkl=torch.tensor(
            [[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
             for l in range(-3, 4) if (h, k, l) != (0, 0, 0)],
            dtype=torch.long),
    )
    s["density_raw"] = splat_density(
        s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], GRID, s["orth"],
        s["taper_width"], compile_core=False,
    )
    s["density_sym"] = symmetrize_sum(s["density_raw"], grid_ops)
    s["cutoff"] = 0.01 * float(s["density_sym"].max())
    s["model"] = SolventModel(cutoff=s["cutoff"], taper_width=0.5 * s["cutoff"])
    return s


def _pipeline(s, solvent):
    return structure_factors_one_config(
        s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
        s["cell_volume"], s["hkl"], s["taper_width"],
        grid_ops=s["grid_ops"], compile_core=False, solvent=solvent,
    )


def test_pipeline_masks_after_symmetrizing(system):
    """Check 1: the pipeline is the correct order, exactly."""
    s = system
    rho = s["density_sym"]
    F_protein = compute_fcalc(rho, s["cell_volume"], s["hkl"], s["orth"])
    # Through the model's own mask construction rather than solvent_mask
    # directly, so this keeps testing the ORDER under whatever the model does
    # to the density first -- the mask_blur added later would otherwise have
    # made this a comparison against a mask the pipeline no longer builds.
    mask = s["model"].build_mask(rho, s["orth"])
    F_mask = f_solvent(mask, s["cell_volume"], s["hkl"], s["orth"])
    expected = f_total(
        F_protein, F_mask, reciprocal_inv_d2(s["orth"], s["hkl"]),
        k_sol=s["model"].k_sol, b_sol=s["model"].b_sol, hkl=s["hkl"])

    assert torch.equal(_pipeline(s, s["model"]), expected)


def test_the_wrong_order_is_different_and_self_identifying(system):
    """Check 2: masking before symmetrizing produces a sum of masks.

    Where the error goes is worth knowing, because it is not where one might
    guess. The summed mask is roughly (|orbit| - 1) + the correct mask, and a
    constant offset lands entirely in F(000) -- measured here, F(000) is 4.3x
    too large against an orbit of 4. What survives for h != 0 is only the
    structure where orbit copies overlap: 22% on this system, and less on a
    sparser one.

    So the reliable detector is the GRID, not F: a mask that exceeds 1 is
    wrong, whatever its transform happens to look like.
    """
    s = system
    wrong = symmetrize_sum(
        solvent_mask(s["density_raw"], s["cutoff"], 0.5 * s["cutoff"]),
        s["grid_ops"])
    order = len(s["grid_ops"]) + 1

    assert float(wrong.max()) > 1.0
    assert float(wrong.max()) == pytest.approx(order, rel=0.05)

    right = solvent_mask(s["density_sym"], s["cutoff"], 0.5 * s["cutoff"])
    assert float(right.max()) <= 1.0

    F_wrong = f_solvent(wrong, s["cell_volume"], s["hkl"], s["orth"])
    F_right = f_solvent(right, s["cell_volume"], s["hkl"], s["orth"])
    rel_diff = float((F_wrong - F_right).abs().mean() / F_right.abs().mean())
    assert rel_diff > 0.1, rel_diff

    # F(000) is the whole cell's mean, so the constant offset shows up there
    # at close to the full orbit order.
    h000 = torch.tensor([[0, 0, 0]], dtype=torch.long)
    f000_wrong = f_solvent(wrong, s["cell_volume"], h000, s["orth"]).abs().item()
    f000_right = f_solvent(right, s["cell_volume"], h000, s["orth"]).abs().item()
    assert f000_wrong / f000_right > 0.5 * order


def test_the_mask_has_the_symmetry_of_the_space_group(system):
    """Check 3: the solvent region must be invariant under every operation of
    the group. A sum over the orbit has that property; a mask built from the
    unsymmetrized density does not, which is the same bug seen from the other
    side."""
    s = system
    mask = solvent_mask(s["density_sym"], s["cutoff"], 0.5 * s["cutoff"])
    for op in s["grid_ops"]:
        assert torch.allclose(op.apply(mask), mask, rtol=1e-12, atol=1e-12)

    unsym = solvent_mask(s["density_raw"], s["cutoff"], 0.5 * s["cutoff"])
    assert not all(
        torch.allclose(op.apply(unsym), unsym, rtol=1e-6, atol=1e-8)
        for op in s["grid_ops"])


def test_solvent_none_leaves_the_symmetric_path_alone(system):
    """Check 4: the two features compose. With solvent off, the symmetry path
    is bit-identical to what it was before solvent existed."""
    s = system
    expected = compute_fcalc(
        s["density_sym"], s["cell_volume"], s["hkl"], s["orth"])
    assert torch.equal(_pipeline(s, None), expected)


def test_occupancy_is_plausible_under_symmetry(system):
    """The diagnostic has to stay meaningful when the cell contents are
    symmetry-expanded: four copies of the asymmetric unit fill more of the
    cell, so occupancy must fall, but not to zero."""
    s = system
    occ_sym = float(mask_occupancy(
        solvent_mask(s["density_sym"], s["cutoff"], 0.5 * s["cutoff"])))
    occ_raw = float(mask_occupancy(
        solvent_mask(s["density_raw"], s["cutoff"], 0.5 * s["cutoff"])))
    assert 0.0 < occ_sym < occ_raw < 1.0
