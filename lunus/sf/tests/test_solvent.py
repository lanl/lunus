"""
Bulk solvent: the mask, the combination formula, and the promise that
solvent=None changes nothing.

Stage 1 of docs/solvent-design.md. Deliberately NOT here: parity against
gemmi's SolventMasker (a geometric mask, so agreement is at the level of the
model difference, not machine precision), R-factor against experimental
amplitudes, and the diffuse convergence study. Those need data and a scaling
fit; these need only torch.

The checks:
  1. solvent=None is bit-identical to the pre-solvent code path.
  2. The mask is 1 in vacuum, 0 inside atoms, and smooth in between.
  3. Occupancy detects the failure mode the atom selection cannot: a cell
     already full of density gives an empty mask and a silent no-op.
  4. k_sol=0 collapses f_total to k_overall * F_protein, exactly.
  5. The b_sol falloff matches exp(-b*s^2/4) evaluated by hand -- the
     convention this formula invites getting wrong by a factor of four.
  6. detach_mask keeps the solvent envelope out of the coordinate gradient
     while F_protein's own gradient still flows.
  7. Per-configuration masks reach the diffuse observable and a shared mask
     does not -- the distinction the ensemble semantics turn on.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from lunus.sf.density_torch import splat_density
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.cell_utils import orth_matrix
from lunus.sf.solvent_torch import (
    SolventModel, f_solvent, f_total, mask_occupancy, shell_voxels, solvent_mask,
)
from lunus.sf.structure_factor_torch import (
    compute_fcalc, mean_and_diffuse, reciprocal_inv_d2,
    structure_factors_batch, structure_factors_one_config,
)

CELL = (20.0, 22.0, 24.0, 90.0, 90.0, 90.0)
GRID = (24, 24, 24)
DTYPE = torch.float64          # exact comparisons, no float32 slack


def _system(n_atoms=6, seed=0):
    """A few carbons in a small cell, with everything the pipeline needs."""
    rng = np.random.default_rng(seed)
    M_np = orth_matrix(*CELL)
    elements = ["C"] * n_atoms
    A, lam, offsets, radius, taper_w, e2i = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, GRID, M_np, cutoff=0.01,
        device="cpu", dtype=DTYPE,
    )
    frac = torch.tensor(rng.random((n_atoms, 3)), dtype=DTYPE)
    element_idx = torch.zeros(n_atoms, dtype=torch.long)
    occ = torch.ones(n_atoms, dtype=DTYPE)
    hkl = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0], [3, 2, 1]],
        dtype=torch.long,
    )
    return dict(
        frac=frac, element_idx=element_idx, occ=occ,
        atom_A=A[element_idx], atom_lam=lam[element_idx],
        elem_offsets=offsets, atom_radius_ang=radius[element_idx],
        grid_shape=GRID, orth=torch.tensor(M_np, dtype=DTYPE),
        cell_volume=float(np.linalg.det(M_np)), hkl=hkl, taper_width=taper_w,
    )


def _sf(sys, **kw):
    return structure_factors_one_config(
        sys["frac"], sys["element_idx"], sys["occ"],
        sys["atom_A"], sys["atom_lam"], sys["elem_offsets"],
        sys["atom_radius_ang"], sys["grid_shape"], sys["orth"],
        sys["cell_volume"], sys["hkl"], sys["taper_width"],
        compile_core=False, **kw
    )


def _density(sys):
    return splat_density(
        sys["frac"], sys["element_idx"], sys["occ"],
        sys["atom_A"], sys["atom_lam"], sys["elem_offsets"],
        sys["atom_radius_ang"], sys["grid_shape"], sys["orth"],
        sys["taper_width"], compile_core=False,
    )


# --------------------------------------------------------------------------
# 1. solvent=None must not perturb anything

def test_solvent_none_is_bit_identical():
    """The promise the argument was added under: no solvent means the old code
    path, not the new one multiplied by zero. Compared against the pipeline
    assembled by hand, so this would catch the solvent branch reordering or
    re-scaling the protein term even when solvent is off."""
    sys = _system()
    expected = compute_fcalc(
        _density(sys), sys["cell_volume"], sys["hkl"], sys["orth"], blur=0.0)
    got = _sf(sys, solvent=None)
    assert torch.equal(got, expected)


# --------------------------------------------------------------------------
# 2-3. the mask itself

def test_mask_is_one_in_vacuum_zero_in_atoms():
    sys = _system()
    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())
    mask = solvent_mask(rho, cutoff, 0.25 * cutoff)

    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    # The densest voxel is inside an atom: solid protein.
    assert float(mask.flatten()[int(rho.argmax())]) == pytest.approx(0.0)
    # The emptiest is vacuum: solid solvent.
    assert float(mask.flatten()[int(rho.argmin())]) == pytest.approx(1.0)
    # And the transition is actually resolved rather than a step.
    assert shell_voxels(mask) > 0


def test_occupancy_catches_the_silent_no_op():
    """A few atoms in a big cell leave most of it solvent. Flood the same cell
    with density and the mask empties out -- F_mask ~ 0, no error, no symptom.
    That is the case the atom selection cannot detect and occupancy can."""
    sys = _system()
    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())

    sparse = mask_occupancy(solvent_mask(rho, cutoff, 0.25 * cutoff))
    assert 0.5 < float(sparse) <= 1.0

    flooded = mask_occupancy(
        solvent_mask(rho + 10.0 * cutoff, cutoff, 0.25 * cutoff))
    assert float(flooded) == pytest.approx(0.0)

    F_mask = f_solvent(
        solvent_mask(rho + 10.0 * cutoff, cutoff, 0.25 * cutoff),
        sys["cell_volume"], sys["hkl"], sys["orth"])
    assert float(F_mask.abs().max()) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 4-5. the combination formula

def test_k_sol_zero_collapses_to_protein():
    sys = _system()
    F_p = _sf(sys, solvent=None)
    F_m = torch.randn_like(F_p)
    inv_d2 = reciprocal_inv_d2(sys["orth"], sys["hkl"])

    assert torch.equal(f_total(F_p, F_m, inv_d2, k_sol=0.0), F_p)
    got = f_total(F_p, F_m, inv_d2, k_sol=0.0, k_overall=2.5)
    assert torch.allclose(got, 2.5 * F_p, rtol=0, atol=1e-12)


def test_b_sol_falloff_matches_the_convention_by_hand():
    """exp(-b_sol * s^2 / 4), the same convention as compute_fcalc's blur --
    the factor of four is the classic way to get this wrong."""
    sys = _system()
    F_p = _sf(sys, solvent=None)
    F_m = torch.ones_like(F_p)
    inv_d2 = reciprocal_inv_d2(sys["orth"], sys["hkl"])

    k_sol, b_sol = 0.35, 46.0
    got = f_total(F_p, F_m, inv_d2, k_sol=k_sol, b_sol=b_sol)
    by_hand = F_p + torch.tensor(
        [k_sol * math.exp(-b_sol * float(s2) / 4.0) for s2 in inv_d2],
        dtype=F_p.real.dtype).to(F_p.dtype) * F_m
    assert torch.allclose(got, by_hand, rtol=1e-12, atol=1e-14)


def test_u_aniso_requires_hkl():
    sys = _system()
    F_p = _sf(sys, solvent=None)
    inv_d2 = reciprocal_inv_d2(sys["orth"], sys["hkl"])
    with pytest.raises(ValueError, match="requires hkl"):
        f_total(F_p, F_p, inv_d2, u_aniso=torch.eye(3, dtype=DTYPE))


# --------------------------------------------------------------------------
# 6. what the mask is allowed to differentiate

def test_detached_mask_carries_no_gradient():
    """The mask tensor itself: detached means autograd cannot reach it, which
    is what keeps the solvent envelope out of the gradient of everything the
    density depends on -- B in particular, once B is differentiable."""
    sys = _system()
    frac = sys["frac"].clone().requires_grad_(True)
    rho = splat_density(
        frac, sys["element_idx"], sys["occ"],
        sys["atom_A"], sys["atom_lam"], sys["elem_offsets"],
        sys["atom_radius_ang"], sys["grid_shape"], sys["orth"],
        sys["taper_width"], compile_core=False,
    )
    cutoff = 0.1 * float(rho.detach().max())
    common = dict(cutoff=cutoff, taper_width=0.25 * cutoff)
    args = (rho, torch.zeros(len(sys["hkl"]), dtype=torch.complex128),
            sys["cell_volume"], sys["hkl"], sys["orth"])

    assert rho.requires_grad
    _, mask_detached = SolventModel(detach_mask=True, **common).apply(*args)
    _, mask_live = SolventModel(detach_mask=False, **common).apply(*args)
    assert not mask_detached.requires_grad
    assert mask_live.requires_grad


def test_detach_mask_removes_the_solvent_path_from_the_gradient():
    """Probed with a LINEAR functional of F, so that the solvent term's effect
    on the VALUE (which is real, and stays) is separated from its effect on the
    gradient (which detaching removes). Under detach the solvent term is a
    constant in frac_coords, so d/dfrac is exactly the no-solvent gradient; a
    |F|^2 loss would not show this, since F itself has moved."""
    sys = _system()
    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())
    common = dict(cutoff=cutoff, taper_width=0.25 * cutoff, k_sol=0.35)

    def grad(**kw):
        frac = sys["frac"].clone().requires_grad_(True)
        F = structure_factors_one_config(
            frac, sys["element_idx"], sys["occ"],
            sys["atom_A"], sys["atom_lam"], sys["elem_offsets"],
            sys["atom_radius_ang"], sys["grid_shape"], sys["orth"],
            sys["cell_volume"], sys["hkl"], sys["taper_width"],
            compile_core=False, **kw
        )
        F.real.sum().backward()
        return frac.grad.clone()

    g_none = grad(solvent=None)
    g_detached = grad(solvent=SolventModel(detach_mask=True, **common))
    g_live = grad(solvent=SolventModel(detach_mask=False, **common))

    # The solvent term does change the VALUE either way...
    F_none = _sf(sys, solvent=None)
    F_det = _sf(sys, solvent=SolventModel(detach_mask=True, **common))
    assert not torch.allclose(F_none, F_det)
    # ...but detached it is a constant, so the coordinate gradient is untouched.
    assert torch.allclose(g_none, g_detached, rtol=1e-10, atol=1e-12)
    # Undetached, the mask carries gradient of its own and they diverge.
    assert not torch.allclose(g_none, g_live, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------
# 7. the ensemble distinction that the diffuse observable turns on

def test_per_configuration_masks_reach_diffuse_and_a_shared_one_does_not():
    """A mask shared by every configuration has zero variance: it moves <F>
    but contributes exactly nothing to <|F|^2> - |<F>|^2. Per-configuration
    masks do contribute. This is why the feature is per_conformer."""
    sys = _system()
    n_cfg = 4
    rng = np.random.default_rng(1)
    batch = sys["frac"].unsqueeze(0) + torch.tensor(
        0.01 * rng.standard_normal((n_cfg,) + tuple(sys["frac"].shape)),
        dtype=DTYPE)

    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())
    model = SolventModel(cutoff=cutoff, taper_width=0.25 * cutoff, k_sol=0.35)

    def batch_sf(**kw):
        return structure_factors_batch(
            batch, sys["element_idx"], sys["occ"],
            sys["atom_A"], sys["atom_lam"], sys["elem_offsets"],
            sys["atom_radius_ang"], sys["grid_shape"], sys["orth"],
            sys["cell_volume"], sys["hkl"], sys["taper_width"],
            compile_core=False, **kw
        )

    _, diffuse_none = mean_and_diffuse(batch_sf(solvent=None))
    _, diffuse_per = mean_and_diffuse(batch_sf(solvent=model))

    # A shared (constant) solvent term added to every configuration: same
    # arithmetic shape, zero variance, so diffuse must be untouched.
    F_prot = batch_sf(solvent=None)
    inv_d2 = reciprocal_inv_d2(sys["orth"], sys["hkl"])
    F_mask_shared = f_solvent(
        solvent_mask(rho, cutoff, 0.25 * cutoff),
        sys["cell_volume"], sys["hkl"], sys["orth"])
    F_combined = torch.stack(
        [f_total(F, F_mask_shared, inv_d2, k_sol=0.35) for F in F_prot], dim=0)
    _, diffuse_combined = mean_and_diffuse(F_combined)

    # Compared as a RATIO rather than with an absolute tolerance. <|F|^2> -
    # |<F>|^2 is a difference of two large nearly-equal numbers, so adding a
    # constant to every F changes its float64 cancellation error even though
    # it cannot change the variance -- exactly the effect docs/performance.md
    # records as diffuse's reproducibility floor. What must hold is that the
    # shared mask's effect is negligible against the per-configuration one.
    shared_effect = (diffuse_combined - diffuse_none).abs().max()
    per_config_effect = (diffuse_per - diffuse_none).abs().max()
    assert float(per_config_effect) > 0.0
    assert float(shared_effect) < 1e-6 * float(per_config_effect)


def test_checkpointing_gives_the_same_answer_with_solvent():
    """use_checkpoint recomputes the splat in backward; the mask FFT lives
    inside the checkpointed region, so the result must be unchanged."""
    sys = _system()
    batch = sys["frac"].unsqueeze(0).repeat(2, 1, 1)
    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())
    model = SolventModel(cutoff=cutoff, taper_width=0.25 * cutoff)

    def run(use_checkpoint):
        return structure_factors_batch(
            batch, sys["element_idx"], sys["occ"],
            sys["atom_A"], sys["atom_lam"], sys["elem_offsets"],
            sys["atom_radius_ang"], sys["grid_shape"], sys["orth"],
            sys["cell_volume"], sys["hkl"], sys["taper_width"],
            compile_core=False, solvent=model, use_checkpoint=use_checkpoint,
        )

    assert torch.allclose(run(False), run(True), rtol=1e-12, atol=1e-14)


def test_taper_width_must_be_positive():
    with pytest.raises(ValueError, match="taper_width"):
        SolventModel(cutoff=0.1, taper_width=0.0)
