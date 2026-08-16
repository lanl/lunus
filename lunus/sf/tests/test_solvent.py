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
  8. The solvent's contribution to diffuse clears the grid-alignment noise it
     also introduces (tools/study_diffuse_solvent.py maps that properly).
"""

import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from lunus.sf.density_torch import splat_density
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.cell_utils import orth_matrix
from lunus.sf.solvent_torch import (
    SolventMaskWarning, SolventModel, f_solvent, f_total, mask_occupancy,
    shell_voxels, solvent_mask,
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
    # check_occupancy off: this fixture is a nearly empty cell, so the
    # degenerate-mask warning is CORRECT here and simply not what is under
    # test. It has its own tests below.
    common = dict(cutoff=cutoff, taper_width=0.25 * cutoff,
                  check_occupancy=False)
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
    common = dict(cutoff=cutoff, taper_width=0.25 * cutoff, k_sol=0.35,
                  check_occupancy=False)

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
    model = SolventModel(cutoff=cutoff, taper_width=0.25 * cutoff,
                         k_sol=0.35, check_occupancy=False)

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
    model = SolventModel(cutoff=cutoff, taper_width=0.25 * cutoff,
                         check_occupancy=False)

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


# --------------------------------------------------------------------------
# 8. the diffuse study's conclusion, in miniature

def test_solvent_diffuse_signal_dominates_grid_alignment_noise():
    """Diffuse is a variance, so mask discretization biases it rather than
    averaging out. The instrument is translation non-commutation: displacing
    every configuration by the same sub-voxel vector leaves the MODEL's diffuse
    exactly unchanged (one common phase on every F), but not the grid
    calculation's, since splat_density resamples at fixed voxel centres rather
    than shifting. Whatever comes out is grid alignment, not structure.

    What must hold is that the solvent's contribution is well clear of it.
    tools/study_diffuse_solvent.py maps this properly; the numbers there show
    the noise falling as roughly the square of the voxel spacing while the
    signal stays put, which is why the recommendation is to refine the grid
    rather than to widen the taper. This is the cheap regression guard.
    """
    # A CLUSTERED blob with a 1.4 A minimum separation, not a loose cloud: the
    # solvent boundary is the thing under test, and uniformly scattered atoms
    # do not have one. Grid spacing matters as much -- at 0.62 A the signal
    # clears the noise 5x, at 0.9 A only 1.9x, which is the h^2 scaling the
    # study measures.
    n_atoms, n_cfg, grid = 60, 4, (32, 32, 32)
    rng = np.random.default_rng(0)
    centre = np.array([10.0, 11.0, 12.0])
    cart = []
    while len(cart) < n_atoms:
        u = rng.normal(size=3)
        p = centre + 4.0 * rng.random() ** (1 / 3.0) * u / np.linalg.norm(u)
        if all(np.linalg.norm(p - q) > 1.4 for q in cart):
            cart.append(p)
    cart = np.array(cart)
    batch = cart[None] + 0.3 * rng.standard_normal((n_cfg, n_atoms, 3))

    M_np = orth_matrix(*CELL)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, grid, M_np, cutoff=0.01,
        device="cpu", dtype=DTYPE,
    )
    idx = torch.zeros(n_atoms, dtype=torch.long)
    common = dict(
        element_idx=idx, occ=torch.ones(n_atoms, dtype=DTYPE),
        atom_A=A[idx], atom_lam=lam[idx], elem_offsets=offsets,
        atom_radius_ang=radius[idx], grid_shape=grid,
        orth_matrix=torch.tensor(M_np, dtype=DTYPE),
        cell_volume=float(np.linalg.det(M_np)),
        hkl=torch.tensor([[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
                          for l in range(-3, 4) if (h, k, l) != (0, 0, 0)],
                         dtype=torch.long),
        taper_width=taper_w,
    )

    def diffuse(coords, solvent):
        frac = torch.tensor(coords @ np.linalg.inv(M_np).T, dtype=DTYPE)
        F = structure_factors_batch(
            frac, common["element_idx"], common["occ"], common["atom_A"],
            common["atom_lam"], common["elem_offsets"],
            common["atom_radius_ang"], common["grid_shape"],
            common["orth_matrix"], common["cell_volume"], common["hkl"],
            common["taper_width"], compile_core=False, solvent=solvent,
        )
        return mean_and_diffuse(F)[1]

    rho = splat_density(
        torch.tensor(batch[0] @ np.linalg.inv(M_np).T, dtype=DTYPE),
        common["element_idx"], common["occ"], common["atom_A"],
        common["atom_lam"], common["elem_offsets"], common["atom_radius_ang"],
        grid, common["orth_matrix"], taper_w, compile_core=False,
    )
    cutoff = 0.01 * float(rho.max())
    model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff)

    voxel = CELL[0] / grid[0]
    d_none = diffuse(batch, None)
    d_sol = diffuse(batch, model)
    d_sol_shifted = diffuse(batch + 0.37 * voxel, model)

    def rel(a, b):
        return float((a - b).abs().mean() / b.abs().mean())

    signal = rel(d_sol, d_none)
    noise = rel(d_sol_shifted, d_sol)
    assert signal > 0.01, signal            # solvent reaches diffuse at all
    # Measured ratio here is ~5; the floor is set at 3 so that a real
    # regression in mask discretization fails while ordinary
    # platform-to-platform drift does not.
    assert noise < signal / 3.0, (signal, noise)


# --------------------------------------------------------------------------
# 9. the occupancy check, wired into the pipeline

def _run_with(sys, model):
    return _sf(sys, solvent=model)


def _flooded(sys):
    """A density with no empty space anywhere -- what an all-atom model with
    explicit bulk water produces. It cannot be reached by thresholding the
    toy fixture instead: outside the atomic cutoff radius the density is
    EXACTLY zero, and zero is always below any positive cutoff, so vacuum is
    solvent however small the cutoff is made."""
    return _density(sys) + 1.0


def test_empty_mask_warns_rather_than_silently_doing_nothing():
    """The failure with no symptom: atoms already filling the cell give
    F_mask ~ 0, no error, and a solvent term that contributes nothing. The
    check exists so that arrives as a warning instead of as a conclusion."""
    sys = _system()
    model = SolventModel(cutoff=0.5, taper_width=0.25)
    F_p = _sf(sys, solvent=None)

    with pytest.warns(SolventMaskWarning, match="essentially empty"):
        model.apply(_flooded(sys), F_p, sys["cell_volume"], sys["hkl"],
                    sys["orth"])
    assert model.last_occupancy == pytest.approx(0.0, abs=1e-6)


def test_saturated_mask_warns_too():
    """The other end: a cutoff above the peak calls the whole cell solvent."""
    sys = _system()
    rho = _density(sys)
    huge = 10.0 * float(rho.max())
    model = SolventModel(cutoff=huge, taper_width=0.5 * huge)

    with pytest.warns(SolventMaskWarning, match="almost the whole cell"):
        _run_with(sys, model)
    assert model.last_occupancy > 0.99


def test_a_sensible_mask_does_not_warn_and_records_its_diagnostics():
    # 200 atoms rather than the 6-atom default: a nearly empty cell is 99.2%
    # solvent, and warning about THAT is correct behaviour, not a bad
    # threshold.
    sys = _system(n_atoms=200)
    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())
    model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff)

    with warnings.catch_warnings():
        warnings.simplefilter("error", SolventMaskWarning)
        _run_with(sys, model)

    assert 0.05 <= model.last_occupancy <= 0.99
    assert model.last_shell_voxels > 0
    assert "occupancy" in repr(model)


def test_the_check_runs_once_not_per_configuration():
    """It reads scalars back from the device, which synchronises. Once per
    model, not once per frame -- and one warning, not N."""
    sys = _system()
    model = SolventModel(cutoff=0.5, taper_width=0.25)
    F_p = _sf(sys, solvent=None)
    args = (_flooded(sys), F_p, sys["cell_volume"], sys["hkl"], sys["orth"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SolventMaskWarning)
        for _ in range(4):
            model.apply(*args)
    assert sum(issubclass(w.category, SolventMaskWarning) for w in caught) == 1


def test_the_check_can_be_turned_off():
    sys = _system()
    model = SolventModel(cutoff=0.5, taper_width=0.25, check_occupancy=False)
    F_p = _sf(sys, solvent=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SolventMaskWarning)
        model.apply(_flooded(sys), F_p, sys["cell_volume"], sys["hkl"],
                    sys["orth"])
    assert model.last_occupancy is None


# --------------------------------------------------------------------------
# 10. float32, which is what production actually runs

def _system_dtype(dtype, n_atoms=200, seed=0):
    """_system() is fixed to float64 for exact comparisons; this one is
    parametrised, so the same structure can be pushed through both."""
    rng = np.random.default_rng(seed)
    M_np = orth_matrix(*CELL)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, GRID, M_np, cutoff=0.01,
        device="cpu", dtype=dtype,
    )
    idx = torch.zeros(n_atoms, dtype=torch.long)
    return dict(
        frac=torch.tensor(rng.random((n_atoms, 3)), dtype=dtype),
        element_idx=idx, occ=torch.ones(n_atoms, dtype=dtype),
        atom_A=A[idx], atom_lam=lam[idx], elem_offsets=offsets,
        atom_radius_ang=radius[idx], grid_shape=GRID,
        orth=torch.tensor(M_np, dtype=dtype),
        cell_volume=float(np.linalg.det(M_np)), taper_width=taper_w,
        hkl=torch.tensor(
            [[h, k, l] for h in range(-4, 5) for k in range(-4, 5)
             for l in range(-4, 5) if (h, k, l) != (0, 0, 0)],
            dtype=torch.long),
    )


@pytest.mark.parametrize("cut_frac", [1e-2, 1e-5])
def test_float32_matches_float64_below_the_pipelines_own_noise(cut_frac):
    """Everything else here runs float64 for exact comparisons; production
    runs float32 on a GPU. The question is whether the mask survives it, and
    especially at the SMALL cutoffs that matching a geometric mask forces,
    where the taper shell is only tens of voxels wide and a float32 density
    might move the level set.

    It does survive. Measured mean relative error on F_total: 1.3e-6 at a
    cutoff of 1% of peak density (1134 shell voxels), 1.2e-5 at 1e-5 of peak
    (NINE shell voxels) -- so the thin-shell case costs about 9x, the expected
    direction, and is still small. The shell voxel count is IDENTICAL in the
    two precisions in both cases, so the level set itself does not move.

    The threshold below is the pipeline's own reproducibility floor rather
    than a round number: docs/performance.md records GPU non-determinism at
    4e-5 on Icalc and 2e-4 on diffuse. float32 stays under that, so it is not
    the limiting precision -- which is the claim worth defending, since
    tightening it further would buy nothing real.
    """
    out = {}
    for dtype in (torch.float32, torch.float64):
        s = _system_dtype(dtype)
        rho = splat_density(
            s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
            s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"],
            s["orth"], s["taper_width"], compile_core=False)
        cutoff = cut_frac * float(rho.max())
        model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff,
                             check_occupancy=False)
        F = structure_factors_one_config(
            s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
            s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"],
            s["orth"], s["cell_volume"], s["hkl"], s["taper_width"],
            compile_core=False, solvent=model)
        mask = solvent_mask(rho, cutoff, 0.5 * cutoff)
        out[dtype] = dict(F=F, occ=float(mask_occupancy(mask)),
                          shell=shell_voxels(mask), cutoff=cutoff)

    lo, hi = out[torch.float32], out[torch.float64]
    assert lo["F"].dtype == torch.complex64
    assert hi["F"].dtype == torch.complex128

    rel = ((lo["F"].to(torch.complex128) - hi["F"]).abs() / hi["F"].abs())
    assert float(rel.mean()) < 4e-5, float(rel.mean())     # the Icalc floor
    assert float(rel.max()) < 1e-3, float(rel.max())

    # The level set itself, which is what a thin shell puts at risk.
    assert lo["shell"] == hi["shell"]
    assert lo["occ"] == pytest.approx(hi["occ"], abs=1e-5)


# --------------------------------------------------------------------------
# 11. calibrating the cutoff, which is the only way to set it

def test_calibrate_cutoff_hits_the_requested_occupancy():
    from lunus.sf.solvent_torch import calibrate_cutoff
    sys = _system(n_atoms=200)
    rho = _density(sys)
    for target in (0.6, 0.75, 0.9):
        cut = calibrate_cutoff(rho, target)
        got = float(mask_occupancy(solvent_mask(rho, cut, 0.5 * cut)))
        assert got == pytest.approx(target, abs=0.02), (target, got, cut)


def test_calibrate_cutoff_rejects_impossible_targets():
    from lunus.sf.solvent_torch import calibrate_cutoff
    rho = _density(_system())
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="target_occupancy"):
            calibrate_cutoff(rho, bad)


def test_calibrate_cutoff_reports_an_unreachable_target():
    """A model with genuine vacuum has an occupancy FLOOR -- no positive
    cutoff can call a zero-density voxel protein. Asking for less than that
    used to bisect to the smallest cutoff tried and return an occupancy
    nothing like the one requested, silently."""
    from lunus.sf.solvent_torch import calibrate_cutoff
    sys = _system(n_atoms=200)
    rho = _density(sys)
    floor = float(mask_occupancy(solvent_mask(rho, 1e-12, 0.5e-12)))
    assert floor > 0.3, floor          # this fixture does have vacuum
    with pytest.raises(ValueError, match="out of reach"):
        calibrate_cutoff(rho, 0.5 * floor)


# --------------------------------------------------------------------------
# 9. mask_blur: the probe radius

def test_mask_blur_zero_reproduces_the_bare_threshold():
    """The escape hatch has to be exact, because every number measured before
    mask_blur existed was measured at 0 and has to stay reproducible."""
    sys = _system()
    rho = _density(sys)
    cutoff = 0.1 * float(rho.max())
    model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff,
                         mask_blur=0.0, check_occupancy=False)

    assert torch.equal(model.build_mask(rho, sys["orth"]),
                       solvent_mask(rho, cutoff, 0.5 * cutoff))


def test_mask_blur_rejects_a_negative_b():
    """A negative B sharpens, which amplifies the boundary roughness the blur
    exists to remove -- the opposite of the intent, and silent."""
    with pytest.raises(ValueError, match="mask_blur"):
        SolventModel(cutoff=0.1, taper_width=0.05, mask_blur=-1.0)


def test_mask_blur_smooths_the_mask_boundary_on_a_dense_model():
    """What the blur is for: closing the crevices between neighbouring atoms,
    and so less high-frequency content in F_mask.

    Asserted on the transform rather than the grid because that is what reaches
    F_total, and with the occupancy held fixed by recalibration so this
    measures SHAPE and not size.

    IT MUST BE A DENSE FIXTURE, and that is the interesting part. Measured on
    the 40-atom fixture the rest of this file uses -- 85% genuine vacuum -- the
    blur changes sum|F(h!=0)| by 0.1%, because isolated atoms have no crevices
    between them to close; blurring merely inflates them into bigger blobs and
    the boundary AREA grows about as fast as its roughness falls. Pack the same
    cell until there is no vacuum, which is what a real structure looks like,
    and the effect appears: 0.93 here, and far larger on 7FPV.

    This is the third time a conclusion drawn from a sparse fixture has failed
    to transfer to a packed one -- see the fixed-fraction cutoff and the parity
    test's voxel agreement, both in docs/solvent-design.md. A sparse model has
    real vacuum between its atoms and a real one has only overlapping tails,
    and the two behave differently for anything that depends on where the
    density crosses a threshold.
    """
    rng = np.random.default_rng(3)
    n_atoms = 400
    frac = torch.tensor(rng.random((n_atoms, 3)), dtype=DTYPE)
    M_np = orth_matrix(*CELL)
    orth = torch.tensor(M_np, dtype=DTYPE)
    volume = float(np.linalg.det(M_np))
    grid = (32, 32, 32)
    hkl = torch.tensor([[h, k, l] for h in range(9) for k in range(9)
                        for l in range(9)], dtype=torch.long)[1:]   # drop (000)

    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, grid, M_np, cutoff=0.01,
        device="cpu", dtype=DTYPE)
    idx = torch.zeros(n_atoms, dtype=torch.long)
    rho = splat_density(frac, idx, torch.ones(n_atoms, dtype=DTYPE),
                        A[idx], lam[idx], offsets, radius[idx], grid, orth,
                        taper_w, compile_core=False)
    target = 0.5
    d = 1.0 / torch.sqrt(reciprocal_inv_d2(orth, hkl))

    def content(blur, sel):
        model = SolventModel(cutoff=1.0, taper_width=0.5, mask_blur=blur,
                             check_occupancy=False)
        model.calibrate(rho, orth, target)
        mask = model.build_mask(rho, orth)
        assert float(mask_occupancy(mask)) == pytest.approx(target, abs=1e-3)
        return float(f_solvent(mask, volume, hkl, orth)[sel].abs().sum())

    # THE EFFECT IS RESOLUTION DEPENDENT, and the crossover is the point.
    # Beyond ~4 A the blur slightly INCREASES |F_mask| -- it is changing the
    # envelope's gross shape, and a smoothed boundary encloses a marginally
    # different volume at fixed occupancy. Below 4 A, where the crevices live
    # and where the excess drove b_sol to 199 on real data, it removes 30-50%.
    # Asserting on the whole reflection set would average these two and pin
    # nothing.
    high = d < 4.0
    assert content(100.0, high) < 0.8 * content(0.0, high)

    low = d >= 4.0
    assert content(100.0, low) == pytest.approx(content(0.0, low), rel=0.15)


def test_calibration_must_see_the_grid_the_mask_is_built_from():
    """The trap mask_source() exists to close.

    Calibrating a cutoff on the raw density and then thresholding a blurred
    one asks for an occupancy from one distribution and applies it to another.
    It does not raise, it just misses -- so this pins that model.calibrate()
    hits the target and the naive two-step does not.
    """
    from lunus.sf.solvent_torch import calibrate_cutoff

    sys = _system(n_atoms=40, seed=5)
    rho = _density(sys)
    target = 0.9
    model = SolventModel(cutoff=1.0, taper_width=0.5, mask_blur=100.0,
                         check_occupancy=False)

    model.calibrate(rho, sys["orth"], target)
    assert float(mask_occupancy(model.build_mask(rho, sys["orth"]))) == \
        pytest.approx(target, abs=1e-3)

    # The naive version: calibrate on the unblurred density, threshold the
    # blurred one. Same target, different answer.
    naive_cutoff = calibrate_cutoff(rho, target)
    naive = SolventModel(cutoff=naive_cutoff, taper_width=0.5 * naive_cutoff,
                         mask_blur=100.0, check_occupancy=False)
    missed = float(mask_occupancy(naive.build_mask(rho, sys["orth"])))
    assert abs(missed - target) > 0.05


def test_mask_blur_is_a_length_and_so_survives_a_change_of_grid():
    """B is in A^2, so the same value means the same physical smoothing on any
    grid. That is what makes it a shippable default when the cutoff -- which
    moves with grid, B convention and system -- is not.

    Measured through the occupancy the same cutoff produces on a 24^3 and a
    36^3 grid of the same cell. Some drift is the density's own resampling and
    is present without any blur; the test asserts the blur does not ADD
    grid dependence, by comparing the two drifts.
    """
    rng = np.random.default_rng(11)
    frac = torch.tensor(rng.random((40, 3)), dtype=DTYPE)
    M_np = orth_matrix(*CELL)
    orth = torch.tensor(M_np, dtype=DTYPE)

    def occupancy(grid, blur, cutoff):
        A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
            ["C"], IT92_COEFFS, 20.0, 0.0, grid, M_np, cutoff=0.01,
            device="cpu", dtype=DTYPE)
        idx = torch.zeros(frac.shape[0], dtype=torch.long)
        rho = splat_density(frac, idx, torch.ones(frac.shape[0], dtype=DTYPE),
                            A[idx], lam[idx], offsets, radius[idx], grid,
                            orth, taper_w, compile_core=False)
        model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff,
                             mask_blur=blur, check_occupancy=False)
        return float(mask_occupancy(model.build_mask(rho, orth)))

    cutoff = 0.05
    drift_bare = abs(occupancy((24, 24, 24), 0.0, cutoff)
                     - occupancy((36, 36, 36), 0.0, cutoff))
    drift_blurred = abs(occupancy((24, 24, 24), 100.0, cutoff)
                        - occupancy((36, 36, 36), 100.0, cutoff))
    assert drift_blurred < drift_bare + 0.02
