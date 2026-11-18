"""
lunus.sf.scaling_torch: the Bragg scaling module.

What is pinned here:

  * The module is the R-factor tool's arithmetic, not a second copy of it:
    BraggScaler, given the structure factors the tool fitted on 7FPV,
    reproduces the tool's single-scale and binned R-work, R-free, k_sol and
    b_sol exactly.
  * The per-step refit: refit("bins") is the closed-form optimum of the bin
    scales (it equals an optimizer's), and the gradient through apply() after
    it is the gradient of the profiled loss (finite differences).
  * The free set never moves a fitted parameter, and the bins are assigned on
    the whole list.
  * fit(warm=True) after a small change lands in the cold fit's basin.
  * return_parts: the three terms reassemble to the default F_total, in a
    single cell and a folded supercell, and through structure_factors_batch.
  * The single-scale exponential model is solvent_torch.f_total.
  * float32 lands within 2e-4 of the float64 R (not to every printed digit).
"""

import contextlib
import copy
import io
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from conftest import REPO_ROOT, requires_cctbx  # noqa: E402

from lunus.sf import scaling_torch as scaling  # noqa: E402
from lunus.sf.cell_utils import orth_matrix  # noqa: E402
from lunus.sf.scaling_torch import BraggScaler  # noqa: E402

TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
COMPARE = os.path.join(REPO_ROOT, "lunus", "sf", "examples", "compare_gemmi")

CELL = (30.0, 40.0, 50.0, 90.0, 90.0, 90.0)


def synthetic(seed=0, d_min=2.0, noise=0.02, dtype=torch.float64):
    """A reflection list in an orthorhombic cell, random F_protein and a
    low-resolution F_mask, and F_obs from the scaling model with known
    parameters (k 0.8, an anisotropic B, k_sol 0.35, b_sol 46) plus noise; a
    5% free set."""
    rng = np.random.default_rng(seed)
    M = orth_matrix(*CELL)
    Minv = np.linalg.inv(M)
    hmax = [int(np.ceil(c / d_min)) for c in CELL[:3]]
    hkl = np.array([(h, k, l) for h in range(hmax[0] + 1)
                    for k in range(hmax[1] + 1) for l in range(hmax[2] + 1)
                    if (h, k, l) != (0, 0, 0)])
    s2 = ((hkl @ Minv) ** 2).sum(1)
    hkl = hkl[s2 <= 1.0 / d_min ** 2]
    s_cart = hkl @ Minv
    s2 = (s_cart ** 2).sum(1)
    n = len(hkl)
    Fp = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 100.0 * np.exp(-5.0 * s2)
    # solvent: large at low resolution, anti-correlated with the protein there
    Fm = (-0.6 * Fp / (100.0 * np.exp(-5.0 * s2)) * 400.0
          + (rng.normal(size=n) + 1j * rng.normal(size=n)) * 200.0) * np.exp(-60.0 * s2)
    B = np.array([[12.0, 0.0, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 25.0]])
    k_mask = 0.35 * np.exp(-0.25 * 46.0 * s2)
    Fc = Fp + k_mask * Fm
    F_obs = 0.8 * np.exp(-0.25 * np.einsum("ni,ij,nj->n", s_cart, B, s_cart)) * np.abs(Fc)
    F_obs = F_obs * (1.0 + noise * rng.normal(size=n))
    work = rng.random(n) >= 0.05
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return dict(hkl=torch.tensor(hkl), orth=torch.tensor(M, dtype=dtype),
                F_obs=torch.tensor(np.abs(F_obs), dtype=dtype),
                work=torch.tensor(work),
                Fp=torch.tensor(Fp, dtype=cdtype), Fm=torch.tensor(Fm, dtype=cdtype))


def scaler_for(data, **kw):
    return BraggScaler(data["hkl"], data["orth"], data["F_obs"], data["work"],
                       dtype=data["orth"].dtype, **kw)


def work_loss(scaler, data, Fp, Fm):
    F = scaler.apply(Fp, Fm)
    w = data["work"]
    return ((data["F_obs"][w] - F[w]) ** 2).sum()


# --------------------------------------------------------------- parity

@requires_cctbx
@pytest.mark.skipif(not os.path.exists(os.path.join(COMPARE, "7FPV-sf.cif")),
                    reason="examples/compare_gemmi/7FPV-sf.cif not present")
def test_scaler_reproduces_the_tool_on_7fpv(monkeypatch):
    """BraggScaler on the structure factors the R-factor tool fitted (7FPV at
    3 A, anisotropic) gives the tool's numbers: the main binned fit (log bins,
    per-bin solvent) and the single-scale fit, R-work, R-free, k_sol, b_sol,
    to the last bit."""
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import fit_solvent_rfactor as fsr

    import lunus.sf.structure_factor_torch as sft

    seen = {}
    real_fit, real_bins, real_inv_d2 = fsr.fit_scales, fsr.assign_scale_bins, sft.reciprocal_inv_d2

    def fit_spy(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso, **kw):
        if F_mask is not None and kw.get("bins") is not None:
            seen["fit"] = (F_obs, F_protein, F_mask, work, aniso)
        return real_fit(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso, **kw)

    def bins_spy(spec, inv_d2, d):
        seen["d"] = d
        return real_bins(spec, inv_d2, d)

    def inv_d2_spy(orth, hkl):
        seen.setdefault("geometry", (orth, hkl))
        return real_inv_d2(orth, hkl)

    monkeypatch.setattr(fsr, "fit_scales", fit_spy)
    monkeypatch.setattr(fsr, "assign_scale_bins", bins_spy)
    monkeypatch.setattr(sft, "reciprocal_inv_d2", inv_d2_spy)
    args = fsr.build_parser().parse_args(
        [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
         "--d-min", "3.0", "--aniso", "--shells", "3"])
    with contextlib.redirect_stdout(io.StringIO()):
        r = fsr.run(args)
    assert r["main_scale_bins"] == 6

    F_obs, Fp, Fm, work, aniso = seen["fit"]
    orth, hkl = seen["geometry"]
    sc = BraggScaler(hkl, orth, F_obs, torch.as_tensor(work), aniso=aniso, d=seen["d"])
    assert torch.equal(sc.bins, real_bins("log", sc.inv_d2, seen["d"]))
    sc.fit(Fp, Fm)
    rw, rf = sc.r_factors(sc.apply(Fp, Fm))
    assert (rw, rf) == (r["R-work"], r["R-free"])
    assert float(sc.state["k_sol"]) == r["k_sol"]
    assert float(sc.state["b_sol"]) == r["b_sol"]
    single = scaling.model_amplitudes(sc.single, Fp, Fm, sc.inv_d2, sc.s_cart)
    assert sc.r_factors(single) == (r["R-work single"], r["R-free single"])
    assert float(sc.single["k_sol"]) == r["k_sol single"]
    assert float(sc.single["b_sol"]) == r["b_sol single"]


# ------------------------------------------------------------- the refit

def test_closed_form_bin_scales_equal_the_optimizer():
    """refit("bins") is the least-squares optimum of the bin scales with
    everything else held: an LBFGS on the same objective lands on it."""
    data = synthetic()
    sc = scaler_for(data, aniso=True)
    sc.fit(data["Fp"], data["Fm"])
    st = sc.state
    lk_fit = st["log_k_bins"].clone()
    # move the bin scales off, then solve them back in closed form
    st["log_k_bins"] = lk_fit + 0.3
    sc.refit(data["Fp"], data["Fm"], what="bins")
    closed = sc.state["log_k_bins"]

    w = data["work"]
    b = sc.bins[w]
    Fo, s2, s = data["F_obs"][w], sc.inv_d2[w], sc.s_cart[w]
    Fc = data["Fp"][w] + sc.state["k_mask_bins"][b].to(torch.complex128) * data["Fm"][w]
    lk = (lk_fit + 0.3).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([lk], max_iter=500, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        model = scaling.scaled_amplitudes(Fc, s2, s, lk[b], sc.state["u_params"], True)
        loss = ((Fo - model) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    assert torch.allclose(closed, lk.detach(), atol=1e-7)
    # and the closed form is where the fit itself left them
    assert torch.allclose(closed, lk_fit, atol=1e-10)


@pytest.mark.parametrize("bins,aniso,what", [
    ("log", False, "bins"), ("log", True, "bins"), (10, True, "bins"),
    (0, False, "bins"), (0, True, "solvent")])
def test_gradient_through_apply_is_the_profiled_gradient(bins, aniso, what):
    """After refit(what), autograd through apply() with the scales held is
    the derivative of the PROFILED loss, min over the refit parameters of
    the loss: the envelope theorem, checked by central differences that
    refit at every perturbed point, along a direction that moves F_protein
    and F_mask together. "bins" is closed form; "solvent" with one scale is
    the exponential's LBFGS, stationary to its tolerance. (The per-bin
    solvent refit is NOT stationary -- see BraggScaler.refit -- and is not
    claimed here.)"""
    data = synthetic(seed=1)
    sc = scaler_for(data, bins=bins, aniso=aniso)
    sc.fit(data["Fp"], data["Fm"])
    Fp0, Fm0 = data["Fp"], data["Fm"]
    rng = np.random.default_rng(2)
    n = len(Fp0)

    def rand():
        return torch.tensor(rng.normal(size=n) + 1j * rng.normal(size=n),
                            dtype=torch.complex128)
    dp, dm = rand() * Fp0.abs().mean(), rand() * Fm0.abs().mean()

    def profiled(t):
        s = copy.deepcopy(sc)
        Fp, Fm = Fp0 + t * dp, Fm0 + t * dm
        s.refit(Fp, Fm, what=what)
        with torch.no_grad():
            return float(work_loss(s, data, Fp, Fm))

    sc.refit(Fp0, Fm0, what=what)
    Fp = Fp0.clone().requires_grad_(True)
    Fm = Fm0.clone().requires_grad_(True)
    work_loss(sc, data, Fp, Fm).backward()
    analytic = float((torch.conj(Fp.grad) * dp).real.sum()
                     + (torch.conj(Fm.grad) * dm).real.sum())

    eps = 1e-4
    numeric = (profiled(eps) - profiled(-eps)) / (2 * eps)
    tol = 1e-5 if what == "bins" else 2e-4
    assert abs(numeric - analytic) <= tol * abs(analytic), (numeric, analytic)


def test_apply_carries_no_parameter_gradient():
    """The parameters apply() uses are plain tensors: a backward pass reaches
    F_protein and F_mask and nothing in the scaler."""
    data = synthetic()
    sc = scaler_for(data, aniso=True)
    sc.fit(data["Fp"], data["Fm"])
    Fp = data["Fp"].clone().requires_grad_(True)
    Fm = data["Fm"].clone().requires_grad_(True)
    work_loss(sc, data, Fp, Fm).backward()
    assert Fp.grad is not None and Fm.grad is not None
    st = sc.state
    for v in [st["log_k_bins"], st["k_mask_bins"], st["k_sol"], st["b_sol"]] + st["u_params"]:
        assert not v.requires_grad


# --------------------------------------------------------- the free set

def test_the_free_set_never_moves_a_fitted_parameter():
    data = synthetic(seed=3)
    sc = scaler_for(data, aniso=True)
    sc.fit(data["Fp"], data["Fm"])
    sc.refit(data["Fp"], data["Fm"], what="solvent")
    before = sc.state_dict()
    r_before = sc.r_factors(sc.apply(data["Fp"], data["Fm"]))

    poisoned = dict(data)
    F_obs = data["F_obs"].clone()
    F_obs[~data["work"]] *= 3.0
    poisoned["F_obs"] = F_obs
    sc2 = scaler_for(poisoned, aniso=True)
    sc2.fit(data["Fp"], data["Fm"])
    sc2.refit(data["Fp"], data["Fm"], what="solvent")
    after = sc2.state_dict()
    for key in ("log_k_bins", "k_mask_bins", "k_sol", "b_sol"):
        assert torch.equal(before["state"][key], after["state"][key]), key
    for a, b in zip(before["state"]["u_params"], after["state"]["u_params"]):
        assert torch.equal(a, b)
    r_after = sc2.r_factors(sc2.apply(data["Fp"], data["Fm"]))
    assert r_after[0] == r_before[0]
    assert r_after[1] > r_before[1] + 0.1


def test_bins_are_assigned_on_the_whole_list():
    """A reflection's bin does not depend on which reflections are free:
    the scale fitted for a bin applies to its free reflections unchanged.
    True by construction (the constructor bins before it looks at work);
    pinned so that it stays so."""
    data = synthetic()
    all_work = dict(data, work=torch.ones_like(data["work"]))
    for spec in ("log", 10):
        assert torch.equal(scaler_for(data, bins=spec).bins,
                           scaler_for(all_work, bins=spec).bins)


# ----------------------------------------------------------- warm start

@pytest.mark.parametrize("bins", ["log", 0])
def test_a_warm_fit_lands_in_the_cold_fits_basin(bins):
    """After a small change to F_protein, fit(warm=True) -- no scan, LBFGS
    from the previous values -- finds what a cold fit finds."""
    data = synthetic(seed=4)
    sc = scaler_for(data, bins=bins, aniso=True)
    sc.fit(data["Fp"], data["Fm"])
    rng = np.random.default_rng(5)
    n = len(data["Fp"])
    Fp1 = data["Fp"] * torch.tensor(1.0 + 0.02 * rng.normal(size=n))
    warm = copy.deepcopy(sc)
    warm.fit(Fp1, data["Fm"], warm=True)
    cold = scaler_for(data, bins=bins, aniso=True)
    cold.fit(Fp1, data["Fm"])
    rw_w, rf_w = warm.r_factors(warm.apply(Fp1, data["Fm"]))
    rw_c, rf_c = cold.r_factors(cold.apply(Fp1, data["Fm"]))
    assert abs(rw_w - rw_c) < 1e-5 and abs(rf_w - rf_c) < 1e-5
    assert abs(float(warm.single["k_sol"]) - float(cold.single["k_sol"])) < 1e-3
    assert abs(float(warm.single["b_sol"]) - float(cold.single["b_sol"])) < 0.5
    assert warm.last_refit["fit_warm"] is True and cold.last_refit["fit_warm"] is False


def test_solvent_refit_right_after_fit_changes_nothing():
    """refit("solvent") with a per-bin solvent is the fit's own last stage, so
    straight after fit() it reproduces the fit's curve and bin scales."""
    data = synthetic(seed=6)
    sc = scaler_for(data, aniso=True)
    st = sc.fit(data["Fp"], data["Fm"])
    km, lk = st["k_mask_bins"].clone(), st["log_k_bins"].clone()
    sc.refit(data["Fp"], data["Fm"], what="solvent")
    assert torch.allclose(sc.state["k_mask_bins"], km, atol=1e-12)
    assert torch.allclose(sc.state["log_k_bins"], lk, atol=1e-12)


def test_exponential_solvent_refit_recovers_a_displaced_k_sol():
    data = synthetic(seed=7)
    sc = scaler_for(data, bins=0, aniso=True)
    sc.fit(data["Fp"], data["Fm"])
    k0, b0 = float(sc.state["k_sol"]), float(sc.state["b_sol"])
    sc.state["k_sol"] = sc.state["k_sol"] * 0.7
    sc.refit(data["Fp"], data["Fm"], what="solvent")
    assert abs(float(sc.state["k_sol"]) - k0) < 1e-3
    assert abs(float(sc.state["b_sol"]) - b0) < 0.5
    assert sc.single is sc.state         # one scale: the single fit is the model


def test_refit_record_and_state_round_trip():
    data = synthetic()
    sc = scaler_for(data)
    with pytest.raises(RuntimeError):
        sc.apply(data["Fp"], data["Fm"])
    sc.fit(data["Fp"], data["Fm"])
    sc.apply(data["Fp"], data["Fm"])
    sc.apply(data["Fp"], data["Fm"])
    sc.refit(data["Fp"], data["Fm"])
    sc.apply_complex(data["Fp"], data["Fm"])
    sc.refit(data["Fp"], data["Fm"], what="solvent")
    rec = sc.state_dict()["last_refit"]
    assert rec == {"fit": 0, "fit_warm": False, "bins": 2, "solvent": 3,
                   "all": None, "what": "solvent"}
    with pytest.raises(ValueError):
        sc.refit(data["Fp"], data["Fm"], what="everything")

    other = scaler_for(data)
    other.load_state_dict(sc.state_dict())
    assert torch.equal(other.apply(data["Fp"], data["Fm"]), sc.apply(data["Fp"], data["Fm"]))
    with pytest.raises(ValueError):
        scaler_for(data, bins=5).load_state_dict(sc.state_dict())


def test_a_bin_without_work_reflections_keeps_its_scale():
    """An equal-count bin that is entirely free has nothing to fit: the
    closed-form refit leaves its scale where it was, and the model stays
    finite there."""
    data = synthetic(seed=11)
    probe = scaler_for(data, bins=20)
    work = data["work"].clone()
    work[probe.bins == 7] = False
    data = dict(data, work=work)
    sc = scaler_for(data, bins=20, solvent_bins=False)
    sc.fit(data["Fp"], data["Fm"])
    lk7 = float(sc.state["log_k_bins"][7])
    sc.state["log_k_bins"] = sc.state["log_k_bins"] + 0.1
    sc.refit(data["Fp"], data["Fm"])
    assert float(sc.state["log_k_bins"][7]) == pytest.approx(lk7 + 0.1, abs=1e-12)
    assert bool(torch.isfinite(sc.apply(data["Fp"], data["Fm"])).all())


def test_load_state_dict_takes_this_scalers_dtype_and_refuses_other_settings():
    data64 = synthetic(seed=12)
    sc = scaler_for(data64, aniso=True)
    sc.fit(data64["Fp"], data64["Fm"])
    state = sc.state_dict()
    data32 = synthetic(seed=12, dtype=torch.float32)
    sc32 = scaler_for(data32, aniso=True)
    sc32.load_state_dict(state)
    for fit in (sc32.state, sc32.single):
        for key in ("log_k", "log_k_bins", "k_mask_bins", "k_sol", "b_sol"):
            if fit[key] is not None:
                assert fit[key].dtype == torch.float32, key
        assert all(v.dtype == torch.float32 for v in fit["u_params"])
    assert sc32.r_factors(sc32.apply(data32["Fp"], data32["Fm"]))[0] == pytest.approx(
        sc.r_factors(sc.apply(data64["Fp"], data64["Fm"]))[0], abs=1e-5)
    with pytest.raises(ValueError, match="aniso"):
        scaler_for(data64, aniso=False).load_state_dict(state)
    # with one scale, solvent_bins has no effect and does not block a load
    one = scaler_for(data64, bins=0)
    one.fit(data64["Fp"], data64["Fm"])
    scaler_for(data64, bins=0, solvent_bins=True).load_state_dict(one.state_dict())
    # equal-count bins: a float32 sort of s^2 orders near-ties differently,
    # so the bins are assigned in float64 whatever the scaler's dtype
    sc10 = scaler_for(data64, bins=10)
    sc10.fit(data64["Fp"], data64["Fm"])
    scaler_for(data32, bins=10).load_state_dict(sc10.state_dict())
    # one scale: the single fit stays the model after a load
    sc0 = scaler_for(data64, bins=0)
    sc0.fit(data64["Fp"], data64["Fm"])
    other0 = scaler_for(data64, bins=0)
    other0.load_state_dict(sc0.state_dict())
    assert other0.single is other0.state
    # to() on the CPU too, so that a CPU-only run exercises it
    moved = copy.deepcopy(sc).to("cpu")
    assert torch.equal(moved.apply(data64["Fp"], data64["Fm"]),
                       sc.apply(data64["Fp"], data64["Fm"]))


def test_no_solvent_scaler_ignores_f_mask_and_needs_none():
    data = synthetic()
    sc = scaler_for(data, fit_solvent=False)
    sc.fit(data["Fp"])
    assert torch.equal(sc.apply(data["Fp"]), sc.apply(data["Fp"], data["Fm"]))
    with pytest.raises(ValueError):
        scaler_for(data).fit(data["Fp"])     # fit_solvent=True needs F_mask


# ------------------------------------------------------------- f_total

def test_single_scale_exponential_model_is_f_total():
    """bins=0 with the exponential solvent is solvent_torch.f_total, with
    U = Mstar^T B_cart Mstar / 4 in the integer-h basis."""
    from lunus.sf.solvent_torch import f_total
    data = synthetic(seed=8)
    sc = scaler_for(data, bins=0, aniso=True)
    st = sc.fit(data["Fp"], data["Fm"])
    B = torch.zeros(3, 3, dtype=torch.float64)
    for v, (i, j) in zip(st["u_params"], scaling.UNIQUE_IJ):
        B[i, j] = B[j, i] = v
    Mstar = torch.linalg.inv(data["orth"]).T
    U = Mstar.T @ B @ Mstar / 4.0
    expected = f_total(data["Fp"], data["Fm"], sc.inv_d2, k_sol=st["k_sol"],
                       b_sol=st["b_sol"], k_overall=torch.exp(st["log_k"]),
                       u_aniso=U, hkl=data["hkl"])
    assert torch.allclose(sc.apply_complex(data["Fp"], data["Fm"]), expected,
                          rtol=1e-10, atol=1e-10)


# ------------------------------------------------------------ precision

@pytest.mark.parametrize("bins", ["log", 10, 0])
def test_float32_gives_the_float64_r_closely(bins):
    """float32 fits land within 2e-4 of float64 in R. Not to the printed
    digit everywhere, measured 2026-11-17: on this data the per-bin solvent's
    high-resolution bins, where F_mask carries almost nothing, are poorly
    determined and float32 moves them (and whether the curve rule cuts the
    last one), 7e-5 in R-work; one scale and equal-count bins agree to 2e-5.
    On 7FPV (full resolution, --aniso) the binned fit printed the same digits
    and the single-scale fit R-free 0.1908 against 0.1909, b_sol 30.0
    against 31.1 -- b_sol's valley is flat."""
    r = {}
    for dtype in (torch.float64, torch.float32):
        data = synthetic(seed=9, dtype=dtype)
        sc = scaler_for(data, aniso=True, bins=bins)
        sc.fit(data["Fp"], data["Fm"])
        r[dtype] = sc.r_factors(sc.apply(data["Fp"], data["Fm"]))
    assert abs(r[torch.float32][0] - r[torch.float64][0]) < 2e-4
    assert abs(r[torch.float32][1] - r[torch.float64][1]) < 2e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_the_scaler_runs_on_cuda_and_agrees_with_cpu():
    data = synthetic(seed=10)
    cpu = scaler_for(data)
    cpu.fit(data["Fp"], data["Fm"])
    gpu = copy.deepcopy(cpu).to("cuda")
    Fp, Fm = data["Fp"].cuda(), data["Fm"].cuda()
    gpu.refit(Fp, Fm)
    cpu.refit(data["Fp"], data["Fm"])
    rg = gpu.r_factors(gpu.apply(Fp, Fm))
    rc = cpu.r_factors(cpu.apply(data["Fp"], data["Fm"]))
    assert abs(rg[0] - rc[0]) < 1e-10 and abs(rg[1] - rc[1]) < 1e-10


# --------------------------------------------------------- return_parts

def small_model(n_atoms=40, seed=0, factors=(1, 1, 1)):
    from lunus.sf.elements import IT92_COEFFS
    from lunus.sf.kernel_torch import build_element_kernels_torch
    cell, grid = (20.0, 22.0, 24.0, 90.0, 90.0, 90.0), (20, 22, 24)
    rng = np.random.default_rng(seed)
    M = orth_matrix(*cell)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, grid, M, cutoff=0.01,
        device="cpu", dtype=torch.float64)
    idx = torch.zeros(n_atoms, dtype=torch.long)
    return dict(
        frac=torch.tensor(rng.random((n_atoms, 3)) * np.array(factors) * 0.6),
        element_idx=idx, occ=torch.ones(n_atoms, dtype=torch.float64),
        atom_A=A[idx], atom_lam=lam[idx], elem_offsets=offsets,
        atom_radius_ang=radius[idx], grid_shape=grid,
        orth=torch.tensor(M, dtype=torch.float64),
        cell_volume=float(np.linalg.det(M)), taper_width=taper_w,
        hkl=torch.tensor([[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
                          for l in range(-3, 4) if (h, k, l) != (0, 0, 0)]))


def sf_one(s, **kw):
    from lunus.sf.structure_factor_torch import structure_factors_one_config
    return structure_factors_one_config(
        s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
        s["cell_volume"], s["hkl"], s["taper_width"], compile_core=False, **kw)


def solvent_model(**kw):
    from lunus.sf.solvent_torch import SolventModel
    return SolventModel(cutoff=0.05, taper_width=0.025, check_occupancy=False,
                        k_sol=0.4, b_sol=40.0, **kw)


@pytest.mark.parametrize("supercell", [None, (2, 1, 1)])
def test_return_parts_reassemble_to_the_default_f_total(supercell):
    """F_total in the parts is the default return, bit for bit, and
    f_total(F_protein, F_mask) with density_scale 1 rebuilds it: F_mask
    carries the fold and the model's density_scale."""
    from lunus.sf.solvent_torch import f_total
    from lunus.sf.structure_factor_torch import SolventParts, reciprocal_inv_d2
    s = small_model(factors=supercell or (1, 1, 1))
    model = solvent_model(density_scale=1.5)
    kw = {} if supercell is None else {"supercell": supercell}
    default = sf_one(s, solvent=model, **kw)
    parts = sf_one(s, solvent=model, return_parts=True, **kw)
    assert isinstance(parts, SolventParts)
    assert torch.equal(parts.F_total, default)
    assert torch.equal(parts.F_protein, sf_one(s, **kw))
    rebuilt = f_total(parts.F_protein, parts.F_mask, reciprocal_inv_d2(s["orth"], s["hkl"]),
                      k_sol=model.k_sol, b_sol=model.b_sol, hkl=s["hkl"])
    assert torch.allclose(rebuilt, default, rtol=1e-12, atol=1e-12)


def test_return_parts_costs_one_mask_fft(monkeypatch):
    """Single cell: F_mask in the parts is the one SolventModel.apply() built
    F_total from, so asking for the parts adds no mask FFT. (It used to call
    f_solvent a second time on the returned mask.) apply() hands back that
    F_mask unscaled: f_solvent of its own mask."""
    import lunus.sf.solvent_torch as solvent_torch
    calls = []
    real = solvent_torch.f_solvent

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(solvent_torch, "f_solvent", counting)
    s = small_model()
    model = solvent_model(density_scale=1.5)
    sf_one(s, solvent=model)
    assert len(calls) == 1
    calls.clear()
    parts = sf_one(s, solvent=model, return_parts=True)
    assert len(calls) == 1
    # and the batch: one mask FFT per configuration, checkpointed or not
    from lunus.sf.structure_factor_torch import structure_factors_batch
    frac = torch.stack([s["frac"], s["frac"] + 0.01])
    rest = (s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
            s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
            s["cell_volume"], s["hkl"], s["taper_width"])
    for ck in (False, True):
        calls.clear()
        with torch.no_grad():
            structure_factors_batch(frac, *rest, compile_core=False, solvent=model,
                                    use_checkpoint=ck, return_parts=True)
        assert len(calls) == 2, (ck, len(calls))

    F, mask, F_mask = model.apply(
        *density_and_protein(s), s["cell_volume"], s["hkl"], s["orth"])
    assert torch.equal(F_mask, real(mask, s["cell_volume"], s["hkl"], s["orth"]))
    assert torch.equal(parts.F_mask, F_mask * model.density_scale)
    assert torch.equal(parts.F_total, F)


def density_and_protein(s):
    from lunus.sf.density_torch import compute_density
    from lunus.sf.structure_factor_torch import compute_fcalc
    density = compute_density(
        s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
        s["taper_width"], compile_core=False)
    return density, compute_fcalc(density, s["cell_volume"], s["hkl"], s["orth"])


def test_return_parts_needs_a_solvent_model():
    s = small_model()
    with pytest.raises(ValueError, match="return_parts"):
        sf_one(s, return_parts=True)


@pytest.mark.parametrize("supercell", [None, (2, 1, 1)])
def test_batch_return_parts_stack_per_configuration(supercell):
    """structure_factors_batch with return_parts: F_total is the default
    batch return, each configuration's parts are structure_factors_one_config's,
    the parts reassemble, and checkpointing changes neither the parts nor their
    coordinate gradient."""
    from lunus.sf.solvent_torch import f_total
    from lunus.sf.structure_factor_torch import reciprocal_inv_d2, structure_factors_batch
    s = small_model(factors=supercell or (1, 1, 1))
    model = solvent_model(density_scale=1.5)
    kw = {} if supercell is None else {"supercell": supercell}
    frac0 = torch.stack([s["frac"], s["frac"] + 0.01])
    rest = (s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
            s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
            s["cell_volume"], s["hkl"], s["taper_width"])
    default = structure_factors_batch(frac0, *rest, compile_core=False, solvent=model, **kw)
    out = {}
    for ck in (False, True):
        frac = frac0.clone().requires_grad_(True)
        parts = structure_factors_batch(frac, *rest, compile_core=False, solvent=model,
                                        use_checkpoint=ck, return_parts=True, **kw)
        (parts.F_protein.abs().sum() + parts.F_mask.abs().sum()).backward()
        out[ck] = (parts, frac.grad)
    parts, grad = out[False]
    assert torch.equal(parts.F_total, default)
    for i in range(2):
        one = sf_one(dict(s, frac=frac0[i]), solvent=model, return_parts=True, **kw)
        for a, b in zip(parts, one):
            assert torch.equal(a[i], b)
    rebuilt = f_total(parts.F_protein, parts.F_mask, reciprocal_inv_d2(s["orth"], s["hkl"]),
                      k_sol=model.k_sol, b_sol=model.b_sol, hkl=s["hkl"])
    assert torch.allclose(rebuilt, default, rtol=1e-12, atol=1e-12)
    assert bool(grad.abs().sum() > 0)
    for a, b in zip(out[True][0], parts):
        assert torch.equal(a, b)
    assert torch.equal(out[True][1], grad)
