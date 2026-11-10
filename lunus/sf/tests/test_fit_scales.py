"""tools/fit_solvent_rfactor.py: fit_scales with and without resolution bins.

Synthetic, torch only, about a second: amplitudes are built from random
protein and mask structure factors with a known k_sol/b_sol and an overall
scale that rises and falls with resolution, which no Gaussian B can follow.
The binned fit must recover it and the single-scale fit must not.
"""

import os
import sys

import pytest
import torch

from conftest import HAVE_CCTBX, REPO_ROOT

TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import fit_solvent_rfactor as fsr  # noqa: E402


def _synthetic(n=6000, n_bins=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64
    inv_d2 = 0.01 + 0.5 * torch.rand(n, generator=g, dtype=dt)
    direction = torch.randn(n, 3, generator=g, dtype=dt)
    s_cart = direction / direction.norm(dim=1, keepdim=True) * inv_d2.sqrt()[:, None]
    F_protein = torch.complex(torch.randn(n, generator=g, dtype=dt),
                              torch.randn(n, generator=g, dtype=dt)) * 100.0
    # The mask term only matters at low resolution, as real bulk solvent does.
    F_mask = torch.complex(torch.randn(n, generator=g, dtype=dt),
                           torch.randn(n, generator=g, dtype=dt)) * 300.0
    bins = fsr.resolution_bins(inv_d2, n_bins)
    k_true = 1.0 + 0.3 * torch.sin(torch.arange(n_bins, dtype=dt) * 1.7)
    F_c = fsr._f_calc(F_protein, F_mask, inv_d2, torch.tensor(0.35, dtype=dt),
                      torch.tensor(46.0, dtype=dt))
    F_obs = k_true[bins] * torch.exp(-0.25 * 10.0 * inv_d2) * F_c.abs()
    work = torch.rand(n, generator=g).numpy() > 0.05
    return F_obs, F_protein, F_mask, inv_d2, s_cart, work, bins


def test_binned_scales_recover_a_non_gaussian_falloff():
    F_obs, Fp, Fm, s2, s, work, bins = _synthetic()
    log = open(os.devnull, "w")
    single = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=True, log=log)
    binned = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=True, bins=bins,
                            log=log)
    free = ~work
    r = {}
    for name, fit in (("single", single), ("binned", binned)):
        m = fsr.model_amplitudes(fit, Fp, Fm, s2, s)
        r[name] = (fsr.r_factor(F_obs[work], m[work]),
                   fsr.r_factor(F_obs[free], m[free]))
    # Noise-free data. The binned fit is nearly exact -- not quite, because
    # the isotropic B is held at the single-scale estimate (10.6 here, truth
    # 10) and a scale that is constant across a bin cannot follow the rest of
    # its slope. The single scale misses by two orders of magnitude more.
    assert r["binned"][0] < 0.005 and r["binned"][1] < 0.005
    assert r["single"][0] > 0.05
    # The bins are applied to the free set too, not just fitted on work.
    assert binned["log_k_bins"] is not None and binned["bins"] is bins
    # The solvent scales stay single numbers and land on the truth.
    assert abs(float(binned["k_sol"]) - 0.35) < 0.01
    assert abs(float(binned["b_sol"]) - 46.0) < 1.0


def test_binned_fit_holds_the_isotropic_b():
    F_obs, Fp, Fm, s2, s, work, bins = _synthetic()
    log = open(os.devnull, "w")
    single = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=True, log=log)
    binned = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=True, bins=bins,
                            log=log)
    # The isotropic part of B_cart is degenerate with the bins and is held at
    # the single-scale value; only the anisotropy refines with them.
    iso = lambda fit: sum(float(p) for p in fit["u_params"][:3]) / 3  # noqa: E731
    assert abs(iso(binned) - iso(single)) < 1e-6


def test_resolution_bins_are_equal_count_and_ordered():
    s2 = torch.rand(1000, dtype=torch.float64)
    bins = fsr.resolution_bins(s2, 8)
    counts = torch.bincount(bins)
    assert len(counts) == 8 and counts.min() == counts.max() == 125
    # Bin 0 is the lowest resolution: its largest s^2 is below bin 1's smallest.
    assert s2[bins == 0].max() < s2[bins == 1].min()


COMPARE = os.path.join(REPO_ROOT, "lunus", "sf", "examples", "compare_gemmi")


def _mmtbx_bins(F):
    """cctbx's log_binning selections as one index per reflection."""
    import numpy as np
    sels = F.log_binning()
    idx = np.full(F.size(), -1)
    for i, sel in enumerate(sels):
        idx[np.array(sel)] = i
    assert (idx >= 0).all()
    return idx


@pytest.mark.skipif(not HAVE_CCTBX, reason="cctbx not installed")
@pytest.mark.skipif(not os.path.exists(os.path.join(COMPARE, "7FPV-sf.cif")),
                    reason="examples/compare_gemmi/7FPV-sf.cif not present")
@pytest.mark.parametrize("d_min, n_expected", [(None, 10), (3.0, 6)])
def test_log_bins_are_mmtbx_bins_on_7fpv(d_min, n_expected):
    """The binned scale must use the bins mmtbx.f_model scales by (cctbx's
    log_binning), not equal-count bins: same reflection in the same bin, on
    the observations the tool actually fits. 7FPV at 1.04 A has 10 such bins,
    and cut to 3 A, 6."""
    import io
    import numpy as np
    from iotbx.pdb import hierarchy

    xrs = hierarchy.input(file_name=os.path.join(COMPARE, "7FPV.pdb"),
                          sort_atoms=False).input.xray_structure_simple()
    F, free = fsr.read_observations(os.path.join(COMPARE, "7FPV-sf.cif"),
                                 d_min=d_min,
                                 crystal_symmetry=xrs.crystal_symmetry(),
                                 log=io.StringIO())
    d = np.array(F.d_spacings().data())
    inv_d2 = torch.tensor(1.0 / d ** 2, dtype=torch.float64)
    bins = fsr.assign_scale_bins("log", inv_d2, d)
    expected = _mmtbx_bins(F)
    assert int(bins.max()) + 1 == n_expected
    assert np.array_equal(bins.numpy(), expected), \
        "%d reflections binned differently from mmtbx" % int(
            (bins.numpy() != expected).sum())
    # And the partition mmtbx.f_model itself scales by, built as mmtbx_fit
    # builds it. Today that is f_obs().log_binning() with the defaults, so
    # this mostly guards against mmtbx changing its binning or reordering
    # f_obs (bin_selections does not change in update_all_scales with
    # remove_outliers=False).
    import mmtbx.f_model
    from cctbx.array_family import flex
    flags = F.customized_copy(data=flex.bool(free.tolist())) \
        .set_observation_type(None)
    fmodel = mmtbx.f_model.manager(f_obs=F, r_free_flags=flags,
                                   xray_structure=xrs)
    assert list(fmodel.f_obs().indices()) == list(F.indices())
    fm = np.full(F.size(), -1)
    for i, sel in enumerate(fmodel.bin_selections):
        fm[np.array(sel)] = i
    assert np.array_equal(bins.numpy(), fm), "not the bins fmodel scales by"


def test_log_bins_order_merge_and_small_data():
    g = torch.Generator().manual_seed(1)
    d = (1.0 / (0.002 + 0.9 * torch.rand(50000, generator=g,
                                         dtype=torch.float64))).sqrt().numpy()
    bins = fsr.log_resolution_bins(d)
    counts = torch.bincount(bins)
    assert 2 <= len(counts) <= 30 and int(counts.min()) >= 50
    # Bin 0 is the lowest resolution and the bins tile the range in order.
    for i in range(len(counts) - 1):
        assert d[(bins == i).numpy()].min() >= d[(bins == i + 1).numpy()].max()
    # Fine at low resolution, wide at high: the first bin is the smallest.
    assert int(counts[0]) <= int(counts[1:].min())
    # Too few reflections for two bins of the lowest size: one bin.
    assert int(fsr.log_resolution_bins(d[:150]).max()) == 0


def test_scale_bins_arg_and_equal_count_still_available():
    import argparse
    assert fsr.scale_bins_arg("log") == "log"
    assert fsr.scale_bins_arg("20") == 20 and fsr.scale_bins_arg("0") == 0
    for bad in ("-1", "logs", "2.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            fsr.scale_bins_arg(bad)
    s2 = torch.rand(1000, dtype=torch.float64)
    assert torch.equal(fsr.assign_scale_bins(20, s2, None),
                       fsr.resolution_bins(s2, 20))
    assert fsr.assign_scale_bins(0, s2, None) is None


@pytest.mark.skipif(not HAVE_CCTBX, reason="cctbx not installed")
@pytest.mark.parametrize("n_refl, kwargs", [
    (200, {}),                                    # one bin, at the boundary
    (201, {}),                                    # just over it
    (20000, {"n_lowest": 5, "min_per_bin": 1}),   # ~220 raw limits: rescaled to 30
    (20000, {"n_lowest": 30, "max_bins": 8, "min_per_bin": 400}),
])
def test_log_bins_equal_cctbx_in_every_branch(n_refl, kwargs):
    """The port against cctbx's own log_binning on synthetic reflection sets
    that reach the branches 7FPV does not: exactly 2*n_lowest reflections,
    the max_bins rescale, and the min-per-bin merge with custom settings."""
    import numpy as np
    from cctbx import crystal, miller
    from cctbx.array_family import flex

    cs = crystal.symmetry(unit_cell=(41, 53, 67, 90, 90, 90),
                          space_group_symbol="P1")
    full = miller.build_set(cs, anomalous_flag=False, d_min=1.2)
    rng = np.random.default_rng(n_refl)
    keep = np.sort(rng.choice(full.size(), size=n_refl, replace=False))
    ms = full.select(flex.size_t(keep.tolist()))
    cctbx_kw = {"n_reflections_in_lowest_resolution_bin":
                kwargs.get("n_lowest", 100),
                "max_number_of_bins": kwargs.get("max_bins", 30),
                "min_reflections_in_bin": kwargs.get("min_per_bin", 50)}
    expected = np.full(ms.size(), -1)
    for i, sel in enumerate(ms.log_binning(**cctbx_kw)):
        expected[np.array(sel)] = i
    d = np.array(ms.d_spacings().data())
    got = fsr.log_resolution_bins(d, **kwargs).numpy()
    assert np.array_equal(got, expected)
    if kwargs.get("n_lowest") == 5:
        assert int(got.max()) + 1 == 30          # the rescale was reached


def test_log_bins_zero_step_raises_instead_of_hanging():
    """cctbx's loop never ends when reflections n_lowest..2*n_lowest share
    one d spacing; the port says so instead."""
    import numpy as np
    d = np.concatenate([np.linspace(20.0, 10.0, 100), np.full(101, 5.0),
                        np.linspace(4.9, 2.0, 99)])
    with pytest.raises(ValueError, match="bin step is zero"):
        fsr.log_resolution_bins(d)


@pytest.mark.skipif(not HAVE_CCTBX, reason="cctbx not installed")
def test_moving_average_matches_mmtbx():
    """The smoother applied to the per-bin solvent curve is mmtbx's own."""
    from mmtbx.bulk_solvent.scaler import moving_average
    from cctbx.array_family import flex
    import numpy as np

    rng = np.random.default_rng(3)
    for n in (3, 5, 12, 30):
        for _ in range(20):
            x = list(rng.uniform(0, 0.5, n))
            ours = fsr.mmtbx_moving_average(x)
            theirs = list(moving_average(flex.double(x)))
            assert np.allclose(ours, theirs), (n, x)


def test_solvent_curve_rule_clamps_and_truncates():
    """mmtbx's rule: no negatives; from the first zero whose bin edge is
    below 3 A, zero outward; a zero above 3 A does not truncate."""
    k = torch.tensor([0.35, 0.34, -0.01, 0.30, 0.02, -0.05, 0.08, -0.02],
                     dtype=torch.float64)
    d_min = [8.0, 5.0, 4.2, 3.5, 2.9, 2.5, 2.2, 2.0]
    out = fsr.solvent_curve_rule(k, d_min, smooth=False)
    assert (out >= 0).all()
    # bin 2 is negative but its edge is 4.2 A: clamped to 0, not a cut-off
    assert float(out[2]) == 0.0 and float(out[3]) == 0.30
    # bin 5 is the first zero below 3 A: it and everything after are zero
    assert out[5:].abs().sum() == 0.0 and float(out[4]) == 0.02
    # with smoothing the surviving bins are averaged at extrema, the cut
    # found on the clamped values still holds, and nothing beyond it is
    # re-admitted
    out_s = fsr.solvent_curve_rule(k, d_min, smooth=True)
    assert (out_s >= 0).all() and out_s[5:].abs().sum() == 0.0
    assert float(out_s[0]) == 0.35 and out_s[:5].sum() > 0


def _synthetic_cliff(n=8000, seed=0, bump=0.0):
    """Amplitudes whose solvent scale is flat to 4 A and zero beyond, which
    no k_sol exp(-b_sol s^2/4) can follow. bump > 0 adds a spurious solvent
    scale of that size between 2.4 and 1.9 A, beyond a run of zeros: the
    kind of thing a free bin can pick up and the rule must remove."""
    g = torch.Generator().manual_seed(seed)
    dt = torch.float64
    d = 1.5 + 18.5 * torch.rand(n, generator=g, dtype=dt) ** 3   # most reflections at low d, as real data
    inv_d2 = 1.0 / d ** 2
    direction = torch.randn(n, 3, generator=g, dtype=dt)
    s_cart = direction / direction.norm(dim=1, keepdim=True) * inv_d2.sqrt()[:, None]
    F_protein = torch.complex(torch.randn(n, generator=g, dtype=dt),
                              torch.randn(n, generator=g, dtype=dt)) * 100.0
    F_mask = torch.complex(torch.randn(n, generator=g, dtype=dt),
                           torch.randn(n, generator=g, dtype=dt)) * 300.0
    k_mask_true = torch.where(d >= 4.0, torch.tensor(0.35, dtype=dt), torch.tensor(0.0, dtype=dt))
    if bump:
        k_mask_true = torch.where((d < 2.4) & (d >= 1.9), torch.tensor(bump, dtype=dt), k_mask_true)
    F_obs = torch.exp(-0.25 * 10.0 * inv_d2) * (F_protein + k_mask_true * F_mask).abs()
    work = torch.rand(n, generator=g).numpy() > 0.05
    bins = fsr.log_resolution_bins(d.numpy())
    return F_obs, F_protein, F_mask, inv_d2, s_cart, work, bins, d


def test_per_bin_solvent_recovers_a_cliff_the_exponential_cannot():
    """fit_scales(solvent_bins=True) follows a flat-then-zero solvent scale;
    the exponential binned fit does not. The truncation lands where the
    data put it."""
    import io
    F_obs, Fp, Fm, s2, s, work, bins, d = _synthetic_cliff()
    quiet = io.StringIO()
    expo = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=False, bins=bins,
                          solvent_bins=False, log=quiet)
    perbin = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=False, bins=bins,
                            solvent_bins=True, log=quiet)
    r = {}
    for name, fit in (("expo", expo), ("perbin", perbin)):
        m = fsr.model_amplitudes(fit, Fp, Fm, s2, s)
        r[name] = (fsr.r_factor(F_obs[work], m[work]), fsr.r_factor(F_obs[~work], m[~work]))
    assert r["perbin"][0] < 0.01 and r["perbin"][1] < 0.01, r
    assert r["expo"][0] > 3 * r["perbin"][0], r
    km = perbin["k_mask_bins"]
    assert km is not None and (km >= 0).all()
    d_min_bins = [float(d[(bins == i).numpy()].min()) for i in range(int(bins.max()) + 1)]
    for v, dm in zip(km.tolist(), d_min_bins):
        if dm >= 4.5:
            assert abs(v - 0.35) < 0.03, (v, dm)
        elif dm < 3.0:
            assert v == 0.0, (v, dm)
    # the reported k_sol/b_sol are the exponential nearest the curve at d >= 4 A
    assert abs(float(perbin["k_sol"]) - 0.35) < 0.03
    assert expo["k_mask_bins"] is None


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_per_bin_solvent_cut_removes_a_bump_beyond_the_first_zero(seed, monkeypatch):
    """A positive solvent scale beyond the first zero below 3 A is something
    the data can support (here it is put there on purpose) and the fit
    still removes it: everything from the cut is zero. The control is the
    same fit with no cut anywhere (the rule reduced to a clamp AND the fit's
    own cut disabled), which keeps the bump; so the difference is the cut,
    not the sign of solver leftovers in the zero zone, which the 0.005 snap
    turns into zeros whatever their sign (hence several seeds)."""
    import io
    F_obs, Fp, Fm, s2, s, work, bins, d = _synthetic_cliff(seed=seed, bump=0.08)
    n_bins = int(bins.max()) + 1
    d_min_bins = [float(d[(bins == i).numpy()].min()) for i in range(n_bins)]
    d_max_bins = [float(d[(bins == i).numpy()].max()) for i in range(n_bins)]
    bump_bins = [i for i in range(n_bins) if d_min_bins[i] >= 1.9 and d_max_bins[i] < 2.4]
    assert len(bump_bins) >= 2                  # bins lying wholly inside the bump

    with monkeypatch.context() as m:
        m.setattr(fsr, "solvent_curve_rule",
                  lambda k, dmin, smooth=True, d_cut=3.0: torch.clamp(k, min=0))
        m.setattr(fsr, "solvent_cut_index", lambda k, dmin, d_cut=3.0: len(k))
        no_cut = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=False, bins=bins,
                                solvent_bins=True, log=io.StringIO())
    assert all(no_cut["k_mask_bins"][i] > 0.04 for i in bump_bins)   # the bump is real

    fit = fsr.fit_scales(F_obs, Fp, Fm, s2, s, work, aniso=False, bins=bins,
                         solvent_bins=True, log=io.StringIO())
    km = fit["k_mask_bins"]
    cut = fsr.solvent_cut_index(km.tolist(), d_min_bins)
    assert cut < min(bump_bins) and d_min_bins[cut] < 3.0
    assert all(v == 0.0 for v in km[cut:].tolist())
    assert (km >= 0).all() and all(abs(float(km[i]) - 0.35) < 0.03
                                  for i in range(n_bins) if d_min_bins[i] >= 4.5)


def test_exponential_from_curve_reports_where_it_fitted():
    """The diagnostic exponential is fitted over the REFLECTIONS at d >= 4 A,
    weighted by count and including the d >= 4 part of a bin that straddles
    4 A, bounded as mmtbx bounds its own; with none there it says so."""
    g = torch.Generator().manual_seed(5)
    d = 2.0 + 10.0 * torch.rand(3000, generator=g, dtype=torch.float64) ** 2
    s2 = 1.0 / d ** 2
    edges = [12.0, 6.0, 4.5, 3.5, 2.8, 2.0]          # bin 2 (4.5-3.5 A) straddles 4 A
    bins = torch.zeros(3000, dtype=torch.long)
    for i in range(1, len(edges) - 1):
        bins[d < edges[i]] = i
    # the straddling bin sits well below its neighbour, so whether its d >= 4
    # reflections are counted changes the answer
    curve = torch.tensor([0.36, 0.34, 0.15, 0.10, 0.05], dtype=torch.float64)
    k, b, n = fsr.exponential_from_curve(curve, bins, s2)
    assert n == int((d >= 4.0).sum()) and n > 0
    # the straddling bin contributes its d >= 4 reflections
    assert int(((d >= 4.0) & (bins == 2)).sum()) > 0
    # brute force on the same reflections, direct cost, same grid
    sel = d >= 4.0
    kr, s2r = curve[bins[sel]], s2[sel]
    kg = torch.arange(0.0, 0.6001, 0.005, dtype=torch.float64)
    bg = torch.arange(0.0, 150.001, 1.0, dtype=torch.float64)
    cost = ((kg[:, None, None] * torch.exp(-0.25 * bg[None, :, None] * s2r) - kr) ** 2).sum(-1)
    i, j = divmod(int(torch.argmin(cost)), len(bg))
    assert (k, b) == (float(kg[i]), float(bg[j]))
    # and it is NOT what a fit over whole bins with edge >= 4 A would give
    sel_bins = bins <= 1
    kr2, s2r2 = curve[bins[sel_bins]], s2[sel_bins]
    cost2 = ((kg[:, None, None] * torch.exp(-0.25 * bg[None, :, None] * s2r2) - kr2) ** 2).sum(-1)
    i2, j2 = divmod(int(torch.argmin(cost2)), len(bg))
    assert (float(kg[i2]), float(bg[j2])) != (k, b)
    # bounds: k pinned at 0.6, b pinned at 150, and the no-data fallback
    k, b, n = fsr.exponential_from_curve(torch.full((5,), 0.9, dtype=torch.float64), bins, s2)
    assert k == 0.6
    # b pinned at 150: three reflections at 20 A see 0.5, a hundred at
    # exactly 4 A see 0, and every increase of b improves the fit
    s2_pin = torch.tensor([1 / 400.0] * 3 + [1 / 16.0] * 100, dtype=torch.float64)
    bins_pin = torch.tensor([0] * 3 + [1] * 100)
    k, b, n = fsr.exponential_from_curve(torch.tensor([0.5, 0.0], dtype=torch.float64), bins_pin, s2_pin)
    assert b == 150.0 and n == 103
    k, b, n = fsr.exponential_from_curve(torch.tensor([0.9, 0.0], dtype=torch.float64),
                                         torch.tensor([0, 1]),
                                         torch.tensor([1 / 9.0, 1 / 4.0], dtype=torch.float64))
    assert (k, b, n) == (0.6, 0.0, 0)        # fallback stays inside the bounds


@pytest.mark.parametrize("dt", [torch.float64, torch.float32])
@pytest.mark.parametrize("fit_solvent", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_coarse_scan_matches_the_loop(fit_solvent, seed, dt):
    """The batched scan picks the node the node-by-node loop picks, and the
    same scale at it, on synthetic data with the fit's own grids, in both
    precisions run() offers. In float32 the expanded residual loses a few
    digits to cancellation, hence the looser scale tolerance there."""
    F_obs, Fp, Fm, s2, s, work, bins = _synthetic(seed=seed)
    cdt = torch.complex128 if dt == torch.float64 else torch.complex64
    Fo, Fp, s2 = F_obs[work].to(dt), Fp[work].to(cdt), s2[work].to(dt)
    Fm = Fm[work].to(cdt) if fit_solvent else None
    grids = fsr.scan_grids(fit_solvent, s2)
    fast = fsr.coarse_scan(Fo, Fp, Fm, s2, *grids)
    slow = fsr.coarse_scan_reference(Fo, Fp, Fm, s2, *grids)
    assert fast[:3] == slow[:3], (fast, slow)
    tol = 1e-9 if dt == torch.float64 else 1e-4
    assert abs(fast[3] - slow[3]) < tol * max(1.0, abs(slow[3]))


@pytest.mark.skipif(not HAVE_CCTBX, reason="cctbx not installed")
@pytest.mark.skipif(not os.path.exists(os.path.join(COMPARE, "7FPV-sf.cif")),
                    reason="examples/compare_gemmi/7FPV-sf.cif not present")
def test_coarse_scan_matches_the_loop_on_7fpv(monkeypatch):
    """The same, on the real inputs run() hands fit_scales for 7FPV cut to
    3 A (captured by wrapping fit_scales), for the with-solvent fit."""
    import contextlib
    import io

    seen = {}
    real = fsr.fit_scales

    def spy(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso, **kw):
        if F_mask is not None and not seen:
            seen.update(Fo=F_obs[work], Fp=F_protein[work], Fm=F_mask[work], s2=inv_d2[work])
        return real(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso, **kw)
    monkeypatch.setattr(fsr, "fit_scales", spy)
    args = fsr.build_parser().parse_args(
        [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
         "--d-min", "3.0", "--aniso", "--shells", "3"])
    with contextlib.redirect_stdout(io.StringIO()):
        fsr.run(args)
    grids = fsr.scan_grids(True, seen["s2"])
    fast = fsr.coarse_scan(seen["Fo"], seen["Fp"], seen["Fm"], seen["s2"], *grids)
    slow = fsr.coarse_scan_reference(seen["Fo"], seen["Fp"], seen["Fm"], seen["s2"], *grids)
    assert fast[:3] == slow[:3], (fast, slow)
    assert abs(fast[3] - slow[3]) < 1e-9 * max(1.0, abs(slow[3]))
