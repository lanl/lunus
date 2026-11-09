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
