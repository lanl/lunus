"""tools/fit_solvent_rfactor.py: fit_scales with and without resolution bins.

Synthetic, torch only, about a second: amplitudes are built from random
protein and mask structure factors with a known k_sol/b_sol and an overall
scale that rises and falls with resolution, which no Gaussian B can follow.
The binned fit must recover it and the single-scale fit must not.
"""

import os
import sys

import torch

from conftest import REPO_ROOT

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
