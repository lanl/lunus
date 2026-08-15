#!/usr/bin/env python
"""
How much of the solvent mask's contribution to diffuse scattering is physics,
and how much is the grid?

    python tools/study_diffuse_solvent.py [--atoms 200] [--configs 8]

Background. Diffuse is a VARIANCE, <|F|^2> - |<F>|^2, so any per-configuration
numerical noise adds a positive bias instead of averaging out the way it does
in the Bragg mean. The solvent mask is the obvious suspect: it is a nearly
discontinuous function of position, so it is far less bandlimited than the
smooth atomic density and correspondingly worse sampled on the same grid. This
script measures the effect rather than assuming it, as a function of the taper
width, and reports it next to the size of the physical solvent contribution it
would contaminate.

Two instruments, neither of which is a null test:

  TRANSLATION NON-COMMUTATION. Displace every configuration by the SAME
  sub-voxel vector. For the underlying model this multiplies every F(h) by one
  phase, so <|F|^2> and |<F>|^2 -- and therefore diffuse -- are unchanged
  exactly. The grid pipeline does not reproduce that: splat_density samples
  each atom at fixed voxel centres, so a displaced structure is RESAMPLED
  rather than shifted, and recovering the true shift would need interpolation.
  What comes out is therefore not zero, and its size is how far the discretized
  calculation is from a translation-invariant one. That is a legitimate
  discretization diagnostic and NOT a bound on the error against the truth.

  GRID REFINEMENT. Recompute on a finer grid and compare. This is the honest
  reference -- the finer answer is closer to the continuum limit -- but it
  moves the sampling of the density and the mask together, so it cannot
  separate them. Use both: refinement says how converged the answer is, the
  translation test says how much of what is left is grid alignment rather than
  structure.

Both are differenced with and without solvent, which is what isolates the
mask's share from the density's own sampling error.

See docs/solvent-design.md, "Diffuse is a variance, so mask noise biases it
upward".
"""

import argparse

import numpy as np
import torch

from lunus.sf.cell_utils import orth_matrix
from lunus.sf.density_torch import splat_density
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.solvent_torch import SolventModel, mask_occupancy, shell_voxels, solvent_mask
from lunus.sf.structure_factor_torch import (
    mean_and_diffuse, structure_factors_batch,
)

CELL = (30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
B_ISO = 20.0
DTYPE = torch.float64


def blob(n, radius=8.0, seed=0):
    """A molecule-shaped cluster: n atoms in a sphere at the cell centre with a
    1.4 A minimum separation, so there is a real solvent boundary to get wrong.
    Uniformly random coordinates would have no boundary at all."""
    rng = np.random.default_rng(seed)
    centre = np.array([15.0, 15.0, 15.0])
    pts = []
    while len(pts) < n:
        u = rng.normal(size=3)
        p = centre + radius * rng.random() ** (1 / 3.0) * u / np.linalg.norm(u)
        if all(np.linalg.norm(p - q) > 1.4 for q in pts):
            pts.append(p)
    return np.array(pts)


def ensemble(cart, n_configs, sigma=0.3, seed=1):
    """n_configs copies with independent Gaussian displacements -- the internal
    disorder whose variance IS the diffuse signal."""
    rng = np.random.default_rng(seed)
    return cart[None, :, :] + sigma * rng.standard_normal(
        (n_configs,) + cart.shape)


def setup(grid, cart_batch):
    M_np = orth_matrix(*CELL)
    n_atoms = cart_batch.shape[1]
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, B_ISO, 0.0, grid, M_np, cutoff=0.01,
        device="cpu", dtype=DTYPE,
    )
    idx = torch.zeros(n_atoms, dtype=torch.long)
    frac = cart_batch @ np.linalg.inv(M_np).T
    return dict(
        frac=torch.tensor(frac, dtype=DTYPE), element_idx=idx,
        occ=torch.ones(n_atoms, dtype=DTYPE),
        atom_A=A[idx], atom_lam=lam[idx], elem_offsets=offsets,
        atom_radius_ang=radius[idx], grid_shape=grid,
        orth=torch.tensor(M_np, dtype=DTYPE),
        cell_volume=float(np.linalg.det(M_np)), taper_width=taper_w,
    )


def hkl_low_resolution(hmax=6):
    """Where a bulk solvent term contributes at all."""
    return torch.tensor(
        [[h, k, l] for h in range(-hmax, hmax + 1)
         for k in range(-hmax, hmax + 1) for l in range(-hmax, hmax + 1)
         if (h, k, l) != (0, 0, 0)],
        dtype=torch.long,
    )


def diffuse_of(s, hkl, solvent):
    F = structure_factors_batch(
        s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
        s["cell_volume"], hkl, s["taper_width"],
        compile_core=False, solvent=solvent,
    )
    return mean_and_diffuse(F)[1]


def rel(a, b):
    """Mean absolute difference relative to the mean magnitude of b."""
    return float((a - b).abs().mean() / b.abs().mean())


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", type=int, default=200)
    p.add_argument("--configs", type=int, default=8)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--fine-grid", type=int, default=72)
    p.add_argument("--sigma", type=float, default=0.3,
                   help="displacement of each configuration, A")
    p.add_argument("--shift", type=float, default=0.37,
                   help="translation applied to every configuration, in VOXELS")
    args = p.parse_args()

    cart = blob(args.atoms)
    batch = ensemble(cart, args.configs, sigma=args.sigma)
    hkl = hkl_low_resolution()
    grid = (args.grid,) * 3
    voxel = CELL[0] / args.grid

    s = setup(grid, batch)
    rho0 = splat_density(
        s["frac"][0], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], grid, s["orth"],
        s["taper_width"], compile_core=False,
    )
    peak = float(rho0.max())
    cutoff = 0.01 * peak            # the parity study's recommendation

    # Same sub-voxel translation applied to every configuration.
    shifted = batch + args.shift * voxel
    s_shift = setup(grid, shifted)
    s_fine = setup((args.fine_grid,) * 3, batch)
    rho_fine = splat_density(
        s_fine["frac"][0], s_fine["element_idx"], s_fine["occ"],
        s_fine["atom_A"], s_fine["atom_lam"], s_fine["elem_offsets"],
        s_fine["atom_radius_ang"], (args.fine_grid,) * 3, s_fine["orth"],
        s_fine["taper_width"], compile_core=False,
    )
    cutoff_fine = 0.01 * float(rho_fine.max())

    print(__doc__.split("\n\n")[0])
    print("\n%d atoms, %d configurations, sigma %.2f A, cell %.0f A"
          % (args.atoms, args.configs, args.sigma, CELL[0]))
    print("grid %d^3 (%.3f A voxels), refinement grid %d^3, translation %.2f voxels"
          % (args.grid, voxel, args.fine_grid, args.shift))
    print("cutoff %.4g = 1%% of peak density %.3f" % (cutoff, peak))

    d_none = diffuse_of(s, hkl, None)
    d_none_shift = diffuse_of(s_shift, hkl, None)
    d_none_fine = diffuse_of(s_fine, hkl, None)
    print("\nno solvent: translation non-commutation %.2e, grid refinement %.2e"
          % (rel(d_none_shift, d_none), rel(d_none_fine, d_none)))

    print("\n%-12s %10s %12s %12s %12s %10s" % (
        "width/cutoff", "shell vox", "solvent sig", "translation", "refinement",
        "sig/noise"))
    for w_frac in (0.05, 0.1, 0.25, 0.5, 0.9):
        model = SolventModel(cutoff=cutoff, taper_width=w_frac * cutoff)
        model_fine = SolventModel(cutoff=cutoff_fine,
                                  taper_width=w_frac * cutoff_fine)
        d = diffuse_of(s, hkl, model)
        d_shift = diffuse_of(s_shift, hkl, model)
        d_fine = diffuse_of(s_fine, hkl, model_fine)

        signal = rel(d, d_none)                 # how much solvent changes diffuse
        noise = rel(d_shift, d)                 # how much grid alignment does
        refine = rel(d_fine, d)
        nvox = shell_voxels(
            solvent_mask(rho0, cutoff, w_frac * cutoff))
        print("%-12.2f %10d %12.3e %12.3e %12.3e %10.1f"
              % (w_frac, nvox, signal, noise, refine,
                 signal / noise if noise > 0 else float("inf")))

    print("\nsolvent signal = |diffuse(solvent) - diffuse(none)| / |diffuse(none)|")
    print("translation    = |diffuse(shifted) - diffuse(unshifted)| / |diffuse|")
    print("refinement     = |diffuse(fine grid) - diffuse(coarse)| / |diffuse|")
    print("A width is usable when the solvent signal is well clear of both.")


if __name__ == "__main__":
    main()
