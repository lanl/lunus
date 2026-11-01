#!/usr/bin/env python
"""
|F| three ways on one model: cctbx direct summation, gemmi, lunus.sf.

    PYTHONPATH=<repo root> python tools/compare_three_way.py \\
        examples/compare_gemmi/7FPV.pdb --d-min 1.5

tools/compare_icalc_mtz.py compares two grid calculations with each other and
cannot say which is right. This one adds the exact answer -- cctbx direct
summation over the same reflection list, no grid, no cutoff -- and reports,
for each density cutoff, how far gemmi (hard cutoff at its own radius) and
lunus.sf (hard, and with the taper) each sit from it, and from each other.
Same grid for both grid methods (gemmi's choice for the cutoff and d_min),
ADPs flattened to isotropic so all three see identical atoms, R on
amplitudes after one least-squares scale, by resolution shell.

MEASURED on 7FPV as deposited, 3,341 atoms, 25,552 reflections to 1.5 A,
2026-11-01, with the taper beginning at the cutoff threshold:

    cutoff   gemmi vs direct   lunus hard vs direct   lunus taper vs direct   taper vs gemmi
    1e-2         0.0104              0.0131                 0.0111               0.0024
    1e-3         0.0012              0.0017                 0.0014               0.0004
    1e-4         0.0003              0.0003                 0.0003               0.0000
    1e-5         0.0002              0.0002                 0.0002               0.0000

At 1e-4 and below the three are indistinguishable; the residual against
direct summation (0.0009 in the top shell) is the same for every grid
method, so it is the grid and the FFT. At coarse cutoffs the hard-cutoff
lunus.sf is FURTHER from the exact answer than gemmi, and --radii shows
why: gemmi's radius estimate overshoots, placing its hard edge where the
density is 0.65-0.9x the cutoff, i.e. 0.05-0.1 A beyond the radius at
which the density equals the cutoff, which lunus.sf solves for exactly.
gemmi therefore keeps a little more tail. The taper on [R, R + 0.1 A]
keeps about the same tail -- for carbon at B = 20, cutoff 1e-2, gemmi's
edge is at 2.249 A and lunus.sf's tapered window ends at 2.259 A -- which
is why the tapered lunus.sf lands close to gemmi and closer to direct
summation than its own hard cutoff.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_solvent_rfactor import read_model, scattering_table  # noqa: E402


def r_factor(a, ref):
    k = (a * ref).sum() / (a * a).sum()
    return float(np.abs(k * a - ref).sum() / ref.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("--d-min", type=float, default=1.5)
    p.add_argument("--cutoffs", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5])
    p.add_argument("--widths", type=float, nargs="+", default=[1e-6, 0.1],
                   help="lunus.sf taper widths, A; 1e-6 is a hard cutoff")
    p.add_argument("--shells", type=float, nargs="+", default=[4.0, 2.5, 2.0, 1.7],
                   help="inner shell edges, A; d_min closes the last")
    p.add_argument("--radii", action="store_true",
                   help="also print gemmi's cutoff radius against lunus.sf's "
                        "for a few atoms, and the density at each")
    args = p.parse_args()

    import gemmi
    from cctbx import miller
    from scitbx.array_family import flex

    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.density_torch import compute_density
    from lunus.sf.kernel import cutoff_radius, precalculate_density_iso
    from lunus.sf.kernel_torch import build_atom_kernels_torch
    from lunus.sf.structure_factor_torch import compute_fcalc
    from lunus.sf.symmetry_torch import build_grid_ops_from_cctbx, symmetrize_sum

    null = open(os.devnull, "w")
    xrs, elements, b, occ, frac, _, _, _ = read_model(args.pdb, log=null)
    ms = miller.build_set(crystal_symmetry=xrs.crystal_symmetry(),
                          anomalous_flag=False, d_min=args.d_min)
    ms = ms.select(flex.bool([h != (0, 0, 0) for h in ms.indices()]))
    hkl = np.array(list(ms.indices()))
    d = np.array(ms.d_spacings().data())
    xrs.scattering_type_registry(table="it1992")
    t0 = time.time()
    F_direct = np.abs(np.array(ms.structure_factors_from_scatterers(
        xray_structure=xrs, algorithm="direct").f_calc().data()))
    print("%s: %d atoms (isotropic), %d reflections to %.2f A; direct summation %.1f s"
          % (args.pdb, len(elements), len(hkl), args.d_min, time.time() - t0))

    st = gemmi.read_structure(args.pdb)
    atoms = [cra.atom for cra in st[0].all()]
    assert len(atoms) == len(elements), "gemmi and cctbx read different atom counts"
    for a, bb in zip(atoms, b):
        a.b_iso = float(bb)
        a.aniso = gemmi.SMat33f(0, 0, 0, 0, 0, 0)

    def gemmi_F(cutoff):
        calc = gemmi.DensityCalculatorX()
        calc.d_min, calc.cutoff = args.d_min, cutoff
        calc.grid.unit_cell = st.cell
        calc.grid.spacegroup = st.find_spacegroup()
        calc.put_model_density_on_grid(st[0])
        rho = np.array(calc.grid, copy=False)
        grid = (calc.grid.nu, calc.grid.nv, calc.grid.nw)
        G = np.fft.ifftn(rho) * st.cell.volume
        return np.abs(G[hkl[:, 0] % grid[0], hkl[:, 1] % grid[1], hkl[:, 2] % grid[2]]), grid, calc

    cell = xrs.unit_cell().parameters()
    M = orth_matrix(*cell)
    present = sorted(set(elements))
    coeffs = scattering_table(present, null)
    grid_ops_cache = {}

    def lunus_F(cutoff, width, grid):
        A, lam, offs, r, tw, e2i = build_atom_kernels_torch(
            elements, present, coeffs, b, 0.0, grid, M, cutoff=cutoff,
            taper_width=width, device="cpu", dtype=torch.float64)
        idx = torch.tensor([e2i[e] for e in elements])
        orth = torch.tensor(M)
        dens = compute_density(torch.tensor(frac), idx, torch.tensor(occ), A, lam,
                               offs, r, grid, orth, tw, compile_core=False)
        if grid not in grid_ops_cache:
            grid_ops_cache[grid] = build_grid_ops_from_cctbx(xrs.space_group(), grid)
        dens = symmetrize_sum(dens, grid_ops_cache[grid])
        with torch.no_grad():
            F = compute_fcalc(dens, float(np.linalg.det(M)), torch.tensor(hkl), orth)
        return F.abs().numpy()

    edges = [np.inf] + sorted(args.shells, reverse=True) + [args.d_min]
    sels = [(d < hi) & (d >= lo) for hi, lo in zip(edges[:-1], edges[1:])]

    def row(name, cutoff, label, a, ref):
        print("%-24s %8g %-7s %8.4f | %s"
              % (name, cutoff, label, r_factor(a, ref),
                 "  ".join("%.4f" % r_factor(a[s], ref[s]) for s in sels)))

    print("shell edges (A): %s" % edges[1:])
    print("%-24s %8s %-7s %8s | %s" % ("comparison", "cutoff", "taper", "R all",
                                       "R by shell, low to high resolution"))
    for cutoff in args.cutoffs:
        Fg, grid, calc = gemmi_F(cutoff)
        print("  [cutoff %g: grid %s]" % (cutoff, grid))
        row("gemmi vs direct", cutoff, "hard", Fg, F_direct)
        for width in args.widths:
            label = "hard" if width <= 1e-6 else "%g A" % width
            Fl = lunus_F(cutoff, width, grid)
            row("lunus.sf vs direct", cutoff, label, Fl, F_direct)
            row("lunus.sf vs gemmi", cutoff, label, Fl, Fg)
        if args.radii:
            print("  radii: gemmi's estimate against the radius where density = cutoff")
            for el, bb in (("C", 20.0), ("C", 50.0), ("O", 20.0), ("S", 20.0), ("H", 20.0)):
                a = gemmi.Atom()
                a.element, a.b_iso, a.occ = gemmi.Element(el), bb, 1.0
                rg = calc.estimate_radius(a)
                A, lam = precalculate_density_iso(coeffs[el] if el in coeffs
                                                  else scattering_table([el], null)[el], bb)
                rl = cutoff_radius(A, lam, cutoff=cutoff)
                rho = lambda r: float(np.sum(A * np.exp(-lam * r * r)))
                print("    %s B=%3.0f  gemmi %.3f A (density %.2e)   lunus.sf %.3f A (density %.2e)"
                      % (el, bb, rg, rho(rg), rl, rho(rl)))


if __name__ == "__main__":
    main()
