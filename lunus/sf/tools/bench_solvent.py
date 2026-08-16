#!/usr/bin/env python
"""
What bulk solvent costs at production scale, on a real structure.

    PYTHONPATH=<repo root> python tools/bench_solvent.py \\
        examples/compare_gemmi/top_bfac.pdb.gz \\
        --cell 88.451,88.451,39.823,90,90,90 --space-group P43 \\
        --grid 300,300,144 [--device cuda|mps] [--selection peptide]

Everything about the solvent path so far has been measured on toy systems --
at most 400 atoms on a 72^3 grid, in float64. This runs the real thing: the
135,834-atom compare_gemmi system on the 300 x 300 x 144 grid every number in
docs/performance.md was taken on, in float32, and reports

  * the cost of the mask and its FFT against the cost of the protein path it
    is added to (the design note's estimate was ~2 ms/frame on CUDA, which is
    arithmetic from an FFT timing, not a measurement);
  * peak device memory with and without solvent, since the mask and its
    transform are extra full-size grids and memory is the binding constraint
    for ensembles;
  * the mask occupancy, which is what says whether the model is doing
    anything at all.

A WARNING ABOUT THE INPUT, learned by running it. The compare_gemmi system
cannot exercise bulk solvent with ANY selection or cutoff. It is a solvated MD
box several crystallographic cells across -- its fractional coordinates span
-0.35 to 5.12 -- so folding it into the crystal cell stacks several protein
copies on top of each other, and symmetry expansion multiplies that again. The
resulting density has minimum 0.16 and median 2.5 e/A^3: there is no empty
space anywhere to call solvent. Dropping the waters does not help, because the
protein copies alone still fill it.

It is still the right input for the COST measurement, which depends on the grid
and the atom count rather than on what the mask contains. For a meaningful
occupancy, use a crystallographic model -- one asymmetric unit in its own cell,
with real solvent channels:

    PYTHONPATH=<repo root> python tools/bench_solvent.py 4wor.cif \
        --cell 48.499,48.499,63.430,90,90,90 --space-group "P 41" \
        --grid 160,160,208 --selection "not water"

which gives occupancy 0.51, a 258k-voxel shell, and |F_total|/|F_protein| =
0.918 -- the low-resolution contrast cancellation the model exists to
represent.

Timings synchronize around each phase; without that a GPU timer measures
dispatch rather than execution.
"""

import argparse
import time

import numpy as np
import torch


def load_structure(path, cell, space_group, selection_text):
    """Coordinates, per-atom B factors and elements, selected the way xtraj
    would select them."""
    import cctbx.sgtbx
    import cctbx.crystal as crystal
    from iotbx.pdb import hierarchy

    pdb_in = hierarchy.input(file_name=path, sort_atoms=False)
    symm = crystal.symmetry(
        unit_cell=tuple(cell),
        space_group_info=cctbx.sgtbx.space_group_info(symbol=space_group))
    xrs = pdb_in.input.xray_structure_simple(crystal_symmetry=symm)
    xrs.convert_to_isotropic()
    xrs.set_occupancies(1.0)

    sel = pdb_in.hierarchy.atom_selection_cache().selection(selection_text)
    xrs = xrs.select(sel)

    elements = [s.scattering_type for s in xrs.scatterers()]
    b_per_atom = np.array(xrs.extract_u_iso_or_u_equiv()) * (8.0 * np.pi ** 2)
    frac = np.array(xrs.sites_frac()).reshape(-1, 3)
    return elements, b_per_atom, frac, xrs


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Timer:
    def __init__(self, device):
        self.device, self.rows = device, []

    def __call__(self, name, fn):
        sync(self.device)
        t0 = time.time()
        out = fn()
        sync(self.device)
        self.rows.append((name, time.time() - t0))
        return out

    def report(self):
        total = sum(dt for _, dt in self.rows)
        print("  %-28s %10s %8s" % ("phase", "ms", "share"))
        for name, dt in self.rows:
            print("  %-28s %10.1f %7.1f%%" % (name, 1e3 * dt, 100 * dt / total))
        print("  %-28s %10.1f" % ("total", 1e3 * total))
        return total


def peak_memory(device):
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("--cell", required=True)
    p.add_argument("--space-group", default="P1")
    p.add_argument("--grid", required=True, help="Nu,Nv,Nw")
    p.add_argument("--device", default="cpu")
    p.add_argument("--selection", default="all",
                   help="cctbx atom selection; 'peptide' excludes water")
    p.add_argument("--cutoff-frac", type=float, default=0.01,
                   help="mask cutoff as a fraction of peak density")
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()

    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.density_torch import splat_density
    from lunus.sf.elements import IT92_COEFFS
    from lunus.sf.kernel_torch import build_atom_kernels_torch
    from lunus.sf.solvent_torch import (
        SolventModel, mask_occupancy, shell_voxels, solvent_mask,
    )
    from lunus.sf.structure_factor_torch import compute_fcalc
    from lunus.sf.symmetry_torch import (
        build_grid_ops_from_cctbx, symmetrize_sum,
    )
    from cctbx import miller

    cell = [float(x) for x in args.cell.split(",")]
    grid = tuple(int(x) for x in args.grid.split(","))
    dev, dtype = args.device, torch.float32

    elements, b_per_atom, frac_np, xrs = load_structure(
        args.pdb, cell, args.space_group, args.selection)
    M_np = orth_matrix(*cell)
    print("%s\n%d atoms after selection %r, B %.1f-%.1f, grid %s, %s, float32"
          % (args.pdb, len(elements), args.selection, b_per_atom.min(),
             b_per_atom.max(), grid, dev))

    present = sorted(set(elements))
    atom_A, atom_lam, offsets, atom_r, taper_w, e2i = build_atom_kernels_torch(
        elements, present, IT92_COEFFS, b_per_atom, 0.0, grid, M_np,
        cutoff=0.01, device=dev, dtype=dtype)
    frac = torch.tensor(frac_np, dtype=dtype, device=dev)
    element_idx = torch.tensor([e2i[e] for e in elements],
                               dtype=torch.long, device=dev)
    occ = torch.ones(len(elements), dtype=dtype, device=dev)
    orth = torch.tensor(M_np, dtype=dtype, device=dev)
    volume = float(np.linalg.det(M_np))

    grid_ops = build_grid_ops_from_cctbx(xrs.space_group(), grid)
    miller_set = miller.build_set(
        crystal_symmetry=xrs.crystal_symmetry(), anomalous_flag=False,
        d_min=2.0)
    hkl = torch.tensor(np.array(miller_set.indices()), dtype=torch.long,
                       device=dev)
    print("%d symmetry ops beyond identity, %d reflections to d_min 2.0\n"
          % (len(grid_ops), hkl.shape[0]))

    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def _one_pass():
        d = splat_density(
            frac, element_idx, occ, atom_A, atom_lam, offsets, atom_r,
            grid, orth, taper_w, compile_core=False)
        if grid_ops:
            d = symmetrize_sum(d, grid_ops)
        f = compute_fcalc(d, volume, hkl, orth)
        c = args.cutoff_frac * float(d.max())
        SolventModel(cutoff=c, taper_width=0.5 * c,
                     check_occupancy=False).apply(d, f, volume, hkl, orth)

    with torch.no_grad():
        # One complete pass before the clock starts. Every phase here has a
        # one-time cost on first use -- cuFFT builds its plan, torch.linalg.inv
        # inside reciprocal_inv_d2 loads cuSOLVER, the allocator grows its pool
        # -- and timing a cold pipeline reports those as the cost of the model.
        # Left uncorrected it read as a 3,165-atom splat taking 137.6 ms, which
        # is twice what 135,834 atoms take warm.
        _one_pass()
        sync(dev)

        t = Timer(dev)
        density = t("splat", lambda: splat_density(
            frac, element_idx, occ, atom_A, atom_lam, offsets, atom_r,
            grid, orth, taper_w, compile_core=False))
        if grid_ops:
            density = t("symmetrize",
                        lambda: symmetrize_sum(density, grid_ops))
        F_protein = t("fft+extract (protein)", lambda: compute_fcalc(
            density, volume, hkl, orth))
        mem_protein = peak_memory(dev)

        cutoff = args.cutoff_frac * float(density.max())
        model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff,
                             check_occupancy=False)
        mask = t("mask", lambda: solvent_mask(density, cutoff, 0.5 * cutoff))
        F_total = t("solvent (mask FFT + combine)", lambda: model.apply(
            density, F_protein, volume, hkl, orth)[0])
        total = t.report()
        mem_total = peak_memory(dev)

    occupancy = float(mask_occupancy(mask))
    print("\ncutoff %.5g (%.1e of peak %.3f), occupancy %.4f, shell %d voxels"
          % (cutoff, args.cutoff_frac, float(density.max()), occupancy,
             shell_voxels(mask)))
    if occupancy < 0.05:
        print("  ^ EMPTY MASK: with this selection the atoms fill the cell, so")
        print("    F_mask is ~0 and the solvent term contributes nothing.")
    solvent_ms = 1e3 * sum(dt for n, dt in t.rows if n in ("mask", "solvent (mask FFT + combine)"))
    print("solvent adds %.1f ms to a %.1f ms configuration: %.1f%%"
          % (solvent_ms, 1e3 * total, 100 * solvent_ms / (1e3 * total)))
    if dev.startswith("cuda"):
        print("peak device memory: %.0f MB protein-only, %.0f MB with solvent"
              % (mem_protein, mem_total))

    print("\n|F_protein| mean %.4g, |F_total| mean %.4g, ratio %.4f"
          % (F_protein.abs().mean(), F_total.abs().mean(),
             float(F_total.abs().mean() / F_protein.abs().mean())))

    # Repeat the solvent block alone, the way bench_splat times its kernel.
    if args.reps > 1:
        with torch.no_grad():
            times = []
            for _ in range(args.reps):
                sync(dev)
                t0 = time.time()
                model.apply(density, F_protein, volume, hkl, orth)
                sync(dev)
                times.append(time.time() - t0)
        print("solvent block repeated: best %.1f ms of %d"
              % (1e3 * min(times), args.reps))


if __name__ == "__main__":
    main()
