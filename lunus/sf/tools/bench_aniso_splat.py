#!/usr/bin/env python
"""
What anisotropic ADPs cost the REAL splat, on a real structure.

    OMP_NUM_THREADS=n PYTHONPATH=<repo root> python tools/bench_aniso_splat.py \\
        examples/compare_gemmi/7FPV.pdb --d-min 2.0 [--device cuda]

tools/bench_aniso_core.py measures the elementwise core in isolation and says
~2.9x compiled. docs/performance.md is emphatic that an isolated benchmark is
an UPPER BOUND on speed -- bench_splat once reported 63 ms for an operation
that cost 255 ms in situ -- so the number that matters is this one: the whole
splat, with the offset prefixes, the chunking, the scatter and the allocator
all present.

THREE CONFIGURATIONS, because the middle one is what a deposited structure
actually is:

    isotropic      every atom on the existing path (the baseline)
    as deposited   each atom on the path its own ADP calls for -- 7FPV is
                   1,814 anisotropic of 3,341, the rest hydrogens
    all anisotropic every atom forced onto the slower path, which is the
                   worst case and not a real one

The gap between the second and the first is the cost you would actually pay.

MEASURED on 7FPV, d_min 2.0, grid 54x72x150, float32, CPU:

    isotropic          5.3 ms   1.00x
    as deposited      12.6 ms   2.37x     <- 1,814 of 3,341 atoms anisotropic
    all anisotropic   13.0 ms   2.45x

so the in-situ cost, 2.37x, is close to the 2.9x the isolated core predicts --
this is one of the cases where the microbenchmark did not badly overstate.

THE MIXED ROW IS NEARLY THE WORST CASE, and that is the surprising part. Only
54% of ATOMS are anisotropic, so a natural guess is that the mixed model pays
about half the penalty. It pays nearly all of it, because the anisotropic
atoms are the EXPENSIVE ones: every non-hydrogen carries ANISOU, and weighting
by r^3 -- what candidate-voxel count scales as -- the anisotropic atoms are
78.4% of the work rather than 54%. Hydrogens are 46% of the atoms and 21.6% of
the cost.

Two consequences. Splitting the isotropic atoms onto the fast path is worth
about 0.08x here, not half; it is still worth having, because a model with NO
anisotropic ADPs -- any MD trajectory -- then pays exactly nothing. And a
deposited structure refined without riding hydrogens has no cheap atoms at
all, so it lands on the "all anisotropic" row.

A secondary effect is folded into the same number: splitting each element
group in two produces more, smaller chunks, so the batching is slightly less
efficient than either pure configuration.

It also reports the ADP extraction, which step 3 needs: u_star as stored in a
PDB is in the fractional reciprocal basis and is NOT what the kernel wants.
Feeding it straight in produces a plausible, wrong density.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def load_adps(pdb_path):
    """Coordinates, elements, occupancies and per-atom CARTESIAN U.

    THE CONVERSION IS THE POINT. A PDB stores anisotropic ADPs as u_star, in
    the fractional reciprocal basis; the kernel measures displacements in
    Cartesian angstroms and needs u_cart. They coincide only for a cubic cell
    with a == 1, so the error is invisible on a toy and wrong on a crystal.
    Isotropic scatterers are promoted to u_iso * I, which the kernel handles
    exactly -- but they are reported separately so they can stay on the fast
    path.
    """
    from cctbx import adptbx
    from iotbx.pdb import hierarchy

    xrs = hierarchy.input(file_name=pdb_path,
                          sort_atoms=False).input.xray_structure_simple()
    uc = xrs.unit_cell()
    is_aniso = np.array(xrs.use_u_aniso(), dtype=bool)
    u_equiv = np.array(xrs.extract_u_iso_or_u_equiv())

    n = xrs.scatterers().size()
    U = np.zeros((n, 3, 3))
    for i, sc in enumerate(xrs.scatterers()):
        if is_aniso[i]:
            c = adptbx.u_star_as_u_cart(uc, sc.u_star)     # u11,u22,u33,u12,u13,u23
            U[i] = [[c[0], c[3], c[4]],
                    [c[3], c[1], c[5]],
                    [c[4], c[5], c[2]]]
        else:
            U[i] = u_equiv[i] * np.eye(3)
    return xrs, U, is_aniso, u_equiv


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("--d-min", type=float, default=2.0)
    p.add_argument("--rate", type=float, default=1.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--blur", type=float, default=None,
                   help="default: recommended_blur for this grid and B")
    p.add_argument("--cutoff", type=float, default=0.01)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    if "OMP_NUM_THREADS" not in os.environ:
        print("WARNING: OMP_NUM_THREADS unset; on a CPU-limited container this "
              "is worth 3.6x. See docs/performance.md.\n", file=sys.stderr)

    from lunus.sf.cell_utils import (grid_shape_for_resolution, orth_matrix,
                                     recommended_blur)
    from lunus.sf.density_torch import splat_density
    from lunus.sf.elements import IT92_COEFFS
    from lunus.sf.kernel_torch import (build_atom_kernels_aniso_torch,
                                       build_atom_kernels_torch)

    dev, dtype = args.device, getattr(torch, args.dtype)
    xrs, U, is_aniso, u_equiv = load_adps(args.pdb)
    cell = xrs.unit_cell().parameters()
    elements = [s.scattering_type for s in xrs.scatterers()]
    present = sorted(set(elements))
    occ_np = np.array(xrs.scatterers().extract_occupancies())
    frac_np = np.array(xrs.sites_frac()).reshape(-1, 3)
    b_per_atom = u_equiv * (8.0 * np.pi ** 2)

    M_np = orth_matrix(*cell)
    grid = grid_shape_for_resolution(cell[0], cell[1], cell[2], args.d_min,
                                     rate=args.rate)
    blur = (args.blur if args.blur is not None
            else recommended_blur(float(b_per_atom.min()), grid, M_np))

    print("%s\n%d atoms, %d anisotropic (%.0f%%), grid %s, %s, blur %.1f"
          % (args.pdb, len(elements), is_aniso.sum(),
             100.0 * is_aniso.mean(), grid, args.dtype, blur))

    evals = np.linalg.eigvalsh(U)
    npd = int((evals.min(axis=1) <= 0).sum())
    print("ADP eigenvalues %.4f to %.4f A^2; %d non-positive-definite"
          % (evals.min(), evals.max(), npd))
    if npd and blur == 0.0:
        print("  -> the constant term has beta = blur, so these would raise. "
              "Pass --blur to proceed.")

    common = dict(grid_shape=grid, orth_matrix_np=M_np, cutoff=args.cutoff,
                  device=dev, dtype=dtype)
    A_i, lam, offs_i, rad_i, tw, e2i = build_atom_kernels_torch(
        elements, present, IT92_COEFFS, b_per_atom, blur, **common)
    A_a, L6, offs_a, rad_a, _, _ = build_atom_kernels_aniso_torch(
        elements, present, IT92_COEFFS, U, blur, **common)

    frac = torch.tensor(frac_np, dtype=dtype, device=dev)
    idx = torch.tensor([e2i[e] for e in elements], device=dev)
    occ = torch.tensor(occ_np, dtype=dtype, device=dev)
    orth = torch.tensor(M_np, dtype=dtype, device=dev)
    mask_dep = torch.tensor(is_aniso, device=dev)
    mask_all = torch.ones(len(elements), dtype=torch.bool, device=dev)
    compile_core = not args.no_compile

    print("\n  mean candidate radius: isotropic %.3f A, anisotropic %.3f A "
          "(%+.1f%%)" % (float(rad_i.mean()), float(rad_a.mean()),
                         100.0 * (float(rad_a.mean()) / float(rad_i.mean()) - 1)))

    configs = [
        ("isotropic", dict(atom_A=A_i, atom_lam=lam, elem_offsets=offs_i,
                           atom_radius_ang=rad_i, atom_L6=None,
                           aniso_mask=None)),
        ("as deposited", dict(atom_A=A_a, atom_lam=lam, elem_offsets=offs_a,
                              atom_radius_ang=rad_a, atom_L6=L6,
                              aniso_mask=mask_dep)),
        ("all anisotropic", dict(atom_A=A_a, atom_lam=lam,
                                 elem_offsets=offs_a, atom_radius_ang=rad_a,
                                 atom_L6=L6, aniso_mask=mask_all)),
    ]

    print("\n  %-18s %10s %10s %12s" % ("configuration", "ms", "x iso", "sum"))
    baseline = None
    with torch.no_grad():
        for name, kw in configs:
            def run():
                return splat_density(frac, idx, occ, kw["atom_A"],
                                     kw["atom_lam"], kw["elem_offsets"],
                                     kw["atom_radius_ang"], grid, orth, tw,
                                     compile_core=compile_core,
                                     atom_L6=kw["atom_L6"],
                                     aniso_mask=kw["aniso_mask"])
            g = run()                       # warm: compile, allocator, plans
            sync(dev)
            best = float("inf")
            for _ in range(args.reps):
                sync(dev)
                t0 = time.perf_counter()
                g = run()
                sync(dev)
                best = min(best, time.perf_counter() - t0)
            baseline = baseline or best
            print("  %-18s %10.1f %10.2f %12.6g"
                  % (name, best * 1e3, best / baseline, float(g.sum())))

    print("\n  The 'as deposited' row is the cost actually paid. 'all "
          "anisotropic'\n  is the worst case and not a real structure -- "
          "hydrogens have no ANISOU.")


if __name__ == "__main__":
    main()
