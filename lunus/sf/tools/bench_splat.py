"""
Benchmarks density_torch.splat_density against gemmi's DensityCalculatorX on a
real structure, with a phase breakdown, so performance claims stay checkable
rather than remembered.

Reports atom-voxel PAIRS as well as wall time: pairs are the unit of work here
(one 5-Gaussian evaluation each), so pairs/second is what compares the two
implementations fairly when they choose slightly different cutoff radii.

Usage:
    python bench_splat.py examples/compare_gemmi/top_bfac.pdb.gz \\
        --cell 34.196,45.558,99.044,90,90,90 --grid 120,160,360

Omit --grid to size it from --d-min the way xtraj.py does. --no-gemmi skips
the reference run (e.g. where gemmi is not installed).
"""

import argparse
import time

import numpy as np
import torch


def load_structure(pdb_path, cell):
    import iotbx.pdb
    import cctbx.crystal as crystal
    import cctbx.sgtbx
    xrs = iotbx.pdb.input(file_name=pdb_path).xray_structure_simple(
        crystal_symmetry=crystal.symmetry(
            unit_cell=tuple(cell),
            space_group_info=cctbx.sgtbx.space_group_info(symbol="P1")))
    elements = [s.scattering_type for s in xrs.scatterers()]
    b_per_atom = np.array(xrs.extract_u_iso_or_u_equiv()) * (8.0 * np.pi ** 2)
    # The REAL fractional coordinates, wrapped into the cell. This matters a
    # great deal on a GPU: the scatter uses index_add_, which is atomic, so the
    # cost depends on how many atoms land on the same voxels. A real structure
    # is densely clustered and contends heavily; uniformly random coordinates
    # spread the writes out and measure a much easier problem.
    frac = np.array(xrs.sites_frac()).reshape(-1, 3) % 1.0
    return elements, b_per_atom, frac


def _sync(device):
    """GPU work is queued asynchronously; without this the timer measures
    dispatch rather than execution."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def bench_torch(elements, b_per_atom, cell, grid, cutoff, max_atoms, reps, compile_core,
                device="cpu", max_pairs=None, frac_np=None):
    from lunus.sf.elements import IT92_COEFFS
    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.kernel_torch import build_atom_kernels_torch
    from lunus.sf.density_torch import splat_density

    dtype = torch.float32          # MPS has no float64 at all
    M_np = orth_matrix(*cell)
    present = sorted(set(elements))

    t0 = time.time()
    atom_A, atom_lam, elem_offsets, atom_r, taper_w, e2i = build_atom_kernels_torch(
        elements, present, IT92_COEFFS, b_per_atom, 0.0, grid, M_np,
        cutoff=cutoff, device=device, dtype=dtype,
    )
    setup_s = time.time() - t0

    if frac_np is None:      # --random-coords: the old, easier, unrealistic case
        frac_np = np.random.default_rng(0).random((len(elements), 3))
    frac = torch.tensor(frac_np, dtype=dtype, device=device)
    occ = torch.ones(len(elements), dtype=dtype, device=device)
    element_idx = torch.tensor([e2i[e] for e in elements], dtype=torch.long, device=device)
    M = torch.tensor(M_np, dtype=dtype, device=device)

    # pairs actually evaluated: each atom uses the offset prefix its own
    # radius plus its own sub-voxel remainder reaches
    grid_t = torch.tensor(grid, dtype=dtype, device=device)
    base = torch.round(frac * grid_t)
    c_rem = (frac - base / grid_t) @ M.T
    reach = atom_r + c_rem.norm(dim=-1)
    pairs = 0
    for e in present:
        eo = elem_offsets[e2i[e]]
        sel = element_idx == e2i[e]
        pairs += int(torch.searchsorted(
            eo.c_off_norm, reach[sel].to(eo.c_off_norm.dtype), right=True).sum())

    args = (frac, element_idx, occ, atom_A, atom_lam, elem_offsets, atom_r,
            grid, M, taper_w)
    kwargs = dict(max_atoms_per_batch=max_atoms, compile_core=compile_core)
    if max_pairs is not None:
        kwargs["max_pairs_per_batch"] = max_pairs

    # splat_density's own count of what the chunked batching evaluates, as
    # against the ideal count above. xtraj reports the same pair of numbers
    # per frame under torch_splat_stats=True, so the two are comparable and
    # can settle whether the splat's in-situ cost is extra work or slower
    # work. Counting is free; it reads values already on the host.
    stats = {}
    kwargs["stats"] = stats

    with torch.no_grad():
        t0 = time.time()
        g = splat_density(*args, **kwargs)
        _sync(device)
        first_s = time.time() - t0                      # includes any compile
        times = []
        for _ in range(reps):
            t0 = time.time()
            g = splat_density(*args, **kwargs)
            _sync(device)
            times.append(time.time() - t0)

    return dict(setup=setup_s, first=first_s, best=min(times), pairs=pairs,
                pairs_actual=stats["pairs_actual"], chunks=stats["chunks"],
                total=float(g.sum()))


def bench_gemmi(pdb_path, cell, d_min, cutoff, reps):
    import gemmi
    import iotbx.pdb
    import cctbx.crystal as crystal
    import cctbx.sgtbx

    xrs = iotbx.pdb.input(file_name=pdb_path).xray_structure_simple(
        crystal_symmetry=crystal.symmetry(
            unit_cell=tuple(cell),
            space_group_info=cctbx.sgtbx.space_group_info(symbol="P1")))
    coords = xrs.sites_cart()
    u = xrs.extract_u_iso_or_u_equiv()
    occs = xrs.scatterers().extract_occupancies()
    els = [s.scattering_type for s in xrs.scatterers()]

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = "P 1"
    model, chain, res = gemmi.Model("1"), gemmi.Chain("A"), gemmi.Residue()
    res.name, res.seqid = "ALL", gemmi.SeqId("1")
    for k in range(len(els)):
        a = gemmi.Atom()
        a.name, a.element = "X", gemmi.Element(els[k])
        a.pos = gemmi.Position(*coords[k])
        a.b_iso = u[k] * 8 * np.pi ** 2
        a.occ = occs[k]
        res.add_atom(a)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)

    calc = gemmi.DensityCalculatorX()
    calc.d_min, calc.cutoff = d_min, cutoff
    calc.grid.unit_cell = st.cell
    calc.grid.spacegroup = st.find_spacegroup()

    times = []
    for _ in range(reps):
        t0 = time.time()
        calc.put_model_density_on_grid(st[0])
        times.append(time.time() - t0)
    arr = np.array(calc.grid, copy=False)
    return dict(best=min(times), grid=arr.shape, total=float(arr.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("--cell", required=True,
                   help="a,b,c,alpha,beta,gamma")
    p.add_argument("--grid", default=None, help="Nu,Nv,Nw (default: from --d-min)")
    p.add_argument("--d-min", type=float, default=0.9)
    p.add_argument("--cutoff", type=float, default=0.01)
    p.add_argument("--max-atoms", type=int, default=10000)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--device", default="cpu",
                   help="cpu, mps, cuda, cuda:0 ... (gemmi is always CPU)")
    p.add_argument("--no-gemmi", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="also benchmark the torch.compile path. Off by default: "
                        "the attempt costs ~10 s on the first call and currently "
                        "fails on CUDA (no cuda.h) and on MPS (inductor's Metal "
                        "backend), falling back to eager either way.")
    p.add_argument("--random-coords", action="store_true",
                   help="splat uniformly random coordinates instead of the "
                        "structure's own. Much faster on a GPU because the "
                        "atomic scatter barely contends -- and correspondingly "
                        "unrepresentative of a real run.")
    p.add_argument("--matmul-precision", default="highest",
                   choices=["highest", "high", "medium"],
                   help="torch.set_float32_matmul_precision(). 'high' is TF32 on "
                        "NVIDIA tensor cores (~10 mantissa bits vs 24); 'medium' "
                        "is bfloat16 (~8). A speed-for-accuracy trade, so measure "
                        "agreement as well as time before adopting one.")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="atom-voxel pairs per batch. The default is tuned to CPU "
                        "cache size and is usually far too small for a GPU; sweep "
                        "it to find this machine's optimum.")
    args = p.parse_args()

    if args.matmul_precision != "highest":
        torch.set_float32_matmul_precision(args.matmul_precision)

    cell = [float(x) for x in args.cell.split(",")]
    if args.grid:
        grid = tuple(int(x) for x in args.grid.split(","))
    else:
        from lunus.sf.cell_utils import grid_shape_for_resolution
        grid = grid_shape_for_resolution(cell[0], cell[1], cell[2], args.d_min)

    elements, b_per_atom, frac_real = load_structure(args.pdb, cell)
    frac_np = None if args.random_coords else frac_real
    print(f"{len(elements):,} atoms, grid {grid}, cutoff {args.cutoff}, "
          f"B {b_per_atom.min():.1f}-{b_per_atom.max():.1f}")
    print(f"torch {torch.__version__}, device {args.device}, "
          f"{torch.get_num_threads()} threads, "
          f"coords {'RANDOM (unrepresentative)' if args.random_coords else 'real'}\n")

    rows = []
    # Eager only unless asked. torch.compile costs a ~10 s attempt on the first
    # call, and it currently FAILS on both platforms this has been run on --
    # no cuda.h for triton in a driver-only container, and a broken Metal
    # backend on MPS in torch 2.13 -- so the default paid that price on every
    # run to measure the eager path twice. It is also worth nothing on a GPU
    # even when it works: 5140e6 pairs/s compiled against 5126e6 eager.
    for compile_core in ((False, True) if args.compile else (False,)):
        r = bench_torch(elements, b_per_atom, cell, grid, args.cutoff,
                        args.max_atoms, args.reps, compile_core, args.device,
                        max_pairs=args.max_pairs, frac_np=frac_np)
        label = "torch (compiled)" if compile_core else "torch (eager)"
        rows.append((label, r["best"], r["pairs"]))
        extra = f"   first call {r['first']:.2f}s (includes compile)" if compile_core else ""
        print(f"{label:18s} {r['best']:6.2f}s   {r['pairs']/r['best']/1e6:7.1f}e6 pairs/s"
              f"   pairs {r['pairs']:,}{extra}")
        print(f"{'':18s} as batched: {r['pairs_actual']:,} pairs in {r['chunks']:,} "
              f"chunks (padding {r['pairs_actual']/r['pairs']:.4f}x)")
        if not compile_core:
            print(f"{'':18s} setup (kernels + offsets, once per run) {r['setup']:.2f}s")

    if not args.no_gemmi:
        try:
            g = bench_gemmi(args.pdb, cell, args.d_min, args.cutoff, args.reps)
        except Exception as e:
            print(f"\ngemmi reference skipped: {type(e).__name__}: {e}")
            return
        if tuple(g["grid"]) != tuple(grid):
            print(f"\nNOTE: gemmi chose grid {g['grid']}, not {grid}; times are not "
                  "directly comparable. Pass --grid to match, or --d-min to agree.")
        print(f"\n{'gemmi (C++)':18s} {g['best']:6.2f}s   (reference)")
        for label, t, _ in rows:
            print(f"  {label:16s} {t/g['best']:5.2f}x gemmi")


if __name__ == "__main__":
    main()
