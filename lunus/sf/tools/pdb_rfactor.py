#!/usr/bin/env python
"""
R and R-free for a deposited PDB entry, by PDB ID, with torch or with mmtbx.

    PYTHONPATH=<repo root> python tools/pdb_rfactor.py 7FPV
    PYTHONPATH=<repo root> python tools/pdb_rfactor.py 7FPV --method mmtbx
    PYTHONPATH=<repo root> python tools/pdb_rfactor.py 7FPV --method torch \\
        --aniso-adp -- --mask gemmi --shells 5

Fetches the model (<ID>.pdb) and the deposited structure factors
(<ID>-sf.cif) from the wwPDB, keeps them in --dir, and computes R-work and
R-free against them on the coordinates exactly as deposited -- no refinement,
only the bulk-solvent and overall scales are fitted. Two ways:

    torch    lunus.sf: density -> symmetry-expand -> FFT, threshold solvent
             mask, k_sol/b_sol/k_overall/B fitted by least squares. This is
             tools/fit_solvent_rfactor.py, and every one of its options can
             be passed through after a bare "--".
    mmtbx    mmtbx.f_model on the same model and the same French-Wilson
             amplitudes and free set: its own mask, its own scaling. No
             lunus.sf code involved.

The default is both, because neither number means much alone. The deposited R
is NOT the target -- it came from a model refined against this data, which
these are not -- so the mmtbx row is what says what this data supports, and
the torch row is read against it. On 7FPV at full resolution, with
--aniso-adp, the two are 0.1312 and 0.1298; without it 0.1802 and 0.1789, the
gap being the isotropic approximation rather than anything in the solvent
model. See docs/solvent-design.md, "What the R-factor found".

The torch fit is run with its anisotropic overall scale tensor (--aniso)
because mmtbx scales anisotropically too; the comparison is unfair otherwise,
and not by a little -- with anisotropic ADPs and an ISOTROPIC overall B, 7FPV
comes out at 0.2060 rather than 0.1312. --iso-scale turns it off.

mmtbx also fits its overall scale per resolution bin. A binned scale absorbs
resolution-dependent errors, including the ones in a structure-factor model
that this comparison exists to expose, so the table always shows both: the
"torch" row with a single overall scale, and beside it (--binned-row) the
"torch, N scale bins" row fitted the way mmtbx is, from the same structure
factors and, by default, on mmtbx's own logarithmic resolution bins ('log'; an
integer N gives N equal-count bins instead). Since 2026-11-17 the binned fit
is the torch tool's main fit (the method write-up's scaling) and the
single-scale fit is made alongside; the rows, their labels, order and numbers
are what they were. --single-scale makes the single scale the main fit again.
The distance between the two rows is the part of the gap that is scaling. On
the 40 sampleworks entries, with 20 equal-count bins (the default
before 2026-11-09; --binned-row 20 reproduces it), the single-scale row is a
median 0.005 above mmtbx in R-work and the binned row 0.002; what remains is
mostly the solvent mask. On data whose falloff is not Gaussian the difference
is large (8VQ1, room-temperature XFEL: 0.198 against mmtbx's 0.137, and 0.139
binned).

Files are cached: an existing <ID>.pdb or <ID>-sf.cif in --dir is used as is,
so running in examples/compare_gemmi/ never touches the network for 7FPV. If
the entry is too large for the PDB format, <ID>.cif is fetched instead; both
cctbx and gemmi read it.
"""

import argparse
import os
import sys
import time

# tools/ is not a package; the torch path is fit_solvent_rfactor.py next door.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_solvent_rfactor  # noqa: E402


def fetch_entry(pdb_id, out_dir, mirror="rcsb", log=sys.stdout):
    """<ID>.pdb (or <ID>.cif) and <ID>-sf.cif in out_dir, downloading what is
    missing. Returns (model_path, sf_path)."""
    from iotbx.pdb import fetch

    pdb_id = pdb_id.upper()
    os.makedirs(out_dir, exist_ok=True)

    model = None
    for ext in (".pdb", ".cif"):
        candidate = os.path.join(out_dir, pdb_id + ext)
        if os.path.exists(candidate):
            model = candidate
            print("using existing %s" % candidate, file=log)
            break
    if model is None:
        for entity, ext in (("model_pdb", ".pdb"), ("model_cif", ".cif")):
            try:
                data = fetch.fetch(pdb_id, entity, mirror=mirror)
            except RuntimeError as e:
                # 404: entries beyond 62 chains or 99,999 atoms have no PDB
                # format file. Fall through to mmCIF.
                print("  %s" % e, file=log)
                continue
            model = os.path.join(out_dir, pdb_id + ext)
            with open(model, "wb") as f:
                f.write(data.read())
            print("fetched %s" % model, file=log)
            break
        if model is None:
            raise SystemExit("no model for %s at %s" % (pdb_id, mirror))

    sf = os.path.join(out_dir, pdb_id + "-sf.cif")
    if os.path.exists(sf):
        print("using existing %s" % sf, file=log)
    else:
        try:
            data = fetch.fetch(pdb_id, "sf", mirror=mirror)
        except RuntimeError as e:
            raise SystemExit("%s\n%s has no deposited structure factors; "
                             "nothing to compute R against." % (e, pdb_id))
        with open(sf, "wb") as f:
            f.write(data.read())
        print("fetched %s" % sf, file=log)
    return model, sf


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb_id", help="4-character PDB ID, e.g. 7FPV")
    p.add_argument("--method", default="both",
                   choices=["torch", "mmtbx", "both"])
    p.add_argument("--dir", default=".",
                   help="where the files live or are downloaded to "
                        "(default: the current directory)")
    p.add_argument("--mirror", default="rcsb", choices=["rcsb", "pdbe", "pdbj"])
    p.add_argument("--d-min", type=float, default=None,
                   help="truncate the observations (default: use them all)")
    p.add_argument("--iso-scale", action="store_true",
                   help="fit a single overall B in the torch method instead "
                        "of the anisotropic scale tensor mmtbx also uses")
    p.add_argument("--aniso-adp", action="store_true",
                   help="use the deposited ANISOTROPIC ADPs. Without this "
                        "both methods flatten them to isotropic, so the two "
                        "stay comparable either way")
    p.add_argument("--binned-row", type=fit_solvent_rfactor.scale_bins_arg,
                   default="log", metavar="log|N",
                   help="the torch row fitted with one scale per resolution "
                        "bin, as mmtbx scales, printed beside the single-scale "
                        "one; it shows how much of a gap to mmtbx is the "
                        "scaling model. Since 2026-11-17 it is the torch fit's "
                        "MAIN fit (--scale-bins), with the single-scale fit "
                        "made alongside; the rows printed are unchanged. 'log' "
                        "uses mmtbx's own logarithmic bins, N uses N "
                        "equal-count bins (20 was the default before "
                        "2026-11-09), 0 omits the row and makes the single "
                        "scale the main fit (default: log)")
    p.add_argument("--single-scale", action="store_true",
                   help="make the torch tool's main fit single-scale, as "
                        "before 2026-11-17 (--scale-bins 0), with the binned "
                        "row as an extra fit (--also-scale-bins); every row "
                        "prints as it does by default")
    p.add_argument("--select", default="all", metavar="SEL",
                   help="an iotbx atom selection applied to the model for "
                        "BOTH methods, e.g. \"not element H\" to drop "
                        "deposited hydrogens (default: all atoms)")
    p.epilog = ("Anything else -- after a bare '--' to be safe -- is passed "
                "through to fit_solvent_rfactor.py: --mask gemmi, --device, "
                "--mask-blur, --mask-source, --protein-blur, --target-spacing, "
                "--reference, ...")
    return p


def main():
    p = build_parser()
    args, passthrough = p.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    model, sf = fetch_entry(args.pdb_id, args.dir, args.mirror)

    rows = []
    r_torch = {}
    r_mmtbx = None
    t_mmtbx = None
    if args.method in ("torch", "both"):
        argv = [model, sf]
        if args.d_min is not None:
            argv += ["--d-min", str(args.d_min)]
        if args.aniso_adp:
            argv.append("--aniso-adp")
        if not args.iso_scale:
            argv.append("--aniso")
        if args.select != "all":
            argv += ["--select", args.select]
        # The binned row is the torch tool's main fit by default (2026-11-17)
        # and the single-scale fit is made alongside it; --single-scale, or
        # --binned-row 0, makes the single scale the main fit, with the
        # binned row as the extra fit (or none). A --scale-bins passed
        # through wins, and then nothing is added here. An --also-scale-bins
        # passed through still wins too, as it always has: with no
        # --single-scale nothing is added, and the torch tool reads it the old
        # way (a single-scale main fit with that row;
        # fit_solvent_rfactor.resolve_scale_bins). What the passthrough sets
        # is read from the torch tool's own parser, so abbreviated and '='
        # forms count, and nothing is added that the tool would then report
        # as ignored.
        given = fit_solvent_rfactor.build_parser().parse_args(argv + passthrough)
        if given.scale_bins is None:
            if args.single_scale or not args.binned_row:
                argv += ["--scale-bins", "0"]
                if given.also_scale_bins is None:
                    argv += ["--also-scale-bins", str(args.binned_row)]
            elif given.also_scale_bins is None:
                argv += ["--scale-bins", str(args.binned_row)]
        elif not given.scale_bins and given.also_scale_bins is None:
            # a single-scale main fit passed through (--scale-bins 0,
            # --single-scale): --binned-row still chooses the extra row, as
            # it always did
            argv += ["--also-scale-bins", str(args.binned_row)]
        argv += passthrough
        print("\ntorch: fit_solvent_rfactor.py %s\n" % " ".join(argv[2:]))
        torch_args = fit_solvent_rfactor.build_parser().parse_args(argv)
        r = fit_solvent_rfactor.run(torch_args)
        r_torch = r
        # The rows keep the labels and order log parsers match (the
        # TorchRef kit's torch_rwork, torch_binned_rwork, ...): "torch, no
        # solvent" and "torch" are the single-scale fits, "torch, N scale
        # bins" the binned one, whichever of them the tool made its main fit.
        # With a binned main fit the single-scale fits come from the "single"
        # keys run() reports alongside it; the bin scheme is printed beside
        # the table.
        single = "R-work single" in r
        rows.append(("torch, no solvent",
                     r["R-work no solvent single" if single else "R-work no solvent"],
                     r["R-free no solvent single" if single else "R-free no solvent"],
                     float("nan"), float("nan")))
        if single:
            rows.append(("torch", r["R-work single"], r["R-free single"],
                         r["k_sol single"], r["b_sol single"]))
        main = ("torch, %d scale bins" % r["main_scale_bins"]
                if r["main_scale_bins"] else "torch")
        rows.append((main, r["R-work"], r["R-free"], r["k_sol"],
                     r["b_sol"]))
        if "R-work binned" in r:
            rows.append(("torch, %d scale bins" % r["scale_bins"],
                         r["R-work binned"], r["R-free binned"],
                         r["k_sol binned"], r["b_sol binned"]))
        n_work, n_free, n_atoms = r["n_work"], r["n_free"], r["n_atoms"]
    if args.method in ("mmtbx", "both"):
        if passthrough:
            print("\nnote: %s apply to the torch method only"
                  % " ".join(passthrough))
        t0 = time.perf_counter()
        r = r_mmtbx = fit_solvent_rfactor.mmtbx_fit(model, sf, args.d_min,
                                          isotropic=not args.aniso_adp,
                                          select=args.select)
        t_mmtbx = time.perf_counter() - t0
        rows.append(("mmtbx.f_model", r["R-work"], r["R-free"], r["k_sol"],
                     r["b_sol"]))
        n_work, n_free, n_atoms = r["n_work"], r["n_free"], r["n_atoms"]
        if r["dropped_note"] and args.method == "mmtbx":
            # The torch run prints this line itself when it runs; print it
            # here when it did not, so the reflection count is not silently
            # short of the file's.
            print(r["dropped_note"])

    print("\n%s: %d work, %d free reflections%s, %s%s"
          % (args.pdb_id.upper(), n_work, n_free,
             "" if args.d_min is None else " to %.2f A" % args.d_min,
             "ADPs as deposited" if args.aniso_adp
             else "ADPs flattened to isotropic",
             "" if args.select == "all"
             else ", %d atoms (--select %r)" % (n_atoms, args.select)))
    print("  %-20s %8s %8s %8s %8s" % ("", "R-work", "R-free", "k_sol", "b_sol"))
    for name, rw, rf, ks, bs in rows:
        print("  %-20s %8.4f %8s %8s %8s"
              % (name, rw, "-" if n_free == 0 else "%.4f" % rf,
                 "-" if ks != ks else "%.3f" % ks,
                 "-" if bs != bs else "%.1f" % bs))
    # One entry per binned row, each with the solvent scale that row used.
    schemes = ["%s, solvent %s" % (r_torch[k], r_torch.get(sk, "exponential"))
               for k, sk in (("main_bin_scheme", "solvent_scheme"),
                             ("bin_scheme", "solvent_scheme binned"))
               if r_torch.get(k)]
    if schemes:
        print("  scale bins: %s" % "; ".join(schemes))
    occ = []
    if r_torch.get("target_occupancy") is not None:
        source = r_torch.get("target_source", "gemmi")
        if source.startswith("none"):
            occ.append("torch gemmi mask %.4f, no calibration (--mask gemmi)"
                       % r_torch["target_occupancy"])
        else:
            occ.append("torch target %.4f (%s)" % (r_torch["target_occupancy"], source))
    elif r_torch.get("mask_level") is not None:
        occ.append("torch mask %.4f at a fixed level of %g sphere overlaps, "
                   "no calibration (--mask-level)"
                   % (r_torch["mask_occupancy"], r_torch["mask_level"]))
    if r_mmtbx is not None and r_mmtbx.get("mask_occupancy") is not None:
        occ.append("mmtbx mask %.4f" % r_mmtbx["mask_occupancy"])
    if occ:
        print("  mask occupancy: %s" % "; ".join(occ))
    if r_torch.get("mask_source") == "spheres":
        print("  mask source: soft spheres, edge %.2f A (--mask-source spheres)"
              % r_torch["sphere_edge"])
    elif r_torch.get("mask_source") == "atoms":
        print("  mask source: atom-presence field, unit weight at B = %.0f A^2 "
              "(--mask-source atoms)" % r_torch["mask_source_b"])
    elif r_torch.get("mask_source") == "density":
        print("  mask source: the electron density (--mask-source density)")
    if r_torch.get("protein_blur"):
        print("  protein blur: %.1f A^2 placed and removed again (sharpest "
              "atom B %.1f)" % (r_torch["protein_blur"], r_torch["b_sharpest"]))
    # Wall time of each method; the torch run's own per-phase table is above.
    parts = []
    if r_torch.get("timing"):
        parts.append("torch %.1f s" % r_torch["timing"]["total"])
    if t_mmtbx is not None:
        parts.append("mmtbx %.1f s" % t_mmtbx)
    if parts:
        print("  timing: %s" % ", ".join(parts))
    if n_free == 0:
        print("  no R-free flags deposited: every reflection is in the work set")
    if args.method == "both":
        print("  The mmtbx row is what this data supports with the model as "
              "deposited;\n  read the torch row against it, not against the "
              "published R.")
        if any(name.endswith("scale bins") for name, *_ in rows):
            print("  mmtbx scales per resolution bin; the binned torch row "
                  "does the same,\n  so its distance from the torch row is "
                  "the part of the gap that is scaling.")


if __name__ == "__main__":
    main()
