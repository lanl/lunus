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

    torch    lunus.sf: splat -> symmetry-expand -> FFT, threshold solvent
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

mmtbx also fits its overall scale per resolution bin, which the torch fit does
not: a binned scale absorbs resolution-dependent errors, including the ones in
a structure-factor model that this comparison exists to expose. So the torch
row stays single-scale, and a second row beside it (--binned-row, 20 bins by
default) is fitted the way mmtbx is, from the same structure factors. The
distance between the two rows is the part of the gap that is scaling. On the
40 sampleworks entries the single-scale row is a median 0.005 above mmtbx in
R-work and the binned row 0.002; what remains is mostly the solvent mask. On
data whose falloff is not Gaussian the difference is large (8VQ1,
room-temperature XFEL: 0.198 against mmtbx's 0.137, and 0.139 binned).

Files are cached: an existing <ID>.pdb or <ID>-sf.cif in --dir is used as is,
so running in examples/compare_gemmi/ never touches the network for 7FPV. If
the entry is too large for the PDB format, <ID>.cif is fetched instead; both
cctbx and gemmi read it.
"""

import argparse
import os
import sys

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


def main():
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
    p.add_argument("--binned-row", type=int, default=20, metavar="N",
                   help="add a torch row fitted with N resolution-bin scales, "
                        "as mmtbx scales, beside the single-scale one; it "
                        "shows how much of a gap to mmtbx is the scaling "
                        "model. 0 omits it (default: 20)")
    p.epilog = ("Anything else -- after a bare '--' to be safe -- is passed "
                "through to fit_solvent_rfactor.py: --mask gemmi, --device, "
                "--mask-blur, --reference, ...")
    args, passthrough = p.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    model, sf = fetch_entry(args.pdb_id, args.dir, args.mirror)

    rows = []
    if args.method in ("torch", "both"):
        argv = [model, sf]
        if args.d_min is not None:
            argv += ["--d-min", str(args.d_min)]
        if args.aniso_adp:
            argv.append("--aniso-adp")
        if not args.iso_scale:
            argv.append("--aniso")
        argv += passthrough
        if args.binned_row:
            argv += ["--also-scale-bins", str(args.binned_row)]
        print("\ntorch: fit_solvent_rfactor.py %s\n" % " ".join(argv[2:]))
        torch_args = fit_solvent_rfactor.build_parser().parse_args(argv)
        r = fit_solvent_rfactor.run(torch_args)
        # A --scale-bins passed through makes the main fit itself binned; say
        # so in its label, and run() then skips the extra row as redundant.
        main = ("torch, %d scale bins" % torch_args.scale_bins
                if torch_args.scale_bins else "torch")
        rows.append(("torch, no solvent", r["R-work no solvent"],
                     r["R-free no solvent"], float("nan"), float("nan")))
        rows.append((main, r["R-work"], r["R-free"], r["k_sol"],
                     r["b_sol"]))
        if "R-work binned" in r:
            rows.append(("torch, %d scale bins" % r["scale_bins"],
                         r["R-work binned"], r["R-free binned"],
                         r["k_sol binned"], r["b_sol binned"]))
        n_work, n_free = r["n_work"], r["n_free"]
    if args.method in ("mmtbx", "both"):
        if passthrough:
            print("\nnote: %s apply to the torch method only"
                  % " ".join(passthrough))
        r = fit_solvent_rfactor.mmtbx_fit(model, sf, args.d_min,
                                          isotropic=not args.aniso_adp)
        rows.append(("mmtbx.f_model", r["R-work"], r["R-free"], r["k_sol"],
                     r["b_sol"]))
        n_work, n_free = r["n_work"], r["n_free"]

    print("\n%s: %d work, %d free reflections%s, %s"
          % (args.pdb_id.upper(), n_work, n_free,
             "" if args.d_min is None else " to %.2f A" % args.d_min,
             "ADPs as deposited" if args.aniso_adp
             else "ADPs flattened to isotropic"))
    print("  %-20s %8s %8s %8s %8s" % ("", "R-work", "R-free", "k_sol", "b_sol"))
    for name, rw, rf, ks, bs in rows:
        print("  %-20s %8.4f %8s %8s %8s"
              % (name, rw, "-" if n_free == 0 else "%.4f" % rf,
                 "-" if ks != ks else "%.3f" % ks,
                 "-" if bs != bs else "%.1f" % bs))
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
