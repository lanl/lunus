"""tools/pdb_rfactor.py: the file cache, the mmtbx path, and the passthrough.

Nothing here touches the network. The cache test runs against the tracked
7FPV files in examples/compare_gemmi/ with the downloader stubbed to raise,
so a regression that re-fetches a file that is already present fails loudly
rather than silently costing a download. The torch path is
fit_solvent_rfactor.run(), which the design note validates by hand on 7FPV at
full resolution (~2 minutes); here it runs only truncated to 3 A, a couple of
seconds, to check how the binned rows are wired, alongside the cheap mmtbx
path at the same truncation.
"""

import os
import sys

import pytest

from conftest import REPO_ROOT, requires_cctbx

TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
COMPARE = os.path.join(REPO_ROOT, "lunus", "sf", "examples", "compare_gemmi")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

pytestmark = [
    requires_cctbx,
    pytest.mark.skipif(
        not os.path.exists(os.path.join(COMPARE, "7FPV-sf.cif")),
        reason="examples/compare_gemmi/7FPV-sf.cif not present"),
]


def test_cached_files_are_used_without_fetching(monkeypatch):
    import iotbx.pdb.fetch
    import pdb_rfactor

    def no_network(*a, **k):
        raise AssertionError("fetch() called although the files exist")
    monkeypatch.setattr(iotbx.pdb.fetch, "fetch", no_network)

    model, sf = pdb_rfactor.fetch_entry("7fpv", COMPARE, log=open(os.devnull, "w"))
    assert model == os.path.join(COMPARE, "7FPV.pdb")
    assert sf == os.path.join(COMPARE, "7FPV-sf.cif")


def test_mmtbx_fit_on_7fpv():
    import fit_solvent_rfactor

    r = fit_solvent_rfactor.mmtbx_fit(
        os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
        d_min=3.0, isotropic=True)
    assert r["n_free"] > 0 and r["n_work"] > 10 * r["n_free"]
    # Unrefined-against-this-truncation scales on a deposited model: a real
    # fit lands well inside this, a broken reading path or an un-updated
    # scale does not.
    assert 0.10 < r["R-work"] < 0.30
    assert 0.10 < r["R-free"] < 0.35
    assert 0.1 < r["k_sol"] < 0.6


def test_passthrough_reaches_the_torch_parser():
    import fit_solvent_rfactor

    args = fit_solvent_rfactor.build_parser().parse_args(
        ["m.pdb", "m-sf.cif", "--d-min", "2.0", "--aniso-adp", "--mask", "gemmi",
         "--shells", "4"])
    assert (args.d_min, args.aniso_adp, args.mask, args.shells) == \
        (2.0, True, "gemmi", 4)


def test_binned_row_is_extra_and_off_in_the_torch_tool_by_default():
    """pdb_rfactor asks run() for the binned fit with --also-scale-bins, which
    must leave the main fit single-scale; fit_solvent_rfactor itself bins
    nothing unless told to."""
    import fit_solvent_rfactor

    parse = fit_solvent_rfactor.build_parser().parse_args
    plain = parse(["m.pdb", "m-sf.cif"])
    assert (plain.scale_bins, plain.also_scale_bins) == (0, 0)
    extra = parse(["m.pdb", "m-sf.cif", "--also-scale-bins", "20"])
    assert (extra.scale_bins, extra.also_scale_bins) == (0, 20)
    # 'log' (mmtbx's own bins) likewise leaves the main fit single-scale.
    log_bins = parse(["m.pdb", "m-sf.cif", "--also-scale-bins", "log"])
    assert (log_bins.scale_bins, log_bins.also_scale_bins) == (0, "log")


def test_charged_scattering_types_get_coefficients():
    """7TX0 types carboxylate oxygens 'O1-'; the default table of neutral
    elements raised KeyError. cctbx has the ion; 'N1+' it does not, and that
    one must fall back to N rather than fail."""
    import io

    import fit_solvent_rfactor

    log = io.StringIO()
    table = fit_solvent_rfactor.scattering_table(["C", "N1+", "O", "O1-"], log)
    assert set(table) == {"C", "N1+", "O", "O1-"}
    assert table["O1-"] != table["O"]            # a real ion entry, not a copy
    assert table["N1+"] == fit_solvent_rfactor.scattering_table(["N"])["N"]
    assert "N1+ -> N" in log.getvalue()


def file_records(pdb):
    """(element, xyz) of every ATOM/HETATM record, in file order."""
    out = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")):
            out.append((line[76:78].strip().upper(),
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    return out


def test_select_keeps_the_file_order_atoms():
    """--select on 7FPV, whose altlocs are interleaved atom by atom: the atoms
    kept must be exactly the file's non-hydrogen records, coordinates and all.
    A mask taken in hierarchy order would keep the right NUMBER of atoms and
    the wrong ones."""
    import io

    import numpy as np

    import fit_solvent_rfactor

    pdb = os.path.join(COMPARE, "7FPV.pdb")
    records = file_records(pdb)
    heavy = np.array([xyz for el, xyz in records if el != "H"])
    assert 0 < len(heavy) < len(records)

    xrs, elements, *_ = fit_solvent_rfactor.read_model(
        pdb, log=io.StringIO(), select="not element H")
    assert len(elements) == len(heavy)
    assert "H" not in elements
    assert np.allclose(np.array(xrs.sites_cart()), heavy, atol=1e-3)

    xrs_all, elements_all, *_ = fit_solvent_rfactor.read_model(pdb, log=io.StringIO())
    assert len(elements_all) == len(records)


def test_select_reaches_the_mmtbx_path():
    import fit_solvent_rfactor

    pdb = os.path.join(COMPARE, "7FPV.pdb")
    n_heavy = sum(1 for el, _ in file_records(pdb) if el != "H")
    r = fit_solvent_rfactor.mmtbx_fit(
        pdb, os.path.join(COMPARE, "7FPV-sf.cif"),
        d_min=3.0, isotropic=True, select="not element H")
    assert r["n_atoms"] == n_heavy
    assert 0.10 < r["R-work"] < 0.30


def test_select_keeping_nothing_is_an_error():
    import io

    import pytest

    import fit_solvent_rfactor

    with pytest.raises(SystemExit):
        fit_solvent_rfactor.read_model(os.path.join(COMPARE, "7FPV.pdb"),
                                       log=io.StringIO(), select="element Xx")


def test_binned_row_defaults_to_mmtbx_log_bins():
    """pdb_rfactor's binned row uses mmtbx's own bins unless told otherwise,
    and an integer still means that many equal-count bins."""
    import pdb_rfactor

    parse = pdb_rfactor.build_parser().parse_known_args
    assert parse(["7FPV"])[0].binned_row == "log"
    assert parse(["7FPV", "--binned-row", "20"])[0].binned_row == 20


def test_log_binned_row_through_run_on_7fpv_at_3a():
    """The wiring: run() hands the binned fit cctbx's d spacings and reports
    mmtbx's 6 bins for 7FPV cut to 3 A, while the main fit stays
    single-scale."""
    import contextlib
    import io

    import fit_solvent_rfactor

    args = fit_solvent_rfactor.build_parser().parse_args(
        [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
         "--d-min", "3.0", "--aniso", "--also-scale-bins", "log", "--shells", "3"])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        r = fit_solvent_rfactor.run(args)
    assert r["main_scale_bins"] == 0 and r["main_bin_scheme"] is None
    assert (r["scale_bins"], r["bin_scheme"]) == (6, "6 log bins")
    assert "binned row: 6 log bins" in out.getvalue()
    # Pinned, so that some other 6-bin partition cannot pass: 6 equal-count
    # bins give R-free 0.1958 here, mmtbx's bins 0.1929.
    assert abs(r["R-work binned"] - 0.1880) < 5e-4
    assert abs(r["R-free binned"] - 0.1929) < 1e-3


def test_main_fit_with_log_bins_through_run_on_7fpv_at_3a():
    """--scale-bins log makes the main fit itself binned on mmtbx's bins."""
    import contextlib
    import io

    import fit_solvent_rfactor

    args = fit_solvent_rfactor.build_parser().parse_args(
        [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
         "--d-min", "3.0", "--aniso", "--scale-bins", "log", "--shells", "3"])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        r = fit_solvent_rfactor.run(args)
    assert (r["main_scale_bins"], r["main_bin_scheme"]) == (6, "6 log bins")
    assert "overall scale per resolution bin: 6 log bins" in out.getvalue()
    assert "fitted per resolution bin (6 bins)" in out.getvalue()
    assert abs(r["R-free"] - 0.1929) < 1e-3


def test_main_forwards_the_default_binned_row(monkeypatch, capsys):
    """pdb_rfactor's main() hands run() --also-scale-bins log by default and
    an integer when given one, and prints the bin scheme under the table."""
    import pdb_rfactor

    seen = []

    def fake_run(args):
        seen.append(args.also_scale_bins)
        return {"R-work no solvent": 0.3, "R-free no solvent": 0.3,
                "R-work": 0.2, "R-free": 0.2, "k_sol": 0.35, "b_sol": 46.0,
                "n_work": 100, "n_free": 5, "n_atoms": 10,
                "main_scale_bins": 0, "main_bin_scheme": None,
                "scale_bins": 6, "bin_scheme": "6 log bins",
                "R-work binned": 0.19, "R-free binned": 0.19,
                "k_sol binned": 0.35, "b_sol binned": 46.0}
    monkeypatch.setattr(pdb_rfactor, "fetch_entry",
                        lambda *a, **k: ("m.pdb", "m-sf.cif"))
    monkeypatch.setattr(pdb_rfactor.fit_solvent_rfactor, "run", fake_run)
    for extra, expected in (([], "log"), (["--binned-row", "20"], 20)):
        monkeypatch.setattr(sys, "argv",
                            ["pdb_rfactor.py", "7FPV", "--method", "torch"] + extra)
        pdb_rfactor.main()
        assert seen[-1] == expected
    out = capsys.readouterr().out
    assert "torch, 6 scale bins" in out and "scale bins: 6 log bins" in out
