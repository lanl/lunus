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
    assert r["dropped_note"] == ""                # 7FPV's file carries none
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
    single-scale. --solvent-bins off keeps the exponential solvent, which is
    what the pinned numbers were measured with."""
    import contextlib
    import io

    import fit_solvent_rfactor

    args = fit_solvent_rfactor.build_parser().parse_args(
        [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
         "--d-min", "3.0", "--aniso", "--also-scale-bins", "log", "--shells", "3",
         "--solvent-bins", "off"])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        r = fit_solvent_rfactor.run(args)
    assert r["main_scale_bins"] == 0 and r["main_bin_scheme"] is None
    assert (r["scale_bins"], r["bin_scheme"]) == (6, "6 log bins")
    assert r["solvent_scheme binned"] == "exponential"
    assert "binned row: 6 log bins, solvent exponential" in out.getvalue()
    # Pinned, so that some other 6-bin partition cannot pass. History of the
    # pins on this run, all on 2026-11-10: 0.1880 / 0.1929 with no protein
    # blur; 0.1847 / 0.1975 with the blur, both on the d_min/3 grid with the
    # calibration target measured on it; 0.1676 / 0.1924 and 0.1678 / 0.1928
    # on the d_min/4 and 0.7 A-capped grids that were the default for a few
    # hours each; 0.1680 / 0.1921 back on the d_min/3 grid with the target on
    # the 0.7 A reference grid; 0.1628 / 0.1845 with the mask cut from the
    # atom-presence field (--mask-source density --target-spacing same
    # --rate 1.5 reproduces the second pair). mmtbx: 0.1561 / 0.1811.
    assert abs(r["R-work binned"] - 0.1628) < 5e-4
    assert abs(r["R-free binned"] - 0.1845) < 1e-3


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
    # The main fit with bins also scales the solvent per bin by default
    # (measured 2026-11-10 with the mask cut from the atom-presence field:
    # 0.1635 / 0.1875), and the exponential is one flag away (0.1628 /
    # 0.1845). With the mask cut from the density these were 0.1689 /
    # 0.1935 and 0.1680 / 0.1921.
    assert r["solvent_scheme"] == "per bin"
    assert "solvent scale is one value per bin" in out.getvalue()
    assert abs(r["R-work"] - 0.1635) < 5e-4 and abs(r["R-free"] - 0.1875) < 1e-3
    args.solvent_bins = "off"
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = fit_solvent_rfactor.run(args)
    assert r2["solvent_scheme"] == "exponential"
    assert abs(r2["R-work"] - 0.1628) < 5e-4 and abs(r2["R-free"] - 0.1845) < 1e-3


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
                "solvent_scheme": "exponential",
                "scale_bins": 6, "bin_scheme": "6 log bins",
                "solvent_scheme binned": "per bin",
                "R-work binned": 0.19, "R-free binned": 0.19,
                "k_sol binned": 0.35, "b_sol binned": 46.0,
                "timing": {"total": 2.5, "density": 0.3}}
    monkeypatch.setattr(pdb_rfactor, "fetch_entry",
                        lambda *a, **k: ("m.pdb", "m-sf.cif"))
    monkeypatch.setattr(pdb_rfactor.fit_solvent_rfactor, "run", fake_run)
    for extra, expected in (([], "log"), (["--binned-row", "20"], 20)):
        monkeypatch.setattr(sys, "argv",
                            ["pdb_rfactor.py", "7FPV", "--method", "torch"] + extra)
        pdb_rfactor.main()
        assert seen[-1] == expected
    out = capsys.readouterr().out
    assert "torch, 6 scale bins" in out and "scale bins: 6 log bins, solvent per bin" in out
    # 2.5 is the stub's timing["total"], not a measurement: this checks that
    # main() prints the total it is handed, in the format the torchref1000
    # kit's parser matches, and with no mmtbx part when only torch ran.
    assert "  timing: torch 2.5 s" in out


def test_binned_row_defaults_to_per_bin_solvent_on_7fpv_at_3a():
    """The default binned row scales the solvent per bin, reports the curve,
    and quotes k_sol/b_sol as the exponential nearest it; the single-scale
    row is untouched."""
    import contextlib
    import io

    import fit_solvent_rfactor

    args = fit_solvent_rfactor.build_parser().parse_args(
        [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
         "--d-min", "3.0", "--aniso", "--also-scale-bins", "log", "--shells", "3"])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        r = fit_solvent_rfactor.run(args)
    text = out.getvalue()
    # The binned row scales the solvent per bin; the single-scale main fit
    # keeps the exponential, and the two are reported under separate keys.
    assert r["solvent_scheme binned"] == "per bin"
    assert r["solvent_scheme"] == "exponential"
    assert "binned row: 6 log bins, solvent per bin" in text
    assert "solvent scale per bin (low to high resolution):" in text
    # Measured 2026-11-10 with the mask cut from the atom-presence field;
    # the exponential gives 0.1628 / 0.1845 (mask cut from the density:
    # 0.1689 / 0.1935, single 0.1682).
    assert abs(r["R-work binned"] - 0.1635) < 5e-4
    assert abs(r["R-free binned"] - 0.1875) < 1e-3
    assert abs(r["R-work"] - 0.1633) < 5e-4    # single-scale row: not moved by the binned row
    assert 0.0 <= r["b_sol binned"] <= 150.0 and 0.0 <= r["k_sol binned"] <= 0.6
    # per-phase timing: every phase this run goes through is present, none
    # negative, the phases do not exceed the total, and the table is printed
    t = r["timing"]
    for phase in ("read model", "read observations", "density", "symmetry expansion",
                  "F_protein (fft)", "gemmi mask (cpu)", "threshold mask", "F_mask (fft)",
                  "gemmi F_mask + compare", "fit, no solvent", "fit, with solvent",
                  "fit, binned row", "tables", "total"):
        assert phase in t and t[phase] >= 0.0, phase
    assert "mmtbx reference" not in t                 # --reference was not asked for
    assert sum(v for k, v in t.items() if k != "total") <= t["total"] + 1e-6
    assert t["fit, with solvent"] > 0.0 and t["total"] > 0.0
    assert "timing (s, wall; device work waited for at each boundary)" in text
    assert "\n  total " in text


def test_equal_count_bins_keep_the_exponential_solvent_by_default():
    """With equal-count bins the lowest bin reaches to about 3 A, so one flat
    solvent scale over it is worse than the exponential (7FPV, 20 bins:
    R-free +0.0035). 'auto' therefore means per-bin solvent for log bins
    only; 'on' forces it."""
    import contextlib
    import io

    import fit_solvent_rfactor

    base = [os.path.join(COMPARE, "7FPV.pdb"), os.path.join(COMPARE, "7FPV-sf.cif"),
            "--d-min", "3.0", "--aniso", "--shells", "3"]
    parse = fit_solvent_rfactor.build_parser().parse_args
    with contextlib.redirect_stdout(io.StringIO()):
        auto = fit_solvent_rfactor.run(parse(base + ["--also-scale-bins", "6"]))
        on = fit_solvent_rfactor.run(parse(base + ["--also-scale-bins", "6",
                                                   "--solvent-bins", "on"]))
    assert auto["solvent_scheme binned"] == "exponential"
    assert on["solvent_scheme binned"] == "per bin"
    assert auto["bin_scheme"] == "6 equal-count bins"
    assert auto["R-free binned"] == auto["R-free binned"]   # a number, not nan
    # the main fit follows the same rule
    with contextlib.redirect_stdout(io.StringIO()):
        main = fit_solvent_rfactor.run(parse(base + ["--scale-bins", "6"]))
    assert main["solvent_scheme"] == "exponential" and main["main_bin_scheme"] == "6 equal-count bins"


def test_drop_unfittable_reflections():
    """Systematically absent reflections and (0,0,0) leave the array and the
    free mask together; everything else, including an allowed axial
    reflection, stays, and the drop is reported. The (0,0,0) case also
    covers the select() that used to raise on a Python list."""
    import io

    import numpy as np
    from cctbx import crystal, miller
    from cctbx.array_family import flex

    import fit_solvent_rfactor

    cs = crystal.symmetry(unit_cell=(40, 50, 60, 90, 90, 90), space_group_symbol="P 21 21 2")
    hkl = [(0, 0, 0), (0, 5, 0), (0, 4, 0), (3, 0, 0), (2, 0, 0), (1, 1, 1), (0, 0, 7)]
    F = miller.set(cs, flex.miller_index(hkl), anomalous_flag=False).array(
        data=flex.double([1.0, 8.2, 300.0, 6.1, 250.0, 120.0, 90.0]))
    free = np.array([False, True, False, False, True, False, True])
    log = io.StringIO()
    F2, free2, note = fit_solvent_rfactor.drop_unfittable_reflections(F, free, log=log, name="toy")
    kept = [tuple(h) for h in F2.indices()]
    # P 21 21 2: h00 and 0k0 absent for odd h, k; 00l is allowed (no screw along c)
    assert kept == [(0, 4, 0), (2, 0, 0), (1, 1, 1), (0, 0, 7)]
    assert list(free2) == [False, True, False, True]
    assert note.startswith("toy: 2 systematically absent reflection(s) of P 21 21 2")
    assert log.getvalue().strip() == note
    # nothing to drop: untouched, silent
    F3, free3, note3 = fit_solvent_rfactor.drop_unfittable_reflections(F2, free2, log=io.StringIO())
    assert F3.size() == 4 and list(free3) == list(free2) and note3 == ""


def test_absent_reflections_are_dropped_by_the_reader_and_mmtbx_fit(tmp_path):
    """The fix end to end: a model and an MTZ in P 21 21 2 whose data hold
    odd 0k0 reflections with non-zero |F|. read_observations (which run()
    also calls) drops them and says so; mmtbx_fit reads through the same
    function, returns the note, and runs -- with the absences left in,
    mmtbx's aniso_u_scaler raises MMTBX_ASSERT(fm_abs > 0) here too."""
    import io

    import numpy as np
    from cctbx import crystal, miller, xray
    from cctbx.array_family import flex

    import fit_solvent_rfactor

    cs = crystal.symmetry(unit_cell=(30, 36, 42, 90, 90, 90), space_group_symbol="P 21 21 2")
    xrs = xray.structure(crystal_symmetry=cs)
    rng = np.random.default_rng(0)
    for i in range(40):
        xrs.add_scatterer(xray.scatterer(label="C%d" % i, site=tuple(rng.uniform(0.05, 0.45, 3)),
                                         scattering_type="C", b=20.0))
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(xrs.as_pdb_file())
    # calculated amplitudes plus deliberate non-zero values on four odd 0k0
    # reflections, as a deposited file can carry
    f = xrs.structure_factors(d_min=2.5, algorithm="direct").f_calc()
    absent_idx = [(0, k, 0) for k in (1, 3, 5, 7)]
    all_idx = flex.miller_index(list(f.indices()) + absent_idx)
    amp = flex.double(list(abs(f).data()) + [5.0, 6.0, 7.0, 8.0])
    F = miller.set(cs, all_idx, anomalous_flag=False).array(data=amp) \
        .set_observation_type_xray_amplitude()
    assert F.sys_absent_flags().data().count(True) == 4
    mtz = tmp_path / "toy.mtz"
    F.as_mtz_dataset(column_root_label="FOBS").mtz_object().write(file_name=str(mtz))

    log = io.StringIO()
    F_read, free = fit_solvent_rfactor.read_observations(str(mtz), crystal_symmetry=cs, log=log)
    assert F_read.sys_absent_flags().data().count(True) == 0
    assert F_read.size() == F.size() - 4 and len(free) == F_read.size()
    assert "4 systematically absent reflection(s) of P 21 21 2 dropped" in log.getvalue()

    r = fit_solvent_rfactor.mmtbx_fit(str(pdb), str(mtz), isotropic=True)
    assert "4 systematically absent reflection(s)" in r["dropped_note"]
    assert r["n_work"] == F.size() - 4 and r["R-work"] == r["R-work"]


def test_main_prints_the_dropped_note_on_an_mmtbx_only_run(monkeypatch, capsys):
    """With --method mmtbx the torch run is not there to print the drop
    line, so main() prints the note mmtbx_fit returns; with both methods it
    does not repeat it."""
    import pdb_rfactor

    note = "m-sf.cif: 42 systematically absent reflection(s) of P 21 21 2 dropped, as refinement programs do"

    def fake_mmtbx(model, sf, d_min, isotropic, select):
        return {"R-work": 0.2, "R-free": 0.25, "k_sol": 0.35, "b_sol": 46.0,
                "n_work": 100, "n_free": 5, "n_atoms": 10, "dropped_note": note}

    def fake_run(args):
        return {"R-work no solvent": 0.3, "R-free no solvent": 0.3, "R-work": 0.2,
                "R-free": 0.2, "k_sol": 0.35, "b_sol": 46.0, "n_work": 100,
                "n_free": 5, "n_atoms": 10, "main_scale_bins": 0, "main_bin_scheme": None,
                "solvent_scheme": "exponential", "timing": {"total": 1.0}}
    monkeypatch.setattr(pdb_rfactor, "fetch_entry", lambda *a, **k: ("m.pdb", "m-sf.cif"))
    monkeypatch.setattr(pdb_rfactor.fit_solvent_rfactor, "mmtbx_fit", fake_mmtbx)
    monkeypatch.setattr(pdb_rfactor.fit_solvent_rfactor, "run", fake_run)

    monkeypatch.setattr(sys, "argv", ["pdb_rfactor.py", "7FPV", "--method", "mmtbx"])
    pdb_rfactor.main()
    assert capsys.readouterr().out.count(note) == 1

    monkeypatch.setattr(sys, "argv", ["pdb_rfactor.py", "7FPV", "--method", "both",
                                      "--binned-row", "0"])
    pdb_rfactor.main()
    assert note not in capsys.readouterr().out     # the (stubbed) torch run would have printed it


def test_zero_amplitudes_are_dropped_only_from_amplitude_data(tmp_path):
    """A deposited AMPLITUDE of exactly zero is an unobserved placeholder and
    is dropped (2GOY), from the raw array, so a zero on one Bijvoet mate
    goes rather than being averaged into its partner; deposited INTENSITIES
    go through French-Wilson and nothing is dropped for being zero or
    (mildly) negative."""
    import io

    import numpy as np
    from cctbx import crystal, miller
    from cctbx.array_family import flex

    import fit_solvent_rfactor

    cs = crystal.symmetry(unit_cell=(30, 36, 42, 90, 90, 90), space_group_symbol="P 1")
    ms = miller.build_set(cs, anomalous_flag=False, d_min=3.0)
    n = ms.size()
    rng = np.random.default_rng(1)
    amp = rng.uniform(50.0, 500.0, n)
    amp[::7] = 0.0                                       # placeholders
    n_zero = int((amp == 0).sum())
    F = ms.array(data=flex.double(amp)).set_observation_type_xray_amplitude()
    mtz_f = tmp_path / "amp.mtz"
    F.as_mtz_dataset(column_root_label="FOBS").mtz_object().write(file_name=str(mtz_f))
    log = io.StringIO()
    F_read, free = fit_solvent_rfactor.read_observations(str(mtz_f), crystal_symmetry=cs, log=log)
    assert F_read.size() == n - n_zero and (np.array(F_read.data()) > 0).all()
    assert len(free) == F_read.size()
    assert "%d amplitude(s) deposited as exactly zero dropped" % n_zero in log.getvalue()

    # a zero deposited for one mate of an anomalous pair: dropped before the
    # Bijvoet average, so the surviving mate keeps its value
    ms_an = miller.build_set(cs, anomalous_flag=True, d_min=3.0)
    amp_an = rng.uniform(50.0, 500.0, ms_an.size())
    minus = np.array(ms_an.indices())[:, 0] < 0
    amp_an[minus] = 0.0
    F_an = ms_an.array(data=flex.double(amp_an)).set_observation_type_xray_amplitude()
    mtz_an = tmp_path / "anom.mtz"
    F_an.as_mtz_dataset(column_root_label="FOBS").mtz_object().write(file_name=str(mtz_an))
    F_an_read, _ = fit_solvent_rfactor.read_observations(str(mtz_an), crystal_symmetry=cs,
                                                         log=io.StringIO())
    assert F_an_read.size() == ms_an.size() // 2 and float(flex.min(F_an_read.data())) >= 50.0

    # the same reflections as intensities, some zero and some negative at
    # -0.5 sigma: the zero-amplitude rule does not apply, and French-Wilson
    # keeps weak and mildly negative intensities (it rejects only below its
    # -4 sigma criterion) rather than stripping them
    inten = amp ** 2
    inten[::7] = 0.0
    inten[1::7] = -5.0
    I = ms.array(data=flex.double(inten), sigmas=flex.double(np.full(n, 10.0))) \
        .set_observation_type_xray_intensity()
    mtz_i = tmp_path / "int.mtz"
    I.as_mtz_dataset(column_root_label="IOBS").mtz_object().write(file_name=str(mtz_i))
    log = io.StringIO()
    F_from_I, _ = fit_solvent_rfactor.read_observations(str(mtz_i), crystal_symmetry=cs, log=log)
    assert "deposited as exactly zero" not in log.getvalue()
    kept = set(map(tuple, F_from_I.indices()))
    hkl = list(map(tuple, ms.indices()))
    assert all(h in kept for h in hkl[::7]) and all(h in kept for h in hkl[1::7])


@pytest.mark.parametrize("flag_label", ["r_free_flags", "R-free-flags", "FreeR_flag"])
def test_zero_amplitudes_reach_the_note_and_the_free_set(tmp_path, flag_label):
    """Through the whole path: an amplitude MTZ with a free-flag column (under
    each of the usual names) and zeros in both the work and the free set.
    The zeros leave both sets, every surviving reflection keeps its own
    flag, the note names the count, and mmtbx_fit returns the same note
    (what pdb_rfactor prints on an mmtbx-only run). A model is built
    alongside so mmtbx_fit can run."""
    import io

    import numpy as np
    from cctbx import crystal, miller, xray
    from cctbx.array_family import flex

    import fit_solvent_rfactor

    cs = crystal.symmetry(unit_cell=(30, 36, 42, 90, 90, 90), space_group_symbol="P 1")
    xrs = xray.structure(crystal_symmetry=cs)
    rng = np.random.default_rng(2)
    for i in range(30):
        xrs.add_scatterer(xray.scatterer(label="C%d" % i, site=tuple(rng.uniform(0.0, 1.0, 3)),
                                         scattering_type="C", b=20.0))
    pdb = tmp_path / "toy.pdb"
    pdb.write_text(xrs.as_pdb_file())
    f = xrs.structure_factors(d_min=3.0, algorithm="direct").f_calc()
    amp = np.array(abs(f).data())
    n = len(amp)
    zero = np.zeros(n, dtype=bool)
    zero[::9] = True
    amp[zero] = 0.0
    flags = np.zeros(n, dtype=int)
    flags[::5] = 1                                    # ~20% free, some of them zero
    n_zero_free = int((zero & (flags == 1)).sum())
    assert n_zero_free > 0
    F = f.set().array(data=flex.double(amp)).set_observation_type_xray_amplitude()
    R = f.set().array(data=flex.int(flags.tolist()))
    ds = F.as_mtz_dataset(column_root_label="FOBS")
    ds.add_miller_array(R, column_root_label=flag_label)
    mtz = tmp_path / "toy.mtz"
    ds.mtz_object().write(file_name=str(mtz))

    info = {}
    F_read, free = fit_solvent_rfactor.read_observations(str(mtz), crystal_symmetry=cs,
                                                         log=io.StringIO(), info=info)
    assert F_read.size() == n - int(zero.sum())
    assert (np.array(F_read.data()) > 0).all()
    # the flags followed their reflections, index by index, not just in count
    expected_free = {tuple(h) for h, fl, z in zip(f.indices(), flags, zero) if fl == 1 and not z}
    got_free = {tuple(h) for h, m in zip(F_read.indices(), free) if m}
    assert got_free == expected_free and len(got_free) > 0
    kept = {tuple(h) for h in F_read.indices()}
    for h, z in zip(f.indices(), zero):
        assert (tuple(h) in kept) == (not z)
    assert "%d amplitude(s) deposited as exactly zero dropped" % int(zero.sum()) in info["dropped_note"]

    r = fit_solvent_rfactor.mmtbx_fit(str(pdb), str(mtz), isotropic=True)
    assert r["dropped_note"] == info["dropped_note"]
    assert r["n_free"] == int(((flags == 1) & ~zero).sum())
    assert r["n_work"] + r["n_free"] == n - int(zero.sum())


def test_status_column_with_wavelength_id_is_still_the_free_set(tmp_path):
    """Deposited mmCIF with a wavelength_id column: iotbx labels the status
    array "<block>,_refln.status,wavelength_id=1" (three labels; 1VME and 10
    of the 1,000 torchref entries). The flag matcher must still find it --
    iotbx's looks_like_r_free_flags_info, which reads only the last label
    of a two-label array, silently did not, and R-free was lost. Zero
    amplitudes are dropped from this file too."""
    import io

    import numpy as np
    from cctbx import crystal

    import fit_solvent_rfactor

    rng = np.random.default_rng(0)
    hkl = [(h, k, l) for h in range(4) for k in range(4) for l in range(4) if (h, k, l) != (0, 0, 0)]
    rows, n_free_expected, n_zero = [], 0, 0
    for i, (h, k, l) in enumerate(hkl):
        status = "f" if i % 5 == 0 else "o"
        F = 0.0 if i % 9 == 0 else float(rng.uniform(50, 500))
        if F == 0.0:
            n_zero += 1
        elif status == "f":
            n_free_expected += 1
        for w in (1, 2):                       # two wavelength ids give the suffixed labels
            rows.append("1 %d 1 %d %d %d %s %.1f %.1f" % (w, h, k, l, status, F, 0.0 if F == 0 else 5.0))
    cif = ("data_toy\n_cell.length_a 30.0\n_cell.length_b 36.0\n_cell.length_c 42.0\n"
           "_cell.angle_alpha 90.0\n_cell.angle_beta 90.0\n_cell.angle_gamma 90.0\n"
           "_symmetry.space_group_name_H-M 'P 1'\nloop_\n_refln.crystal_id\n_refln.wavelength_id\n"
           "_refln.scale_group_code\n_refln.index_h\n_refln.index_k\n_refln.index_l\n_refln.status\n"
           "_refln.F_meas_au\n_refln.F_meas_sigma_au\n" + "\n".join(rows) + "\n")
    path = tmp_path / "toy-sf.cif"
    path.write_text(cif)
    from iotbx.reflection_file_reader import any_reflection_file
    labels = [a.info().label_string() for a in any_reflection_file(file_name=str(path)).as_miller_arrays()]
    assert "toy,_refln.status,wavelength_id=1" in labels   # the three-label form is what is tested

    cs = crystal.symmetry(unit_cell=(30, 36, 42, 90, 90, 90), space_group_symbol="P 1")
    log = io.StringIO()
    F, free = fit_solvent_rfactor.read_observations(str(path), crystal_symmetry=cs, log=log)
    assert "no R-free flags found" not in log.getvalue()
    assert int(free.sum()) == n_free_expected and n_free_expected > 0
    assert F.size() == len(hkl) - n_zero
    assert "%d amplitude(s) deposited as exactly zero dropped" % n_zero in log.getvalue()


def sharp_toy(tmp_path, d_min=2.5, sharp=True, aniso=False):
    """A P1 cell with atoms at B = 20 and, with sharp, at B = 2 (narrower
    than the voxel), and an MTZ whose 'observations' are the exact
    |F_calc| of that model from cctbx direct summation, so that any R above a
    few 1e-3 in the no-solvent fit is the grid's error and nothing else. With
    aniso, one more atom carries an ANISOU with principal B 1.6, 2.4, 3.2
    (U = 0.02, 0.03, 0.04 A^2, rotated off the axes)."""
    from cctbx.array_family import flex
    import iotbx.pdb

    lines = ["CRYST1   20.000   22.000   24.000  90.00  90.00  90.00 P 1           1"]
    atoms = [("C", 12.0, 14.0, 15.0, 20.0), ("N", 15.5, 14.5, 15.5, 20.0), ("O", 12.5, 18.0, 16.0, 20.0)]
    if sharp:
        atoms += [("C", 4.0, 5.0, 6.0, 2.0), ("N", 7.5, 5.5, 6.5, 2.0), ("O", 4.5, 9.0, 7.0, 2.0)]
    for i, (el, x, y, z, b) in enumerate(atoms, 1):
        lines.append("ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00%6.2f          %2s"
                     % (i, el, i, x, y, z, b, el))
    if aniso:
        i = len(atoms) + 1
        # U_cart = R diag(0.02, 0.03, 0.04) R^T for a 30 degree rotation about z,
        # in 1e-4 A^2 as ANISOU stores it: U11 U22 U33 U12 U13 U23
        u11, u22, u12 = 225, 275, -43          # 0.0225, 0.0275, -0.00433
        lines.append("ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00%6.2f          %2s"
                     % (i, "S", i, 9.0, 16.0, 9.5, 8 * 3.14159265 ** 2 * 0.03, "S"))
        lines.append("ANISOU%5d  %-3s ALA A%4d  %7d%7d%7d%7d%7d%7d       %2s"
                     % (i, "S", i, u11, u22, 400, u12, 0, 0, "S"))
    lines.append("END")
    pdb = tmp_path / "sharp.pdb"
    pdb.write_text("\n".join(lines) + "\n")
    xrs = iotbx.pdb.input(file_name=str(pdb)).xray_structure_simple()
    xrs.scattering_type_registry(table="it1992")
    fc = xrs.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    F = fc.customized_copy(data=flex.abs(fc.data())).set_observation_type_xray_amplitude()
    mtz = tmp_path / "sharp.mtz"
    F.as_mtz_dataset(column_root_label="FOBS").mtz_object().write(file_name=str(mtz))
    return str(pdb), str(mtz)


@requires_cctbx
def test_protein_blur_fixes_sharp_atoms_on_the_default_grid(tmp_path):
    """An atom at B = 2 A^2 is a Gaussian narrower than the voxel
    and aliases when placed on the grid: against exact |F_calc| the
    no-solvent fit cannot get below R ~ 0.1. Placing the density with
    recommended_blur's extra B and removing it from F_protein (the default,
    --protein-blur auto) brings the same fit to the few-1e-3 floor of the
    grid; --protein-blur 0 reproduces the aliased placement. The threshold
    mask's smoothing is reduced by the same B, which the run says."""
    import contextlib
    import io

    import fit_solvent_rfactor

    pdb, mtz = sharp_toy(tmp_path)

    def fit(*extra):
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb, mtz, "--shells", "2"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = fit_solvent_rfactor.run(args)
        return r, out.getvalue()

    r, text = fit()
    assert r["R-work no solvent"] < 0.01, (
        "default run: no-solvent R-work %.4f against exact |F_calc|; the B = 2 "
        "atoms alias on the grid" % r["R-work no solvent"])
    assert r["b_sharpest"] == pytest.approx(2.0)
    assert r["protein_blur"] > 10.0, r["protein_blur"]
    assert "protein density placed with B + %.1f A^2" % r["protein_blur"] in text
    assert "(%.1f of it already in the placed density)" % r["protein_blur"] in text

    r0, text0 = fit("--protein-blur", "0")
    assert r0["protein_blur"] == 0.0
    assert "placed as is (--protein-blur 0;" in text0
    assert "clears the grid's blur target" not in text0     # 0 was asked for, not earned
    assert r0["R-work no solvent"] > 0.05, r0["R-work no solvent"]
    assert "already in the placed density" not in text0

    # an explicit value wins over the rule, and reaches the density: a third
    # of the recommended blur lands between the aliased and the corrected fit
    r12, _ = fit("--protein-blur", "12.5")
    assert r12["protein_blur"] == 12.5
    assert r0["R-work no solvent"] > r12["R-work no solvent"] > r["R-work no solvent"]


@requires_cctbx
def test_protein_blur_reaches_the_anisotropic_kernel(tmp_path):
    """With --aniso-adp the anisotropic kernel builder gets the same blur, and
    the sharpest B is the smallest principal B of the ANISOU atom (1.6 A^2
    here), not its equivalent isotropic B (2.4) nor the isotropic minimum
    (2.0); without --aniso-adp the atom is placed with its equivalent B and
    the isotropic minimum rules."""
    import contextlib
    import io

    import numpy as np

    import fit_solvent_rfactor

    pdb, mtz = sharp_toy(tmp_path, aniso=True)

    def fit(*extra):
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb, mtz, "--shells", "2"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_solvent_rfactor.run(args)

    r = fit("--aniso-adp")
    assert r["b_sharpest"] == pytest.approx(0.02 * 8 * np.pi ** 2, abs=0.02)
    assert r["protein_blur"] > 10.0
    assert r["R-work no solvent"] < 0.01, r["R-work no solvent"]
    r0 = fit("--aniso-adp", "--protein-blur", "0")
    assert r0["R-work no solvent"] > 0.05, r0["R-work no solvent"]
    r_iso = fit()
    assert r_iso["b_sharpest"] == pytest.approx(2.0)
    assert r_iso["R-work no solvent"] < 0.01


@requires_cctbx
def test_protein_blur_leaves_the_threshold_mask_as_shipped(tmp_path):
    """--mask-blur is a total: the mask is thresholded from a density smoothed
    by mask_blur in all, whether the protein blur put part of it there or
    not. On a model with nothing sharp (B = 20 only, so nothing aliases) the
    calibrated cutoff and the occupancy with the blur match those without it;
    smoothing twice would move the cutoff."""
    import contextlib
    import io

    import fit_solvent_rfactor

    pdb, mtz = sharp_toy(tmp_path, sharp=False)

    def fit(*extra):
        # on the d_min/3 grid, where B = 20 atoms still earn a blur of ~20
        # A^2, pinned so the margin below does not depend on the grid rule
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb, mtz, "--shells", "2", "--rate", "1.5"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_solvent_rfactor.run(args)

    r = fit()
    r0 = fit("--protein-blur", "0")
    assert r["protein_blur"] > 5.0 and r0["protein_blur"] == 0.0
    assert r["mask_cutoff"] == pytest.approx(r0["mask_cutoff"], rel=2e-3)
    # the occupancy is calibrated to gemmi's target either way, so it cannot
    # tell the two apart; the cutoff is the quantity that moves
    assert r["mask_occupancy"] == pytest.approx(r0["mask_occupancy"], abs=2e-3)
    # the same density smoothed by the full --mask-blur on top of the protein
    # blur is what the remainder rule prevents: run it as if it were not a
    # total, by asking for a mask blur larger by the protein blur
    r2 = fit("--mask-blur", "%.3f" % (100.0 + r["protein_blur"]))
    assert abs(r2["mask_cutoff"] - r0["mask_cutoff"]) > 5e-3 * r0["mask_cutoff"]


@requires_cctbx
def test_sharpest_b_and_protein_blur_setting():
    """sharpest_b is the smallest B the density is placed with: isotropic B,
    or an anisotropic atom's smallest principal B; protein_blur resolves
    'auto' through recommended_blur and takes a number as given."""
    import numpy as np

    import fit_solvent_rfactor
    from lunus.sf.cell_utils import orth_matrix, recommended_blur

    b = np.array([30.0, 12.0, 45.0])
    assert fit_solvent_rfactor.sharpest_b(b) == 12.0
    assert fit_solvent_rfactor.sharpest_b(b, occ=np.array([1.0, 0.0, 0.5])) == 30.0
    assert fit_solvent_rfactor.sharpest_b(b, occ=np.zeros(3)) == 12.0      # all zero: keep all
    u = np.zeros((3, 3, 3))
    u[2] = np.diag([0.05, 0.2, 0.3])                    # principal B 3.9, 15.8, 23.7
    is_aniso = np.array([False, False, True])
    assert abs(fit_solvent_rfactor.sharpest_b(b, u, is_aniso) - 0.05 * 8 * np.pi ** 2) < 1e-9
    u[2] = np.diag([0.5, 0.6, 0.7])                     # all above the isotropic 12
    assert fit_solvent_rfactor.sharpest_b(b, u, is_aniso) == 12.0
    assert fit_solvent_rfactor.sharpest_b(b, u, np.zeros(3, dtype=bool)) == 12.0

    M = orth_matrix(20.0, 22.0, 24.0, 90.0, 90.0, 90.0)
    grid = (24, 27, 30)
    assert fit_solvent_rfactor.protein_blur("auto", 2.0, grid, M, 2.5) == \
        recommended_blur(2.0, grid, M, d_min=2.5)
    assert fit_solvent_rfactor.protein_blur("AUTO", 500.0, grid, M, 2.5) == 0.0
    assert fit_solvent_rfactor.protein_blur("0", 2.0, grid, M, 2.5) == 0.0
    assert fit_solvent_rfactor.protein_blur(7.5, 2.0, grid, M, 2.5) == 7.5
    # a negative blur would place the atoms sharper than the model says and
    # sharpen F_protein's error: rejected at the parser and in the resolver
    with pytest.raises(ValueError, match=">= 0"):
        fit_solvent_rfactor.protein_blur(-1.0, 2.0, grid, M, 2.5)
    parse = fit_solvent_rfactor.build_parser().parse_args
    assert parse(["m.pdb", "d.mtz", "--protein-blur", "AUTO"]).protein_blur == "auto"
    assert parse(["m.pdb", "d.mtz", "--protein-blur", "12.5"]).protein_blur == 12.5
    for bad in ("-1", "yes"):
        with pytest.raises(SystemExit):
            parse(["m.pdb", "d.mtz", "--protein-blur", bad])


@requires_cctbx
def test_default_grid_is_d_min_over_3_with_the_cap_opt_in(tmp_path):
    """The tool's default grid is gemmi's and xtraj's rate 1.5, voxels of
    d_min/3, the grid every number before 2026-11-10 was measured on: at
    2.5 A the voxels are 0.83 A and nothing caps them. --max-voxel caps the
    edge on request (0.7 A binds at 2.5 A and not at 1.8 A) and --rate
    overrides both. The cap was the default for part of 2026-11-10; with the
    calibration target on its own reference grid it changes nothing."""
    import contextlib
    import io

    import fit_solvent_rfactor

    cell = (20.0, 22.0, 24.0)

    def fit(d_min, *extra):
        pdb, mtz = sharp_toy(tmp_path, d_min=d_min, sharp=False)
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb, mtz, "--shells", "2"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = fit_solvent_rfactor.run(args)
        return args, r, out.getvalue()

    args, r, text = fit(2.5)
    assert args.rate is None and args.max_voxel is None, (
        "default grid rule is --rate %r --max-voxel %r, not plain d_min/3"
        % (args.rate, args.max_voxel))
    assert r["rate"] == 1.5 and r["max_voxel"] is None
    assert "grid: voxels of d_min/3" in text and "the default" in text
    assert r["grid_rule"].startswith("voxels of d_min/3")
    assert any(edge / n > 0.7 for edge, n in zip(cell, r["grid"]))        # not capped
    _, r15, _ = fit(2.5, "--rate", "1.5")
    assert r15["grid"] == r["grid"]

    # the cap, opted in: binds at 2.5 A (d_min/3 = 0.83 A), not at 1.8 A (0.6 A)
    _, rc, textc = fit(2.5, "--max-voxel", "0.7")
    assert rc["max_voxel"] == 0.7 and 1.75 < rc["rate"] < 1.85
    assert "grid: voxels capped at --max-voxel 0.7 A" in textc
    for edge, n in zip(cell, rc["grid"]):
        assert edge / n <= 0.7 + 1e-9, (edge, n)
    assert all(a <= b for a, b in zip(r["grid"], rc["grid"])) and rc["grid"] != r["grid"]
    _, r18c, text18 = fit(1.8, "--max-voxel", "0.7")
    assert r18c["rate"] == 1.5 and "finer than the --max-voxel cap" in text18
    _, r18, _ = fit(1.8)
    assert r18["grid"] == r18c["grid"]

    # an explicit --rate wins over both
    _, r2, text2 = fit(2.5, "--rate", "2", "--max-voxel", "0.7")
    assert r2["rate"] == 2.0 and r2["grid_rule"].startswith("--rate 2")
    assert all(a >= b for a, b in zip(r2["grid"], rc["grid"]))


def test_grid_rate_rule():
    """grid_rate: d_min/3 by default; a cap binds only where d_min/3 is
    coarser than it; an explicit rate wins; non-positive settings are errors."""
    import pytest

    import fit_solvent_rfactor as f

    assert f.MAX_VOXEL_DEFAULT is None
    assert f.grid_rate(3.0)[0] == 1.5 and "the default" in f.grid_rate(3.0)[1]
    assert f.grid_rate(1.8, None, 0.7)[0] == 1.5             # d_min/3 = 0.60 < 0.7
    assert f.grid_rate(2.1, None, 0.7)[0] == 1.5             # exactly at the cap
    assert abs(f.grid_rate(3.0, None, 0.7)[0] - 3.0 / 1.4) < 1e-12
    assert abs(f.grid_rate(3.0, None, 0.6)[0] - 2.5) < 1e-12
    assert f.grid_rate(3.0, 2.0, 0.7)[0] == 2.0 and "--rate 2" in f.grid_rate(3.0, 2.0, 0.7)[1]
    assert f.grid_rate(3.0, 1.5, 0.1)[0] == 1.5             # the override ignores the cap
    with pytest.raises(ValueError):
        f.grid_rate(3.0, 0.0)
    with pytest.raises(ValueError):
        f.grid_rate(3.0, None, 0.0)


@requires_cctbx
def test_calibration_target_does_not_move_with_the_working_grid(tmp_path):
    """The threshold mask is calibrated to gemmi's mask occupancy, which steps
    with the voxel size of the grid it is measured on (3I7M: 0.297 on d_min/3
    voxels, 0.316 on d_min/4), so a finer working grid used to change the
    answer through the target. The target is now measured on a fixed reference
    spacing (--target-spacing, default 0.7 A): the same number on every
    working grid, while gemmi's occupancy on the working grid still moves;
    --target-spacing same restores the old coupling."""
    import contextlib
    import io

    import fit_solvent_rfactor

    pdb, mtz = sharp_toy(tmp_path, sharp=False)

    def cell_edges():
        return (20.0, 22.0, 24.0)

    def fit(*extra):
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb, mtz, "--shells", "2"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = fit_solvent_rfactor.run(args)
        return args, r, out.getvalue()

    args, r_def, text = fit()
    assert args.target_spacing == fit_solvent_rfactor.TARGET_SPACING_DEFAULT
    _, r_15, _ = fit("--rate", "1.5")
    _, r_3, _ = fit("--rate", "3")
    assert r_15["grid"] != r_3["grid"]
    assert r_15["target_occupancy"] == r_3["target_occupancy"] == r_def["target_occupancy"], (
        "the calibration target moved with the working grid: %r"
        % [r_15["target_occupancy"], r_def["target_occupancy"], r_3["target_occupancy"]])
    assert r_15["gemmi_occupancy_working_grid"] != r_3["gemmi_occupancy_working_grid"]
    assert "reference grid" in text and "independent of the working grid" in text
    assert "actual spacing up to" in text
    assert 0.0 < r_def["target_occupancy"] < 1.0
    # the value itself is the reference-grid measurement, so a quiet fallback
    # to the working grid fails on a number, not only on the keys
    ref, ref_grid = fit_solvent_rfactor.gemmi_mask_occupancy_at_spacing(
        pdb, fit_solvent_rfactor.TARGET_SPACING_DEFAULT, "Refmac")
    assert r_def["target_occupancy"] == ref
    assert max(e / n for e, n in zip(cell_edges(), ref_grid)) <= 0.7 + 1e-9

    _, s_15, text_s = fit("--rate", "1.5", "--target-spacing", "same")
    _, s_3, _ = fit("--rate", "3", "--target-spacing", "same")
    assert s_15["target_occupancy"] == s_15["gemmi_occupancy_working_grid"]
    assert s_3["target_occupancy"] == s_3["gemmi_occupancy_working_grid"]
    assert s_15["target_occupancy"] != s_3["target_occupancy"]
    assert "--target-spacing same" in text_s

    # an explicit --target-occupancy still wins over either, and is reported
    # as the override it is, not as a gemmi measurement
    _, o, text_o = fit("--target-occupancy", "0.95")     # the toy is mostly empty cell
    assert o["target_occupancy"] == 0.95 and o["target_source"] == "--target-occupancy"
    assert "reference grid" not in text_o
    assert r_def["target_source"] == "gemmi, 0.7 A reference grid"
    assert s_15["target_source"] == "gemmi, working grid"

    # --mask gemmi uses gemmi's mask on the WORKING grid, and the occupancy
    # reported is that mask's, whatever the target was measured on or set to
    _, g, _ = fit("--mask", "gemmi", "--target-occupancy", "0.95")
    assert g["mask_occupancy"] == g["gemmi_occupancy_working_grid"] != 0.95


def test_target_spacing_arg():
    import argparse

    import pytest

    import fit_solvent_rfactor as f

    assert f.target_spacing_arg("same") == "same" and f.target_spacing_arg("SAME") == "same"
    assert f.target_spacing_arg("0.6") == 0.6
    for bad in ("0", "-1", "fine"):
        with pytest.raises(argparse.ArgumentTypeError):
            f.target_spacing_arg(bad)
    assert f.TARGET_SPACING_DEFAULT > 0.0


def test_shrink_step_distance():
    """The near-step check looks at voxel edges and face diagonals against
    gemmi's shrink radius, the lengths at which its occupancy steps; the
    7FPV grids the reviewer measured are the cases."""
    import numpy as np

    import fit_solvent_rfactor as f
    from lunus.sf.cell_utils import orth_matrix

    M = orth_matrix(34.196, 45.558, 99.044, 90.0, 90.0, 90.0)     # 7FPV
    assert f.shrink_step_distance(M, (50, 72, 144), 0.8) is None    # edges .68 .63 .69
    assert f.shrink_step_distance(M, (50, 82, 144), 0.8) is None    # one edge .56, diagonals > .8
    near = f.shrink_step_distance(M, (62, 82, 144), 0.8)            # (a,b) diagonal 0.784
    assert near is not None and abs(near - 0.784) < 2e-3
    assert f.shrink_step_distance(M, (43, 57, 124), 0.8) is not None  # an edge at ~0.8
    # an oblique cell: the shorter face diagonal is the one that counts.
    # gamma = 75 deg with 0.652 A edges puts |a - b| at 0.794 A, within 5% of
    # 0.8, while the edges (0.652) and the other diagonals (0.92, 1.04) are
    # clear; the same edges in a rectangular cell have no diagonal near 0.8
    n = 46
    Mo = orth_matrix(30.0, 30.0, 30.0, 90.0, 90.0, 75.0)
    d_short = np.linalg.norm(Mo[:, 0] / n - Mo[:, 1] / n)
    assert abs(d_short - 0.794) < 2e-3
    near = f.shrink_step_distance(Mo, (n, n, n), 0.8)
    assert near is not None and abs(near - d_short) < 1e-9
    Mr = orth_matrix(30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
    assert f.shrink_step_distance(Mr, (n, n, n), 0.8) is None


@requires_cctbx
def test_mask_gemmi_calibrates_nothing(tmp_path, monkeypatch):
    """--mask gemmi uses gemmi's mask on the working grid as it is: no
    reference-grid target is built, the run says so, and the result names the
    occupancy as that mask's and not a target. On voxels no finer than gemmi's
    shrink radius it warns, with or without a --target-occupancy."""
    import contextlib
    import io
    import warnings

    import fit_solvent_rfactor
    from lunus.sf.solvent_torch import SolventMaskWarning

    pdb, mtz = sharp_toy(tmp_path, sharp=False)
    calls = []
    real = fit_solvent_rfactor.gemmi_mask_occupancy_at_spacing

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(fit_solvent_rfactor, "gemmi_mask_occupancy_at_spacing", spy)
    args = fit_solvent_rfactor.build_parser().parse_args(
        [pdb, mtz, "--shells", "2", "--mask", "gemmi"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = fit_solvent_rfactor.run(args)
    assert calls == []
    assert "calibration target: none (--mask gemmi" in out.getvalue()
    assert r["target_source"] == "none (--mask gemmi)"
    assert r["target_occupancy"] == r["gemmi_occupancy_working_grid"] == r["mask_occupancy"]
    # the toy at 2.5 A sits on 0.83 A voxels, no finer than the 0.8 A Refmac
    # shrink radius: gemmi's mask has no shrink step there, and the run says so
    assert any(issubclass(w.category, SolventMaskWarning)
               and "no finer than gemmi's Refmac shrink" in str(w.message)
               for w in caught)
    args2 = fit_solvent_rfactor.build_parser().parse_args(
        [pdb, mtz, "--shells", "2", "--mask", "gemmi", "--max-voxel", "0.7"])
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            fit_solvent_rfactor.run(args2)
    assert not any(issubclass(w.category, SolventMaskWarning) for w in caught2)
    assert calls == []
    # the warning is about the mask in use, so an explicit target does not silence it
    args3 = fit_solvent_rfactor.build_parser().parse_args(
        [pdb, mtz, "--shells", "2", "--mask", "gemmi", "--target-occupancy", "0.95"])
    with warnings.catch_warnings(record=True) as caught3:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            fit_solvent_rfactor.run(args3)
    assert any(issubclass(w.category, SolventMaskWarning) for w in caught3)


@requires_cctbx
def test_mask_source_is_the_atom_presence_field(tmp_path):
    """The threshold mask is cut from an atom-presence field (every atom a
    unit-weight Gaussian at --mask-source-b, default 20 A^2) and not from the
    electron density: on a model whose atoms differ in B and element the two
    sources give different masks and different R, the field's occupancy is
    still calibrated to the target, and --mask-source density reproduces the
    old mask. F_protein (the no-solvent fit) is the same under both."""
    import contextlib
    import io

    import fit_solvent_rfactor

    pdb, mtz = sharp_toy(tmp_path, aniso=True)      # B 2-20, C/N/O/S

    def fit(*extra):
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb, mtz, "--shells", "2"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = fit_solvent_rfactor.run(args)
        return args, r, out.getvalue()

    args, r, text = fit()
    assert args.mask_source == "atoms" and args.mask_source_b == 20.0, (
        "default mask source is %r at B %r, not the atom-presence field at 20"
        % (getattr(args, "mask_source", None), getattr(args, "mask_source_b", None)))
    assert r["mask_source"] == "atoms" and r["mask_source_b"] == 20.0
    assert "mask source: atom-presence field" in text
    assert abs(r["mask_occupancy"] - r["target_occupancy"]) < 2e-3
    _, rd, textd = fit("--mask-source", "density")
    assert rd["mask_source"] == "density" and rd["mask_source_b"] is None
    assert "mask source: the electron density" in textd
    assert rd["R-work no solvent"] == r["R-work no solvent"]      # F_protein untouched
    assert rd["mask_cutoff"] != r["mask_cutoff"]
    assert rd["R-work"] != r["R-work"]
    _, r60, _ = fit("--mask-source-b", "60")
    assert r60["mask_source_b"] == 60.0 and r60["mask_cutoff"] != r["mask_cutoff"]

    # the unit weighting: relabel the toy's sulfur as a carbon at the same
    # place and the atoms-source cutoff barely moves, the density's does
    text_pdb = open(pdb).read()
    pdb_c = str(tmp_path / "sharp_c.pdb")
    def relabel(line):
        # name in columns 13-16, element symbol the last two characters; the
        # ANISOU record carries both too and must match its ATOM
        if line.startswith(("ATOM", "ANISOU")) and line.rstrip()[-2:] == " S":
            return line[:13] + "C  " + line[16:-2] + " C"
        return line

    relabelled = "\n".join(relabel(l) for l in text_pdb.splitlines())
    assert relabelled != text_pdb and not any(
        l.rstrip()[-2:] == " S" for l in relabelled.splitlines() if l.startswith("ATOM"))
    open(pdb_c, "w").write(relabelled + "\n")

    def fit_c(*extra):
        args = fit_solvent_rfactor.build_parser().parse_args(
            [pdb_c, mtz, "--shells", "2"] + list(extra))
        with contextlib.redirect_stdout(io.StringIO()):
            return fit_solvent_rfactor.run(args)

    r_c = fit_c()
    rd_c = fit_c("--mask-source", "density")
    rel_atoms = abs(r_c["mask_cutoff"] - r["mask_cutoff"]) / r["mask_cutoff"]
    rel_dens = abs(rd_c["mask_cutoff"] - rd["mask_cutoff"]) / rd["mask_cutoff"]
    # what is left is the IT92 shape difference between S and C, a fraction
    # of the electron-count effect (measured 0.08 against 0.20 here)
    assert rel_atoms < 0.5 * rel_dens, (rel_atoms, rel_dens)
