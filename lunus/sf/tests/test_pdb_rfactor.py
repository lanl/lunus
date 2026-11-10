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
    # The main fit with bins also scales the solvent per bin by default
    # (measured 2026-11-10: 0.1886 / 0.1935), and the exponential is one
    # flag away (0.1880 / 0.1929, the numbers before per-bin solvent).
    assert r["solvent_scheme"] == "per bin"
    assert "solvent scale is one value per bin" in out.getvalue()
    assert abs(r["R-work"] - 0.1886) < 5e-4 and abs(r["R-free"] - 0.1935) < 1e-3
    args.solvent_bins = "off"
    with contextlib.redirect_stdout(io.StringIO()):
        r2 = fit_solvent_rfactor.run(args)
    assert r2["solvent_scheme"] == "exponential"
    assert abs(r2["R-work"] - 0.1880) < 5e-4 and abs(r2["R-free"] - 0.1929) < 1e-3


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
    # Measured 2026-11-10 on this branch; the exponential gives 0.1880 / 0.1929.
    assert abs(r["R-work binned"] - 0.1886) < 5e-4
    assert abs(r["R-free binned"] - 0.1935) < 1e-3
    assert abs(r["R-work"] - 0.1901) < 5e-4          # single-scale row unchanged
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
