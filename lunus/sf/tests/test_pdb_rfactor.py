"""tools/pdb_rfactor.py: the file cache, the mmtbx path, and the passthrough.

Nothing here touches the network. The cache test runs against the tracked
7FPV files in examples/compare_gemmi/ with the downloader stubbed to raise,
so a regression that re-fetches a file that is already present fails loudly
rather than silently costing a download. The torch path is
fit_solvent_rfactor.run(), which the design note validates by hand on 7FPV at
full resolution; at ~2 minutes it is not a unit test, so only the cheap mmtbx
path is run here, truncated to 3 A.
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
