"""Atom selections in the tools must be in the scatterers' (file) order.

iotbx's hierarchy groups atoms by altloc within each residue, so on a file
whose altlocs are interleaved atom by atom -- as deposited models often are,
7FPV among them -- a mask from the hierarchy's atom_selection_cache() is not
in the order of the scatterers xray_structure_simple builds, and applying it
to them selects the wrong atoms. On 7FPV the documented single-conformer
selection picked 109 B or C conformers and missed 109 A atoms.
"""

import io
import os
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

pytest.importorskip("cctbx")

TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

# File order N, CB A, CB B, OG A, OG B; the hierarchy's is N, CB A, OG A,
# CB B, OG B. A and B differ in x so each atom is identifiable by position.
INTERLEAVED = """\
CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1
ATOM      1  N   SER A   1       2.000   3.000   4.000  1.00 20.00           N
ATOM      2  CB ASER A   1       3.000   3.000   4.000  0.50 20.00           C
ATOM      3  CB BSER A   1       3.300   3.000   4.000  0.50 20.00           C
ATOM      4  OG ASER A   1       3.000   4.400   4.000  0.50 20.00           O
ATOM      5  OG BSER A   1       3.300   4.400   4.000  0.50 20.00           O
END
"""
ONE_CONFORMER = "altloc ' ' or altloc A"


def write(tmp_path):
    path = tmp_path / "interleaved.pdb"
    path.write_text(INTERLEAVED)
    return str(path)


def test_selection_in_file_order(tmp_path):
    from lunus.sf.atom_order import selection_in_file_order
    from iotbx.pdb import hierarchy

    pdb_in = hierarchy.input(file_name=write(tmp_path), sort_atoms=False)
    np.testing.assert_array_equal(selection_in_file_order(pdb_in, ONE_CONFORMER),
                                  [True, True, False, True, False])
    np.testing.assert_array_equal(selection_in_file_order(pdb_in, "name OG"),
                                  [False, False, False, True, True])
    # The hierarchy's own mask, the old code, is in the other order.
    old = np.array(pdb_in.hierarchy.atom_selection_cache().selection(ONE_CONFORMER))
    np.testing.assert_array_equal(old, [True, True, True, False, False])


def test_study_taper_selects_the_named_conformer(tmp_path):
    """Through load_system: --select keeps N, CB A and OG A (not CB B), and
    --free marks OG A among the kept atoms."""
    torch = pytest.importorskip("torch")
    from study_taper import load_system

    s = load_system(write(tmp_path), 3.0, "cpu", torch.float64, io.StringIO(),
                    select=ONE_CONFORMER, free="name OG")
    kept = [(sc.scattering_type, round(sc.site[0] * 20.0, 3))
            for sc in s["xrs"].scatterers()]
    assert kept == [("N", 2.0), ("C", 3.0), ("O", 3.0)]
    assert s["elements"] == ["N", "C", "O"]
    np.testing.assert_allclose(s["frac0"].numpy()[:, 0] * 20.0, [2.0, 3.0, 3.0])
    np.testing.assert_array_equal(s["free_np"], [False, False, True])


def test_selection_survives_repeated_labels(tmp_path):
    """An MD box repeats atom labels once residue numbers wrap (the
    compare_gemmi box has 21,690 duplicates); the mapping must not depend on
    labels being unique."""
    from lunus.sf.atom_order import selection_in_file_order
    from iotbx.pdb import hierarchy

    path = tmp_path / "repeats.pdb"
    path.write_text("""\
CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1
HETATM    1  O   HOH W   1       1.000   1.000   1.000  1.00 20.00           O
HETATM    2  O   HOH W   1       5.000   5.000   5.000  1.00 20.00           O
ATOM      3  N   SER A   1       2.000   3.000   4.000  1.00 20.00           N
END
""")
    pdb_in = hierarchy.input(file_name=str(path), sort_atoms=False)
    np.testing.assert_array_equal(selection_in_file_order(pdb_in, "water"),
                                  [True, True, False])
    np.testing.assert_array_equal(selection_in_file_order(pdb_in, "all"), [True] * 3)


def test_selection_after_a_pickled_copy_says_why(tmp_path):
    """xtraj broadcasts its hierarchy.input by pickling it, and a pickled copy
    no longer shares atoms between the file records and the hierarchy. The
    mapping must say so, not fail with a bare KeyError: xtraj computes it
    before the broadcast for this reason."""
    import pickle

    from iotbx.pdb import hierarchy
    from lunus.sf.atom_order import selection_in_file_order

    pdb_in = hierarchy.input(file_name=write(tmp_path), sort_atoms=False)
    copy = pickle.loads(pickle.dumps(pdb_in, protocol=4))
    with pytest.raises(ValueError, match="compute the selection before copying"):
        selection_in_file_order(copy, ONE_CONFORMER)


def test_selection_permutation_direction(tmp_path):
    """A reordering that is not its own inverse, so the two directions of the
    file/hierarchy permutation give different masks: file order N, CB A, C,
    OG A, O; hierarchy order N, C, O, CB, OG."""
    from iotbx.pdb import hierarchy
    from lunus.sf.atom_order import selection_in_file_order

    path = tmp_path / "cycle.pdb"
    path.write_text("""\
CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1
ATOM      1  N   SER A   1       2.000   3.000   4.000  1.00 20.00           N
ATOM      2  CB ASER A   1       3.000   3.000   4.000  1.00 20.00           C
ATOM      3  C   SER A   1       5.000   7.000   9.000  1.00 20.00           C
ATOM      4  OG ASER A   1       3.000   4.400   4.000  1.00 20.00           O
ATOM      5  O   SER A   1       6.000   8.000  11.000  1.00 20.00           O
END
""")
    pdb_in = hierarchy.input(file_name=str(path), sort_atoms=False)
    np.testing.assert_array_equal(selection_in_file_order(pdb_in, "name CB"),
                                  [False, True, False, False, False])
    np.testing.assert_array_equal(selection_in_file_order(pdb_in, "name O"),
                                  [False, False, False, False, True])
