"""The grid search behind xtraj's nearest_waters= against brute force.

min_distances() and nearest_points() use a cell list; the reference here is
the full distance matrix over all 27 periodic images, which shares none of
the grid's bookkeeping. The end-to-end check, through the xtraj command
line, is test_xtraj_nearest_waters.py.
"""

import numpy as np
import pytest

from lunus.sf.nearest_waters import (
    heavy_atom_mask, min_distances, nearest_points, protein_atom_mask,
    water_atom_mask, water_tables,
)

ORTHORHOMBIC = np.diag([31.0, 24.0, 40.0])
# Rows are the lattice vectors, as mdtraj gives them.
TRICLINIC = np.array([[30.0, 0.0, 0.0],
                      [9.0, 26.0, 0.0],
                      [-7.0, 5.0, 33.0]])


def brute_force(points, ref, cell):
    """Minimum distance from each point to any reference atom, over every
    periodic image that could hold it. Two boxes each way: system() scatters
    points up to a box and a half from the reference."""
    if cell is None:
        shifts = np.zeros((1, 3))
    else:
        span = (-2, -1, 0, 1, 2)
        steps = np.array([(i, j, k) for i in span for j in span for k in span],
                         dtype=float)
        shifts = steps @ cell
    best = np.full(len(points), np.inf)
    for shift in shifts:
        d = points[:, None, :] - (ref[None, :, :] + shift)
        best = np.minimum(best, np.sqrt((d ** 2).sum(axis=2)).min(axis=1))
    return best


def system(cell, n_points=400, n_ref=150, seed=0, spread=3.0):
    """A compact clump of reference atoms, and points scattered through the
    box. In a periodic box the points also fill the boxes either side of it:
    unwrapped, as MD coordinates are."""
    rng = np.random.default_rng(seed)
    box = np.eye(3) * 30.0 if cell is None else cell
    centre = np.array([0.5, 0.5, 0.5]) @ box
    ref = centre + rng.normal(scale=spread, size=(n_ref, 3))
    low, high = (0.0, 1.0) if cell is None else (-1.0, 2.0)
    points = rng.uniform(low, high, size=(n_points, 3)) @ box
    return points, ref


@pytest.mark.parametrize("cell", [ORTHORHOMBIC, TRICLINIC, None],
                         ids=["orthorhombic", "triclinic", "open"])
@pytest.mark.parametrize("r_max", [3.5, 6.0, 11.0])
def test_min_distances_match_brute_force(cell, r_max):
    points, ref = system(cell)
    expected = brute_force(points, ref, cell)
    got = min_distances(points, ref, cell, r_max)

    within = expected <= r_max
    assert within.sum() > 5, "the test system puts nothing inside the radius"
    assert (~within).sum() > 5, "the test system puts nothing outside it"
    np.testing.assert_allclose(got[within], expected[within], rtol=0, atol=1e-9)
    assert np.all(np.isinf(got[~within]))


def test_reference_spanning_the_boundary():
    """The clump sits on a box corner, so its neighbours are found through
    the periodic wrap in all three directions."""
    rng = np.random.default_rng(3)
    ref = rng.normal(scale=2.5, size=(120, 3))            # around the origin
    points = rng.uniform(0.0, 1.0, size=(500, 3)) @ TRICLINIC
    expected = brute_force(points, ref, TRICLINIC)
    got = min_distances(points, ref, TRICLINIC, 5.0)
    within = expected <= 5.0
    assert within.sum() > 20
    np.testing.assert_allclose(got[within], expected[within], rtol=0, atol=1e-9)
    assert np.all(np.isinf(got[~within]))


def test_pair_budget_does_not_change_the_answer():
    points, ref = system(ORTHORHOMBIC, n_points=300, n_ref=400)
    whole = min_distances(points, ref, ORTHORHOMBIC, 8.0)
    pieces = min_distances(points, ref, ORTHORHOMBIC, 8.0, max_pairs=37)
    np.testing.assert_array_equal(whole, pieces)


def test_radius_beyond_half_the_box_is_refused():
    points, ref = system(ORTHORHOMBIC)
    with pytest.raises(ValueError, match="narrowest width"):
        min_distances(points, ref, ORTHORHOMBIC, 12.5)    # box is 24 wide


@pytest.mark.parametrize("cell", [ORTHORHOMBIC, TRICLINIC, None],
                         ids=["orthorhombic", "triclinic", "open"])
def test_nearest_points_are_the_n_closest(cell):
    points, ref = system(cell)
    expected = brute_force(points, ref, cell)
    n = 25
    picked, farthest = nearest_points(points, ref, cell, n, r_start=4.0)
    np.testing.assert_array_equal(picked, np.sort(np.argsort(expected)[:n]))
    assert farthest == pytest.approx(np.sort(expected)[n - 1], abs=1e-9)


def test_nearest_points_widens_a_radius_that_starts_too_small():
    points, ref = system(ORTHORHOMBIC)
    expected = brute_force(points, ref, ORTHORHOMBIC)
    n = 40
    assert (expected <= 0.5).sum() < n          # the start cannot hold them
    picked, _ = nearest_points(points, ref, ORTHORHOMBIC, n, r_start=0.5)
    np.testing.assert_array_equal(picked, np.sort(np.argsort(expected)[:n]))


def test_nearest_points_refuses_what_it_cannot_find():
    points, ref = system(ORTHORHOMBIC, n_points=50)
    with pytest.raises(ValueError, match="only 50 to choose from"):
        nearest_points(points, ref, ORTHORHOMBIC, 51, r_start=4.0)
    # All 50 exist, but not within half the box of a tight clump.
    points, ref = system(ORTHORHOMBIC, n_points=400, spread=0.5)
    with pytest.raises(ValueError, match="widest search the box allows"):
        nearest_points(points, ref, ORTHORHOMBIC, 400, r_start=4.0)


def test_nearest_points_of_none():
    points, ref = system(ORTHORHOMBIC)
    picked, farthest = nearest_points(points, ref, ORTHORHOMBIC, 0, r_start=4.0)
    assert picked.size == 0 and farthest == 0.0


PDB = """\
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1
ATOM      1  N   ALA A   1       8.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       9.400  10.000  10.000  1.00 20.00           C
ATOM      3  HA  ALA A   1       9.800  11.000  10.000  1.00 20.00           H
ATOM      4  O   HOH W   1       3.000   3.000   3.000  1.00 20.00           O
ATOM      5  H1  HOH W   1       3.900   3.000   3.000  1.00 20.00           H
ATOM      6  H2  HOH W   1       2.700   3.900   3.000  1.00 20.00           H
ATOM      7  H1  HOH W   2      15.900   3.000   3.000  1.00 20.00           H
ATOM      8  O   HOH W   2      15.000   3.000   3.000  1.00 20.00           O
ATOM      9  H2  HOH W   2      14.700   3.900   3.000  1.00 20.00           H
END
"""


@pytest.fixture
def topology(tmp_path):
    md = pytest.importorskip("mdtraj")
    path = tmp_path / "waters.pdb"
    path.write_text(PDB)
    return md.load_topology(str(path))


def test_masks(topology):
    np.testing.assert_array_equal(water_atom_mask(topology),
                                  [0, 0, 0, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(protein_atom_mask(topology),
                                  [1, 1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(heavy_atom_mask(topology),
                                  [1, 1, 0, 1, 0, 0, 0, 1, 0])


def test_water_tables(topology):
    position, atoms = water_tables(topology, np.ones(9, dtype=bool))
    np.testing.assert_array_equal(position, [3, 7])       # the oxygens
    np.testing.assert_array_equal(atoms, [[3, 4, 5], [6, 7, 8]])

    # Hydrogens deselected: one atom per water, still located by its oxygen.
    position, atoms = water_tables(topology, heavy_atom_mask(topology))
    np.testing.assert_array_equal(position, [3, 7])
    np.testing.assert_array_equal(atoms, [[3], [7]])

    # A water with nothing selected is not offered at all.
    mask = np.ones(9, dtype=bool)
    mask[3:6] = False
    position, atoms = water_tables(topology, mask)
    np.testing.assert_array_equal(position, [7])
    np.testing.assert_array_equal(atoms, [[6, 7, 8]])

    # No waters selected: empty tables, not an error.
    position, atoms = water_tables(topology, ~water_atom_mask(topology))
    assert position.size == 0 and atoms.shape[0] == 0


def test_water_tables_refuse_uneven_waters(topology):
    mask = np.ones(9, dtype=bool)
    mask[4] = False                              # one hydrogen of one water
    with pytest.raises(ValueError, match="same number of selected atoms"):
        water_tables(topology, mask)


@pytest.mark.parametrize("cell", [ORTHORHOMBIC, None], ids=["orthorhombic", "open"])
def test_tiny_radius_is_capped_not_a_huge_grid(cell):
    """A radius far below the box size would ask for millions of cells per
    axis; the grid is capped and the answer is unchanged."""
    points, ref = system(cell, n_points=50, n_ref=40)
    points[:5] = ref[:5] + 5e-6                  # five points at an atom
    got = min_distances(points, ref, cell, 1e-5)
    expected = brute_force(points, ref, cell)
    within = expected <= 1e-5
    assert within.sum() == 5
    np.testing.assert_allclose(got[within], expected[within], rtol=0, atol=1e-9)
    assert np.all(np.isinf(got[~within]))
