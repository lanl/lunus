"""The waters nearest a solute, found frame by frame on a grid.

For xtraj's nearest_waters= option: in each frame of an MD trajectory, keep
the same NUMBER of waters -- the ones closest to the solute in that frame --
rather than the same waters. A water is ordered only on average; any one
molecule moves in and out of the ordered sites.

The search is a cell list, not an all-against-all distance matrix. The box is
cut into cells no narrower than the search radius, so a water within the
radius of a solute atom is in that atom's cell or one of the 26 around it.
Waters with no solute atom in those 27 cells are never tested, and the rest
are tested against those cells' atoms only.

numpy only. The mdtraj topology the table builders read is passed in.
"""

import itertools

import numpy as np

# The most cells the grid is cut into along one axis: 128**3 of them is 17 MB
# per index array. Only a search radius under 1/128 of the box reaches it.
MAX_CELLS_PER_AXIS = 128


def min_distances(points, ref, cell, r_max, max_pairs=4000000):
    """Distance from each point to the nearest reference atom, where that is
    no more than r_max; np.inf where it is more.

    points (n, 3) and ref (m, 3) are Cartesian, in the units of r_max. cell
    is the periodic box as a (3, 3) array whose ROWS are the lattice vectors
    (mdtraj's unitcell_vectors convention), or None for no periodicity.
    Coordinates need not be wrapped into the box.

    A periodic search needs r_max no more than half the box's narrowest
    perpendicular width: up to there the nearest image is the one whose
    fractional separation rounds to zero, in a triclinic box as well, and
    beyond it that is not so. ValueError otherwise.

    max_pairs bounds the point-atom pairs held at once, and so the memory.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    best = np.full(len(points), np.inf)       # squared, until the end
    if len(points) == 0 or len(ref) == 0:
        return best
    if not r_max > 0:
        raise ValueError("r_max must be positive; got %r" % (r_max,))

    periodic = cell is not None
    if periodic:
        cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        inv = np.linalg.inv(cell)
        # Column i of the inverse is the reciprocal vector a*_i, and
        # 1/|a*_i| the spacing of the planes that bound the box along i.
        widths = 1.0 / np.linalg.norm(inv, axis=0)
        if r_max > 0.5 * widths.min():
            raise ValueError(
                "search radius %.3f exceeds half the box's narrowest width "
                "(%.3f); the nearest periodic image is not well defined "
                "beyond that" % (r_max, widths.min()))
        # >= 2 by the check. Capped: cells wider than r_max find the same
        # pairs, and a very small r_max would otherwise ask for more cells
        # than memory holds.
        n = np.minimum(np.floor(widths / r_max), MAX_CELLS_PER_AXIS).astype(np.intp)
        frac_points = points @ inv
        frac_points -= np.floor(frac_points)
        frac_ref = ref @ inv
        frac_ref -= np.floor(frac_ref)
        # x - floor(x) rounds up to exactly 1.0 for a tiny negative x.
        cell_points = np.minimum((frac_points * n).astype(np.intp), n - 1)
        cell_ref = np.minimum((frac_ref * n).astype(np.intp), n - 1)
        inside = np.ones(len(points), dtype=bool)
        # With two cells along an axis, -1 and +1 are the same neighbour.
        steps = [(0, 1) if k == 2 else (-1, 0, 1) for k in n]
    else:
        origin = ref.min(axis=0) - r_max
        extent = ref.max(axis=0) + r_max - origin
        # Cells r_max wide, or wider where that would be too many of them.
        size = np.maximum(r_max, extent / MAX_CELLS_PER_AXIS)
        n = np.floor(extent / size).astype(np.intp) + 1
        cell_points = np.floor((points - origin) / size).astype(np.intp)
        cell_ref = np.floor((ref - origin) / size).astype(np.intp)
        # A point outside the reference's bounding box grown by r_max has
        # nothing within r_max.
        inside = np.all((cell_points >= 0) & (cell_points < n), axis=1)
        steps = [(-1, 0, 1)] * 3

    def flat(c):
        return (c[:, 0] * n[1] + c[:, 1]) * n[2] + c[:, 2]

    ref_cell = flat(cell_ref)
    counts = np.bincount(ref_cell, minlength=int(np.prod(n)))
    starts = np.cumsum(counts) - counts
    order = np.argsort(ref_cell, kind="stable")     # ref atoms, cell by cell

    for step in itertools.product(*steps):
        neighbour = cell_points + np.array(step, dtype=np.intp)
        if periodic:
            neighbour %= n
            valid = inside
        else:
            valid = inside & np.all((neighbour >= 0) & (neighbour < n), axis=1)
            neighbour = np.clip(neighbour, 0, n - 1)
        neighbour = flat(neighbour)
        # Reference atoms each point meets in this neighbouring cell. Zero
        # for most points far from the solute: those are never tested.
        count = np.where(valid, counts[neighbour], 0)
        start = starts[neighbour]
        running = np.cumsum(count)

        lo = 0
        while lo < len(points):
            done = running[lo - 1] if lo else 0
            hi = int(np.searchsorted(running, done + max_pairs, side="right"))
            hi = min(max(hi, lo + 1), len(points))
            c = count[lo:hi]
            total = int(c.sum())
            if total:
                first = np.cumsum(c) - c
                point = np.repeat(np.arange(lo, hi), c)
                atom = order[np.repeat(start[lo:hi] - first, c) + np.arange(total)]
                if periodic:
                    d = frac_points[point] - frac_ref[atom]
                    d -= np.round(d)
                    d = d @ cell
                else:
                    d = points[point] - ref[atom]
                d2 = np.einsum("ij,ij->i", d, d)
                has = c > 0
                tested = np.arange(lo, hi)[has]
                best[tested] = np.minimum(
                    best[tested], np.minimum.reduceat(d2, first[has]))
            lo = hi

    best = np.sqrt(best)
    # Atoms in a neighbouring cell can be farther than r_max; a minimum
    # beyond it is not known to be the minimum.
    best[best > r_max] = np.inf
    return best


def nearest_points(points, ref, cell, n, r_start):
    """The n points nearest the reference atoms.

    Returns (indices into points, ascending; the distance of the farthest of
    them). The search starts at radius r_start and widens until it holds n
    points, so r_start is a speed setting, not a limit. Arguments as
    min_distances. ValueError when n points cannot be found: more asked for
    than there are, or, in a periodic box, fewer than n within half the box's
    narrowest width.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1, 3)
    if n > len(points):
        raise ValueError("%d points requested but only %d to choose from"
                         % (n, len(points)))
    if n <= 0:
        return np.zeros(0, dtype=np.intp), 0.0
    if len(ref) == 0:
        raise ValueError("no reference atoms to measure distances to")

    if cell is not None:
        inv = np.linalg.inv(np.asarray(cell, dtype=np.float64).reshape(3, 3))
        limit = 0.5 * (1.0 / np.linalg.norm(inv, axis=0)).min()
    else:
        both = np.concatenate((points, ref))
        limit = np.linalg.norm(both.max(axis=0) - both.min(axis=0)) + 1.0

    r = r_start
    while True:
        r = min(r, limit)
        d = min_distances(points, ref, cell, r)
        near = np.nonzero(np.isfinite(d))[0]
        if len(near) >= n:
            pick = near[np.argsort(d[near], kind="stable")[:n]]
            return np.sort(pick), float(d[pick].max())
        if r >= limit:
            raise ValueError(
                "%d points requested but only %d lie within %.2f of the "
                "reference atoms, the widest search the box allows (half "
                "its narrowest width)" % (n, len(near), limit))
        r *= 1.5


def water_atom_mask(topology):
    """Which atoms of an mdtraj topology belong to water residues."""
    return np.array([a.residue.is_water for a in topology.atoms], dtype=bool)


def heavy_atom_mask(topology):
    """Which atoms of an mdtraj topology are not hydrogen (or deuterium).
    Atoms with no element, virtual sites for one, are not heavy."""
    return np.array(
        [a.element is not None and a.element.symbol not in ("H", "D")
         for a in topology.atoms], dtype=bool)


def water_tables(topology, selection_mask):
    """The waters a selection offers, as index tables.

    Returns (position_idx, atoms). position_idx[w] is the atom water w is
    located by: its oxygen, whether or not the selection includes it.
    atoms[w] holds the water's SELECTED atoms, so shape (n_waters, k). Waters
    with no selected atom are left out. Every water must contribute the same
    number of atoms -- that is what keeps the atom count fixed from frame to
    frame -- and ValueError says so when they do not.
    """
    position_idx = []
    atoms = []
    for residue in topology.residues:
        if not residue.is_water:
            continue
        indices = [a.index for a in residue.atoms]
        kept = [i for i in indices if selection_mask[i]]
        if not kept:
            continue
        oxygens = [a.index for a in residue.atoms
                   if a.element is not None and a.element.symbol == "O"]
        position_idx.append(oxygens[0] if oxygens else indices[0])
        atoms.append(kept)
    sizes = sorted(set(len(a) for a in atoms))
    if len(sizes) > 1:
        raise ValueError(
            "the selected waters do not all have the same number of selected "
            "atoms (found %s per water), so the atom count could not be kept "
            "fixed from frame to frame" % ", ".join(str(s) for s in sizes))
    return (np.array(position_idx, dtype=np.intp),
            np.array(atoms, dtype=np.intp).reshape(len(atoms), sizes[0] if sizes else 0))
