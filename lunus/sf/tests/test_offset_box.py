"""density_torch.precompute_element_offsets: the candidate box reaches every
grid offset within an atom's radius plus the half-voxel margin, in any cell.

The callers size the box as ceil(radius / (|a_i| / N_i)) steps per axis, the
step length ALONG the axis. In an oblique cell the spacing between grid
planes, 1 / (|a*_i| N_i), is smaller, so the sphere of kept offsets reaches
further in steps than that, and the box used to clip it. The reference here
is the same call with the box padded by 10 steps on every axis, which the
distance filter then trims back to the sphere: the two must keep exactly the
same offsets. Torch only, well under a second.
"""

import numpy as np
import pytest
import torch

from lunus.sf.cell_utils import orth_matrix
from lunus.sf.density_torch import precompute_element_offsets
from lunus.sf.mask_spheres import sphere_kernels

# 8VPW's cell (C 2, beta 115.9), a hexagonal cell (P 65 2 2, gamma 120,
# 2XLW-like), and an orthogonal one (P 21 21 21) as the control.
CELLS = {
    "monoclinic_8vpw": (101.6, 33.6, 74.0, 90.0, 115.9, 90.0),
    "hexagonal": (60.0, 60.0, 150.0, 90.0, 90.0, 120.0),
    "orthorhombic": (34.2, 45.6, 99.0, 90.0, 90.0, 90.0),
}
SPACING = 0.6           # A per step along each axis, near d_min/3 at 1.8 A
RADII = [2.9, 3.6, 4.4, 9.0]  # per "element", A: R + taper at 1e-4, and a high-B
# atom. At 0.6 A steps the 9.0 A sphere reaches 3 steps beyond the callers' box
# on the hexagonal cell, so a fix that pads by a constant 2 does not pass.


def _setup(cell):
    M = torch.tensor(orth_matrix(*cell), dtype=torch.float64)
    grid = tuple(int(np.ceil(n)) for n in np.array(cell[:3]) / SPACING)
    # the callers' sizing (kernel_torch, mask_spheres)
    spacing = M.norm(dim=0).numpy() / np.array(grid)
    vox = torch.tensor(np.stack([np.ceil(r / spacing) for r in RADII]).astype(np.int64))
    return M, grid, vox, torch.tensor(RADII, dtype=torch.float64)


def _offset_sets(elem_offsets):
    return [set(map(tuple, o.offsets.tolist())) for o in elem_offsets]


@pytest.mark.parametrize("name", sorted(CELLS))
def test_offsets_equal_those_of_a_generously_padded_box(name):
    M, grid, vox, radii = _setup(CELLS[name])
    got = _offset_sets(precompute_element_offsets(vox, radii, M, grid, "cpu"))
    ref = _offset_sets(precompute_element_offsets(vox + 10, radii, M, grid, "cpu"))
    for r, g, f in zip(RADII, got, ref):
        missing = len(f - g)
        assert missing == 0, ("%s, radius %.1f A: %d of %d offsets within reach are "
                              "outside the box" % (name, r, missing, len(f)))
        assert g == f


@pytest.mark.parametrize("name", sorted(CELLS))
def test_every_voxel_within_reach_of_an_off_centre_atom_is_a_candidate(name):
    """Independent of the function's own margin: put the atom at each corner
    of its sub-voxel range, the worst case for the remainder, and check by
    brute force that every grid offset within the radius is in the list. The
    padded-box test above cannot see a margin that is too small, since both
    of its calls share it."""
    M, grid, vox, radii = _setup(CELLS[name])
    elem_offsets = precompute_element_offsets(vox, radii, M, grid, "cpu")
    N = torch.tensor(grid, dtype=torch.float64)
    reach = [int(v) + 12 for v in vox.max(dim=0).values]
    axes = [torch.arange(-r, r + 1) for r in reach]
    cube = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    corners = torch.tensor([[su, sv, sw] for su in (-0.499, 0.499)
                            for sv in (-0.499, 0.499) for sw in (-0.499, 0.499)],
                           dtype=torch.float64)
    for r, eo in zip(RADII, elem_offsets):
        kept = set(map(tuple, eo.offsets.tolist()))
        for s in corners:
            d = ((cube.to(torch.float64) - s) / N @ M.T).norm(dim=1)
            within = set(map(tuple, cube[d <= r].tolist()))
            missing = len(within - kept)
            assert missing == 0, ("%s, radius %.1f A, remainder %s: %d voxels within "
                                  "the radius are not candidates"
                                  % (name, r, s.tolist(), missing))


def test_solvent_sphere_offsets_equal_those_of_a_padded_box(monkeypatch):
    """The same through mask_spheres.sphere_kernels, the solvent mask's
    caller, on the monoclinic cell: its box comes from the same function."""
    import lunus.sf.density_torch as dt
    cell = CELLS["monoclinic_8vpw"]
    M = orth_matrix(*cell)
    grid = tuple(int(np.ceil(n)) for n in np.array(cell[:3]) / SPACING)
    radii = np.array([1.6, 1.9, 2.0])
    got = _offset_sets(sphere_kernels(radii, grid, M)[2])
    real = dt.precompute_element_offsets
    calls = []

    def padded(vox, *a):
        calls.append(1)
        return real(vox + 10, *a)

    monkeypatch.setattr(dt, "precompute_element_offsets", padded)
    ref = _offset_sets(sphere_kernels(radii, grid, M)[2])
    # sphere_kernels imports the function at call time; if that ever moves to
    # module level the patch would not take, and got == ref would be vacuous
    assert calls, "sphere_kernels did not go through the patched function"
    missing = len(ref[0] - got[0])
    assert missing == 0, ("solvent spheres: %d of %d offsets within reach are outside "
                          "the box" % (missing, len(ref[0])))
    assert got == ref

