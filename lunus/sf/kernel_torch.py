"""
Builds the per-element (A, lam, offsets) tables as PyTorch tensors, ready
for density_torch.compute_density -- including the sphere-restricted local
offsets, computed ONCE here rather than per frame (see
density_torch.precompute_element_offsets's docstring for why that
matters for speed).

The B-factor/kernel-coefficient computation is NOT differentiated
(B-factor/blur are fixed hyperparameters, not learned), so it's done
once in NumPy via kernel.py and then converted to tensors.

WHERE THE TAPER BEGINS. kernel.py returns R, the radius at which an atom's
density equals the cutoff threshold. The raised-cosine window that
compute_density applies BEGINS AT THE THRESHOLD: it occupies
[R, R + taper_width], so inside R the density is identical to what a hard
cutoff gives, and only density already below the threshold is attenuated,
over a tail of width taper_width. So the radius handed to compute_density,
and the radius the candidate offset boxes are sized by, is R + taper_width.
The three builders below add the width in one place each; kernel.py keeps
the meaning "density equals the threshold" for R.

This corrects an error in how the method worked before 2026-10-30. The
window used to REACH ZERO AT THE THRESHOLD, occupying [R - w, R], which
made the density inside the cutoff radius lower than the hard cutoff's,
and the amount removed grow with the width -- the taper study
(docs/design.md) measured a 0.3 A window costing a factor of two in forward
error at cutoff 1e-2. With the taper beginning at the threshold the density
removed is bounded by the threshold value times the tail volume whatever
the width, and parity with gemmi improved 4.5x, at the price of ~9-15% more
candidate voxels per atom (work ~ r^3; carbon at 1e-4: 3.43 -> 3.53 A).
"""

import numpy as np
import torch

from .kernel import build_element_kernels, build_atom_kernels
from .density_torch import precompute_element_offsets


def build_element_kernels_torch(
    element_symbols,          # list of distinct element symbols present, defines index order
    coeff_table,               # e.g. elements.IT92_COEFFS
    b_iso, blur,
    grid_shape,                 # (Nu, Nv, Nw)
    orth_matrix_np,              # (3,3) numpy, Cartesian = orth_matrix @ frac
    cutoff=1e-5,
    taper_width=None,           # A; if None, defaults to 0.1 A. This is
                                 # NOT tied to grid spacing -- a numerical
                                 # experiment showed gradient correctness
                                 # holds to <0.06% relative error even at
                                 # taper_width as narrow as 0.05 A (an
                                 # order of magnitude below typical grid
                                 # spacing), since any nonzero width
                                 # removes the discrete in/out state that
                                 # breaks autograd -- grid spacing only
                                 # affects how finely the taper's SHAPE is
                                 # sampled, a minor secondary concern, not
                                 # gradient correctness. The taper begins
                                 # AT the threshold (module docstring), so
                                 # the width no longer sets how much density
                                 # is removed -- only how many extra
                                 # candidate voxels are visited. The ~0.5%
                                 # loss and the forward-value degradation
                                 # at 0.6 A were measured when the taper
                                 # reached zero at the threshold instead,
                                 # and are gone with that; 0.1 A stays the
                                 # default because the taper study found
                                 # every width from 0.01 to 0.3 A
                                 # equivalent for the optimizer.
    device="cpu",
    dtype=torch.float32,
):
    kernels = build_element_kernels(element_symbols, coeff_table, b_iso, blur, cutoff)

    Nu, Nv, Nw = grid_shape
    spacing = np.array([
        np.linalg.norm(orth_matrix_np[:, 0]) / Nu,
        np.linalg.norm(orth_matrix_np[:, 1]) / Nv,
        np.linalg.norm(orth_matrix_np[:, 2]) / Nw,
    ])

    if taper_width is None:
        taper_width = 0.1

    A_list, lam_list, radius_vox_list, radius_ang_list = [], [], [], []
    for el in element_symbols:
        k = kernels[el]
        A_list.append(k["A"])
        lam_list.append(k["lam"])
        outer = k["radius"] + taper_width        # window on [R, R + w]
        radius_vox_list.append(np.ceil(outer / spacing).astype(np.int64))
        radius_ang_list.append(outer)

    elem_A = torch.tensor(np.stack(A_list), device=device, dtype=dtype)
    elem_lam = torch.tensor(np.stack(lam_list), device=device, dtype=dtype)
    elem_radius_vox = torch.tensor(np.stack(radius_vox_list), device=device, dtype=torch.long)
    elem_radius_ang = torch.tensor(np.array(radius_ang_list), device=device, dtype=dtype)

    orth_matrix = torch.tensor(orth_matrix_np, device=device, dtype=dtype)
    elem_offsets = precompute_element_offsets(
        elem_radius_vox, elem_radius_ang, orth_matrix, grid_shape, device
    )

    element_to_index = {el: i for i, el in enumerate(element_symbols)}

    return elem_A, elem_lam, elem_offsets, elem_radius_ang, taper_width, element_to_index


def build_atom_kernels_aniso_torch(
    elements_per_atom,
    element_symbols,
    coeff_table,
    u_cart_per_atom,            # (n_atoms,3,3), CARTESIAN A^2
    blur,
    grid_shape,
    orth_matrix_np,
    cutoff=1e-5,
    taper_width=None,
    device="cpu",
    dtype=torch.float32,
):
    """
    build_atom_kernels_torch's anisotropic counterpart.

    Returns (atom_A, atom_L6, elem_offsets, atom_radius_ang, taper_width,
    element_to_index), with atom_L6 (n_atoms,5,6) where the isotropic builder
    returns atom_lam (n_atoms,5).

    Everything else is deliberately identical, INCLUDING the offset candidate
    list, which stays keyed by element and sized by the largest radius among
    that element's atoms. An anisotropic atom needs a sphere containing its
    longest semi-axis, which cutoff_radius_aniso_batch already returns, so the
    existing per-atom prefix machinery handles the wider spread the same way
    it handles a high-B atom -- no new code path, and small-radius atoms in the
    same element group still take a short prefix.
    """
    from .kernel import build_atom_kernels_aniso

    A_np, L6_np, radius_np = build_atom_kernels_aniso(
        elements_per_atom, coeff_table, u_cart_per_atom, blur, cutoff)

    elements_per_atom = np.asarray(elements_per_atom)
    if taper_width is None:
        taper_width = 0.1
    radius_np = radius_np + taper_width          # window on [R, R + w]

    Nu, Nv, Nw = grid_shape
    spacing = np.array([np.linalg.norm(orth_matrix_np[:, i]) / n
                        for i, n in enumerate((Nu, Nv, Nw))])

    radius_vox_list, max_radius = [], []
    for el in element_symbols:
        el_mask = elements_per_atom == el
        r = float(radius_np[el_mask].max()) if el_mask.any() else 0.0
        radius_vox_list.append(np.ceil(r / spacing).astype(np.int64))
        max_radius.append(r)

    orth_matrix = torch.tensor(orth_matrix_np, device=device, dtype=dtype)
    elem_offsets = precompute_element_offsets(
        torch.tensor(np.stack(radius_vox_list), device=device, dtype=torch.long),
        torch.tensor(max_radius, device=device, dtype=dtype),
        orth_matrix, grid_shape, device)

    return (torch.tensor(A_np, device=device, dtype=dtype),
            torch.tensor(L6_np, device=device, dtype=dtype),
            elem_offsets,
            torch.tensor(radius_np, device=device, dtype=dtype),
            taper_width,
            {el: i for i, el in enumerate(element_symbols)})


def build_atom_kernels_torch(
    elements_per_atom,          # (n_atoms,) list/array of element symbols, one per atom
    element_symbols,            # distinct symbols present, defines elem_offsets index order
    coeff_table,                 # e.g. elements.IT92_COEFFS
    b_per_atom,                  # (n_atoms,) array, isotropic B-factor per atom, A^2
    blur,
    grid_shape,                   # (Nu, Nv, Nw)
    orth_matrix_np,                # (3,3) numpy, Cartesian = orth_matrix @ frac
    cutoff=1e-5,
    taper_width=None,             # A; None -> 0.1 A default, same reasoning as
                                   # build_element_kernels_torch (see there)
    device="cpu",
    dtype=torch.float32,
):
    """
    Per-ATOM version of build_element_kernels_torch: each atom gets its
    own (A, lam, radius) from its own B-factor, since B doesn't have to
    be uniform across the structure (though it's still assumed fixed
    across configurations -- if B needs to vary by configuration too,
    this would need to move from setup-time into the per-frame loop,
    which it currently is NOT built for).

    The offset candidate list stays keyed by ELEMENT (not atom), sized
    using the MAXIMUM radius among all atoms of that element -- generous
    enough to safely contain every atom of that type, including the
    highest-B (most spread out) ones. This keeps the vectorized,
    grouped-by-element density structure in density_torch.py unchanged;
    only the actual density formula and taper radius become per-atom
    (using each atom's own A/lam/radius, gathered via array indexing)
    rather than shared across a whole element group.

    Returns (atom_A, atom_lam, elem_offsets, atom_radius_ang, taper_width,
    element_to_index) -- atom_A/atom_lam/atom_radius_ang are (n_atoms,5),
    (n_atoms,5), (n_atoms,) tensors, already in per-atom order matching
    elements_per_atom; elem_offsets is a list of (K_e,3) tensors, one per
    DISTINCT element (same shape/role as build_element_kernels_torch's).
    """
    A_np, lam_np, radius_np = build_atom_kernels(elements_per_atom, coeff_table, b_per_atom, blur, cutoff)

    elements_per_atom = np.asarray(elements_per_atom)
    if taper_width is None:
        taper_width = 0.1
    radius_np = radius_np + taper_width          # window on [R, R + w]

    Nu, Nv, Nw = grid_shape
    spacing = np.array([
        np.linalg.norm(orth_matrix_np[:, 0]) / Nu,
        np.linalg.norm(orth_matrix_np[:, 1]) / Nv,
        np.linalg.norm(orth_matrix_np[:, 2]) / Nw,
    ])

    radius_vox_list = []
    for el in element_symbols:
        el_mask = elements_per_atom == el
        max_radius_el = float(radius_np[el_mask].max()) if el_mask.any() else 0.0
        radius_vox_list.append(np.ceil(max_radius_el / spacing).astype(np.int64))
        # per-element max radius is only used to size the offset candidate
        # box below -- the actual taper radius stays per-atom (atom_radius_ang)

    elem_radius_vox = torch.tensor(np.stack(radius_vox_list), device=device, dtype=torch.long)
    max_radius_per_elem = torch.tensor(
        [float(radius_np[elements_per_atom == el].max()) if (elements_per_atom == el).any() else 0.0
         for el in element_symbols],
        device=device, dtype=dtype,
    )

    orth_matrix = torch.tensor(orth_matrix_np, device=device, dtype=dtype)
    elem_offsets = precompute_element_offsets(
        elem_radius_vox, max_radius_per_elem, orth_matrix, grid_shape, device
    )

    atom_A = torch.tensor(A_np, device=device, dtype=dtype)
    atom_lam = torch.tensor(lam_np, device=device, dtype=dtype)
    atom_radius_ang = torch.tensor(radius_np, device=device, dtype=dtype)

    element_to_index = {el: i for i, el in enumerate(element_symbols)}

    return atom_A, atom_lam, elem_offsets, atom_radius_ang, taper_width, element_to_index
