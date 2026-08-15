"""
FFT + extraction half of the pipeline, matching xtraj.py's gemmi-engine
convention exactly:
    complex_grid = ifftn(density) * cell_volume
    F(hkl) = conjugate(complex_grid[h % Nu, k % Nv, l % Nw])
    (optionally: F(hkl) *= exp(blur * 0.25 * inv_d2)  -- undoes gemmi's blur)

torch.fft.ifftn is a standard differentiable linear operator, so this
composes with density_torch.splat_density to give end-to-end gradients
from F(hkl) all the way back to atomic fractional coordinates.
"""

import torch


def reciprocal_inv_d2(orth_matrix: torch.Tensor, hkl: torch.Tensor) -> torch.Tensor:
    """1/d^2 for each row of hkl (n,3), given the (3,3) orthogonalization
    matrix (Cartesian = orth_matrix @ frac)."""
    Mstar = torch.linalg.inv(orth_matrix).T             # reciprocal-lattice matrix
    s_cart = hkl.to(Mstar.dtype) @ Mstar.T               # (n,3)
    return (s_cart * s_cart).sum(-1)


def compute_fcalc(
    density_grid: torch.Tensor,   # (Nu,Nv,Nw) real, differentiable
    cell_volume: float,
    hkl: torch.Tensor,            # (n_refl,3) long
    orth_matrix: torch.Tensor = None,
    blur: float = 0.0,
) -> torch.Tensor:
    """Returns F(hkl), shape (n_refl,), complex, differentiable through
    density_grid (and therefore through atomic coordinates)."""
    Nu, Nv, Nw = density_grid.shape
    complex_dtype = torch.complex128 if density_grid.dtype == torch.float64 else torch.complex64
    Fgrid = torch.fft.ifftn(density_grid.to(complex_dtype)) * cell_volume

    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    u = torch.remainder(h, Nu)
    v = torch.remainder(k, Nv)
    w = torch.remainder(l, Nw)
    F = torch.conj(Fgrid[u, v, w]).resolve_conj()

    if blur != 0.0:
        if orth_matrix is None:
            raise ValueError("orth_matrix is required to apply the blur correction")
        inv_d2 = reciprocal_inv_d2(orth_matrix, hkl)
        F = F * torch.exp(blur * 0.25 * inv_d2).to(F.dtype)

    return F


def structure_factors_one_config(
    frac_coords, element_idx, occ,
    atom_A, atom_lam, elem_offsets, atom_radius_ang,
    grid_shape, orth_matrix, cell_volume,
    hkl, taper_width, blur=0.0, max_atoms_per_batch=50_000,
    grid_ops=None, compile_core=True, solvent=None,
):
    """One configuration, start to finish: splat -> symmetrize -> FFT ->
    extract F(hkl). Differentiable with respect to frac_coords. atom_A/atom_lam/
    atom_radius_ang are PER-ATOM (n_atoms, 5)/(n_atoms, 5)/(n_atoms,) --
    for the uniform-B path (kernel_torch.build_element_kernels_torch),
    broadcast with atom_A = elem_A[element_idx] etc. before calling this;
    kernel_torch.build_atom_kernels_torch already returns per-atom
    tensors directly. elem_offsets stays per-element (a generous
    candidate box, not exact per-atom data).

    grid_ops: from symmetry_torch.build_grid_ops(); when given, the density
    grid is symmetry-expanded before the FFT, exactly as gemmi's
    put_model_density_on_grid() does via symmetrize_sum(). Pass None (or an
    empty list, which is what a P1 space group produces) to splat the atoms
    as given with no symmetry expansion.

    solvent: a solvent_torch.SolventModel, or None. None is the default and
    runs no solvent code at all, so results are bit-identical to before this
    argument existed. When given, F_total = k_overall * exp(-h.U.h) *
    [F_protein + k_sol*exp(-b_sol*s^2/4)*F_mask] is returned instead, with the
    mask thresholded from THIS configuration's density -- see
    docs/solvent-design.md, and note that the mask is built after the symmetry
    expansion below, which is the only correct order."""
    from .density_torch import splat_density
    from .symmetry_torch import symmetrize_sum

    density = splat_density(
        frac_coords, element_idx, occ,
        atom_A, atom_lam, elem_offsets, atom_radius_ang,
        grid_shape, orth_matrix, taper_width,
        max_atoms_per_batch=max_atoms_per_batch, compile_core=compile_core,
    )
    if grid_ops:
        density = symmetrize_sum(density, grid_ops)
    F_protein = compute_fcalc(density, cell_volume, hkl, orth_matrix, blur=blur)
    if solvent is None:
        return F_protein
    # Built from the symmetry-expanded density, per the note above. The mask
    # FFT happens here rather than outside so that it lands INSIDE the
    # checkpointed region in structure_factors_batch -- otherwise the density
    # grid is retained for every configuration at once and the peak memory
    # checkpointing exists to control comes straight back.
    F_total, _mask = solvent.apply(density, F_protein, cell_volume, hkl, orth_matrix)
    return F_total


def structure_factors_batch(
    frac_coords_batch, element_idx, occ,
    atom_A, atom_lam, elem_offsets, atom_radius_ang,
    grid_shape, orth_matrix, cell_volume,
    hkl, taper_width, blur=0.0, max_atoms_per_batch=50_000,
    grid_ops=None, compile_core=True,
    use_checkpoint=False, solvent=None,
):
    """N configurations at once: (N, n_atoms, 3) -> (N, n_refl) complex.

    Every configuration shares one atom set -- same elements, occupancies,
    kernels and offsets -- and differs only in coordinates, which is the
    ensemble case this exists for (N members of a guided step, N frames of a
    trajectory). The shared setup is therefore done once rather than N times.

    Gradients flow to ALL N coordinate sets, so a single backward() on a loss
    built from the ensemble observables reaches every member.

    use_checkpoint (default False) recomputes each configuration's splat
    during backward instead of retaining its intermediates. This is usually
    the difference between "runs" and "out of memory": the splat holds
    (atoms x voxels) working buffers whose size dwarfs the density grid, and
    without checkpointing they are retained for all N simultaneously.
    Measured on a 3000-atom / 90^3 case, peak RSS above baseline:

        N     retained      checkpointed
        8      839 MB          361 MB
        16    1604 MB          371 MB

    i.e. flat in N rather than linear, for ~2.4x the time. Gradients are
    bit-identical either way (verified in tests/test_batch.py), so this is a
    pure time-for-memory trade with no numerical consequence.

    NOTE: this flag is all-or-nothing, and is tuned for MEMORY rather than for
    speed. It exists to make large N possible; when N does not fit it takes the
    memory-minimising extreme and pays the recompute penalty on every member,
    including those that would have fitted. Retaining as many members as the
    device can hold and checkpointing only the remainder would give the same
    peak at a fraction of the penalty -- see "Not yet built" in README.md.

    solvent: a solvent_torch.SolventModel, or None (default: no solvent code
    runs, bit-identical to before the argument existed). Applied PER
    CONFIGURATION -- each member gets its own mask from its own density.

    That is the only choice that reaches the diffuse observable: one mask
    shared across the ensemble has zero variance, so it changes <F> but
    contributes exactly nothing to <|F|^2> - |<F>|^2. Per-configuration masks
    fluctuate anti-correlated with the protein (where an atom moves out,
    solvent moves in), which is the excluded-volume contrast correction.

    Two things that follow, both in docs/solvent-design.md. The atom array must
    CORRESPOND across configurations -- same slots, same count -- which is a
    mechanical requirement, not a claim that the hydration cannot vary: an
    ensemble is exactly where it SHOULD vary, and that variation is signal in
    the diffuse term. Express it as coordinates and occupancies differing
    between configurations, not as atoms entering and leaving the array. And
    since diffuse is a variance, NUMERICAL noise in the mask (level-set
    discretization) biases it upward rather than averaging out, so the taper
    wants to be wider here than Bragg accuracy alone would suggest.

    Returns a (N, n_refl) complex tensor; pass it straight to
    mean_and_diffuse() for the ensemble observables.
    """
    if frac_coords_batch.dim() != 3 or frac_coords_batch.shape[-1] != 3:
        raise ValueError(
            "frac_coords_batch must be (n_configs, n_atoms, 3), got %s"
            % (tuple(frac_coords_batch.shape),)
        )
    n_atoms = frac_coords_batch.shape[1]
    if element_idx.shape[0] != n_atoms:
        raise ValueError(
            "element_idx has %d entries but each configuration has %d atoms; "
            "all configurations must share one atom set"
            % (element_idx.shape[0], n_atoms)
        )

    def one(frac_coords):
        return structure_factors_one_config(
            frac_coords, element_idx, occ,
            atom_A, atom_lam, elem_offsets, atom_radius_ang,
            grid_shape, orth_matrix, cell_volume,
            hkl, taper_width, blur=blur,
            max_atoms_per_batch=max_atoms_per_batch,
            grid_ops=grid_ops, compile_core=compile_core,
            solvent=solvent,
        )

    if not use_checkpoint:
        return torch.stack([one(fc) for fc in frac_coords_batch], dim=0)

    from torch.utils.checkpoint import checkpoint

    # preserve_rng_state=False: the splat draws no random numbers, so there is
    # no RNG state worth saving and restoring per configuration.
    return torch.stack(
        [checkpoint(one, fc, use_reentrant=False, preserve_rng_state=False)
         for fc in frac_coords_batch],
        dim=0,
    )


def mean_and_diffuse(F_configs: torch.Tensor):
    """
    F_configs: (n_configs, n_refl) complex tensor, one F(hkl) row per
    configuration (e.g. MD trajectory frame).

    Returns:
        mean_F:   (n_refl,) complex -- <F(hkl)>, the Bragg amplitude
        diffuse:  (n_refl,) real    -- <|F|^2> - |<F>|^2  (Eq. 3.65-style
                                       diffuse/variance term, matching
                                       xtraj.py's diffuse_data = avg_icalc
                                       - |avg_fcalc|^2)
    Both are differentiable with respect to every configuration's atomic
    coordinates that produced F_configs.
    """
    mean_F = F_configs.mean(dim=0)
    mean_I = (F_configs.abs() ** 2).mean(dim=0)
    diffuse = mean_I - mean_F.abs() ** 2
    return mean_F, diffuse
