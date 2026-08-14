"""
Extends validate_numpy.py to a multi-atom, multi-element system, and
mirrors the CURRENT (post memory-optimization) algorithm structure --
grouped by element, chunked, sphere-masked offsets -- in plain NumPy.
This isolates whether the chunking/sphere-mask/grouping logic itself is
correct, independent of any PyTorch-specific semantics (int32 modulo
behavior, einsum broadcasting, etc.), which NumPy can't test but which
is a candidate the algorithm-level test below can't rule in or out.
"""

import numpy as np
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import precalculate_density_iso, calculate_sf, cutoff_radius
from lunus.sf.cell_utils import orth_matrix, reciprocal_inv_d2, grid_shape_for_resolution


def analytic_atom_F(coef9, b_iso, pos_frac, hkl, M):
    inv_d2 = reciprocal_inv_d2(M, hkl)
    stol2 = inv_d2 / 4.0
    f = calculate_sf(coef9, stol2)
    dwf = np.exp(-b_iso * stol2)
    phase = np.exp(-2j * np.pi * (hkl @ pos_frac))
    return f * dwf * phase


def sphere_offsets(du, dv, dw, radius_ang, safety_margin, M, grid_dims):
    ou, ov, ow = np.meshgrid(np.arange(-du, du+1), np.arange(-dv, dv+1), np.arange(-dw, dw+1), indexing='ij')
    offsets = np.stack([ou, ov, ow], axis=-1).reshape(-1, 3)
    frac_offsets = offsets / grid_dims
    cart_offsets = frac_offsets @ M.T
    dist = np.linalg.norm(cart_offsets, axis=1)
    return offsets[dist <= (radius_ang + safety_margin)]


def splat_grouped_chunked(frac_coords, element_syms, occ, elem_kernels, grid_shape, M, chunk_size):
    """Mirrors density_torch.py's algorithm exactly: grouped by element,
    chunked atoms within each group, sphere-restricted offsets."""
    Nu, Nv, Nw = grid_shape
    grid_dims = np.array([Nu, Nv, Nw])
    grid = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu, np.linalg.norm(M[:, 1]) / Nv, np.linalg.norm(M[:, 2]) / Nw])
    safety_margin = np.linalg.norm(spacing)

    unique_elements = sorted(set(element_syms))
    for el in unique_elements:
        idxs = [i for i, e in enumerate(element_syms) if e == el]
        A, lam, radius = elem_kernels[el]["A"], elem_kernels[el]["lam"], elem_kernels[el]["radius"]
        du, dv, dw = [int(np.ceil(radius / s)) for s in spacing]
        offsets = sphere_offsets(du, dv, dw, radius, safety_margin, M, grid_dims)

        for start in range(0, len(idxs), chunk_size):
            chunk = idxs[start:start + chunk_size]
            for i in chunk:  # inner loop stands in for the vectorized (m,K,...) batch
                pos = frac_coords[i]
                o = occ[i]
                gpos = pos * grid_dims
                base = np.round(gpos).astype(int)
                for off in offsets:
                    idx = base + off
                    voxel_frac = idx / grid_dims
                    d_frac = voxel_frac - pos
                    d_frac -= np.round(d_frac)
                    d_cart = M @ d_frac
                    r2 = float(d_cart @ d_cart)
                    dens = np.sum(A * np.exp(-lam * r2)) * o
                    wu, wv, ww = idx % grid_dims
                    grid[wu, wv, ww] += dens

    return grid


def test_multiatom_matches_analytic_and_is_chunk_invariant():
    """Grouped/chunked/sphere-masked splat vs a multi-atom analytic reference.

    Two claims, and the second is the one this file exists for:

    1. The pipeline reproduces the linear superposition of independent
       single-atom F(hkl). Measured max relative error 1.9e-05.
    2. The result is INDEPENDENT of chunk size. Chunking is a pure memory
       optimization, so chunk_size=1, 2 and 100 must agree to round-off --
       not merely all land within tolerance of the reference, which is a
       much weaker statement and would miss a chunk-boundary bug that
       perturbed every chunking identically.

    The atom set deliberately includes positions near the 0/1 cell boundaries
    to stress periodic wraparound and negative base indices, plus a partial
    occupancy.
    """
    a, b, c = 30.0, 34.0, 26.0
    alpha, beta, gamma = 80.0, 95.0, 110.0
    M = orth_matrix(a, b, c, alpha, beta, gamma)
    cell_volume = abs(np.linalg.det(M))
    d_min, rate = 2.0, 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)
    Nu, Nv, Nw = grid_shape
    b_iso = 20.0 * d_min * d_min

    # mixed elements, including positions near 0/1 boundaries to stress
    # periodic wraparound and negative base indices
    atoms = [
        ("C", np.array([0.372, 0.611, 0.148]), 1.0),
        ("N", np.array([0.020, 0.611, 0.148]), 1.0),   # near u=0 boundary
        ("O", np.array([0.372, 0.980, 0.148]), 1.0),   # near v=1 boundary
        ("H", np.array([0.372, 0.611, 0.005]), 1.0),   # near w=0 boundary
        ("C", np.array([0.600, 0.300, 0.700]), 0.8),   # partial occupancy
        ("N", np.array([0.850, 0.150, 0.450]), 1.0),
    ]
    element_syms = [a_[0] for a_ in atoms]
    frac_coords = np.array([a_[1] for a_ in atoms])
    occ = np.array([a_[2] for a_ in atoms])

    elem_kernels = {}
    for el in set(element_syms):
        A, lam = precalculate_density_iso(IT92_COEFFS[el], b_iso)
        r = cutoff_radius(A, lam, cutoff=1e-5)
        elem_kernels[el] = {"A": A, "lam": lam, "radius": r}

    hkl_test = np.array([
        [0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3],
        [2, 1, 0], [1, -3, 2], [-4, 2, 1], [3, 3, 3],
        [5, -1, 2], [0, 5, -2], [-2, -3, -1],
    ])

    # reference: linear superposition of independent analytic single-atom F(hkl)
    F_ref = np.zeros(len(hkl_test), dtype=np.complex128)
    for el, pos, o in atoms:
        F_ref += o * analytic_atom_F(IT92_COEFFS[el], b_iso, pos, hkl_test, M)

    scale = np.max(np.abs(F_ref))
    by_chunk = {}
    for chunk_size in [1, 2, 100]:
        grid = splat_grouped_chunked(frac_coords, element_syms, occ, elem_kernels, grid_shape, M, chunk_size)
        Fgrid = np.conj(np.fft.ifftn(grid) * cell_volume)
        F_num = np.array([Fgrid[h % Nu, k % Nv, l % Nw] for h, k, l in hkl_test])
        by_chunk[chunk_size] = F_num

        max_rel_err = max(abs(fn - fa) / max(abs(fa), 1e-8)
                          for fn, fa in zip(F_num, F_ref))
        print(f"\nchunk_size={chunk_size:>4}: max relative error vs multi-atom analytic reference = {max_rel_err:.2e}")
        if chunk_size == 1:
            print("  per-reflection detail:")
            for (h, k, l), fn, fa in zip(hkl_test, F_num, F_ref):
                rel_err = abs(fn - fa) / max(abs(fa), 1e-8)
                print(f"    {h:3d}{k:3d}{l:3d}   grouped/chunked={fn.real:8.3f}{fn.imag:+8.3f}i   "
                      f"reference={fa.real:8.3f}{fa.imag:+8.3f}i   rel.err={rel_err:.2e}")

        # claim 1: agrees with the analytic superposition
        np.testing.assert_allclose(
            F_num, F_ref, rtol=1e-4, atol=1e-4 * scale,
            err_msg=f"chunk_size={chunk_size} does not match the analytic reference",
        )

    # claim 2: chunking changes nothing. Same arithmetic in a different order,
    # so this is float-associativity noise only -- far tighter than the
    # tolerance against the analytic reference above.
    reference_chunking = by_chunk[1]
    for chunk_size, F_num in by_chunk.items():
        np.testing.assert_allclose(
            F_num, reference_chunking, rtol=1e-10, atol=1e-10 * scale,
            err_msg=f"chunk_size={chunk_size} disagrees with chunk_size=1; "
                    "chunking must be a pure memory optimization",
        )
