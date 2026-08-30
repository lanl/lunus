import numpy as np
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import precalculate_density_iso, calculate_sf, cutoff_radius
from lunus.sf.cell_utils import orth_matrix, reciprocal_inv_d2, grid_shape_for_resolution

def splat_and_fft_sphere(pos_frac, A, lam, radius, grid_shape, M, cell_volume, safety_margin):
    Nu, Nv, Nw = grid_shape
    grid = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu, np.linalg.norm(M[:, 1]) / Nv, np.linalg.norm(M[:, 2]) / Nw])
    du, dv, dw = [int(np.ceil(radius / s)) for s in spacing]

    # Precompute sphere-filtered offsets (approximate: ignores the atom's
    # sub-voxel remainder, padded with a safety margin to compensate)
    ou, ov, ow = np.meshgrid(np.arange(-du, du+1), np.arange(-dv, dv+1), np.arange(-dw, dw+1), indexing='ij')
    offsets = np.stack([ou, ov, ow], axis=-1).reshape(-1, 3)
    frac_offsets = offsets / np.array([Nu, Nv, Nw])
    cart_offsets = frac_offsets @ M.T
    dist = np.linalg.norm(cart_offsets, axis=1)
    keep = dist <= (radius + safety_margin)
    offsets = offsets[keep]
    n_cube = (2*du+1)*(2*dv+1)*(2*dw+1)
    print(f"  sphere-filtered offsets: {len(offsets)} / {n_cube} cube points ({100*len(offsets)/n_cube:.1f}%)")

    gpos = pos_frac * np.array([Nu, Nv, Nw])
    base = np.round(gpos).astype(int)
    for off in offsets:
        idx = base + off
        voxel_frac = idx / np.array([Nu, Nv, Nw])
        d_frac = voxel_frac - pos_frac
        d_frac -= np.round(d_frac)
        d_cart = M @ d_frac
        r2 = float(d_cart @ d_cart)
        dens = np.sum(A * np.exp(-lam * r2))
        wu, wv, ww = idx % np.array([Nu, Nv, Nw])
        grid[wu, wv, ww] += dens

    Fgrid = np.fft.ifftn(grid) * cell_volume
    return np.conj(Fgrid), len(offsets), n_cube

def analytic_single_atom_F(coef9, b_iso, pos_frac, hkl, M):
    inv_d2 = reciprocal_inv_d2(M, hkl)
    stol2 = inv_d2 / 4.0
    f = calculate_sf(coef9, stol2)
    dwf = np.exp(-b_iso * stol2)
    phase = np.exp(-2j * np.pi * (hkl @ pos_frac))
    return f * dwf * phase

def test_sphere_masked_splat_matches_analytic():
    """Restricting the local box to the cutoff SPHERE rather than its bounding
    cube must not change F(hkl): the dropped corner points lie beyond the
    cutoff radius, where the density is already negligible.

    Measured max relative error 7.4e-06, and the sphere keeps 4305/6859 =
    62.8% of the cube's points.
    """
    a, b, c = 30.0, 34.0, 26.0
    alpha, beta, gamma = 80.0, 95.0, 110.0
    M = orth_matrix(a, b, c, alpha, beta, gamma)
    cell_volume = abs(np.linalg.det(M))
    d_min, rate = 2.0, 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)
    Nu, Nv, Nw = grid_shape
    b_iso = 20.0 * d_min * d_min
    coef9 = IT92_COEFFS["C"]
    A, lam = precalculate_density_iso(coef9, b_iso)
    radius = cutoff_radius(A, lam, cutoff=1e-5)
    pos_frac = np.array([0.372, 0.611, 0.148])
    hkl_test = np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3],[2,1,0],[1,-3,2],[-4,2,1],[3,3,3],[5,-1,2],[0,5,-2]])

    # per-axis grid spacing, used to size a conservative safety margin (~1 voxel diagonal)
    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu, np.linalg.norm(M[:, 1]) / Nv, np.linalg.norm(M[:, 2]) / Nw])
    safety_margin = np.linalg.norm(spacing)

    print(f"\ncutoff radius: {radius:.3f} A, safety margin: {safety_margin:.3f} A")
    Fgrid, n_kept, n_cube = splat_and_fft_sphere(
        pos_frac, A, lam, radius, grid_shape, M, cell_volume, safety_margin)
    F_num = np.array([Fgrid[h % Nu, k % Nv, l % Nw] for h, k, l in hkl_test])
    F_ana = analytic_single_atom_F(coef9, b_iso, pos_frac, hkl_test, M)

    print("\nhkl            F_sphere_splat                F_analytic                   rel.err")
    for (h, k, l), fn, fa in zip(hkl_test, F_num, F_ana):
        print(f"{h:3d}{k:3d}{l:3d}   {fn.real:8.4f}{fn.imag:+8.4f}i   "
              f"{fa.real:8.4f}{fa.imag:+8.4f}i   {abs(fn - fa) / max(abs(fa), 1e-8):.2e}")

    np.testing.assert_allclose(
        F_num, F_ana,
        rtol=1e-4, atol=1e-4 * np.max(np.abs(F_ana)),
        err_msg="sphere-masked splat does not match the analytic reference",
    )

    # The mask has to actually be doing something -- if a change made it keep
    # the whole cube, the parity check above would still pass and silently
    # stop testing anything.
    assert n_kept < n_cube, "sphere mask kept the entire bounding cube"
    assert n_kept / n_cube < 0.8, (
        f"sphere kept {n_kept}/{n_cube} = {100 * n_kept / n_cube:.1f}% of the "
        "cube; expected ~63%"
    )
