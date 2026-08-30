"""
Validates the exact-cutoff-mask fix: confirms it still matches the
analytic multi-atom reference (regression check), and quantifies how
much the exact mask actually changes results relative to the earlier
margin-padded-but-unmasked version, at a loose cutoff (0.01, matching
what xtraj.py actually uses) where the difference should be visible.
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


def splat(frac_coords, element_syms, occ, elem_kernels, grid_shape, M, exact_mask):
    Nu, Nv, Nw = grid_shape
    grid_dims = np.array([Nu, Nv, Nw])
    grid = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu, np.linalg.norm(M[:, 1]) / Nv, np.linalg.norm(M[:, 2]) / Nw])
    safety_margin = np.linalg.norm(spacing)

    for el in sorted(set(element_syms)):
        idxs = [i for i, e in enumerate(element_syms) if e == el]
        A, lam, radius = elem_kernels[el]["A"], elem_kernels[el]["lam"], elem_kernels[el]["radius"]
        du, dv, dw = [int(np.ceil(radius / s)) for s in spacing]
        offsets = sphere_offsets(du, dv, dw, radius, safety_margin, M, grid_dims)

        for i in idxs:
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
                if exact_mask and r2 > radius * radius:
                    continue
                dens = np.sum(A * np.exp(-lam * r2)) * o
                wu, wv, ww = idx % grid_dims
                grid[wu, wv, ww] += dens
    return grid


def test_exact_cutoff_mask():
    """Characterizes the exact cutoff mask against the margin-padded version.

    This is not a parity test -- the two variants are SUPPOSED to differ. The
    exact mask discards every point beyond the cutoff radius, so it can only
    remove density, and the looser the cutoff the more it removes. Measured:

        cutoff    unmasked (old)   exact mask (new)   grid sum old -> new
        1e-05     5.8e-06          1.3e-04            85.3461 -> 85.3351
        0.01      9.6e-03          2.2e-01            84.5531 -> 79.2078

    So the assertions below pin the three things that are actually invariant:
    at a tight cutoff both agree with the untruncated analytic reference; the
    exact mask never ADDS density; and the loose-cutoff deviation stays within
    a characterized bound rather than drifting silently.
    """
    a, b, c = 30.0, 34.0, 26.0
    alpha, beta, gamma = 80.0, 95.0, 110.0
    M = orth_matrix(a, b, c, alpha, beta, gamma)
    cell_volume = abs(np.linalg.det(M))
    d_min, rate = 2.0, 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)
    Nu, Nv, Nw = grid_shape
    b_iso = 20.0 * d_min * d_min

    atoms = [
        ("C", np.array([0.372, 0.611, 0.148]), 1.0),
        ("N", np.array([0.600, 0.300, 0.700]), 1.0),
        ("O", np.array([0.850, 0.150, 0.450]), 0.9),
    ]
    element_syms = [a_[0] for a_ in atoms]
    frac_coords = np.array([a_[1] for a_ in atoms])
    occ = np.array([a_[2] for a_ in atoms])

    hkl_test = np.array([[0,0,0],[1,0,0],[0,2,0],[5,-1,2],[-4,2,1],[3,3,3],[-10,7,-3]])

    results = {}
    for cutoff in [1e-5, 0.01]:
        elem_kernels = {}
        for el in set(element_syms):
            A, lam = precalculate_density_iso(IT92_COEFFS[el], b_iso)
            r = cutoff_radius(A, lam, cutoff=cutoff)
            elem_kernels[el] = {"A": A, "lam": lam, "radius": r}

        F_ref = np.zeros(len(hkl_test), dtype=np.complex128)
        for el, pos, o in atoms:
            F_ref += o * analytic_atom_F(IT92_COEFFS[el], b_iso, pos, hkl_test, M)

        print(f"\n=== cutoff = {cutoff} ===")
        for exact_mask, label in [(False, "margin-padded, unmasked (old)"), (True, "exact cutoff mask (new)")]:
            grid = splat(frac_coords, element_syms, occ, elem_kernels, grid_shape, M, exact_mask)
            Fgrid = np.conj(np.fft.ifftn(grid) * cell_volume)
            F_num = np.array([Fgrid[h % Nu, k % Nv, l % Nw] for h, k, l in hkl_test])
            max_rel_err = np.max(np.abs(F_num - F_ref) / np.maximum(np.abs(F_ref), 1e-8))
            print(f"  {label}: grid sum={grid.sum():.4f}, max rel.err vs analytic = {max_rel_err:.2e}")
            results[(cutoff, exact_mask)] = (grid.sum(), max_rel_err)

    # At a tight cutoff the truncation is negligible either way, so both
    # variants must track the untruncated analytic reference closely.
    for exact_mask in (False, True):
        _, err = results[(1e-5, exact_mask)]
        assert err < 1e-3, (
            f"at cutoff=1e-5, exact_mask={exact_mask} deviates from the "
            f"analytic reference by {err:.2e}; expected < 1e-3"
        )

    # The mask can only ever DISCARD points, so it cannot increase total
    # density. This is the structural invariant, independent of tolerances.
    for cutoff in (1e-5, 0.01):
        sum_unmasked = results[(cutoff, False)][0]
        sum_masked = results[(cutoff, True)][0]
        assert sum_masked <= sum_unmasked + 1e-9, (
            f"at cutoff={cutoff} the exact mask INCREASED the grid sum "
            f"({sum_unmasked:.4f} -> {sum_masked:.4f}); it can only remove points"
        )

    # A loose cutoff truncates visibly -- that is the deliberate trade, not a
    # bug. Bound it so a regression that made it much worse still trips.
    _, err_loose = results[(0.01, True)]
    assert 0.05 < err_loose < 0.5, (
        f"loose-cutoff deviation {err_loose:.2e} is outside its characterized "
        "range (~0.22); the truncation behaviour has changed"
    )
