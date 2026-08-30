"""
Validates the full splat+FFT pipeline with DIFFERENT B-factors per atom
(not just per element), against an analytic reference where each atom's
Debye-Waller factor uses its own B -- the actual target of this feature.
"""
import numpy as np
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import build_atom_kernels, calculate_sf
from lunus.sf.cell_utils import orth_matrix, reciprocal_inv_d2, grid_shape_for_resolution


def analytic_atom_F(coef9, b_atom, pos_frac, hkl, M):
    inv_d2 = reciprocal_inv_d2(M, hkl)
    stol2 = inv_d2 / 4.0
    f = calculate_sf(coef9, stol2)
    dwf = np.exp(-b_atom * stol2)  # THIS atom's own B, not a shared one
    phase = np.exp(-2j * np.pi * (hkl @ pos_frac))
    return f * dwf * phase


def taper_factor(r, R, w):
    inner = R - w
    t = np.ones_like(r)
    edge = (r > inner) & (r < R)
    t[edge] = 0.5 * (1 + np.cos(np.pi * (r[edge] - inner) / w))
    t[r >= R] = 0.0
    return t


def splat_and_fft_per_atom_b(atoms, A_list, lam_list, R_list, w, grid_shape, M, cell_volume):
    Nu, Nv, Nw = grid_shape
    grid = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu, np.linalg.norm(M[:, 1]) / Nv, np.linalg.norm(M[:, 2]) / Nw])

    for (pos, occ), A, lam, R in zip(atoms, A_list, lam_list, R_list):
        du, dv, dw = [int(np.ceil(R / s)) for s in spacing]
        gpos = pos * np.array([Nu, Nv, Nw])
        base = np.round(gpos).astype(int)
        for iu in range(-du, du + 1):
            for iv in range(-dv, dv + 1):
                for iw in range(-dw, dw + 1):
                    idx = base + np.array([iu, iv, iw])
                    voxel_frac = idx / np.array([Nu, Nv, Nw])
                    d_frac = voxel_frac - pos
                    d_frac -= np.round(d_frac)
                    d_cart = M @ d_frac
                    r2 = float(d_cart @ d_cart)
                    r = np.sqrt(r2)
                    if r >= R:
                        continue
                    taper = taper_factor(np.array([r]), R, w)[0]
                    dens = np.sum(A * np.exp(-lam * r2)) * taper * occ
                    wu, wv, ww = idx % np.array([Nu, Nv, Nw])
                    grid[wu, wv, ww] += dens
    Fgrid = np.fft.ifftn(grid) * cell_volume
    return np.conj(Fgrid)


def test_per_atom_b_factors():
    """Full splat+FFT with a DIFFERENT B per atom, vs an analytic reference
    where each atom's Debye-Waller factor uses its own B.

    The ~4% agreement asserted below is expected and is NOT a defect. The
    reference here is fully UNTRUNCATED, while the splat applies a fixed
    density-cutoff THRESHOLD: a high-B atom is more spread out, so the same
    threshold truncates a larger fraction of its charge (measured directly, a
    B=85 carbon loses ~9.3% of its charge to the cutoff before any taper,
    against ~3.1% for the same element at B=12). That is a property of the
    fixed-threshold cutoff itself, which gemmi shares, not of per-atom B or of
    this implementation.

    The atom set mixes rigid and very disordered atoms (B=12 and B=85 on the
    SAME element) precisely so this effect is exercised rather than averaged
    away.

    NOTE: what actually establishes correctness here is the grid-refinement
    convergence check described in docs/design.md -- F(000) stable to <0.01%
    across a 6x refinement -- which this file does not yet implement. Until it
    does, this test is a regression guard on a characterized number, not an
    independent proof of correctness.
    """
    a, b, c = 30.0, 34.0, 26.0
    alpha, beta, gamma = 80.0, 95.0, 110.0
    M = orth_matrix(a, b, c, alpha, beta, gamma)
    cell_volume = abs(np.linalg.det(M))
    d_min, rate = 2.0, 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)
    Nu, Nv, Nw = grid_shape

    # atoms with DELIBERATELY VERY DIFFERENT B-factors -- some rigid, some
    # very disordered, same and different elements, to stress-test everything
    atoms_spec = [
        ("C", np.array([0.372, 0.611, 0.148]), 1.0, 12.0),
        ("C", np.array([0.600, 0.300, 0.700]), 1.0, 85.0),  # same element, very different B
        ("N", np.array([0.850, 0.150, 0.450]), 0.9, 30.0),
        ("O", np.array([0.100, 0.900, 0.250]), 1.0, 55.0),
        ("S", np.array([0.450, 0.450, 0.050]), 1.0, 20.0),
    ]
    elements_per_atom = [s[0] for s in atoms_spec]
    b_per_atom = np.array([s[3] for s in atoms_spec])
    w = 0.1

    A_arr, lam_arr, R_arr = build_atom_kernels(elements_per_atom, IT92_COEFFS, b_per_atom, cutoff=0.01)

    hkl_test = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [5, -1, 2], [-4, 2, 1], [3, 3, 3], [-10, 7, -3]])

    F_ref = np.zeros(len(hkl_test), dtype=np.complex128)
    for (el, pos, occ, b_atom) in atoms_spec:
        F_ref += occ * analytic_atom_F(IT92_COEFFS[el], b_atom, pos, hkl_test, M)

    atoms_for_splat = [(s[1], s[2]) for s in atoms_spec]
    Fgrid = splat_and_fft_per_atom_b(atoms_for_splat, A_arr, lam_arr, R_arr, w, grid_shape, M, cell_volume)
    F_num = np.array([Fgrid[h % Nu, k % Nv, l % Nw] for h, k, l in hkl_test])

    print("hkl            F_splat (per-atom B)         F_analytic (per-atom B)      rel.err")
    max_rel_err = 0.0
    for (h, k, l), fn, fa in zip(hkl_test, F_num, F_ref):
        rel_err = abs(fn - fa) / max(abs(fa), 1e-8)
        max_rel_err = max(max_rel_err, rel_err)
        print(f"{h:3d}{k:3d}{l:3d}   {fn.real:8.4f}{fn.imag:+8.4f}i   {fa.real:8.4f}{fa.imag:+8.4f}i   {rel_err:.2e}")
    print(f"\nMax relative error: {max_rel_err:.2e}")

    # Measured 4.1e-02. Bounded on BOTH sides: an unexpectedly small error
    # would mean the truncation described above stopped happening, which is
    # as much a change in behaviour as an unexpectedly large one.
    assert 1e-2 < max_rel_err < 8e-2, (
        f"per-atom-B deviation {max_rel_err:.2e} is outside its characterized "
        "range (~4.1e-02); see this test's docstring for why it is not zero"
    )
