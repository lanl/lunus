"""
The anisotropic-ADP density kernel: the closed form, and the conventions.

Named for ANISOTROPIC DISPLACEMENT PARAMETERS. Not to be confused with
aniso.py / test_aniso.py, which are the isotropic-background removal that
produces the anisotropic COMPONENT of a diffuse map -- an unrelated operation
that happens to share the word.

Stage 1 of the anisotropic work: numpy only, no splat and no torch, so a
mistake in the algebra surfaces here rather than inside a fused kernel.

The checks:
  1. Setting U = u*I reproduces the isotropic kernel exactly. The anisotropic
     path is a strict generalization, so anything else is an algebra error.
  2. integral rho d^3r = f(0) = Z, the same normalization the isotropic kernel
     is held to.
  3. Splat + FFT of one anisotropic atom against the closed-form
     f(s) exp(-2 pi^2 s.U.s) exp(-2 pi i h.x). This is the one that pins the
     CONVENTIONS -- a transposed U, eigenvectors as rows instead of columns, or
     U taken in the fractional rather than Cartesian frame all reproduce checks
     1 and 2 and fail here.
  4. A non-positive-definite ADP raises instead of returning nonsense.
  5. The cutoff sphere contains the whole ellipsoid.
"""

import numpy as np
import pytest

from lunus.sf.cell_utils import orth_matrix, reciprocal_inv_d2
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import (
    calculate_sf, cutoff_radius_aniso, density_at, density_at_aniso,
    precalculate_density_aniso, precalculate_density_iso,
)

CELL = (18.0, 20.0, 22.0, 90.0, 90.0, 90.0)
GRID = (40, 44, 48)
M_NP = orth_matrix(*CELL)
VOLUME = float(np.linalg.det(M_NP))
COEF = IT92_COEFFS["C"]


def u_aniso(u1, u2, u3, axis=(1.0, 1.0, 0.3), angle=0.7):
    """A positive-definite ADP with principal values (u1,u2,u3) in a frame
    rotated away from the Cartesian axes, so that a convention error in V has
    somewhere to show up."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = (np.eye(3) + np.sin(angle) * K
         + (1 - np.cos(angle)) * (K @ K))              # Rodrigues
    return R @ np.diag([u1, u2, u3]) @ R.T


# --------------------------------------------------------------------------
# 1. the isotropic limit

@pytest.mark.parametrize("u_iso, blur", [(0.25, 0.0), (0.25, 5.0), (1.2, 2.0)])
def test_isotropic_adp_reproduces_the_isotropic_kernel(u_iso, blur):
    A_a, w_a, V = precalculate_density_aniso(COEF, u_iso * np.eye(3), blur=blur)
    A_i, lam_i = precalculate_density_iso(COEF, 8 * np.pi ** 2 * u_iso + blur)

    assert np.allclose(A_a, A_i, rtol=0, atol=1e-12)
    # every principal axis carries the same coefficient, equal to lam
    assert np.allclose(w_a, lam_i[:, None], rtol=0, atol=1e-12)

    r = np.random.default_rng(0).normal(0.0, 1.2, (200, 3))
    assert np.allclose(density_at_aniso(A_a, w_a, V, r),
                       density_at(A_i, lam_i, (r * r).sum(-1)),
                       rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# 2. normalization

@pytest.mark.parametrize("u", [(0.2, 0.2, 0.2), (0.1, 0.3, 0.8), (0.05, 0.6, 1.5)])
def test_total_charge_is_f_zero_whatever_the_adp(u):
    """integral of an ellipsoidal Gaussian is A pi^{3/2}/sqrt(prod w), and the
    sum has to come back to f(0) = Z regardless of how the ADP redistributes
    it. Charge must not depend on the ADP any more than it depends on B."""
    A, w, _ = precalculate_density_aniso(COEF, u_aniso(*u), blur=1.0)
    total = float((A * np.pi ** 1.5 / np.sqrt(np.prod(w, axis=1))).sum())
    assert total == pytest.approx(float(calculate_sf(COEF, np.zeros(1))[0]),
                                  rel=1e-12)


# --------------------------------------------------------------------------
# 3. the convention test

def splat_and_fft_aniso(pos_frac, A, w, V, radius, grid_shape, M, cell_volume):
    """One anisotropic atom, occupancy 1, splatted by brute force. Same
    conventions as tests/test_numpy_reference.py's isotropic version:
    conjugated, volume-scaled ifft."""
    Nu, Nv, Nw = grid_shape
    grid = np.zeros(grid_shape, dtype=np.float64)
    spacing = [np.linalg.norm(M[:, i]) / n for i, n in enumerate(grid_shape)]
    du, dv, dw = [int(np.ceil(radius / s)) for s in spacing]

    base = np.round(pos_frac * np.array(grid_shape)).astype(int)
    for iu in range(-du, du + 1):
        for iv in range(-dv, dv + 1):
            for iw in range(-dw, dw + 1):
                idx = base + np.array([iu, iv, iw])
                d_frac = idx / np.array(grid_shape) - pos_frac
                d_frac -= np.round(d_frac)                 # minimum image
                d_cart = M @ d_frac
                wu, wv, ww = idx % np.array(grid_shape)
                grid[wu, wv, ww] += float(density_at_aniso(A, w, V, d_cart))

    return np.conj(np.fft.ifftn(grid) * cell_volume)


def analytic_single_atom_F_aniso(coef9, u_cart, pos_frac, hkl, M):
    """f(s) * exp(-2 pi^2 s.U.s) * exp(-2 pi i h.x), with s the CARTESIAN
    reciprocal vector in A^-1 -- the same s whose square reciprocal_inv_d2
    returns."""
    s_cart = hkl @ np.linalg.inv(M)                    # (n,3), A^-1
    stol2 = reciprocal_inv_d2(M, hkl) / 4.0
    dwf = np.exp(-2.0 * np.pi ** 2 * ((s_cart @ u_cart) * s_cart).sum(-1))
    phase = np.exp(-2j * np.pi * (hkl @ pos_frac))
    return calculate_sf(coef9, stol2) * dwf * phase


def test_splat_and_fft_matches_the_closed_form():
    """Two independently derived expressions for the same quantity: a
    real-space ellipsoidal Gaussian summed onto a grid and transformed, against
    the reciprocal-space Debye-Waller form. Agreement is only possible if U,
    its eigenvectors and the Cartesian frame are all used consistently."""
    U = u_aniso(0.12, 0.35, 0.75)
    blur = 3.0
    A, w, V = precalculate_density_aniso(COEF, U, blur=blur)
    radius = cutoff_radius_aniso(A, w, V, cutoff=1e-6)
    pos_frac = np.array([0.2137, 0.3391, 0.4523])

    Fgrid = splat_and_fft_aniso(pos_frac, A, w, V, radius, GRID, M_NP, VOLUME)

    hkl = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 0],
                    [1, 2, 3], [3, -2, 1], [4, 4, 4], [-5, 2, 3]])
    got = np.array([Fgrid[h % GRID[0], k % GRID[1], l % GRID[2]]
                    for h, k, l in hkl])
    # The splat carries the blur; the closed form does not, so undo it the way
    # compute_fcalc does.
    got = got * np.exp(blur * 0.25 * reciprocal_inv_d2(M_NP, hkl))
    want = analytic_single_atom_F_aniso(COEF, U, pos_frac, hkl, M_NP)

    # Measured max relative error 4.2e-09, and FLAT across a 3x grid
    # refinement (0.45 -> 0.15 A voxels: 4.2e-9, 1.6e-8, 2.6e-8, 3.9e-8), so
    # what is left is the 1e-6 density truncation and float64 accumulation over
    # ~10^5 voxels rather than discretization. The bound is set just clear of
    # that; a loose one here would pass on a wrong convention.
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-6


def test_an_isotropic_u_would_not_have_caught_a_transposed_frame():
    """Guards the guard: the fixture's ADP must actually be anisotropic and
    rotated, or check 3 passes on a kernel that ignores V entirely."""
    U = u_aniso(0.12, 0.35, 0.75)
    assert np.linalg.eigvalsh(U).ptp() > 0.5           # genuinely anisotropic
    assert np.abs(U - np.diag(np.diag(U))).max() > 0.05   # and not axis-aligned


# --------------------------------------------------------------------------
# 4-5. the failure modes

def test_non_positive_definite_adp_raises():
    """Deposited models contain these. The constant term has beta = blur, so
    with blur = 0 a non-positive eigenvalue is fatal, not merely unphysical --
    and the failure without this check is a negative or nan amplitude rather
    than an exception."""
    npd = u_aniso(0.4, 0.2, -0.05)
    with pytest.raises(ValueError, match="positive"):
        precalculate_density_aniso(COEF, npd, blur=0.0)

    # Enough blur rescues it: q_5k = blur/4 + 2 pi^2 u_k.
    ok = precalculate_density_aniso(COEF, npd, blur=20.0)
    assert np.all(np.isfinite(ok[0]))


def test_cutoff_sphere_contains_the_whole_ellipsoid():
    U = u_aniso(0.10, 0.40, 1.10)
    A, w, V = precalculate_density_aniso(COEF, U, blur=2.0)
    cutoff = 1e-5
    radius = cutoff_radius_aniso(A, w, V, cutoff=cutoff)

    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(2000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    outside = density_at_aniso(A, w, V, dirs * (radius * 1.001))
    assert np.abs(outside).max() < cutoff

    # And not absurdly conservative: some direction reaches the cutoff near it.
    inside = density_at_aniso(A, w, V, dirs * (radius * 0.85))
    assert np.abs(inside).max() > cutoff


# --------------------------------------------------------------------------
# 6. the batched path, which is what a real structure uses

def _lam_from_l6(l6):
    """(...,6) -> (...,3,3), the inverse of the packing L6 uses."""
    xx, yy, zz, xy, xz, yz = [l6[..., i] for i in range(6)]
    return np.stack([np.stack([xx, xy, xz], -1),
                     np.stack([xy, yy, yz], -1),
                     np.stack([xz, yz, zz], -1)], axis=-2)


def _batch(n=6, blur=2.0, seed=1):
    from lunus.sf.kernel import precalculate_density_aniso_batch
    rng = np.random.default_rng(seed)
    coef = np.array([IT92_COEFFS[e] for e in
                     ["C", "N", "O", "C", "S", "O"][:n]])
    U = np.stack([u_aniso(*(0.08 + 0.9 * rng.random(3)),
                          angle=float(rng.random()) * 2.0)
                  for _ in range(n)])
    return coef, U, precalculate_density_aniso_batch(coef, U, blur=blur), blur


def test_batched_matches_the_single_atom_path():
    coef, U, (A, L6, w, V), blur = _batch()
    for i in range(len(coef)):
        A1, w1, V1 = precalculate_density_aniso(coef[i], U[i], blur=blur)
        assert np.allclose(A[i], A1, rtol=0, atol=1e-14)
        assert np.allclose(w[i], w1, rtol=0, atol=1e-14)


def test_l6_reproduces_the_eigenbasis_quadratic_form():
    """The equivalence the splat depends on. The hot loop expands
    d.Lam_i.d from the six unique components; the derivation is stated in the
    eigenbasis. If these ever disagree the splat computes a different -- and
    entirely plausible-looking -- density."""
    coef, U, (A, L6, w, V), blur = _batch()
    r = np.random.default_rng(4).normal(size=(20, 3))
    Lam = _lam_from_l6(L6)                                   # (n,5,3,3)

    from_l6 = np.einsum("pm,njmk,pk->npj", r, Lam, r)        # (n,20,5)
    e = np.einsum("pm,nmk->npk", r, V)                       # into each frame
    from_eig = np.einsum("npk,njk->npj", e * e, w)
    assert np.allclose(from_l6, from_eig, rtol=1e-12, atol=0.0)


def test_batched_cutoff_radius_matches_the_per_atom_scan():
    from lunus.sf.kernel import cutoff_radius_aniso_batch
    coef, U, (A, L6, w, V), blur = _batch()
    batched = cutoff_radius_aniso_batch(A, w, cutoff=1e-5)
    for i in range(len(coef)):
        one = cutoff_radius_aniso(A[i], w[i], V[i], cutoff=1e-5)
        # The batch scanner bisects where the single-atom one scans a fixed
        # grid, so they agree to the grid step rather than exactly.
        assert abs(batched[i] - one) < 0.05


def test_batched_npd_says_how_many_and_which():
    """One bad ADP in a large model should not produce a bare LinAlgError or,
    worse, a silent nan -- it should say how many atoms are affected and point
    at the first."""
    from lunus.sf.kernel import precalculate_density_aniso_batch
    coef, U, _, _ = _batch()
    U = U.copy()
    U[3] = u_aniso(0.4, 0.2, -0.05)
    with pytest.raises(ValueError, match="1 of 6 atoms fail, first at index 3"):
        precalculate_density_aniso_batch(coef, U, blur=0.0)


# --------------------------------------------------------------------------
# 7. the splat: dispatch, and what it must not cost an isotropic model

def _splat_system(n_atoms=24, grid=(28, 30, 32), seed=2):
    import torch
    from lunus.sf.kernel_torch import (
        build_atom_kernels_aniso_torch, build_atom_kernels_torch)

    rng = np.random.default_rng(seed)
    elements = ["C", "N", "O"] * (n_atoms // 3)
    frac = torch.tensor(rng.random((n_atoms, 3)), dtype=torch.float64)
    occ = torch.tensor(rng.uniform(0.4, 1.0, n_atoms), dtype=torch.float64)
    u_iso = 0.25
    U = np.stack([u_iso * np.eye(3)] * n_atoms)
    b = np.full(n_atoms, 8 * np.pi ** 2 * u_iso)
    common = dict(grid_shape=grid, orth_matrix_np=M_NP, cutoff=1e-4,
                  device="cpu", dtype=torch.float64)
    present = sorted(set(elements))
    iso = build_atom_kernels_torch(elements, present, IT92_COEFFS, b, 0.0,
                                   **common)
    ani = build_atom_kernels_aniso_torch(elements, present, IT92_COEFFS, U,
                                         0.0, **common)
    idx = torch.tensor([present.index(e) for e in elements])
    return dict(frac=frac, occ=occ, element_idx=idx, grid=grid, iso=iso,
                ani=ani, U=U, elements=elements, present=present)


def test_isotropic_model_is_bit_identical_to_before_the_aniso_path():
    """The promise the argument was added under. aniso_mask=None must not
    merely give the same answer -- it must run none of the new code, including
    the compile of a core it will never call."""
    import torch
    from lunus.sf.density_torch import splat_density

    s = _splat_system()
    A, lam, offs, rad, tw, _ = s["iso"]
    args = (s["frac"], s["element_idx"], s["occ"], A, lam, offs, rad,
            s["grid"], torch.tensor(M_NP), tw)
    before = splat_density(*args, compile_core=False)
    after = splat_density(*args, compile_core=False, atom_L6=None,
                          aniso_mask=None)
    assert torch.equal(before, after)


def test_an_isotropic_adp_through_the_aniso_path_matches_the_isotropic_one():
    """End to end on the grid, not just in the core: U = u*I splatted through
    the anisotropic dispatch must reproduce the isotropic splat."""
    import torch
    from lunus.sf.density_torch import splat_density

    s = _splat_system()
    A_i, lam, offs_i, rad_i, tw, _ = s["iso"]
    A_a, L6, offs_a, rad_a, _, _ = s["ani"]
    n = s["frac"].shape[0]

    iso = splat_density(s["frac"], s["element_idx"], s["occ"], A_i, lam,
                        offs_i, rad_i, s["grid"], torch.tensor(M_NP), tw,
                        compile_core=False)
    ani = splat_density(s["frac"], s["element_idx"], s["occ"], A_a, None,
                        offs_a, rad_a, s["grid"], torch.tensor(M_NP), tw,
                        compile_core=False, atom_L6=L6,
                        aniso_mask=torch.ones(n, dtype=torch.bool))
    assert torch.allclose(iso, ani, rtol=1e-10, atol=1e-12)


def test_mixed_model_is_the_sum_of_its_two_halves():
    """A real structure is mixed -- 7FPV is 1,814 anisotropic of 3,341. The
    dispatch splits by kind WITHIN each element group, so this checks the two
    halves land in one grid correctly rather than one overwriting the other or
    an atom being splatted twice."""
    import torch
    from lunus.sf.density_torch import splat_density

    s = _splat_system()
    A_i, lam, offs, rad, tw, _ = s["iso"]
    A_a, L6, _, rad_a, _, _ = s["ani"]
    n = s["frac"].shape[0]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[::2] = True                       # every other atom anisotropic

    mixed = splat_density(s["frac"], s["element_idx"], s["occ"], A_a, lam,
                          offs, rad, s["grid"], torch.tensor(M_NP), tw,
                          compile_core=False, atom_L6=L6, aniso_mask=mask)

    # Same thing built as two splats, with the other half's occupancies zeroed.
    zero = torch.zeros_like(s["occ"])
    half_a = splat_density(s["frac"], s["element_idx"],
                           torch.where(mask, s["occ"], zero), A_a, lam, offs,
                           rad, s["grid"], torch.tensor(M_NP), tw,
                           compile_core=False, atom_L6=L6, aniso_mask=mask)
    half_i = splat_density(s["frac"], s["element_idx"],
                           torch.where(mask, zero, s["occ"]), A_a, lam, offs,
                           rad, s["grid"], torch.tensor(M_NP), tw,
                           compile_core=False, atom_L6=L6, aniso_mask=mask)
    assert torch.allclose(mixed, half_a + half_i, rtol=1e-10, atol=1e-12)


def test_dispatch_rejects_an_incoherent_call():
    import torch
    from lunus.sf.density_torch import splat_density

    s = _splat_system()
    A, lam, offs, rad, tw, _ = s["iso"]
    n = s["frac"].shape[0]
    args = (s["frac"], s["element_idx"], s["occ"], A, lam, offs, rad,
            s["grid"], torch.tensor(M_NP), tw)

    with pytest.raises(ValueError, match="without atom_L6"):
        splat_density(*args, compile_core=False,
                      aniso_mask=torch.ones(n, dtype=torch.bool))

    # atom_lam=None is only allowed when nothing is isotropic.
    A_a, L6, _, _, _, _ = s["ani"]
    partial = torch.zeros(n, dtype=torch.bool)
    partial[0] = True
    with pytest.raises(ValueError, match="atom_lam is required"):
        splat_density(s["frac"], s["element_idx"], s["occ"], A_a, None, offs,
                      rad, s["grid"], torch.tensor(M_NP), tw,
                      compile_core=False, atom_L6=L6, aniso_mask=partial)
