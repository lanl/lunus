"""
Validates the splat -> FFT -> extract pipeline against an independent,
closed-form single-atom structure factor, using plain NumPy (executable
in this sandbox, unlike PyTorch). This is the reference the PyTorch
version in density_torch.py / structure_factor_torch.py is a direct
translation of.

Two checks:
  1. VALUE check: F(hkl) from splat+FFT vs. the direct analytic
     single-atom formula F(hkl) = f(s) * exp(-B*s^2) * exp(-2 pi i h.x)
  2. GRADIENT check: finite-difference dF/dx from the splat+FFT pipeline
     vs. finite-difference dF/dx from the closed-form analytic formula.
     These are two independently-derived expressions for the same
     physical quantity -- agreement here is strong evidence that the
     splat+FFT pipeline (and therefore its PyTorch autograd gradient,
     since autograd differentiates the identical arithmetic) is correct.
"""

import numpy as np
import pytest

from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import precalculate_density_iso, calculate_sf, cutoff_radius
from lunus.sf.cell_utils import orth_matrix, reciprocal_inv_d2, grid_shape_for_resolution

# Measured on this system: 5.4e-06 (values), 4.0e-05 (gradients, over the
# reflections where the reference derivative is not identically zero).
# Thresholds are set an order of magnitude above what is observed, so they
# catch a real regression without tripping on float noise.
VALUE_RTOL = 1e-4
GRAD_RTOL = 1e-3


def splat_and_fft(pos_frac, A, lam, radius, grid_shape, M, cell_volume):
    """One atom, occupancy 1. Returns the full (Nu,Nv,Nw) F(hkl) grid
    (conjugated, volume-scaled ifft), matching xtraj.py's gemmi-engine
    convention exactly."""
    Nu, Nv, Nw = grid_shape
    grid = np.zeros((Nu, Nv, Nw), dtype=np.float64)

    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu,
                         np.linalg.norm(M[:, 1]) / Nv,
                         np.linalg.norm(M[:, 2]) / Nw])
    du, dv, dw = [int(np.ceil(radius / s)) for s in spacing]

    gpos = pos_frac * np.array([Nu, Nv, Nw])
    base = np.round(gpos).astype(int)

    for iu in range(-du, du + 1):
        for iv in range(-dv, dv + 1):
            for iw in range(-dw, dw + 1):
                idx = base + np.array([iu, iv, iw])
                voxel_frac = idx / np.array([Nu, Nv, Nw])
                d_frac = voxel_frac - pos_frac
                d_frac -= np.round(d_frac)  # minimum-image wrap
                d_cart = M @ d_frac
                r2 = float(d_cart @ d_cart)
                dens = np.sum(A * np.exp(-lam * r2))
                wu, wv, ww = idx % np.array([Nu, Nv, Nw])
                grid[wu, wv, ww] += dens

    Fgrid = np.fft.ifftn(grid) * cell_volume
    return np.conj(Fgrid)


def analytic_single_atom_F(coef9, b_iso, pos_frac, hkl, M):
    inv_d2 = reciprocal_inv_d2(M, hkl)
    stol2 = inv_d2 / 4.0
    f = calculate_sf(coef9, stol2)
    dwf = np.exp(-b_iso * stol2)
    phase = np.exp(-2j * np.pi * (hkl @ pos_frac))
    return f * dwf * phase


@pytest.fixture(scope="module")
def system():
    """Generic triclinic cell, one carbon atom. Shared by both checks."""
    a, b, c = 30.0, 34.0, 26.0
    alpha, beta, gamma = 80.0, 95.0, 110.0
    M = orth_matrix(a, b, c, alpha, beta, gamma)

    d_min = 2.0
    rate = 1.5
    grid_shape = grid_shape_for_resolution(a, b, c, d_min, rate)

    b_iso = 20.0 * d_min * d_min
    coef9 = IT92_COEFFS["C"]
    A, lam = precalculate_density_iso(coef9, b_iso)

    return dict(
        M=M,
        cell_volume=abs(np.linalg.det(M)),
        grid_shape=grid_shape,
        b_iso=b_iso,
        coef9=coef9,
        A=A,
        lam=lam,
        radius=cutoff_radius(A, lam, cutoff=1e-5),
        pos_frac=np.array([0.372, 0.611, 0.148]),
        hkl=np.array([
            [0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3],
            [2, 1, 0], [1, -3, 2], [-4, 2, 1], [3, 3, 3],
            [5, -1, 2], [0, 5, -2],
        ]),
    )


def test_values_match_analytic(system):
    """splat+FFT F(hkl) vs the closed-form single-atom formula."""
    s = system
    Nu, Nv, Nw = s["grid_shape"]

    Fgrid = splat_and_fft(s["pos_frac"], s["A"], s["lam"], s["radius"],
                          s["grid_shape"], s["M"], s["cell_volume"])
    F_num = np.array([Fgrid[h % Nu, k % Nv, l % Nw] for h, k, l in s["hkl"]])
    F_ana = analytic_single_atom_F(s["coef9"], s["b_iso"], s["pos_frac"], s["hkl"], s["M"])

    print("\nhkl            F_splat_fft                  F_analytic                   rel.err")
    for (h, k, l), fn, fa in zip(s["hkl"], F_num, F_ana):
        print(f"{h:3d}{k:3d}{l:3d}   {fn.real:8.4f}{fn.imag:+8.4f}i   "
              f"{fa.real:8.4f}{fa.imag:+8.4f}i   {abs(fn - fa) / max(abs(fa), 1e-8):.2e}")

    # atol is scaled off the largest reference magnitude so that reflections
    # with near-zero |F| are judged on absolute, not relative, agreement.
    np.testing.assert_allclose(
        F_num, F_ana,
        rtol=VALUE_RTOL, atol=VALUE_RTOL * np.max(np.abs(F_ana)),
        err_msg="splat+FFT does not reproduce the analytic single-atom F(hkl)",
    )


def test_gradients_match_analytic(system):
    """Finite-difference dF/dx from splat+FFT vs from the analytic formula.

    Two independently-derived expressions for the same quantity, so agreement
    is strong evidence the pipeline (and hence its autograd gradient, which
    differentiates identical arithmetic) is correct.

    Only reflections with h != 0 are compared. d/d(frac_x) of the analytic
    formula carries a factor of h, so h == 0 rows have an IDENTICALLY ZERO
    reference derivative -- a relative error against them is meaningless and
    blows up to ~1e+04 purely from dividing float noise by float noise.
    test_torch_pipeline.py restricts its gradient check the same way and for
    the same reason. Those rows are still checked here, but for absolute
    smallness instead.
    """
    s = system
    Nu, Nv, Nw = s["grid_shape"]
    eps = 1e-5

    def dF_splat(pos, h, k, l):
        pos_p = pos.copy(); pos_p[0] += eps
        pos_m = pos.copy(); pos_m[0] -= eps
        args = (s["A"], s["lam"], s["radius"], s["grid_shape"], s["M"], s["cell_volume"])
        Fp = splat_and_fft(pos_p, *args)[h % Nu, k % Nv, l % Nw]
        Fm = splat_and_fft(pos_m, *args)[h % Nu, k % Nv, l % Nw]
        return (Fp - Fm) / (2 * eps)

    def dF_analytic(pos, hkl1):
        pos_p = pos.copy(); pos_p[0] += eps
        pos_m = pos.copy(); pos_m[0] -= eps
        Fp = analytic_single_atom_F(s["coef9"], s["b_iso"], pos_p, hkl1, s["M"])[0]
        Fm = analytic_single_atom_F(s["coef9"], s["b_iso"], pos_m, hkl1, s["M"])[0]
        return (Fp - Fm) / (2 * eps)

    print("\nGradient check (finite difference, d F(hkl) / d frac_x):")
    checked, zero_rows, scale = [], [], 0.0
    for h, k, l in s["hkl"][:5]:
        num = dF_splat(s["pos_frac"], h, k, l)
        ana = dF_analytic(s["pos_frac"], np.array([[h, k, l]]))
        scale = max(scale, abs(ana))
        tag = "compared" if h != 0 else "h=0, reference is identically zero"
        print(f"{h:3d}{k:3d}{l:3d}   dF_splat={num:10.3f}   dF_analytic={ana:10.3f}   ({tag})")
        (checked if h != 0 else zero_rows).append((num, ana))

    assert checked, "no h != 0 reflections in the test set -- nothing was compared"
    num_arr = np.array([c[0] for c in checked])
    ana_arr = np.array([c[1] for c in checked])
    np.testing.assert_allclose(
        num_arr, ana_arr, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale,
        err_msg="splat+FFT gradient does not match the analytic gradient",
    )

    # h == 0: the reference is exactly zero, so require the computed value to
    # be negligible against the scale of the gradients that are nonzero.
    for num, _ in zero_rows:
        assert abs(num) < GRAD_RTOL * scale, (
            f"h=0 reflection should have ~zero d/d(frac_x), got {num}"
        )
