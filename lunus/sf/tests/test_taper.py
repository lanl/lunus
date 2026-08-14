"""
Extends the isolated real-space taper demonstration to the FULL
splat -> FFT -> F(hkl) pipeline (with phase factors, not just a real
scalar sum), confirming the gradient is well-behaved (converges cleanly
across shrinking eps) with the smooth taper, unlike the hard cutoff's
erratic finite-difference behavior shown in demo_taper_gradient.py.
"""
import numpy as np
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel import precalculate_density_iso, cutoff_radius
from lunus.sf.cell_utils import orth_matrix, grid_shape_for_resolution


def taper_factor(r, R, w):
    inner = R - w
    t = np.ones_like(r)
    edge = (r > inner) & (r < R)
    t[edge] = 0.5 * (1 + np.cos(np.pi * (r[edge] - inner) / w))
    t[r >= R] = 0.0
    return t


def splat_and_fft_tapered(pos_frac, A, lam, R, w, grid_shape, M, cell_volume):
    Nu, Nv, Nw = grid_shape
    grid = np.zeros((Nu, Nv, Nw), dtype=np.float64)
    spacing = np.array([np.linalg.norm(M[:, 0]) / Nu, np.linalg.norm(M[:, 1]) / Nv, np.linalg.norm(M[:, 2]) / Nw])
    du, dv, dw = [int(np.ceil(R / s)) for s in spacing]

    gpos = pos_frac * np.array([Nu, Nv, Nw])
    base = np.round(gpos).astype(int)
    for iu in range(-du, du + 1):
        for iv in range(-dv, dv + 1):
            for iw in range(-dw, dw + 1):
                idx = base + np.array([iu, iv, iw])
                voxel_frac = idx / np.array([Nu, Nv, Nw])
                d_frac = voxel_frac - pos_frac
                d_frac -= np.round(d_frac)
                d_cart = M @ d_frac
                r2 = float(d_cart @ d_cart)
                r = np.sqrt(r2)
                if r >= R:
                    continue
                taper = taper_factor(np.array([r]), R, w)[0]
                dens = np.sum(A * np.exp(-lam * r2)) * taper
                wu, wv, ww = idx % np.array([Nu, Nv, Nw])
                grid[wu, wv, ww] += dens
    Fgrid = np.fft.ifftn(grid) * cell_volume
    return np.conj(Fgrid)


def test_tapered_gradient_converges():
    """The smooth taper must give a finite-difference gradient that CONVERGES
    as eps shrinks, unlike a hard cutoff, whose gradient swings erratically
    because atoms cross the boundary discontinuously (see
    examples/demo_taper_gradient.py for that contrast).

    Convergence, not any particular value, is the property under test: a
    well-behaved derivative means successive halvings of eps stop changing the
    answer. Measured dF/d(frac_x):

        eps=1e-03  -3.4672-8.6577j
        eps=1e-04  -2.9785-8.4010j
        eps=1e-05  -2.9664-8.3995j
        eps=1e-06  -2.9663-8.3995j

    i.e. the last two agree to ~1e-05 relative while the first two differ by
    ~17%, which is exactly the signature of a converging finite difference.
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
    R = cutoff_radius(A, lam, cutoff=0.01)
    w = 0.1  # matches kernel_torch.py's corrected default

    pos_frac = np.array([0.372, 0.611, 0.148])
    hkl = np.array([5, -3, 7])  # a generic, nontrivial reflection

    print(f"R = {R:.3f} A, taper width = {w} A, grid spacing ~ {a/Nu:.3f} A")
    print("\nConvergence check (gradient should settle to a stable value as eps shrinks,")
    print("unlike the hard-cutoff case's erratic swings):")
    eps_values = [1e-3, 1e-4, 1e-5, 1e-6]
    dF_values = []
    for eps in eps_values:
        pos_p = pos_frac.copy(); pos_p[0] += eps
        pos_m = pos_frac.copy(); pos_m[0] -= eps
        Fp = splat_and_fft_tapered(pos_p, A, lam, R, w, grid_shape, M, cell_volume)[hkl[0] % Nu, hkl[1] % Nv, hkl[2] % Nw]
        Fm = splat_and_fft_tapered(pos_m, A, lam, R, w, grid_shape, M, cell_volume)[hkl[0] % Nu, hkl[1] % Nv, hkl[2] % Nw]
        dF = (Fp - Fm) / (2 * eps)
        dF_values.append(dF)
        print(f"  eps={eps:.0e}: dF/d(frac_x) = {dF:.4f}")

    scale = abs(dF_values[-1])
    assert scale > 1e-6, "gradient is ~zero; this reflection tests nothing"

    steps = [abs(dF_values[i + 1] - dF_values[i]) / scale
             for i in range(len(dF_values) - 1)]
    print("  successive relative changes: " + ", ".join(f"{s:.2e}" for s in steps))

    # The last refinement must barely move the answer...
    assert steps[-1] < 1e-3, (
        f"gradient still changing by {steps[-1]:.2e} between eps=1e-05 and "
        "1e-06; it has not converged"
    )
    # ...and each refinement must improve on the one before it. A hard cutoff
    # fails here: its steps do not shrink monotonically, they jitter.
    for i in range(1, len(steps)):
        assert steps[i] < steps[i - 1], (
            f"finite difference is not settling monotonically: step {i} "
            f"({steps[i]:.2e}) is not smaller than step {i-1} ({steps[i-1]:.2e})"
        )
