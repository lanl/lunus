"""Shared, framework-independent (NumPy) unit-cell geometry helpers."""

import numpy as np


def orth_matrix(a, b, c, alpha, beta, gamma):
    """Cartesian = M @ fractional. Standard crystallographic convention
    (a along x, b in the xy-plane), angles in degrees."""
    al, be, ga = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos(al), np.cos(be), np.cos(ga)
    sg = np.sin(ga)
    v = np.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, 0.0))
    M = np.array([
        [a, b * cg, c * cb],
        [0.0, b * sg, c * (ca - cb * cg) / sg],
        [0.0, 0.0, c * v / sg],
    ])
    return M


def reciprocal_inv_d2(M, hkl):
    """1/d^2 for each row of hkl (n,3), given orthogonalization matrix M."""
    Mstar = np.linalg.inv(M).T
    s_cart = hkl @ Mstar.T
    return np.sum(s_cart * s_cart, axis=1)


def next_fft_friendly(n, primes=(2, 3, 5)):
    """Smallest EVEN integer >= n whose only prime factors are in `primes`.
    Standard practice for FFT grid sizing (gemmi's GridSizeRounding::Up
    does the equivalent) -- confirmed against a real gemmi run: for a raw
    (unrounded) requirement of 661 on one axis, gemmi's actual grid used
    720, which is exactly the smallest even 5-smooth number >= 661 (675,
    an earlier version of this function's answer without the evenness
    constraint, is 3^3*5^2 -- entirely odd, and did not match gemmi)."""
    n = int(n)
    if n % 2 == 1:
        n += 1
    while True:
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 2


def grid_shape_for_resolution(a, b, c, d_min, rate=1.5):
    """Matches gemmi's requested_grid_spacing() = d_min / (2*rate), then
    rounds each axis up to an FFT-friendly size (see next_fft_friendly)."""
    spacing_target = d_min / (2 * rate)
    Nu = next_fft_friendly(int(np.ceil(a / spacing_target)))
    Nv = next_fft_friendly(int(np.ceil(b / spacing_target)))
    Nw = next_fft_friendly(int(np.ceil(c / spacing_target)))
    return Nu, Nv, Nw


def recommended_blur(b_min, grid_shape, orth_matrix_np, safety=1.0):
    """
    How much extra B to splat with so the grid samples the atoms adequately.

    THE FFT-ARTIFACT TRICK. Atoms are splatted with B + blur, which spreads
    them, and compute_fcalc removes it again with exp(+blur * s^2 / 4). The
    answer is unchanged in exact arithmetic; what changes is that a sharp
    Gaussian sampled on a coarse grid is badly represented, and a spread one is
    not.

    IT IS NOT OPTIONAL AT LOW B, and the size of the error is the argument.
    Measured against exact direct summation, 7FPV, 0.633 A grid, d_min 2.0:

        B_iso    blur=0     blur=20    blur=80
          2      0.186       0.0004     0.0006
         10      0.0078      0.0002     0.0007
         30      0.0002      0.0002     0.0012

    R = 0.186 at B = 2 with no blur, which is 37x the pipeline's whole
    disagreement with gemmi. But BLUR IS NOT FREE EITHER: the un-blur
    multiplies by exp(+blur*s^2/4), which at d_min 2 A and blur 80 is a factor
    of 148 applied to whatever error is present, and the B = 30 row shows it
    costing 5x. So this returns a blur that reaches a target and no more.

    The target is set by sampling: the real-space Gaussian has
    sigma = sqrt(B / 8 pi^2), and it needs to be resolved by the voxel. The
    measured floor above is reached by B_eff ~ 20-25 at 0.633 A spacing, i.e.
    sigma ~ 0.85 voxels, which gives

        B_target = 8 pi^2 (0.85 h)^2 ~ 57 h^2

    with h the LARGEST voxel dimension, since the coarsest direction is what
    limits. safety scales the target for anyone who wants more margin.

    Returns 0.0 when the model's own B already clears the target -- which is
    the B = 30 row, where any blur is a net loss.
    """
    import numpy as np

    # Voxel edge lengths: the columns of the orthogonalization matrix are the
    # cell axis vectors, so their norms are a, b, c.
    spacing = (np.linalg.norm(np.asarray(orth_matrix_np), axis=0)
               / np.asarray(grid_shape, dtype=float))
    h = float(spacing.max())
    b_target = safety * 57.0 * h * h
    return max(0.0, b_target - float(b_min))
