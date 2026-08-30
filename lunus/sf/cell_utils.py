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


def recommended_blur(b_min, grid_shape, orth_matrix_np, safety=1.0,
                     d_min=None):
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
    sigma = sqrt(B / 8 pi^2), and it needs to be resolved by the voxel, giving
    B_target = C h^2 with h the LARGEST voxel dimension, since the coarsest
    direction is what limits. safety scales it for anyone wanting more margin.

    VALIDATED, and with a stated range. C was originally fitted to one curve on
    one structure at one spacing. It has since been measured against exact
    direct summation on TWO unrelated crystals -- 7FPV (P2_1 2_1 2_1) and 4WOR
    (P4_1) -- across four grid spacings at fixed d_min, so that sampling is
    isolated from resolution, and three B values. Fitting B_eff = C h^2 through
    the origin over the rows where blur was actually needed:

        7FPV alone   C = 66.3
        4WOR alone   C = 64.7

    Two unrelated structures agreeing to ~1% is the transferability question
    answered: the h^2 FORM is sound and is not a property of one crystal.

    The constant that matters is the one covering the worst row, not the fit,
    and it depends on how finely the grid samples the resolution:

        rate >= 1.5  (h <= d_min/3, what grid_shape_for_resolution gives)
                     C = 55 covers every row -> 57 is adequate, 4% margin
        rate <  1.5  C = 78 needed -> 57 under-blurs by up to 38%

    So 57 is right for the regime this pipeline actually runs in and wrong
    below it. Raising it to 78 is NOT the fix: it would over-blur by 1.4x
    everywhere normal, and over-blurring is what costs 5x at high B. Below
    rate 1.5 the achievable floor degrades anyway (R 0.00025-0.00029 against
    0.00022-0.00024), so no blur recovers it -- the grid is too coarse and
    that is the thing to fix. Pass d_min to be warned when you are there.

    One judgement call is baked in: "adequate" means R within 30% of the floor
    that structure reaches at any blur. A stricter tolerance would ask for a
    larger C.

    Returns 0.0 when the model's own B already clears the target, which is the
    common case for deposited structures and is why turning this on moved no
    previously published number.
    """
    import numpy as np

    # Voxel edge lengths: the columns of the orthogonalization matrix are the
    # cell axis vectors, so their norms are a, b, c.
    spacing = (np.linalg.norm(np.asarray(orth_matrix_np), axis=0)
               / np.asarray(grid_shape, dtype=float))
    h = float(spacing.max())
    if d_min is not None and h > d_min / 3.0:
        import warnings
        warnings.warn(
            "grid spacing %.3f A is coarser than d_min/3 (%.3f A), i.e. rate "
            "< 1.5. The blur target is calibrated for rate >= 1.5 and "
            "under-blurs by up to 38%% below it -- and the accuracy floor is "
            "worse there whatever the blur, so more blur will not rescue it. "
            "Use a finer grid." % (h, d_min / 3.0), RuntimeWarning, stacklevel=2)
    b_target = safety * 57.0 * h * h
    return max(0.0, b_target - float(b_min))
