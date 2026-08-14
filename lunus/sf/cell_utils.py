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
