"""
Real-space isotropic atomic density kernel, derived from the Cromer-Mann
form factor.

DERIVATION
----------
The IT92 parameterization gives the reciprocal-space scattering factor as
a sum of Gaussians in s = sin(theta)/lambda:

    f(s) = sum_{i=1}^{4} a_i exp(-b_i s^2) + c

An isotropic Debye-Waller factor exp(-B s^2), with B = b_iso + blur,
multiplies through and simply shifts each width:

    f(s) exp(-B s^2) = sum_i a_i exp(-(b_i + B) s^2) + c exp(-B s^2)

so every term has the form  alpha * exp(-beta s^2), with
(alpha, beta) = (a_i, b_i + B) for the four Gaussian terms and (c, B) for
the constant term -- the constant is just a Gaussian of width b = 0, which
the Debye-Waller factor gives a finite width.

The real-space density is the Fourier transform of the scattering factor.
Writing S for the scattering vector, whose magnitude is |S| = 2 sin(theta)/
lambda = 2s, so that s^2 = |S|^2 / 4:

    rho(r) = integral f(S) exp(-2 pi i S.r) d^3S

Each term is therefore the transform of a Gaussian, alpha exp(-beta |S|^2 / 4).
The 3D Gaussian transform factorizes into three identical 1D transforms,
each a standard result:

    integral exp(-a S^2) exp(-2 pi i S x) dS = sqrt(pi/a) exp(-pi^2 x^2 / a)

so in three dimensions

    integral exp(-a |S|^2) exp(-2 pi i S.r) d^3S = (pi/a)^{3/2} exp(-pi^2 r^2 / a)

Substituting a = beta/4:

    (4 pi / beta)^{3/2} exp(-4 pi^2 r^2 / beta)

Defining t = 4 pi / beta, the amplitude is alpha * t^{3/2} and the exponent
coefficient is 4 pi^2 / beta = pi * (4 pi / beta) = pi * t. Hence the density
is a sum of five Gaussians in r^2,

    rho_atom(r^2) = sum_{i=1}^{5} A_i exp(-lam_i r^2)

with, for i = 1..4:

    t_i   = 4*pi / (b_i + B)
    A_i   = a_i * t_i^1.5
    lam_i = pi * t_i

and for i = 5, the constant term:

    t_5   = 4*pi / B
    A_5   = c * t_5^1.5
    lam_5 = pi * t_5

which is what the code below computes.

Because this is the exact transform rather than an approximation of it, any
correct implementation of the same parameterization agrees to floating-point
rounding -- including gemmi's, which is what the diffuse-scattering pipeline
compares against. That agreement is verified by test, not assumed: see
tests/test_density_kernel.py, which checks the closed form against direct
numerical integration of f(s) and against the normalization
integral rho(r) d^3r = f(0) = Z.
"""

import math
import numpy as np

PI = math.pi
FOUR_PI = 4.0 * math.pi


def precalculate_density_iso(coef9, B, addend=0.0):
    """
    coef9 : (a1,a2,a3,a4,b1,b2,b3,b4,c) -- IT92 coefficients for one element
    B     : effective isotropic B-factor (b_iso + blur), in A^2
    addend: added to the constant term (e.g. anomalous f'); 0 for our case

    Returns (A, lam), each shape (5,) float64 arrays, such that
        rho_atom(r2) = sum(A * exp(-lam * r2))
    """
    a = np.asarray(coef9[0:4], dtype=np.float64)
    b = np.asarray(coef9[4:8], dtype=np.float64)
    c = float(coef9[8])

    A = np.empty(5, dtype=np.float64)
    lam = np.empty(5, dtype=np.float64)

    t = FOUR_PI / (b + B)              # shape (4,)
    A[:4] = a * t ** 1.5
    lam[:4] = PI * t

    t_c = FOUR_PI / B
    A[4] = (c + addend) * t_c ** 1.5
    lam[4] = PI * t_c

    return A, lam


def density_at(A, lam, r2):
    """rho_atom(r2) = sum_i A_i * exp(-lam_i * r2). r2 may be an array."""
    r2 = np.asarray(r2, dtype=np.float64)
    return np.tensordot(A, np.exp(-np.outer(lam, r2.ravel())), axes=1).reshape(r2.shape)


def cutoff_radius(A, lam, cutoff=1e-5, r_max=25.0, n=20000):
    """Local-support radius: the largest r at which |rho_atom(r)| is still
    >= cutoff, found by a direct radial scan.

    The only requirement on this number is that it not be too SMALL. The
    density beyond it is discarded, so an under-estimate loses real charge,
    while an over-estimate merely evaluates extra grid points whose density
    is below the cutoff by construction -- costing compute, never accuracy.
    Structure factors are therefore unchanged by any radius at or above the
    true one, which is why this does not need to agree exactly with the
    radius any other implementation happens to pick.
    """
    r = np.linspace(0.0, r_max, n)
    dens = density_at(A, lam, r * r)
    above = np.where(np.abs(dens) >= cutoff)[0]
    if len(above) == 0:
        return 0.0
    # small safety margin
    return float(r[above[-1]]) + (r[1] - r[0])


def calculate_sf(coef9, stol2):
    """
    Direct Cromer-Mann evaluation in reciprocal space:
        f(s) = sum_i a_i * exp(-b_i * s^2) + c
    stol2 = (sin(theta)/lambda)^2, in 1/A^2. Used only for the
    independent analytic cross-check, not in the splat pipeline itself.
    """
    a = np.asarray(coef9[0:4], dtype=np.float64)
    b = np.asarray(coef9[4:8], dtype=np.float64)
    c = float(coef9[8])
    stol2 = np.asarray(stol2, dtype=np.float64)
    return c + np.tensordot(a, np.exp(-np.outer(b, stol2.ravel())), axes=1).reshape(stol2.shape)


def build_element_kernels(elements, coeff_table, b_iso, blur=0.0, cutoff=1e-5):
    """
    elements: iterable of element symbols present in the system
    coeff_table: dict element -> IT92 9-tuple (e.g. elements.IT92_COEFFS)
    b_iso, blur: scalars (A^2) -- effective B = b_iso + blur, uniform
                 across the structure (matches xtraj.py's
                 xrs.convert_to_isotropic(); xrs.set_b_iso(...))

    Returns dict: element -> {"A": (5,), "lam": (5,), "radius": float}
    """
    B = b_iso + blur
    out = {}
    for el in elements:
        A, lam = precalculate_density_iso(coeff_table[el], B)
        r = cutoff_radius(A, lam, cutoff)
        out[el] = {"A": A, "lam": lam, "radius": r}
    return out


def precalculate_density_iso_batch(coef9_batch, B_batch, addend=0.0):
    """
    Vectorized precalculate_density_iso across many atoms at once --
    same exact formula, just broadcast over an array of (possibly
    different) B values and IT92 coefficient rows instead of one scalar
    B and one coefficient tuple.

    coef9_batch: (N, 9) array, one IT92 row per atom (already gathered
                 by each atom's element -- see build_atom_kernels)
    B_batch:     (N,) array, one effective B (= b_iso_atom + blur) per atom
    Returns (A, lam), each (N, 5).
    """
    coef9_batch = np.asarray(coef9_batch, dtype=np.float64)
    B_batch = np.asarray(B_batch, dtype=np.float64)
    a = coef9_batch[:, 0:4]          # (N,4)
    b = coef9_batch[:, 4:8]          # (N,4)
    c = coef9_batch[:, 8]            # (N,)
    B = B_batch[:, None]             # (N,1)

    N = coef9_batch.shape[0]
    A = np.empty((N, 5), dtype=np.float64)
    lam = np.empty((N, 5), dtype=np.float64)

    t = FOUR_PI / (b + B)            # (N,4)
    A[:, :4] = a * t ** 1.5
    lam[:, :4] = PI * t

    t_c = FOUR_PI / B_batch          # (N,)
    A[:, 4] = (c + addend) * t_c ** 1.5
    lam[:, 4] = PI * t_c

    return A, lam


def cutoff_radius_batch(A_batch, lam_batch, cutoff=1e-5, r_max=12.0, n_coarse=60, n_bisect=30, chunk_size=20000):
    """
    Vectorized version of cutoff_radius, chunked to bound memory for
    large atom counts, and using a coarse bracket + bisection refinement
    instead of a brute-force fine scan -- the original brute-force
    version (n=2000 points) took 47.6s for 135,834 atoms; this takes a
    small fraction of that, since it needs ~n_coarse + n_bisect density
    evaluations per atom (~90) instead of ~2000.

    r_max=12 A is still generous: measured radii across H/C/N/O/S at
    B up to 120 (well beyond typical protein B-factor ranges) top out
    under 6.2 A.

    Two-stage approach: a coarse scan (n_coarse points) finds the LAST
    crossing of |density|=cutoff, same robustness as the original
    brute-force scan (handles the same rare non-monotonic behavior
    noted for cutoff_radius, e.g. some elements' negative constant
    term), then bisection (n_bisect iterations) refines within that
    bracket to a precision far exceeding what's needed -- 30 bisection
    steps over a ~0.2 A coarse-scan bracket resolves to ~2e-10 A,
    dramatically finer than the original's 0.0125 A scan resolution.
    """
    N = A_batch.shape[0]
    radius = np.empty(N, dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        A_c = A_batch[start:end]        # (m,5)
        lam_c = lam_batch[start:end]    # (m,5)
        m = end - start

        r_coarse = np.linspace(0.0, r_max, n_coarse)
        dr_coarse = r_coarse[1] - r_coarse[0]
        dens_coarse = np.einsum("mi,mni->mn", A_c, np.exp(-lam_c[:, None, :] * (r_coarse * r_coarse)[None, :, None]))
        above = np.abs(dens_coarse) >= cutoff              # (m, n_coarse)
        has_any = above.any(axis=1)
        last_idx = np.where(has_any, n_coarse - 1 - np.argmax(above[:, ::-1], axis=1), 0)
        last_idx = np.clip(last_idx, 0, n_coarse - 2)       # leave room for a hi bracket point

        lo = r_coarse[last_idx]                              # density still >= cutoff here
        hi = r_coarse[last_idx + 1]                           # density < cutoff here (or at least no worse)

        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            dens_mid = np.einsum("mi,mi->m", A_c, np.exp(-lam_c * (mid * mid)[:, None]))
            still_above = np.abs(dens_mid) >= cutoff
            lo = np.where(still_above, mid, lo)
            hi = np.where(still_above, hi, mid)

        radius[start:end] = np.where(has_any, hi, 0.0)

    return radius


def build_atom_kernels(elements_per_atom, coeff_table, b_per_atom, blur=0.0, cutoff=1e-5):
    """
    Per-ATOM version of build_element_kernels: each atom gets its own
    (A, lam, radius) from its own element's IT92 coefficients and its
    own B-factor, rather than every atom of a given element sharing one
    kernel. Since B doesn't vary by configuration, this is meant to be
    called ONCE at setup (not per frame) just like build_element_kernels.

    elements_per_atom: (N,) list/array of element symbols, one per atom
    coeff_table: dict element -> IT92 9-tuple
    b_per_atom: (N,) array of isotropic B-factors (A^2), one per atom
                (blur gets added uniformly on top, same role as before)
    Returns (A, lam, radius): (N,5), (N,5), (N,) numpy arrays.
    """
    coef9_batch = np.array([coeff_table[el] for el in elements_per_atom], dtype=np.float64)
    B_batch = np.asarray(b_per_atom, dtype=np.float64) + blur
    A, lam = precalculate_density_iso_batch(coef9_batch, B_batch)
    radius = cutoff_radius_batch(A, lam, cutoff=cutoff)
    return A, lam, radius
