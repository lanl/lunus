"""
Real-space atomic density kernel, derived from the Cromer-Mann form factor.
Isotropic B first; the anisotropic generalization is at the end of this note.

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

ANISOTROPIC ADPs
----------------
The Debye-Waller factor is then exp(-2 pi^2 S.U.S), with U in A^2 and S the
scattering vector in A^-1, so term i becomes

    a_i exp(-b_i |S|^2/4) exp(-2 pi^2 S.U.S) = a_i exp(-S.Q_i.S)

    Q_i = (beta_i/4) I + 2 pi^2 U ,   beta_i = b_i + blur   (beta_5 = blur)

a symmetric 3x3 in A^2. The same Gaussian transform, now in matrix form,

    integral exp(-S.Q.S) exp(-2 pi i S.r) d^3S
        = pi^{3/2} / sqrt(det Q) * exp(-pi^2 r.Q^{-1}.r)

gives an ELLIPSOIDAL Gaussian per term:

    rho_i(r) = A_i exp(-r.Lam_i.r),   A_i = a_i pi^{3/2}/sqrt(det Q_i),
                                      Lam_i = pi^2 Q_i^{-1}

THE FIVE Lam_i SHARE EIGENVECTORS. They differ only by a multiple of the
identity, so all five are diagonal in U's own eigenbasis, and ONE
eigendecomposition of U serves all five rather than five 3x3 inversions.
Writing U = V diag(u) V^T and e = V^T r,

    q_ik = beta_i/4 + 2 pi^2 u_k          (the eigenvalues of Q_i)
    A_i  = alpha_i pi^{3/2} / sqrt(prod_k q_ik)
    w_ik = pi^2 / q_ik
    rho(r) = sum_i A_i exp(-sum_k w_ik e_k^2)

which is the cheap way to PRECOMPUTE the kernel. It is NOT the way to evaluate
it in the splat, which was the natural inference and is measured false.

Rotating into the eigenbasis in the hot loop needs e = V^T d per atom-voxel
pair, an (m,K,3) intermediate and a batched (m,3,3) matmul. Expanding the
quadratic form instead,

    d.Lam_i.d = c_off.Lam_i.c_off - 2 c_rem.Lam_i.c_off + c_rem.Lam_i.c_rem

keeps every intermediate at (m,K) and needs only plain 2-D matmuls, at the
price of doing that per Gaussian -- 5x the matmul flops. The expanded form
still WINS, because this core is memory-bound rather than flop-bound: 16 MB of
intermediates per chunk against 96 MB, and 2-D matmuls where a batched one
cannot fuse with the surrounding elementwise chain. Measured, compiled, over
four chunk shapes: expanded 2.5-3.4x the isotropic core, eigenbasis 3.6-6.0x.
(CPU. The flop-to-bandwidth balance differs on CUDA and the winner may not.)

Setting U = u I recovers the isotropic case exactly: q_ik = (b_i + blur +
8 pi^2 u)/4 = beta/4 with B = 8 pi^2 u, hence A_i = alpha_i (4 pi/beta)^{3/2}
and w_ik = 4 pi^2/beta = lam_i. tests/test_adp_aniso.py asserts it rather than
taking it on trust.

POSITIVE DEFINITENESS is required and is not automatic. q_5k = blur/4 +
2 pi^2 u_k, so the constant term needs U strictly positive definite whenever
blur is 0 -- and deposited models do contain non-positive-definite ADPs. The
isotropic path carries the same constraint in the milder form B > 0.
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


def precalculate_density_aniso(coef9, u_cart, blur=0.0):
    """
    (A, w, V) for one atom with an anisotropic ADP -- see the derivation above.

    u_cart is the (3,3) symmetric ADP in CARTESIAN A^2, the frame the
    displacements are measured in. Returns

        A  (5,)    amplitudes
        w  (5,3)   exponent coefficients, per Gaussian per principal axis
        V  (3,3)   eigenvectors of u_cart, as COLUMNS

    with rho(r) = sum_i A_i exp(-sum_k w[i,k] (V[:,k] . r)^2).

    Raises on a non-positive-definite Q, which the constant term makes easy to
    hit: q_5k = blur/4 + 2 pi^2 u_k, so with blur = 0 any non-positive ADP
    eigenvalue is fatal rather than merely unphysical.
    """
    coef9 = np.asarray(coef9, dtype=np.float64)
    alpha = np.concatenate([coef9[0:4], coef9[8:9]])          # a_1..a_4, c
    beta = np.concatenate([coef9[4:8] + blur, [blur]])        # b_i + blur, blur

    u_cart = np.asarray(u_cart, dtype=np.float64)
    u_cart = 0.5 * (u_cart + u_cart.T)          # eigh reads one triangle only
    u_eval, V = np.linalg.eigh(u_cart)

    q = beta[:, None] / 4.0 + 2.0 * PI ** 2 * u_eval[None, :]          # (5,3)
    if not np.all(q > 0.0):
        raise ValueError(
            "anisotropic kernel needs every (beta_i/4 + 2 pi^2 u_k) > 0; the "
            "minimum here is %.4g, from ADP eigenvalues %s with blur %.4g. "
            "The constant term has beta = blur, so a non-positive-definite ADP "
            "is fatal unless blur > 0."
            % (q.min(), np.array2string(u_eval, precision=4), blur))

    A = alpha * PI ** 1.5 / np.sqrt(np.prod(q, axis=1))
    w = PI ** 2 / q
    return A, w, V


def density_at_aniso(A, w, V, r):
    """rho at Cartesian displacements r, shape (..., 3)."""
    e2 = (np.asarray(r) @ V) ** 2                    # (...,3), in the eigenbasis
    expo = np.tensordot(e2, np.asarray(w), axes=([-1], [1]))       # (...,5)
    return np.tensordot(np.exp(-expo), np.asarray(A), axes=([-1], [0]))


def cutoff_radius_aniso(A, w, V, cutoff=1e-5, r_max=25.0, n=20000):
    """Radius of a SPHERE containing every point where |rho| >= cutoff.

    Scanned along each principal axis and maximised, since the density is an
    ellipsoid and the sphere must contain its longest semi-axis. Along axis k
    the density is sum_i A_i exp(-w[i,k] t^2) -- the isotropic problem with
    lam = w[:,k] -- so this is cutoff_radius three times.

    A sphere rather than the exact ellipsoid, because the candidate-offset
    machinery is spherical and real ADPs are only mildly anisotropic. It costs
    extra candidate voxels for an elongated atom and never loses density.
    """
    return max(cutoff_radius(A, np.asarray(w)[:, k], cutoff=cutoff,
                             r_max=r_max, n=n) for k in range(3))


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


def build_atom_kernels_aniso(elements_per_atom, coeff_table, u_cart_per_atom,
                             blur=0.0, cutoff=1e-5):
    """
    Per-atom anisotropic kernels: (A, L6, radius) for n atoms.

    The anisotropic counterpart of build_atom_kernels. u_cart_per_atom is
    (n,3,3), the ADP of each atom in CARTESIAN A^2 -- the frame displacements
    are measured in, which is NOT the u_star a PDB stores; convert first.

    Returns A (n,5), L6 (n,5,6) and radius (n,). L6 packs Lam_i as
    (xx, yy, zz, xy, xz, yz), which is what the splat consumes.
    """
    coef = np.array([coeff_table[e] for e in elements_per_atom], dtype=np.float64)
    A, L6, w, _V = precalculate_density_aniso_batch(coef, u_cart_per_atom,
                                                    blur=blur)
    radius = cutoff_radius_aniso_batch(A, w, cutoff=cutoff)
    return A, L6, radius


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


def precalculate_density_aniso_batch(coef9_batch, u_cart_batch, blur=0.0):
    """
    Vectorized precalculate_density_aniso: (n,9), (n,3,3) -> (A, L6, w, V).

    Batched because a Python loop over 10^5 atoms is not viable -- the same
    reason precalculate_density_iso_batch exists.

    Returns, per atom,
        A   (n,5)     amplitudes
        L6  (n,5,6)   Lam_i as its six unique components, ordered
                      (xx, yy, zz, xy, xz, yz)
        w   (n,5,3)   exponent coefficients in the eigenbasis
        V   (n,3,3)   eigenvectors as columns

    L6 IS WHAT THE SPLAT WANTS and w, V are returned for tests and for anyone
    who needs the principal axes: the hot loop expands d.Lam_i.d rather than
    rotating into the eigenbasis, because that keeps every intermediate at
    (m,K). Measured, that is 1.7x the isotropic core on CUDA against 2.6x for
    the rotation -- see tools/bench_aniso_core.py.
    """
    coef9_batch = np.asarray(coef9_batch, dtype=np.float64)
    u_cart_batch = np.asarray(u_cart_batch, dtype=np.float64)
    u_cart_batch = 0.5 * (u_cart_batch + np.swapaxes(u_cart_batch, -1, -2))

    alpha = np.concatenate([coef9_batch[:, 0:4], coef9_batch[:, 8:9]], axis=1)
    beta = np.concatenate([coef9_batch[:, 4:8] + blur,
                           np.full((len(coef9_batch), 1), blur)], axis=1)

    u_eval, V = np.linalg.eigh(u_cart_batch)                    # (n,3), (n,3,3)
    q = beta[:, :, None] / 4.0 + 2.0 * PI ** 2 * u_eval[:, None, :]   # (n,5,3)
    if not np.all(q > 0.0):
        bad = np.unique(np.where(~(q > 0.0))[0])
        raise ValueError(
            "anisotropic kernel needs every (beta_i/4 + 2 pi^2 u_k) > 0; %d of "
            "%d atoms fail, first at index %d with ADP eigenvalues %s and blur "
            "%.4g. The constant term has beta = blur, so a non-positive-"
            "definite ADP is fatal unless blur > 0."
            % (len(bad), len(q), bad[0],
               np.array2string(u_eval[bad[0]], precision=4), blur))

    A = alpha * PI ** 1.5 / np.sqrt(np.prod(q, axis=2))          # (n,5)
    w = PI ** 2 / q                                              # (n,5,3)

    # Lam_i = V diag(w_i) V^T, kept as its six unique components.
    Lam = np.einsum("nmk,njk,nlk->njml", V, w, V)                # (n,5,3,3)
    L6 = np.stack([Lam[:, :, 0, 0], Lam[:, :, 1, 1], Lam[:, :, 2, 2],
                   Lam[:, :, 0, 1], Lam[:, :, 0, 2], Lam[:, :, 1, 2]], axis=-1)
    return A, L6, w, V


def cutoff_radius_aniso_batch(A, w, cutoff=1e-5, **kwargs):
    """Per-atom sphere radius containing the whole ellipsoid.

    The slowest decay is along the principal axis with the SMALLEST exponent
    coefficient, so the bounding sphere is set by taking, per atom, the
    smallest w over axes and handing the isotropic batch scanner that. Doing
    it per axis and maximising would give the same answer more slowly, since
    the radius is monotone decreasing in w.
    """
    return cutoff_radius_batch(A, np.asarray(w).min(axis=2), cutoff=cutoff,
                               **kwargs)


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
