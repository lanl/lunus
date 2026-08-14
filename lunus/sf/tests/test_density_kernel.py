"""
Verifies the real-space density kernel against the mathematics it is
derived from, independently of any other implementation.

kernel.precalculate_density_iso() returns the closed-form Fourier transform
of the Cromer-Mann form factor (the derivation is written out in kernel.py's
module docstring). Two checks establish that the closed form is right
without reference to how anyone else computes it:

  1. NUMERICAL TRANSFORM. Fourier-transform f(s) exp(-B s^2) numerically and
     compare to the closed form, point by point in r.
  2. NORMALIZATION. integral rho(r) d^3r must equal f(0) = sum(a_i) + c,
     which for a neutral atom is the atomic number. This is the total charge,
     so it is a physical statement, not an algebraic one.

Check 1 would catch an algebra error in the transform; check 2 would catch a
wrong overall scale factor, which check 1 alone could miss if the same error
appeared in both expressions.
"""

import numpy as np
import pytest

from lunus.sf.elements import it92_coefficients
from lunus.sf.kernel import density_at, precalculate_density_iso

ELEMENTS = ["H", "C", "N", "O", "S", "Fe"]
B_VALUES = [8.0, 20.0, 80.0]

# np.trapz was renamed np.trapezoid in numpy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def numerical_density(coef9, B, r):
    """rho(r) by direct numerical integration, from the definition.

    The 3D transform of an isotropic function reduces to a 1D sine
    transform. With S the scattering vector (|S| = 2 s) and the
    crystallographic convention rho(r) = integral f(S) exp(-2 pi i S.r) d^3S,
    angular integration gives

        rho(r) = 4 pi integral_0^inf f(S) sinc(2 S r) S^2 dS

    where sinc(x) = sin(pi x) / (pi x). Everything here is evaluated in terms
    of s = S/2, which is the variable the IT92 coefficients are defined in.
    """
    a = np.asarray(coef9[0:4], dtype=np.float64)
    b = np.asarray(coef9[4:8], dtype=np.float64)
    c = float(coef9[8])

    # f(s) exp(-B s^2), sampled densely enough that the Gaussians are resolved.
    # The narrowest Gaussian has width ~1/sqrt(b_min + B); integrate well past
    # where all of them have died.
    s_max = 6.0 / np.sqrt(min(b.min() + B, B))
    s = np.linspace(0.0, s_max, 200001)
    f = (a[:, None] * np.exp(-np.outer(b, s * s))).sum(axis=0) + c
    f = f * np.exp(-B * s * s)

    S = 2.0 * s
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    # sinc(2 S r) with numpy's normalized sinc(x) = sin(pi x)/(pi x)
    kernel = np.sinc(2.0 * np.outer(r, S))
    integrand = kernel * (f * S * S)[None, :]
    return 4.0 * np.pi * _trapezoid(integrand, S, axis=1)


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("B", B_VALUES)
def test_closed_form_matches_numerical_transform(element, B):
    """The closed form equals a direct numerical Fourier transform."""
    coef9 = it92_coefficients([element])[element]
    A, lam = precalculate_density_iso(coef9, B)

    # Sample inside the region that carries essentially all the density; the
    # far tail is ~0 in both expressions and its relative error is noise.
    r = np.linspace(0.05, 2.0, 15)
    closed = density_at(A, lam, r * r)
    numeric = numerical_density(coef9, B, r)

    scale = np.max(np.abs(numeric))
    worst = np.max(np.abs(closed - numeric)) / scale
    print(f"\n{element} B={B}: max rel dev {worst:.2e} (peak density {scale:.3f})")

    np.testing.assert_allclose(
        closed, numeric, rtol=2e-4, atol=2e-4 * scale,
        err_msg=f"{element}, B={B}: closed-form density disagrees with the "
                "numerical Fourier transform of f(s)",
    )


@pytest.mark.parametrize("element", ELEMENTS)
@pytest.mark.parametrize("B", B_VALUES)
def test_density_integrates_to_total_charge(element, B):
    """integral rho(r) d^3r = f(0) = sum(a_i) + c.

    Analytically, integral A exp(-lam r^2) d^3r = A (pi/lam)^{3/2}, so the
    total is sum_i A_i (pi/lam_i)^{3/2}. Substituting the closed form gives
    exactly sum_i a_i + c, independently of B -- smearing an atom changes its
    shape, never its charge. A wrong power of t, or a missing factor of
    4 pi, breaks this.
    """
    coef9 = it92_coefficients([element])[element]
    A, lam = precalculate_density_iso(coef9, B)

    total = float(np.sum(A * (np.pi / lam) ** 1.5))
    expected = float(sum(coef9[0:4]) + coef9[8])
    print(f"\n{element} B={B}: integral rho = {total:.6f}, f(0) = {expected:.6f}")

    np.testing.assert_allclose(
        total, expected, rtol=1e-12,
        err_msg=f"{element}, B={B}: density does not integrate to f(0); "
                "the kernel's normalization is wrong",
    )


@pytest.mark.parametrize("element", ELEMENTS)
def test_charge_is_independent_of_b_factor(element):
    """Total charge must not depend on B. Broadening an atom redistributes
    density; it cannot create or destroy electrons.
    """
    coef9 = it92_coefficients([element])[element]
    totals = []
    for B in [1.0, 10.0, 50.0, 200.0]:
        A, lam = precalculate_density_iso(coef9, B)
        totals.append(float(np.sum(A * (np.pi / lam) ** 1.5)))
    print(f"\n{element}: totals across B = {totals}")
    np.testing.assert_allclose(
        totals, totals[0], rtol=1e-12,
        err_msg=f"{element}: total charge varies with B",
    )


def test_batch_matches_scalar():
    """The vectorized per-atom path must agree with the scalar one to
    machine precision -- same formula, different loop structure.
    """
    from lunus.sf.kernel import precalculate_density_iso_batch

    elements = ELEMENTS
    coeffs = it92_coefficients(elements)
    coef_batch = np.array([coeffs[e] for e in elements], dtype=np.float64)
    b_batch = np.array([12.0, 20.0, 35.0, 50.0, 65.0, 80.0], dtype=np.float64)

    A_batch, lam_batch = precalculate_density_iso_batch(coef_batch, b_batch)
    for i, element in enumerate(elements):
        A, lam = precalculate_density_iso(coeffs[element], b_batch[i])
        np.testing.assert_allclose(
            A_batch[i], A, rtol=1e-14,
            err_msg=f"{element}: batched A disagrees with scalar",
        )
        np.testing.assert_allclose(
            lam_batch[i], lam, rtol=1e-14,
            err_msg=f"{element}: batched lam disagrees with scalar",
        )
