"""
Checks the IT92 scattering coefficients this package reads from gemmi or
cctbx.

Nothing here is a self-consistency check against a stored copy of the
table -- there is no stored copy. Instead:

  1. the two independent implementations are required to agree,
  2. the coefficients are checked against PHYSICS rather than against any
     implementation: for a neutral atom, f(s -> 0) = sum(a_i) + c must
     equal the atomic number.

Check 2 is the one that would catch a wrong table entirely.
"""

import numpy as np
import pytest

from lunus.sf.elements import DEFAULT_ELEMENTS, it92_coefficients


def _have(source):
    try:
        it92_coefficients(["C"], source=source)
    except ImportError:
        return False
    return True


HAVE_GEMMI = _have("gemmi")
HAVE_CCTBX = _have("cctbx")

pytestmark = pytest.mark.skipif(
    not (HAVE_GEMMI or HAVE_CCTBX),
    reason="neither gemmi nor cctbx installed; no coefficient source",
)


def test_table_is_well_formed():
    coeffs = it92_coefficients()
    assert coeffs, "no coefficients returned"
    assert set(coeffs) == set(DEFAULT_ELEMENTS)

    for element, values in sorted(coeffs.items()):
        assert len(values) == 9, (
            f"{element}: expected 9 coefficients (a1..a4, b1..b4, c), got {len(values)}"
        )
        assert np.all(np.isfinite(values)), f"{element}: non-finite coefficient"
        # The b values are Gaussian widths in A^2. kernel.py forms
        # 4*pi/(b + B), so a non-positive b would divide by zero or flip the
        # sign of the real-space kernel.
        assert np.all(np.array(values[4:8]) > 0.0), (
            f"{element}: non-positive b coefficient {values[4:8]}"
        )


@pytest.mark.skipif(not (HAVE_GEMMI and HAVE_CCTBX),
                    reason="needs both gemmi and cctbx to cross-check")
def test_gemmi_and_cctbx_agree():
    """Two independently maintained implementations of the same published
    table must agree. cctbx stores it in single precision, so agreement is
    to float32 resolution rather than exact -- e.g. it reports b4(H) as
    57.799702 where gemmi has the published 57.7997.
    """
    from_gemmi = it92_coefficients(source="gemmi")
    from_cctbx = it92_coefficients(source="cctbx")

    assert set(from_gemmi) == set(from_cctbx)
    worst, worst_el = 0.0, None
    for element in sorted(from_gemmi):
        g = np.array(from_gemmi[element], dtype=np.float64)
        c = np.array(from_cctbx[element], dtype=np.float64)
        rel = np.max(np.abs(g - c) / np.maximum(np.abs(g), 1e-12))
        if rel > worst:
            worst, worst_el = rel, element
        np.testing.assert_allclose(
            c, g, rtol=1e-6, atol=1e-6,
            err_msg=f"{element}: gemmi and cctbx disagree on the IT92 coefficients",
        )
    print(f"\nworst relative difference {worst:.2e} ({worst_el}), "
          "consistent with float32 storage in cctbx")


@pytest.mark.parametrize("element", sorted(DEFAULT_ELEMENTS))
def test_forward_scattering_equals_atomic_number(element):
    """f(s -> 0) = sum(a_i) + c must equal Z for a neutral atom.

    Independent of any implementation: it follows from the definition of
    the scattering factor, which at zero scattering angle counts the
    electrons. Catches a wrong row, a misordered tuple (a's swapped with
    b's), or a truncated table.

    The IT92 fit is a 4-Gaussian approximation, so it reproduces Z to a few
    parts in 1e4 rather than exactly. Measured over the default element set,
    the worst are Mg (11.9865 vs 12, 1.1e-03), N (6.9946 vs 7, 7.7e-04) and
    Na (10.9924 vs 11, 6.9e-04); the tolerance below is set just above that.
    It is loose enough to pass any correct row and still tight enough to
    reject a different element -- adjacent elements differ by ~1/Z, i.e.
    several percent, which is more than an order of magnitude above this.
    """
    atomic_numbers = {
        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Mg": 12,
        "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20, "Mn": 25,
        "Fe": 26, "Ni": 28, "Cu": 29, "Zn": 30, "Br": 35, "I": 53,
    }
    values = it92_coefficients([element])[element]
    f0 = sum(values[0:4]) + values[8]
    z = atomic_numbers[element]
    print(f"\n{element:3s} f(0) = {f0:9.4f}   Z = {z}   rel = {abs(f0 - z) / z:.2e}")
    assert abs(f0 - z) / z < 2e-3, (
        f"{element}: f(0) = {f0:.4f} does not recover Z = {z}; "
        "the coefficients are not this element's"
    )


def test_arbitrary_element_needs_no_code_change():
    """Elements outside the default set are available on request -- the
    point of reading the table instead of storing a curated copy.
    """
    coeffs = it92_coefficients(["Se"])
    assert set(coeffs) == {"Se"}
    f0 = sum(coeffs["Se"][0:4]) + coeffs["Se"][8]
    assert abs(f0 - 34) / 34 < 2e-3, f"Se: f(0) = {f0:.4f} does not recover Z = 34"


def test_unknown_element_is_rejected():
    with pytest.raises((KeyError, ValueError, RuntimeError)):
        it92_coefficients(["Xx"])
