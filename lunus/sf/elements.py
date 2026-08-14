"""
Neutral-atom Cromer-Mann analytical scattering-factor coefficients.

The coefficients are the standard IT92 parameterization -- Cromer, D. T. &
Mann, J. B. (1968), Acta Cryst. A24, 321-324, tabulated as International
Tables for Crystallography Vol. C (1992), Table 6.1.1.4 -- giving the
reciprocal-space scattering factor as

    f(s) = sum_{i=1}^{4} a_i * exp(-b_i * s^2) + c

with s = sin(theta)/lambda in 1/A, so the b_i are in A^2.

This module does not carry its own copy of that table. It reads it from
whichever crystallography library is installed, via `it92_coefficients()`:
gemmi by preference (it stores the published decimals in double precision,
so values are exact), otherwise cctbx (which stores them in single
precision, differing by ~5e-8 relative -- physically irrelevant, but it
means results are not bit-identical between the two sources).

Fetching rather than republishing has two practical benefits beyond
avoiding a redundant copy of a large table: coefficients cannot drift out
of sync with the engine they are compared against, and supporting a new
element requires no edit here at all -- just ask for it.

NEUTRAL atoms only; formal atomic charge is ignored. That is the usual
convention for this kind of calculation and matches gemmi's default
(`ignore_charge=True`), so the two engines agree.

Note that nothing in the computational core imports this module.
`kernel.build_element_kernels()` and `build_atom_kernels()` take the
coefficient table as an argument, so a caller with neither gemmi nor cctbx
can supply its own mapping and keep the numpy/torch-only path intact.

Usage:

    from lunus.sf.elements import IT92_COEFFS          # default element set
    from lunus.sf.elements import it92_coefficients

    coeffs = it92_coefficients(["C", "N", "O", "Se"])  # any elements
    coeffs = it92_coefficients(source="cctbx")         # force a source
"""

# Elements typically present in biomolecular MD systems (protein, nucleic
# acid, lipid, water, common ions). This list is only the DEFAULT set built
# for IT92_COEFFS -- it is not a limit. Pass any symbols you like to
# it92_coefficients(); the published table runs H..Cf.
DEFAULT_ELEMENTS = (
    "H", "C", "N", "O", "F", "Na", "Mg", "P", "S", "Cl",
    "K", "Ca", "Mn", "Fe", "Ni", "Cu", "Zn", "Br", "I",
)


def _from_gemmi(symbols):
    import gemmi

    out = {}
    for sym in symbols:
        element = gemmi.Element(sym)
        # gemmi.Element does not raise on an unknown symbol; it yields the
        # X ("unknown") element with atomic number 0, whose coefficients are
        # meaningless. Catch that rather than silently scattering off nothing.
        if element.atomic_number == 0:
            raise KeyError("gemmi does not recognize element symbol %r" % (sym,))
        coef = element.it92
        out[sym] = tuple(coef.a) + tuple(coef.b) + (coef.c,)
    return out


def _from_cctbx(symbols):
    from cctbx.eltbx import xray_scattering

    out = {}
    for sym in symbols:
        gaussian = xray_scattering.it1992(sym, True).fetch()
        out[sym] = (
            tuple(gaussian.array_of_a())
            + tuple(gaussian.array_of_b())
            + (gaussian.c(),)
        )
    return out


_SOURCES = {"gemmi": _from_gemmi, "cctbx": _from_cctbx}


def it92_coefficients(symbols=None, source="auto"):
    """Return {symbol: (a1..a4, b1..b4, c)} for the requested elements.

    symbols: iterable of element symbols; defaults to DEFAULT_ELEMENTS.
    source:  "gemmi", "cctbx", or "auto" (gemmi first, then cctbx).

    Raises KeyError for an unrecognized element, and ImportError if no
    source library is available.
    """
    symbols = tuple(DEFAULT_ELEMENTS if symbols is None else symbols)

    if source != "auto":
        if source not in _SOURCES:
            raise ValueError(
                "unknown source %r; expected one of %s or 'auto'"
                % (source, sorted(_SOURCES))
            )
        return _SOURCES[source](symbols)

    errors = []
    for name in ("gemmi", "cctbx"):
        try:
            return _SOURCES[name](symbols)
        except ImportError as e:
            errors.append("%s: %s" % (name, e))

    raise ImportError(
        "IT92 scattering coefficients need either gemmi or cctbx installed "
        "(%s). Alternatively pass your own {symbol: (a1..a4, b1..b4, c)} "
        "mapping directly to kernel.build_element_kernels() or "
        "build_atom_kernels(), which take the table as an argument."
        % "; ".join(errors)
    )


def __getattr__(name):
    """Build IT92_COEFFS on first access (PEP 562).

    Deferred so that importing this module costs nothing and does not
    require gemmi or cctbx until the coefficients are actually wanted.
    """
    if name == "IT92_COEFFS":
        coeffs = it92_coefficients()
        globals()["IT92_COEFFS"] = coeffs      # cache; __getattr__ won't run again
        return coeffs
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
