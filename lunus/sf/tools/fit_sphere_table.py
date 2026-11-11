#!/usr/bin/env python
"""Regenerate mask_spheres.SPHERE_FIT_TABLE: the five-Gaussian fit of the
soft sphere for every radius the default united-atom table and element
fallback produce, at offset 0 and the default edge. Prints the Python
literal to paste over the table in lunus/sf/mask_spheres.py.

    PYTHONPATH=<repo root> python lunus/sf/tools/fit_sphere_table.py > table.txt
"""
import sys

from lunus.sf import mask_spheres as ms

ELEMENTS = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Se", "Na", "K", "Mg", "Ca",
            "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg", "B", "Si", "As", "Pt", "Au", "Li",
            "Rb", "Cs", "Sr", "Ba", "W", "Mo", "V", "Cr", "Al", "Ga", "Yb", "Gd", "Tb", "Eu",
            "Sm", "Lu", "Ir", "Os", "Ru", "Rh", "Pd", "Ag", "Pb", "Tl", "Xe", "Kr"]


def main():
    radii = set(round(0.5 * s, 3) for s in ms.SIGMA_UA.values())
    radii |= {round(0.5 * ms.lj_sigma("", "", e, by_element=True), 3) for e in ELEMENTS}
    print("SPHERE_FIT_TABLE = {")
    for r in sorted(radii):
        A, lam = ms.gaussian_fit.__wrapped__(r, ms.EDGE_WIDTH_DEFAULT, 5, 20.0, 8)
        print("    %.3f: ((%s), (%s))," % (r, ", ".join("%.6f" % a for a in A),
                                          ", ".join("%.6f" % l for l in lam)))
        sys.stdout.flush()
    print("}")


if __name__ == "__main__":
    main()
