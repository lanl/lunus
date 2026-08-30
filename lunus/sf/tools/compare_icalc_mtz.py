"""
Compares Icalc across the FULL reflection set between the gemmi and
torch engines, broken down by resolution shell -- rather than eyeballing
a handful of reflections (which, if they're just the first few lines of
a printed list, are likely all clustered at one extreme corner of the
resolution sphere and not representative of overall agreement).

Requires cctbx (same environment xtraj.py itself runs in) -- this can't
be run in the sandbox that produced everything else in this package.

Usage: python compare_icalc_mtz.py icalc_gemmi.mtz icalc_torch.mtz

Note: xtraj.py's default icalc_file="icalc.mtz" is the SAME for both
engines, so the second run will silently overwrite the first run's
output unless you pass distinct icalc_file= values (or rename the file
between runs) -- e.g.:
    python xtraj.py ... engine=gemmi icalc_file=icalc_gemmi.mtz ...
    python xtraj.py ... engine=torch icalc_file=icalc_torch.mtz ...
"""

import sys
import numpy as np
from iotbx.reflection_file_reader import any_reflection_file


def load_icalc(path):
    hkl_in = any_reflection_file(file_name=path)
    arrays = hkl_in.as_miller_arrays()
    return arrays[0]  # written as the sole array via avg_icalc.as_mtz_dataset('Iavg')


def main():
    path_gemmi = sys.argv[1] if len(sys.argv) > 1 else "icalc_gemmi.mtz"
    path_torch = sys.argv[2] if len(sys.argv) > 2 else "icalc_torch.mtz"

    a_gemmi = load_icalc(path_gemmi)
    a_torch = load_icalc(path_torch)

    print(f"gemmi: {a_gemmi.indices().size()} reflections, d_min={a_gemmi.d_min():.3f}")
    print(f"torch: {a_torch.indices().size()} reflections, d_min={a_torch.d_min():.3f}")

    common_gemmi, common_torch = a_gemmi.common_sets(a_torch)
    n = common_gemmi.indices().size()
    print(f"common reflections: {n}")
    if n < 10:
        print("Too few common reflections to draw conclusions -- check that both files "
              "cover the same resolution range / crystal symmetry.")
        return

    ig = np.array(common_gemmi.data())
    it = np.array(common_torch.data())
    d = np.array(common_gemmi.d_spacings().data())

    corr = np.corrcoef(ig, it)[0, 1]
    r_factor = np.sum(np.abs(ig - it)) / np.sum(ig + it)
    print(f"\nOVERALL (all {n} common reflections):")
    print(f"  correlation = {corr:.6f}")
    print(f"  R-factor    = {r_factor:.6f}  (sum|gemmi-torch| / sum(gemmi+torch))")

    print("\nBy resolution shell (highest resolution first, equal reflection count per shell):")
    print(f"{'d range (A)':>18}  {'n':>8}  {'correlation':>12}  {'R-factor':>10}")
    edges = np.percentile(d, np.linspace(0, 100, 11))
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d <= hi)
        if mask.sum() < 2:
            continue
        shell_corr = np.corrcoef(ig[mask], it[mask])[0, 1]
        shell_r = np.sum(np.abs(ig[mask] - it[mask])) / np.sum(ig[mask] + it[mask])
        print(f"  {lo:7.3f} - {hi:7.3f}  {mask.sum():8d}  {shell_corr:12.4f}  {shell_r:10.4f}")

    print("\nIf the highest-resolution shell (smallest d, top row) shows much worse "
          "correlation/R-factor than the rest, that's consistent with expected physical "
          "sensitivity near the resolution limit rather than a pervasive bug. If ALL shells "
          "show poor agreement, something is still wrong more broadly.")


if __name__ == "__main__":
    main()
