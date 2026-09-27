"""
Diagnostic: compare pairs of MTZ files and print relative amplitude differences.
Usage: ccp4-python diag_phenix.py ref.mtz test.mtz [label_ref] [label_test]
"""
import gemmi
import numpy as np
import sys

ref_path  = sys.argv[1]
test_path = sys.argv[2]
label_ref  = sys.argv[3] if len(sys.argv) > 3 else None
label_test = sys.argv[4] if len(sys.argv) > 4 else None

mr = gemmi.read_mtz_file(ref_path)
mt = gemmi.read_mtz_file(test_path)

ar = np.array(mr)
at = np.array(mt)

ir = np.lexsort((ar[:,2], ar[:,1], ar[:,0]))
it = np.lexsort((at[:,2], at[:,1], at[:,0]))
ar = ar[ir]; at = at[it]

# intersect on HKL
hkl_r = {tuple(row[:3].astype(int)): i for i, row in enumerate(ar)}
hkl_t = {tuple(row[:3].astype(int)): i for i, row in enumerate(at)}
common = sorted(set(hkl_r) & set(hkl_t))
if not common:
    sys.exit("ERROR: no common reflections")

rc = {c.label: i for i, c in enumerate(mr.columns)}
tc = {c.label: i for i, c in enumerate(mt.columns)}

# pick amplitude column
def amp_col(cols):
    for name in ('FC', 'FMODEL', 'F'):
        if name in cols:
            return name
    return list(cols.keys())[3]  # fallback: 4th column

acol_r = amp_col(rc)
acol_t = amp_col(tc)

Fr = np.array([ar[hkl_r[h], rc[acol_r]] for h in common])
Ft = np.array([at[hkl_t[h], tc[acol_t]] for h in common])

denom = np.maximum(Fr, 1e-6)
rel = np.abs(Fr - Ft) / denom

label_r = label_ref  or f"{ref_path.split('/')[-1]}:{acol_r}"
label_t = label_test or f"{test_path.split('/')[-1]}:{acol_t}"
print(f"ref : {label_r}")
print(f"test: {label_t}")
print(f"common reflections: {len(common)}")
print()
print(f"{'F range':>15s}  {'N':>8s}  {'mean_rel%':>10s}  {'max_rel%':>10s}")
print("-" * 50)
for lo, hi in [(100, 1e9), (10, 100), (1, 10), (0, 1)]:
    mask = (Fr >= lo) & (Fr < hi) if hi < 1e9 else (Fr >= lo)
    n = mask.sum()
    if n == 0:
        continue
    label = f">={lo:.0f}" if hi >= 1e9 else f"[{lo:.0f},{hi:.0f})"
    print(f"{label:>15s}  {n:>8d}  {rel[mask].mean()*100:>10.3f}  {rel[mask].max()*100:>10.2f}")
print(f"{'overall':>15s}  {len(rel):>8d}  {rel.mean()*100:>10.3f}  {rel.max()*100:>10.2f}")
