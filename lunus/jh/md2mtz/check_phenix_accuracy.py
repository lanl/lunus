"""
Compare phenix.fmodel vs gemmi sfcalc structure factor amplitudes.
Usage: ccp4-python check_phenix_accuracy.py
"""
import gemmi
import numpy as np
import sys

gemmi_mtz  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gemmi_ref_1k.mtz"
phenix_mtz = sys.argv[2] if len(sys.argv) > 2 else "/tmp/phenix_1k.mtz"

mg = gemmi.read_mtz_file(gemmi_mtz)
mp = gemmi.read_mtz_file(phenix_mtz)

ag = np.array(mg)
ap = np.array(mp)

# sort both by H,K,L
ig = np.lexsort((ag[:,2], ag[:,1], ag[:,0]))
ip = np.lexsort((ap[:,2], ap[:,1], ap[:,0]))
ag = ag[ig]
ap = ap[ip]

if not np.all(ag[:,:3] == ap[:,:3]):
    sys.exit("ERROR: HKL indices do not match between files")

# column indices
gc = {c.label: i for i, c in enumerate(mg.columns)}
pc = {c.label: i for i, c in enumerate(mp.columns)}

Fg = ag[:, gc['FC']]        # gemmi amplitude
Fp = ap[:, pc['FMODEL']]    # phenix amplitude

denom = np.maximum(Fg, 1e-6)
rel = np.abs(Fg - Fp) / denom

print(f"{'F range':>20s}  {'N':>8s}  {'mean_rel%':>10s}  {'max_rel%':>10s}")
print("-" * 55)
for lo, hi in [(100, 1e9), (10, 100), (1, 10), (0, 1)]:
    if hi < 1e9:
        mask = (Fg >= lo) & (Fg < hi)
        label = f"[{lo:.0f}, {hi:.0f})"
    else:
        mask = Fg >= lo
        label = f">= {lo:.0f}"
    n = mask.sum()
    if n == 0:
        continue
    print(f"{label:>20s}  {n:>8d}  {rel[mask].mean()*100:>10.3f}  {rel[mask].max()*100:>10.1f}")

print(f"\nOverall: N={len(rel)}  mean_rel={rel.mean()*100:.3f}%  median_rel={np.median(rel)*100:.3f}%  max_rel={rel.max()*100:.1f}%")
