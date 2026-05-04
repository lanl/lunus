#!/usr/bin/env ccp4-python
import sys
import gemmi
import numpy as np

f1 = sys.argv[1]
f2 = sys.argv[2]

def mtz_amp_dict(path):
    """Return dict (h,k,l)->|F|, normalising to positive-H half-sphere."""
    mtz = gemmi.read_mtz_file(path)
    cols = [col.label for col in mtz.columns]
    arr = np.array(mtz.array)
    hi = cols.index('H'); ki = cols.index('K'); li = cols.index('L')
    fi = cols.index('FC')
    H = arr[:,hi].astype(int); K = arr[:,ki].astype(int); L = arr[:,li].astype(int)
    F = arr[:,fi]
    # Normalise: put each (H,K,L) into positive-H half-sphere so both files are comparable
    flip = H < 0
    H[flip] = -H[flip]; K[flip] = -K[flip]; L[flip] = -L[flip]
    # Also handle H=0, K<0
    flip2 = (H == 0) & (K < 0)
    K[flip2] = -K[flip2]; L[flip2] = -L[flip2]
    d = {}
    for h,k,l,f in zip(H.tolist(), K.tolist(), L.tolist(), F.tolist()):
        d[(h,k,l)] = f
    return d, mtz.nreflections

d1, n1 = mtz_amp_dict(f1)
d2, n2 = mtz_amp_dict(f2)

print(f"{f1}: {n1} reflections")
print(f"{f2}: {n2} reflections")

common = set(d1) & set(d2)
only1  = len(d1) - len(common)
only2  = len(d2) - len(common)
print(f"common: {len(common)}  only-in-1: {only1}  only-in-2: {only2}")

common_list = list(common)
diffs = np.array([(abs(d1[hkl] - d2[hkl])) / (max(d1[hkl], d2[hkl]) + 1e-6) for hkl in common_list])
print(f"mean rel |F| diff : {diffs.mean():.6f}")
print(f"median rel diff   : {np.median(diffs):.6f}")
imax = diffs.argmax()
hkl_max = common_list[imax]
print(f"max  rel diff     : {diffs[imax]:.6f}  at HKL {hkl_max}  "
      f"F1={d1[hkl_max]:.4f}  F2={d2[hkl_max]:.4f}")
