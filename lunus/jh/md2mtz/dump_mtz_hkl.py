#!/usr/bin/env ccp4-python
"""Dump first 20 HKL from an MTZ file and their systematic absence status."""
import sys
import gemmi
import numpy as np

path = sys.argv[1]
mtz  = gemmi.read_mtz_file(path)
cols = [col.label for col in mtz.columns]
arr  = np.array(mtz.array)
H    = arr[:, cols.index('H')].astype(int)
K    = arr[:, cols.index('K')].astype(int)
L    = arr[:, cols.index('L')].astype(int)

print(f"{path}: {mtz.nreflections} reflections, SG={mtz.spacegroup.hm if mtz.spacegroup else 'None'}")
print(f"H range: {H.min()}..{H.max()},  K: {K.min()}..{K.max()},  L: {L.min()}..{L.max()}")
print("First 20:")
for i in range(min(20, len(H))):
    print(f"  {H[i]:4d} {K[i]:4d} {L[i]:4d}   H+K+L={H[i]+K[i]+L[i]:+d}")
