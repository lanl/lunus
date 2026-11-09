#!/usr/bin/env ccp4-python
"""Print ratio of GPU/gemmi amplitudes for first 10 common reflections."""
import sys
import gemmi
import numpy as np

def load(path):
    mtz  = gemmi.read_mtz_file(path)
    cols = [col.label for col in mtz.columns]
    arr  = np.array(mtz.array)
    return {(int(row[0]), int(row[1]), int(row[2])): row[3] for row in arr}

gemmi_mtz = sys.argv[1]
gpu_mtz   = sys.argv[2]

g = load(gemmi_mtz)
c = load(gpu_mtz)
common = sorted(set(g) & set(c))[:15]
print('  HKL                 gemmi       GPU      ratio')
for hkl in common:
    r = c[hkl] / g[hkl] if g[hkl] else float('nan')
    print(f'  {str(hkl):25s}  {g[hkl]:9.2f}  {c[hkl]:9.2f}  {r:.4f}')
