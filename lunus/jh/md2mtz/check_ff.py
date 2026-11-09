#!/usr/bin/env ccp4-python
"""Compare GPU IT92 constants vs gemmi's built-in form factors."""
import gemmi
import numpy as np

# GPU constants (from sfcalc_gpu.cu)
gpu_a = {
    'C': [2.31000, 1.02000, 1.58860, 0.86500],
    'H': [0.48992, 0.26200, 0.19677, 0.04990],
    'N': [12.2126, 3.13220, 2.01250, 1.16630],
    'O': [3.04850, 2.28680, 1.54630, 0.86700],
    'S': [6.90530, 5.20340, 1.43790, 1.58630],
}
gpu_b = {
    'C': [20.8439, 10.2075, 0.56870, 51.6512],
    'H': [20.6593,  7.74039, 49.5519,  2.20159],
    'N': [0.00570,  9.89330, 28.9975,  0.58260],
    'O': [13.2771,  5.70110,  0.32390, 32.9089],
    'S': [1.46790, 22.2151,  0.25360, 56.1720],
}
gpu_c = {'C': 0.2156, 'H': 0.0013, 'N': -11.529, 'O': 0.2508, 'S': 0.86630}

def gpu_f(el, s2):
    return sum(a*np.exp(-b*s2) for a,b in zip(gpu_a[el], gpu_b[el])) + gpu_c[el]

print(f"{'el':4s}  {'s':6s}  {'gpu_f':10s}  {'gemmi_f':10s}  {'diff%':8s}")
for el in ['C', 'H', 'N', 'O', 'S']:
    coef = gemmi.Element(el).it92
    for s in [0.0, 0.1, 0.2, 0.3, 0.4]:
        s2 = s*s
        gf  = coef.calculate_sf(stol2=s2)
        gpf = gpu_f(el, s2)
        print(f"{el:4s}  {s:6.3f}  {gpf:10.5f}  {gf:10.5f}  {100*(gpf-gf)/(abs(gf)+1e-10):8.4f}%")
    print()
