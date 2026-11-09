#!/usr/bin/env python3
"""Generate a 10-atom random P1 PDB for axis-ordering diagnostics."""
import random
import sys

random.seed(42)
N = 10
A = 25.0  # cubic cell, Angstroms

print(f"CRYST1 {A:8.3f} {A:8.3f} {A:8.3f}  90.00  90.00  90.00 P 1           1")
for i in range(N):
    x = random.uniform(1.0, A - 1.0)
    y = random.uniform(1.0, A - 1.0)
    z = random.uniform(1.0, A - 1.0)
    print(f"ATOM  {i+1:5d}  CA  ALA A {i+1:3d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C")
print("END")
