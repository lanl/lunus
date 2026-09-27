#!/usr/bin/env ccp4-python
"""Copy P1test_B20.pdb but assign random B factors in [2, 999]."""
import sys, random, gemmi
random.seed(7)
st = gemmi.read_structure('P1test_B20.pdb')
for model in st:
    for chain in model:
        for res in chain:
            for atom in res:
                atom.b_iso = random.uniform(2.0, 999.0)
st.write_pdb('P1test_randB.pdb')
print("Written P1test_randB.pdb")
