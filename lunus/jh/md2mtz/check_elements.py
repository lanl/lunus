#!/usr/bin/env ccp4-python
import sys, gemmi, collections
st = gemmi.read_structure(sys.argv[1])
counts = collections.Counter()
for model in st:
    for chain in model:
        for res in chain:
            for atom in res:
                counts[atom.element.name] += 1
for el, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {el:4s}: {n}")
