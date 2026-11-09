#!/usr/bin/env ccp4-python
import sys, gemmi
m = gemmi.read_mtz_file(sys.argv[1])
print('Columns:', [c.label for c in m.columns])
print('Spacegroup:', m.spacegroup.hm)
print('Nref:', m.nreflections)
