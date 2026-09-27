#!/usr/bin/env ccp4-python
import gemmi
for el in ['C', 'H', 'N', 'O', 'S']:
    coef = gemmi.Element(el).it92
    print(f"{el}: a={list(coef.a)}  b={list(coef.b)}  c={coef.c}")
