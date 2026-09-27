#!/usr/bin/env ccp4-python
"""Print the first 20 ASU reflections from gemmi for a space group,
and check our hand-coded ASU condition against gemmi's."""
import sys
import math
import gemmi

sg_name = sys.argv[1] if len(sys.argv) > 1 else 'I23'
dmin    = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

sg   = gemmi.find_spacegroup_by_name(sg_name)
cell = gemmi.UnitCell(20, 20, 20, 90, 90, 90)  # cubic

print(f"SG: {sg.hm}  Laue: {sg.laue_str()}  Ops: {len(list(sg.operations()))}")

asu = gemmi.ReciprocalAsu(sg)

# Enumerate reflections and print first 20
rc      = cell.reciprocal()
H_max   = int(math.ceil(1.0 / (dmin * rc.a)))
in_asu  = []
for H in range(0, H_max+1):
    for K in range(-H_max, H_max+1):
        for L in range(-H_max, H_max+1):
            m = (H, K, L)
            if not asu.is_in(m):
                continue
            inv_d2 = H**2*rc.a**2 + K**2*rc.b**2 + L**2*rc.c**2
            if inv_d2 <= 0 or inv_d2 > 1/dmin**2:
                continue
            in_asu.append((H, K, L))

print(f"Total in gemmi ASU at dmin={dmin}: {len(in_asu)}")
print(f"First 20:")
for hkl in in_asu[:20]:
    h,k,l = hkl
    print(f"  {h:3d} {k:3d} {l:3d}   H+K+L={h+k+l:+d}")

# Count how many satisfy H>=K>=L>=0 (our m-3/m-3m condition)
our_asu = [(h,k,l) for h,k,l in in_asu if h>=k>=l>=0]
print(f"\nOf those, satisfying H>=K>=L>=0: {len(our_asu)}")
print(f"Of those, satisfying H>=K>=0, L>=0 (but NOT L<=K): {len([x for x in in_asu if x[0]>=x[1]>=0 and x[2]>=0])}")

# Show which gemmi ASU reflections are NOT in our H>=K>=L>=0 condition
not_in_ours = [(h,k,l) for h,k,l in in_asu if not (h>=k>=l>=0)]
print(f"\nGemmi ASU reflections NOT satisfying H>=K>=L>=0: {len(not_in_ours)}")
for hkl in not_in_ours[:15]:
    h,k,l = hkl
    print(f"  {h:3d} {k:3d} {l:3d}   H+K+L={h+k+l:+d}")
