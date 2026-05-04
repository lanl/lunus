#!/bin/bash
# Test the full supercell collapse workflow:
#   1. Tile P41212.pdb (20x20x30, P41212, 2870 atoms) into a 3x3x3 P1 supercell
#   2. Calculate structure factors on the GPU for the supercell
#   3. Collapse supercell F's back to P41212 primitive-cell ASU
#   4. Compare with gemmi sfcalc on the original P41212 structure
#
# Prerequisites: sfcalc_gpu.so (run compile_gpu.sh first), ccp4-python, GPU

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f sfcalc_gpu.so ]; then
    echo "ERROR: sfcalc_gpu.so not found. Run compile_gpu.sh first."
    exit 1
fi

DMIN=2.0          # resolution limit (use 2.0 A to keep gemmi fast)
SG=P41212         # space group of the primitive (single) cell
MULT=3,3,3        # supercell multipliers

echo "=== Step 1: tile P41212.pdb into a ${MULT} P1 supercell ==="
ccp4-python make_supercell_pdb.py P41212.pdb super_mult=${MULT} out=P41212_super.pdb

echo ""
echo "=== Step 2: GPU structure factor calculation on supercell ==="
ccp4-python sfcalc_gpu.py P41212_super.pdb \
    dmin=${DMIN} outmtz=P41212_super_gpu.mtz outmap= bmax=0

echo ""
echo "=== Step 3: collapse supercell MTZ to ${SG} primitive-cell ASU ==="
ccp4-python supercell_collapse \
    P41212_super_gpu.mtz ${SG} super_mult=${MULT} dmin=${DMIN} \
    outfile=P41212_collapsed.mtz

echo ""
echo "=== Step 4: reference calculation with gemmi sfcalc on primitive cell ==="
gemmi sfcalc --dmin=${DMIN} --rate=2.5 --to-mtz=P41212_gemmi.mtz P41212.pdb

echo ""
echo "=== Step 5: compare collapsed GPU vs gemmi ==="
ccp4-python compare_mtz.py P41212_gemmi.mtz P41212_collapsed.mtz
