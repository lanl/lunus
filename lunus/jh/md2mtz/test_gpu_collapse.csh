#! /bin/tcsh -f
#
# Test sfcalc_gpu_collapse.py against gemmi sfcalc
# Analogous to fixthis.com but using the GPU program and an orthogonal cell.
#

set SG = P212121

set CELL = ( 10 20 30 90 90 90 )
set superCELL = `echo $CELL | awk '{print 2*$1,2*$2,2*$3,$4,$5,$6}'`

echo "SG = $SG"
echo "primitive CELL = $CELL"
echo "supercell CELL = $superCELL"

randompdb.com $superCELL

pdbset xyzin random.pdb xyzout ASU.pdb << EOF
space $SG
CELL $CELL
EOF

# GPU sfcalc on supercell + collapse to primitive-cell ASU
ccp4-python sfcalc_gpu_collapse.py random.pdb sg=$SG super_mult=2,2,2 dmin=2 \
    outmtz=gpu_collapsed.mtz outI=gpu_supercell_I.mtz

# Reference: gemmi sfcalc on the ASU structure
gemmi sfcalc --to-mtz ASU.mtz --dmin=2 ASU.pdb

# Compare collapsed vs reference
echo ""
echo "=== Phased ASU MTZ: GPU collapse vs gemmi ==="
ccp4-python compare_mtz.py ASU.mtz gpu_collapsed.mtz
