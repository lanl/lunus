#!/bin/bash
# Test sfcalc_gpu.py against gemmi sfcalc on a 1000-atom PDB with random B factors.
#
# Prerequisites:
#   - sfcalc_gpu.so built via compile_gpu.sh
#   - ccp4-python (CCP4 suite) in PATH  --> provides gemmi
#   - GPU with CUDA support

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f sfcalc_gpu.so ]; then
    echo "ERROR: sfcalc_gpu.so not found. Run compile_gpu.sh first."
    exit 1
fi

echo "=== Step 1: generate 1000-atom PDB with random B factors ==="
ccp4-python make_randb_pdb.py

echo ""
echo "=== Step 2: reference calculation with gemmi sfcalc ==="
gemmi sfcalc --dmin=1.5 --rate=2.5 --to-mtz=P1test_randB_gemmi.mtz P1test_randB.pdb

echo ""
echo "=== Step 3: GPU calculation with sfcalc_gpu.py ==="
ccp4-python sfcalc_gpu.py P1test_randB.pdb outmtz=P1test_randB_gpu.mtz outmap= bmax=0

echo ""
echo "=== Step 4: compare results ==="
ccp4-python compare_mtz.py P1test_randB_gemmi.mtz P1test_randB_gpu.mtz
