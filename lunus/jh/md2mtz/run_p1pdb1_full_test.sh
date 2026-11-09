#!/bin/bash
cd /home/jamesh/projects/fft_symmetry/claude_test
source /home/jamesh/projects/fft_symmetry/claude_test/setup_cuda.sh

if [ ! -f P1pdb1_gemmi.mtz ]; then
    gemmi sfcalc --dmin=1.5 --rate=2.5 --to-mtz=P1pdb1_gemmi.mtz P1pdb1.pdb
fi
ccp4-python sfcalc_gpu.py P1pdb1.pdb outmtz=P1pdb1_gpu.mtz outmap= bmax=0
ccp4-python compare_mtz.py P1pdb1_gemmi.mtz P1pdb1_gpu.mtz
