#!/bin/bash
cd /home/jamesh/projects/fft_symmetry/claude_test
source /home/jamesh/projects/fft_symmetry/claude_test/setup_cuda.sh

gemmi sfcalc --dmin=1.5 --rate=2.5 --to-mtz=random_gemmi.mtz random.pdb
ccp4-python sfcalc_gpu.py random.pdb outmtz=random_gpu.mtz outmap= bmax=0
ccp4-python compare_mtz.py random_gemmi.mtz random_gpu.mtz
