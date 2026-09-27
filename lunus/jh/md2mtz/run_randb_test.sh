#!/bin/bash
cd /home/jamesh/projects/fft_symmetry/claude_test
source /home/jamesh/projects/fft_symmetry/claude_test/setup_cuda.sh

ccp4-python make_randb_pdb.py
gemmi sfcalc --dmin=1.5 --rate=2.5 --to-mtz=P1test_randB_gemmi.mtz P1test_randB.pdb
ccp4-python sfcalc_gpu.py P1test_randB.pdb outmtz=P1test_randB_gpu.mtz outmap= bmax=0
ccp4-python compare_mtz.py P1test_randB_gemmi.mtz P1test_randB_gpu.mtz
