#!/bin/tcsh -f
source /programs/cuda/setup_cuda.csh
cd /home/jamesh/projects/fft_symmetry/claude_test
ccp4-python diag_axes.py
