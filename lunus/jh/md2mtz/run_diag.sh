#!/bin/bash
cd /home/jamesh/projects/fft_symmetry/claude_test
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64
ccp4-python diag_axes.py
