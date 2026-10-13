#!/bin/bash
cd /home/jamesh/projects/fft_symmetry/claude_test
source /home/jamesh/projects/fft_symmetry/claude_test/setup_cuda.sh

nvcc -O3 -arch=sm_70 -shared -Xcompiler -fPIC -lcufft sfcalc_gpu.cu -o sfcalc_gpu.so 2>&1
if [ $? -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi
echo "BUILD OK"

bash run_p1pdb1_10k_test.sh
