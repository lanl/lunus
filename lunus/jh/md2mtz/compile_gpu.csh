#!/bin/tcsh -f
# Compile sfcalc_gpu.cu into a shared library on voltron
# Run this on voltron (not via srun -- compilation is CPU-only)

source /programs/cuda/setup_cuda.csh

cd /home/jamesh/projects/fft_symmetry/claude_test

echo "Compiling sfcalc_gpu.cu ..."
nvcc -O3 \
     -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     -shared -Xcompiler -fPIC \
     -Xlinker -rpath -Xlinker '$ORIGIN' \
     -lcufft \
     sfcalc_gpu.cu \
     -o sfcalc_gpu.so

# Copy the versioned cuFFT so alongside sfcalc_gpu.so.
# When distributing, ship both files together; $ORIGIN rpath lets
# the loader find libcufft.so.11 in the same directory without
# needing the CUDA toolkit installed on the target machine.
echo "Copying libcufft.so.11 ..."
cp /usr/local/cuda/lib64/libcufft.so.11 .

if ( $status == 0 ) then
    echo "OK: sfcalc_gpu.so"
    ls -lh sfcalc_gpu.so
else
    echo "FAILED"
    exit 1
endif

echo "Compiling sfcalc_gpu_collapse.cpp ..."
/opt/rh/devtoolset-7/root/usr/bin/g++ -O2 -std=c++14 \
    -I./include \
    sfcalc_gpu_collapse.cpp \
    -ldl -lm \
    -o sfcalc_gpu_collapse

if ( $status == 0 ) then
    echo "OK: sfcalc_gpu_collapse"
    ls -lh sfcalc_gpu_collapse
else
    echo "FAILED"
    exit 1
endif
