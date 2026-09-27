#!/bin/bash
# Compile sfcalc_gpu.cu into sfcalc_gpu.so
# Requires: nvcc (CUDA toolkit), cufft library
# Tested with CUDA 11/12, sm_70 (Volta) and sm_80 (Ampere)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect GPU architecture (falls back to sm_70 if nvidia-smi not available)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "$ARCH" ]; then
    echo "Warning: could not detect GPU arch, using sm_70"
    ARCH=70
fi

echo "Compiling sfcalc_gpu.cu for sm_${ARCH} ..."
nvcc -O3 -arch=sm_${ARCH} \
     -shared -Xcompiler -fPIC \
     -lcufft \
     sfcalc_gpu.cu \
     -o sfcalc_gpu.so

echo "OK: sfcalc_gpu.so"
ls -lh sfcalc_gpu.so
