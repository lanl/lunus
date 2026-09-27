#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/lib64
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/usr/local/cuda
