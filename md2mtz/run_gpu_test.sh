#!/bin/bash
lunus=/home/jamesh/projects/lunus
conda deactivate 2> /dev/null
source ${lunus}/miniconda3/etc/profile.d/conda.sh
conda activate lunus
which python
export PATH=~/projects/lunus/lunus/c/bin/:$PATH

here=/home/jamesh/projects/lunus/lunus/md2mtz
tmpdir=/dev/shm/${USER}/lunus_gpu_test_$$
mkdir -p $tmpdir

python ${lunus}/lunus/lunus/command_line/xtraj.py \
  top=${here}/amber_1000.pdb \
  traj=${here}/amber_1000.nc \
  first=0 last=4 \
  engine=gpu \
  lib=${here}/sfcalc_gpu.so \
  diffuse=${tmpdir}/diffuse_gpu.hkl \
  fcalc=${tmpdir}/fcalc_gpu.mtz \
  icalc=${tmpdir}/icalc_gpu.mtz

echo "=== output ==="
ls -lh ${tmpdir}/
echo "=== first 5 lines of diffuse ==="
head -5 ${tmpdir}/diffuse_gpu.hkl
echo "=== NaN/Inf check ==="
awk 'NF==4 && ($4!=$4 || $4+0!=int($4+0)*0+$4) {print "BAD:",$0}' ${tmpdir}/diffuse_gpu.hkl | head -5
echo "done"
