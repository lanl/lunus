#!/bin/bash
lunus=/home/jamesh/projects/lunus
conda deactivate 2> /dev/null
source ${lunus}/miniconda3/etc/profile.d/conda.sh
conda activate lunus
which python
export PATH=~/projects/lunus/lunus/c/bin/:$PATH

here=/home/jamesh/projects/lunus/lunus/md2mtz
tmpdir=/dev/shm/${USER}/lunus_compare_test_$$
mkdir -p $tmpdir
xtraj=${lunus}/lunus/lunus/command_line/xtraj.py

echo "=== GPU run (d_min=2.0, 3 frames) ==="
date
python $xtraj \
  top=${here}/amber_1000.pdb \
  traj=${here}/amber_1000.nc \
  first=0 last=2 d_min=2.0 \
  engine=gpu \
  lib=${here}/sfcalc_gpu.so \
  diffuse=${tmpdir}/diffuse_gpu.hkl \
  fcalc=${tmpdir}/fcalc_gpu.mtz \
  icalc=${tmpdir}/icalc_gpu.mtz
date

echo "=== CPU run (d_min=2.0, 3 frames) ==="
date
python $xtraj \
  top=${here}/amber_1000.pdb \
  traj=${here}/amber_1000.nc \
  first=0 last=2 d_min=2.0 \
  engine=cctbx \
  diffuse=${tmpdir}/diffuse_cpu.hkl \
  fcalc=${tmpdir}/fcalc_cpu.mtz \
  icalc=${tmpdir}/icalc_cpu.mtz
date

echo "=== CPU vs GPU diffuse correlation ==="
awk '
NR==FNR && NF==4 { cpu[$1" "$2" "$3] = $4; next }
NF==4 && ($1" "$2" "$3) in cpu {
    n++
    x = cpu[$1" "$2" "$3]; y = $4
    xsum += x; ysum += y
    x2sum += x*x; y2sum += y*y; xysum += x*y
}
END {
    if(n < 2) { print "too few common reflections"; exit }
    CC = (n*xysum - xsum*ysum) / sqrt((n*x2sum-xsum^2)*(n*y2sum-ysum^2))
    printf "Common reflections: %d\n", n
    printf "Pearson CC (CPU vs GPU diffuse): %.4f\n", CC
}' ${tmpdir}/diffuse_cpu.hkl ${tmpdir}/diffuse_gpu.hkl
