#!/bin/bash
# Benchmark the diffuse backends (numpy / torch) at a given resolution.
# Usage: RES=1.6 BACKENDS="numpy|torch cuda|torch cpu" bash bench_backends.sh
#
# pinv.rcond is set explicitly (default 1e-14) so the Hessian pseudoinverse is
# deterministic and portable: the numpy default (~1e-15) sits on the softest
# surviving mode of the near-singular ANMEXP Hessian and blows up to negative
# V.diagonal on some LAPACK builds (e.g. the remote V100 box). 1e-14 truncates
# that precision-floor mode and gives the reproducible 0.238 aniso correlation.
cd "$(dirname "$0")"
LUNUS_BIN="$(cd ../../../../c/bin && pwd)"; export PATH="$LUNUS_BIN:$PATH"
EXP_LAT=snc_exafel_mean_mirror_sym_aniso_cull.lat
# The experimental lattice is stored gzipped in the repo; decompress on demand
# (mirrors calcenm.sh) so scoring works on a fresh checkout.
if [ ! -e "$EXP_LAT" ] && [ -e "$EXP_LAT.gz" ]; then
  gunzip -k "$EXP_LAT.gz"
fi
RES=${RES:-1.6}
RCOND=${RCOND:-1e-14}
# Backends to test, "|"-separated. On a CUDA box use "numpy|torch cuda|torch cpu".
BACKENDS=${BACKENDS:-"numpy|torch cuda|torch cpu"}
echo "=== resolution=$RES  pinv.rcond=$RCOND ==="
IFS='|' read -ra BKS <<< "$BACKENDS"
for BK in "${BKS[@]}"; do
  set -- $BK; backend=$1; dev=${2:-}
  extra="diffuse.backend=$backend"
  [ -n "$dev" ] && extra="$extra diffuse.device=$dev"
  tag="$backend${dev}"
  OMP_NUM_THREADS=2 python ../../xenm.py input.pdb=4wor.pdb output.hkl=tmp_b.hkl \
    anmexp.lambda=0.16 nproc=32 model=ANMEXP kpoints=1 resolution=$RES \
    pinv.rcond=$RCOND $extra > bench_$tag.log 2>&1
  t=$(grep "Diffuse calculation took" bench_$tag.log | awk '{print $4}')
  if [ ! -e tmp_b.hkl ]; then
    printf "%-14s FAILED (see bench_%s.log)\n" "$BK" "$tag"
    continue
  fi
  lunus.hkl2lat tmp_b.hkl tmp_b.lat "$EXP_LAT" >/dev/null 2>&1
  lunus.symlt tmp_b.lat tmp_b_s.lat >/dev/null 2>&1
  lunus.anisolt tmp_b_s.lat tmp_b_sub.lat 48.499,48.499,63.430,90.0,90.0,90.0 >/dev/null 2>&1
  c=$(lunus.corrlt tmp_b_sub.lat "$EXP_LAT" | tail -1)
  printf "%-14s diffuse=%ss  aniso_corr=%s\n" "$BK" "$t" "$c"
  rm -f tmp_b.hkl tmp_b.lat tmp_b_s.lat tmp_b_sub.lat
done
