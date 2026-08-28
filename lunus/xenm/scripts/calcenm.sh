#!/bin/bash
# Calculate ENM and compute correlation with experimental data
# Usage: calcenm.sh <pdb_file> <lambda_value>
#
# This script runs xenm.py to calculate diffuse scattering from an elastic
# network model, then computes the correlation with experimental data.

# Find xenm.py relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XENM_DIR="$(dirname "$SCRIPT_DIR")"
XENM_PY="$XENM_DIR/xenm.py"

# Ensure the lunus C lattice tools (lunus.hkl2lat, lunus.symlt, etc.) are on PATH.
# Repo layout: <root>/lunus/xenm/scripts -> <root>/c/bin
LUNUS_BIN="$(cd "$XENM_DIR/../../c/bin" 2>/dev/null && pwd)"
if [ -n "$LUNUS_BIN" ]; then
  export PATH="$LUNUS_BIN:$PATH"
fi

EXP_LAT=snc_exafel_mean_mirror_sym_aniso_cull.lat

# The experimental lattice is stored gzipped in the repo; decompress on demand.
if [ ! -e "$EXP_LAT" ] && [ -e "$EXP_LAT.gz" ]; then
  gunzip -k "$EXP_LAT.gz"
fi

# Resolution cutoff for the diffuse calculation (Angstroms). Lower = finer grid,
# slower. 4.0 is a fast setting for testing; use ~1.6 to match the full
# experimental grid.
RES=${RES:-4.0}

OMP_NUM_THREADS=2 python "$XENM_PY" input.pdb=$1 output.hkl=tmp.hkl anmexp.lambda=$2 nproc=32 model=ANMEXP kpoints=1 resolution=$RES > /dev/null
if [ -e tmp.hkl ]; then
lunus.hkl2lat tmp.hkl tmp.lat "$EXP_LAT" >/dev/null
lunus.symlt tmp.lat tmp_sym.lat 1>/dev/null
lunus.anisolt tmp_sym.lat tmp_sym_sub.lat 48.499,48.499,63.430,90.0,90.0,90.0>/dev/null
corr_aniso=`lunus.corrlt tmp_sym_sub.lat "$EXP_LAT" | tail -1`
echo "0.0 $corr_aniso"
else
echo "0.0 0.0"
fi
if [ -e tmp.hkl ]; then
rm tmp.hkl 
fi
if [ -e tmp.lat ]; then
rm tmp.lat
fi
