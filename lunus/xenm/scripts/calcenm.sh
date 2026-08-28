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

# Model selection. Default is the single-tier ANMEXP (backward compatible: two
# positional args <pdb> <lambda>). Set MODEL=ANMEXP_DOMAIN to use the two-tier
# domain-coupled model; the inter-molecular logistic springs are then controlled
# by the environment variables INTER_THRESH, INTER_K and INTER_WIDTH.
MODEL=${MODEL:-ANMEXP}
INTER_THRESH=${INTER_THRESH:-10.0}
INTER_K=${INTER_K:-1.0}
INTER_WIDTH=${INTER_WIDTH:-1.5}

# Backbone-enhanced ENM factor (BENM, Ming & Wall 2005). w_benm multiplies the
# force constant of sequence-adjacent backbone (CA-CA) bonds. Default 1.0 = off
# (plain model); ~40 reproduces the all-atom density of states for an ANM.
W_BENM=${W_BENM:-1.0}

# Covariance shrinkage toward the diagonal. Off-diagonal covariance elements are
# scaled by COV_SHRINK after renormalization (diagonal preserved). Default 1.0 =
# off (raw model). Set COV_SHRINK=auto to derive the minimal shrinkage from the
# matrix itself (from its most-negative correlation eigenvalue), restoring
# positive definiteness with no user-supplied value. A numeric value (e.g. 0.95)
# still works as a manual override.
COV_SHRINK=${COV_SHRINK:-1.0}

if [ "$MODEL" = "ANMEXP_DOMAIN" ]; then
  OMP_NUM_THREADS=2 python "$XENM_PY" input.pdb=$1 output.hkl=tmp.hkl anmexp.lambda=$2 \
    inter.thresh=$INTER_THRESH inter.k=$INTER_K inter.width=$INTER_WIDTH \
    backbone.enhancement=$W_BENM covariance.shrinkage=$COV_SHRINK \
    nproc=32 model=ANMEXP_DOMAIN kpoints=1 resolution=$RES > /dev/null
else
  OMP_NUM_THREADS=2 python "$XENM_PY" input.pdb=$1 output.hkl=tmp.hkl anmexp.lambda=$2 \
    backbone.enhancement=$W_BENM covariance.shrinkage=$COV_SHRINK \
    nproc=32 model=$MODEL kpoints=1 resolution=$RES > /dev/null
fi
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
