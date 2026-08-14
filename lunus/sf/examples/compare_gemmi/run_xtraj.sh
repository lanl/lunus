#!/bin/bash
# torch-vs-gemmi comparison for xtraj.py, on the 135,834-atom system that
# docs/performance.md's tables were measured on.
#
#   ./run_xtraj.sh gemmi
#   ./run_xtraj.sh torch                       # picks cuda / mps / cpu itself
#   LUNUS_TORCH_DEVICE=cpu ./run_xtraj.sh torch
#   mpirun -np 4 ./run_xtraj.sh torch          # one GPU per rank, if on cuda
#   python ../../tools/compare_icalc_mtz.py icalc_gemmi.mtz icalc_torch.mtz

set -eu

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SF_DIR=$(cd "${HERE}/../.." && pwd)
REPO_ROOT=$(cd "${SF_DIR}/../.." && pwd)

# sf is an ordinary subpackage, so the repo root goes on the path.
# A cctbx module build already provides it.
export PYTHONPATH=${PYTHONPATH:-}:${REPO_ROOT}

var=${1:-torch}

# Inputs are tracked. top_bfac.pdb.gz carries a random B per residue, made once
# with tools/make_random_bfacs.py --bmin 10 --bmax 100 --seed 0; iotbx.pdb and
# mdtraj both read .pdb.gz directly. use_top_bfacs=True reads those per-atom
# B-factors instead of applying one uniform B, exercising that path in BOTH
# engines (it also forces apply_bfac=False).
TOP="${HERE}/top_bfac.pdb.gz"
TRAJ="${HERE}/traj.xtc"

# P4(3), 88.451/88.451/39.823 is the crystal's real symmetry -- observed
# experimentally, and approximately obeyed by the simulation. It is passed on
# the command line only because the topology's CRYST1 records the P1 simulation
# BOX; the information is absent from the file, not fictitious. The deviations
# from it are what the diffuse signal is made of.
#
# d_min=0.9 (xtraj.py's default, explicit here because the benchmark numbers
# depend on it) gives the documented 300 x 300 x 144 grid. Use 1.8 for a quick
# coarse check.
UNIT_CELL=88.451,88.451,39.823,90.00,90.00,90.00
SPACE_GROUP=P43
D_MIN=0.9

for f in "${TOP}" "${TRAJ}"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done

ARGS="top=${TOP} use_top_bfacs=True traj=${TRAJ}"
ARGS="${ARGS} first=0 last=9 d_min=${D_MIN}"
ARGS="${ARGS} unit_cell=${UNIT_CELL} space_group=${SPACE_GROUP}"
ARGS="${ARGS} fcalc=fcalc_${var}.mtz icalc=icalc_${var}.mtz diffuse=diffuse_${var}.hkl"
ARGS="${ARGS} engine=${var}"

# torch.compile is off here because it currently FAILS on both platforms this
# has been run on -- Linux/CUDA has no cuda.h for triton to compile against in
# a driver-only container, and inductor's Metal backend is broken on MPS in
# torch 2.13. The fallback to eager is automatic and numerically identical, but
# the attempt itself costs ~9 s per run.
#
# It also buys nothing on a GPU even when it works: measured on an NVIDIA card,
# 5140e6 pairs/s compiled against 5126e6 eager, i.e. no difference. Fusing
# memory-bound passes mattered on CPU (~2.9x); a GPU does not need it.
#
# Set torch_compile=True to re-enable if you are on CPU, or once the headers
# are available.
if [ "${var}" = "torch" ]; then
  ARGS="${ARGS} torch_compile=False"
fi

# Useful extra flags:
#   torch_compile=False           skip the splat's ~1.5 s one-off compile
#                                 (worth it for a single-frame run only)
#   torch_max_pairs_per_batch=N   atom-voxel pair budget; re-tune on a new
#                                 machine with tools/bench_splat.py
#   save_density=True             dump frame 0's real-space grid for
#                                 tools/compare_density.py (221 MB at this grid)

# --- device selection --------------------------------------------------------
#
# Which device is available and how many MPI ranks there are are unrelated
# questions, so they are answered separately.
#
# The device is chosen by asking torch, which is the only authority that
# accounts for how it was built: a CUDA machine running an MPS build, or a
# CPU-only wheel, both report accurately. Override with
# LUNUS_TORCH_DEVICE=cpu|cuda|mps to force one.
#
# The MPI local rank is used ONLY to pin each rank to its own physical GPU,
# and only when the device is actually CUDA. With CUDA_VISIBLE_DEVICES
# restricted to one card, torch sees it as cuda:0 from that process, whatever
# its physical index. The LOCAL rank (rank within the node) is the right one --
# the global rank would break on multi-node runs where each node has only some
# of the GPUs. Other MPI flavours: MV2_COMM_WORLD_LOCAL_RANK, MPI_LOCALRANKID.

if [ "${var}" = "torch" ]; then
  if [ -n "${LUNUS_TORCH_DEVICE:-}" ]; then
    DEVICE=${LUNUS_TORCH_DEVICE}
  else
    DEVICE=$(python - <<'EOF'
import torch
if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
EOF
)
  fi
  echo "torch device: ${DEVICE}"
  ARGS="${ARGS} torch_device=${DEVICE}"

  if [ "${DEVICE}" = "cuda" ]; then
    LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-}}}
    if [ -n "${LOCAL_RANK}" ]; then
      export CUDA_VISIBLE_DEVICES=$LOCAL_RANK
      echo "rank-local GPU pinning: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
  fi
fi

exec python "${SF_DIR}/xtraj.py" ${ARGS}


# --- other configurations, kept for reference ------------------------------
#
# P4(1) on the tracked examples/xtraj/ data (52,224 atoms, cell halved so
# a == b). Needs no data of its own, so it runs from a bare clone:
#XTRAJ_DATA="${REPO_ROOT}/examples/xtraj"
#python ../../tools/make_random_bfacs.py "${XTRAJ_DATA}/top_ref.pdb" top_ref_bfac.pdb --bmin 10 --bmax 100 --seed 0
#exec python "${SF_DIR}/xtraj.py" top=top_ref_bfac.pdb use_top_bfacs=True traj=${XTRAJ_DATA}/traj_ref.xtc first=0 last=9 d_min=1.8 unit_cell=84.522,84.522,76.623,90.00,90.00,90.00 space_group=P41 fcalc=fcalc_${var}.mtz icalc=icalc_${var}.mtz diffuse=diffuse_${var}.hkl engine=${var} torch_device=mps
#
# P2(1)2(1)2(1), and runs against other systems:
#exec python "${SF_DIR}/xtraj.py" ${ARGS} torch_device=mps unit_cell=34.196,45.558,99.044,90.00,90.00,90.00 space_group=P212121
#mpirun -np 1 python ~/packages/lunus/lunus/command_line/xtraj_new.py top=top.pdb traj=traj.xtc first=0 last=1 unit_cell=88.451,88.451,39.823,90.00,90.00,90.00 space_group=P43 fcalc=fcalc_${var}.mtz icalc=icalc_${var}.mtz diffuse=diffuse_${var}.hkl engine=${var} selection="peptide"
#mpirun -np 26 python ~/packages/lunus/lunus/command_line/xtraj.py top=md_restrained_amber.pdb traj=md_restrained_last_10ns.xtc first=0 last=250 fcalc=fcalc_md_restrained_p1.mtz icalc=icalc_md_restrained_p1.mtz diffuse=diffuse_md_restrained_p1.hkl
#
# Preparing an Amber-compatible topology + trajectory from a GROMACS run:
#gmx editconf -f md_restrained.tpr -o md_restrained.pdb
#grep -v -e MODEL -e ENDMDL md_restrained.pdb > temp.pdb
#mv temp.pdb md_restrained.pdb
#pdb4amber -i md_restrained.pdb -o md_restrained_amber.pdb
#gmx trjconv -f md_restrained.xtc -s md_restrained.tpr -b 90000 -e 100000 -o md_restrained_last_10ns.xtc<<EOF
#0
#EOF
