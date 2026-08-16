#!/bin/bash
# Re-measure everything docs/performance.md publishes, on the corrected cell.
#
#     OMP_NUM_THREADS=3 ./tools/remeasure.sh                 # full, ~30 min
#     OMP_NUM_THREADS=3 ./tools/remeasure.sh --quick         # smoke test, ~2 min
#     OMP_NUM_THREADS=3 TRAJ=md_restrained_last_10ns.xtc LAST=250 ./tools/remeasure.sh
#
# The interpreter is found rather than assumed: where the python on PATH cannot
# import torch and pixi is installed -- which is the pod -- every call goes
# through `pixi run python`. Override with PYTHON_CMD="pixi run python" or
# PYTHON_CMD=/path/to/python if the guess is wrong.
#
# WHY THIS EXISTS. Every absolute number in docs/performance.md was measured
# with unit_cell=88.451,88.451,39.823 and space_group=P43, carried over from a
# different example. The crystal is 7FPV -- 34.196 x 45.558 x 99.044,
# P 21 21 21 -- and the simulation box is exactly a 2x2x2 supercell of it. Two
# things changed as a result and each invalidates a different set of numbers:
#
#   the CELL      moves the grid (120x160x360 = 6.9M voxels against
#                 300x300x144 = 12.96M), so every ms/frame figure goes, and
#                 none of them can be rescaled by hand because the atom-voxel
#                 pair count moves with the grid too;
#   expand_symmetry
#                 now defaults to False, so the structure factors themselves
#                 change -- applying the operations to a model that already
#                 holds the full cell contents symmetry-AVERAGES it rather
#                 than completing it. Measured: Icalc x15, diffuse/Icalc down
#                 3.4x. Numbers from before and after are different
#                 calculations, not the same calculation on a different grid.
#
# Four runs, in the order their results are needed. Each writes a log; the
# summary at the end points at the section of docs/performance.md it replaces.

set -eu

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SF_DIR=$(cd "${HERE}/.." && pwd)
REPO_ROOT=$(cd "${SF_DIR}/../.." && pwd)
DATA="${SF_DIR}/examples/compare_gemmi"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# ---- the interpreter --------------------------------------------------------
# An array, not a string: `pixi run python` is three words and has to survive
# being passed as a command.
if [ -n "${PYTHON_CMD:-}" ]; then
  read -r -a PY <<< "${PYTHON_CMD}"
elif python -c "import torch" >/dev/null 2>&1; then
  PY=(python)
elif command -v pixi >/dev/null 2>&1; then
  PY=(pixi run python)
else
  echo "no python with torch on PATH, and no pixi to reach one." >&2
  echo "Set PYTHON_CMD to the interpreter to use." >&2
  exit 1
fi

# ---- the corrected configuration -------------------------------------------
CELL=${CELL:-34.196,45.558,99.044,90.00,90.00,90.00}
SPACE_GROUP=${SPACE_GROUP:-P212121}
D_MIN=${D_MIN:-0.9}
EXPAND=${EXPAND_SYMMETRY:-False}
TOP=${TOP:-${DATA}/top_bfac.pdb.gz}
# The tracked traj.xtc holds 10 frames. The 251-frame budget in the docs came
# from md_restrained_last_10ns.xtc, which lives only on the machine that ran
# it; pass TRAJ= and LAST= to use it.
TRAJ=${TRAJ:-${DATA}/traj.xtc}
LAST=${LAST:-9}

if [ "${QUICK}" = "1" ]; then
  D_MIN=2.5
  LAST=1
fi

# ---- the two things most likely to invalidate the whole run -----------------
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  cat >&2 <<'MSG'
OMP_NUM_THREADS is not set. On a CPU-limited container torch sizes its thread
pool from the HOST's core count -- 104 threads on 3 CPUs, measured -- and the
spinning threads exhaust the cgroup quota, freezing the process for most of
every 100 ms period. That cost 4.3x on the splat and made every timing here a
measurement of the quota rather than of the code. Set it to the pod's real CPU
count (`cat /sys/fs/cgroup/cpu.max`) and run again.
MSG
  exit 1
fi

for f in "${TOP}" "${TRAJ}"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done

# Device, asked of torch rather than assumed -- a CUDA machine running an MPS
# build reports accurately, a CPU-only wheel likewise.
if [ -n "${LUNUS_TORCH_DEVICE:-}" ]; then
  DEVICE=${LUNUS_TORCH_DEVICE}
else
  DEVICE=$("${PY[@]}" - <<'EOF'
import torch
print("cuda" if torch.cuda.is_available()
      else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
      else "cpu")
EOF
)
fi

GRID=$("${PY[@]}" - "$CELL" "$D_MIN" <<'EOF'
import sys
from lunus.sf.cell_utils import grid_shape_for_resolution
a, b, c = [float(x) for x in sys.argv[1].split(",")[:3]]
print(",".join(str(n) for n in grid_shape_for_resolution(a, b, c, float(sys.argv[2]))))
EOF
)

LOGDIR="${DATA}/remeasure-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOGDIR}"

cat <<MSG | tee "${LOGDIR}/config.txt"
cell             ${CELL}
space group      ${SPACE_GROUP}
d_min            ${D_MIN}  ->  grid ${GRID}
expand_symmetry  ${EXPAND}
trajectory       $(basename "${TRAJ}"), frames 0..${LAST}
device           ${DEVICE}
interpreter      ${PY[*]}
OMP_NUM_THREADS  ${OMP_NUM_THREADS}
logs             ${LOGDIR}
MSG
echo

step() { echo; echo "=== $* ==="; }

# ---- 1. the splat, which is most of the frame -------------------------------
step "1/4  splat benchmark  -> replaces the CPU / MPS / CUDA tables"
"${PY[@]}" "${HERE}/bench_splat.py" "${TOP}" \
    --cell "${CELL}" --grid "${GRID}" --device "${DEVICE}" --no-gemmi \
    2>&1 | tee "${LOGDIR}/1-bench_splat.log"

# ---- 2. where the frame goes ------------------------------------------------
step "2/4  frame budget  -> replaces 'Where the time actually goes'"
"${PY[@]}" "${SF_DIR}/xtraj.py" \
    top="${TOP}" use_top_bfacs=True traj="${TRAJ}" first=0 last="${LAST}" \
    d_min="${D_MIN}" unit_cell="${CELL}" space_group="${SPACE_GROUP}" \
    expand_symmetry="${EXPAND}" engine=torch torch_compile=False \
    torch_device="${DEVICE}" torch_timing=True torch_splat_stats=True \
    fcalc="${LOGDIR}/f_budget.mtz" icalc="${LOGDIR}/i_budget.mtz" \
    diffuse="${LOGDIR}/d_budget.hkl" \
    2>&1 | tee "${LOGDIR}/2-frame_budget.log"

# ---- 3. do the engines still agree? -----------------------------------------
# The first full-resolution check under expand_symmetry=False. Parity is the
# headline correctness claim, and it is the one result the cell change should
# NOT have moved -- both engines fold identically, whatever cell they are given.
step "3/4  engine parity  -> replaces the gemmi-vs-torch correlation / R"
for eng in gemmi torch; do
  "${PY[@]}" "${SF_DIR}/xtraj.py" \
      top="${TOP}" use_top_bfacs=True traj="${TRAJ}" first=0 last="${LAST}" \
      d_min="${D_MIN}" unit_cell="${CELL}" space_group="${SPACE_GROUP}" \
      expand_symmetry="${EXPAND}" engine="${eng}" gemmi_cutoff=0.01 \
      torch_compile=False torch_device="${DEVICE}" \
      fcalc="${LOGDIR}/fcalc_${eng}.mtz" icalc="${LOGDIR}/icalc_${eng}.mtz" \
      diffuse="${LOGDIR}/diffuse_${eng}.hkl" \
      > "${LOGDIR}/3-engine_${eng}.log" 2>&1
  echo "  ${eng} done"
done
"${PY[@]}" "${HERE}/compare_icalc_mtz.py" \
    "${LOGDIR}/icalc_gemmi.mtz" "${LOGDIR}/icalc_torch.mtz" \
    2>&1 | tee "${LOGDIR}/3-parity.log"

# ---- 4. what bulk solvent costs ---------------------------------------------
#
# On 7FPV itself, NOT on the trajectory. The MD model is all-atom with explicit
# water, so its production path is all atoms and no mask -- which is what runs
# 1-3 above measure. Forcing a mask onto it by selecting the peptide (as an
# earlier version of this script did) benchmarks a configuration nobody should
# use: the mask is for a crystallographic model, where the disordered solvent
# is absent from the coordinates rather than explicitly simulated.
step "4/4  solvent cost, on the crystal structure  -> 'At production scale'"
"${PY[@]}" "${HERE}/bench_solvent.py" "${DATA}/7FPV.pdb" \
    --cell "${CELL}" --space-group "${SPACE_GROUP}" --grid "${GRID}" \
    --device "${DEVICE}" --selection "not water" \
    2>&1 | tee "${LOGDIR}/4-bench_solvent.log"

# ---- summary ----------------------------------------------------------------
cat <<MSG

=== done ===
Logs in ${LOGDIR}

Into docs/performance.md:
  1-bench_splat.log     the CPU / Apple GPU / NVIDIA GPU tables, and the
                        atom-voxel pair count in the configuration block
  2-frame_budget.log    'Where the time actually goes'. Read the median
                        column rather than the mean: one-time costs divide by
                        the frame count, so a short run inflates the mean. For
                        a speedup figure use tools/compare_baseline.sh, which
                        runs both sides rather than comparing against numbers
                        from another configuration
  3-parity.log          the gemmi-vs-torch correlation and R-factor
  4-bench_solvent.log   'At production scale'. Note this one runs on 7FPV
                        rather than on the trajectory: the MD model carries
                        its solvent explicitly, so a mask has nothing to add
                        to it, and runs 1-3 are the all-atom production path

Then delete the "configuration above is wrong for this system" note at the top
of docs/performance.md, since it will no longer be true.
MSG
