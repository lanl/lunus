#!/bin/bash
# What the frame-loop optimizations are worth, measured honestly.
#
#     OMP_NUM_THREADS=3 ./tools/compare_baseline.sh
#     OMP_NUM_THREADS=3 ./tools/compare_baseline.sh --quick
#     OMP_NUM_THREADS=3 BASELINE=036d550 ./tools/compare_baseline.sh
#
# The page used to claim 6.5x, from 571.2 ms/frame to 87.3. That number was
# wrong twice over. It was measured on a cell carried over from another example,
# and -- worse -- the baseline ran with 104 torch threads on 3 CPUs while the
# final run did not, so most of it was a container setting rather than code:
#
#     571.2 -> 315.2   1.81x   the two host-side fixes, both runs throttled
#     315.2 ->  87.3   3.60x   setting OMP_NUM_THREADS
#
# What the code is worth in a HEALTHY environment has never been measured. This
# measures it: the same trajectory, cell, resolution, device and thread count,
# through the last commit before any optimization and through the current one.
#
# BASELINE defaults to 824b800, "Instrument the host half of the xtraj frame
# loop" -- the commit that added the phase table and changed nothing else, so
# both sides report the same breakdown. It predates:
#
#     a836f27  flex.vec3_double built from a flat buffer   (~190 ms/frame then)
#     7ef3f8c  the torch path skipping xray.structure      (~97 ms/frame then)
#
# It is checked out into a git worktree, so the branch you are on is untouched.

set -eu

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SF_DIR=$(cd "${HERE}/.." && pwd)
REPO_ROOT=$(cd "${SF_DIR}/../.." && pwd)
DATA="${SF_DIR}/examples/compare_gemmi"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

BASELINE=${BASELINE:-824b800}
WORKTREE=${WORKTREE:-/tmp/lunus-baseline-${BASELINE}}

CELL=${CELL:-34.196,45.558,99.044,90.00,90.00,90.00}
SPACE_GROUP=${SPACE_GROUP:-P212121}
D_MIN=${D_MIN:-0.9}
TRAJ=${TRAJ:-${DATA}/traj.xtc}
TOP=${TOP:-${DATA}/top_bfac.pdb.gz}
LAST=${LAST:-9}
[ "${QUICK}" = "1" ] && { D_MIN=2.5; LAST=3; }

if [ -z "${OMP_NUM_THREADS:-}" ]; then
  echo "OMP_NUM_THREADS is not set. Setting it is the whole point of this" >&2
  echo "comparison: leave it unset and you re-measure the container's CPU" >&2
  echo "quota instead of the code, which is what produced the 6.5x." >&2
  exit 1
fi

if [ -n "${PYTHON_CMD:-}" ]; then
  read -r -a PY <<< "${PYTHON_CMD}"
elif python -c "import torch" >/dev/null 2>&1; then
  PY=(python)
elif command -v pixi >/dev/null 2>&1; then
  PY=(pixi run python)
else
  echo "no python with torch, and no pixi to reach one" >&2; exit 1
fi

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

# ---- the baseline worktree --------------------------------------------------
if [ ! -d "${WORKTREE}" ]; then
  echo "checking out ${BASELINE} into ${WORKTREE}"
  git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE}" "${BASELINE}"
fi

LOGDIR="${DATA}/baseline-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOGDIR}"

cat <<MSG | tee "${LOGDIR}/config.txt"
baseline commit  ${BASELINE}  (worktree ${WORKTREE})
current          $(git -C "${REPO_ROOT}" rev-parse --short HEAD)
cell             ${CELL}   ${SPACE_GROUP}
d_min            ${D_MIN}
frames           0..${LAST} of $(basename "${TRAJ}")
device           ${DEVICE}
interpreter      ${PY[*]}
OMP_NUM_THREADS  ${OMP_NUM_THREADS}
MSG

common=(top="${TOP}" use_top_bfacs=True traj="${TRAJ}" first=0 last="${LAST}"
        d_min="${D_MIN}" unit_cell="${CELL}" space_group="${SPACE_GROUP}"
        engine=torch torch_compile=False torch_device="${DEVICE}"
        torch_timing=True)

# The baseline has no expand_symmetry and always expands, so the current run is
# told to expand too. Both then compute the same thing. It costs ~0 ms either
# way -- symmetrize measures 0.0 ms/frame on this grid -- but a comparison
# where the two sides calculate different quantities is not one.
echo; echo "=== baseline: ${BASELINE} ==="
( cd "${WORKTREE}/lunus/sf" && PYTHONPATH="${WORKTREE}" "${PY[@]}" xtraj.py \
    "${common[@]}" fcalc="${LOGDIR}/f_before.mtz" icalc="${LOGDIR}/i_before.mtz" \
    diffuse="${LOGDIR}/d_before.hkl" ) 2>&1 | tee "${LOGDIR}/before.log" \
  | grep -E "processed chunk|^  [a-z].*[0-9]|TOTAL|frame loop|unaccounted" || true

echo; echo "=== current ==="
( cd "${SF_DIR}" && PYTHONPATH="${REPO_ROOT}" "${PY[@]}" xtraj.py \
    "${common[@]}" expand_symmetry=True torch_splat_stats=True \
    fcalc="${LOGDIR}/f_after.mtz" icalc="${LOGDIR}/i_after.mtz" \
    diffuse="${LOGDIR}/d_after.hkl" ) 2>&1 | tee "${LOGDIR}/after.log" \
  | grep -E "processed chunk|^  [a-z].*[0-9]|TOTAL|frame loop|unaccounted|per frame" || true

# ---- the comparison ---------------------------------------------------------
"${PY[@]}" - "${LOGDIR}" <<'EOF'
import re, sys, pathlib
log = pathlib.Path(sys.argv[1])

def budget(path):
    """phase -> ms/frame, plus the frame-loop wall, from a phase table."""
    rows, wall, frames = {}, None, None
    for line in path.read_text().splitlines():
        m = re.match(r"=== torch engine phase timing \(rank 0\), (\d+) frames", line)
        if m:
            frames = int(m.group(1))
        m = re.match(r"\s{2}([a-z][a-z ._+>-]*?)\s{2,}[\d.]+\s+([\d.]+)\s", line)
        if m and frames:
            rows[m.group(1).strip()] = float(m.group(2))
        m = re.match(r"\s{2}frame loop wall\s+[\d.]+\s+([\d.]+)", line)
        if m:
            wall = float(m.group(1))
    rows.pop("unaccounted", None)          # a residual, not a phase
    return rows, wall, frames

before, wall_b, n = budget(log / "before.log")
after, wall_a, _ = budget(log / "after.log")
if not (wall_b and wall_a):
    print("\ncould not parse both phase tables; read the logs in", log)
    raise SystemExit(0)

# Every phase from BOTH sides. Keeping only the intersection would drop
# exactly the rows the optimization deleted -- set_sites_cart, xrs.select,
# sites_frac+occ, element idx -- which is to say the entire explanation for
# whatever the frame loop saved.
names = list(before) + [k for k in after if k not in before]
print("\n=== ms/frame, %d frames, same cell / device / threads ===" % n)
print("  %-22s %10s %10s %10s" % ("phase", "before", "after", "saved"))
for k in names:
    b, a = before.get(k, 0.0), after.get(k, 0.0)
    print("  %-22s %10.1f %10.1f %10.1f" % (k, b, a, b - a))
print("  %-22s %10.1f %10.1f %10.1f" % ("FRAME LOOP", wall_b, wall_a, wall_b - wall_a))
print("\n  speedup %.2fx  (code only: same threads, same cell, both sides)"
      % (wall_b / wall_a))
print("""
Read it against the retired claim: 6.5x was 1.81x of code measured under
throttling times 3.60x of setting OMP_NUM_THREADS. The number above is the
code's contribution with the container already healthy, which is the only
version of it worth publishing.""")
EOF

echo
echo "logs in ${LOGDIR}"
echo "remove the worktree with: git -C ${REPO_ROOT} worktree remove ${WORKTREE}"
