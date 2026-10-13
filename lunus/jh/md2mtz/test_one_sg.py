#!/usr/bin/env ccp4-python
"""
test_one_sg.py
==============
Test sfcalc_gpu_collapse.py vs gemmi sfcalc for one space group.

Usage:
  ccp4-python test_one_sg.py <sg_name_or_number> [dmin=2.0] [workdir=sg_tests]

Exits 0 on PASS, 1 on FAIL.
Prints one summary line: PASS/FAIL sgnum sgname  common/total  max_reldiff
"""

import sys
import os
import math
import subprocess
import tempfile
import shutil
import numpy as np
import gemmi


# ---------------------------------------------------------------------------
# Cell parameters appropriate for each crystal system
# ---------------------------------------------------------------------------

CELLS = {
    'triclinic':    (15., 20., 25.,  80.,  85.,  95.),
    'monoclinic':   (15., 20., 25.,  90., 110.,  90.),
    'orthorhombic': (15., 20., 25.,  90.,  90.,  90.),
    'tetragonal':   (20., 20., 30.,  90.,  90.,  90.),
    'trigonal':     (20., 20., 30.,  90.,  90., 120.),   # hexagonal setting
    'hexagonal':    (20., 20., 30.,  90.,  90., 120.),
    'cubic':        (20., 20., 20.,  90.,  90.,  90.),
}


def cell_for_sg(sg):
    """Return (a,b,c,alpha,beta,gamma) consistent with the space group constraints."""
    cs = sg.crystal_system_str().lower()
    return CELLS.get(cs, (20., 20., 20., 90., 90., 90.))


def super_mult_for_sg(sg):
    """Return (na,nb,nc) supercell multipliers.
    R-centering requires multiples of 3; all others work with 2.
    """
    # centering letter is the first char of the short HM symbol
    centering = sg.hm.split()[0][0]
    if centering == 'R':
        return (3, 3, 3)
    else:
        return (2, 2, 2)


def run(cmd, cwd, stdin=None):
    """Run cmd in cwd; return (returncode, combined_output)."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        input=stdin,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return result.returncode, result.stdout


def compare_mtz(path1, path2):
    """
    Compare FC amplitudes in two MTZ files by HKL matching.
    Returns (n_common, n_only1, n_only2, mean_rel, median_rel, max_rel, hkl_max).
    Both files must have an 'FC' column.
    """
    def load(path):
        mtz  = gemmi.read_mtz_file(path)
        cols = [col.label for col in mtz.columns]
        arr  = np.array(mtz.array)
        hi   = cols.index('H'); ki = cols.index('K'); li = cols.index('L')
        fi   = cols.index('FC')
        H = arr[:, hi].astype(int)
        K = arr[:, ki].astype(int)
        L = arr[:, li].astype(int)
        F = arr[:, fi]
        # normalise to H>0 half-space
        flip  = H < 0
        H[flip] = -H[flip]; K[flip] = -K[flip]; L[flip] = -L[flip]
        flip2 = (H == 0) & (K < 0)
        K[flip2] = -K[flip2]; L[flip2] = -L[flip2]
        return {(int(h), int(k), int(l)): float(f)
                for h, k, l, f in zip(H, K, L, F)}

    d1 = load(path1)
    d2 = load(path2)
    common = set(d1) & set(d2)
    only1  = len(d1) - len(common)
    only2  = len(d2) - len(common)
    if not common:
        return 0, only1, only2, float('nan'), float('nan'), float('nan'), None
    diffs = np.array(
        [abs(d1[hkl] - d2[hkl]) / (max(d1[hkl], d2[hkl]) + 1e-6)
         for hkl in common])
    imax = diffs.argmax()
    hkl_list = list(common)
    return (len(common), only1, only2,
            float(diffs.mean()), float(np.median(diffs)),
            float(diffs[imax]), hkl_list[imax])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit(__doc__)

    sg_arg  = argv[0]
    dmin    = 2.0
    workdir = 'sg_tests'
    for a in argv[1:]:
        if a.startswith('dmin='):   dmin    = float(a.split('=', 1)[1])
        if a.startswith('workdir='): workdir = a.split('=', 1)[1]

    # Resolve space group
    try:
        sg_num = int(sg_arg)
        sg = gemmi.find_spacegroup_by_number(sg_num)
    except ValueError:
        sg = gemmi.find_spacegroup_by_name(sg_arg)
    if sg is None:
        sys.exit(f"ERROR: unknown space group '{sg_arg}'")

    sg_tag  = f"{sg.number:03d}_{sg.hm.replace(' ', '_').replace('/', '-')}"
    wdir    = os.path.join(workdir, sg_tag)
    os.makedirs(wdir, exist_ok=True)

    a, b, c, alpha, beta, gamma = cell_for_sg(sg)
    na, nb, nc = super_mult_for_sg(sg)
    sa, sb, sc = na * a, nb * b, nc * c

    prefix = f"[{sg.number:3d} {sg.hm}]"
    sg_hm  = sg.hm             # gemmi HM name for sfcalc_gpu_collapse.py

    log_lines = []

    def note(msg):
        log_lines.append(msg)

    # --- Step 1: generate random supercell PDB ---
    cell_str  = f"{sa:.3f} {sb:.3f} {sc:.3f} {alpha:.2f} {beta:.2f} {gamma:.2f}"
    rc, out = run(f'randompdb.com {cell_str}', cwd=wdir)
    if rc != 0:
        note(out[-500:])
        print(f"FAIL  {prefix}  randompdb.com failed")
        return 1

    # --- Step 2: patch CRYST1 to use primitive cell + correct SG ---
    # (avoids pdbset space-group name lookup issues)
    _patch_cryst1(os.path.join(wdir, 'random.pdb'),
                  os.path.join(wdir, 'ASU.pdb'),
                  a, b, c, alpha, beta, gamma, sg.hm)

    # --- Step 3: GPU sfcalc + collapse ---
    collapse_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'sfcalc_gpu_collapse.py')
    mult_str = f"{na},{nb},{nc}"
    rc, out = run(
        f'ccp4-python {collapse_py} random.pdb sg="{sg_hm}" '
        f'super_mult={mult_str} dmin={dmin} '
        f'outmtz=gpu_collapsed.mtz outI=gpu_supercell_I.mtz',
        cwd=wdir)
    note(out)
    if rc != 0:
        print(f"FAIL  {prefix}  sfcalc_gpu_collapse failed")
        _write_log(wdir, log_lines)
        return 1

    # --- Step 4: gemmi sfcalc reference ---
    rc, out = run(f'gemmi sfcalc --to-mtz ASU.mtz --dmin={dmin} ASU.pdb',
                  cwd=wdir)
    note(out)
    if rc != 0 or not os.path.exists(os.path.join(wdir, 'ASU.mtz')):
        print(f"FAIL  {prefix}  gemmi sfcalc failed")
        _write_log(wdir, log_lines)
        return 1

    # --- Step 5: compare ---
    p1 = os.path.join(wdir, 'ASU.mtz')
    p2 = os.path.join(wdir, 'gpu_collapsed.mtz')
    n_common, n_only1, n_only2, mean_r, med_r, max_r, hkl_max = compare_mtz(p1, p2)

    n_gemmi = n_common + n_only1
    frac    = n_common / max(n_gemmi, 1)

    # Pass criteria:
    #   - at least 90% of gemmi reflections are found in GPU output
    #   - mean relative difference < 0.5%
    #   - max  relative difference < 5%
    passed = (frac >= 0.90 and
              (math.isnan(mean_r) or mean_r < 0.005) and
              (math.isnan(max_r)  or max_r  < 0.05))

    verdict = 'PASS' if passed else 'FAIL'
    hkl_str = str(hkl_max) if hkl_max else 'N/A'
    print(f"{verdict}  {prefix}  "
          f"common={n_common}/{n_gemmi}({frac:.0%})  "
          f"mean={mean_r:.4f}  max={max_r:.4f}@{hkl_str}")
    sys.stdout.flush()

    _write_log(wdir, log_lines)
    return 0 if passed else 1


def _patch_cryst1(pdb_in, pdb_out, a, b, c, alpha, beta, gamma, sg_hm):
    """Copy pdb_in to pdb_out, replacing the CRYST1 line with the given cell/SG.
    SCALE1/2/3 records are dropped so gemmi recomputes them from CRYST1;
    keeping the supercell SCALE in a primitive-cell PDB causes wrong fractional
    coordinates and hence wrong structure factors."""
    cryst1 = (f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
              f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {sg_hm:<11s}   1\n")
    with open(pdb_in) as fin, open(pdb_out, 'w') as fout:
        replaced = False
        for line in fin:
            if line.startswith('CRYST1') and not replaced:
                fout.write(cryst1)
                replaced = True
            elif line.startswith('SCALE'):
                pass  # drop SCALE records; gemmi recomputes from CRYST1
            else:
                fout.write(line)
        if not replaced:
            fout.write(cryst1)


def _write_log(wdir, lines):
    with open(os.path.join(wdir, 'test.log'), 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
