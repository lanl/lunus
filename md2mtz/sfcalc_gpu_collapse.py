#!/usr/bin/env ccp4-python
"""
sfcalc_gpu_collapse.py
======================
GPU-accelerated structure factor calculation from a P1 supercell PDB,
producing two MTZ outputs for diffuse scatter computation:

  outI    -- supercell squared amplitudes   I(h,k,l) = |F_super(h,k,l)|²
             in P1 supercell space; average across MD frames gives <|F|²>

  outmtz  -- primitive-cell ASU structure factors  FC, PHIC
             collapsed from the supercell via:
             F_SG(H,K,L) = Σ_i  exp(2πi H·t_i) * F_super(na*R_i^T H, ...)
             average across MD frames gives <F>; then |<F>|² = <F_avg>²

Diffuse scatter: I_diffuse(H) = <|F_SG(H)|²> - |<F_SG(H)>|²

If super_mult=1,1,1 and sg=P1 both outputs cover the same P1 ASU reflections.

Usage
-----
  sfcalc_gpu_collapse.py  input.pdb
      [dmin=1.5]  [rate=2.5]
      [sg=P1]  [super_mult=1,1,1]
      [outmtz=collapsed.mtz]  [outI=supercell_I.mtz]  [outmap=]
      [bmax=0]  [noise=0.01]  [lib=sfcalc_gpu.so]

  sg          : space group of the PRIMITIVE cell  (default P1)
  super_mult  : supercell multipliers na,nb,nc      (default 1,1,1)
  outmtz      : primitive-cell ASU phased MTZ       (FC + PHIC columns)
  outI        : supercell intensity MTZ             (I column, P1 space)
"""

import sys
import os
import ctypes
import math
import time
import numpy as np
import gemmi

ELEM_IDX    = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'P': 4, 'S': 5}
DEFAULT_ELEM = 0
LEVEL_FACTOR = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv):
    args = {
        'pdb':        None,
        'dmin':       1.5,
        'rate':       2.5,
        'outmtz':     'collapsed.mtz',
        'outI':       'supercell_I.mtz',
        'outmap':     '',
        'bmax':       0.0,
        'noise':      0.01,
        'lib':        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'sfcalc_gpu.so'),
        'sg':         'P 1',
        'super_mult': (1, 1, 1),
    }
    for arg in argv[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower().strip()
            if   key == 'dmin':                    args['dmin']   = float(val)
            elif key == 'rate':                    args['rate']   = float(val)
            elif key in ('outmtz', 'mtz'):         args['outmtz'] = val
            elif key in ('outi', 'outsq', 'outisq'):  args['outI'] = val
            elif key in ('outmap', 'map'):         args['outmap'] = val
            elif key in ('bmax', 'bmax_skip'):     args['bmax']   = float(val)
            elif key == 'noise':                   args['noise']  = float(val)
            elif key == 'lib':                     args['lib']    = val
            elif key in ('sg', 'spacegroup', 'smallsg', 'space_group'):
                args['sg'] = val
            elif key in ('super_mult', 'mult', 'multipliers', 'md_mult'):
                parts = [p.strip() for p in val.replace('x', ',').split(',')]
                if len(parts) != 3:
                    sys.exit("ERROR: super_mult must be na,nb,nc  e.g. 3,3,3")
                args['super_mult'] = tuple(int(p) for p in parts)
        elif arg.endswith('.pdb') or arg.endswith('.cif'):
            args['pdb'] = arg
    return args


# ---------------------------------------------------------------------------
# Grid helpers  (same as sfcalc_gpu.py)
# ---------------------------------------------------------------------------

def good_fft_size(n):
    best = n * 10
    i2 = 1
    while i2 <= n * 2:
        i3 = i2
        while i3 <= n * 2:
            i5 = i3
            while i5 <= n * 2:
                if i5 >= n:
                    best = min(best, i5)
                i5 *= 5
            i3 *= 3
        i2 *= 2
    return best


def compute_levels(dmin, rate, ax, ay, az, min_pts=4):
    levels = []
    d = dmin
    while True:
        s  = d / (2.0 * rate)
        nx = good_fft_size(max(min_pts, math.ceil(ax / s)))
        ny = good_fft_size(max(min_pts, math.ceil(ay / s)))
        nz = good_fft_size(max(min_pts, math.ceil(az / s)))
        levels.append((d, nx, ny, nz))
        if min(nx, ny, nz) <= min_pts:
            break
        d *= LEVEL_FACTOR
    return levels


def noise_wpx(noise_frac, neighbors_2d=4.0, neighbors_3d=6.0):
    ln2 = math.log(2.0)
    noise_2d = noise_frac * (neighbors_2d / neighbors_3d)
    return (4.0 / 9.0) * math.sqrt(-math.log(math.log(noise_2d + 1.0)) / ln2)


def assign_levels(B_arr, pixel_fine, noise_frac, n_levels):
    ln2   = math.log(2.0)
    log_f = math.log(LEVEL_FACTOR)
    w_px  = noise_wpx(noise_frac)
    fwhm      = np.sqrt(ln2 * (B_arr + 0.0)) / (2.0 * math.pi)
    pixel_req = fwhm / w_px
    ratio = pixel_req / pixel_fine
    lev = np.where(ratio >= 1.0,
                   np.floor(np.log(ratio.clip(min=1.0)) / log_f).astype(np.int32),
                   0)
    return np.clip(lev, 0, n_levels - 1).astype(np.int32)


def b_threshold_for_level(L, pixel_fine, noise_frac):
    if L == 0:
        return 0.0
    ln2  = math.log(2.0)
    w_px = noise_wpx(noise_frac)
    pixel_L  = pixel_fine * (LEVEL_FACTOR ** L)
    fwhm_min = pixel_L * w_px
    return fwhm_min ** 2 * (4.0 * math.pi ** 2) / ln2 - 0.0


# ---------------------------------------------------------------------------
# GPU spreading helper  (same as sfcalc_gpu.py)
# ---------------------------------------------------------------------------

def run_gpu_raw(lib, x, y, z, B, el, nx, ny, nz, ax, ay, az,
                alpha=90., beta=90., gamma=90., do_map=False):
    nx2   = nx // 2 + 1
    fft_n = nx2 * ny * nz
    _rbuf  = bytearray(fft_n * 4)
    _ibuf  = bytearray(fft_n * 4)
    F_real = np.frombuffer(_rbuf, dtype=np.float32)
    F_imag = np.frombuffer(_ibuf, dtype=np.float32)
    if do_map:
        _mbuf   = bytearray(nx * ny * nz * 4)
        map_buf = np.frombuffer(_mbuf, dtype=np.float32)
    else:
        map_buf = None

    def fptr(arr):
        if arr is None:
            return ctypes.cast(None, ctypes.POINTER(ctypes.c_float))
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    nkept = lib.spread_and_fft(
        len(x),
        fptr(x), fptr(y), fptr(z), fptr(B),
        el.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nx, ny, nz,
        ctypes.c_float(ax), ctypes.c_float(ay), ctypes.c_float(az),
        ctypes.c_float(alpha), ctypes.c_float(beta), ctypes.c_float(gamma),
        ctypes.c_float(0.0),
        fptr(map_buf), fptr(F_real), fptr(F_imag),
    )
    if nkept < 0:
        sys.exit(f"ERROR: spread_and_fft returned {nkept}")
    return F_real, F_imag, map_buf, nkept


# ---------------------------------------------------------------------------
# Add coarse level into fine-grid accumulator  (same as sfcalc_gpu.py)
# ---------------------------------------------------------------------------

def add_to_fine(acc, nx, ny, nz, coarse, nx_c, ny_c, nz_c):
    nx_c2 = nx_c // 2 + 1
    Ln = nz_c // 2 + 1;  Lh = nz_c - Ln
    Kn = ny_c // 2 + 1;  Kh = ny_c - Kn
    acc[0:Ln,    0:Kn,   :nx_c2] += coarse[0:Ln, 0:Kn, :]
    if Kh:
        acc[0:Ln,    ny-Kh:, :nx_c2] += coarse[0:Ln, Kn:,  :]
    if Lh:
        acc[nz-Lh:,  0:Kn,   :nx_c2] += coarse[Ln:,  0:Kn, :]
    if Lh and Kh:
        acc[nz-Lh:,  ny-Kh:, :nx_c2] += coarse[Ln:,  Kn:,  :]


# ---------------------------------------------------------------------------
# Vectorised ASU mask for the primitive cell
# ---------------------------------------------------------------------------

def asu_mask_for_laue(laue, H3, K3, L3):
    if laue in ('-1',):
        return ((L3 > 0) | ((L3 == 0) & (K3 > 0)) |
                ((L3 == 0) & (K3 == 0) & (H3 > 0)))
    elif laue in ('2/m',):
        return (H3 >= 0) & (L3 >= 0) & ((H3 > 0) | (K3 >= 0))
    elif laue in ('mmm',):
        return (H3 >= 0) & (K3 >= 0) & (L3 >= 0) & ~((H3==0) & (K3==0) & (L3==0))
    elif laue in ('4/m', '4/mmm'):
        return (H3 >= K3) & (K3 >= 0) & (L3 >= 0) & ~((H3==0) & (K3==0) & (L3==0))
    elif laue in ('-3', '-3m', '6/m', '6/mmm'):
        return ((H3 >= 0) & (K3 >= 0) & (L3 >= 0) & (H3 >= K3) &
                ~((H3==0) & (K3==0) & (L3==0)))
    elif laue in ('m-3', 'm-3m'):
        return (H3 >= K3) & (K3 >= L3) & (L3 >= 0) & ~((H3==0) & (K3==0) & (L3==0))
    else:
        sys.exit(f"ERROR: unsupported Laue class '{laue}'")


def build_prim_asu(sg, prim_cell, dmin):
    """Return (H, K, L) int32 arrays for all primitive-cell ASU reflections.
    Uses gemmi.ReciprocalAsu to match gemmi's exact ASU convention and excludes
    systematically absent reflections."""
    rc = prim_cell.reciprocal()
    H_max = int(math.ceil(1.0 / (dmin * rc.a)))
    K_max = int(math.ceil(1.0 / (dmin * rc.b)))
    L_max = int(math.ceil(1.0 / (dmin * rc.c)))

    # Reciprocal metric tensor for general cell
    rca = math.cos(math.radians(rc.alpha))
    rcb = math.cos(math.radians(rc.beta))
    rcg = math.cos(math.radians(rc.gamma))
    gs11 = rc.a**2;  gs22 = rc.b**2;  gs33 = rc.c**2
    gs12 = rc.a * rc.b * rcg
    gs13 = rc.a * rc.c * rcb
    gs23 = rc.b * rc.c * rca
    inv_dmin2 = 1.0 / (dmin * dmin)

    asu = gemmi.ReciprocalAsu(sg)
    ops = sg.operations()

    Hlist, Klist, Llist = [], [], []
    for H in range(-H_max, H_max + 1):
        for K in range(-K_max, K_max + 1):
            for L in range(-L_max, L_max + 1):
                if not asu.is_in((H, K, L)):
                    continue
                inv_d2 = (H*H*gs11 + K*K*gs22 + L*L*gs33
                          + 2*(H*K*gs12 + H*L*gs13 + K*L*gs23))
                if inv_d2 <= 0 or inv_d2 > inv_dmin2:
                    continue
                if ops.is_systematically_absent((H, K, L)):
                    continue
                Hlist.append(H)
                Klist.append(K)
                Llist.append(L)

    return (np.array(Hlist, dtype=np.int32),
            np.array(Klist, dtype=np.int32),
            np.array(Llist, dtype=np.int32))


# ---------------------------------------------------------------------------
# Supercell collapse: look up F_super from acc arrays for all ASU reflections
# ---------------------------------------------------------------------------

def collapse_to_prim_asu(acc_real, acc_imag, nx, ny, nz,
                          na, nb, nc, sg, H_asu, K_asu, L_asu):
    """
    For each primitive-cell ASU reflection (H,K,L), compute:
        F_SG(H) = Σ_op  exp(2πi H·t_op)  *  F_super(na*R^T_op*H, nb*..., nc*...)

    R^T is the TRANSPOSED direct-space rotation matrix (see supercell_collapse).
    Friedel mates are used when the required supercell index is in the H<0 half.

    Returns (F_re, F_im) as float64 arrays (length = number of ASU reflections).
    """
    nx2  = nx // 2 + 1
    ops  = list(sg.operations())
    den  = gemmi.Op.DEN

    H = H_asu.astype(np.int64)
    K = K_asu.astype(np.int64)
    L = L_asu.astype(np.int64)

    F_re = np.zeros(len(H), dtype=np.float64)
    F_im = np.zeros(len(H), dtype=np.float64)

    for op in ops:
        rot  = op.rot
        tran = op.tran

        # R_direct^T * (H,K,L)  — transposed indexing (col,row) instead of (row,col)
        Hr = (rot[0][0]*H + rot[1][0]*K + rot[2][0]*L) // den
        Kr = (rot[0][1]*H + rot[1][1]*K + rot[2][1]*L) // den
        Lr = (rot[0][2]*H + rot[1][2]*K + rot[2][2]*L) // den

        # Scale to supercell Miller indices
        SH = na * Hr
        SK = nb * Kr
        SL = nc * Lr

        # Put into the H>=0 half-space stored by the R2C FFT
        friedel = ((SH < 0) |
                   ((SH == 0) & (SK < 0)) |
                   ((SH == 0) & (SK == 0) & (SL < 0)))
        SH = np.where(friedel, -SH, SH)
        SK = np.where(friedel, -SK, SK)
        SL = np.where(friedel, -SL, SL)

        # Convert K,L to grid storage indices (wrap negatives)
        ix = SH                                          # H >= 0
        iy = np.where(SK >= 0, SK, SK + ny).astype(np.int64)
        iz = np.where(SL >= 0, SL, SL + nz).astype(np.int64)

        # Bounds check (some operators may map to beyond the grid Nyquist)
        valid = ((ix < nx2) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz))
        if not valid.all():
            n_oor = int((~valid).sum())
            print(f"    [{op.triplet()}] {n_oor} reflections out of grid range (skipped)")

        ix_s = np.where(valid, ix, 0).astype(np.intp)
        iy_s = np.where(valid, iy, 0).astype(np.intp)
        iz_s = np.where(valid, iz, 0).astype(np.intp)

        re = np.where(valid, acc_real[iz_s, iy_s, ix_s].astype(np.float64), 0.0)
        im = np.where(valid, acc_imag[iz_s, iy_s, ix_s].astype(np.float64), 0.0)
        # Friedel mate: F(-H) = conj(F(H)) for real density → negate imag
        im = np.where(friedel, -im, im)

        # Phase factor: exp(2πi H·t) = exp(-i * op.phase_shift(H))
        # gemmi op.phase_shift returns -2π H·t, so multiply by -1
        phase = 2.0 * math.pi * (H * tran[0] + K * tran[1] + L * tran[2]) / den
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)

        F_re += re * cos_p - im * sin_p
        F_im += re * sin_p + im * cos_p

    return F_re, F_im


def precompute_collapse(nx, ny, nz, na, nb, nc, sg, H_asu, K_asu, L_asu):
    """Pre-compute index and phase arrays for symmetry collapse (call once per setup).

    Returns ops_data: list of (iz_s, iy_s, ix_s, valid, friedel, cos_p, sin_p),
    one entry per symmetry operator.  Pass to collapse_fast() for each frame.
    """
    nx2 = nx // 2 + 1
    den = gemmi.Op.DEN
    H = H_asu.astype(np.int64)
    K = K_asu.astype(np.int64)
    L = L_asu.astype(np.int64)
    ops_data = []
    for op in sg.operations():
        rot = op.rot; tran = op.tran
        Hr = (rot[0][0]*H + rot[1][0]*K + rot[2][0]*L) // den
        Kr = (rot[0][1]*H + rot[1][1]*K + rot[2][1]*L) // den
        Lr = (rot[0][2]*H + rot[1][2]*K + rot[2][2]*L) // den
        SH = na * Hr; SK = nb * Kr; SL = nc * Lr
        friedel = ((SH < 0) | ((SH == 0) & (SK < 0)) |
                   ((SH == 0) & (SK == 0) & (SL < 0)))
        SH = np.where(friedel, -SH, SH)
        SK = np.where(friedel, -SK, SK)
        SL = np.where(friedel, -SL, SL)
        ix = SH
        iy = np.where(SK >= 0, SK, SK + ny).astype(np.int64)
        iz = np.where(SL >= 0, SL, SL + nz).astype(np.int64)
        valid = ((ix < nx2) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz))
        if not valid.all():
            n_oor = int((~valid).sum())
            print(f"    [{op.triplet()}] {n_oor} reflections out of grid range (skipped)")
        ix_s = np.where(valid, ix, 0).astype(np.intp)
        iy_s = np.where(valid, iy, 0).astype(np.intp)
        iz_s = np.where(valid, iz, 0).astype(np.intp)
        phase = 2.0 * math.pi * (H * tran[0] + K * tran[1] + L * tran[2]) / den
        cos_p = np.cos(phase); sin_p = np.sin(phase)
        ops_data.append((iz_s, iy_s, ix_s, valid, friedel, cos_p, sin_p))
    return ops_data


def collapse_fast(acc_real, acc_imag, ops_data):
    """Symmetry collapse using pre-computed ops_data from precompute_collapse()."""
    F_re = np.zeros(len(ops_data[0][0]), dtype=np.float64)
    F_im = np.zeros(len(ops_data[0][0]), dtype=np.float64)
    for iz_s, iy_s, ix_s, valid, friedel, cos_p, sin_p in ops_data:
        re = np.where(valid, acc_real[iz_s, iy_s, ix_s].astype(np.float64), 0.0)
        im = np.where(valid, acc_imag[iz_s, iy_s, ix_s].astype(np.float64), 0.0)
        im = np.where(friedel, -im, im)
        F_re += re * cos_p - im * sin_p
        F_im += re * sin_p + im * cos_p
    return F_re, F_im


# ---------------------------------------------------------------------------
# Write output MTZ
# ---------------------------------------------------------------------------

def write_mtz(H, K, L, amp, phi, cell, sg, outpath):
    out_mtz            = gemmi.Mtz(with_base=False)
    out_mtz.spacegroup = sg
    out_mtz.cell       = cell
    base_ds = out_mtz.add_dataset("HKL_base");   base_ds.wavelength = 0.0
    data_ds = out_mtz.add_dataset("SFCALC_GPU"); data_ds.wavelength = 1.0
    ds_id   = data_ds.id
    for label, ctype in [('H','H'),('K','H'),('L','H'),('FC','F'),('PHIC','P')]:
        col = out_mtz.add_column(label, ctype)
        col.dataset_id = 0 if ctype == 'H' else ds_id
    out_data = np.column_stack([H.astype(np.float32), K.astype(np.float32),
                                L.astype(np.float32), amp, phi])
    if len(out_data):
        out_mtz.set_data(out_data)
    out_mtz.write_to_file(outpath)
    print(f"  Written: {outpath}  ({len(out_data)} reflections)")


def write_mtz_I(H, K, L, intensity, cell, outpath):
    """Write a P1 intensity MTZ (supercell h,k,l + I = |F|² column)."""
    sg_p1              = gemmi.find_spacegroup_by_name('P 1')
    out_mtz            = gemmi.Mtz(with_base=False)
    out_mtz.spacegroup = sg_p1
    out_mtz.cell       = cell
    base_ds = out_mtz.add_dataset("HKL_base");   base_ds.wavelength = 0.0
    data_ds = out_mtz.add_dataset("SFCALC_GPU"); data_ds.wavelength = 1.0
    ds_id   = data_ds.id
    for label, ctype in [('H','H'),('K','H'),('L','H'),('I','J')]:
        col = out_mtz.add_column(label, ctype)
        col.dataset_id = 0 if ctype == 'H' else ds_id
    out_data = np.column_stack([H.astype(np.float32), K.astype(np.float32),
                                L.astype(np.float32), intensity.astype(np.float32)])
    if len(out_data):
        out_mtz.set_data(out_data)
    out_mtz.write_to_file(outpath)
    print(f"  Written: {outpath}  ({len(out_data)} reflections)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args(sys.argv)
    if args['pdb'] is None:
        print(__doc__)
        sys.exit("ERROR: no PDB file specified")

    dmin  = args['dmin']
    rate  = args['rate']
    bmax  = args['bmax']
    noise = args['noise']
    na, nb, nc = args['super_mult']

    # Resolve space group
    sg_name = args['sg']
    sg = gemmi.find_spacegroup_by_name(sg_name)
    if sg is None:
        sys.exit(f"ERROR: unknown space group '{sg_name}'")

    # ------------------------------------------------------------------
    # 1. Load PDB
    # ------------------------------------------------------------------
    print(f"Reading {args['pdb']} ...")
    st   = gemmi.read_structure(args['pdb'])
    cell = st.cell
    ax, ay, az = cell.a, cell.b, cell.c
    print(f"  Supercell: {ax:.3f} x {ay:.3f} x {az:.3f}  "
          f"angles: {cell.alpha:.2f} {cell.beta:.2f} {cell.gamma:.2f}")
    prim_ax = ax / na;  prim_ay = ay / nb;  prim_az = az / nc
    prim_cell = gemmi.UnitCell(prim_ax, prim_ay, prim_az,
                               cell.alpha, cell.beta, cell.gamma)
    print(f"  super_mult: {na} x {nb} x {nc}")
    print(f"  Primitive cell: {prim_ax:.3f} x {prim_ay:.3f} x {prim_az:.3f}")
    print(f"  Space group: {sg.xhm()}")

    xs, ys, zs, Bs, els = [], [], [], [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    xs.append(p.x); ys.append(p.y); zs.append(p.z)
                    Bs.append(atom.b_iso)
                    els.append(ELEM_IDX.get(atom.element.name.upper().strip(),
                                            DEFAULT_ELEM))

    x_arr  = np.array(xs,  dtype=np.float32)
    y_arr  = np.array(ys,  dtype=np.float32)
    z_arr  = np.array(zs,  dtype=np.float32)
    B_arr  = np.maximum(np.array(Bs, dtype=np.float32), 0.0)
    el_arr = np.array(els, dtype=np.int32)
    natoms = len(x_arr)
    print(f"  Atoms: {natoms}  B: {B_arr.min():.2f} .. {B_arr.max():.2f}")

    if bmax > 0:
        keep  = B_arr <= bmax
        nskip = int((~keep).sum())
        if nskip:
            print(f"  Skipping {nskip} atoms with B > {bmax}")
        x_arr, y_arr, z_arr, B_arr, el_arr = (
            a[keep] for a in (x_arr, y_arr, z_arr, B_arr, el_arr))
        natoms = int(keep.sum())

    # Auto-blur: add b_add to all B-factors before spreading so no Gaussian is
    # sub-pixel; the corresponding exp(-b_add*stol^2) envelope is corrected
    # after the FFT by multiplying each F(H) by exp(+b_add*stol^2).
    b_add = (dmin * rate) ** 2 / math.pi ** 2
    sigma_min = math.sqrt(b_add / (4.0 * math.pi ** 2))
    pixel_fine_pre = dmin / (2.0 * rate)
    print(f"  Auto-blur: b_add = {b_add:.4f} A^2  "
          f"(sigma_min = {sigma_min:.3f} A, pixel = {pixel_fine_pre:.3f} A)")
    B_arr = B_arr + np.float32(b_add)

    # ------------------------------------------------------------------
    # 2. Multi-grid levels
    # ------------------------------------------------------------------
    levels     = compute_levels(dmin, rate, ax, ay, az)
    n_levels   = len(levels)
    pixel_fine = dmin / (2.0 * rate)
    atom_lev   = assign_levels(B_arr, pixel_fine, noise, n_levels)

    d0, nx, ny, nz = levels[0]
    nx2    = nx // 2 + 1
    V_cell = cell.volume
    w_px   = noise_wpx(noise)

    last_occupied = max((L for L in range(n_levels) if (atom_lev == L).any()),
                        default=0)
    print(f"  Multi-grid levels (noise<={noise*100:.1f}%, w_px={w_px:.3f}, step=sqrt(2)):")
    for L, (d_L, nx_L, ny_L, nz_L) in enumerate(levels):
        if L > last_occupied:
            break
        n_L   = int((atom_lev == L).sum())
        B_lo  = b_threshold_for_level(L,     pixel_fine, noise)
        B_hi  = b_threshold_for_level(L + 1, pixel_fine, noise) if L < n_levels - 1 else float('inf')
        vfrac = (nx_L * ny_L * nz_L) / (nx * ny * nz) * 100
        print(f"    L{L}: pixel={d_L/rate/2:.3f}A  grid {nx_L}x{ny_L}x{nz_L} "
              f"({vfrac:.1f}%)  B=[{B_lo:.1f},{B_hi:.1f})  {n_L} atoms")

    # ------------------------------------------------------------------
    # 3. Load GPU library
    # ------------------------------------------------------------------
    lib_path = args['lib']
    if not os.path.exists(lib_path):
        sys.exit(f"ERROR: shared library not found: {lib_path}")
    lib = ctypes.CDLL(lib_path)
    lib.spread_and_fft.restype  = ctypes.c_int
    lib.spread_and_fft.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,   # ax, ay, az
        ctypes.c_float, ctypes.c_float, ctypes.c_float,   # alpha, beta, gamma
        ctypes.c_float,                                    # Bmax_skip
        ctypes.POINTER(ctypes.c_float),   # map_out (float32)
        ctypes.POINTER(ctypes.c_float),   # F_real  (float32)
        ctypes.POINTER(ctypes.c_float),   # F_imag  (float32)
    ]

    # ------------------------------------------------------------------
    # 4. Spread + FFT each level → accumulate into fine-grid float32 arrays
    # ------------------------------------------------------------------
    do_map = bool(args['outmap'])

    _acc_r = bytearray(nz * ny * nx2 * 8)
    _acc_i = bytearray(nz * ny * nx2 * 8)
    acc_real = np.frombuffer(_acc_r, dtype=np.float64).reshape(nz, ny, nx2)
    acc_imag = np.frombuffer(_acc_i, dtype=np.float64).reshape(nz, ny, nx2)
    map_buf  = None
    nkept_total = 0

    t_start = time.perf_counter()
    for L, (d_L, nx_L, ny_L, nz_L) in enumerate(levels):
        mask_L = (atom_lev == L)
        n_L    = int(mask_L.sum())
        if n_L == 0:
            continue
        t0 = time.perf_counter()
        Fr, Fi, mb, nk = run_gpu_raw(
            lib,
            x_arr[mask_L], y_arr[mask_L], z_arr[mask_L],
            B_arr[mask_L], el_arr[mask_L],
            nx_L, ny_L, nz_L, ax, ay, az,
            cell.alpha, cell.beta, cell.gamma,
            do_map=(do_map and L == 0),
        )
        t1 = time.perf_counter()
        norm_L = np.float32(V_cell / (nx_L * ny_L * nz_L))
        Fr *= norm_L;  Fi *= -norm_L
        nx_L2 = nx_L // 2 + 1
        Fr3 = Fr.reshape(nz_L, ny_L, nx_L2)
        Fi3 = Fi.reshape(nz_L, ny_L, nx_L2)
        if L == 0:
            acc_real += Fr3;  acc_imag += Fi3
            if mb is not None:
                map_buf = mb
        else:
            add_to_fine(acc_real, nx, ny, nz, Fr3, nx_L, ny_L, nz_L)
            add_to_fine(acc_imag, nx, ny, nz, Fi3, nx_L, ny_L, nz_L)
        t2 = time.perf_counter()
        print(f"    L{L}: {n_L} atoms  GPU {(t1-t0)*1000:.0f} ms  "
              f"combine {(t2-t1)*1000:.0f} ms")
        nkept_total += nk
    t_end = time.perf_counter()
    print(f"  GPU total: {(t_end-t_start)*1000:.0f} ms  ({nkept_total} atoms spread)")

    # Blur correction is applied at ASU level below (not to the full grid).
    # Pre-compute reciprocal metric for stol^2 calculations.
    rc_s = cell.reciprocal()
    cg = math.cos(math.radians(rc_s.gamma))
    cb = math.cos(math.radians(rc_s.beta))
    ca = math.cos(math.radians(rc_s.alpha))

    # ------------------------------------------------------------------
    # 5. Write CCP4 map if requested (level-0 atoms only when multi-grid)
    # ------------------------------------------------------------------
    if do_map and map_buf is not None:
        n_coarse = int((atom_lev > 0).sum())
        if n_coarse:
            print(f"  Note: map contains level-0 atoms only ({natoms - n_coarse} of {natoms})")
        rho_3d = map_buf.reshape(nz, ny, nx)
        ccp4   = gemmi.Ccp4Map()
        ccp4.grid = gemmi.FloatGrid(rho_3d.astype(np.float32))
        ccp4.grid.unit_cell  = cell
        ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(args['outmap'])
        print(f"  Written: {args['outmap']}")

    # ------------------------------------------------------------------
    # 6. Extract supercell P1 ASU reflections → write supercell I MTZ
    # ------------------------------------------------------------------
    print("Extracting supercell structure factors ...")
    H_1d = np.arange(nx2, dtype=np.int32)
    K_1d = np.arange(ny,  dtype=np.int32)
    L_1d = np.arange(nz,  dtype=np.int32)
    K_1d = np.where(K_1d > ny // 2, K_1d - ny, K_1d)
    L_1d = np.where(L_1d > nz // 2, L_1d - nz, L_1d)
    H3 = H_1d[None, None, :];  K3 = K_1d[None, :, None];  L3 = L_1d[:, None, None]
    inv_d2    = (H3 / ax) ** 2 + (K3 / ay) ** 2 + (L3 / az) ** 2
    inv_dmin2 = 1.0 / (dmin * dmin)
    asu_p1    = ((L3 > 0) | ((L3 == 0) & (K3 > 0)) | ((L3 == 0) & (K3 == 0) & (H3 > 0)))
    sc_mask   = (inv_d2 <= inv_dmin2) & asu_p1

    re_sc = acc_real[sc_mask].astype(np.float64)
    im_sc = acc_imag[sc_mask].astype(np.float64)
    H_sc  = np.broadcast_to(H3, (nz, ny, nx2))[sc_mask].astype(np.float32)
    K_sc  = np.broadcast_to(K3, (nz, ny, nx2))[sc_mask].astype(np.float32)
    L_sc  = np.broadcast_to(L3, (nz, ny, nx2))[sc_mask].astype(np.float32)

    # Apply blur correction at supercell ASU points only (not the full grid)
    Hf = H_sc.astype(np.float64); Kf = K_sc.astype(np.float64); Lf = L_sc.astype(np.float64)
    stol2_sc = 0.25 * (Hf**2 * rc_s.a**2 + Kf**2 * rc_s.b**2 + Lf**2 * rc_s.c**2
                       + 2.0 * (Hf*Kf*rc_s.a*rc_s.b*cg + Hf*Lf*rc_s.a*rc_s.c*cb
                                + Kf*Lf*rc_s.b*rc_s.c*ca))
    blur_sc = np.exp(b_add * stol2_sc)
    re_sc *= blur_sc; im_sc *= blur_sc

    if args['outI']:
        I_sc = (re_sc**2 + im_sc**2).astype(np.float32)
        write_mtz_I(H_sc, K_sc, L_sc, I_sc, cell, args['outI'])

    # ------------------------------------------------------------------
    # 7. Collapse to primitive-cell ASU → write phased MTZ
    # ------------------------------------------------------------------
    simple_p1 = (na == 1 and nb == 1 and nc == 1 and sg.hm == 'P 1')

    if simple_p1:
        # Fast path: supercell IS the primitive cell
        amp = np.hypot(re_sc, im_sc).astype(np.float32)
        phi = np.degrees(np.arctan2(im_sc, re_sc)).astype(np.float32)
        write_mtz(H_sc, K_sc, L_sc, amp, phi, cell, sg, args['outmtz'])

    else:
        # General path: collapse supercell → primitive-cell ASU
        print(f"Collapsing supercell to primitive-cell ASU "
              f"({sg.xhm()}, {len(list(sg.operations()))} operators) ...")
        t0 = time.perf_counter()
        H_asu, K_asu, L_asu = build_prim_asu(sg, prim_cell, dmin)
        t1 = time.perf_counter()
        print(f"  Primitive-cell ASU reflections: {len(H_asu)}  "
              f"(enumerated in {(t1-t0)*1000:.0f} ms)")

        t0 = time.perf_counter()
        ops_data = precompute_collapse(nx, ny, nz, na, nb, nc, sg, H_asu, K_asu, L_asu)
        F_re, F_im = collapse_fast(acc_real, acc_imag, ops_data)
        t1 = time.perf_counter()
        print(f"  Collapse complete in {(t1-t0)*1000:.0f} ms")

        # Apply blur correction at primitive ASU level
        prim_rc = prim_cell.reciprocal()
        pcg = math.cos(math.radians(prim_rc.gamma))
        pcb = math.cos(math.radians(prim_rc.beta))
        pca = math.cos(math.radians(prim_rc.alpha))
        Ha = H_asu.astype(np.float64); Ka = K_asu.astype(np.float64); La = L_asu.astype(np.float64)
        stol2_asu = 0.25 * (Ha**2 * prim_rc.a**2 + Ka**2 * prim_rc.b**2 + La**2 * prim_rc.c**2
                            + 2.0 * (Ha*Ka*prim_rc.a*prim_rc.b*pcg
                                     + Ha*La*prim_rc.a*prim_rc.c*pcb
                                     + Ka*La*prim_rc.b*prim_rc.c*pca))
        blur_asu = np.exp(b_add * stol2_asu)
        F_re *= blur_asu; F_im *= blur_asu

        amp = np.hypot(F_re, F_im).astype(np.float32)
        phi = np.degrees(np.arctan2(F_im, F_re)).astype(np.float32)
        write_mtz(H_asu, K_asu, L_asu, amp, phi, prim_cell, sg, args['outmtz'])

    print("Done.")


if __name__ == '__main__':
    main()
