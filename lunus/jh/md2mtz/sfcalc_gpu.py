#!/usr/bin/env ccp4-python
"""
sfcalc_gpu.py
=============
GPU-accelerated structure factor calculator.

Matches: gemmi sfcalc --dmin=DMIN --rate=RATE --write-map=MAP --to-mtz=MTZ PDB

Usage:
  sfcalc_gpu.py  input.pdb  [dmin=1.5]  [rate=2.5]  [outmtz=out.mtz]
                 [outmap=out.map]  [bmax=0]  [noise=0.01]  [lib=sfcalc_gpu.so]

  dmin   : resolution cutoff in Angstroms  (default 1.5)
  rate   : grid oversampling rate          (default 2.5)
  outmtz : output MTZ filename             (default sfcalc_gpu.mtz)
  outmap : output CCP4 map filename        (default sfcalc_gpu.map, empty = skip)
  bmax   : skip atoms with B > bmax        (default 0 = keep all)
  noise  : target RMS aliasing noise as fraction of peak  (default 0.01 = 1%)
  lib    : path to compiled .so            (default ./sfcalc_gpu.so)

Multi-grid algorithm
--------------------
Atoms are binned by B-factor onto grids with progressively coarser spacing.
For each atom, the required pixel size is determined so that spreading a
Gaussian of its FWHM onto that grid meets a target aliasing noise floor.

Atom FWHM (Angstroms) from B-factor:
    fwhm = sqrt(ln2 * (B + 8)) / (2*pi)
    (the +8 represents the intrinsic width of a B=0 atom)

2D aliasing noise for a Gaussian with fwhm/pixel = w_px spread on a grid:
    noise = max * (exp(gauss(w_px * 1.125)) - 1)
    gauss(x) = exp(-4*ln2*x^2)  (unit-FWHM Gaussian)

Inverted for w_px given target noise (per the user's formula):
    w_px = (4/9) * sqrt(-ln(ln(noise+1)) / ln2)

3D correction: a 3D cubic grid has 6 nearest-neighbor replicas vs 4 in 2D,
so aliasing is ~1.5x larger. The 3D formula therefore solves the 2D equation
with noise_2d = noise_target * (4/6) to get the equivalent safety margin.

Required pixel size:  pixel_req = fwhm / w_px
Level assignment:     L = floor(log2(pixel_req / pixel_fine))

Each level has ~8x fewer voxels than the previous one.
All F accumulation is done in float32 to avoid large complex128 temporaries.
"""

import sys
import os
import ctypes
import math
import numpy as np
import gemmi

ELEM_IDX   = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'P': 4, 'S': 5}
DEFAULT_ELEM = 0


def parse_args(argv):
    args = {
        'pdb':    None,
        'dmin':   1.5,
        'rate':   2.5,
        'outmtz': 'sfcalc_gpu.mtz',
        'outmap': 'sfcalc_gpu.map',
        'bmax':   0.0,
        'noise':  0.01,
        'lib':    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sfcalc_gpu.so'),
    }
    for arg in argv[1:]:
        if '=' in arg:
            key, val = arg.split('=', 1)
            key = key.lower().strip()
            if   key == 'dmin':                args['dmin']   = float(val)
            elif key == 'rate':                args['rate']   = float(val)
            elif key in ('outmtz', 'mtz'):     args['outmtz'] = val
            elif key in ('outmap', 'map'):     args['outmap'] = val
            elif key in ('bmax','bmax_skip'):  args['bmax']   = float(val)
            elif key == 'noise':               args['noise']  = float(val)
            elif key == 'lib':                 args['lib']    = val
        elif arg.endswith('.pdb') or arg.endswith('.cif'):
            args['pdb'] = arg
    return args


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


LEVEL_FACTOR = math.sqrt(2.0)   # pixel doubles every 2 levels; finer than 2x steps


def compute_levels(dmin, rate, ax, ay, az, min_pts=4):
    """Return list of (d_level, nx, ny, nz) for each level.

    Each level's pixel size is LEVEL_FACTOR (= sqrt(2)) times the previous,
    so grid volume halves per level rather than per two levels.  This gives
    finer B-factor granularity: B=2 and B=20 no longer share the same grid.
    """
    levels = []
    d = dmin
    while True:
        s = d / (2.0 * rate)
        nx = good_fft_size(max(min_pts, math.ceil(ax / s)))
        ny = good_fft_size(max(min_pts, math.ceil(ay / s)))
        nz = good_fft_size(max(min_pts, math.ceil(az / s)))
        levels.append((d, nx, ny, nz))
        if min(nx, ny, nz) <= min_pts:
            break
        d *= LEVEL_FACTOR
    return levels


def noise_wpx(noise_frac, neighbors_2d=4.0, neighbors_3d=6.0):
    """Required fwhm/pixel ratio to keep 3D aliasing noise < noise_frac * peak.

    2D aliasing formula (empirical, for a Gaussian spread on a 2D pixel grid):
        noise = max * (exp(gauss(w_px * 1.125)) - 1)
        gauss(x) = exp(-4*ln2*x^2)  [unit-FWHM Gaussian]
    Inverted: w_px = (4/9) * sqrt(-ln(ln(noise+1)) / ln2)

    3D correction: 3D cubic grid has ~neighbors_3d nearest replicas vs
    ~neighbors_2d in 2D, so noise_3d ≈ (n3/n2)*noise_2d.
    We solve the 2D formula for noise_2d = noise_frac*(n2/n3).
    """
    ln2 = math.log(2.0)
    noise_2d = noise_frac * (neighbors_2d / neighbors_3d)
    return (4.0 / 9.0) * math.sqrt(-math.log(math.log(noise_2d + 1.0)) / ln2)


def assign_levels(B_arr, pixel_fine, noise_frac, n_levels):
    """Assign each atom to its coarsest adequate grid level (vectorised).

    For each atom: compute required pixel size from B-factor FWHM and the
    aliasing noise target, then find the coarsest level whose pixel size
    does not exceed that requirement.

    fwhm(B) = sqrt(ln2 * (B+8)) / (2*pi)          [+8 = intrinsic B=0 width]
    pixel_req = fwhm / w_px                         [w_px from noise_wpx()]
    level L:  pixel_fine * LEVEL_FACTOR^L <= pixel_req
           => L = floor(log(ratio) / log(LEVEL_FACTOR))
    """
    ln2 = math.log(2.0)
    log_f = math.log(LEVEL_FACTOR)
    w_px = noise_wpx(noise_frac)
    fwhm = np.sqrt(ln2 * (B_arr + 8.0)) / (2.0 * math.pi)
    pixel_req = fwhm / w_px
    ratio = pixel_req / pixel_fine
    lev = np.where(ratio >= 1.0,
                   np.floor(np.log(ratio.clip(min=1.0)) / log_f).astype(np.int32),
                   0)
    return np.clip(lev, 0, n_levels - 1).astype(np.int32)


def b_threshold_for_level(L, pixel_fine, noise_frac):
    """Minimum B-factor to be assigned to level L (for display purposes)."""
    if L == 0:
        return 0.0
    ln2 = math.log(2.0)
    w_px = noise_wpx(noise_frac)
    pixel_L = pixel_fine * (LEVEL_FACTOR ** L)
    fwhm_min = pixel_L * w_px
    return fwhm_min ** 2 * (4.0 * math.pi ** 2) / ln2 - 8.0


def run_gpu_raw(lib, x, y, z, B, el, nx, ny, nz, ax, ay, az,
                alpha=90., beta=90., gamma=90., do_map=False):
    """Call the GPU spreading+FFT library.
    Returns (F_real_flat, F_imag_flat, map_buf_or_None, nkept).
    F_real/F_imag are float32, shape (nz*(ny*(nx//2+1)),), NOT yet normalised.
    Cell angles in degrees; defaults to orthogonal (90,90,90).
    """
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
        ctypes.c_float(0.0),          # bmax_skip disabled; Python pre-filters
        fptr(map_buf), fptr(F_real), fptr(F_imag),
    )
    if nkept < 0:
        sys.exit(f"ERROR: spread_and_fft returned {nkept}")
    return F_real, F_imag, map_buf, nkept


def add_to_fine(acc, nx, ny, nz, coarse, nx_c, ny_c, nz_c):
    """Add coarse float32 array (nz_c, ny_c, nx_c2) into fine (nz, ny, nx2)
    using physical Miller-index mapping.

    The K and L axes each split into two contiguous blocks at the Nyquist
    point.  Using four slice operations avoids fancy-indexing overhead.
    """
    nx_c2 = nx_c // 2 + 1

    # Number of elements in the "positive-frequency" block vs "negative-freq" block
    Ln = nz_c // 2 + 1   # L_c = 0 .. nz_c//2  → iz_f = 0 .. nz_c//2
    Lh = nz_c - Ln       # L_c = Ln .. nz_c-1  → iz_f = nz-Lh .. nz-1
    Kn = ny_c // 2 + 1   # K_c = 0 .. ny_c//2  → iy_f = 0 .. ny_c//2
    Kh = ny_c - Kn       # K_c = Kn .. ny_c-1  → iy_f = ny-Kh .. ny-1

    acc[0:Ln,    0:Kn,    :nx_c2] += coarse[0:Ln, 0:Kn, :]
    if Kh:
        acc[0:Ln,    ny-Kh:, :nx_c2] += coarse[0:Ln, Kn:,  :]
    if Lh:
        acc[nz-Lh:,  0:Kn,   :nx_c2] += coarse[Ln:,  0:Kn, :]
    if Lh and Kh:
        acc[nz-Lh:,  ny-Kh:, :nx_c2] += coarse[Ln:,  Kn:,  :]


def main():
    args = parse_args(sys.argv)
    if args['pdb'] is None:
        print(__doc__)
        sys.exit("ERROR: no PDB file specified")

    dmin  = args['dmin']
    rate  = args['rate']
    bmax  = args['bmax']
    noise = args['noise']

    # ------------------------------------------------------------------
    # 1. Load PDB
    # ------------------------------------------------------------------
    print(f"Reading {args['pdb']} ...")
    st   = gemmi.read_structure(args['pdb'])
    cell = st.cell
    sg   = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    print(f"  Space group: {sg.hm}")

    ax, ay, az = cell.a, cell.b, cell.c
    print(f"  Cell: {ax:.3f} x {ay:.3f} x {az:.3f}  "
          f"angles: {cell.alpha:.2f} {cell.beta:.2f} {cell.gamma:.2f}")

    xs, ys, zs, Bs, els = [], [], [], [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    xs.append(p.x); ys.append(p.y); zs.append(p.z)
                    Bs.append(atom.b_iso)
                    els.append(ELEM_IDX.get(atom.element.name.upper().strip(), DEFAULT_ELEM))

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

    # Auto-blur: add b_add to all B-factors so no Gaussian is sub-pixel;
    # corrected after FFT by multiplying each F(H) by exp(+b_add*stol^2).
    b_add = (dmin * rate) ** 2 / math.pi ** 2
    sigma_min = math.sqrt(b_add / (4.0 * math.pi ** 2))
    pixel_fine_pre = dmin / (2.0 * rate)
    print(f"  Auto-blur: b_add = {b_add:.4f} A^2  "
          f"(sigma_min = {sigma_min:.3f} A, pixel = {pixel_fine_pre:.3f} A)")
    B_arr = B_arr + np.float32(b_add)

    # ------------------------------------------------------------------
    # 2. Build multi-grid level table and assign atoms
    # ------------------------------------------------------------------
    levels     = compute_levels(dmin, rate, ax, ay, az)
    n_levels   = len(levels)
    pixel_fine = dmin / (2.0 * rate)
    atom_lev   = assign_levels(B_arr, pixel_fine, noise, n_levels)

    d0, nx, ny, nz = levels[0]
    nx2    = nx // 2 + 1
    V_cell = cell.volume
    w_px   = noise_wpx(noise)

    # Find last occupied level so we don't print a long tail of empty levels
    last_occupied = max((L for L in range(n_levels) if (atom_lev == L).any()),
                        default=0)

    print(f"  Multi-grid levels (noise<={noise*100:.1f}%, w_px={w_px:.3f}, "
          f"step=sqrt(2)):")
    for L, (d_L, nx_L, ny_L, nz_L) in enumerate(levels):
        if L > last_occupied:
            break
        n_L   = int((atom_lev == L).sum())
        B_lo  = b_threshold_for_level(L,     pixel_fine, noise)
        B_hi  = b_threshold_for_level(L + 1, pixel_fine, noise) if L < n_levels - 1 else float('inf')
        vfrac = (nx_L * ny_L * nz_L) / (nx * ny * nz) * 100
        print(f"    L{L}: pixel={d_L/rate/2:.3f}A  grid {nx_L}x{ny_L}x{nz_L} "
              f"({vfrac:.1f}% voxels)  B=[{B_lo:.1f},{B_hi:.1f})  {n_L} atoms")

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
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    # ------------------------------------------------------------------
    # 4. Spread each level; accumulate into fine-grid float32 R/I arrays.
    #    Conjugate and normalise in-place before accumulating to avoid
    #    large complex128 temporaries.
    # ------------------------------------------------------------------
    do_map = bool(args['outmap'])

    # Fine-grid accumulators (float32, shape (nz, ny, nx2)).
    # Use bytearray backing so pages are pre-faulted (avoids ~1 µs/page overhead
    # when cudaMemcpy / numpy writes touch lazy-mmap pages from np.zeros).
    _acc_r = bytearray(nz * ny * nx2 * 4)
    _acc_i = bytearray(nz * ny * nx2 * 4)
    acc_real = np.frombuffer(_acc_r, dtype=np.float32).reshape(nz, ny, nx2)
    acc_imag = np.frombuffer(_acc_i, dtype=np.float32).reshape(nz, ny, nx2)
    map_buf  = None
    nkept_total = 0

    import time
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

        # Normalise: V_cell/(nx*ny*nz); negate imag for cuFFT->crystallographic conjugate
        norm_L = np.float32(V_cell / (nx_L * ny_L * nz_L))
        Fr *= norm_L
        Fi *= -norm_L

        nx_L2 = nx_L // 2 + 1
        Fr3 = Fr.reshape(nz_L, ny_L, nx_L2)
        Fi3 = Fi.reshape(nz_L, ny_L, nx_L2)

        if L == 0:
            acc_real += Fr3
            acc_imag += Fi3
            if mb is not None:
                map_buf = mb
        else:
            add_to_fine(acc_real, nx, ny, nz, Fr3, nx_L, ny_L, nz_L)
            add_to_fine(acc_imag, nx, ny, nz, Fi3, nx_L, ny_L, nz_L)

        t2 = time.perf_counter()
        print(f"    L{L}: {n_L} atoms  GPU {(t1-t0)*1000:.0f} ms  combine {(t2-t1)*1000:.0f} ms")
        nkept_total += nk

    t_end = time.perf_counter()
    print(f"  GPU total time: {(t_end-t_start)*1000:.0f} ms  ({nkept_total} atoms spread)")

    # ------------------------------------------------------------------
    # 5. Extract ASU reflections -> MTZ
    #    Work entirely in float32; no large complex128 array created.
    # ------------------------------------------------------------------
    print("Extracting structure factors from FFT output ...")

    H_1d = np.arange(nx2, dtype=np.int32)
    K_1d = np.arange(ny,  dtype=np.int32)
    L_1d = np.arange(nz,  dtype=np.int32)
    K_1d = np.where(K_1d > ny // 2, K_1d - ny, K_1d)
    L_1d = np.where(L_1d > nz // 2, L_1d - nz, L_1d)

    H3 = H_1d[None, None, :]
    K3 = K_1d[None, :,  None]
    L3 = L_1d[:,  None, None]

    # Reciprocal metric tensor for general cell (reduces to (H/ax)^2+... for orthogonal)
    rc = cell.reciprocal()
    rca = math.cos(math.radians(rc.alpha))
    rcb = math.cos(math.radians(rc.beta))
    rcg = math.cos(math.radians(rc.gamma))
    gs11 = rc.a**2;  gs22 = rc.b**2;  gs33 = rc.c**2
    gs12 = rc.a * rc.b * rcg
    gs13 = rc.a * rc.c * rcb
    gs23 = rc.b * rc.c * rca
    inv_d2 = (H3**2 * gs11 + K3**2 * gs22 + L3**2 * gs33
              + 2*(H3*K3*gs12 + H3*L3*gs13 + K3*L3*gs23))
    inv_dmin2 = 1.0 / (dmin * dmin)

    laue = sg.laue_str()
    if laue in ('-1',):
        asu_mask = (L3 > 0) | ((L3 == 0) & (K3 > 0)) | ((L3 == 0) & (K3 == 0) & (H3 > 0))
    elif laue in ('2/m',):
        asu_mask = (H3 >= 0) & (L3 >= 0) & ((H3 > 0) | (K3 >= 0))
    elif laue in ('mmm',):
        asu_mask = (H3 >= 0) & (K3 >= 0) & (L3 >= 0) & ~((H3 == 0) & (K3 == 0) & (L3 == 0))
    elif laue in ('4/m', '4/mmm'):
        asu_mask = (H3 >= K3) & (K3 >= 0) & (L3 >= 0) & ~((H3 == 0) & (K3 == 0) & (L3 == 0))
    elif laue in ('-3', '-3m', '6/m', '6/mmm'):
        asu_mask = ((H3 >= 0) & (K3 >= 0) & (L3 >= 0) & (H3 >= K3) &
                    ~((H3 == 0) & (K3 == 0) & (L3 == 0)))
    elif laue in ('m-3', 'm-3m'):
        asu_mask = (H3 >= K3) & (K3 >= L3) & (L3 >= 0) & ~((H3 == 0) & (K3 == 0) & (L3 == 0))
    else:
        sys.exit(f"ERROR: unsupported Laue class '{laue}' for space group {sg.hm}")

    mask = (inv_d2 <= inv_dmin2) & asu_mask

    # Extract at ASU points and apply blur correction there (not to the full grid).
    re_sel = acc_real[mask].astype(np.float64)
    im_sel = acc_imag[mask].astype(np.float64)
    H_sel  = np.broadcast_to(H3, (nz, ny, nx2))[mask].astype(np.float32)
    K_sel  = np.broadcast_to(K3, (nz, ny, nx2))[mask].astype(np.float32)
    L_sel  = np.broadcast_to(L3, (nz, ny, nx2))[mask].astype(np.float32)

    rc  = cell.reciprocal()
    cg  = math.cos(math.radians(rc.gamma))
    cb  = math.cos(math.radians(rc.beta))
    ca  = math.cos(math.radians(rc.alpha))
    Hf  = H_sel.astype(np.float64); Kf = K_sel.astype(np.float64); Lf = L_sel.astype(np.float64)
    stol2 = 0.25 * (Hf**2 * rc.a**2 + Kf**2 * rc.b**2 + Lf**2 * rc.c**2
                    + 2.0 * (Hf*Kf*rc.a*rc.b*cg + Hf*Lf*rc.a*rc.c*cb + Kf*Lf*rc.b*rc.c*ca))
    blur  = np.exp(b_add * stol2)
    re_sel *= blur; im_sel *= blur

    amp = np.hypot(re_sel, im_sel).astype(np.float32)
    phi = np.degrees(np.arctan2(im_sel, re_sel)).astype(np.float32)

    out_data = np.column_stack([H_sel, K_sel, L_sel, amp, phi])
    print(f"  ASU reflections: {len(out_data)}")

    out_mtz            = gemmi.Mtz(with_base=False)
    out_mtz.spacegroup = sg
    out_mtz.cell       = cell
    base_ds = out_mtz.add_dataset("HKL_base");   base_ds.wavelength = 0.0
    data_ds = out_mtz.add_dataset("SFCALC_GPU"); data_ds.wavelength = 1.0
    ds_id   = data_ds.id

    for label, ctype in [('H','H'),('K','H'),('L','H'),('FC','F'),('PHIC','P')]:
        col = out_mtz.add_column(label, ctype)
        col.dataset_id = 0 if ctype == 'H' else ds_id

    if len(out_data):
        out_mtz.set_data(out_data)
    out_mtz.write_to_file(args['outmtz'])
    print(f"  Written: {args['outmtz']}")

    # ------------------------------------------------------------------
    # 6. Write CCP4 map (level-0 atoms only when multi-grid is active)
    # ------------------------------------------------------------------
    if do_map and map_buf is not None:
        n_coarse = int((atom_lev > 0).sum())
        if n_coarse:
            print(f"  Note: map contains level-0 atoms only "
                  f"({natoms - n_coarse} of {natoms})")
        rho_3d = map_buf.reshape(nz, ny, nx)
        ccp4   = gemmi.Ccp4Map()
        ccp4.grid = gemmi.FloatGrid(rho_3d.astype(np.float32))
        ccp4.grid.unit_cell  = cell
        ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(args['outmap'])
        print(f"  Written: {args['outmap']}")

    print("Done.")


if __name__ == '__main__':
    main()
