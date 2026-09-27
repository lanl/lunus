#!/usr/bin/env ccp4-python
"""
diag_axes.py — axis-ordering diagnostic for sfcalc_gpu.

For each test reflection (H,K,L), computes:
  1. Direct sum  : F = Σ_atoms f(s) * exp(2πi(Hx+Ky+Lz)/a)  (ground truth)
  2. GPU cuFFT   : value from spread_and_fft output (current MTZ extraction)
  3. PY-FFT-map  : numpy rfftn of GPU map, various axis orderings

Prints comparison so the axis bug is obvious.
"""
import sys, os, ctypes, math
import numpy as np
import gemmi

# -------------------------------------------------------------------
# IT92 C parameters (matches sfcalc_gpu.cu)
# -------------------------------------------------------------------
a_C = np.array([2.31000, 1.02000, 1.58860, 0.86500])
b_C = np.array([20.8439, 10.2075, 0.56870, 51.6512])
c_C = 0.2156

def f_C(s2, B):
    """Carbon form factor at sin²θ/λ² = s2, with Debye-Waller exp(-B*s2)."""
    dw = np.exp(-B * s2)
    f = sum(a * np.exp(-b * s2) for a, b in zip(a_C, b_C)) * dw + c_C * dw
    return f

def direct_sum(atoms, hkl, cell_a, B=10.0):
    """Ground-truth F(H,K,L) via direct summation."""
    H, K, L = hkl
    s2 = ((H/cell_a)**2 + (K/cell_a)**2 + (L/cell_a)**2) / 4.0  # sin²θ/λ²
    F = 0+0j
    for (x, y, z) in atoms:
        phase = 2*math.pi * (H*x + K*y + L*z) / cell_a
        F += f_C(s2, B) * complex(math.cos(phase), math.sin(phase))
    return F

# -------------------------------------------------------------------
# Load PDB and GPU library
# -------------------------------------------------------------------
pdb   = 'rand10.pdb'
lib_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sfcalc_gpu.so')

st   = gemmi.read_structure(pdb)
cell = st.cell
A    = cell.a   # 25 Å cubic

atoms = []
xs, ys, zs, Bs, els = [], [], [], [], []
for model in st:
    for chain in model:
        for res in chain:
            for atom in res:
                p = atom.pos
                atoms.append((p.x, p.y, p.z))
                xs.append(p.x); ys.append(p.y); zs.append(p.z)
                Bs.append(atom.b_iso); els.append(0)   # all Carbon

x_arr = np.array(xs, dtype=np.float32)
y_arr = np.array(ys, dtype=np.float32)
z_arr = np.array(zs, dtype=np.float32)
B_arr = np.array(Bs, dtype=np.float32)
el_arr = np.array(els, dtype=np.int32)
natoms = len(atoms)

# Grid: dmin=1.5, rate=1.5 => spacing = dmin/(2*rate) = 0.5 A => n = ceil(25/0.5) = 50
dmin, rate = 1.5, 1.5
spacing = dmin / (2*rate)

def good_fft_size(n):
    best = n * 10
    i2 = 1
    while i2 <= n*2:
        i3 = i2
        while i3 <= n*2:
            i5 = i3
            while i5 <= n*2:
                if i5 >= n: best = min(best, i5)
                i5 *= 5
            i3 *= 3
        i2 *= 2
    return best

nx = good_fft_size(math.ceil(A/spacing))
ny = good_fft_size(math.ceil(A/spacing))
nz = good_fft_size(math.ceil(A/spacing))
print(f"Grid: {nx}x{ny}x{nz}  (spacing ~{A/nx:.3f} A)")

# -------------------------------------------------------------------
# Call GPU library
# -------------------------------------------------------------------
lib = ctypes.CDLL(lib_p)
lib.spread_and_fft.restype  = ctypes.c_int
lib.spread_and_fft.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]

def fptr(arr): return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
def iptr(arr): return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

map_buf = np.zeros(nx*ny*nz, dtype=np.float32)
nx2 = nx//2 + 1
F_real = np.zeros(nx2*ny*nz, dtype=np.float32)
F_imag = np.zeros(nx2*ny*nz, dtype=np.float32)

nkept = lib.spread_and_fft(
    natoms, fptr(x_arr), fptr(y_arr), fptr(z_arr), fptr(B_arr), iptr(el_arr),
    nx, ny, nz, ctypes.c_float(A), ctypes.c_float(A), ctypes.c_float(A),
    ctypes.c_float(0.0),
    fptr(map_buf), fptr(F_real), fptr(F_imag),
)
print(f"Atoms spread: {nkept}")

V   = A**3
norm = V / (nx*ny*nz)

# cuFFT output: Fc[iz,iy,ix] = F(H=ix, K=iy, L=iz)  (current assumption)
Fc_gpu = (F_real.reshape(nz, ny, nx2).astype(np.float64)
         - 1j * F_imag.reshape(nz, ny, nx2).astype(np.float64)) * norm
# conjugate for FFT sign convention
Fc_gpu = np.conj(Fc_gpu)

# -------------------------------------------------------------------
# Python FFT of GPU map — try two reshapes
# -------------------------------------------------------------------
# Option A: reshape (nz,ny,nx) → correct C-order for memory layout
#   rfftn gives F[ikz,iky,ikx]  with H=ikx (last/fastest)
mapA = map_buf.reshape(nz, ny, nx)
FcA  = np.conj(np.fft.rfftn(mapA)) * norm   # shape (nz,ny,nx//2+1)

# Option B: reshape (nx,ny,nz) → treats data as if X is slowest
mapB = map_buf.reshape(nx, ny, nz)
FcB  = np.conj(np.fft.rfftn(mapB)) * norm   # shape (nx,ny,nz//2+1)

# -------------------------------------------------------------------
# Test reflections
# -------------------------------------------------------------------
test_hkls = [
    (1,0,0),(0,1,0),(0,0,1),
    (3,0,0),(0,3,0),(0,0,3),
    (5,2,0),(0,5,2),(2,0,5),
    (4,3,1),(1,4,3),(3,1,4),
]

print(f"\n{'hkl':>12}  {'direct':>10}  {'GPU-MTZ':>10}  {'PyFFT-A(nz,ny,nx)':>18}  {'PyFFT-B(nx,ny,nz)':>18}")
print("-"*76)

def gpu_mtz(H, K, L):
    """Look up |F| from cuFFT output using current extraction logic."""
    # Current assumption: Fc_gpu[iz,iy,ix] = F(H=ix, K=iy, L=iz)
    # need iz=L, iy=K, ix=H  (H>=0 by R2C half-sphere)
    if H < 0:   # use Friedel mate
        H,K,L = -H,-K,-L
    if H < 0 or H > nx//2: return float('nan')
    iy = K % ny;  iz = L % nz
    return abs(Fc_gpu[iz, iy, H])

def pyfft_A(H, K, L):
    """FcA[iz,iy,ix]: F indexed by (L,K,H)."""
    if H < 0: H,K,L = -H,-K,-L
    if H < 0 or H > nx//2: return float('nan')
    iy = K % ny; iz = L % nz
    return abs(FcA[iz, iy, H])

def pyfft_B(H, K, L):
    """FcB[ix,iy,iz]: rfftn of (nx,ny,nz) — half-complex on last axis (Z)."""
    # half-complex is L here, so F indexed by (H,K,L) with L>=0
    if L < 0: H,K,L = -H,-K,-L
    if L < 0 or L > nz//2: return float('nan')
    ix = H % nx; iy = K % ny
    return abs(FcB[ix, iy, L])

for hkl in test_hkls:
    H, K, L = hkl
    Fd = direct_sum(atoms, hkl, A)
    print(f"({H:2d},{K:2d},{L:2d})  {abs(Fd):10.4f}  {gpu_mtz(H,K,L):10.4f}  "
          f"{pyfft_A(H,K,L):18.4f}  {pyfft_B(H,K,L):18.4f}")
