# sfcalc_gpu — GPU-accelerated structure factor calculation

Computes crystallographic structure factors F(H,K,L) from a PDB file on the GPU
using a multi-level Gaussian-spread density approach, then FFTs to reciprocal space.

Designed for very large systems (100k–500k atoms) where CPU tools are too slow.
Results agree with `gemmi sfcalc` to better than 0.1% RMS for typical B factors.

## Requirements

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU  | Volta (sm_70) or newer recommended |
| CUDA toolkit | nvcc + libcufft |
| CCP4 suite  | provides `ccp4-python` and `gemmi` |

## Quick start

```bash
# 1. Compile the CUDA shared library
bash compile_gpu.sh

# 2. Test 1: 1000-atom random-B test (GPU sfcalc vs gemmi, single P1 cell)
bash run_randb_test.sh

# 3. Test 2: supercell collapse test (GPU sfcalc on P1 supercell → collapse to P41212)
bash run_collapse_test.sh
```

### Test 1 expected output
```
N reflections  : 412068
mean rel diff  : 0.000NNN
RMS  rel diff  : 0.000NNN
max  rel diff  : 0.00NN   at HKL (...)  F1=...  F2=...
```
RMS relative difference should be well below 0.001 (0.1%).

### Test 2 expected output
Same statistics comparing the collapsed GPU result against gemmi sfcalc
on the original P41212 primitive cell. RMS should similarly be < 0.1%.

## Files

| File | Purpose |
|------|---------|
| `sfcalc_gpu.cu`        | CUDA kernel: atom-centric Gaussian spread + cuFFT |
| `sfcalc_gpu.py`        | Python driver: level assignment, GPU dispatch, MTZ output |
| `supercell_collapse`   | Collapse P1 supercell MTZ to primitive-cell SG ASU |
| `compare_mtz.py`       | Compare two MTZ files: mean/RMS/max relative F difference |
| `make_randb_pdb.py`    | Generate 1000-atom test PDB with random B factors |
| `make_supercell_pdb.py`| Tile a unit-cell PDB into a P1 supercell |
| `P1test_B20.pdb`       | 1000-atom P1 test structure (base coordinates) |
| `P41212.pdb`           | 2870-atom P41212 test structure (full unit cell, 20×20×30 Å) |
| `compile_gpu.sh`       | Build sfcalc_gpu.so |
| `run_randb_test.sh`    | Test 1: P1 random-B end-to-end test |
| `run_collapse_test.sh` | Test 2: supercell tiling → GPU sfcalc → collapse → compare |

## Supercell collapse workflow

The target application is diffuse X-ray scatter from MD simulations:

```
I_diffuse(H) = <|F(H)|²> - |<F(H)>|²
```

MD simulations run in a P1 supercell (na×nb×nc copies of the crystal unit cell).
After calculating structure factors for each frame with `sfcalc_gpu.py`, the
P1 supercell F values are collapsed to the primitive-cell ASU using
`supercell_collapse`:

```
F_SG(H,K,L) = Σ_i  exp(2πi H·t_i) * F_super(na·R_i^T·H, nb·K_rot, nc·L_rot)
```

where the sum is over all symmetry operators of the small space group SG,
R_i^T is the transposed direct-space rotation, and t_i is the translation.

## How it works

Atoms are sorted into logarithmically spaced "levels" by B factor
(spacing factor √2).  Each level uses a grid fine enough for that B factor.
The electron density from all levels is accumulated and FFT'd to give F(H,K,L).

The atom-centric spreading kernel assigns one GPU thread per atom.
Each thread scatters to all nearby voxels within its cutoff radius via `atomicAdd`.
This avoids the O(N_atoms × N_voxels) voxel-centric approach.

## Usage

### sfcalc_gpu.py
```
ccp4-python sfcalc_gpu.py  input.pdb
    [dmin=1.5]   [rate=2.5]
    [outmtz=sfcalc_gpu.mtz]  [outmap=sfcalc_gpu.map]
    [bmax=0]     [noise=0.01]  [lib=sfcalc_gpu.so]
```

- `dmin`  : resolution limit in Å (default 1.5)
- `rate`  : FFT oversampling rate (default 2.5, higher = more accurate)
- `bmax`  : skip atoms with B > bmax (0 = keep all)
- `noise` : small noise floor added to avoid divide-by-zero in comparisons

### supercell_collapse
```
ccp4-python supercell_collapse  super.mtz  SPACEGROUP
    super_mult=na,nb,nc  [dmin=VALUE]  [outfile=collapsed.mtz]
```

- `super.mtz`      : P1 supercell MTZ from sfcalc_gpu.py (FC + PHIC columns)
- `SPACEGROUP`     : target primitive-cell space group  (e.g. P41212, P212121)
- `super_mult`     : supercell multipliers along a, b, c  (e.g. 3,3,3)

### make_supercell_pdb.py
```
ccp4-python make_supercell_pdb.py  input.pdb
    [super_mult=na,nb,nc]  [out=supercell.pdb]
```
Input PDB must contain the **complete unit cell** (all symmetry copies in P1).
