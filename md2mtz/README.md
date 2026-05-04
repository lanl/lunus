# sfcalc_gpu_collapse — GPU Structure Factors for Diffuse Scatter

GPU-accelerated calculation of crystallographic structure factors from MD supercell trajectories, for diffuse scatter analysis.

## Background

X-ray diffuse scatter carries information about correlated atomic motion that is invisible to conventional Bragg analysis. For an MD ensemble of supercell frames, the diffuse intensity is:

```
I_diffuse(H) = <|F_SG(H)|²> - |<F_SG(H)>|²
```

where `H = (H,K,L)` is a primitive-cell Miller index and the angle brackets denote averaging over MD frames. Computing this requires two quantities per frame:

- `|F_SG(H)|²` — squared amplitude in the primitive-cell ASU
- `F_SG(H)` — complex structure factor (amplitude + phase) in the primitive-cell ASU

This software calculates both efficiently using GPU FFT.

## Method

Each MD frame is stored as a P1 supercell PDB (`na × nb × nc` copies of the primitive cell). The GPU spreads atoms onto a density grid and computes the 3-D FFT, giving `F_super(h,k,l)` for the supercell in P1. These are then **collapsed** to the primitive-cell ASU using the space-group symmetry operators:

```
F_SG(H,K,L) = Σ_op  exp(2πi H·t_op)  ×  F_super(na·R_op^T·H, nb·R_op^T·K, nc·R_op^T·L)
```

where `R_op` and `t_op` are the rotation and translation of each symmetry operator (fractional coordinates). The collapse is exact for any space group in any crystal system.

Two MTZ files are written per frame:

| Output | Contents | Used for |
|--------|----------|----------|
| `outI` | Supercell P1 intensities `I = \|F_super\|²` at all h,k,l to `dmin` | Computing `<\|F\|²>` by averaging I across frames |
| `outmtz` | Primitive-cell ASU amplitudes FC + phases PHIC | Computing `\|<F>\|²` by averaging complex F across frames, then squaring |

## Requirements

- **NVIDIA GPU**, compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **CCP4 suite** — only needed to run `sfcalc_gpu_collapse.py` (Python alternative); not required for the C++ executable
- **CUDA driver** 520+ (CUDA 11 runtime is bundled in `libcufft.so.11`)
- **GCC 7+** for building `sfcalc_gpu_collapse.cpp` (devtoolset-7 on RHEL/CentOS 7)
- `sfcalc_gpu.so`, `libcufft.so.11`, and `sfcalc_gpu_collapse` in the same directory (see Build)

## Build

Compile once on a machine with the CUDA toolkit installed (e.g. voltron):

```csh
tcsh compile_gpu.csh
```

This produces:
- `sfcalc_gpu.so` (multi-arch: sm_70 through sm_90 + PTX fallback)
- `sfcalc_gpu_collapse` (C++ executable, no external runtime dependencies)

**Distribution**: ship `sfcalc_gpu.so`, `libcufft.so.11`, and `sfcalc_gpu_collapse` together in the same directory. The `$ORIGIN` rpath means neither the CUDA toolkit nor the CCP4 suite needs to be installed on the target machine — only an NVIDIA driver is required.

## Usage

```
./sfcalc_gpu_collapse  <input.pdb>
    [dmin=1.5]          resolution limit in Angstroms (default 1.5)
    [rate=2.5]          Shannon rate (grid oversampling, default 2.5)
    [sg=P1]             space group of the PRIMITIVE cell
    [super_mult=1,1,1]  supercell multipliers na,nb,nc
    [outmtz=collapsed.mtz]   primitive-cell ASU phased MTZ
    [outI=supercell_I.mtz]   supercell P1 intensity MTZ
    [outmap=]           optional: write electron density map (CCP4 format)
    [bmax=0]            skip atoms with B > bmax (0 = keep all)
    [noise=0.01]        multi-grid noise tolerance (default 1%)
    [lib=sfcalc_gpu.so] path to the GPU shared library
```

### Example: orthorhombic 2×2×2 supercell

```csh
./sfcalc_gpu_collapse  frame_001.pdb \
    sg="P 21 21 21"  super_mult=2,2,2  dmin=2.0 \
    outmtz=frame_001_collapsed.mtz  outI=frame_001_I.mtz
```

The input PDB must be a P1 supercell with `CRYST1` giving the supercell dimensions. The `sg` argument specifies the space group of the **primitive** unit cell. The script derives the primitive-cell dimensions by dividing the supercell lengths by `na`, `nb`, `nc`.

### P1 single-cell calculation

If `super_mult=1,1,1` and `sg=P1`, both outputs cover the same P1 ASU reflections and the collapse step is skipped.

### Python alternative

The Python script `sfcalc_gpu_collapse.py` accepts identical arguments and produces identical output, but has ~400 ms additional startup overhead per frame (Python + gemmi import):

```csh
ccp4-python sfcalc_gpu_collapse.py  frame_001.pdb \
    sg="P 21 21 21"  super_mult=2,2,2  dmin=2.0 \
    outmtz=frame_001_collapsed.mtz  outI=frame_001_I.mtz
```

## Averaging over an MD trajectory

For each frame `i`, run the script and save both MTZ files. Then:

1. **`<|F|²>`**: average the `I` column across all `outI` MTZs (standard MTZ merging).
2. **`|<F>|²`**: sum the complex structure factors (FC, PHIC) from all `outmtz` files, divide by N to get the mean complex F, then square the amplitude.
3. **`I_diffuse = <|F|²> - |<F>|²`** at each primitive-cell ASU reflection.

CCP4 programs `SCALEIT`, `CAD`, and `ADDUP` can assist with step 1. Steps 2–3 are most conveniently done in Python with gemmi or cctbx.

## Multi-grid B-factor treatment

Atoms are assigned to FFT grid levels based on their isotropic B-factor, so high-B (diffuse) atoms are spread on coarser grids and low-B (well-ordered) atoms use the full fine grid. This reduces GPU memory and time without sacrificing accuracy. The `noise` parameter controls the coarsening threshold (default 1%).

## Accuracy

Validated against gemmi's direct-sum structure factor calculator on a 1000-atom, 170 Å orthorhombic P1 cell at `dmin=2.0` Å (585 k reflections):

| F bin | N reflections | Mean rel. diff | Max rel. diff |
|-------|--------------|----------------|---------------|
| F > 100 | 130 004 | **0.028%** | 0.4% |
| 10 < F ≤ 100 | 445 099 | 0.13% | 4.0% |
| 1 < F ≤ 10 | 9 994 | 1.0% | 23% |
| F ≤ 1 | 102 | 7.7% | 47% |

Errors at weak reflections are dominated by near-zero denominators, not systematic bias. All 230 space groups pass an independent collapse test vs `gemmi sfcalc` (mean <0.5%, max <5%).

### Accuracy vs phenix.fmodel

`phenix.fmodel` was run on the same 1000-atom P1 cell and its output compared against the same gemmi direct-sum reference, for two B-factor distributions:

**Uniform B = 20 Å²** (all atoms identical):

| F bin | N | phenix mean | GPU mean |
|-------|---|-------------|----------|
| F ≥ 100 | 114 041 | 0.53% | **0.03%** |
| [10, 100) | 964 757 | 1.23% | 0.13% |
| [1, 10) | 80 458 | 6.4% | 1.0% |

**Random B = 2–998 Å²**:

| F bin | N | phenix mean | GPU mean |
|-------|---|-------------|----------|
| F ≥ 100 | 8 860 | 0.24% | **0.03%** |
| [10, 100) | 580 297 | 0.46% | 0.13% |
| [1, 10) | 561 304 | 1.29% | 1.0% |

phenix is ~17× less accurate than this code for strong reflections. Notably, phenix accuracy is *better* for the random-B structure than for uniform B=20 — the opposite of what a high-B cutoff radius problem would produce.

**Root cause: phenix uses a 2.25× coarser FFT grid.**

| | phenix.fmodel | this code |
|--|---------------|-----------|
| Grid (170 Å cell, dmin=2.0) | 256×256×240 | 576×576×512 |
| Pixel size | 0.665 Å | 0.296 Å |
| b_add (auto-blur) | 0.91 Å² | 2.53 Å² |
| σ/pixel for narrowest C Gaussian (B=10) | **0.55** (sub-pixel) | **1.37** (above pixel) |

Phenix's `grid_resolution_factor=1/3` sets the pixel to dmin/3 = 0.667 Å. This code uses `rate=2.5`, which sets the pixel to dmin/5 = 0.40 Å (576 is the next FFT-friendly grid size, giving 0.30 Å actual pixel). The narrowest IT92 Gaussian for a carbon atom (b₃=0.57 Å²) at B=10 has σ=0.37 Å, which is sub-pixel on phenix's grid even after the auto-blur correction (b_add=0.91 Å²). Sub-pixel Gaussians alias regardless of b_add. This code's finer grid keeps all Gaussians above the pixel size after auto-blur (σ_eff/pixel ≥ 1.37).

Switching phenix to `scattering_table=it1992` (matching gemmi's form factors) reduced the F≥100 error only from 0.53% to 0.36%, confirming that form factor choice is a minor contributor. The dominant cause is the coarser grid.

### Aliasing correction

The IT92 form factors include a constant term `c` (effective `b = 0`) that would be sub-pixel at low B-factors, causing ~0.7% aliasing error per Å² of B. The code applies **auto-blur** (identical to gemmi's approach): a correction `b_add = (dmin·rate)²/π²` is added to every atom's B-factor before spreading, and the resulting `exp(−b_add·stol²)` envelope is divided out of each F(H) after the FFT. At `dmin=2.0`, `rate=2.5`: `b_add = 2.53 Å²`, `σ_min = 0.25 Å` (pixel = 0.40 Å).

## Performance

Benchmarked on an NVIDIA Volta GPU (sm_70) with a 1000-atom 170 Å orthorhombic P1 cell, `dmin=2.0`, `rate=2.5`:

| Implementation | Total wall time | Spreading | FFT |
|----------------|----------------|-----------|-----|
| gemmi sfcalc (CPU, single-thread) | 2230 ms | — | — |
| Python + float64 (D2Z) | 2843 ms | 1626 ms | 11.2 ms |
| Python + float32 (R2C) | 1912 ms | 1129 ms | 5.3 ms |
| **C++ + float32 (R2C)** | **1040 ms** | 1129 ms | 5.3 ms |

### Scaling with atom count

Same cell (170 × 170 × 153 Å, `dmin=2.0`, grid 576×576×512), Volta GPU (sm_70):

| Atoms | gemmi sfcalc (CPU) | phenix.fmodel (CPU) | GPU C++ R2C | GPU speedup |
|-------|--------------------|---------------------|-------------|-------------|
| 1 000 | 2.2 s | 13 s | 1.0 s | 2× vs gemmi |
| 10 000 | 80 s | 19 s | ~3.4 s | ~6× vs phenix |
| 473 000 ¹ | 39 min | 77 s | ~2.6 s | ~30× vs phenix |

¹ GPU run with `bmax=100` (296k/473k atoms kept); CPU codes processed all 473k.

phenix.fmodel uses FFT-based spreading internally and outperforms gemmi sfcalc at large atom counts, but carries ~13 s of Python/PHIL startup overhead. The GPU kernel eliminates both startup cost and the CPU spreading bottleneck, with growing advantage as atom count increases.

### Implementations compared (1 000-atom baseline)

The Python → C++ port eliminates ~400 ms of Python/gemmi import overhead per frame. Switching from float64/D2Z to float32/R2C halves all device memory sizes (cudaMalloc, cudaMemset, PCIe transfer). The float32 noise floor (~10⁻⁷) is 2500× below the 0.03% grid-discretisation error, so accuracy is unaffected.

For a 10 000-atom MD trajectory at `dmin=2.0` with a 2×2×2 supercell (~150 Å cell), expect roughly 5–10 s per frame on a modern NVIDIA GPU.

## Testing

Validate the GPU collapse against a gemmi reference for any space group:

```csh
# Single space group (by number or HM name)
ccp4-python test_one_sg.py 19
ccp4-python test_one_sg.py "P 21 21 21"  dmin=2.0

# All 230 space groups in parallel
ccp4-python test_all_sg.py
```

Pass criteria: ≥90% of gemmi reflections found, mean relative amplitude difference <0.5%, max <5%.

## Files

| File | Purpose |
|------|---------|
| `sfcalc_gpu_collapse` | **Main executable** (C++, no Python required) |
| `sfcalc_gpu_collapse.cpp` | C++ source for the main executable |
| `sfcalc_gpu_collapse.py` | Python equivalent (identical interface, ~400 ms slower startup) |
| `sfcalc_gpu.cu` | CUDA C source for GPU FFT kernel |
| `sfcalc_gpu.so` | Compiled GPU shared library |
| `libcufft.so.11` | cuFFT runtime (bundled for portability) |
| `compile_gpu.csh` | Build script (run on GPU machine with CUDA toolkit) |
| `test_one_sg.py` | Per-space-group validation |
| `test_all_sg.py` | Full 230 SG test suite |
