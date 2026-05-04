# LUNUS — Diffuse Scattering Analysis

LUNUS software for analyzing X-ray diffuse scattering from protein crystals.

Los Alamos National Laboratory C21101

## Overview

The central tool is `xtraj.py`, which reads an MD trajectory and computes the diffuse scatter signal:

```
I_diffuse(H) = <|F(H)|²> - |<F(H)>|²
```

where the angle brackets denote averaging over MD frames and `H = (H,K,L)` is a Miller index in the crystal's ASU.

## Quick Start

```bash
source ~/projects/lunus/miniconda3/etc/profile.d/conda.sh
conda activate lunus
export PATH=~/projects/lunus/lunus/c/bin/:$PATH

python lunus/command_line/xtraj.py \
  top=my_topology.pdb \
  traj=my_trajectory.nc \
  first=0 last=99 d_min=0.9 \
  engine=gpu \
  lib=md2mtz/sfcalc_gpu.so \
  diffuse=diffuse.hkl \
  fcalc=fcalc.mtz \
  icalc=icalc.mtz
```

For MPI parallel execution:
```bash
mpirun -n 8 python lunus/command_line/xtraj.py top=... traj=... engine=gpu ...
```

## Structure Factor Engines

### `engine=cctbx` (default)

Uses CCTBX direct-sum structure factor calculation. Accurate but scales as O(N\_atoms) per reflection.

### `engine=gpu` (recommended)

GPU-accelerated FFT spreading. Spreads atoms onto a 3-D density grid, computes the FFT, then collapses to the primitive-cell ASU using space-group symmetry operators. Requires an NVIDIA GPU (Volta or newer).

**Performance** (469k atoms, d\_min=0.9, 3 frames):

| Engine | Time |
|--------|------|
| cctbx (CPU) | ~9 min |
| GPU (GV100) | ~1.5 min |

**Accuracy** (Pearson CC between CPU and GPU diffuse):

| d\_min | Reflections | CC |
|--------|-------------|----|
| 2.0 Å | 1,146,362 | 0.9987 |
| 0.9 Å | 12,581,024 | 0.9985 |

### GPU Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lib` | `md2mtz/sfcalc_gpu.so` | Path to the GPU shared library |
| `rate` | 2.5 | FFT grid oversampling factor |
| `noise` | 0.01 | Multi-grid noise tolerance (fraction) |
| `bmax` | 0.0 | Skip atoms with B > bmax (0 = keep all) |
| `super_mult` | 1,1,1 | Supercell multipliers `na,nb,nc` |

### Supercell Trajectories

If the trajectory contains a supercell (`na × nb × nc` copies of the primitive cell), set `super_mult=na,nb,nc`. The GPU engine collapses supercell F values to the primitive-cell ASU via symmetry:

```
F_SG(H) = Σ_op  exp(2πi H·t_op)  ×  F_super(na·R^T_op·H, nb·R^T_op·K, nc·R^T_op·L)
```

## GPU Library

The precompiled GPU library (`md2mtz/sfcalc_gpu.so`) targets NVIDIA Volta through Hopper (sm\_70–sm\_90). `libcufft.so.11` is bundled in `md2mtz/` for portability — no CUDA toolkit installation required on the compute node, only a compatible NVIDIA driver.

To recompile from source (requires CUDA toolkit on voltron):
```csh
ssh voltron "cd /path/to/lunus/md2mtz ; tcsh compile_gpu.csh"
```

See `md2mtz/README.md` for full details on the GPU library.

## Repository Structure

```
lunus/command_line/xtraj.py   Main MD-to-diffuse script
lunus/                        Python package (cctbx-based routines)
c/                            C source and binaries for diffuse routines
md2mtz/                       GPU FFT library and test suite
  sfcalc_gpu.so               Compiled GPU kernel (ctypes interface)
  libcufft.so.11              Bundled cuFFT runtime
  sfcalc_gpu.cu               CUDA source
  sfcalc_gpu_collapse.cpp     C++ standalone executable source
  README.md                   Detailed GPU library documentation
examples/                     Example datasets and scripts
scripts/                      Utility scripts
test/                         Test suite
```

## Dependencies

- **Python**: CCTBX, mdtraj, numpy, scipy, mpi4py
- **GPU engine**: NVIDIA driver (Volta/GV100 or newer recommended); no CUDA toolkit needed at runtime
- **MPI**: OpenMPI or MPICH for parallel execution

## License

BSD license. See `README` for full license text.

Copyright (c) 1993–1997, Michael E. Wall  
Copyright (c) 2007–2015, Los Alamos National Security, LLC  
Copyright © 2022. Triad National Security, LLC. All rights reserved.
