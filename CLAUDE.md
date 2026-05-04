# Claude Code Guide — lunus

## Project Overview

LUNUS is a software package for analyzing X-ray diffuse scattering from protein crystals. The core workflow is:

1. **MD trajectory** (AMBER `.nc` + topology PDB) → `xtraj.py` → structure factor statistics
2. **Diffuse scatter**: `I_diffuse(H) = <|F(H)|²> - |<F(H)>|²`, averaged over MD frames

The primary performance-critical script is `lunus/command_line/xtraj.py`, which reads each MD frame, computes structure factors, and accumulates the running sums for the diffuse calculation.

## Key Files

| Path | Purpose |
|------|---------|
| `lunus/command_line/xtraj.py` | **Main script** — MD trajectory → diffuse scatter HKL + MTZ |
| `md2mtz/sfcalc_gpu.so` | GPU FFT library (precompiled for NVIDIA Volta–Hopper) |
| `md2mtz/libcufft.so.11` | Bundled cuFFT runtime |
| `md2mtz/sfcalc_gpu.cu` | CUDA source for the GPU kernel |
| `md2mtz/CLAUDE.md` | Detailed guide for the GPU library / sfcalc_gpu_collapse tool |

## Environment

- **Python**: `lunus` conda env at `~/projects/lunus/miniconda3/envs/lunus`
- **Activation**:
  ```bash
  source ~/projects/lunus/miniconda3/etc/profile.d/conda.sh
  conda activate lunus
  export PATH=~/projects/lunus/lunus/c/bin/:$PATH
  ```
- **MPI**: `mpirun -n <N> python xtraj.py ...` for parallel frame processing
- **SLURM**: GPU partition is `gpu`; voltron has 7× GV100 GPUs (`gpu:GV100`)
- **Shell**: tcsh on voltron. Write bash scripts for SSH execution; don't mix shells.

## xtraj.py Parameters

```
top=<file>        topology PDB (atom names, elements, B-factors)
traj=<file>       trajectory file (AMBER .nc or other mdtraj-readable format)
first=0           first frame index (0-based)
last=N            last frame index (inclusive)
stride=1          frame stride
d_min=0.9         resolution cutoff in Å (default 0.9)
engine=cctbx      structure factor engine: cctbx (default) or gpu
diffuse=<file>    output diffuse scatter as space-separated HKL text
fcalc=<file>      output <F> as MTZ
icalc=<file>      output <|F|²> as MTZ
```

GPU-specific parameters (used when `engine=gpu`):

```
lib=<path>        path to sfcalc_gpu.so (default: md2mtz/sfcalc_gpu.so)
rate=2.5          FFT grid oversampling factor (default 2.5)
noise=0.01        multi-grid noise tolerance fraction (default 0.01 = 1%)
bmax=0.0          skip atoms with B > bmax (0 = keep all)
super_mult=1,1,1  supercell multipliers na,nb,nc (default 1,1,1)
```

## GPU Engine Design

`engine=gpu` replaces cctbx direct-sum `structure_factors()` with GPU FFT spreading:

1. **Multi-level spreading**: atoms binned by B-factor onto coarser grids; high-B atoms use fewer grid points. Controlled by `noise` and `rate`.
2. **Auto-blur correction**: `b_add = (dmin·rate)²/π²` added to all B-factors before spreading; divided out after FFT via `exp(b_add·stol²)`. Prevents sub-pixel Gaussian aliasing.
3. **ASU enumeration**: `cctbx.miller.build_set()` (no gemmi dependency).
4. **Symmetry collapse**: `F_SG(H) = Σ_op exp(2πi H·t_op) · F_super(R^T_op·H)` using `sg.all_ops()`.

All GPU code is self-contained — no gemmi, no CCP4. Only `ctypes`, `numpy`, and `cctbx`.

## Validation Results

| Resolution | Common reflections | Pearson CC (CPU vs GPU diffuse) |
|------------|-------------------|---------------------------------|
| d_min=2.0 | 1,146,362 | **0.9987** |
| d_min=0.9 | 12,581,024 | **0.9985** |

Test system: 469,326 atoms, 3 frames each.

## Running Tests

```bash
# GPU-only sanity check (5 frames, d_min=0.9)
bash md2mtz/run_gpu_test.sh

# CPU vs GPU correlation at d_min=2.0 (quick, ~2 min)
bash md2mtz/run_compare_test.sh

# CPU vs GPU correlation at d_min=0.9 (submit to SLURM)
sbatch md2mtz/slurm_compare_dmin09.sh
```

The SLURM script targets voltron (`--nodelist=voltron`, `--gres=gpu:GV100:1`) and writes output to `md2mtz/slurm_compare_dmin09_<jobid>.out`.

## MPI Parallel Execution

xtraj.py uses MPI (via `mpi4py`) when launched with `mpirun`. Each rank processes a subset of frames; rank 0 collects and writes output. The GPU engine is compatible with MPI — each rank uses the GPU independently.

```bash
mpirun -n 8 python lunus/command_line/xtraj.py top=... traj=... engine=gpu lib=md2mtz/sfcalc_gpu.so
```

## Building the GPU Library

The GPU library must be compiled on a machine with the CUDA toolkit (voltron):

```csh
ssh voltron "cd /home/jamesh/projects/lunus/lunus/md2mtz ; tcsh compile_gpu.csh"
```

See `md2mtz/CLAUDE.md` for full details on the GPU library build and test.
