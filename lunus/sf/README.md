# lunus.sf — differentiable structure factors and diffuse scattering

A PyTorch implementation of a **gemmi-like** engine: splat atomic Gaussian
density onto a real-space grid → symmetrize → FFT → extract F(hkl), made
differentiable with respect to atomic fractional coordinates.

Gemmi-like rather than a clone — the density cutoff is tapered smoothly rather
than truncated hard, so that gradients are well behaved. See
[docs/design.md](docs/design.md) for what matches gemmi and what deliberately
does not.

The point is the gradients. On CPU, gemmi is faster per core (see
[docs/performance.md](docs/performance.md)); what this buys you is
`loss.backward()` reaching every atom in every configuration.

## Installing

`lunus.sf` installs as an ordinary Python package — **no cctbx build
required**. The repository root's `pyproject.toml` packages the pure-Python
tree (`lunus`, `lunus.sf`, `lunus.md`, `lunus.command_line`); the C library,
the command-line tools and the `lunus_ext` Boost extension are built
separately by SCons or libtbx and are unaffected by it.

```bash
pip install -e ".[sf]"        # numpy + torch + gemmi
pip install -e ".[sf,xtraj]"  # adds mdtraj, scipy, h5py for the MD pipeline
```

With **pixi**, the conda-forge stack supplies pytorch/gemmi/cctbx/mdtraj/mpi4py
and the project is installed on top of it — `[tool.pixi]` tables are already in
`pyproject.toml`, in a single environment so a plain `pixi run` works:

```bash
pixi install
pixi run test                    # the lunus.sf test suite
pixi run compare                 # both engines end to end, then compare
pixi shell                       # then: python -c "import lunus.sf"

cd lunus/sf/examples/compare_gemmi
pixi run bash run_xtraj.sh torch
```

cctbx is deliberately not a PyPI dependency: it installs far more reliably
from conda-forge, and `elements.py` needs only one of gemmi *or* cctbx.

### What needs what

numpy and torch for the core; **cctbx only for `xtraj.py` and the comparison
tools**; gemmi (or cctbx) for the scattering coefficients, and gemmi for the
parity checks. `lunus/__init__.py` imports the `lunus_ext` Boost extension
inside a `try/except` precisely so this stays true — code that genuinely needs
`Process` or `LunusDIFFIMAGE` still fails loudly, at first use.

## Importing

The public API is available from the package root, resolved lazily so a bare
`import lunus.sf` costs ~15 ms and does not import torch at all:

```python
from lunus.sf import structure_factors_batch, mean_and_diffuse
```

Submodules work too, if you prefer to be explicit:

```python
from lunus.sf.kernel_torch import build_element_kernels_torch
```

Working from a clone without installing also works, with the repository root
on `sys.path` — which a cctbx module build arranges for you, and which
`tests/conftest.py` does for the test suite.

## Quickstart

```bash
cd lunus/sf
python -m pytest tests/ -q                     # 44 passed, 2 skipped

PYTHONPATH=<repo root> python examples/example_usage.py       # forward + backward
mpirun -n 4 python examples/example_mpi_training.py           # MPI training step

cd examples/compare_gemmi                      # end-to-end vs gemmi
./run_xtraj.sh gemmi && ./run_xtraj.sh torch
python ../../tools/compare_icalc_mtz.py icalc_gemmi.mtz icalc_torch.mtz
# expect correlation 0.999989, R-factor 0.0077
```

## Layout

### Library

| module | role |
|---|---|
| `elements.py` | Reads IT92 neutral-atom Cromer-Mann coefficients from gemmi or cctbx at run time. Stores no table of its own, so any element those libraries know works without a code change. |
| `kernel.py` | The real-space density kernel: reciprocal-space coefficients + effective B → the 5-Gaussian real-space form, derived from the closed-form Fourier transform of the Cromer-Mann sum (derivation in the module docstring). Also the per-atom (batched) variants. |
| `cell_utils.py` | Orthogonalization matrix, 1/d², grid-size-from-resolution (matches gemmi's `requested_grid_spacing() = d_min / (2*rate)`). |
| `density_torch.py` | The differentiable splat. Grouped by element, batched by atom-voxel PAIRS, restricted to the cutoff sphere rather than its bounding cube, with the hot blocks factored for `torch.compile` fusion. |
| `kernel_torch.py` | Builds per-element or per-atom (A, lam, radius) as torch tensors. Not differentiated — B and blur are fixed hyperparameters. |
| `symmetry_torch.py` | Space group → validated integer **grid** operations, applied differentiably. Also `adjust_grid_for_symmetry()` for the grid-size constraints those operations impose. |
| `structure_factor_torch.py` | FFT + Miller-index extraction, matching xtraj.py's `ifftn(...) * volume` then `conjugate(...)` convention including the optional blur-undo. `structure_factors_batch()` runs N configurations in one call with gradients to all N (optionally gradient-checkpointed, so memory stays flat in N), and `mean_and_diffuse()` does the `<F>` / `<\|F\|²> − \|<F>\|²` ensemble reduction. |
| `training_mpi.py` | MPI-distributed training step. Configurations split across ranks with no `torch.distributed`: one small `Allreduce` on the aggregate sums, a tiny shared graph to produce the upstream gradient, then each rank finishes its own backward locally. No per-atom gradient communication, since each configuration's atoms belong to one rank alone. |
| `xtraj.py` | MD trajectory → `<F>` and diffuse intensity. **This is the live xtraj variant** (engines: `gemmi`, `sfall`, `torch`). `lunus/command_line/xtraj.py` is the separate shipped dispatcher and has no torch engine; merging them is deliberately out of scope for now. |

### tests/

`python -m pytest tests/ -q`. Thresholds come from measured values, recorded in
each test's docstring. Characterization tests carry two-sided bounds so
behaviour changing in either direction trips them.

| file | what it pins |
|---|---|
| `test_numpy_reference.py` | splat+FFT vs a closed-form single-atom F(hkl), values and gradients. The NumPy ground truth the torch code is a translation of. |
| `test_torch_pipeline.py` | torch reproduces the NumPy reference; autograd matches a finite difference of the analytic formula. |
| `test_multiatom.py` | multi-element superposition, and that chunk size changes nothing (chunking is a pure memory optimization). |
| `test_sphere_mask.py` | restricting the box to the cutoff sphere is lossless, and actually discards ~37% of the cube. |
| `test_exact_cutoff.py` | the exact cutoff mask against the margin-padded one; asserts it can only *remove* density. |
| `test_per_atom_b.py` | per-atom B end to end (see the caveat in docs/design.md). |
| `test_taper.py` | the tapered gradient converges monotonically as eps shrinks, unlike a hard cutoff. |
| `test_symmetry.py` | parity against gemmi's own `symmetrize_sum()` across P1/P4₃/P2₁2₁2₁/P6₁/I23, fast path vs gather path, orbit invariance, autograd, and grid adjustment. |
| `test_gradient_seeding.py` | the MPI reduce-then-seed scheme is identical to one combined backward pass. |

`conftest.py` puts the repository root on `sys.path` and skips on missing
torch/cctbx/gemmi/mpi4py, so a partial environment skips rather than errors.

### tools/ and examples/

| | |
|---|---|
| `tools/bench_splat.py` | splat benchmark, gemmi vs torch, CPU/MPS/CUDA. The thing to re-run rather than trusting docs/performance.md. |
| `tools/compare_icalc_mtz.py` | two Icalc MTZs → correlation and R-factor, overall and by resolution shell. |
| `tools/compare_density.py` | two real-space density grids (needs `save_density=True`). |
| `tools/make_random_bfacs.py` | writes a PDB with a random B per residue, for exercising the per-atom-B path. Seeded. |
| `tools/analyze_trace.py` | a `torch.profiler` Chrome trace (from `torch_profile_frames=N`) → GPU busy vs idle, the gap size distribution, what each large gap was blocked in, and a test for cgroup CPU-throttling. Standard library only. |
| `examples/example_usage.py` | several configurations → `<F>`/diffuse → scalar loss → `backward()` → gradients on every atom. |
| `examples/example_mpi_training.py` | the same under `mpirun`, via `training_mpi.py`. |
| `examples/demo_taper_gradient.py` | why a hard cutoff is unusable for gradients. |
| `examples/compare_gemmi/` | end-to-end torch-vs-gemmi run, with its tracked input data. |

`examples/compare_gemmi/` carries `top_bfac.pdb.gz` (135,834 atoms) and
`traj.xtc` — the system every measured number in docs/ was taken on, kept in the
repo so those numbers stay reproducible. Both `iotbx.pdb` and `mdtraj` read
`.pdb.gz` directly. Its outputs are gitignored.

## Further reading

- **[docs/design.md](docs/design.md)** — what matches gemmi exactly and what
  deliberately does not, the assumptions inherited from `xtraj.py`, per-atom
  B-factors, the three-iteration history behind density-space symmetry
  expansion, and measured agreement with gemmi.
- **[docs/performance.md](docs/performance.md)** — benchmarks (all labelled with
  the configuration they were measured on), Apple MPS and NVIDIA CUDA, where
  the per-frame time actually goes, an unresolved splat discrepancy, how to
  measure this pipeline without fooling yourself, and the tuning knobs.
- **[docs/bugfixes.md](docs/bugfixes.md)** — bugs found and fixed, several of
  which produced wrong numbers rather than a traceback. Kept because the
  reasoning that exposed them is worth more than the diffs.

## Not yet built

- **Tune `structure_factors_batch()` for speed within a memory budget.**
  `use_checkpoint` is currently all-or-nothing: retain every member's splat
  intermediates (fastest, memory linear in N) or recompute all of them
  (memory flat in N, ~2.4x slower). It was built to make large N *possible*,
  not fast, so it takes the memory-minimising extreme whenever N does not fit.

  The optimum is in between: retain as many members as the device can hold and
  checkpoint only the remainder. Memory then scales with the retained count
  rather than N, and the recompute penalty falls to roughly the fraction
  checkpointed — at N=16 on a device with room for 8, that is half the penalty
  for the same peak. A `max_retained` (or a byte budget queried from
  `torch.cuda.mem_get_info()`) would let it pick automatically instead of
  making the caller choose between two extremes.

  Worth measuring first: the per-member cost is dominated by the splat, and on
  CUDA the splat's in-loop cost is not yet understood (see
  docs/performance.md), so calibrate against real timings rather than assuming
  the recompute penalty is a clean 2.4x at every N.
- **Fuse the batched splat into a single kernel.** `structure_factors_batch()`
  currently shares setup across members and loops the splat; fusing the N
  splats into one launch would help GPU throughput for small systems, where
  per-launch overhead is significant. It would not help memory — the
  checkpointing path already makes that flat in N — and it would touch the
  carefully tuned `_density_core` / `_scatter_core` fast path, so it is worth
  doing only if profiling shows launch overhead actually dominating.
- **Port the MPI reduction to `torch.distributed`** for multi-GPU
  data-parallelism over configurations.
- **Wire up a real loss** against experimental diffuse data; `example_usage.py`
  currently has a placeholder.
- **Anisotropic ADPs** — the anisotropic case needs a different,
  tensor-valued kernel, derived the same way as the isotropic one.
- **A grid-refinement convergence check** in `test_per_atom_b.py`; see
  docs/design.md.
