# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

LUNUS is a toolkit for analyzing and modeling **diffuse X-ray scattering** from protein crystals (LANL, BSD-licensed). It has three layers that can be used independently:

1. A **C library + ~280 small command-line tools** (`c/`) for image and reciprocal-lattice processing.
2. A **Boost.Python extension** (`lunus/lunus_ext.cpp`) that exposes the C processing pipeline to Python as a cctbx/libtbx module.
3. **Python command-line programs** (`lunus/command_line/`) built on cctbx/DIALS/mdtraj for integration, MD-trajectory structure-factor calculation, and simulation.

## Building

There are two independent build paths; which one you want depends on whether you need the Python extension.

**Standalone C tools only** (top-level `SConstruct`, no cctbx needed):

```bash
scons                              # -> c/lib/liblunus.a and c/bin/lunus.*
scons enable-openmp=1              # OpenMP threading
scons enable-cuda=1                # Linux only; also builds lunus/cuda
scons enable-kokkos=1              # Linux only; also builds lunus/kokkos
export PATH=$PATH:$PWD/c/bin
```
`CC` / `AR` env vars select the compiler/archiver (default `gcc`/`ar`).

**As a cctbx module** (`SConscript` + `libtbx_config`): place the repo inside a cctbx `modules/` directory and run `libtbx.configure lunus; make`. `libtbx_config` requires `boost`, `scitbx`, `dxtbx` at build time and `scitbx` at run time. This path builds `lib/lunus_ext.so`, the `bin/lunus.*` C tools, and the `lunus.*` Python dispatchers (see "Dispatchers" below).

The legacy `c/src/makemake` + `Makefile.head.<platform>` build still exists and is what `.gitlab-ci.yml` uses; SCons is the current path.

**Python environment**: `lunus.yaml` is the conda/mamba spec (`conda env create -f lunus.yaml`) — cctbx, gemmi, mdtraj, mpi4py, scons, ambertools. Bare system `python3` will not have cctbx/gemmi/mpi4py; always run Python code with the `lunus` env interpreter.

## Tests

```bash
export LUNUS_HOME=$PWD
export PATH=$PATH:$LUNUS_HOME/c/bin
cd test && bash run_tests.sh          # runs every test/unit_tests/Test*.sh
```

`run_tests.sh` first substitutes `${LUNUS_HOME}` into `lunus_params_unique.sh` to produce `test_inputs.sh`, then runs each `Test*.sh` and reports pass/fail from its exit status.

Run a single test:

```bash
cd test/unit_tests && bash TestModeim.sh    # exit 0 = pass
```

Tests are pure shell: they run a binary on `test/data/`, write to `test/proc/`, and `diff` against a golden file in `test/ref/`. `TestLunus.sh` is the exception — it runs the full `lunus.lunus` pipeline through `TestLunusDir/run_lunus.sh` and passes if `lunus.corrlt` correlation with the reference lattice exceeds 0.99999. Scripts named `noTest*.sh` are deliberately disabled (they need cctbx/DIALS or extra data) and are skipped by the glob.

Set `LUNUS_MPI=YES` to make `TestLunusDir/run_lunus.sh` launch under `mpirun -np 4` (or `srun` under SLURM).

## C layer architecture

The conventions here are strict and pervasive — follow them when adding functionality.

- **One function per file.** `c/lib/lXXX.c` defines `lXXX()`; `c/src/XXX.c` is a thin `main()` that parses argv, reads files, calls `lXXX()`, and writes results. SCons globs both directories, so adding a file is enough — no build-file edit. `c/src/XXX.c` becomes the binary `lunus.XXX`.
- **All prototypes live in `c/include/lunus.h`** (which `mwmask.h` aliases). A new library function must be declared there or nothing will link.
- **Name suffixes encode the data type operated on**, and the two core structs are `DIFFIMAGE` and `LAT3D` (both in `lunus.h`):
  - `*im` — 2D diffraction image (`DIFFIMAGE`), e.g. `lmodeim`, `lpunchim`, `lthrshim`, `lpolarim`
  - `*lt` — 3D reciprocal-space lattice (`LAT3D`, `.lat` format), e.g. `lanisolt`, `lculllt`, `lcorrlt`
  - `*rf` — 1D radial-average file ("rfile"), e.g. `lavgrf`, `lmulrf`
  - `*map` / `*vtk` / `*hkl` — CCP4 map, VTK, and reflection-file conversions (`lat2vtk`, `lat2hkl`, `map2lat`, …)
- **Reentry protocol**: long-running image routines take `imdiff->reentry` (0 = init+free, 1 = init only, 2 = neither, 3 = free only) so they can be called repeatedly in a pipeline without reallocating. See the header comment in `c/lib/lmodeim.c`.
- **Acceleration is compile-time.** `USE_OPENMP`, `USE_CUDA`, `USE_KOKKOS` guard alternate implementations; the Kokkos/CUDA variants live in `lunus/kokkos/` and `lunus/cuda/` with their own SConscripts and are Linux-only. `LUNUS_NUM_IBLOCKS`/`LUNUS_NUM_JBLOCKS` set the tiling used by those paths.

### `lunus.lunus` — the pipeline driver

`c/src/lunus.c` is the main production entry point: `lunus.lunus <input_deck>`. The input deck is a shell-style `key=value` file (see `test/unit_tests/lunus_params_unique.sh`) parsed by string search — `lgettag(deck, "\nkeyword")`. Keys cover the whole pipeline: image masking (`thrshim_*`, `punchim_*`, `windim_*`), mode filtering (`modeim_*`), polarization (`polarim_*`), and integration (`jsonlist_name`, `xvectors_path`, `diffuse_lattice_dir`, `resolution`, `points_per_hkl`, `writevtk`, …). The deck buffer is broadcast over MPI and each rank processes a subset of images, so the driver is MPI-aware end to end; `writevtk`/`spacegroup`/`amatrix_format` are optional keys with defaults.

DIALS/DXTBX experiment JSON files supply per-image geometry; the deck points at a `jsonlist` file listing them. `scripts/get_amatrix_from_json.py` and `scripts/get_xvectors_from_json.py` generate the A-matrix and x-vector inputs from those JSONs.

## Python layer

`lunus/__init__.py` imports the Boost extension (`lunus_ext`) into the package namespace. The extension wraps two classes:

- `Process` — the image→lattice integration pipeline (`set_image`, `set_amatrix`, `set_xvectors`, `LunusProcimlt`, `get_lattice`, `write_as_vtk`/`_hkl`/`_cube`/`_lat`, `divide_by_counts`).
- `LunusDIFFIMAGE` — single-image filters (`LunusPunchim`, `LunusWindim`, `LunusThrshim`, `LunusPolarim`, …).

Python code that imports `lunus` therefore requires the cctbx-module build, not the standalone one.

### Dispatchers

Each `lunus/command_line/*.py` carries a `# LIBTBX_SET_DISPATCHER_NAME lunus.<name>` comment; libtbx generates the corresponding `bin/lunus.<name>` wrapper at configure time. Current set: `lunus.process`, `lunus.integrate`, `lunus.xtraj`, `lunus.md`, `lunus.filter_peaks`, `lunus.stills_process`, `lunus.simulate_diffraction_image`.

Most of these detect MPI by checking `'OMPI_COMM_WORLD_SIZE' in os.environ` and degrade to serial otherwise, so they are launched as `mpirun -np N python <script> ...` (usually with `OMP_NUM_THREADS=1` to avoid oversubscription).

### `xtraj` — MD trajectory → diffuse scattering

`lunus/command_line/xtraj.py` reads an MD trajectory (mdtraj) plus topology and computes ensemble structure-factor statistics: `<F>` (Bragg) and `<|F|²> − |<F>|²` (diffuse). Arguments are `key=value` on the command line, parsed positionally out of `sys.argv`. Key options: `top=`, `traj=`, `d_min=`, `first=`, `last=`, `nproc=`, `chunk=`, `engine=` (`cctbx` default | `sfall` | `gemmi`), `gemmi_cutoff=`, `cctbx_method=` (`fft` | `direct`). Output filenames are set with `fcalc=`, `icalc=`, `diffuse=`. `examples/xtraj/run.sh` is a working smoke test. Note that its reference outputs (`*.ref.mtz`, `*.ref.hkl`) are **not** in git — they are caught by the blanket `examples/xtraj/*.mtz` / `*.hkl` patterns in `.gitignore`, so a fresh clone has only `run.sh`, `top_ref.pdb` and `traj_ref.xtc`.

**Two xtraj variants exist, deliberately.** `lunus/command_line/xtraj.py` is the shipped dispatcher (`gemmi`/`sfall`/`cctbx` engines). `lunus/sf/xtraj.py` is the live development variant and additionally has `engine=torch`, plus `torch_device=`, `torch_compile=`, `torch_max_pairs_per_batch=`, `torch_num_threads=`, `torch_taper_width=`, `torch_blur=` (`auto` by default: the FFT-artifact blur, sized from the grid and the model's minimum B — it matters a lot at low B, see `docs/solvent-design.md`), `save_density=` and `use_top_bfacs=`. Merging the torch engine into the dispatcher is a separate change that has not been made.

### `lunus/sf/` — differentiable structure factors

A PyTorch implementation of a **gemmi-like** engine: differentiable splat → symmetrize → FFT → F(hkl) with respect to atomic coordinates. Gemmi-like rather than a clone — the density cutoff is tapered smoothly rather than truncated hard, so that gradients are well behaved, and that taper is the main source of the (measured, understood) difference from gemmi. See `docs/design.md`.

It installs as an ordinary Python package (`pip install -e ".[sf]"`, or via pixi — see the repo-root `pyproject.toml`, which packages the pure-Python tree only and leaves the SCons/libtbx builds untouched). The public API resolves lazily from the package root, so `from lunus.sf import structure_factors_batch` works and a bare `import lunus.sf` does not import torch. Working from a clone with the repository root on `sys.path` also works. Only `xtraj.py` and the cctbx-based tools in `tools/` require cctbx; the rest needs numpy and torch alone. `lunus/__init__.py` imports the `lunus_ext` Boost extension inside a try/except with a PEP 562 `__getattr__` fallback specifically so that stays true, and so `lunus.md` (pure Python) is importable without a cctbx module build. Code that needs `Process`/`LunusDIFFIMAGE` still fails loudly, at first use.

Layout: library modules at the top level; `tests/` (pytest, `python -m pytest tests/ -q`), `tools/`, `examples/`, `docs/`. See `lunus/sf/README.md` for the module-by-module table, `docs/design.md` for where it matches gemmi and where it deliberately does not, and `docs/performance.md` for benchmarks.

`examples/compare_gemmi/` holds tracked input data (`top_bfac.pdb.gz`, `traj.xtc` — 135,834 atoms) and `run_xtraj.sh`, the end-to-end torch-vs-gemmi comparison. **Every measured number in `docs/` was taken on that system**, at cell 88.451,88.451,39.823 / P4₃ / `d_min` 0.9 / grid 300×300×144 — a cell and space group that were carried over from a different example and are **wrong for this crystal**. It is PDB 7FPV (34.196 × 45.558 × 99.044, P2₁2₁2₁), kept alongside as `7FPV.pdb`, and the simulation box is exactly a 2×2×2 supercell of it. `run_xtraj.sh` now passes the correct cell; the docs numbers have not been re-measured, and `docs/performance.md` says what that does and does not invalidate. The data is kept in the repo so those numbers stay reproducible. Its outputs are gitignored, as is the optional `save_density=True` grid dump (221 MB per run).

`lunus/md/` is a small self-contained MD engine (`energy`, `integrate`, `maxwell`, `parse`, `units`) driven by `lunus.md`.

## Conventions

- **Don't delete commented-out scaffolding** (profilers, alternate algorithm paths, debug timers) during cleanup passes — this repo intentionally keeps them for later reactivation. Genuinely abandoned code is fine to remove.
- `test/ref/` holds golden outputs; regenerate deliberately, never as a convenience when a test fails.
- Contributors must sign the CLA linked in `CONTRIBUTING/CONTRIBUTING.md`.
