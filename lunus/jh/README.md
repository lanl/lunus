# `lunus/jh` — James Holton's GPU FFT structure-factor engine

An archive of PR #23, "Add GPU FFT engine to xtraj (`engine=gpu`)", opened
against `lanl/lunus` on 2026-05-03 by **James Holton (jmholton)** and never
merged. Preserved here so the work is not lost with the pull request, and so
it is discoverable from the tree rather than only from GitHub.

| | |
|---|---|
| source | `jmholton/lunus`, branch `gpu-engine` |
| head | `29335b7bbf6e3ae4e9ef086ceec31674cf9c7f29`, 2026-05-03 |
| commits | 8, from `a9e04d5` (2026-04-22) to `29335b7` |
| pull request | <https://github.com/lanl/lunus/pull/23> |

## What it is

A structure-factor engine that spreads atomic density onto a grid and FFTs it
on the GPU with cuFFT, in place of cctbx's CPU direct summation. `md2mtz/`
holds the kernel and a standalone command-line tool; `xtraj_gpu.py` is the
version of `lunus/command_line/xtraj.py` that calls it.

Reported by the author, on a GV100:

| | |
|---|---|
| system | 473,064 atoms, 1000 frames, `d_min` 0.9 |
| CPU (cctbx) | 45.4 s/frame, 12.6 h total |
| GPU | 8.6 s/frame, 2.4 h total — **5.2x** |
| agreement | Pearson **0.9999** against the CPU path, over 12M+ ASU reflections |

Those numbers are his, on his hardware and his system. They have not been
reproduced here, and nothing in `lunus/sf/docs/performance.md` was measured on
a comparable configuration, so do not compare them directly with the torch
engine's figures.

## Why it is archived rather than active

The repository went a different way, for a reason that is about capability
rather than speed: **`lunus/sf`'s torch engine is differentiable**, and the
guidance work depends on gradients with respect to atomic coordinates. This
engine is forward-only. It is also written against CUDA directly, where the
torch path runs on CUDA, MPS or CPU from one implementation.

`xtraj_gpu.py` is a **snapshot of May 2026** and is not a drop-in replacement
for the current `lunus/command_line/xtraj.py`. It predates the torch engine,
the `gemmi_cutoff` default moving to 1e-4, the auto-tuned pair budget and the
`do_opt` rework. Treat it as a record of how `engine=gpu` was wired in, not as
a maintained variant.

Nothing here is imported by the rest of lunus, and nothing here is on any
build path. The PR's changes to the root `README.md`, root `CLAUDE.md` and
`SConstruct` (which enabled OpenMP globally) were deliberately NOT taken, as
those touch the shared tree.

## One omitted file

**`md2mtz/P1pdb1.pdb` is not included.** It is 43 MB — 473,064 atoms with
explicit hydrogens in a P1 box, `CRYST1 170.266 170.266 152.840 90 90 90` —
and accounted for essentially all of the pull request's 493,671 added lines.
It is an MD supercell from the author's own simulation, so it cannot be
fetched from the PDB.

`run_p1pdb1_10k_test.sh` and `run_p1pdb1_full_test.sh` still refer to it. They
are kept verbatim rather than repointed, so that what is his stays his. To run
them, substitute a large P1 structure the repository already tracks:

```bash
cd lunus/jh/md2mtz
zcat ../../sf/examples/compare_gemmi/top_bfac.pdb.gz > P1pdb1.pdb   # 135,834 atoms
bash run_p1pdb1_full_test.sh
```

That is the 7FPV 2x2x2 MD box every number in `lunus/sf/docs/performance.md`
was measured on, which makes any comparison against the torch engine
meaningful rather than incidental. It will not reproduce the 5.2x above, which
was specific to 473k atoms on a GV100.

`run_p1pdb1_10k_test.sh` is the cheap entry point: it slices the first 10,000
atoms and randomises B over [2, 999], so it works against any large P1
structure and exercises the kernel in seconds.

## Building

There are no compiled artefacts in this archive — no `.so`, no bundled cuFFT.
`md2mtz/sfcalc_gpu.cu` (17 KB) is the kernel and `md2mtz/compile_gpu.csh`
builds it; `md2mtz/setup_cuda.sh` sets the environment. The author's build
targeted NVIDIA Volta through Hopper. Anything newer — Blackwell, including
GB10 / DGX Spark — needs recompiling for its own architecture.

`md2mtz/include/gemmi/` vendors two gemmi headers (`symmetry.hpp`,
`fail.hpp`). They duplicate a dependency lunus already has, and are kept only
so the sources compile as the author left them.

## The author's own documentation

`md2mtz/README.md` and `md2mtz/CLAUDE.md` are his, unedited. They are the
better reference for the kernel's internals, its parameters and its validation
than anything above.
