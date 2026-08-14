# Bugs found and fixed

Kept because several of these failed *silently* — wrong numbers rather than a
traceback — and the reasoning that exposed them is worth more than the diffs.
Each entry records how it manifested, why it was easy to miss, and how the fix
was verified.

## do_opt summed uninitialised memory

**Symptom:** none. Wrong optimization results, non-deterministically.

`fcalc_list` / `icalc_list` are allocated with `np.empty()`, sized from the
*planned* frame count `chunklist * nchunklist`, then filled once per frame
*actually processed*. A final short chunk — or a trajectory ending earlier than
the plan expects — left rows unwritten. Those rows are then consumed by
`np.sum(fcalc_list, axis=0)` and by iterating over the whole array, folding
uninitialised memory straight into the result.

**Why it hid:** `np.empty` returns *zeroed* pages when the allocator hands back
fresh memory from the OS, so the bug is invisible until the heap is dirty
enough to recycle used pages. A first attempt to demonstrate it showed no
contamination at all. Forcing reuse makes it stark:

```
unwritten rows contain: [1e+30+1e+30j ...]
sum over ALL rows      : [2e+30+2e+30j ...]
sum over filled rows   : [6.247+5.612j ...]
```

**Fix:** both arrays are trimmed to the number of frames actually processed,
and `last_this` is derived from that count rather than the planned one, so the
HDF5 slices and per-frame indexing stay aligned.

**Verification:** build simulated data from frames 0–4, then optimise over all
10. The optimiser removes exactly frames 9, 8, 7, 6, 5 — precisely those absent
from the reference — and reaches correlation 1.0. Contaminated rows could not
produce that. Byte-identical with no chunking, `chunk=5` (even), `chunk=3`
(uneven, 3+3+3+1) and under `mpirun -np 2`.

## chunk= divided by zero on a single rank

**Symptom:** `ZeroDivisionError` whenever `chunk=` was given and the frame count
was not a multiple of it, on one rank.

The frame-assignment scheme reserves the *last* rank to mop up the remainder
and spreads whole chunks over the other `mpi_size - 1` ranks. With one rank
that divisor is zero.

**Why it matters beyond the crash:** the fallback would have handed the single
rank *one chunk containing every frame*. Chunking is what bounds peak memory —
the entire reason to pass `chunk=` — so the "working" path would have defeated
its own purpose.

**Fix:** single-rank runs get their own assignment; rank 0 runs every chunk
itself, including a final short one.

**Verification:** `chunk=3` over 10 frames runs 3+3+3+1, the fourth chunk
measurably shorter (0.22 s against 0.67 s), and results are identical to an
unchunked run, as with `chunk=5`. The multi-rank path is unchanged.

## do_opt could not run serially at all

`mpi_comm` is `None` without MPI, but six `bcast` / `Allreduce` / `Barrier`
calls were unguarded — in the `ID_file` read and throughout the optimization
loop — so any serial `do_opt` run died on `AttributeError: 'NoneType' object
has no attribute 'bcast'`. They now follow the `if mpi_enabled():` convention
used elsewhere in the file.

Same species as the `chunk=` bug: MPI-only code on a path a serial run reaches.

## The torch device was inferred from MPI rank

`run_xtraj.sh` chose `cuda` if an MPI local rank was set and `mps` otherwise.
Device availability and rank count are unrelated, so this failed on Linux (no
MPS), on CPU-only builds, and on non-MPI CUDA machines.

It now asks torch which device exists — the only authority that accounts for
how torch was *built* — overridable with `LUNUS_TORCH_DEVICE`. The MPI local
rank is used only to pin each rank to its own GPU, and only when the device is
actually CUDA.

## lunus/__init__.py's __getattr__ was too aggressive

Making `lunus_ext` optional introduced a module `__getattr__` that raised an
explanatory `ImportError` for *any* missing attribute. But `__getattr__` fires
for every optional attribute that tooling probes for: adding `pyproject.toml`
moved pytest's rootdir to the repository root, pytest probed
`lunus.pytest_plugins`, and a routine lookup became a collection error across
the whole suite.

It now raises `ImportError` only for names the extension actually provides
(`Process`, `LunusDIFFIMAGE`, `LunusLAT3D`, and the `Lunus*` prefix) and plain
`AttributeError` otherwise, which is what the attribute protocol requires.
`tests/test_package_import.py` pins this, including the probe case.

## Notes

Two of these — the uninitialised rows and the `chunk=` divisor — were reachable
only together: the crash masked the silent corruption, so fixing the crash
first exposed the worse bug. Worth remembering when a fix opens a path that was
previously unreachable: the newly reachable code has never run.
