# Performance

**Every number on this page was measured on one specific configuration.** They
are meaningless against a different system, so it is stated once here and
assumed throughout:

| | |
|---|---|
| topology | `examples/compare_gemmi/top_bfac.pdb.gz`, 135,834 atoms |
| B-factors | per-residue, uniform on [10, 100] |
| cell / space group | 88.451, 88.451, 39.823 / P4₃ |
| resolution / grid | `d_min` 0.9 → 300 × 300 × 144 |
| cutoff | 0.01 |
| machine | Apple M5 Pro (6 performance + 12 efficiency cores), macOS 26.5.1, **except** the CUDA section and anything about a 251-frame run, which are an NVIDIA container, driver CUDA 13.0 |
| torch | 2.13.0 |
| atom-voxel pairs | 324,986,306 |

Reproduce all of it with `tools/bench_splat.py`, which is the thing to trust
rather than the tables below:

```bash
PYTHONPATH=<repo root> python tools/bench_splat.py \
    examples/compare_gemmi/top_bfac.pdb.gz \
    --cell 88.451,88.451,39.823,90,90,90 --grid 300,300,144 [--device mps|cuda]
```

It synchronizes around each timed region, without which GPU timings measure
dispatch rather than execution. Set `OMP_NUM_THREADS` to control CPU threads.

Figures below re-measured 2026-08-11.

## CPU

**gemmi's `DensityCalculatorX` is single-threaded** (measured: 1.00 cores
busy), so the thread count torch is given decides which way the comparison
goes. Both are worth stating:

| | 1 thread | 6 threads |
|---|---|---|
| gemmi `put_model_density_on_grid` | 1.37 s | n/a (single-threaded) |
| torch, eager | 4.23 s | 1.84 s |
| torch, `torch.compile` (default) | 3.34 s | **0.78 s** |

So torch beats gemmi in wall time when it can spread over cores (0.57x on 6
threads), but **gemmi is ~2.4x faster per core**. That distinction decides
multi-rank runs: under `mpirun -n N` on CPU, `xtraj.py` gives each rank one
torch thread to avoid oversubscription, so the per-core figure is the one that
applies.

Rebalancing ranks against threads does not rescue it. Measured over 10 frames
with a warm compile cache:

| | total |
|---|---|
| torch, 10 ranks x 1 thread | 7.44 s |
| torch, 6 ranks x 3 threads | 7.15 s |
| torch, 3 ranks x 6 threads | 8.02 s |
| torch, 2 ranks x 9 threads | 21.37 s (each rank serializes 5 frames) |
| **gemmi, 10 ranks x 1 thread** | **3.12 s** |

`torch_num_threads=N` sets threads per rank if you want to explore this, but the
honest summary is: **expect gemmi to win multi-rank CPU runs.** The torch engine
earns its keep through differentiability, and on GPU or in a single
multi-threaded process — not as a faster CPU replacement for gemmi.

## Apple GPU (MPS)

`torch_device=mps` works end to end — splat, symmetrization, FFT and extraction
— and is the fastest single-process option here:

| device | splat | vs gemmi | pairs/s |
|---|---|---|---|
| gemmi (C++, 1 core) | 1.33 s | 1.00x | — |
| torch CPU, eager, 6 threads | 1.84 s | 1.34x | 177e6 |
| torch CPU, compiled, 6 threads | 0.78 s | 0.57x | 419e6 |
| **torch MPS, eager** | **0.72 s** | **0.54x** | **451e6** |

MPS agrees with CPU to float32 rounding (whole-grid correlation 1.0000000000,
max difference 1.8e-7 relative). End to end through `xtraj.py`, 10 frames took
15.5 s on MPS against 20.9 s for gemmi.

Two things to know:

- **`torch.compile` does not work on MPS in torch 2.13** — inductor's Metal
  backend raises `InductorError: 'c10/metal/reduction_utils.h' file not found`.
  `density_torch` catches this on the first call, warns once, and runs eager for
  the rest of the process, so `torch_device=mps` works with the default
  `torch_compile=True`; it just does not benefit. Pass `torch_compile=False` to
  skip the failed attempt and its one-off cost. Notably MPS does not *need* it:
  eager MPS already edges out compiled CPU.
- **MPS has no float64 at all.** Everything here is float32 by design, and the
  offset-prefix search deliberately stays in the compute dtype rather than
  promoting — the extra precision would buy nothing, since rounding can shift
  the prefix by at most one voxel shell whose density the taper has already
  driven to ~0.

## NVIDIA GPU (CUDA)

Measured on a container with an NVIDIA card (driver CUDA 13.0), torch 2.13.0
`cuda130_mkl` from conda-forge, on the same 135,834-atom system and
300 × 300 × 144 grid. The splat *in isolation*:

| device | splat | vs gemmi | pairs/s |
|---|---|---|---|
| gemmi (C++, 1 core) | 1.37 s | 1.00x | — |
| torch CPU, compiled, 6 threads | 0.78 s | 0.57x | 419e6 |
| torch MPS, eager | 0.72 s | 0.53x | 451e6 |
| **torch CUDA, eager** | **0.06 s** | **0.04x** | **5126e6** |

About 23x faster than gemmi and 11x faster than MPS. **But see the next
section: the splat costs ~255 ms/frame inside `xtraj.py`, not 60 ms, and that
discrepancy is unexplained.**

Environment notes, all of which cost a round to discover:

- conda-forge ships CPU *and* CUDA build variants of the same `pytorch`
  package, and a bare `pytorch = "*"` resolves to the CPU one. Ask for
  `pytorch-gpu`, which forces a `cuda*` build.
- Those builds need the `__cuda` virtual package, which pixi derives by probing
  the host. **That probe fails in many containers even when the GPU works**,
  giving `__cuda *, for which no candidates were found`. Declare it instead:
  `platforms = [{ platform = "linux-64", cuda = "13" }]`, with the value being
  the maximum CUDA version the *driver* supports (`nvidia-smi`, top right).
  Under-declaring is what causes spurious failures.
- `torch.compile` fails here: triton compiles a small `cuda_utils.c` needing
  the driver API header, and a container that mounts the driver still has no
  headers. `cuda-driver-dev` provides `cuda.h` but the compile still failed,
  and **it is worth nothing on a GPU regardless** — 5140e6 pairs/s compiled
  against 5126e6 eager. Fusing memory-bound passes mattered on CPU (~2.9x); a
  GPU does not need it. The eager fallback is automatic and numerically
  identical, but the attempt costs ~10 s per run, so it is off by default in
  `examples/compare_gemmi/run_xtraj.sh` and opt-in (`--compile`) in
  `bench_splat.py`.

### GPU runs are not bit-reproducible

`index_add_` uses atomics, so summation order varies between runs. Two
identical CUDA runs differ by up to **4e-5 relative on Icalc** and **~2e-4 on
diffuse**. CPU is deterministic; MPS is not.

Relative noise grows as reflections weaken — absolute error is roughly
constant, so a strong reflection sees ~1e-7 while one at Icalc 904 sees 4.4e-5.
**Diffuse is hit hardest, structurally**: `<|F|²> − |<F>|²` is a difference of
two large, nearly equal numbers, so the cancellation amplifies it. A diffuse
value of 1520 moved by 1.7e-4 between runs while its Icalc moved by 9e-6.

That is the reproducibility floor for anything downstream. Averaging more
frames helps; this noise does not average away *within* a run.
`torch.use_deterministic_algorithms(True)` would pin it at some cost.

## Where the time actually goes

`torch_timing=True` reports per-phase timings with device synchronization on
both sides of each phase. On CUDA, 251 frames, 140 s total (≈553 ms/frame):

| phase | ms/frame | share of timed |
|---|---|---|
| splat | 255.4 | 97.0% |
| into cctbx | 4.4 | 1.7% |
| host→device | 1.0 | 0.4% |
| symmetrize | 1.1 | 0.4% |
| device→host | 0.8 | 0.3% |
| fft+extract | 0.6 | 0.2% |
| **total timed** | **263** | |

Two things worth internalising:

**The FFT is 0.6 ms.** So are symmetrization and both transfers. Everything in
the GPU pipeline except the splat is free; there is nothing to win there.

**About half of each frame is host-side**, outside these timers — 290 ms of
553 ms. Of that, cProfile attributes only ~39 ms/frame to identifiable calls
(`add_scatterers` 14 ms, `structure.select` 14 ms, `resolution_filter` 4 ms,
`set_sites_frac` 3 ms, misc 4 ms), leaving ~250 ms/frame **unattributed and
uninstrumented**. That is the largest single unknown in the pipeline and the
obvious place for the next measurement — the trajectory read inside mdtraj's
`iterload` generator is the leading suspect, since cProfile charges it to one
call rather than per frame.

## An unresolved discrepancy: the splat in situ vs in isolation

**The observation.** On CUDA the splat costs ~255 ms/frame inside the frame
loop, but repeating the identical call on the identical tensors in the same
process gives 62.8 ms — matching `bench_splat` exactly. Same data, same
process, seconds apart, fourfold different.

**Ruled out, each by measurement rather than argument:**

| hypothesis | test | result |
|---|---|---|
| cProfile overhead | run with `profile=False` | 255.4 vs 264.8 ms — 4% |
| real vs random coordinates | `bench_splat --random-coords` | no difference |
| GPU idle clocks between frames | `bench_splat --idle-gap 0.3` | no difference |
| allocator churn from the per-frame FFT buffer | CUDA allocator stats | retries 0, segment frees 0 |
| host-side contention | repeat with the host kept busy | 63.2 vs 62.8 ms |
| kernel-launch overhead | `torch_max_pairs_per_batch=64000000` | 259 → 247 ms, 5% |
| thermal/power throttling | first-5 vs last-5 frame trend | *falls*, 319 → 254 ms |

**A methodological error worth recording.** The repeat test originally ran only
on frame 0 — whose in-loop time is the slowest of the run because it carries
one-time warmup. Comparing a warm repeat against a warmup-polluted baseline
proves nothing, and may have manufactured part of the puzzle. On MPS that
confound inflates the ratio from 1.18x (frame 3, clean) to 1.76x (frame 0).

**Still open.** Whether later frames' coordinates genuinely cost more than
frame 0's — which would make this a data effect and would also explain why
`bench_splat`, which uses the *topology's* coordinates rather than the
trajectory's, has always been fast. The instrument for this is a like-for-like
comparison on a later frame; if that shows a large ratio, the next step is
`torch.profiler` with CUDA activity tracing rather than more black-box tests.

**Perspective before anyone spends more on it:** the splat is ~255 ms of a
~553 ms frame, so eliminating it entirely caps the win near 45%. The host-side
half is comparable in size, better understood, and completely uninstrumented.

## How to measure this pipeline

Hard-won, in the order the mistakes were made.

**cProfile misleads badly for torch.** It charges per *Python* call, and the
splat makes ~90 batched calls per frame each issuing many ops, so it inflates
the splat several-fold while the FFT — one call doing heavy GPU work — looks
free. In a 251-frame run it showed `splat_density` at 65 s cumulative against a
real ~15 s. Its `tottime` column *is* trustworthy for low-call-count functions
doing real work, which is how `miller.__add__` was correctly identified at 53 ms
per call.

**Never run `profile=True` and `torch_timing=True` together.** cProfile's
overhead lands inside the timed regions. `xtraj.py` warns if you do.

**GPU kernels are asynchronous.** An unsynchronized timer measures how long it
took to *enqueue* work, and the cost surfaces on whichever later call happens
to block. Every phase timer synchronizes on both sides.

**Back-to-back benchmarks measure a different problem than a real loop.** They
keep caches warm, the device busy, and the allocator stable. `bench_splat`
reports 60 ms for an operation that costs 255 ms in situ, and that gap is still
unexplained — treat isolated kernel benchmarks as upper bounds on achievable
speed, not as predictions.

**Benchmark inputs must be the real thing.** `bench_splat` originally splatted
uniformly *random* coordinates, having discarded the structure's own — a
materially easier problem for an atomic scatter than a clustered structure. Now
fixed; `--random-coords` retains the old behaviour for comparison.

**Measure repeats before believing a difference.** Run-to-run variation on the
same command reached 25% on the 10-frame case. A single pair of runs once
showed the numpy accumulation making things *slower*; three runs each showed
the opposite, matching the microbenchmark's prediction.

## How the splat got fast

Starting point was 12.2 s (6 threads). Every step below is numerically neutral:
density correlation against gemmi is unchanged at 0.99977378 and the sum ratio
at 1.016887, both to eight figures, and the autograd-vs-analytic gradient check
in `tests/test_torch_pipeline.py` is unchanged at 2.00e-04.

**The governing fact is that this work is memory-bound.** Each elementary torch
operation is its own kernel pass over an (atoms, voxels) array, so the cost is
the number of passes, not the number of flops. That explains both what worked
and what did not:

- **Separable distances** (12.2 s → 3.5 s). With `base = round(pos*N)`, the
  displacement is `c_off[k] - c_rem[i]`, so `r2` comes from one matrix multiply
  plus a precomputed `|c_off|^2` instead of four `(m,K,3)` intermediates. Also
  removes the minimum-image wrap, which provably never fires (now checked at
  setup).
- **Fewer evaluated points** (−24%). The candidate list is sorted by distance
  from the box centre, so each atom uses only the prefix its own radius reaches;
  and the safety margin is half a voxel diagonal, not a full one, which is the
  tight bound since `round()` puts the atom within half a voxel of the centre.
  Dropped points have true distance beyond the taper, so both changes are
  exactly lossless.
- **Kernel fusion** (2.7 s → 0.97 s, the single biggest win). The two hot blocks
  are factored into `_density_core` and `_scatter_core` purely so
  `torch.compile` can fuse each into a couple of passes instead of ~20. Fusion
  imitates what gemmi gets for free by keeping each voxel in registers in one
  pass, which is why it closes the gap. Enabled by default; `compile_core=False`
  (or `torch_compile=False` on xtraj.py's command line) falls back to eager, as
  does any compile failure, automatically.
- **What did NOT work:** replacing the wrap-around modulo with an arithmetically
  equivalent compare-and-shift made it *slower*, because two `torch.where`s cost
  about six passes against one for a `remainder`, despite integer division being
  the more expensive instruction.

One wrinkle worth knowing: compiled autograd's donated-buffer optimization
requires every `backward()` to be its graph's last, so it raises under
`retain_graph=True` or `create_graph=True` — which any gradient check does.
`density_torch` therefore sets `torch._functorch.config.donated_buffer = False`
when it compiles, trading a little backward-pass memory for the freedom to call
`backward()` however you like.

Compiled and eager differ by ~1e-7 relative, which is float32 `index_add_`
accumulation ordering, not a fusion error.

## The running sums: numpy, not miller arrays

`xtraj.py` used to accumulate `sig_fcalc = sig_fcalc + fcalc` once per frame.
Every frame shares one Miller set — same cell, same `d_min` — so
`miller.__add__` repeated identical index matching every time, at **53 ms per
call, twice per frame**: 26 s of a 251-frame run, about 15% of it, against
roughly 8% for the actual GPU calculation.

The sums are now numpy arrays, and `|F|²` is formed directly rather than via
`abs() → set_observation_type → f_as_f_sq()` (agreeing to 2e-18 relative).
`sig_fcalc`/`sig_icalc` are kept as the first frame's miller arrays because
downstream code uses them as containers, overwriting `.data()` with the reduced
totals. Adding data arrays elementwise is only valid if the Miller sets agree,
so that is asserted per frame; the check costs 0.01 ms.

Measured on the accumulation alone: 0.88 → 0.01 ms/frame at 6,388 reflections,
3.86 → 0.04 ms/frame at 21,164. End to end on the 251-frame CUDA run,
**176 s → 140 s**. Output is bit-identical.

## Ensembles: memory is the binding constraint, not speed

Backpropagating through N configurations at once — a guided step over an
ensemble, say — is limited by memory, and the limit is not what it looks like.

The density grid is the obvious cost but the small one. The splat's working
buffers are `(atoms × voxels)` arrays, far larger than the grid, and autograd
retains them for the backward pass. Do that for N members simultaneously and
the retained intermediates, not the grids, are what fills memory. `xtraj.py`
never meets this because it runs forward-only per frame and reduces
immediately; guidance cannot, because every member's graph has to stay live
until the loss is backward-ed.

`structure_factors_batch(..., use_checkpoint=True)` recomputes each member's
splat during backward instead of retaining it. Measured on a 3000-atom system
at a 90³ grid, peak RSS above baseline:

| N | retained | checkpointed |
|---|---|---|
| 8 | 839 MB | **361 MB** |
| 16 | 1604 MB | **371 MB** |

Flat in N rather than linear, for about 2.4x the wall time. Gradients are
**bit-identical** either way — `tests/test_batch.py` asserts exact equality,
not approximate — so this is a pure time-for-memory trade with no numerical
consequence.

Rules of thumb: leave it off for small N or small grids where the retained
path fits comfortably; turn it on when N or the grid grows, which at
production grid sizes is essentially always. Since it changes no numbers, it
is safe to flip based purely on whether the run fits.

### This is tuned for memory, not for speed

The flag is deliberately all-or-nothing, and that is a limitation rather than a
design goal. It exists to make large N *possible*; when N does not fit it takes
the memory-minimising extreme and pays the full recompute penalty on every
member, including the ones that would have fitted.

The optimum is in between: retain as many members as the device can hold and
checkpoint only the remainder. Peak memory then tracks the retained count
instead of N, and the time penalty falls to roughly the fraction checkpointed —
at N=16 on a device with room for 8, half the penalty at the same peak. A
`max_retained` parameter, or a byte budget read from
`torch.cuda.mem_get_info()`, would let it choose rather than forcing the caller
to pick an extreme.

Calibrate against measurement rather than arithmetic when doing this. The
per-member cost is dominated by the splat, and the splat's in-loop cost on CUDA
is not yet understood (see the unresolved discrepancy above), so the 2.4x
recompute penalty measured at one N on one device should not be assumed to hold
generally.

## Compiling under MPI

`torch.compile`'s cost is per-process, so under `mpirun -n N` every rank
compiles the same kernels and N concurrent compilations saturate the machine. On
10 ranks with a cold cache that showed up as frame times staggered in ~1 s steps
(10.0, 10.1, 10.3, 10.8, 11.8, 12.8, 13.9, 15.0, 16.0, 17.0 s; total 17.5 s). It
is *not* contention on inductor's cache-directory file lock — giving each rank
its own `TORCHINDUCTOR_CACHE_DIR` reproduces the stagger exactly.

`xtraj.py` therefore calls `density_torch.warmup_compile()` on rank 0 and
barriers before the others start, which brings the same cold-cache run to 7.4 s
with no stagger. The penalty only appears on a cold cache, so it looks
intermittent — re-running the same job is fast because the on-disk cache
survives.

Writing that warmup correctly is fiddly, and both mistakes are silent (the
warmup succeeds, the cache looks populated, every rank recompiles anyway). It
has to vary the per-element atom count above 1, since size-1 dimensions are
specialized; and it has to pass both partial and full-length offset prefixes,
because the guard is literally
`c_off_sq._base.size()[0] == c_off_sq.size()[0]`. Both were found by reading
`TORCH_LOGS=recompiles`, which is the tool to reach for if this ever regresses —
the check is that a real splat immediately after the warmup adds no new graphs
to `torch._dynamo.utils.counters["stats"]["unique_graphs"]`.

## Tuning knobs

| option | effect |
|---|---|
| `torch_device=cpu\|mps\|cuda` | where the splat, symmetrization and FFT run |
| `torch_compile=False` | skip the one-off compile; worth it for single-frame runs, or on MPS where it cannot succeed |
| `torch_max_pairs_per_batch=N` | atom-voxel pair budget, the knob that actually sets intermediate tensor size. The default keeps each working buffer cache-resident, which measured faster than both larger and smaller values. Re-tune on a new machine with `bench_splat.py`. |
| `torch_num_threads=N` | threads per rank |
| `torch_taper_width=W` | taper width in Å; narrower is closer to gemmi but harsher on gradients |

The old `torch_max_atoms_per_batch` is deliberately gone: batching is budgeted
by pairs now, and the atom cap almost never bound.
