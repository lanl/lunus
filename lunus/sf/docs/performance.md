# Performance

**Every number on this page was measured on one specific configuration.** They
are meaningless against a different system, so it is stated once here and
assumed throughout:

| | |
|---|---|
| topology | `examples/compare_gemmi/top_bfac.pdb.gz`, 135,834 atoms |
| B-factors | per-residue, uniform on [10, 100] |
| crystal | PDB **7FPV**, kept as `examples/compare_gemmi/7FPV.pdb` |
| cell / space group | 34.196, 45.558, 99.044 / **P2₁2₁2₁**, Z = 4 |
| simulation box | 68.392 × 91.116 × 198.088, P1 — exactly a **2×2×2 supercell** |
| resolution / grid | `d_min` 0.9 → 120 × 160 × 360 |
| cutoff | 0.01 |
| machine | Apple M5 Pro (6 performance + 12 efficiency cores), macOS 26.5.1, **except** the CUDA section, which is an NVIDIA container, driver CUDA 13.0, 3 CPUs |
| torch | 2.13.0 |
| atom-voxel pairs | 348,765,878 ideal, 360,747,104 as batched in 95 chunks |

Reproduce all of it with `tools/bench_splat.py`, which is the thing to trust
rather than the tables below:

```bash
PYTHONPATH=<repo root> python tools/bench_splat.py \
    examples/compare_gemmi/top_bfac.pdb.gz \
    --cell 34.196,45.558,99.044,90,90,90 --grid 120,160,360 [--device mps|cuda]
```

It synchronizes around each timed region, without which GPU timings measure
dispatch rather than execution. Set `OMP_NUM_THREADS` to control CPU threads —
**on a CPU-limited container this is not optional**, see below.

Figures re-measured 2026-08-16 on the configuration above. Numbers published
before 2026-08-15 used a cell carried over from another example (88.451,
88.451, 39.823 / P4₃) and do not compare; where one is quoted below for
context it says so.

## Where this stands (2026-08-16)

On CUDA, 135,834 atoms, `d_min` 0.9, steady state per frame. **Measured
eager**; with `torch.compile` the splat drops to 25.5 ms and the frame to
~36 ms, which re-ranks everything below it — see the CUDA section:

| phase | median ms/frame |
|---|---|
| **splat** | **65.9** |
| into cctbx | 1.9 |
| accumulate | 0.8 |
| host→device | 0.6 |
| frac coords | 0.3 |
| fft+extract | 0.3 |
| device→host | 0.3 |
| symmetrize | 0.0 |
| **per-frame total** | **≈ 70** |

plus the trajectory read, which is ~12.8 ms/frame and **does not depend on
`chunk=`**: 251 frames cost ~3.2 s in total whether that is 26 calls of 10
frames or 6 of 50 (12.6 against 12.9 ms/frame measured). It is batched into
per-chunk calls, so its `calls` column and median are per chunk, but the WORK
scales with frames. An earlier version of this note read the per-call figure as
a fixed per-chunk cost and concluded that larger chunks would amortize it; they
do not.

The splat is 94% of the per-frame work and nothing else is above 2 ms. It is
also remarkably steady: over ten frames, median 65.9, min 65.8, max 65.9, with
frame 0 at 195.0 — that first frame carries ~130 ms of one-time warmup, which
is why the table reports a median as well as a mean.

### What the optimizations are worth

Measured directly — `tools/compare_baseline.sh` runs `824b800`, the last commit
before any of them, and the current tip under the same cell, resolution,
device, trajectory and thread count, in one session:

| phase | before | after | saved |
|---|---|---|---|
| `set_sites_cart` | 187.6 | 0.0 | **187.6** |
| `element idx` | 45.8 | 0.0 | 45.8 |
| `xrs.select` | 26.0 | 0.0 | 26.0 |
| `sites_frac+occ` | 22.7 | 0.0 | 22.7 |
| splat | 78.8 | 79.7 | −0.9 |
| everything else | ~7 | ~7 | ~0 |
| **per-frame work** | **368.3** | **87.2** | **281.1 → 4.2×** |
| whole frame loop, 10 frames/chunk | 555.4 | 280.3 | 1.98× |

**The transferable number is 281 ms/frame of host work removed**, all of it in
the four rows that disappear. The splat does not move, which is the point: none
of this touched it.

Two ratios, because one of them is not a property of the code. The trajectory
read fires once per *chunk*, so it is 185 ms/frame over 10 frames and ~7 over
251; include it and the same code reads 1.98× here and would read ~3.9× at 251
frames per chunk. The per-frame ratio, 4.2×, is the one that means something.

**A speedup figure is only as good as its baseline.** An earlier version of
this page led with 6.5×, from a baseline that ran throttled against a final run
that did not — so most of it was a container setting, not code. Measure both
sides in one session, under the same conditions, which is what
`compare_baseline.sh` does.

## CPU

**gemmi's `DensityCalculatorX` is single-threaded** (measured: 1.00 cores
busy), so the thread count torch is given decides which way the comparison
goes. Both are worth stating:

| | 1 thread | 6 threads |
|---|---|---|
| gemmi `put_model_density_on_grid` | 1.42 s | n/a (single-threaded) |
| torch, eager | 4.65 s | 2.02 s |

So gemmi wins outright on one thread (3.27×) and stays ahead on six (1.40×) —
**gemmi is ~3× faster per core**. That distinction decides
multi-rank runs: under `mpirun -n N` on CPU, `xtraj.py` gives each rank one
torch thread to avoid oversubscription, so the per-core figure is the one that
applies.

The `torch.compile` row is gone from this table: it fails on the CPU/MPS
platforms this has been run on (see below), and the numbers it used to carry
were from the superseded configuration.

**On CUDA it does work** — see the CUDA section below, including why the
entry that said otherwise was measuring the eager fallback against itself.

**On CPU, compiling is worth 2.35x on the real splat**: 135,834 atoms on the
120 x 160 x 360 grid, 6 threads, **2.00 s eager against 0.85 s compiled**
(174e6 against 410e6 atom-voxel pairs/s). The first call costs ~1.4 s extra
for the compilation itself, which matters for short runs and not for a
trajectory.

Read that against **3.5x on the isolated elementwise core** (15.1 → 4.3 ms,
`tools/bench_aniso_core.py`). The microbenchmark overstates by ~1.5x, which is
the usual direction and the reason this page keeps saying so: the real splat
also pays the scatter, the offset prefixes and the allocator, none of which
fusion touches.

One trap that comes with it. Compiling on CUDA emits a suggestion to enable
TF32 matmuls. **Do not take it without measuring.** The splat computes `r2`
as `c_off_sq - 2*(c_rem @ c_off.T) + |c_rem|^2` — a difference of terms much
larger than the result, which is why it carries a `clamp_min(0.0)` — and the
anisotropic quadratic form cancels the same way. TF32 keeps 10 mantissa bits
against float32's 23, so it would attack precisely the operation this pipeline
is least able to afford it on.

Rebalancing ranks against threads does not rescue it. Measured over 10 frames
on the SUPERSEDED cell, and not re-measured — the ratios are the point and they
do not depend on the grid:

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
| gemmi (C++, 1 core) | 1.42 s | 1.00x | — |
| torch CPU, eager, 1 thread | 4.65 s | 3.27x | 75e6 |
| torch CPU, eager, 6 threads | 2.02 s | 1.40x | 173e6 |
| **torch MPS, eager** | **0.75 s** | **0.53x** | **463e6** |

MPS agrees with CPU to float32 rounding (whole-grid correlation 1.0000000000,
max difference 1.8e-7 relative, measured on the superseded configuration and
not a property of the cell).

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
120 × 160 × 360 grid. The splat *in isolation*:

| device | splat | vs gemmi | pairs/s |
|---|---|---|---|
| gemmi (C++, 1 core) | 1.42 s | 1.00x | — |
| torch CPU, eager, 6 threads | 2.02 s | 1.42x | 173e6 |
| torch MPS, eager | 0.75 s | 0.53x | 463e6 |
| **torch CUDA, eager** | **0.07 s** | **0.05x** | **5330e6** |

About 21x faster than gemmi and 11x faster than MPS.

In the frame loop the same splat takes **65.9 ms** — the isolated benchmark is
0.07 s for one call on warm buffers, so the two now agree to within the
difference between a benchmark and a loop. That gap used to be 4.6x and is the
subject of the section after next; it closed when the container's thread count
was fixed, not when the code changed.

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
- `torch.compile` **does** work here now, and the entry that used to say
  otherwise was wrong in a way worth keeping. It read: "`cuda-driver-dev`
  provides `cuda.h` but the compile still failed, and it is worth nothing on a
  GPU regardless — 5140e6 pairs/s compiled against 5126e6 eager."

  **Those two numbers are 0.3% apart, which is run-to-run noise, and the same
  sentence says the compile failed.** So the comparison was eager against the
  eager fallback, and the conclusion drawn from it — that fusion does not help
  a GPU — does not follow from it. The fallback is automatic and silent enough
  to be measured by mistake.

  What was actually missing is the include path, not the package: inductor
  shells out to gcc for its `cuda_utils.c` shim, and `CPATH` is what gcc
  honours whatever flags triton constructs. `density_torch` now derives that
  from `CUDA_HOME` and sets it itself, so this needs no environment setup, and
  a compile failure now warns with its cost and likely cause instead of
  quietly halving throughput.

  **Measured properly, compiling is worth 2.69x on the real CUDA splat**,
  135,834 atoms on the 120 x 160 x 360 grid:

  | | s | atom-voxel pairs/s |
  |---|---|---|
  | eager | 0.07 | 5,258e6 |
  | **compiled** | **0.02** | **14,120e6** |

  Note that today's *eager* figure, 5,258e6, is within noise of the number the
  retracted entry reported as *compiled* (5,140e6) — which is the confirmation
  that it had been measuring the fallback.

  The scatter-bound hypothesis was wrong too: fusion helps the GPU roughly as
  much as the CPU (2.69x against 2.35x in situ). The one-off compile costs
  ~12 s on CUDA, which a trajectory amortizes and a two-frame smoke test does
  not.

  **Confirmed in situ**, 251 frames, `torch_timing=True`: the splat's median
  is **25.5 ms/frame against 65.9 eager, 2.58x**. `bench_splat`'s compiled
  0.02 s understated the real loop by 27%, which is the usual direction.

  **Read the MEDIAN, not the mean.** The same run reports a splat mean of
  61.6 ms, nearly the eager figure, because `torch.compile` spends **9.07 s on
  frame 0** and the mean spreads it over 251 frames. That very nearly produced
  the conclusion that compiling had done nothing.

  **The first frame costs 8.77 s, and that is compilation** — separated by
  running the identical trajectory with `torch_compile=False`:

  | | frame 0 | median |
  |---|---|---|
  | eager | 195.9 ms | 66.7 ms |
  | compiled | 8961.5 ms | 25.0 ms |

  So ~196 ms is CUDA and library initialization, which both pay, and the
  remaining 8.77 s is inductor. Worth measuring rather than assuming: this
  page already records an "85.9% of the frame" figure that turned out to be
  `torch.linalg.inv` loading cuSOLVER, so a large first frame is not
  self-evidently the compiler. The eager median of 66.7 ms also confirms the
  65.9 ms baseline above has not drifted.

  **Break-even is therefore ~210 frames**: 8.77 s against 41.7 ms saved per
  frame. A 251-frame run is only ~8% faster overall, and anything shorter is a
  net LOSS.

  **A persistent cache helps by a third, and does not remove it.** With
  `TORCHINDUCTOR_CACHE_DIR` on real storage, frame 0 goes from ~12.5 s cold to
  **8.35 s warm** — so ~4 s was codegen, and ~8.15 s is not. Break-even barely
  moves, ~197 frames warm against ~210 cold, and compiling stays a long-run
  optimization rather than becoming unconditional.

  What survives caching is per-PROCESS rather than per-machine: Dynamo re-traces
  the graph and rebuilds its guards in every new interpreter, whatever is on
  disk.

  **It is ONE compilation, not many.** `TORCH_LOGS=recompiles` counts zero over
  a full run, so `dynamic=True` is covering all ~95 per-frame chunk shapes with
  a single specialization rather than re-specializing as `K` varies. That was
  worth checking because it was the one cheap lever available — bucketing `K`
  to a handful of sizes would have removed most of the residue if it had been
  recompilation. It is not, so there is nothing there to remove, and ~8.1 s of
  single-trace cost is simply what compiling this graph costs per process.

  **Conclusion, so this does not get re-litigated.** Compiling is worth 2.67x
  on the splat and costs ~8.1 s per process that no cache or configuration
  recovers. Break-even is ~196 frames. Turn it on for trajectories, leave it
  off for smoke tests, and do not expect a persistent cache to change that.

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

## How it got here

The frame loop was four times slower before 2026-08-14. Three things fixed it,
two of them in the code and one in the container, and all three generalise
beyond this program.

### `flex.vec3_double(ndarray)` converts row by row

**85.5 ms for 135,834 atoms, against 0.5 ms for
`flex.vec3_double(flex.double(a.ravel()))`, with `max|diff| = 0`.** The ndarray
overload goes row by row; handing it a `flex.double` over the flattened array
takes the bulk path. It was 187.6 ms/frame here — the largest single line-level
win in this codebase — and it applies to **every** engine, since gemmi and cctbx
cannot skip the `xray.structure` the way the torch path now does. Fixed at both
conversion sites in `lunus/command_line/xtraj.py`.

### The torch path does not need the `xray.structure` at all

Fractional coordinates are one 3×3 matmul from what mdtraj already provides,
and the selection, occupancies and element indices do not change between
frames. Skipping the round trip removed `xrs.select` (26.0), `sites_frac+occ`
(22.7) and `element idx` (45.8) — 94 ms/frame — and setup asserts the
fractionalization matrix reproduces cctbx's own `sites_frac()` to 2e-16, since
a silent disagreement there would move every atom.

**cProfile found neither.** It attributed 14 ms/frame to the first and 14 to
the second, an order of magnitude low, because it charges per Python call and
these are few calls doing bulk work. The phase timers in `torch_timing` exist
because of that.

### The container was CPU-throttled, which was worth more than the code

The pod had **3 CPUs** and torch, sizing its pool from the host, started **104
threads**. They spin-wait, so the cgroup quota was spent in ~12 ms of every
100 ms period and the process sat frozen for the rest. Setting
`OMP_NUM_THREADS` was worth **3.6×** — more than the two code fixes together.

It was found in a `torch.profiler` trace, and the signature is worth
recognising: the GPU was idle 82% of the frame, and nine gaps of 24–90 ms
carried 82% of that idle, **all ending at the same phase of a 100 ms period**:

```
45.86  45.67  45.86  45.83  45.82  45.66  45.86  45.90  45.76   ms mod 100 ms
```

Nothing in a numerical kernel ends on a wall-clock boundary. That is the CFS
bandwidth controller refilling the quota. `tools/analyze_trace.py` now tests
for it directly, and `xtraj.py` sizes torch's pool from `cpu.max` rather than
from the host's core count — but that cannot cover numpy/BLAS, which reads the
environment before Python starts, so **export `OMP_NUM_THREADS` on every run**.

Do not trust `nproc`, `os.cpu_count()` or `sched_getaffinity` here: all three
report the host's cores, because a CPU limit is enforced by accounting rather
than by restricting which cores a process may use.

### The "splat is slow in situ" mystery was the same throttling

For weeks the splat cost ~255 ms/frame in the loop against 62.8 ms repeated
back-to-back on the same tensors. The gap was the throttle: `bench_splat`
reports `min()` over its repetitions, so it reported the repetition that was
not frozen, while the loop paid on every frame. With the thread count fixed the
two agree — 65.9 ms in the loop against 0.07 s for one benchmark call.

The eliminated hypotheses are kept because each cost a round to test, and
because the list is a decent template for the next time something is slow for
no visible reason:

| hypothesis | test | result |
|---|---|---|
| cProfile overhead | run with `profile=False` | 255.4 vs 264.8 ms — 4% |
| real vs random coordinates | `bench_splat --random-coords` | no difference |
| GPU idle clocks between frames | `bench_splat --idle-gap 0.3` | no difference |
| allocator churn from the per-frame FFT buffer | CUDA allocator stats | retries 0, segment frees 0 |
| host-side contention | repeat with the host kept busy | 63.2 vs 62.8 ms |
| kernel-launch overhead | `torch_max_pairs_per_batch=64000000` | 259 → 247 ms, 5% |
| thermal/power throttling | first-5 vs last-5 frame trend | *falls*, 319 → 254 ms |
| a fresh grid allocation per frame | `torch_reuse_grid=True`, 251 frames | 290.0 vs 290.6 ms — 0.2% |
| previous frame's grid and F still alive | `torch_free_early=True`, 251 frames | same, 0.2% |

The last two are still available as flags, defaulting to off. The figures in
this section are from the superseded configuration; the effects they describe
are not.

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
overhead lands inside the timed regions. Nothing stops you -- the only
combination `xtraj.py` rejects is `torch_profile_frames` with `profile`.

**GPU kernels are asynchronous.** An unsynchronized timer measures how long it
took to *enqueue* work, and the cost surfaces on whichever later call happens
to block. Every phase timer synchronizes on both sides.

**Back-to-back benchmarks measure a different problem than a real loop.** They
keep caches warm, the device busy, and the allocator stable. `bench_splat`
reported 63 ms for an operation that cost 255 ms in situ. That gap turned out
not to be about the loop at all — the benchmark was short enough to fit inside
one CPU-quota period and reports `min()` over its repetitions, so it never paid
the throttle the loop paid every frame. Two lessons, not one: treat isolated
benchmarks as upper bounds, and be suspicious of a `min()` when the thing you
are chasing is intermittent.

**On a container, check for CPU throttling before believing any wall-clock
number.** `nproc`, `os.cpu_count()` and `os.sched_getaffinity()` all report the
host's cores; a CPU limit is enforced by quota accounting instead. The
signature is visible in a profiler trace as stalls that all end at the same
phase modulo 100 ms, and `tools/analyze_trace.py` tests for it. This one factor
was worth 4.3× on the splat and had been distorting every number here.

**Do not count `ProfilerStep` as device work.** torch projects it onto the GPU
track as a `gpu_user_annotation` spanning the whole step, so summing that
category reports the device ~100% busy whatever it is doing — and hides the
idle you are looking for. In `key_averages()` the tell is `ProfilerStep*` with
a CUDA share above 100%.

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
in `tests/test_torch_pipeline.py` is unchanged at 2.00e-04. (Those two values
are from the superseded configuration. What matters here is that they did not
move across the optimization steps, which is a statement about the steps.)

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
per-member cost is dominated by the splat, so the 2.4x recompute penalty
measured at one N on one device should not be assumed to hold generally.

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
| `torch_num_threads=N` | threads per rank. Defaults to the cgroup CPU quota, or to 1 on a device run when no quota is discoverable. Set `OMP_NUM_THREADS` too — see "The container was CPU-throttled, which was worth more than the code". |
| `torch_taper_width=W` | taper width in Å; narrower is closer to gemmi but harsher on gradients |

`torch_max_atoms_per_batch` still exists but is now secondary: batching is
budgeted by pairs, and the atom cap almost never binds.

### Diagnostics

Off by default, all of them; they exist because each one answered a question
that wall-clock timing could not, and the same questions will come back.

| option | what it answers |
|---|---|
| `torch_timing=True` | where the frame goes, host and device, plus the loop's wall time and the residual the phases do not account for. Read `ms/frame` against the `calls` column: `traj read` fires once per *chunk*. |
| `torch_splat_stats=True` | is it more work or slower work? Counts atoms, chunks, and pairs both ideal and as batched, comparable directly with `bench_splat.py`. Also prints the splat's per-frame series with frame 0 separated, which distinguishes a one-time cost from a per-frame one. |
| `torch_profile_frames=N` | records N frames with `torch.profiler` (CPU + CUDA), writes a Chrome trace and prints the per-operator table. Skips frame 0, warms on frame 1. Cannot be combined with `profile=True`, and perturbs what it measures. |
| `torch_reuse_grid=True` | splat into one preallocated grid instead of a fresh allocation per frame. Measured: no effect, per-frame or one-time. |
| `torch_free_early=True` | drop the previous frame's density and F before the splat. Measured: no effect. |
| `profile=True` | cProfile. Misleads badly for torch — see "How to measure this pipeline". |

`tools/analyze_trace.py <trace.json>` reads what `torch_profile_frames` writes:
GPU busy vs idle, the gap size distribution, the largest gaps with the CPU op
and CUDA API call each lands in, and a direct test for the CFS-throttle
signature. Standard library only, so it runs wherever the trace was produced.
