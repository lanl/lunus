# Performance

**Every number on this page was measured on one specific configuration.** They
are meaningless against a different system, so it is stated once here and
assumed throughout:

| | |
|---|---|
| topology | `examples/compare_gemmi/top_bfac.pdb.gz`, 135,834 atoms |
| B-factors | per-residue, uniform on [10, 100] |
| cell / space group | **88.451, 88.451, 39.823 / P4₃ — WRONG, see below** |
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
    --cell 34.196,45.558,99.044,90,90,90 --grid 120,160,360 [--device mps|cuda]
    # the tables below were taken with --cell 88.451,88.451,39.823,90,90,90
    # --grid 300,300,144; see the note directly above
```

It synchronizes around each timed region, without which GPU timings measure
dispatch rather than execution. Set `OMP_NUM_THREADS` to control CPU threads —
**on a CPU-limited container this is not optional**, see below.

Figures re-measured 2026-08-11; the frame-loop optimization work is 2026-08-14.

## The configuration above is wrong for this system (2026-08-15)

Every absolute number on this page was measured with `unit_cell=88.451,88.451,
39.823` and `space_group=P43`, which were carried over from a different example.
The crystal this trajectory was simulated from is **PDB 7FPV: 34.196 × 45.558 ×
99.044, P2₁2₁2₁, Z = 4** (now kept alongside the data as `7FPV.pdb`), and the
simulation box is exactly a 2×2×2 supercell of it — 68.392/34.196,
91.116/45.558 and 198.088/99.044 are all 2.000000, where the old cell had no
integer relationship on any axis.

**What this invalidates:**

- **Every ms/frame figure**, because the grid changes: `d_min` 0.9 gives
  120 × 160 × 360 = 6.9M voxels on the correct cell against 300 × 300 × 144 =
  12.96M on the old one, a factor of 1.9 in the splat and the FFTs alike. The
  timings need re-measuring; nothing about them can be rescaled by hand,
  because the atom-voxel pair count changes with the grid too.
- **Any physical reading of the Icalc/diffuse output** from those runs, since
  folding into an unrelated cell scattered the 32 protein copies at
  incommensurate positions rather than stacking them.

**What survives unchanged:**

- **Torch-vs-gemmi parity.** Both engines were given the same cell and fold
  identically, so they agree with each other whatever cell they are given.
- **The float32 study, the CPU-throttling diagnosis, and every ratio between
  phases** — splat against FFT, host against device, and the conclusion that
  the pipeline was quota-throttled. Those are properties of the code and the
  machine, not of the cell.
- **The optimization results**, since before and after were measured under
  identical settings.

The superseded configuration is kept commented in `run_xtraj.sh` so the numbers
below can be reproduced until they are replaced.

## Where this stands (2026-08-14)

The CUDA frame loop went **571.2 → 87.3 ms/frame, 6.5×** — 251 frames in 21.9 s
against 143 s. Three changes, each measured, outputs bit-identical throughout:

| | ms/frame removed | |
|---|---|---|
| `flex.vec3_double` built from a flat buffer instead of row by row | ~190 | every engine |
| the torch path no longer round-trips through `xray.structure` | ~97 | torch only |
| torch's thread pool sized to the cgroup CPU quota, not the host's cores | ~228 | any container |

The splat is now 62.9 ms/frame against `bench_splat`'s 62.6 for identical work,
so the frame loop runs it at benchmark speed and the long-standing "the splat
is slow in situ" discrepancy is closed. It was CPU throttling.

Read in order: "Where the time actually goes" for how the frame was
instrumented and what it showed, then "Resolved: the container was
CPU-throttled" for the largest single factor and the one most likely to bite
again elsewhere.

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

`torch_timing=True` reports per-phase timings for the whole frame loop, host
and device. Device phases synchronize on both sides; host phases do not need
to, since they queue no device work and every device phase has already
synchronized on exit. The report ends with the loop's wall time and the
residual the phases do not account for — if that residual is large, the timers
are missing something and the rest of the table should not be trusted as a
budget.

The host phases are engine-independent, so the report is printed for
`engine=gemmi` and `engine=cctbx` too, with the engine's own call bracketed as
a single phase.

On CUDA, `engine=torch`, the documented `compare_gemmi` configuration —
135,834 atoms, `d_min` 0.9, grid 300×300×144, 251 frames in one chunk, 143 s
(571 ms/frame). **This is the "before" state**; the after is at the end of the
section:

| phase | | ms/frame | share |
|---|---|---|---|
| splat | device | 257.6 | 45.1% |
| set_sites_cart | host | 191.5 | 33.5% |
| element idx | host | 43.7 | 7.6% |
| xrs.select | host | 29.9 | 5.2% |
| sites_frac+occ | host | 23.0 | 4.0% |
| traj read | host | 12.9 | 2.3% |
| into cctbx | device | 4.6 | 0.8% |
| accumulate | host | 2.3 | 0.4% |
| coords→numpy | host | 2.1 | 0.4% |
| device→host | device | 1.1 | 0.2% |
| host→device | device | 0.9 | 0.2% |
| symmetrize | device | 0.8 | 0.1% |
| fft+extract | device | 0.6 | 0.1% |
| **total timed** | | **571.0** | |
| frame loop wall | | 571.2 | |
| unaccounted | | 0.2 | 0.0% |

The residual is 0.2 ms/frame, so this is the whole loop, not a sample of it.
(An earlier device-only run of the same configuration gave 553 ms/frame; the
difference is within the run-to-run variation documented below, not timer
overhead.)

**The FFT is 0.6 ms.** So are symmetrization and both transfers. Everything in
the GPU pipeline except the splat is free; there is nothing to win there.

**The host half is 305 ms/frame, 53% of the run, and it is almost all cctbx
marshalling.** `set_sites_cart` alone — at the time, one phase covering both
building a `flex.vec3_double` from the frame's coordinates and pushing it into
the `xray.structure` — is 191.5 ms/frame, a third of total runtime and the
second-largest cost in the pipeline after the splat. Add `xrs.select`,
`sites_frac+occ` and `element idx` and it is 288 ms/frame, 50% of the run,
spent converting coordinates the trajectory already gave us into an
`xray.structure` and immediately reading them back out.

All of that host cost has since been removed — 191.5 ms of it by one line, the
rest by dropping the round trip from the torch path. Both are below; the table
is the state of the pipeline when the problem was found, not as it stands now.

cProfile saw none of this proportionally: it attributed ~39 ms/frame to
identifiable calls (`add_scatterers` 14 ms, `structure.select` 14 ms,
`resolution_filter` 4 ms, `set_sites_frac` 3 ms) — the right functions, an
order of magnitude too small.

**The trajectory read is not the problem.** It was the leading suspect and it
is 12.9 ms/frame, 2.3%. Worth recording *why* it looked otherwise: a coarse
local check (Mac, CPU, `d_min` 2.5, 3 frames) put `traj read` at 262 ms/frame
and top of the table. That run had one chunk of 3 frames, so a mostly fixed
per-chunk cost was divided by 3; here the same kind of cost is divided by 251.
`traj read` fires **once per chunk**, which is why the report prints call
counts — read its ms/frame against the `calls` column or it will mislead you
exactly this way.

### `flex.vec3_double(ndarray)` converts row by row

Splitting the `set_sites_cart` phase in two put **99% of it in the
`flex.vec3_double` construction, not in cctbx's setter** — 87.9 ms against
0.8 ms per frame, on the local CPU configuration. Benchmarked directly on
135,834 atoms:

| | ms |
|---|---|
| `flex.vec3_double(a)` — a as an (N, 3) ndarray | 85.5 |
| `flex.vec3_double(flex.double(a.ravel()))` | 0.5 |

170×, with `max|diff| = 0` between the results: the ndarray overload converts
row by row, while `flex.double` on a contiguous flat buffer takes the bulk
path. The fix is the second form, and it is applied at both conversion sites in
`xtraj.py`. Verified on a 5-frame run: Icalc correlation 1.000000, R 0.000000,
`diffuse.hkl` byte-identical, `fcalc.mtz` data bit-identical (the file differs
only in its embedded timestamp). Local frame loop went 469.7 → 374.5 ms/frame.

This is **not** torch-specific — every engine pays it, and the gemmi and cctbx
engines cannot avoid the `xray.structure` the way the torch path could.

The CUDA table above has not been re-measured since; on those numbers the fix
should take ~190 ms off the 571 ms frame, but that is arithmetic, not a
measurement.

### The torch path skips the `xray.structure` entirely

The remaining round trip — `xrs.select`, `sites_frac+occ`, `element idx`, plus
the `coords->flex`/`set_sites_cart` pair feeding it — asks cctbx for
information that either does not change between frames or is one matmul from
what mdtraj already handed over. `engine=torch` no longer asks:

- **fractional coordinates** are `tsites[i] @ frac_matrix.T`, using
  `unit_cell().fractionalization_matrix()`;
- **the selection** becomes row indices into the trajectory's atom axis once,
  at setup (and is skipped altogether when every atom is selected);
- **occupancies and element indices** are read from the structure once, at
  setup.

Setup asserts that the matrix reproduces cctbx's own `sites_frac()` on the
topology coordinates — measured agreement 2.2e-16 — because a bypass that
disagreed would move every atom and change every structure factor without
raising anything.

`translational_fit=True` keeps the round trip: it fits against
`xrs.sites_frac()` and writes back with `set_sites_frac()`, so it genuinely
needs the structure maintained. The other engines are untouched.

Measured locally (5 frames, CPU, `d_min` 2.5): the five host phases collapse
into one `frac coords` phase at **0.3 ms/frame**, and the frame loop goes
374.5 → 322.9 ms/frame — 469.7 before either change. Output is bit-identical
to the pre-bypass code: Icalc correlation 1.000000, R 0.000000, `fcalc` data
identical, `diffuse.hkl` byte-identical. The gemmi engine's output is
byte-identical across the change as well.

### After both changes, on CUDA

Same configuration, 251 frames, 79 s (315 ms/frame) against 143 s (571):

| phase | | ms/frame | share |
|---|---|---|---|
| splat | device | 290.6 | 92.2% |
| traj read | host | 13.0 | 4.1% |
| into cctbx | device | 4.1 | 1.3% |
| coords→numpy | host | 2.1 | 0.7% |
| accumulate | host | 1.9 | 0.6% |
| symmetrize | device | 0.8 | 0.3% |
| host→device | device | 0.7 | 0.2% |
| device→host | device | 0.7 | 0.2% |
| fft+extract | device | 0.6 | 0.2% |
| frac coords | host | 0.5 | 0.2% |
| **total timed** | | **315.0** | |
| frame loop wall | | 315.2 | |
| unaccounted | | 0.2 | 0.0% |

**1.81× overall.** The host half went from 305 ms/frame to 17.5, and what is
left of it is the trajectory read (13.0, once per chunk) rather than anything
per-atom. The `xray.structure` marshalling that was 50% of the frame is now
0.5 ms.

The splat *appeared* to get slower over the same change, 257.6 → 290.6
ms/frame. It had not: with the host work gone, the splat became the only thing
left to absorb the CPU-throttle stalls that the next section is about. Its true
cost is 62.9 ms.

### After the thread fix as well: the state of the pipeline

Same configuration and trajectory as the 571 ms baseline, 251 frames,
`OMP_NUM_THREADS=1`, 21.9 s:

| phase | | ms/frame | share |
|---|---|---|---|
| splat | device | 62.9 | 72.2% |
| traj read | host | 13.0 | 14.9% |
| into cctbx | device | 4.0 | 4.5% |
| coords→numpy | host | 2.1 | 2.4% |
| accumulate | host | 1.8 | 2.1% |
| host→device | device | 0.8 | 0.9% |
| symmetrize | device | 0.8 | 1.0% |
| device→host | device | 0.6 | 0.7% |
| fft+extract | device | 0.6 | 0.7% |
| frac coords | host | 0.5 | 0.5% |
| **total timed** | | **87.2** | |
| frame loop wall | | 87.3 | |
| unaccounted | | 0.1 | 0.2% |

**571.2 → 87.3 ms/frame, 6.5× end to end**, 143 s → 21.9 s for the run. The
three contributions, in the order they were found:

| | ms/frame removed |
|---|---|
| `flex.vec3_double` built from a flat buffer | ~190 |
| the torch path skipping `xray.structure` | ~97 |
| torch's thread pool sized to the CPU quota | ~228 |

**The splat is 62.9 ms against `bench_splat`'s 62.6 for the same work.** The
frame loop now runs the splat at benchmark speed; there is no gap left to
explain. What remains in the frame is 13.0 ms of trajectory read (once per
chunk, so this is `chunk=`-dependent) and about 9 ms of everything else.

## Resolved: the container was CPU-throttled

**The frame loop was not slow. The container was frozen for most of every
100 ms.**

`torch_profile_frames=3` plus `tools/analyze_trace.py` on the documented CUDA
configuration: the GPU is busy 54.1 ms of a 303 ms frame, **82% idle**, and
that idle is not spread across the ~5,500 kernel launches per frame. Nine gaps
of 24–90 ms carry **82% of it**, and they end at:

```
45.86  45.67  45.86  45.83  45.82  45.66  45.86  45.90  45.76   ms mod 100 ms
```

All nine, within ±0.1 ms, across 900 ms of trace. During those gaps the host is
not in a CUDA call at all — `cudaLaunchKernel` covers 0% of the six longest —
and the CUDA API totals show no allocator activity whatsoever: 141 ms of
`cudaLaunchKernel` across 16,134 calls, `cudaMalloc` not even in the top ten.

Nothing in a numerical kernel ends on a wall-clock boundary. The Linux CFS
bandwidth controller — what a Kubernetes CPU limit becomes — accounts in fixed
100 ms periods and unfreezes a throttled cgroup when the quota refills at the
period boundary. The process burns its quota in ~12 ms of wall time and then
sits stopped for ~88 ms.

`tools/analyze_trace.py` now tests for this directly and prints the phase and
its spread. Confirm from inside the container with:

```bash
cat /sys/fs/cgroup/cpu.max     # quota and period, in us
cat /sys/fs/cgroup/cpu.stat    # nr_throttled, throttled_usec -- diff over a run
```

**What this invalidates.** Every wall-clock number in this file measured on
that pod is partly a measurement of the CPU quota. Specifically:

- The 4.6× "in situ vs isolation" gap below. `bench_splat` reports `min()` over
  its repetitions, so it systematically reports the repetition that was not
  throttled; the frame loop has no such escape and pays on every frame.
- The splat's 257.6 → 290.6 ms/frame rise after the host-side work was removed.
  The throttle stalls did not go away with that work — they landed on the splat
  instead, which is the only thing left to absorb them.
- The 35% frame-to-frame spread on identical work, and the 25% run-to-run
  variation recorded throughout: whether a frame straddles a refill boundary.
- The allocation probes, whose 10-frame signal was never about allocation.

The **pair counts, kernel times and op counts are unaffected** — they are
counts and device-side durations, not host wall time. So the work-volume
conclusion stands: the loop and the benchmark do identical work.

**The cause, confirmed.** The pod has **3 CPUs**. torch sized its thread pool
from the host's core count and got **104**. Those threads spin-wait, so the
quota is spent far faster than useful work consumes it, and the process spends
most of every 100 ms period frozen.

**The fix, measured.** One environment variable; 10 frames, CUDA, everything
else identical:

| | splat median | splat min | frame 0 |
|---|---|---|---|
| default (104 threads) | 287.9 ms | 247.4 | 351.2 |
| `OMP_NUM_THREADS=2` | **66.3 ms** | 64.9 | 189.2 |

**4.3×** — and 66.3 ms lands on `bench_splat`'s 62.6 ms for the same work.
There was never a discrepancy between the loop and the benchmark. There was a
thread-count difference between two ways of starting a process on a 3-CPU
container.

**Confirmed at full scale**, 251 frames, `OMP_NUM_THREADS=1`, unprofiled, same
trajectory as the 571 ms baseline: **87.3 ms/frame**, splat **62.9 ms**. Table
in "After the thread fix as well" above.

The host profile shows the same thing: across the three profiled frames
`aten::index` self CPU falls from 190.9 ms to 14.8 ms and `aten::remainder`
from 90.1 ms to 5.8 ms, for *identical* op counts. That is time spent waiting
to be scheduled, attributed to whichever op was unlucky enough to be running.

**Do not trust `nproc` here.** It reports 104, as do `os.cpu_count()` and
`os.sched_getaffinity()`: a CPU limit is enforced by accounting, not by
restricting which cores a process may run on. Only the quota knows.

`xtraj.py` now reads it — `/sys/fs/cgroup/cpu.max`, falling back to cgroup v1 —
and sets torch's thread count to match, reporting what it did and why. When no
quota is discoverable and the run is on a device, it defaults to **one thread**,
since the host only dispatches and one thread cannot oversubscribe a limit that
cannot be seen. A single-process CPU run, where threads are the point, is left
alone. `torch_num_threads=N` overrides all of it.

That covers torch. **It cannot cover numpy/BLAS**, whose thread pool is sized
from the environment before this program starts — so still export
`OMP_NUM_THREADS` to the pod's real CPU count on every run.

## The former discrepancy: the splat in situ vs in isolation

**Resolved — it was the CPU throttling above.** With the thread pool sized to
the quota, the in-loop splat is 62.9 ms/frame against `bench_splat`'s 62.6, and
nothing is left to explain. The section is kept because the eliminations below
are still valid results about the splat, and because the shape of the mistake
is worth remembering: a wall-clock difference was attributed to the code for
weeks while the actual cause was the container's CPU quota, invisible to every
measurement that did not look at *when* the device was idle.

**The observation, as it stood.** On CUDA the splat cost ~255 ms/frame inside
the frame loop, while repeating the identical call on the identical tensors in
the same process gave 62.8 ms — matching `bench_splat` exactly. Same data, same
process, seconds apart, fourfold different. The repeat was fast because it was
short enough to fit inside one quota period; the frame loop paid the throttle
on every frame.

**It is not extra work.** `torch_splat_stats=True` counts what the loop
actually evaluates, against `bench_splat`'s count on the same structure and
grid:

| | pairs (ideal) | pairs (as batched) | chunks | time | rate |
|---|---|---|---|---|---|
| `bench_splat`, back-to-back | 324,986,306 | — | — | 62.8 ms | ~5,175e6 pairs/s |
| frame loop, 10 frames | 324,928,670–325,024,090 | 336,533,010–336,808,766 | 90 | 289.8 ms | ~1,121e6 pairs/s |

The ideal counts agree to **0.02%**, the chunk count is identical frame to
frame, and the chunk padding is 1.036 with a spread of 0.07%. So the loop
evaluates the same number of atom-voxel pairs in the same number of batches as
the benchmark, and does it at **a fifth of the rate**. Whatever this is, it is
not the coordinates, the batching, or the amount of arithmetic.

**And the rate is not even stable within one run.** Over frames 1–9 of a
10-frame run, with the pair count fixed to 0.02%, the splat ranges 219.2 to
296.0 ms — **35%**. Identical work, same process, consecutive frames. Whatever
sets the rate varies on the timescale of a few hundred milliseconds, which is
the timescale of clock and power management rather than of anything in this
code. It also means a single frame, or a 10-frame average, cannot be used to
compare two configurations; that is what burned the allocation probes above.
The 251-frame series (`torch_splat_stats=True` prints first-tenth vs
last-tenth, 25 frames each at that length) is the trend measurement worth
having, and has not been run.

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
| a fresh grid allocation per frame | `torch_reuse_grid=True`, 251 frames | 290.0 vs 290.6 ms — 0.2% |
| previous frame's grid and F still alive during the splat | `torch_free_early=True`, 251 frames | same, 0.2% |

The last two were motivated by the difference between how the splat is
benchmarked (repeated calls reusing one set of buffers) and how the frame loop
calls it (a new 51.8 MB grid each time, with the previous frame's grid and F
still referenced). Neither matters. **Both flags remain, defaulting to off** —
they cost nothing and re-testing on another machine is a command-line flag.

**Including as a one-time cost.** The 251-frame result bounds only the
*persistent per-frame* effect. Preallocation also moves any allocator growth
out of the loop entirely, and a one-time saving divides by the frame count —
over 10 frames it would look ten times larger than over 251, which is
arithmetically consistent with what the 10-frame runs showed. The per-frame
series settles it, because a cost moved to setup disappears from **frame 0**
and nowhere else:

| | frame 0 | rest: min / median / max |
|---|---|---|
| baseline | 365.0 | 219.2 / 287.3 / 296.0 |
| `torch_reuse_grid=True` | 377.0 | 211.9 / 277.5 / 294.1 |

Frame 0 does not move (it is marginally *higher*), so there is no one-time cost
to recover either. Frame 0 does carry ~78 ms of warmup over the median — that
is real, and unrelated to the grid buffer.

For the record of how the 10-frame comparison misled: the four combinations
spanned 237.8–294.9 ms/frame with an ordering that was not self-consistent
(`free_early` alone was the *worst* of the four and the best in combination),
and ~7–10 ms of jitter was visibly landing in unrelated phases (`frac coords`
at 9.5 ms where it is normally 0.5, `fft+extract` at 10.4 where it is 0.6).

**A methodological error worth recording.** The repeat test originally ran only
on frame 0 — whose in-loop time is the slowest of the run because it carries
one-time warmup. Comparing a warm repeat against a warmup-polluted baseline
proves nothing, and may have manufactured part of the puzzle. On MPS that
confound inflates the ratio from 1.18x (frame 3, clean) to 1.76x (frame 0).

**The GPU is idle 82% of the frame.** `torch_profile_frames=3` on the
documented CUDA configuration:

| | per frame |
|---|---|
| frame wall (`ProfilerStep`) | 294.6 ms |
| **total CUDA kernel time** | **54.1 ms** |
| total CPU time | 303.1 ms |

The kernels are not slow. 54.1 ms of device time per frame is in the same place
as `bench_splat`'s 62.8 ms *wall* for the same work — so whatever the splat is
waiting on, it is not the arithmetic. The scatter, the part that uses atomics
and was the standing suspect, is 19.3 ms across all three frames (6.4 ms/frame,
270 calls at 71.6 µs) — 12% of device time.

Meanwhile the host is busy for essentially the whole frame, and the op counts
say why. Per frame, in eager mode across 90 chunks × 9 elements:

| op | calls/frame | self CPU (3 frames) | self CUDA (3 frames) |
|---|---|---|---|
| `aten::mul` | 1,623 | 27.3 ms | 52.2 ms |
| `aten::add` | 904 | 15.1 ms | 31.7 ms |
| `aten::index` | 568 | **103.0 ms** | 8.6 ms |
| `aten::exp` | 450 | 6.8 ms | 9.8 ms |
| `aten::index_add_` | 90 | 1.7 ms | 19.3 ms |

~4,400 dispatched ops per frame. `aten::index` costs 12× more CPU than device
time — it is the per-chunk `c_rem_all[sel]`, `atom_A[sel]` … gathers, which are
cheap on the GPU and are being paid for on the host.

**Caveat on the CPU column:** profiling inflates per-op host cost, so 303
ms/frame is an upper bound. The device column does not have that problem —
kernel durations come from CUPTI. The 82%-idle conclusion rests on the device
column and the wall time, both of which are sound.

**What this does not yet explain** is why `bench_splat` gets 62.8 ms *wall* for
the same op sequence, which would require the same ~4,400 dispatches to cost
almost nothing. That figure has not been reproduced on this machine in the
current state — it dates from 2026-08-11 — and everything called a
"discrepancy" here depends on it. Re-measure it before theorising further.

**The instrument.** With work volume ruled out, the question
is where 4.6× of *rate* goes, and wall-clock timing cannot answer it.
`torch_profile_frames=N` records N frames with `torch.profiler` (CPU + CUDA
activities), skipping frame 0 and using frame 1 as the profiler's own warmup,
then writes a Chrome trace (`torch_trace_rank0.json`, load in
`chrome://tracing` or `ui.perfetto.dev`) and prints the per-operator table.
N=3 is enough. It cannot be combined with `profile=True`, and it perturbs what
it measures — take timings from an unprofiled run.

Three shapes the answer can take, distinguishable at a glance in the trace:

- **one kernel 4.6× slower** than in the benchmark → occupancy, clocks, or
  memory behaviour of the kernel itself;
- **the same kernels with gaps between them** → the loop is stalling on the
  host. The nine per-frame `K_needed.detach().to("cpu")` calls in
  `splat_density` are the obvious candidate: each is a device→host copy that
  synchronizes, once per element per frame;
- **more kernel launches than the benchmark issues** → the batching differs
  after all, though the chunk counts already say it does not.

**Still open.** Whether later frames' coordinates genuinely cost more than
frame 0's — which would make this a data effect and would also explain why
`bench_splat`, which uses the *topology's* coordinates rather than the
trajectory's, has always been fast. The instrument for this is a like-for-like
comparison on a later frame; if that shows a large ratio, the next step is
`torch.profiler` with CUDA activity tracing rather than more black-box tests.

**Perspective, after the fact:** the splat is 62.9 ms of an 87.3 ms frame, 72%,
and it matches the benchmark exactly. Making it faster is now a question about
the kernel itself — see "How the splat got fast" below for what has already
been done there — not about anything the frame loop does around it.

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
| `torch_num_threads=N` | threads per rank. Defaults to the cgroup CPU quota, or to 1 on a device run when no quota is discoverable. Set `OMP_NUM_THREADS` too — see "Resolved: the container was CPU-throttled". |
| `torch_taper_width=W` | taper width in Å; narrower is closer to gemmi but harsher on gradients |

The old `torch_max_atoms_per_batch` is deliberately gone: batching is budgeted
by pairs now, and the atom cap almost never bound.

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
