"""Automatic selection of the torch engine's performance knobs.

The knobs in `docs/performance.md`'s "Tuning knobs" table are all defensible
defaults and all wrong somewhere: `torch_compile` is a net LOSS below a couple
of hundred frames on CUDA, and `max_pairs_per_batch`'s default was fitted to a
CPU cache and the splat's own docstring says to expect a GPU to want more. This
module picks them from things that are known before the frame loop starts --
frame count, grid shape, atom count, and what the device reports about itself
-- so that the common case needs no experiment.

EVERY RULE HERE TRACES TO A MEASUREMENT OR TO A HARDWARE QUERY. Nothing is
invented; where a number would have to be guessed the function says so and
keeps the measured default instead. The measurements are in
`docs/performance.md`, and each function names the section it came from.

Framework-independent by design: no torch import at module scope, so
`import lunus.sf` stays torch-free (see the package docstring). The device is
described by the small `DeviceInfo` record, which `describe_device()` fills in
from torch when torch is in hand.
"""

from collections import namedtuple

# What the splat's (m, K) working buffers cost per atom-voxel pair. float32,
# and the fused kernel is float32 by design -- MPS has no float64 at all.
BYTES_PER_PAIR = 4

# docs/performance.md, "How the splat got fast" / splat_density's docstring:
# 4M pairs is ~16 MB per buffer, chosen to stay cache-resident. The curve is
# flat from 2M to 12M on the 135k-atom CPU case and rises on both sides --
# 0.86 s at 4M, 1.00 s at 16M, 1.25 s at 48M, 1.01 s at 0.5M.
MAX_PAIRS_CPU = 4_000_000

# How many (m, K) buffers may be live at once, for the memory cap only. The
# fused kernel keeps "a couple"; 4 is deliberate headroom, since the penalty
# for being wrong here is an out-of-memory abort rather than a slow run.
LIVE_PAIR_BUFFERS = 4

# Fraction of FREE device memory the pair buffers may occupy. The grid, the
# FFT workspace and the offset tables all come out of the same pool.
PAIR_BUDGET_FRACTION = 0.25

# docs/performance.md, "NVIDIA GPU (CUDA)": frame 0 costs 8961.5 ms compiled
# against 195.9 ms eager, so ~8.77 s is inductor; the median goes 66.7 -> 25.0
# ms, saving 41.7 ms/frame. A persistent TORCHINDUCTOR_CACHE_DIR takes frame 0
# to ~8.35 s warm, so this barely moves with a warm cache.
COMPILE_SECONDS_CUDA = 8.77
COMPILE_SAVED_PER_FRAME_CUDA = 0.0417

# docs/performance.md, "CPU": 2.00 s eager against 0.85 s compiled at 6
# threads, for ~1.4 s of one-off compilation.
COMPILE_SECONDS_CPU = 1.4
COMPILE_SAVED_PER_FRAME_CPU = 1.15


DeviceInfo = namedtuple("DeviceInfo", "kind l2_bytes free_bytes total_bytes name")
DeviceInfo.__new__.__defaults__ = (None, None, None, None)


def describe_device(device, torch_module=None):
    """A DeviceInfo for 'cpu', 'mps', 'cuda' or 'cuda:N'.

    Everything except `kind` is best-effort: a field stays None when torch
    cannot answer, and every rule below treats None as "fall back to the
    measured default" rather than guessing.
    """
    kind = str(device).split(":")[0].lower()
    if kind != "cuda":
        return DeviceInfo(kind=kind)

    torch = torch_module
    if torch is None:                      # pragma: no cover - import guard
        try:
            import torch
        except ImportError:
            return DeviceInfo(kind=kind)
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:                      # no device, bad index, no driver
        return DeviceInfo(kind=kind)

    # L2_cache_size is not present on every torch/driver combination, hence
    # getattr rather than attribute access -- its absence must not be fatal.
    l2 = getattr(props, "L2_cache_size", None)
    total = getattr(props, "total_memory", None)
    try:
        free, total_q = torch.cuda.mem_get_info(device)
    except Exception:
        free, total_q = None, None
    return DeviceInfo(kind=kind, l2_bytes=l2 or None,
                      free_bytes=free, total_bytes=total_q or total,
                      name=getattr(props, "name", None))


def recommended_max_pairs(info, bytes_per_pair=BYTES_PER_PAIR):
    """Atom-voxel pair budget for splat_density. Returns (pairs, why).

    The CPU value is measured (see MAX_PAIRS_CPU). The criterion behind it is
    "one working buffer stays resident in cache", so the transferable rule is
    not the number 4M but the cache it was sized against -- applied to a GPU
    that means its L2, which is tens of MB rather than the tens of MB of an
    L3 slice, and is why splat_density's docstring says to expect a GPU to
    prefer a larger value.

    So: on CUDA, size the buffer to the L2 the device reports, then clamp so
    LIVE_PAIR_BUFFERS of them fit in PAIR_BUDGET_FRACTION of free memory. If
    the device will not report its L2, KEEP THE CPU DEFAULT and say so -- a
    guess here is a silent performance regression, and the honest fallback is
    the value that was actually measured.

    The clamp never raises the budget, only lowers it, so a small free-memory
    reading cannot make this recommend something larger than the cache rule.
    """
    if info.kind != "cuda":
        return MAX_PAIRS_CPU, "measured CPU default (cache-resident at ~%.0f MB)" % (
            MAX_PAIRS_CPU * bytes_per_pair / 1e6)

    if not info.l2_bytes:
        return MAX_PAIRS_CPU, (
            "device did not report an L2 size, so the measured CPU default is "
            "kept; calibrate with tools/bench_splat.py --device cuda")

    pairs = int(info.l2_bytes // bytes_per_pair)
    why = "sized to the device L2 of %.0f MB" % (info.l2_bytes / 1e6)

    if info.free_bytes:
        cap = int(info.free_bytes * PAIR_BUDGET_FRACTION
                  / (LIVE_PAIR_BUFFERS * bytes_per_pair))
        if cap < pairs:
            pairs, why = cap, (
                "capped by free device memory (%.1f GB) rather than L2"
                % (info.free_bytes / 1e9))

    # Never go below the measured default: that value is known to work, and a
    # tiny budget costs per-call overhead (0.5M measured at 1.01 s against
    # 0.86 s at 4M) on top of whatever pressure prompted the cap.
    if pairs < MAX_PAIRS_CPU:
        pairs, why = MAX_PAIRS_CPU, why + "; floored at the measured default"
    return pairs, why


def recommended_compile(n_frames, info):
    """Whether torch.compile pays for itself. Returns (bool, why).

    n_frames is the count THIS PROCESS will splat -- compilation is per
    process, so under `mpirun -n N` it is frames/N even though xtraj warms the
    inductor cache on rank 0 first.

    The trade is a fixed one-off against a per-frame saving, so the break-even
    is a frame count and the decision is arithmetic rather than a preference.
    """
    if info.kind == "mps":
        return False, ("torch.compile has no working Metal backend "
                       "(InductorError on c10/metal/reduction_utils.h); "
                       "skipping it avoids the failed attempt")
    if info.kind == "cuda":
        one_off, saved = COMPILE_SECONDS_CUDA, COMPILE_SAVED_PER_FRAME_CUDA
    else:
        one_off, saved = COMPILE_SECONDS_CPU, COMPILE_SAVED_PER_FRAME_CPU

    break_even = one_off / saved
    if n_frames >= break_even:
        return True, ("%d frames per process against a break-even of %d "
                      "(%.1f s one-off, %.0f ms/frame saved)"
                      % (n_frames, round(break_even), one_off, saved * 1e3))
    return False, ("%d frames per process is below the break-even of %d "
                   "(%.1f s one-off would not be repaid)"
                   % (n_frames, round(break_even), one_off))


def estimated_peak_bytes(n_atoms, grid_shape, max_pairs,
                         bytes_per_pair=BYTES_PER_PAIR):
    """Rough peak device bytes for one forward splat + FFT.

    This is the term that grows with the system and the resolution, which is
    what makes a run at a finer d_min fail where a coarser one fits: the grid
    goes as d_min^-3.

    Counted: the real density grid, the rfft output (complex64, last axis
    halved), and the live pair buffers. NOT counted: the element offset
    tables, torch's caching-allocator slack, or anything autograd retains --
    xtraj runs forward-only, but a guided step does not, and there the
    retained intermediates dominate (see "Ensembles: memory is the binding
    constraint"). Treat this as a floor, not a budget.
    """
    nu, nv, nw = (int(g) for g in grid_shape)
    density = nu * nv * nw * 4
    rfft = nu * nv * (nw // 2 + 1) * 8
    pairs = LIVE_PAIR_BUFFERS * int(max_pairs) * bytes_per_pair
    coords = int(n_atoms) * 3 * 4
    return density + rfft + pairs + coords


def memory_warning(n_atoms, grid_shape, max_pairs, info,
                   bytes_per_pair=BYTES_PER_PAIR):
    """A one-line warning when the run looks close to the device, else None."""
    if not info.free_bytes:
        return None
    need = estimated_peak_bytes(n_atoms, grid_shape, max_pairs, bytes_per_pair)
    if need < 0.7 * info.free_bytes:
        return None
    return ("estimated peak %.1f GB against %.1f GB free on %s -- this is a "
            "floor and excludes allocator slack, so consider a coarser d_min, "
            "a smaller torch_max_pairs_per_batch, or more ranks"
            % (need / 1e9, info.free_bytes / 1e9, info.name or info.kind))
