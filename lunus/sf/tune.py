"""Automatic selection of the torch engine's performance knobs.

`max_pairs_per_batch`'s default was fitted to a CPU cache, and the splat's own
docstring says to expect a GPU to want more -- so on a GPU the shipped default
is knowingly wrong and the user is left to discover that with `bench_splat.py`.
This module picks it instead, from things known before the frame loop starts:
grid shape, atom count, and what the device reports about itself.

Keep its worth in proportion. Swept on GB10, the whole 8x range of budgets
spans 1.089x on the compiled splat, so this is a sub-1% knob there -- it is
here to stop a GPU silently inheriting a CPU-shaped default, not because
tuning it is where the time goes. On that machine the splat was 68% of the
frame and the FFT and host phases were the rest; see docs/performance.md,
"A second machine".

WHAT IS AND IS NOT DECIDED HERE. `recommended_max_pairs()` and
`memory_warning()` are wired into xtraj; nothing else is. `torch_compile` in
particular is the user's to set -- its break-even is a frame count, and the
one measured on one machine turned out not to transfer (2.8 frames on GB10
against ~210 on a faster card), so choosing it automatically would be worse
than leaving it alone. docs/performance.md, "A second machine".

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


DeviceInfo = namedtuple("DeviceInfo", "kind l2_bytes free_bytes name",
                        defaults=(None, None, None))


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
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        free = None
    return DeviceInfo(kind=kind, l2_bytes=l2 or None, free_bytes=free,
                      name=getattr(props, "name", None))


def recommended_max_pairs(info, bytes_per_pair=BYTES_PER_PAIR):
    """Atom-voxel pair budget for splat_density. Returns (pairs, why).

    On CUDA, size the buffer to the L2 the device reports, then clamp so
    LIVE_PAIR_BUFFERS of them fit in PAIR_BUDGET_FRACTION of free memory. If
    the device will not report its L2, KEEP THE CPU DEFAULT and say so -- a
    guess here is a silent performance regression, and the honest fallback is
    the value that was actually measured. The clamp never raises the budget,
    only lowers it, so a small free-memory reading cannot recommend something
    larger than the cache rule.

    THE CACHE JUSTIFICATION FOR THIS IS WRONG, and the rule is kept anyway.
    It was written from MAX_PAIRS_CPU's criterion -- "one working buffer stays
    resident in cache across the fused kernel's passes" -- on the assumption
    that it transferred to a GPU with L2 in place of L3. Swept on GB10 (24 MiB
    L2) the COMPILED splat is monotonically faster with a bigger budget, right
    past 2x the L2, with no optimum: 91.8 / 95.6 / 97.4 / 98.2 / 100% of peak
    at 1.5 / 3.1 / 4.0 / 6.3 / 12.6 M pairs. Fusion keeps the intermediates
    out of memory, so whole-buffer residency is not what binds; per-chunk
    launch overhead is, and fewer chunks wins. Cache residency IS real for the
    eager path, which runs the other way over the same sweep (852 -> 560 Ge/s,
    best at the SMALLEST budget) -- which is also why the CPU default looked
    like a genuine optimum when it was measured.

    The rule survives on the numbers rather than the reasoning: 98.2% of peak
    on GB10, against 97.4% for the old 4M default. The whole 8x sweep spans
    1.089x, so this knob is worth under 1% there and is not where tuning time
    should go. Going bigger is not free either -- padding grows 1.010 -> 1.058
    across the sweep, so at 12.6 M you waste 5.8% of pair work to save 1.8% of
    overhead. Sized to L2 sits near where those cross, which is the defensible
    version of this rule.

    Measured on ONE device. A less bandwidth-starved card may well flatten or
    turn over where GB10 does not; docs/performance.md, "A second machine".
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
