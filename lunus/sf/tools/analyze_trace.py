#!/usr/bin/env python
"""
Where the GPU idle time in a torch.profiler Chrome trace actually sits.

    python tools/analyze_trace.py torch_trace_rank0.json

Written for one question: xtraj's frame loop spends ~290 ms per frame doing
54 ms of GPU work, while bench_splat.py does the identical work -- same pair
count, same 90 chunks -- in 62.6 ms wall. So ~240 ms per frame is device idle,
and the shape of that idle says what is causing it:

  * thousands of small gaps, one between each pair of kernels
        -> host dispatch cannot keep up; the fix is fewer, larger ops
           (raise max_pairs_per_batch, or CUDA graphs)
  * a handful of large gaps
        -> the loop is blocking on something. Look at what CPU op each gap
           lands in: splat_density does one K_needed.detach().to("cpu") per
           element per frame, which is a synchronizing device->host copy
  * one long gap at the start or end of the step
        -> the cost is outside the splat entirely

The distinction is not visible in the profiler's summary table, which reports
totals per operator with no notion of when the device was doing nothing.

Reads the trace with the standard library only -- no torch, no pandas -- so it
runs wherever the trace was produced.
"""

import argparse
import collections
import json
import sys


# Chrome-trace categories torch uses for real device work. Kernels are the
# ones that matter; memcpy/memset are included because a device that is
# copying is not idle.
#
# gpu_user_annotation is deliberately NOT here. torch projects each
# ProfilerStep onto the GPU track as an annotation spanning the whole step,
# so counting it as device work reports the device busy ~100% of the time
# whatever it is doing -- and hides exactly the idle this tool exists to find.
# The same artifact shows up in profiler.key_averages() as ProfilerStep* with
# a CUDA share over 100%.
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"cpu_op", "user_annotation"}
# CUDA API calls (cudaMalloc, cudaFree, cudaLaunchKernel,
# cudaStreamSynchronize, ...). When a gap sits inside an innocent-looking
# elementwise op, this is the layer that says what the host was actually
# blocked in: an allocation going to the driver, a synchronizing copy, or a
# launch waiting on a full queue.
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}


def load_events(path):
    with open(path) as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    gpu, cpu, steps, rt = [], [], [], []
    seen = collections.Counter()          # every category, counted or not
    for e in events:
        if e.get("ph") != "X" or "ts" not in e or "dur" not in e:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        seen[cat] += 1
        rec = (float(e["ts"]), float(e["dur"]), name)
        if cat in GPU_CATS:
            gpu.append(rec)
        elif cat in RUNTIME_CATS:
            rt.append(rec)
        elif cat in CPU_CATS:
            cpu.append(rec)
            if name.startswith("ProfilerStep"):
                steps.append(rec)
    gpu.sort()
    cpu.sort()
    rt.sort()
    steps.sort()
    return gpu, cpu, steps, rt, seen


def merge_busy(intervals):
    """Union of possibly-overlapping intervals: concurrent streams must not
    double-count as busy time."""
    merged = []
    for ts, dur, _ in intervals:
        if merged and ts <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ts + dur)
        else:
            merged.append([ts, ts + dur])
    return merged


def enclosing_cpu_op(cpu, t):
    """Innermost CPU op whose span contains t -- what the host was doing while
    the device sat idle. cpu is sorted by start time."""
    best = None
    for ts, dur, name in cpu:
        if ts > t:
            break
        if ts + dur >= t and (best is None or dur <= best[1]):
            best = (ts, dur, name)
    return best[2] if best else "(none)"


def check_cfs_throttling(gaps, period_us=100000.0, min_gap_us=10000.0):
    """Do the large gaps all END at the same phase within a fixed period?

    The Linux CFS bandwidth controller (cgroup cpu.max, what a Kubernetes CPU
    limit becomes) accounts in fixed periods, 100 ms by default. A container
    that exhausts its quota is frozen until the period boundary refills it, so
    every stall ends at the SAME phase modulo the period, whatever the program
    was doing. Nothing in a numerical kernel produces that signature.

    Returns (phase_ms, spread_ms, n) or None. Uses circular statistics, since
    phases near 0 and near the period are adjacent.
    """
    import math

    ends = [(at + g) % period_us for g, at in gaps if g >= min_gap_us]
    if len(ends) < 3:
        return None
    angles = [2 * math.pi * e / period_us for e in ends]
    mx = sum(math.cos(a) for a in angles) / len(angles)
    my = sum(math.sin(a) for a in angles) / len(angles)
    if math.hypot(mx, my) < 0.9:            # not concentrated: no signature
        return None
    mean = math.atan2(my, mx) % (2 * math.pi)
    spread = max(
        abs((a - mean + math.pi) % (2 * math.pi) - math.pi) for a in angles)
    return (mean * period_us / (2 * math.pi) / 1e3,
            spread * period_us / (2 * math.pi) / 1e3,
            len(ends))


def longest_overlapping(rt, lo, hi):
    """The CUDA API call that covers most of [lo, hi] -- what the host was
    blocked in while the device had nothing to do."""
    best, best_ov = None, 0.0
    for ts, dur, name in rt:
        if ts >= hi:
            break
        ov = min(ts + dur, hi) - max(ts, lo)
        if ov > best_ov:
            best, best_ov = (name, dur), ov
    if best is None:
        return "(none)"
    return "{0} ({1:.1f} ms, {2:.0f}% of gap)".format(
        best[0], best[1] / 1e3, 100.0 * best_ov / (hi - lo))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--top", type=int, default=15, help="largest gaps to list")
    p.add_argument("--attribute", type=int, default=10,
                   help="how many of those to attribute to a CPU op (this is "
                        "the slow part -- it scans the CPU op list per gap)")
    args = p.parse_args()

    gpu, cpu, steps, rt, seen = load_events(args.trace)
    # Which categories were counted as device work, so a miscategorisation is
    # visible in the output rather than silently changing the conclusion.
    print("categories in trace (* = counted as GPU busy)")
    for cat, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print("  {0} {1:<24} {2:,}".format(
            "*" if cat in GPU_CATS else " ", cat or "(none)", n))
    print("")
    if not gpu:
        print("No GPU events in this trace. Was it recorded with "
              "ProfilerActivity.CUDA, on a CUDA device?")
        return 1

    busy = merge_busy(gpu)
    span_start = min(busy[0][0], steps[0][0] if steps else busy[0][0])
    span_end = max(busy[-1][1], (steps[-1][0] + steps[-1][1]) if steps else busy[-1][1])
    span = span_end - span_start
    busy_us = sum(b - a for a, b in busy)

    n_steps = len(steps) or 1
    print("trace           {0}".format(args.trace))
    print("frames (steps)  {0}".format(len(steps)))
    print("span            {0:10.1f} ms   ({1:.1f} ms/frame)".format(
        span / 1e3, span / 1e3 / n_steps))
    print("GPU busy        {0:10.1f} ms   ({1:.1f} ms/frame, {2:.1f}% of span)".format(
        busy_us / 1e3, busy_us / 1e3 / n_steps, 100.0 * busy_us / span))
    print("GPU idle        {0:10.1f} ms   ({1:.1f} ms/frame, {2:.1f}% of span)".format(
        (span - busy_us) / 1e3, (span - busy_us) / 1e3 / n_steps,
        100.0 * (span - busy_us) / span))
    print("GPU events      {0:,} kernels/copies".format(len(gpu)))

    # Gaps between consecutive busy intervals.
    gaps = []
    for i in range(1, len(busy)):
        g = busy[i][0] - busy[i - 1][1]
        if g > 0:
            gaps.append((g, busy[i - 1][1]))
    # The idle before the first kernel and after the last one counts too --
    # without them the gap total would not add up to the idle total above.
    lead = busy[0][0] - span_start
    if lead > 0:
        gaps.append((lead, span_start))
    tail = span_end - busy[-1][1]
    if tail > 0:
        gaps.append((tail, busy[-1][1]))
    gaps.sort(reverse=True)

    print("\ngap size distribution ({0:,} gaps totalling {1:.1f} ms)".format(
        len(gaps), sum(g for g, _ in gaps) / 1e3))
    print("  {0:>14}  {1:>8}  {2:>10}  {3:>7}".format(
        "size", "count", "total ms", "share"))
    total_gap = sum(g for g, _ in gaps) or 1.0
    buckets = [(0, 10), (10, 100), (100, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, 1e9)]
    labels = ["<10us", "10-100us", "0.1-1ms", "1-10ms", "10-100ms", ">100ms"]
    for (lo, hi), lab in zip(buckets, labels):
        sel = [g for g, _ in gaps if lo <= g < hi]
        if not sel:
            continue
        print("  {0:>14}  {1:>8,}  {2:>10.1f}  {3:>6.1f}%".format(
            lab, len(sel), sum(sel) / 1e3, 100.0 * sum(sel) / total_gap))

    # The shape question, stated as a number rather than left to the eye.
    small = sum(g for g, _ in gaps if g < 1e3)
    print("\ngaps under 1 ms account for {0:.1f}% of all idle time".format(
        100.0 * small / total_gap))
    print("  high -> host dispatch bound (fewer, larger ops is the fix)")
    print("  low  -> a few long blocks; see what they land in, below")

    cfs = check_cfs_throttling(gaps)
    if cfs is not None:
        phase, spread, n = cfs
        print("\n*** CPU THROTTLING: all {0} gaps over 10 ms end at {1:.1f} ms "
              "+/- {2:.2f} ms into the 100 ms cgroup period.".format(
                n, phase, spread))
        print("    A stall that always ends on the period boundary is the CFS")
        print("    bandwidth controller refilling the quota, not anything in")
        print("    this program. Confirm with:")
        print("      cat /sys/fs/cgroup/cpu.max      # quota period, in us")
        print("      cat /sys/fs/cgroup/cpu.stat     # nr_throttled, "
              "throttled_usec -- diff across a run")
        print("    Every wall-clock number measured under this is a "
              "measurement of the quota, not of the code.")

    print("\n{0} largest gaps".format(min(args.top, len(gaps))))
    print("  {0:>10}  {1:>12}  {2:<22}  {3}".format(
        "gap ms", "at ms", "host was in", "longest CUDA API call in the gap"))
    for i, (g, at) in enumerate(gaps[:args.top]):
        if i < args.attribute:
            where = enclosing_cpu_op(cpu, at + g / 2)
            api = longest_overlapping(rt, at, at + g)
        else:
            where, api = "(not attributed)", ""
        print("  {0:>10.2f}  {1:>12.1f}  {2:<22}  {3}".format(
            g / 1e3, (at - span_start) / 1e3, where, api))

    # Which CPU ops the idle time lands in overall, not just for the largest.
    by_op = collections.Counter()
    for g, at in gaps[:200]:
        by_op[enclosing_cpu_op(cpu, at + g / 2)] += g
    print("\nidle time by enclosing CPU op (200 largest gaps)")
    for name, tot in by_op.most_common(10):
        print("  {0:>10.2f} ms  {1}".format(tot / 1e3, name))

    # The CUDA API calls the host spent its time in, whatever the surrounding
    # torch op was named. A long total against cudaMalloc/cudaFree points at
    # the allocator; against cudaStreamSynchronize or a blocking memcpy, at a
    # sync; against cudaLaunchKernel, at a full launch queue.
    if rt:
        agg = collections.Counter()
        cnt = collections.Counter()
        for _, dur, name in rt:
            agg[name] += dur
            cnt[name] += 1
        print("\nCUDA API calls by total host time")
        print("  {0:>10}  {1:>9}  {2:>10}  {3}".format(
            "total ms", "calls", "avg us", "call"))
        for name, tot in agg.most_common(10):
            print("  {0:>10.1f}  {1:>9,}  {2:>10.1f}  {3}".format(
                tot / 1e3, cnt[name], tot / cnt[name], name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
