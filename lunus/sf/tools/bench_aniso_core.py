#!/usr/bin/env python
"""
Which anisotropic quadratic form belongs in the splat's inner loop.

    OMP_NUM_THREADS=n python tools/bench_aniso_core.py [--device cuda]

Only the elementwise core, at the shapes splat_density actually chunks to. No
offsets, no scatter, no kernel build -- this isolates the one number the
anisotropic-ADP splat depends on, before any of it is written.

TWO FORMULATIONS of rho(r) = sum_i A_i exp(-r.Lam_i.r):

  EIGENBASIS. The five Lam_i share eigenvectors (see kernel.py), so rotate the
  displacement into the atom's principal axes once, e = V^T d, and each
  Gaussian is sum_k w_ik e_k^2. Fewest flops -- one batched (K,3)@(3,3) per
  atom -- but it materializes (m,K,3) intermediates, and the taper's |d|^2
  comes free since V is orthogonal.

  EXPANDED. d.Lam_i.d = c_off.Lam_i.c_off - 2 c_rem.Lam_i.c_off + const,
  contracted against a per-atom 6-vector. Every intermediate stays (m,K) and
  every matmul is plain 2-D, but it costs that PER GAUSSIAN: 5x the matmul
  flops, plus one more matmul for the taper's r^2.

ON CPU THE EXPANDED FORM WINS, 2.5-3.4x the isotropic core against 3.6-6.0x,
because the core is memory-bound rather than flop-bound: 16 MB of
intermediates per chunk against 96 MB, and a batched matmul cannot fuse with
the surrounding elementwise chain where 2-D ones produce (m,K) directly.

MEASURED ON CUDA. The ordering holds on both platforms, so EXPANDED is what to
build -- but read the compiled rows, not the eager ones:

    shape        mode        iso ms  eigen ms  expand ms  eigen x  expand x
    1000x4000    eager         0.44      1.19       0.72     2.71      1.64
    1000x4000    compiled      0.09      0.27       0.26     2.99      2.83
    2000x2000    eager         0.45      1.19       0.77     2.64      1.71
    2000x2000    compiled      0.09      0.27       0.26     3.11      2.95

    peak device memory   iso 143 MB, expanded 177 MB, eigen 236 MB

COMPILING NEARLY ERASES THE DIFFERENCE between the two forms -- 2.8-3.0x
against 3.0-3.1x, where eager showed 1.6-1.7x against 2.6-2.7x. Inductor fuses
away most of the (m,K,3) traffic the eigenbasis form pays for in eager, which
was the entire basis of the eager result. Expanded still wins, on speed by ~5%
and on peak memory by 59 MB, so the choice stands; the MARGIN does not.

The anisotropic core therefore costs ~2.9x compiled, not the ~1.7x the eager
row suggests, and production compiles.

Compiling is also worth 4.9x on the isotropic core itself (0.44 -> 0.09 ms),
which matters far more than anything above. It needs the CUDA driver header on
the include path -- CPATH=/usr/local/cuda/include -- without which inductor
cannot build its cuda_utils.c shim and silently falls back.

Both cores are checked against each other before timing, so this compares two
spellings of one function rather than a fast implementation of the wrong
thing.

GPU timing discipline, per docs/performance.md: every region synchronizes on
both sides, and a full warm pass runs first so cuBLAS initialization does not
land in the measurement. Treat the result as an upper bound on speed -- an
isolated benchmark keeps caches warm and the allocator stable in ways a real
frame loop does not.
"""

import argparse
import contextlib
import os
import sys
import time

import torch

from lunus.sf.density_torch import taper_factor


@contextlib.contextmanager
def _quiet_stderr():
    """Silence fd 2, not just sys.stderr.

    torch.compile shells out to gcc, so a failure arrives as subprocess output
    on the real file descriptor and a try/except around the call catches the
    exception while the compiler error still lands on the terminal. Redirecting
    the descriptor is the only thing that suppresses it.
    """
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def compile_works(device):
    """Probe torch.compile ONCE on this device, quietly.

    Attempting it per shape produced four identical walls of gcc output in the
    first CUDA run. It fails for one reason -- inductor compiles a cuda_utils.c
    shim that needs cuda.h, the driver API header, which a runtime-only
    container does not ship -- so one probe settles it for the whole run.
    """
    try:
        with _quiet_stderr():
            f = torch.compile(lambda x: x * 2.0 + 1.0, dynamic=False)
            f(torch.zeros(8, device=device))
        return True
    except Exception:
        return False


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def core_iso(c_rem, c_off, c_off_sq, A_e, lam_e, R_e, occ_e, tw):
    r2 = (c_off_sq.unsqueeze(0)
          - 2.0 * (c_rem @ c_off.T)
          + (c_rem * c_rem).sum(-1, keepdim=True)).clamp_min(0.0)
    dens = A_e[:, 0:1] * torch.exp(-lam_e[:, 0:1] * r2)
    for j in range(1, 5):
        dens = dens + A_e[:, j:j + 1] * torch.exp(-lam_e[:, j:j + 1] * r2)
    dens = dens * occ_e[:, None]
    return dens * taper_factor(torch.sqrt(r2), R_e[:, None], tw)


def core_eigen(c_rem, c_off, V_e, w_e, A_e, R_e, occ_e, tw):
    e = c_off.unsqueeze(0) @ V_e - (c_rem.unsqueeze(1) @ V_e)      # (m,K,3)
    e2 = e * e
    dens = A_e[:, 0:1] * torch.exp(-(e2 * w_e[:, 0, :].unsqueeze(1)).sum(-1))
    for j in range(1, 5):
        dens = dens + A_e[:, j:j + 1] * torch.exp(
            -(e2 * w_e[:, j, :].unsqueeze(1)).sum(-1))
    dens = dens * occ_e[:, None]
    r2 = e2.sum(-1)                    # free: V orthogonal, so this is |d|^2
    return dens * taper_factor(torch.sqrt(r2.clamp_min(0.0)), R_e[:, None], tw)


def core_expanded(c_rem, c_off, c_off_sq, P_off, L6_e, g_e, const_e, A_e,
                  R_e, occ_e, tw):
    r2 = (c_off_sq.unsqueeze(0)
          - 2.0 * (c_rem @ c_off.T)
          + (c_rem * c_rem).sum(-1, keepdim=True)).clamp_min(0.0)
    dens = None
    for j in range(5):
        q = (L6_e[:, j, :] @ P_off.T
             - 2.0 * (g_e[:, j, :] @ c_off.T)
             + const_e[:, j:j + 1])
        term = A_e[:, j:j + 1] * torch.exp(-q.clamp_min(0.0))
        dens = term if dens is None else dens + term
    dens = dens * occ_e[:, None]
    return dens * taper_factor(torch.sqrt(r2), R_e[:, None], tw)


def make_inputs(m, K, device, dtype, seed=0):
    """One set of physical parameters, expressed both ways, so the two cores
    compute the same function."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    c_rem = (torch.randn(m, 3, generator=g) * 0.3).to(device, dtype)
    c_off = (torch.randn(K, 3, generator=g) * 2.0).to(device, dtype)
    A_e = torch.rand(m, 5, generator=g).to(device, dtype)
    occ_e = torch.ones(m, device=device, dtype=dtype)
    R_e = torch.full((m,), 3.0, device=device, dtype=dtype)

    V = torch.linalg.qr(torch.randn(m, 3, 3, generator=g))[0].to(device, dtype)
    w = (torch.rand(m, 5, 3, generator=g) + 0.5).to(device, dtype)

    # Lam_i = V diag(w_i) V^T, the same tensor the eigenbasis form applies
    # implicitly, so the expanded form is not a different model.
    Lam = torch.einsum("amk,ajk,ank->ajmn", V, w, V)               # (m,5,3,3)
    L6 = torch.stack([Lam[:, :, 0, 0], Lam[:, :, 1, 1], Lam[:, :, 2, 2],
                      Lam[:, :, 0, 1], Lam[:, :, 0, 2], Lam[:, :, 1, 2]],
                     dim=-1)                                        # (m,5,6)
    g_e = torch.einsum("ajmn,an->ajm", Lam, c_rem)                  # (m,5,3)
    const_e = torch.einsum("am,ajmn,an->aj", c_rem, Lam, c_rem)     # (m,5)
    P_off = torch.stack([c_off[:, 0] ** 2, c_off[:, 1] ** 2, c_off[:, 2] ** 2,
                         2 * c_off[:, 0] * c_off[:, 1],
                         2 * c_off[:, 0] * c_off[:, 2],
                         2 * c_off[:, 1] * c_off[:, 2]], dim=1)     # (K,6)

    c_off_sq = (c_off * c_off).sum(-1)
    lam_e = w[:, :, 0].contiguous()          # isotropic stand-in, same shapes
    return dict(
        iso=(c_rem, c_off, c_off_sq, A_e, lam_e, R_e, occ_e, 0.1),
        eigen=(c_rem, c_off, V, w, A_e, R_e, occ_e, 0.1),
        expanded=(c_rem, c_off, c_off_sq, P_off, L6, g_e, const_e, A_e,
                  R_e, occ_e, 0.1))


def bench(fn, args, device, reps):
    fn(*args)
    sync(device)
    best = float("inf")
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        fn(*args)
        sync(device)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    dev, dtype = args.device, getattr(torch, args.dtype)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    if "OMP_NUM_THREADS" not in os.environ:
        print("WARNING: OMP_NUM_THREADS unset. On a CPU-limited container torch "
              "sizes its pool from the host's cores and throttles; worth 3.6x "
              "on this pipeline. See docs/performance.md.\n", file=sys.stderr)

    print("device %s, %s, best of %d\n" % (dev, args.dtype, args.reps))

    # Equivalence and ACCURACY, before any timing -- speed is not the only
    # axis on which these two differ.
    ref = make_inputs(600, 800, dev, torch.float64)
    a64 = core_eigen(*ref["eigen"])
    b64 = core_expanded(*ref["expanded"])
    rel = float((a64 - b64).abs().max() / a64.abs().max())
    print("the two forms agree to %.2e in float64 -- same function, two "
          "spellings." % rel)
    if rel > 1e-5:
        raise SystemExit("formulations disagree by more than cancellation "
                         "explains; the timing below would be comparing "
                         "different computations")

    # THE EXPANDED FORM CANCELS. It evaluates
    # c_off.Lam.c_off - 2 c_rem.Lam.c_off + const, a difference of terms much
    # larger than the result, so it loses digits where the eigenbasis form --
    # which differences the displacement FIRST and then squares it -- does not.
    # The isotropic core already accepts exactly this for r^2 (see its
    # clamp_min), but Lam has more dynamic range than a scalar lam, so it is
    # worth measuring rather than assuming it carries over.
    ref32 = make_inputs(600, 800, dev, torch.float32)
    err_e = float((core_eigen(*ref32["eigen"]).double() - a64).abs().max()
                  / a64.abs().max())
    err_x = float((core_expanded(*ref32["expanded"]).double() - b64).abs().max()
                  / b64.abs().max())
    print("float32 against float64: eigenbasis %.2e, expanded %.2e  (%.1fx)"
          % (err_e, err_x, err_x / max(err_e, 1e-30)))
    print("  production runs float32, and docs/solvent-design.md puts the")
    print("  pipeline's own reproducibility floor at 4e-5 on Icalc -- so read")
    print("  these against that, not against zero.\n")

    use_compile = not args.no_compile and compile_works(dev)
    if not args.no_compile and not use_compile:
        print("torch.compile does not work here, so the rows below are EAGER.")
        print("  On CUDA this is usually inductor's cuda_utils.c needing cuda.h,")
        print("  the driver API header, absent from a runtime-only image. It is")
        print("  not a GPU or torch fault. Production is eager anyway -- see")
        print("  docs/performance.md -- so the comparison stands; note only that")
        print("  compiling is NOT neutral between these two forms (on CPU it")
        print("  moved expand x from 2.0-2.3 to 2.9-3.8).\n")

    print("%-12s %-9s %9s %9s %9s %8s %8s"
          % ("m x K", "mode", "iso ms", "eigen ms", "expand ms",
             "eigen x", "expand x"))
    for m, K in ((1000, 4000), (2000, 2000), (4000, 1000), (500, 8000)):
        d = make_inputs(m, K, dev, dtype)
        modes = ["eager"] + (["compiled"] if use_compile else [])
        for mode in modes:
            fns = [core_iso, core_eigen, core_expanded]
            if mode == "compiled":
                fns = [torch.compile(f, dynamic=False) for f in fns]
                for f, key in zip(fns, ("iso", "eigen", "expanded")):
                    f(*d[key])
            t = [bench(f, d[k], dev, args.reps)
                 for f, k in zip(fns, ("iso", "eigen", "expanded"))]
            print("%-12s %-9s %9.2f %9.2f %9.2f %8.2f %8.2f"
                  % ("%dx%d" % (m, K), mode, t[0], t[1], t[2],
                     t[1] / t[0], t[2] / t[0]))

    if dev.startswith("cuda"):
        print("\npeak device memory per formulation, m*K = 4e6:")
        for key, fn in (("iso", core_iso), ("eigen", core_eigen),
                        ("expanded", core_expanded)):
            d = make_inputs(2000, 2000, dev, dtype)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            fn(*d[key])
            torch.cuda.synchronize()
            print("  %-10s %6.0f MB" % (key, torch.cuda.max_memory_allocated() / 1e6))


if __name__ == "__main__":
    main()
