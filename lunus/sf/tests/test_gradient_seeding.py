"""
Validates the core mathematical claim behind the MPI training-step
design: that gradients computed via

    local forward -> Allreduce local sums -> tiny backward on the
    reduced aggregate -> seed each rank's local backward with that
    result

are IDENTICAL to gradients from one combined backward pass over
everything at once. Uses a real-valued toy analog of the actual
mean_F/mean_I -> bragg/diffuse -> loss structure (avoiding complex-
number Wirtinger-calculus conventions here, since that part is handled
natively and correctly by PyTorch itself in the real implementation --
this test is about the reduce/seed mechanism, which is agnostic to
real vs complex).
"""
import numpy as np


def g(x):
    return np.sin(x) + 0.1 * x**3


def dg_dx(x):
    return np.cos(x) + 0.3 * x**2


def h(x):
    return g(x) ** 2  # stands in for |F_c|^2


def dh_dx(x):
    return 2 * g(x) * dg_dx(x)


def loss_fn(mean, mean_I, target_diffuse, target_bragg):
    bragg = mean ** 2
    diffuse = mean_I - bragg
    return (diffuse - target_diffuse) ** 2 + (bragg - target_bragg) ** 2


def test_reduce_then_seed_matches_combined_backward():
    """The MPI training step's reduce-then-seed scheme must give EXACTLY the
    gradients of one combined backward pass over all configurations.

    This is the claim the whole design rests on: because each configuration's
    atoms belong to one rank alone, no per-atom gradient ever needs to be
    communicated -- one small Allreduce of the aggregate sums is enough, and
    each rank then finishes its own backward pass locally.

    Measured: combined vs split agree to 5.6e-17 (machine precision -- they
    are the same arithmetic), and both agree with a finite-difference ground
    truth to 1.2e-10 (limited by the finite difference, not by either method).

    The rank split is deliberately uneven (3/2/3) so that a bug assuming equal
    per-rank counts would show up.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, size=8)  # 8 "configs" total
    ranks = [x[0:3], x[3:5], x[5:8]]  # 3 ranks, uneven split on purpose
    N = len(x)
    target_diffuse, target_bragg = 0.3, 0.5

    # --- combined: one big graph, direct analytic chain rule ---
    mean = g(x).mean()
    mean_I = h(x).mean()
    bragg = mean ** 2
    diffuse = mean_I - bragg
    dloss_ddiffuse = 2 * (diffuse - target_diffuse)
    dloss_dbragg_direct = 2 * (bragg - target_bragg)
    dbragg_dmean = 2 * mean
    combined_grad = np.array([
        dloss_ddiffuse * (dh_dx(xi) / N - dbragg_dmean * dg_dx(xi) / N)
        + dloss_dbragg_direct * dbragg_dmean * dg_dx(xi) / N
        for xi in x
    ])

    # --- split: local sums -> allreduce -> tiny backward -> seed local backward ---
    local_sums_g = [g(r).sum() for r in ranks]    # per-rank local_sum_F analog
    local_sums_h = [h(r).sum() for r in ranks]     # per-rank local_sum_I analog
    global_sum_g = sum(local_sums_g)              # "Allreduce"
    global_sum_h = sum(local_sums_h)

    mean2 = global_sum_g / N
    mean_I2 = global_sum_h / N
    bragg2 = mean2 ** 2
    diffuse2 = mean_I2 - bragg2
    dloss_dmean = 2 * (diffuse2 - target_diffuse) * (-2 * mean2) + 2 * (bragg2 - target_bragg) * (2 * mean2)
    dloss_dmean_I = 2 * (diffuse2 - target_diffuse) * 1.0
    dloss_dglobal_sum_g = dloss_dmean / N       # this is what .backward() on the tiny graph gives
    dloss_dglobal_sum_h = dloss_dmean_I / N     # same, for the other leaf

    # every rank independently computes the SAME dloss_dglobal_sum_g/h (no
    # broadcast needed -- they all have the same Allreduce'd data), then
    # seeds its own local backward using ONLY its own local atoms
    split_grad = np.array([dloss_dglobal_sum_g * dg_dx(xi) + dloss_dglobal_sum_h * dh_dx(xi) for xi in x])

    # --- finite-difference ground truth, on the actual masked/discretized-free loss ---
    def full_loss(xvec):
        m = g(xvec).mean()
        mI = h(xvec).mean()
        return loss_fn(m, mI, target_diffuse, target_bragg)

    eps = 1e-6
    fd_grad = np.zeros(N)
    for i in range(N):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        fd_grad[i] = (full_loss(xp) - full_loss(xm)) / (2 * eps)

    print(f"{'i':>2}  {'combined':>12}  {'split(reduce+seed)':>20}  {'finite-diff':>12}")
    for i in range(N):
        print(f"{i:>2}  {combined_grad[i]:12.6f}  {split_grad[i]:20.6f}  {fd_grad[i]:12.6f}")

    print(f"\nmax |combined - split|        = {np.max(np.abs(combined_grad - split_grad)):.2e}")
    print(f"max |combined - finite-diff|  = {np.max(np.abs(combined_grad - fd_grad)):.2e}")
    print(f"max |split - finite-diff|     = {np.max(np.abs(split_grad - fd_grad)):.2e}")

    # Combined vs split is the actual claim, and it is an identity rather than
    # an approximation -- the two evaluate the same chain rule in a different
    # order, so only float associativity separates them.
    np.testing.assert_allclose(
        split_grad, combined_grad, rtol=1e-12, atol=1e-12,
        err_msg="reduce-then-seed does not reproduce the combined backward pass",
    )

    # Both against an independent finite-difference ground truth, which bounds
    # the possibility that combined_grad's hand-derived chain rule is itself
    # wrong in a way split_grad happens to share.
    fd_tol = 1e-7 * max(np.max(np.abs(fd_grad)), 1.0)
    np.testing.assert_allclose(
        combined_grad, fd_grad, rtol=1e-6, atol=fd_tol,
        err_msg="analytic combined gradient disagrees with finite differences",
    )
    np.testing.assert_allclose(
        split_grad, fd_grad, rtol=1e-6, atol=fd_tol,
        err_msg="reduce-then-seed gradient disagrees with finite differences",
    )
