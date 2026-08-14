"""
Differentiable PyTorch atomic density splat, direct translation of the
validated NumPy reference (validate_numpy.py / validate_multiatom.py),
vectorized and grouped by element for large atom counts.

Memory-bounded: the naive fully-vectorized approach materializes one
(atoms_of_this_element, local_box_points, 3) tensor per element group,
which scales as O(n_atoms * box_volume) and blows up memory at large
scale (~90GB for 200k atoms with a 17^3 local box). Fixed by:

  1. Atoms are processed in bounded-size chunks, budgeted by ATOM-VOXEL
     PAIRS (max_pairs_per_batch) since that is what actually sets
     intermediate size, with max_atoms_per_batch as a secondary cap. Peak
     memory is a fixed budget regardless of total atom count.
  2. Grid indices use int32 instead of int64, with a single conversion
     immediately before index_add_ (which requires int64).
  3. The local box is restricted to the actual cutoff sphere rather than
     its full bounding cube, and the candidate list is SORTED by distance
     from the box center so each atom uses only the prefix its own radius
     needs -- an atom no longer pays for its element's highest-B, most
     spread out one.
  4. Per-element offsets are precomputed ONCE (via
     precompute_element_offsets, typically called from
     kernel_torch.build_element_kernels_torch) and reused across every
     frame/configuration, rather than rebuilt on every call -- they
     only depend on the unit cell and element radii, neither of which
     changes frame to frame.

PERFORMANCE. The per-voxel work is memory-bound, not compute-bound: every
elementary torch operation is its own kernel pass over an
(atoms, voxels) array, so what matters is the number of passes, not the
number of flops. Three consequences shape this module, all measured on a
135,834-atom P4_3 case against gemmi's C++ DensityCalculatorX (1.42 s):

  * The distance is computed separably, so r2 comes from one matrix
    multiply plus a precomputed |c_off|^2 rather than four
    (atoms, voxels, 3) intermediates. 12.2 s -> 3.5 s.
  * The two hot blocks (_density_core, _scatter_core) are factored into
    their own functions so torch.compile can FUSE them, collapsing ~20
    passes into a couple. 2.7 s -> 0.96 s, i.e. faster than gemmi, which
    keeps each voxel in registers in a single pass and is the thing
    fusion is imitating.
  * Cheaper arithmetic is not automatically faster: replacing the
    wrap-around modulo with an arithmetically equivalent compare-and-shift
    made things SLOWER eagerly, because two torch.where's cost about six
    passes against one for a remainder.

Periodic wraparound uses torch.remainder explicitly (not the `%`
operator), whose non-negative-result-for-positive-divisor behavior is
documented and version-independent, to avoid any ambiguity for negative
grid-index candidates near a fractional-coordinate boundary.

GRADIENT CORRECTNESS: density is tapered smoothly to zero over the last
`taper_width` of the cutoff radius (a C^1 raised-cosine window) instead
of being hard-masked at the radius. A hard `r2 <= radius^2` mask exactly
matches gemmi's forward values, but is a genuine correctness problem for
gradients: torch.where's backward rule holds the mask fixed and never
accounts for grid points crossing the boundary as an atom moves, which a
direct numerical experiment showed makes the true (finite-difference)
gradient near a hard cutoff essentially unstable/grid-dependent -- ~100%
relative error against the reported autograd gradient, with the true
value's sign and magnitude varying wildly between trials. The smooth
taper removes the discrete in/out state entirely, so autograd correctly
differentiates it (same experiment: ~0.01-0.15% agreement). This trades
a small, already-quantified forward-value difference from gemmi's hard
cutoff for reliable gradients, which is the right trade given this
engine's purpose. taper_width doesn't need to track grid spacing at all
(gradient correctness held to <0.06% error even at taper_width an order
of magnitude narrower than typical grid spacing, in direct testing) --
what it needs to stay small against is the cutoff radius, since removing
density from a specific outer shell reshapes the effective form factor
at whatever resolution corresponds to that shell's width, not just at
the true resolution limit. build_element_kernels_torch defaults to 0.1 A
(~0.5% density loss, measured) if not given explicitly.

Every operation here is a standard differentiable PyTorch primitive
(arithmetic, exp, index_add_), so autograd differentiates it exactly as
validated in test_torch.py's autograd-vs-analytic gradient check.
"""

import math
import warnings

import torch


def taper_factor(r: torch.Tensor, R, w) -> torch.Tensor:
    """
    C^1 raised-cosine taper: 1.0 for r <= R-w, smoothly to 0.0 at r = R,
    0.0 for r >= R. Both the taper's value AND its first derivative
    match at both transition endpoints (r=R-w and r=R), which is what
    makes it safe for autograd -- no discrete in/out state anywhere.

    Written as a clamp rather than two torch.where branches: with
    x = clamp((R - r)/w, 0, 1) the three regimes collapse into
    0.5 - 0.5*cos(pi*x), since cos(pi*(1-x)) = -cos(pi*x). Algebraically
    identical to the branched form (x=1 -> 1, x=0 -> 0), but branch-free
    and without the two full-size boolean masks the where-version built.
    """
    x = ((R - r) / w).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * x)


class ElementOffsets:
    """
    The precomputed local-box candidate list for one element, SORTED by
    distance from the box center. Sorting is what makes the per-atom radius
    usable: the candidates an atom of radius R needs are exactly a PREFIX of
    this list (every offset with |c_off| <= R + margin), so a group of
    similar-radius atoms can use a tight prefix instead of the whole list
    sized for the element's largest-B, most spread out atom.

    Also caches the Cartesian offsets and their squared norms, which turn the
    per-atom-per-voxel distance into a single matrix multiply -- see
    splat_density.

    Fields:
      offsets    (K,3) int32   grid-index offsets, sorted by |c_off|
      c_off      (K,3) dtype   Cartesian offsets, orth_matrix @ (offsets/N)
      c_off_sq   (K,)  dtype   |c_off|^2
      c_off_norm (K,)  dtype   |c_off|, sorted ascending (for searchsorted)
      margin     float         half a voxel diagonal; see below
    """

    __slots__ = ("offsets", "c_off", "c_off_sq", "c_off_norm", "margin")

    def __init__(self, offsets, c_off, c_off_sq, c_off_norm, margin):
        self.offsets = offsets
        self.c_off = c_off
        self.c_off_sq = c_off_sq
        self.c_off_norm = c_off_norm
        self.margin = margin

    def __len__(self):
        return self.offsets.shape[0]

    def prefix_for(self, radius_ang):
        """Number of leading offsets an atom of this radius can possibly need."""
        return int(torch.searchsorted(
            self.c_off_norm, torch.as_tensor(radius_ang + self.margin,
                                             dtype=self.c_off_norm.dtype,
                                             device=self.c_off_norm.device),
            right=True,
        ))


def _sphere_offsets(du, dv, dw, radius_ang, safety_margin, orth_matrix, grid_dims, device):
    """
    Grid-index offsets whose Cartesian distance from the box center is within
    radius_ang + safety_margin, returned SORTED by that distance along with
    the Cartesian offsets themselves. Called once per element at setup time.

    The center-distance |c_off| is not the atom-to-voxel distance: it ignores
    the atom's sub-voxel remainder within its own cell. safety_margin covers
    exactly that, and HALF A VOXEL DIAGONAL is the tight bound, not a heuristic
    -- since base = round(pos * N), the atom sits within half a voxel of the
    box center on every axis, so |c_rem| <= half the voxel diagonal. A dropped
    offset therefore has true distance >= |c_off| - |c_rem| > R, where the
    taper is exactly zero, so nothing that could contribute is ever excluded.
    (An earlier version used a full voxel diagonal, which is safe but keeps
    ~1.2-1.4x more candidate points than can ever matter.)
    """
    ou = torch.arange(-du, du + 1, device=device)
    ov = torch.arange(-dv, dv + 1, device=device)
    ow = torch.arange(-dw, dw + 1, device=device)
    OU, OV, OW = torch.meshgrid(ou, ov, ow, indexing="ij")
    offsets = torch.stack([OU, OV, OW], dim=-1).reshape(-1, 3)  # (K_cube, 3)

    frac_offsets = offsets.to(orth_matrix.dtype) / grid_dims
    cart_offsets = torch.einsum("ij,kj->ki", orth_matrix, frac_offsets)
    dist = cart_offsets.norm(dim=-1)

    keep = dist <= (radius_ang + safety_margin)
    offsets, cart_offsets, dist = offsets[keep], cart_offsets[keep], dist[keep]

    order = torch.argsort(dist)
    offsets, cart_offsets, dist = offsets[order], cart_offsets[order], dist[order]
    return offsets.to(torch.int32).contiguous(), cart_offsets.contiguous(), dist.contiguous()


def precompute_element_offsets(elem_radius_vox, elem_radius_ang, orth_matrix, grid_shape, device):
    """
    Call this ONCE per structure/unit-cell (e.g. in setup code, alongside
    building elem_A/elem_lam), not per frame. Returns a list of ElementOffsets,
    one per element, for splat_density's elem_offsets arg.
    """
    Nu, Nv, Nw = grid_shape
    grid_dims = torch.tensor([Nu, Nv, Nw], device=device, dtype=orth_matrix.dtype)
    spacing_approx = torch.stack([
        orth_matrix[:, 0].norm() / Nu,
        orth_matrix[:, 1].norm() / Nv,
        orth_matrix[:, 2].norm() / Nw,
    ])
    safety_margin = 0.5 * spacing_approx.norm()

    # splat_density's separable distance skips the minimum-image wrap, which is
    # only valid while a local box stays well inside the cell. Check it here
    # rather than silently returning wrong distances for a pathologically small
    # cell (or an enormous B-factor).
    half_cell = 0.5 * min(float(orth_matrix[:, k].norm()) for k in range(3))
    max_reach = float(max(float(r) for r in elem_radius_ang)) + float(safety_margin)
    if max_reach >= half_cell:
        raise ValueError(
            f"local box reach ({max_reach:.2f} A) is at least half the shortest cell "
            f"axis ({half_cell:.2f} A); the minimum-image assumption in splat_density "
            "no longer holds. Use a larger cell, a tighter density cutoff, or restore "
            "an explicit minimum-image wrap in the distance calculation."
        )

    n_elements = elem_radius_vox.shape[0]
    offsets_list = []
    for e in range(n_elements):
        du, dv, dw = elem_radius_vox[e].tolist()
        offsets, c_off, c_norm = _sphere_offsets(
            du, dv, dw, elem_radius_ang[e], safety_margin,
            orth_matrix, grid_dims, device,
        )
        offsets_list.append(ElementOffsets(
            offsets, c_off, (c_off * c_off).sum(-1).contiguous(),
            c_norm, float(safety_margin),
        ))
    return offsets_list


def _density_core(c_rem, c_off, c_off_sq, A_e, lam_e, R_e, occ_e, taper_width):
    """
    The whole per-chunk elementwise body: separable r2 -> 5 Gaussians -> taper,
    returning the (m,K) density contributions.

    Split into its own function purely so torch.compile can fuse it. Every step
    is memory-bound and individually trivial, so run eagerly it costs about
    fifteen separate passes over an (m,K) array; fused it is closer to two,
    which measures as a 3.2-3.6x speedup on this block (float32 agreement to
    ~1e-7, i.e. rounding).
    """
    r2 = (
        c_off_sq.unsqueeze(0)
        - 2.0 * (c_rem @ c_off.T)
        + (c_rem * c_rem).sum(-1, keepdim=True)
    ).clamp_min(0.0)                    # clamp guards float round-off at r ~ 0

    # accumulate the 5 Gaussians into (m,K) rather than building an (m,K,5)
    # intermediate and reducing it
    dens = A_e[:, 0:1] * torch.exp(-lam_e[:, 0:1] * r2)
    for j in range(1, 5):
        dens = dens + A_e[:, j:j + 1] * torch.exp(-lam_e[:, j:j + 1] * r2)
    dens = dens * occ_e[:, None]
    return dens * taper_factor(torch.sqrt(r2), R_e[:, None], taper_width)


def _scatter_core(grid_flat, base_i, offsets, dens, Nu, Nv, Nw):
    """
    Wrap each candidate voxel into the cell, flatten it, and accumulate the
    density. Also split out for torch.compile: fused, the flat index never has
    to be materialized as a full (m,K) int64 array and handed to a separate
    index_add_ pass, which is worth another ~2.2-2.5x on this block.

    The whole index is computed in int32 and converted once at the end
    (index_add_ requires int64). The largest value is Nu*Nv*Nw - 1, which
    splat_density's assert keeps inside int32, and Horner's form folds the
    strides into the accumulation.

    Wrapping uses torch.remainder rather than a compare-and-shift, even though
    |offset| is small enough that the latter is arithmetically equivalent: run
    eagerly, two torch.where's cost about six passes over an (m,K) array
    against one for a remainder, and measurably lose despite integer division
    being the more expensive instruction. Cost is memory passes, not flops.
    """
    flat = None
    for axis, Nax in ((0, Nu), (1, Nv), (2, Nw)):
        x = torch.remainder(base_i[:, axis].unsqueeze(1) + offsets[:, axis].unsqueeze(0), Nax)
        flat = x if flat is None else flat * Nax + x
    grid_flat.index_add_(0, flat.to(torch.int64).reshape(-1), dens.reshape(-1))
    return grid_flat


_COMPILED = {}              # built lazily, once per process
_COMPILE_PROVEN = set()     # functions whose compiled form has actually run
_COMPILE_FAILED = False


def _resolve_compiled(fn, enable):
    """
    Returns fn, torch.compile'd when asked for and available.

    Compilation is lazy and cached process-wide, and any failure (older torch,
    no compiler backend, unsupported platform) falls back to the eager version
    permanently rather than raising -- the eager path is correct, just slower.
    dynamic=True so the varying per-chunk (m, K) shapes do not trigger a
    recompile for every chunk.

    The fallback has to wrap the first CALL, not just torch.compile() itself:
    torch.compile returns immediately and does the real work lazily, so a
    broken backend raises from the call site. This is not hypothetical -- MPS
    on torch 2.13 raises InductorError ("'c10/metal/reduction_utils.h' file not
    found") from the first invocation, which without this guard takes down any
    torch_device=mps run, since compilation is on by default.

    Falling back mid-call is safe even for _scatter_core, which mutates
    grid_flat: inductor compiles the whole graph before executing any of it, so
    a compilation failure leaves the grid untouched and the eager retry cannot
    double-count.
    """
    global _COMPILE_FAILED
    if not enable or _COMPILE_FAILED:
        return fn
    if fn not in _COMPILED:
        try:
            # Donated buffers let the compiled backward reuse forward
            # intermediates, but that requires every backward() to be the
            # graph's last -- it raises on retain_graph=True or
            # create_graph=True. Gradient checks and any higher-order use do
            # exactly that, so turn it off: it costs some backward-pass memory
            # and buys back the ability to call backward however you like.
            import torch._functorch.config as _ft_config
            _ft_config.donated_buffer = False
        except Exception:
            pass                       # older torch: option simply not present
        try:
            _COMPILED[fn] = torch.compile(fn, dynamic=True)
        except Exception:
            _COMPILE_FAILED = True
            return fn

    compiled = _COMPILED[fn]
    if fn in _COMPILE_PROVEN:
        return compiled

    def guarded(*args, **kwargs):
        global _COMPILE_FAILED
        # re-checked on every call, not just when the wrapper was built: a
        # single splat_density invocation calls this once per CHUNK, and the
        # other core's failure (or an earlier chunk's) must stop the retries
        # immediately rather than re-attempting a known-broken compile.
        if _COMPILE_FAILED:
            return fn(*args, **kwargs)
        try:
            out = compiled(*args, **kwargs)
        except Exception as exc:
            _COMPILE_FAILED = True
            warnings.warn(
                f"torch.compile failed for {fn.__name__} "
                f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]}); "
                "falling back to the eager implementation for the rest of this "
                "process. Results are unaffected, but the splat will be slower "
                "-- pass compile_core=False (xtraj.py: torch_compile=False) to "
                "skip the attempt entirely.",
                RuntimeWarning, stacklevel=2,
            )
            return fn(*args, **kwargs)
        _COMPILE_PROVEN.add(fn)
        return out

    return guarded


def warmup_compile(elem_offsets, grid_shape, orth_matrix, taper_width,
                   max_atoms_per_batch=50_000, max_pairs_per_batch=4_000_000):
    """
    Runs a minimal splat purely to populate torch.compile's on-disk cache.

    UNDER MPI THIS MUST BE DONE BY ONE RANK BEFORE THE OTHERS START. Every rank
    otherwise compiles the same kernels itself, and N concurrent compilations
    saturate the machine, so the cost is paid roughly once per rank in sequence
    rather than once in total. Measured on 10 ranks with a cold cache, frame
    times came out staggered in ~1 s steps -- 10.0, 10.1, 10.3, 10.8, 11.8,
    12.8, 13.9, 15.0, 16.0, 17.0 s, total 17.5 s; warming on rank 0 first
    brings the same cold-cache run to 7.4 s with no stagger.

    It is NOT contention on inductor's cache-directory file lock: giving every
    rank its own TORCHINDUCTOR_CACHE_DIR reproduces the stagger exactly (17.3
    s). It is the compile work itself.

    The penalty only appears on a COLD cache, which is what makes it look
    intermittent -- re-running the same job is fast because the on-disk cache
    survives.

    Uses the REAL grid shape, dtype and offset tables, since the compiled
    artifact is keyed on those; the actual A/lam values are irrelevant to
    compilation, so they are synthesized. Each element's radius is set to its
    full candidate reach, so the shapes compiled for are the largest the real
    run will use.

    Called several times, varying two things that torch specializes on and that
    a naive warmup gets wrong (each was found by reading the guard failure from
    TORCH_LOGS=recompiles, not by guesswork):

      * the ATOM COUNT per element must vary and exceed 1. Size-1 dimensions
        are special-cased, so a warmup with one atom per element compiles a
        graph the real run cannot reuse.
      * the OFFSET PREFIX must be a partial slice for some calls and the whole
        tensor for others. The guard is literally
        `c_off_sq._base.size()[0] == c_off_sq.size()[0]`: a full-length prefix
        passes `eo.c_off_sq[:K]` as a complete view of its base, a shorter one
        does not, and the two compile separately.

    Getting either wrong is silent -- the warmup succeeds, the cache looks
    populated, and every rank recompiles anyway on its first real frame.
    """
    n_elements = len(elem_offsets)
    if n_elements == 0:
        return
    device, dtype = orth_matrix.device, orth_matrix.dtype

    reach = torch.tensor(
        [float(eo.c_off_norm[-1]) if len(eo) else 0.0 for eo in elem_offsets],
        device=device, dtype=dtype,
    )

    for per_element, reach_fraction in ((2, 0.6), (5, 0.85), (3, 1.0)):
        n = n_elements * per_element
        frac = torch.rand((n, 3), device=device, dtype=dtype)
        element_idx = torch.arange(n_elements, device=device).repeat_interleave(per_element)
        occ = torch.ones(n, device=device, dtype=dtype)
        A = torch.ones((n, 5), device=device, dtype=dtype)
        lam = torch.ones((n, 5), device=device, dtype=dtype)
        radius = reach[element_idx] * reach_fraction
        with torch.no_grad():
            splat_density(
                frac, element_idx, occ, A, lam, elem_offsets, radius,
                grid_shape, orth_matrix, taper_width,
                max_atoms_per_batch=max_atoms_per_batch,
                max_pairs_per_batch=max_pairs_per_batch,
                compile_core=True,
            )


def _chunk_length(K_sorted, start, n_left, max_atoms, max_pairs):
    """
    Largest n in [1, min(max_atoms, n_left)] with n * K_sorted[start+n-1] <= max_pairs.
    K_sorted is non-decreasing (atoms were sorted by radius), so n * K[start+n-1]
    is non-decreasing in n and a binary search is exact.
    """
    hi = min(max_atoms, n_left)
    if hi <= 1 or int(K_sorted[start]) * 1 >= max_pairs:
        return 1
    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * int(K_sorted[start + mid - 1]) <= max_pairs:
            lo = mid
        else:
            hi = mid - 1
    return lo


def splat_density(
    frac_coords: torch.Tensor,       # (n_atoms, 3), requires_grad=True, in [0,1)
    element_idx: torch.Tensor,       # (n_atoms,) long
    occ: torch.Tensor,               # (n_atoms,)
    atom_A: torch.Tensor,            # (n_atoms, 5)  -- no grad. Per-ATOM, not per-element:
                                      # different atoms of the same element can have
                                      # different B-factors (see kernel_torch.build_atom_kernels_torch),
                                      # so build_element_kernels_torch's uniform-B path
                                      # must broadcast elem_A[element_idx] -> (n_atoms,5)
                                      # before calling this function.
    atom_lam: torch.Tensor,          # (n_atoms, 5)  -- no grad, same per-atom note as atom_A
    elem_offsets,                    # list of (K_e,3) int32 tensors, one per DISTINCT element
                                      # -- from precompute_element_offsets(), NOT recomputed here.
                                      # Stays per-element: it's only a generous candidate box,
                                      # sized for the largest radius among that element's atoms.
    atom_radius_ang: torch.Tensor,   # (n_atoms,) -- outer radius where taper reaches 0, A.
                                      # Per-atom, same broadcasting note as atom_A.
    grid_shape,                      # (Nu, Nv, Nw)
    orth_matrix: torch.Tensor,       # (3,3), Cartesian = orth_matrix @ frac -- no grad
    taper_width,                     # scalar, A -- taper spans [radius-taper_width, radius]
    max_atoms_per_batch: int = 50_000,
    max_pairs_per_batch: int = 4_000_000,
    compile_core: bool = True,
    out: torch.Tensor = None,        # optional preallocated flat grid, see below
    stats: dict = None,              # optional dict; work counters written into it
) -> torch.Tensor:
    """
    Returns the real-space electron density grid, shape (Nu, Nv, Nw),
    differentiable with respect to frac_coords. Peak memory is bounded by
    max_pairs_per_batch (atom-voxel pairs, the quantity that actually sets
    intermediate size) and secondarily by max_atoms_per_batch.

    max_pairs_per_batch is a SPEED knob as much as a memory one, and bigger is
    not better: 4M pairs means ~16 MB per (m,K) buffer, which stays resident in
    cache across the fused kernel's passes. Measured on the 135k-atom CPU case,
    the curve is flat-ish from 2M to 12M and rises on both sides -- 4M gave
    0.86 s, 16M gave 1.00 s, 48M gave 1.25 s, and 0.5M gave 1.01 s as
    per-call overhead starts to dominate. The optimum tracks cache size, so
    re-tune it with bench_splat.py on a new machine, and expect a GPU to prefer
    a larger value than this CPU-derived default.

    elem_offsets (from precompute_element_offsets) is only a candidate
    pre-filter, sized for the LARGEST radius among that element's atoms and
    padded by half a voxel diagonal to cover each atom's sub-voxel remainder.
    Because it is sorted by distance from the box center, each atom's real
    requirement is a PREFIX of it: atoms are processed in radius-sorted
    chunks, and a chunk only touches the prefix its own largest radius needs,
    rather than every atom of an element paying for that element's most
    spread out (highest-B) atom. This function then applies a SMOOTH taper
    using the true per-atom, per-voxel r2 and each atom's OWN radius,
    reaching exactly zero at atom_radius_ang -- see taper_factor() and this
    module's docstring for why this is smooth rather than a hard mask
    (gradient correctness).

    DISTANCES ARE SEPARABLE. Writing base = round(pos*N) and
    frac_rem = pos - base/N, the atom-to-voxel displacement is

        d_cart[i,k] = orth @ (offset[k]/N) - orth @ frac_rem[i]
                    = c_off[k] - c_rem[i]

    so  r2[i,k] = |c_off[k]|^2 - 2 c_rem[i].c_off[k] + |c_rem[i]|^2,
    where |c_off|^2 is precomputed per element and the cross term is ONE
    matrix multiply. The earlier formulation built four (m, K, 3) tensors per
    batch to get the same numbers; this builds a single (m, K) one and hands
    the heavy part to BLAS. It also removes the minimum-image wrap, which can
    never fire here (precompute_element_offsets checks that a local box stays
    well inside the cell, and raises if it does not).

    out, if given, is a preallocated FLAT (Nu*Nv*Nw,) grid, zeroed in place and
    accumulated into instead of allocating a fresh one per call. It exists to
    test whether repeated calls that reuse one buffer run faster than repeated
    calls that each get a new allocation -- the difference between how this is
    benchmarked and how xtraj's frame loop calls it. Forward-only: reusing a
    buffer that an autograd graph still references would corrupt it, so grad
    must be disabled.

    stats, if given, is filled with the work actually done -- atoms, chunks,
    pairs_ideal (each atom's own offset prefix, the figure bench_splat.py
    reports) and pairs_actual (what the chunked batching evaluates, where every
    atom in a chunk pays the chunk's largest prefix). Their ratio is the
    padding overhead, which depends on how the per-atom reach is distributed
    and so can differ between one set of coordinates and another. Counting is
    free: both come from values already resident on the host.
    """
    Nu, Nv, Nw = grid_shape
    device = frac_coords.device
    dtype = frac_coords.dtype

    assert Nu * Nv * Nw < 2**31, (
        f"grid has {Nu*Nv*Nw} points, too large for the int32 grid coordinates "
        "used here to save memory; either reduce grid resolution or widen "
        "the index dtypes in this function."
    )

    grid_dims = torch.tensor([Nu, Nv, Nw], device=device, dtype=dtype)
    extents = (Nu, Nv, Nw)

    if out is None:
        grid_flat = torch.zeros(Nu * Nv * Nw, device=device, dtype=dtype)
    else:
        assert not torch.is_grad_enabled(), (
            "splat_density(out=...) reuses the caller's buffer in place, which "
            "would corrupt any autograd graph still holding it; use it only "
            "under torch.no_grad()")
        assert out.shape == (Nu * Nv * Nw,) and out.dtype == dtype and out.device == device, (
            f"out must be a flat ({Nu*Nv*Nw},) {dtype} tensor on {device}, got "
            f"{tuple(out.shape)} {out.dtype} on {out.device}")
        grid_flat = out.zero_()
    core = _resolve_compiled(_density_core, compile_core)
    scatter = _resolve_compiled(_scatter_core, compile_core)

    # Sub-voxel geometry for EVERY atom at once (cheap, (n_atoms,3)): base is
    # piecewise-constant in pos so it carries no gradient (round()'s derivative
    # is zero a.e.), and c_rem is the only part the gradient flows through.
    base = torch.round(frac_coords * grid_dims).detach()            # (n,3)
    c_rem_all = (frac_coords - base / grid_dims) @ orth_matrix.T    # (n,3), differentiable
    base_i_all = base.to(torch.int32)

    # An offset can matter only if |c_off| <= radius + |c_rem|, since the true
    # distance is at least |c_off| - |c_rem|. Using each atom's OWN |c_rem|
    # makes the prefix EXACT rather than padding every atom by the worst-case
    # half-voxel-diagonal margin: |c_rem| averages well under that bound, so
    # this genuinely removes evaluated points rather than just relabelling them.
    # Kept in the compute dtype rather than promoted to float64: MPS has no
    # float64 at all, and the extra precision buys nothing here. reach only
    # selects how many candidate offsets to look at, and rounding can shift
    # that by at most one voxel shell -- whose density the taper has already
    # driven to ~0, since a point near |c_off| = reach sits at true r ~ R.
    reach_all = atom_radius_ang + c_rem_all.detach().norm(dim=-1)

    if stats is not None:
        stats["atoms"] = int(frac_coords.shape[0])
        stats["chunks"] = 0
        stats["pairs_ideal"] = 0
        stats["pairs_actual"] = 0

    n_elements = len(elem_offsets)
    for e in range(n_elements):
        mask = element_idx == e
        if not torch.any(mask):
            continue
        atom_indices = torch.nonzero(mask, as_tuple=True)[0]  # (M,)
        eo = elem_offsets[e]

        # Sort by reach so each chunk spans a narrow range and uses a short
        # offset prefix, instead of every atom paying for this element's
        # largest-B, most spread out one.
        reach = reach_all[atom_indices]
        order = torch.argsort(reach)
        atom_indices = atom_indices[order]
        K_needed = torch.searchsorted(
            eo.c_off_norm, reach[order].to(eo.c_off_norm.dtype), right=True,
        )
        K_needed_cpu = K_needed.detach().to("cpu", torch.int64).numpy()
        if stats is not None:
            stats["pairs_ideal"] += int(K_needed_cpu.sum())

        M = atom_indices.shape[0]
        start = 0
        while start < M:
            n = _chunk_length(K_needed_cpu, start, M - start,
                              max_atoms_per_batch, max_pairs_per_batch)
            sel = atom_indices[start:start + n]
            K = int(K_needed_cpu[start + n - 1])
            start += n
            if K == 0:
                continue
            if stats is not None:
                stats["chunks"] += 1
                stats["pairs_actual"] += n * K

            dens = core(
                c_rem_all[sel], eo.c_off[:K], eo.c_off_sq[:K],
                atom_A[sel], atom_lam[sel], atom_radius_ang[sel], occ[sel],
                taper_width,
            )                                                      # (m,K)

            scatter(grid_flat, base_i_all[sel], eo.offsets[:K], dens, Nu, Nv, Nw)

    return grid_flat.view(Nu, Nv, Nw)
