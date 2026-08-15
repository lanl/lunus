# Design note: bulk solvent for lunus.sf

Status: proposed, not implemented. Written 2026-08-15, revised the same day
against the current code — the API assumptions below were checked, the cost
estimate was replaced with a measurement, and three constraints were added
(symmetry ordering, B-factor differentiation, and the explicit/bulk boundary).

Scope, decided: the model is **explicit ordered waters plus a mask for the
disordered remainder**, and it is aimed at **ensemble models** — N
configurations sharing one atom list — rather than at MD trajectories, where a
frozen hydration shell stops meaning anything. Solvent **is** applied to the
diffuse observable, `per_conformer`.

lunus.sf currently computes `F_protein` only. Fitting experimental amplitudes —
and matching what SFcalculator, Phenix and REFMAC produce — needs a bulk-solvent
contribution. This note proposes the model, where it plugs into the existing
pipeline, and the two design constraints that make it non-trivial here.

## The model

The flat / mask-based bulk solvent model (Jiang & Brünger 1994), which is what
every refinement program implements:

```
F_total(h) = k_overall · exp(-h·U_aniso·h) · [ F_protein(h) + k_sol · exp(-B_sol·s²/4) · F_mask(h) ]
```

`F_mask` is the Fourier transform of a real-space solvent mask. Conventional
defaults are `k_sol ≈ 0.35 e/Å³` and `B_sol ≈ 46 Å²` (Phenix; confirm against a
current reference before pinning). `s² = 1/d²` already comes from
`structure_factor_torch.reciprocal_inv_d2()`.

Everything outside `F_mask` is arithmetic on quantities the pipeline already has.

## Building the mask: threshold the density we already compute

Two families are in use:

- **Geometric (Phenix / REFMAC).** Solvent is everything beyond `vdW radius +
  r_probe` from any atom center, then eroded by `r_shrink`. Conventional and
  accurate, but a distance transform is awkward on GPU and unpleasant to
  differentiate.
- **Density threshold.** Solvent is wherever the protein density falls below a
  cutoff. This is what SFC_Torch does.

**Recommendation: density threshold.** `splat_density()` already produces exactly
that grid, and `compute_fcalc()` already FFTs and extracts Miller indices from an
arbitrary grid — so `F_mask` is one call to existing code, and the mask inherits
the same symmetry expansion and grid conventions as the protein density.

Two details, both checked against the current code:

**The units work out with no new normalization.** `compute_fcalc` computes
`ifftn(grid) · cell_volume`, so for a dimensionless mask,
`k_sol · compute_fcalc(mask, …)` is in electrons — `e/Å³ · Å³`. And its existing
blur term is `exp(blur · 0.25 · inv_d2)`, the same convention as
`exp(−B_sol·s²/4)`, so `f_total` and `compute_fcalc` cannot disagree by a factor
of four.

**Mask the density AFTER symmetry expansion.** `symmetrize_sum` *sums* over the
orbit. Build the mask from the pre-symmetrization grid and sum it, and the
result reaches |orbit| rather than 1 and means nothing. The natural insertion
point is right after the `if grid_ops:` line in `structure_factors_one_config`,
which is correct — but this is a silent wrong answer, not an exception, so it is
written down here rather than left to be rediscovered.

**A known limitation, for the parity test's sake.** A density threshold is not
invariant to grid spacing, B-factors or `taper_width`, and a geometric mask is
invariant to all three. The same structure at `d_min` 0.9 and 1.8 gives
different solvent regions, and with per-atom B up to 100 a high-B surface atom
can fall below cutoff and be labelled solvent. That is acceptable for a
self-consistent reward function and it is a real difference from
Phenix/REFMAC — expect the gemmi parity check to show it, and do not read it as
a bug.

## Constraint 1: the mask must not kill gradients

A binary mask `rho < cutoff` has zero gradient almost everywhere and an undefined
one at the boundary. This is the same problem the density cutoff had, solved once
already with `taper_factor()` — see `examples/demo_taper_gradient.py` for why the
hard version is unusable.

Reuse that function, not just its shape. `taper_factor(r, R, w)` returns 1 for
`r ≤ R−w` and 0 for `r ≥ R`, so the mask is literally

```python
mask = taper_factor(density, cutoff, width)   # 1 in solvent, 0 in protein
```

with no rearrangement, and it inherits the C¹ guarantee and `tests/test_taper.py`
along with it.

**`width` is in density units (e/Å³), not Å.** Structurally it is the same class
of hyperparameter as `taper_width`; numerically it is nothing like it, and a
value copied across from the 0.1 Å atomic taper would be meaningless.

**Choose the width from the shell, not from forward accuracy.**
`∂mask/∂ρ = (π/2w)·sin(πx)`, so the gradient scales as **1/w** and is nonzero
only in the thin shell where `cutoff−w < ρ < cutoff`. Narrow width means a large
gradient carried by few voxels, which is grid-discretization sensitive: which
voxels land in the shell changes with grid spacing, so `dF/dparam` can be
unstable across `d_min` even where the forward `F` is not. The shell should span
**at least 2–3 voxels**, which is directly checkable — count voxels with
`0 < mask < 1` and compare against the boundary area divided by the voxel
spacing squared. Assert it in the test.

Note this tension is much sharper than for the atomic taper. There the taper is
per-atom and the gradient is a sum over ~10⁵ atoms, which averages; here it is a
single global level set with nothing to average against.

### Two mistakes not to inherit from SFC_Torch

Both are documented in sampleworks'
`core/rewards/structure_factor.py::_compute_ensemble_ftotal`, observed in
SFC_Torch 0.3.3's `rsgrid2realmask`:

1. **Not permutation-invariant over an ensemble.** Its batch path takes the
   quantile cutoff from batch element 0 and applies it to every configuration.
2. **Nondeterministic above ~5M voxels per configuration.** The cutoff is then a
   quantile of an *unseeded* `torch.randperm` subsample, so the mask and its
   gradients vary run to run.

Both disappear by deriving the cutoff from an absolute density scale rather than
a per-batch quantile. That also makes the mask independent of ensemble
composition — a property worth having when the ensemble is the thing being
refined.

## Constraint 2: ensemble semantics are a choice, not a detail

The mask operator is nonlinear, so `mask(⟨ρ⟩) ≠ ⟨mask(ρ)⟩`:

- **One mask from the summed density** — the ensemble is one averaged structure
  with one solvent region.
- **Mean of per-configuration masks** — each configuration carries its own
  solvent region, averaged after masking.

These are different physical models. sampleworks already exposes exactly this
distinction as `bulk_solvent="combined"` vs `"per_conformer"`. Make it a named
argument in `structure_factors_batch`, not an accident of where the sum lands.

### The diffuse observable: compute it, and only per conformer

`mean_and_diffuse` computes `⟨|F|²⟩ − |⟨F⟩|²`, which is what lunus exists for,
and solvent belongs in it. **Decision: compute it, `per_conformer`.**

The physics is the excluded volume. The mask fluctuation is *anti-correlated*
with the protein's — where an atom moves out, solvent moves in — so per-conformer
solvent does not add diffuse near the surface so much as **cancel** part of it.
Diffuse computed from `F_protein` alone overestimates the contrast at low
resolution, and this is the correction.

Note that `combined` is not a weaker version of this: one mask shared by all
configurations has zero variance and contributes **exactly nothing** to diffuse
while still changing Bragg. The `combined` / `per_conformer` choice is the
difference between having the effect and not having it.

**What the mask cannot give back.** A static mask fluctuates only where the
protein boundary moves. Explicit water additionally produces water–water
correlated scattering — the diffuse water ring near ~3.5 Å, and low-angle
solvent density fluctuations — which are real, measurable components of a
crystal's diffuse signal. Replacing waters with a mask removes that
contribution rather than approximating it. For the diffuse observable an
all-atom model is therefore *more* physical than the hybrid; the hybrid's
justification is cost, or a trajectory whose far-field solvent is not
trustworthy. Say which claim is being made, because a low-resolution diffuse
comparison will show the difference.

### Diffuse is a variance, so mask noise biases it upward

This is the constraint that should drive the parameter choices. In the Bragg
average, per-configuration numerical jitter in the mask averages out. In
`⟨|F|²⟩ − |⟨F⟩|²` it does not: **variance of noise adds**, so any
frame-to-frame jitter that is not physics becomes a systematic *positive bias*
in the diffuse signal, not a random error.

Two sources, both controllable:

- **Level-set discretization.** The mask is a level set sampled on a grid; as
  atoms move, voxels flip across the taper shell partly discretely. This argues
  for a **wider** taper for diffuse work than forward Bragg accuracy would
  suggest — the opposite of the usual pressure, and a decision to make
  deliberately rather than by inheriting a Bragg-tuned width.
- **Discrete changes in the atom set** — see the next section.

And the existing precision floor gets worse exactly where solvent matters:
`docs/performance.md` records diffuse reproducing to 2e-4 on CUDA against 4e-5
for Icalc, because it is a difference of two large nearly-equal numbers. The
cancellation above shrinks the low-resolution diffuse signal further, so the
same absolute noise is a larger relative error in the shells the solvent model
was added for. Re-measure that floor with solvent on — two identical CUDA runs —
before trusting low-resolution diffuse; float64 for the mask FFT is the
fallback.

Finally, `k_overall` and `exp(−h·U·h)` are Bragg scaling parameters. Applied per
conformer they scale diffuse by their square. That is probably what is wanted
for comparison, but it should be explicit rather than incidental.

## The explicit/bulk boundary: ordered waters stay explicit

The intended model is the crystallographic one: **ordered waters are explicit
atoms, and the mask covers the disordered remainder** — what Phenix and REFMAC
do. The density-threshold mask suits this better than a geometric one, because
it does not care which atoms produced the density: explicit waters raise the
density where they sit and so exclude themselves from the solvent region, with
no special handling of the boundary between the two models.

**Do not police this with the atom selection.** "Water is selected" is not the
failure; a fully solvated box is. Check the mask instead, once, at setup:

```python
occupancy = mask.mean()        # or the fraction of voxels above 0.5
```

| occupancy | meaning |
|---|---|
| ~0.3–0.7 | a plausible crystal solvent content; the hybrid is working |
| ≈ 0 | the selected atoms already fill the cell — `F_mask ≈ 0` and the feature is a **silent no-op** |
| ≈ 1 | cutoff too high, or almost nothing was selected |

The middle row is the one that matters. `xtraj.py`'s `selection` defaults to
`all`, and the tracked example system is 38.1% explicit water (51,690 of 135,834
atoms), so the default path on a solvated trajectory produces an empty mask, no
error, and the conclusion that bulk solvent does not matter. Print the occupancy
and warn at the extremes.

### The explicit waters must correspond across configurations

Whatever waters are explicit must sit at the **same positions in the coordinate
array in every configuration**. Two reasons:

1. **Physics.** A water that enters or leaves the explicit set between
   configurations is a whole atom's worth of density appearing and
   disappearing. In a variance observable that lands as spurious diffuse
   *precisely in the surface region the model is trying to get right*. This is
   the second source of bias from the previous section, and it is much larger
   than the discretization one.
2. **Mechanics.** The atom set is fixed at setup throughout: kernels are built
   once from a fixed element and B list, and `xtraj.py`'s `xray.structure`
   bypass resolves the selection, occupancies and element indices once. A
   per-configuration selection would not be slow, it would be **silently
   ignored**, with the setup-time set used for every configuration.

This suits an **ensemble model** — N configurations sharing one atom list, each
with the same explicit waters in the same slots — which is the case this feature
is for. It does **not** suit an MD trajectory, where waters wander: a first-shell
water at frame 0 is in the bulk a few hundred frames later, so a fixed subset
stops meaning anything. For MD, use all atoms and no mask, or protein only and a
mask, and do not try to freeze a hydration shell out of a diffusing solvent.

## Constraint 3: what a threshold mask does to ∂/∂B

Today `b_per_atom` reaches `build_atom_kernels_torch` as a numpy array and the
kernels are built once at setup, so **B is not in the autograd graph** and
`∂mask/∂B = 0`. That is not a property of the mask; it is a property of nobody
differentiating B yet. The day `kernel_torch` becomes differentiable in B — a
change with no diff in `solvent_torch.py` and no failing test — the solvent
envelope silently becomes a function of thermal parameters. This section exists
so that decision is made deliberately, while it still costs one keyword.

**The derivative has the wrong sign structure.** Raising B spreads an atom's
density and lowers it near the surface, pushing fringe voxels below cutoff, so
the solvent region *grows into the protein* as B increases. But the bulk-solvent
envelope is a property of the molecular boundary: a larger B means an atom's
position is less well determined, not that there is more disordered solvent
there. Phenix and REFMAC define the mask from vdW radii plus a probe precisely so
that it carries no thermal dependence, and recompute it per macro-cycle rather
than per gradient step — the mask is deliberately held constant while
gradients flow.

**It opens a cheap and wrong route for the optimizer.** `k_sol · F_mask` is
large at low resolution and partly cancels `F_protein` there. With a
B-differentiable mask, low-resolution amplitudes can be moved by inflating
surface B — growing the solvent region — without moving any atom. B refinement
is already the worst-conditioned part of low-resolution refinement; this couples
it to the solvent model and the refined B values stop meaning what B means.

**It is asymmetric between surface and core.** With per-atom B, `∂mask/∂B_i` is
nonzero only for atoms whose density crosses the cutoff, i.e. surface atoms.
Buried atoms get exactly zero. The solvent term therefore changes B refinement
only where B is already least constrained by the data.

**Recommendation: the mask should not see B by default.** Make the mask's
density source explicit rather than implicitly "the grid `F_protein` used", and
default to a detached grid (or one splatted at a fixed reference B):

```python
def solvent_mask(density, cutoff, taper_width):
    """... The mask carries whatever gradients `density` carries. Passing the
    same grid used for F_protein makes the solvent envelope a function of every
    parameter that grid depends on -- including B, once B is differentiable."""
```

That gives `∂F_mask/∂B = 0`, matching geometric-mask semantics and what
refinement programs do in practice, and leaves B-refinement conditioning
identical to the no-solvent case. Keep the fully differentiable path behind a
flag, with the shell-thickness check above attached to it, for anyone who wants
to experiment.

The cheap version of this is free *today*: passing the shared density already
gives coordinate gradients and no B gradient, because B is not in the graph. The
flag is what stops that from changing behind your back.

### A related trap for whoever makes B differentiable

`build_atom_kernels_torch` sizes each element's offset candidate list from the
**maximum B among that element's atoms, at setup**. Refine B upward past that
envelope and the density is truncated at the candidate-box boundary — and
non-smoothly, since the taper was sized for the old radius, so the very
discontinuity the taper exists to remove comes back. Either recompute the
offsets per B update (discrete, and not cheap) or size the envelope for a
maximum allowed B and clamp to it. Same future change, same day.

## Proposed API

A new `lunus/sf/solvent_torch.py`, mirroring `structure_factor_torch.py`:

```python
def solvent_mask(density, cutoff, taper_width):
    """(Nu,Nv,Nw) density → smooth solvent mask, differentiable in density.
    Thin wrapper over density_torch.taper_factor(density, cutoff, taper_width).
    Carries whatever gradients `density` carries -- see Constraint 3."""

def f_solvent(mask, cell_volume, hkl, orth_matrix):
    """Mask grid → (n_refl,) complex. Wraps compute_fcalc."""

def f_total(F_protein, F_mask, inv_d2, *, k_sol=0.35, b_sol=46.0,
            k_overall=1.0, u_aniso=None):
    """Combine per the equation above."""
```

The pipeline should take the mask's density source as an argument
(`mask_density=None` → detached copy of the protein grid), so that "does the
solvent envelope depend on B" is a visible choice rather than a consequence of
what `kernel_torch` happens to differentiate.

`structure_factors_one_config` / `structure_factors_batch` grow an optional
`solvent=` argument; `solvent=None` must leave today's numerical behaviour
bit-for-bit unchanged.

Keep `k_sol` / `b_sol` / `k_overall` as plain tensors so they can later carry
`requires_grad` and be refined, but ship them fixed. Fixed, unrefined scales are
what the sampleworks structure-factor reward already assumes, so the two stay
comparable.

## Validation

- **Parity against gemmi's solvent masking** (`SolventMasker`, with its Refmac /
  cctbx radii sets — check the current API), the same way `test_symmetry.py`
  pins `symmetrize_sum` against `symmetrize_sum()`. Note this compares a
  geometric mask against a density-threshold one, so expect agreement in
  `F_mask` at the level of the model difference, not to machine precision.
- **R-factor against experimental amplitudes** for a structure with a known
  refined R. Bulk solvent mostly affects low resolution, so a shell-resolved
  comparison below ~5 Å is where a wrong model shows up.
  `tools/compare_icalc_mtz.py` already reports by shell.

  **This criterion conflicts with shipping fixed scales.** A published refined R
  comes from refined `k_sol`, `b_sol`, `k_overall` *and* anisotropic scaling.
  With those pinned at 0.35 / 46 / 1 and `u_aniso=None` it will not be
  reproduced, and the gap will not distinguish a wrong mask from missing
  scaling. Either fit those two or three parameters by least squares **in the
  validation harness** — standard practice, and cheap, while the shipped default
  stays fixed — or restate the acceptance criterion as "R improves by X against
  no solvent, shell-resolved below 5 Å", which is measurable without any
  scaling. The first is worth the ~20 lines; without it the R number does not
  mean much.
- **Gradient check** in the style of `tests/test_torch_pipeline.py`: autograd
  through the mask against a finite difference.

- **Mask occupancy**, asserted to be in a plausible range on the example
  configuration, so the silent-no-op case above fails loudly in CI rather than
  in someone's results.

- **Diffuse null test.** Replicate one configuration N times: the true diffuse
  is exactly zero, so whatever comes out is the numerical floor. Run it with
  and without solvent — the difference is what the mask contributes to the
  floor.

- **Diffuse convergence in the taper width and the grid.** Compute diffuse with
  solvent at width `w` and `2w`, and at two grid spacings. The physical part
  converges; the bias from level-set discretization scales with `w` and the
  spacing. Until this is done, a diffuse difference attributed to excluded
  volume cannot be distinguished from a discretization artifact.

  With explicit ordered waters this study is two-dimensional: vary the width
  *and* the number of explicit waters. With waters present the density in the
  hydration region is intermediate between protein and bulk, so where the level
  set falls depends on how many were kept. If the diffuse does not change
  smoothly in both, the cutoff is sitting inside the hydration shell rather than
  outside it.

## Cost

One extra FFT per configuration, plus the mask evaluation over the grid.

**Measured, not assumed** — CUDA, 300 × 300 × 144, 251 frames
(`docs/performance.md`): `fft+extract` is **0.6 ms/frame** of an 87.3 ms frame,
and `symmetrize`, a comparable grid-wide pass, is 0.8 ms. So the solvent path is
one more FFT plus a pointwise taper over 12.96M voxels: **roughly 2 ms, 2–3% of
a frame.** The earlier note here — "at 300³ the FFT is not the cheap part" —
was wrong; the splat is ~72% of the frame and everything else together is under
10 ms.

Two caveats that survive the measurement. Under guidance the cost lands per
configuration per guided step, so it multiplies with the ensemble. And it
interacts with `use_checkpoint`: the mask FFT must happen **inside** the
checkpointed region, or the density grid is retained for all N configurations
simultaneously and the peak memory that checkpointing exists to control comes
back.

## Why this matters outside lunus

sampleworks' `StructureFactorRewardFunction` (SFC_Torch-backed) is memory bound
on its ASU-grid batch and does not reach large systems. A lunus path with bulk
solvent is what would let it be replaced there. Until then, lunus can generate
`F_protein` only, and any synthetic MTZ it writes will lack the `Ftotal` set that
the SFC-based generator produces.
