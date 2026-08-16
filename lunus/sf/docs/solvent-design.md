# Design note: bulk solvent for lunus.sf

Status: **stages 1-3 implemented** — `solvent_torch.py`, `solvent=` on both entry
points in `structure_factor_torch.py`, `tests/test_solvent.py`, gemmi parity in
`tests/test_solvent_gemmi.py`, and the diffuse discretization study in
`tools/study_diffuse_solvent.py` (findings in the two sections named "What the
... test found" below). Symmetry coverage is in
`tests/test_solvent_symmetry.py`, the degenerate-mask check is wired into
`SolventModel`, and float32 is measured (below). Still to do: the scaling fit
and shell-resolved R against experimental amplitudes, per-configuration
occupancies (see the API gap below), a numerical test of `u_aniso`, and a run
at production scale for cost.

Written 2026-08-15, revised the same day against the current code — the API
assumptions below were checked, the cost estimate was replaced with a
measurement, and three constraints were added (symmetry ordering, B-factor
differentiation, and the explicit/bulk boundary).

Scope, decided: the model is **explicit ordered waters plus a mask for the
disordered remainder**, and it is aimed at **ensemble models** — N
configurations sharing one atom array, in which the hydration is free to differ
between members and that difference is part of what is being modelled — rather
than at MD trajectories, where a frozen explicit water dissociates into the bulk
that the mask already represents. Solvent **is** applied to the diffuse
observable, `per_conformer`.

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
pinned in `tests/test_solvent_symmetry.py`.

Measured there, on P4₃, the error is not distributed the way one would guess.
The summed mask is roughly `(|orbit| − 1) +` the correct mask, and a constant
offset lands **entirely in F(000)** — 4.3× too large against an orbit of 4.
What survives for `h ≠ 0` is only the structure where orbit copies overlap:
22% on that system, and less on a sparser one. So the reliable detector is the
**grid**, not the transform: a mask whose values exceed 1 is wrong whatever its
`F_mask` happens to look like. `mask_occupancy()` sees this immediately.

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

The source is **level-set discretization**: the mask is sampled on a grid, and
as atoms move, voxels flip across the taper shell partly discretely.

*Measured since* — see "What the diffuse study found" below. The effect is real
(the mask raises the pipeline's grid sensitivity about sevenfold) but the taper
width barely controls it: a 16× wider shell buys ~25% less noise, because what
dominates is where the sampled density crosses the cutoff rather than how
smoothly it crosses. **Grid spacing is the lever**, with the noise falling as
roughly its square. The "wider taper for diffuse" instinct is directionally
right and quantitatively almost irrelevant.

Note this is about *numerical* noise. Genuine variation between configurations —
including which waters are associated and at what occupancy — is signal, not
bias; see "The array must correspond; the hydration may vary" below.

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
error, and the conclusion that bulk solvent does not matter.

**Implemented**: `SolventModel` measures the mask once, on the first
configuration, and raises `SolventMaskWarning` outside [0.05, 0.99] — a band
deliberately much wider than a real crystal's 0.3–0.7, since the point is to
catch a model doing *nothing* rather than to second-guess an unusual one. The
measurement is kept in `last_occupancy` / `last_shell_voxels` for callers that
would rather report than be warned, `check_occupancy=False` disables it, and
`warnings.simplefilter("error", SolventMaskWarning)` turns it into a failure —
worth doing in a pipeline, since both degenerate cases otherwise produce
numbers rather than exceptions. It costs one reduction per run, not per
configuration, because reading the scalars back synchronises the device.

### The array must correspond; the hydration may vary

Whatever waters are explicit sit at the **same positions in the coordinate array
in every configuration**, with the same count. That is a **mechanical**
requirement: kernels are built once from a fixed element and B list, and
`xtraj.py`'s `xray.structure` bypass resolves the selection, occupancies and
element indices once at setup. A per-configuration *selection* would not be
slow, it would be **silently ignored**, with the setup-time set used for every
configuration.

**It is not a claim that the hydration must be identical across
configurations.** The opposite: an ensemble model is exactly where which waters
are associated with the protein, and how strongly, is *meant* to differ between
members. That variation is part of the model being fitted, and its contribution
to the diffuse term is signal — the surface disorder the ensemble exists to
represent. Express it as **coordinates and occupancies differing between
configurations within a fixed array layout**, rather than as atoms entering and
leaving the array.

The density-threshold mask handles that gracefully, which is a second point in
its favour here: as an explicit water's occupancy falls, the density where it
sits falls with it, and the threshold hands that region back to the bulk. The
explicit and bulk contributions stay continuous across the transition instead of
double-counting at one end and leaving a hole at the other. How completely the
bulk takes over is then set by `k_sol` and the cutoff, which makes it a modelling
knob rather than an accident.

**Two things that genuinely are artifacts**, and should not be confused with the
above:

- **A per-configuration selection rule.** Re-picking "waters within X Å of the
  protein" per configuration makes membership a property of the rule and the
  grid rather than of the model, and it is silently ignored anyway for the
  mechanical reason above.
- **A frozen explicit subset of an MD trajectory.** Waters wander: a first-shell
  water at frame 0 is in the bulk a few hundred frames later, and keeping it
  explicit there both double-counts against the region the mask represents and
  injects one molecule's diffusion into the surface diffuse. For MD, use all
  atoms and no mask, or protein only and a mask. Do not try to freeze a
  hydration shell out of a diffusing solvent.

### API gap: occupancy is currently shared across configurations

`structure_factors_batch` takes one `occ` tensor for the whole ensemble, so
per-configuration occupancies — the natural way to express varying hydration
above — cannot be passed today. Accepting `occ` of shape `(N, n_atoms)` as well
as `(n_atoms,)` is a small change to the per-member call and is on the path for
this feature, not an optional extra.

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

- ~~**Parity against gemmi's solvent masking**~~ **done** —
  `tests/test_solvent_gemmi.py`. See "What the parity test found" below.
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

- ~~**Diffuse null test / convergence study.**~~ **done** —
  `tools/study_diffuse_solvent.py`, findings below. Note the null test as
  originally proposed does not exist: replicating one configuration gives a
  diffuse that is zero by construction and measures nothing, and the sub-voxel
  translation that replaces it is not a null either, because the grid pipeline
  does not commute with translation.

  **Still open**, and now better targeted: the study run so far varies the width
  and the grid on a water-free cluster. With explicit ordered waters the density
  in the hydration region is intermediate between protein and bulk, so where the
  level set falls depends on how many were kept — vary the **number of explicit
  waters** as the second axis (the width having turned out to be the weak one).
  If the diffuse does not change smoothly in it, the cutoff is sitting inside
  the hydration shell rather than outside it.

## What the parity test found

`tests/test_solvent_gemmi.py`, on 200 carbons clustered in a 30 Å cell at 0.625 Å
grid spacing, cutoff calibrated so both models call the same fraction of the
cell solvent (0.886).

**The convention matches.** gemmi's mask is 1 in solvent and 0 in the molecule,
same as `solvent_mask`. Asserted by sampling at atom centres, since a flipped
mask is the easiest catastrophic error to make here and would otherwise show up
only as a strange `F_mask`.

**Agreement is inside the spread of gemmi's own conventions**, which is the
only defensible yardstick for comparing two different models:

| | binarised voxel agreement |
|---|---|
| ours vs gemmi `Cctbx` | **0.995** |
| gemmi `Refmac` vs `Cctbx` | 0.987 |
| gemmi `VanDerWaals` vs `Cctbx` | 0.985 |

`|F_mask|` correlates at 0.9998 with R = 0.036 over the low-resolution
reflections where bulk solvent contributes at all. So the threshold mask is a
defensible stand-in — the model difference is smaller than the disagreement the
field already tolerates between radii sets.

### But matching gemmi exactly forces a nearly hard threshold

This is the substantive finding, and it changes how the cutoff should be
chosen. Matching gemmi's boundary puts the cutoff far out in the density tail
(1.45e-3 of peak here), where the density falls through the taper's range in
well under a voxel. The taper width cannot rescue it: `mask = 1` requires
`ρ ≤ cutoff − width`, so a width approaching the cutoff eliminates the fully
solvent region altogether (measured: width = 4× cutoff drops occupancy from
0.83 to 0.12).

The result is a mask that is smooth in principle and a hard threshold in
practice — 31 shell voxels out of 110,592 — which is precisely what the
gradients and the diffuse variance depend on.

The curve, on the same fixture, with width = ½ cutoff throughout:

| cutoff / peak ρ | solvent fraction | agreement vs gemmi | shell voxels |
|---|---|---|---|
| 1e-5 | 0.828 | 0.995 | 13 |
| 1e-3 | 0.830 | 0.995 | 93 |
| **1e-2** | **0.835** | **0.994** | **1,464** |
| 3e-2 | 0.850 | 0.978 | 2,331 |
| 1e-1 | 0.874 | 0.954 | 3,746 |
| 3e-1 | 0.920 | 0.908 | 8,701 |

(rows measured on the larger 400-atom fixture; the test's own numbers differ in
magnitude and not in shape)

**Recommendation, superseded — see "A real case" below.** The reasoning here
("put the cutoff near 1% of peak") was drawn from this sparse fixture and does
not transfer to a real structure, where the same calibration lands near 11% of
peak. Calibrate on occupancy, with `calibrate_cutoff()`; do not use a fixed
fraction of peak.

Note what this means for the diffuse work: a 2–3 voxel shell, which is what the
variance-bias argument asks for, is at the expensive end of this curve on a
0.625 Å grid and will be more expensive on the finer grids real `d_min` values
imply. The width, the grid and the diffuse bias have to be chosen together —
the convergence study in Validation is where that gets settled, not here.

### The cutoff is a calibration, not a constant

The matched cutoff moves by 21× between B = 20 and B = 60 on the test fixture,
and by three orders of magnitude on a larger one. Borrowing B = 20's cutoff for
a B = 60 structure costs real agreement (0.996 → 0.974). A geometric mask does
not move with B and a density threshold does — so the cutoff must be calibrated
for the system, B convention and grid in use. The test asserts this rather than
describing it, so that a future hard-coded default fails loudly.

## What the diffuse study found

`tools/study_diffuse_solvent.py`, on a 200-atom cluster in a 30 Å cell, 8
configurations displaced by σ = 0.3 Å, low-resolution reflections only, cutoff
at 1% of peak density per the parity recommendation.

**The instrument, and what it is not.** Displace every configuration by the
same sub-voxel vector. For the underlying model this multiplies every `F(h)` by
one common phase, so `⟨|F|²⟩` and `|⟨F⟩|²` are both unchanged and diffuse is
exactly invariant. The grid pipeline does not reproduce that: `splat_density`
samples each atom at fixed voxel centres, so a displaced structure is
*resampled* rather than shifted, and recovering the true shift would require
interpolation. The residual is therefore **not a null test** and not a bound on
the error against the truth — it measures how far the discretized calculation
is from a translation-invariant one. Grid refinement is the complementary
measurement: it is closer to the continuum limit but moves the sampling of the
density and the mask together, so it cannot separate them. Differencing with
and without solvent is what isolates the mask's share.

**The mask multiplies the pipeline's grid sensitivity by about seven.** At
0.625 Å voxels, translation non-commutation is 3.6e-3 of the diffuse signal
without solvent and 2.0e-2 with it. The reason is not subtle: the mask is
nearly a step function, so it is far less bandlimited than the smooth atomic
density and correspondingly worse sampled.

**The solvent's contribution to diffuse is real and large** — 16% of the
protein-only diffuse in this system, and stable across grids (0.166, 0.161,
0.166 at three spacings). It is signal, not an artifact.

**Grid spacing is the lever; the taper width is not.**

| voxel | solvent signal | translation noise | refinement | signal / noise |
|---|---|---|---|---|
| 0.833 Å | 0.166 | 4.3e-2 | 2.8e-2 | 3.8 |
| 0.625 Å | 0.161 | 2.0e-2 | 1.5e-2 | 8.1 |
| 0.417 Å | 0.166 | 8.9e-3 | 5.7e-3 | 18.6 |

The noise falls roughly as the **square of the voxel spacing** while the signal
stays put. Against that, widening the taper barely helps: over a 16× change in
shell voxels (width 0.05 → 0.9 × cutoff) the noise moves only ~25%, because
what dominates is *where the sampled density crosses the cutoff*, not how
smoothly it crosses. That is worth stating plainly because the earlier reasoning
in this note — "diffuse wants a wider taper than Bragg" — is directionally right
and quantitatively almost irrelevant.

**Recommendation: oversample the grid for solvent + diffuse work.** The default
`rate=1.5` in `grid_shape_for_resolution` gives spacing `d_min/3`, so a diffuse
study at `d_min` 2 Å lands at 0.67 Å — signal/noise ≈ 8. Raising `rate` to 2–3
buys the h² improvement and costs one FFT on a larger grid. Choose it from the
required signal/noise, and re-run this study when the system changes; the
numbers above are one cluster in one cell, and the *scaling* is the
transferable part, not the values.

A cheap regression guard lives in `tests/test_solvent.py`
(`test_solvent_diffuse_signal_dominates_grid_alignment_noise`), pinned at a
factor of three on a 0.62 Å grid where the measured ratio is ~5.

## At production scale

`tools/bench_solvent.py`, float32, one configuration, CPU (a GPU run is still
owed — see below).

**Measured on CUDA**, 7FPV in its own cell (P2₁2₁2₁, 120 × 160 × 360, waters
excluded, cutoff 1% of peak), whole pipeline warmed before timing:

| phase | ms |
|---|---|
| splat (3,165 atoms) | 5.3 |
| symmetrize | 0.6 |
| fft+extract (protein) | 0.3 |
| mask | 0.1 |
| **solvent (mask FFT + combine)** | **0.9** |
| total | 7.2 |

**Solvent costs ~1 ms per configuration** — 13.8% of this small structure's
frame, and it would be ~1.4% of the 135,834-atom trajectory frame (70 ms),
because the cost is **per-grid, not per-atom**: the mask and its FFT scale with
voxels while the splat scales with atoms. Peak device memory goes 201 → 229 MB,
one extra grid's worth.

Two earlier figures for this were wrong and are worth recording as traps. A
"3.9% on compare_gemmi" number came from forcing a mask onto an all-atom MD
model by selecting the peptide — a configuration nobody should run, since that
model carries its solvent explicitly. And an "85.9% of the frame" number was
one-time CUDA library initialisation, `torch.linalg.inv` loading cuSOLVER on
first use, landing in the timed row: the same block repeated measured 0.7 ms.
Both benchmarks now warm the whole pipeline before timing.

### Fold last: the mask belongs in the model's own frame

Folding a model into a smaller cell destroys the boundary the mask needs.
Measured on the compare_gemmi topology, protein only, cutoff 1% of peak:

| frame | median ρ | occupancy |
|---|---|---|
| the model's own P1 box | 0.076 | **0.344** |
| folded into the crystallographic cell | 0.578 | 0.010 |

So a mask thresholded from the folded density describes a cell with no empty
space left in it. The order has to be: build the density in the model's frame,
**build the mask there**, then fold.

`supercell=(na, nb, nc)` on both entry points does that. The atoms are splatted
onto the larger grid, the mask is built on it, and density and mask are folded
into the target cell separately — which is exact, because folding is linear, and
which keeps `exp(−B_sol·s²/4)` in reciprocal space where it belongs rather than
forcing the mask to be blurred in real space before it is added.

**Commensurate or refuse.** `symmetry_torch.supercell_factors()` derives the
factors from the two cells and raises unless each axis ratio is an integer,
because folding a density *grid* is only exact when the target tiles the source.
(Folding *atoms* is always well defined — the modulo in the splat's scatter —
which is why the constraint only appears once masks are folded.) A model
spanning several cells already contains its symmetry copies, so `supercell=`
refuses `grid_ops=` as well: expanding again would double-count, and summing a
mask over the symmetry orbit does not give a mask.

Without solvent, `supercell=` changes nothing observable, and there is a test
asserting exactly that against the direct path — if the two foldings ever
disagree, every supercell result is suspect.

### Retracted: "the compare_gemmi system cannot exercise this model"

An earlier version of this section said that, on the strength of a mask that
came out empty with any selection or cutoff. The measurement was real; the
explanation was wrong, and so was the conclusion.

The example was being run with `unit_cell=88.451,88.451,39.823` and
`space_group=P43`, carried over from a different example. The crystal is **PDB
7FPV, 34.196 × 45.558 × 99.044, P2₁2₁2₁**, and the simulation box is exactly a
2×2×2 supercell of it. Folding into the correct cell stacks the eight images
coherently and the boundary survives; folding into a cell with no integer
relationship to the box scatters 32 protein copies at unrelated positions and
fills it solid.

| configuration | atoms | occupancy in the box | occupancy after folding |
|---|---|---|---|
| wrong cell, 88.451 / P4₃, protein only | 77,056 | — | **0.0100** |
| **7FPV cell, P2₁2₁2₁, 2×2×2, protein only** | 77,056 | 0.3443 | **0.3122** |
| 7FPV cell, all atoms (explicit water) | 135,834 | 0.0022 | 0.0000 |

So the system exercises the model perfectly well, and the all-atom row is still
the degenerate case this note warns about — explicit bulk water leaves nothing
for a mask to find, which is a property of the model, not of the cell.

`calibrate_cutoff(density, 0.50)` on the folded protein-only density lands at
**1.1e-01 of peak**, the same ~11% found independently on 4WOR. Two unrelated
real crystals agreeing on the calibration is worth more than either alone.

**The lesson worth keeping** is about the diagnosis, not the cell: an empty
mask says the density has no empty space in it, and that has at least two
causes — a model that genuinely fills the cell (explicit solvent), and a model
folded into the wrong cell. `mask_occupancy()` detects the symptom; it cannot
distinguish the causes. Check that the box is a supercell of the cell before
concluding anything about the solvent, which is what
`symmetry_torch.supercell_factors()` is now for.

### Fold last: the mask belongs in the model's own frame

Folding a model into a smaller cell destroys the boundary the mask needs.
Measured on the compare_gemmi topology, protein only, cutoff 1% of peak:

| frame | median ρ | occupancy |
|---|---|---|
| the model's own P1 box | 0.076 | **0.344** |
| folded into the crystallographic cell | 0.578 | 0.010 |

So a mask thresholded from the folded density describes a cell with no empty
space left in it. The order has to be: build the density in the model's frame,
**build the mask there**, then fold.

`supercell=(na, nb, nc)` on both entry points does that. The atoms are splatted
onto the larger grid, the mask is built on it, and density and mask are folded
into the target cell separately — which is exact, because folding is linear, and
which keeps `exp(−B_sol·s²/4)` in reciprocal space where it belongs rather than
forcing the mask to be blurred in real space before it is added.

**Commensurate or refuse.** `symmetry_torch.supercell_factors()` derives the
factors from the two cells and raises unless each axis ratio is an integer,
because folding a density *grid* is only exact when the target tiles the source.
(Folding *atoms* is always well defined — the modulo in the splat's scatter —
which is why the constraint only appears once masks are folded.) A model
spanning several cells already contains its symmetry copies, so `supercell=`
refuses `grid_ops=` as well: expanding again would double-count, and summing a
mask over the symmetry orbit does not give a mask.

Without solvent, `supercell=` changes nothing observable, and there is a test
asserting exactly that against the direct path — if the two foldings ever
disagree, every supercell result is suspect.

### The compare_gemmi system cannot exercise this model at all

Worth stating plainly, because it is the repo's standard benchmark input and it
looks like the obvious thing to test on. Its fractional coordinates span
**−0.35 to 5.12**: the MD box is several crystallographic cells across, so
folding it into the 88.451 × 88.451 × 39.823 cell stacks several protein copies
on top of one another, and the P4₃ expansion multiplies that again. The
resulting density has **minimum 0.163 and median 2.48 e/Å³** — there is no
empty space anywhere to call solvent. Selecting `peptide` does not help: the
protein copies alone still fill the cell.

Run it anyway and the mask comes out empty, `F_mask ≈ 0`, and
`|F_total|/|F_protein| = 1.0000` exactly. `SolventModel` warns, which is the
degenerate-mask check earning its place on real data rather than on a
constructed fixture.

The right input is a **crystallographic model** — one asymmetric unit in its
own cell, with solvent channels. Two of them, independently:

| structure | occupancy | shell voxels | `\|F_total\|/\|F_protein\|` |
|---|---|---|---|
| 4WOR, P4₁ | 0.5097 | 258,256 | 0.9176 |
| 7FPV, P2₁2₁2₁ | 0.5516 | 672,888 | 0.9275 |

— textbook solvent contents, and a 7–8% low-resolution contrast cancellation,
which is what this model exists to represent.

This also sharpens the scope statement at the top. Bulk solvent belongs with
crystallographic models and ensembles built from them; an all-atom MD box in
its own P1 frame already contains its solvent explicitly and needs no mask.

## A real case: 4WOR

P4₁, 48.499 × 48.499 × 63.430, waters excluded, 160 × 160 × 208 grid, d_min 2.0,
9,988 reflections. `F_total` built two ways — our threshold mask, and gemmi's
geometric mask through the same FFT and the same `f_total` — so the comparison
isolates the mask model in the quantity that would actually be fitted.

| cutoff | occupancy | shell voxels | R vs gemmi-mask `F_total` |
|---|---|---|---|
| 1% of peak (the sparse-fixture recommendation) | 0.3817 | 57,076 | 0.0528 |
| 0.3% of peak | 0.3752 | 5,932 | 0.0549 |
| **occupancy-matched, = 11% of peak** | 0.5085 | 377,216 | **0.0246** |

**The fixed-fraction recommendation was wrong for real structures.** Matching
gemmi's solvent content on 4WOR needs a cutoff two orders of magnitude higher
than the sparse fixture implied, and it is simultaneously twice as accurate
(R 0.025 against 0.053) and an order of magnitude smoother (377k shell voxels
against 57k). The earlier tension — "matching gemmi forces a nearly hard
threshold" — was an artifact of a model with genuine vacuum between its atoms.
A real structure has no vacuum, only overlapping tails, so carving out the same
volume takes a much higher threshold, and that threshold sits where the density
is varying quickly and the shell is thick.

Hence `calibrate_cutoff(density, target_occupancy)`: bisection to a requested
solvent content, run once at setup. It also refuses targets outside the
reachable range, which is **not** (0, 1) — a model with vacuum has an occupancy
floor at the vacuum fraction, and asking for less than that used to return a
meaningless cutoff without complaint.

**How large is the model difference, in context?** At the calibrated cutoff,
`F_total` from the two masks agrees to R = 0.025 overall and 0.036 in the worst
shell (> 6 Å). The solvent contribution itself — `F_protein` against `F_total` —
is R = 0.066 overall and **0.267 beyond 10 Å**. So the choice of mask model is
roughly a third of the size of the effect it models at low resolution: real, and
subdominant.

## float32 is not the limiting precision

Everything in the tests runs float64 for exact comparisons; production runs
float32 on a GPU, and the mask is the part of the pipeline most likely to care,
since its level set sits in the density tail and its taper shell can be a few
tens of voxels wide.

Measured, same structure through both precisions:

| cutoff | shell voxels (f32 / f64) | mean rel. error on F_total | max |
|---|---|---|---|
| 1% of peak | 1134 / 1134 | 1.3e-6 | 3.1e-5 |
| 1e-5 of peak | 9 / 9 | 1.2e-5 | 1.6e-4 |

The thin-shell case — the one a geometric-mask match forces — costs about 9×,
which is the expected direction and still small. **The shell voxel count is
identical in the two precisions in both cases**, so the level set does not move;
what differs is only the taper value within it.

For the diffuse observable, the cancellation-sensitive one, float32 against
float64 is **1.6e-6 with solvent and 3.0e-6 without** — the mask does not make
it worse.

Against `docs/performance.md`'s reproducibility floor for this pipeline — GPU
non-determinism of 4e-5 on Icalc and 2e-4 on diffuse — float32 sits one to two
orders of magnitude below the noise already present. It is not worth switching
the solvent path to float64.

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
