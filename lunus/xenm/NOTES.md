# Development Notes

Summary of the work that brought `xenm` into the lunus repository. These are
the essential notes behind the initial commit; see the READMEs for usage.

The code was first used in Wall, Wolff & Fraser (2018), "Bringing diffuse X-ray
scattering into focus," *Current Opinion in Structural Biology* 50, 109-116
(https://doi.org/10.1016/j.sbi.2018.01.009).

## Integration and modernization

- Tidied the original ENM refinement code and integrated it under `lunus/xenm/`
  as a proper package (`__init__.py`, top-level `README.md`, example docs).
- Modernized Python 2 -> 3: `print`, integer division, `xrange` -> `range`,
  `np.complex` -> builtin `complex`.
- Fixed cctbx atom selection for the modern API. After the structure is
  expanded to P1 and its scatterers are reordered, selections are built by
  parsing scatterer labels (` CA `, backbone ` N `) into `flex.bool` arrays.
- Switched the parallel diffuse calculation from `multiprocessing.Pool` to
  `multiprocessing.pool.ThreadPool` so worker functions share module globals
  (the original code relied on fork-inherited globals).

## `calcenm.sh` workflow

- Runs with plain `python` (cctbx is expected in the active environment).
- Uses the `lunus.`-prefixed C lattice tools (`lunus.hkl2lat`, `lunus.symlt`,
  `lunus.anisolt`, `lunus.corrlt`) and adds `<lunus_root>/c/bin` to `PATH`
  automatically.
- `resolution` is exposed via the `RES` environment variable (default 4.0 for
  fast testing; ~1.6 matches the full experimental grid).
- Decompresses the experimental `.lat.gz` on demand.

## Experimental data

- The 4WOR example ships the experimental diffuse data as a gzipped lunus
  lattice, `snc_exafel_mean_mirror_sym_aniso_cull.lat.gz` (smaller than the raw
  hkl and directly usable as both the `hkl2lat` template and correlation
  target).
- Regenerate a `.lat` from raw `h k l I` with `lunus.makelt` (build a template
  at the target cell and resolution) followed by `lunus.hkl2lat`.

## 4WOR identity and unit cell

- PDB 4WOR is **staphylococcal nuclease** (thermonuclease) in complex with
  Ca2+ and thymidine-3',5'-diphosphate (pdTp) at room temperature -- not
  nitrogenase reductase.
- The authoritative unit cell is the CRYST1 record:
  **a = b = 48.499 A, c = 63.430 A, alpha = beta = gamma = 90 deg, space group P41.**
  The experimental diffuse data uses this same cell. All cell references (docs,
  the `anisolt` call in `calcenm.sh`, the `makelt` template) are aligned to it.

## Diffuse-scattering vectorization (~3x faster)

The per-reciprocal-point loops spent nearly all their time on the coupling
contraction `fr^H . dc . fr`, where `dc = exp(qsq * V) - 1` on the (nsel x nsel)
covariance matrix, evaluated at every grid point.

Key observation: `dc` depends only on the scalar `qsq`, and the grid has
~14-17x fewer distinct `qsq` values than points (h^2/k^2/l^2 degeneracy plus
the a = b cell symmetry). `calc_diffuse_vectorized` exploits this:

- Coarse-grained scattering factors `fr` are computed in large vectorized
  blocks, using a precomputed residue-grouping matrix `G` (`fr = f @ G^T`) that
  replaces the inner per-atom accumulation loop.
- Points are grouped by unique `qsq`; `dc` is computed once per group and
  applied to all its points with a single BLAS matrix multiply.
- The intra-residue correction is folded into the block loop to bound memory.

Numerically identical to the original loops (relative difference ~1e-9). The CA
path with `kpoints` 0 or 1 (used by `calcenm.sh` and the refinement) routes
through the fast path; the all-atom and multi-R `kpoints > 1` cases keep the
original loops.

**Memory note:** at 1.6 A the `fr` array is ~2.7 GB (npts x nsel complex). Fine
on a workstation, but larger cells would need the `qsq` grouping itself blocked.

## Interaction cutoff is not a physical knob (it is conditioning noise)

Tightening the interaction `cutoff` from 25 to 10 A changed the anisotropic
diffuse correlation (0.361 -> 0.391 at `resolution=4.0`, `anmexp.lambda=0.16`),
which looked like a physical effect worth optimizing. It is not. A cutoff sweep
is rugged and non-monotonic -- 6/8/10/12/15/20/25 A give
0.043/0.238/0.391/0.352/0.388/0.410/0.361 -- the fingerprint of numerical
instability, not an optimum. The findings, all verified:

- **The toggled springs are negligible.** With `anmexp.lambda = 0.157` the
  ANMEXP force constant is `k = (1/r) exp(-r/2λ)`, i.e. a decay length of
  `2λ ≈ 0.31 A`. Summed over the real 4WOR CA network (27 cells), the springs
  in the 10-25 A band total ~1e-13 against ~4e-4 for the 0-4 A contact shell --
  a relative ~1e-9 perturbation.
- **In the real-space Hessian the change is below rounding.** For `kpoints=0`,
  `||H(25) - H(10)|| / ||H(10)|| = 1.3e-18` -- the two Hessians are bit-for-bit
  identical.
- **But the Hessian is near-singular and we invert it.** The softest physical
  mode / largest eigenvalue ratio is ~1.6e-15, essentially on top of
  `numpy.linalg.pinv`'s default `rcond = 1e-15`. Perturbing H by *any* relative
  amount (1e-6 down to 1e-12) reshuffles the pseudoinverse by 80-98%, because it
  flips whether borderline soft modes survive `rcond` truncation, and the
  covariance `V ≈ Σ vvᵀ/λ` is dominated by exactly those soft modes.
- **Decisive control.** Raising `rcond` to 1e-9 makes the diffuse correlation
  *identical* (-0.0248) for every cutoff 8-25 A -- the cutoff dependence
  vanishes entirely once the borderline modes are truncated consistently. (1e-9
  is too aggressive: it also removes the real signal, showing the diffuse signal
  lives in modes right at the `rcond` edge.)

Conclusion: with a short `anmexp.lambda`, cutoffs above ~6 A are equivalent in
exact arithmetic; the observed differences are near-singular-pinv noise.
Choosing a cutoff by maximizing the diffuse correlation is fitting to that
noise. The default cutoff is left at 10 A but should not be read as tuned.

## The model is the renormalized covariance; its content is the correlation matrix

A subtlety that reframes everything below: **the model is not the Hessian, and
not the raw covariance -- it is the renormalized covariance V.** The pipeline is

```
H  ->  V0 = truncated pseudoinverse(H)  ->  V = S · V0 · S,   S = diag(sigs/Vsigs)
```

Renormalization is a diagonal congruence. Two consequences, both verified
directly on 4WOR:

- **It forces `diag(V)` to equal the experimental B-factors exactly, for any
  parameters.** The model B-factors are therefore an *imposed input*, not a
  fitting target -- there is nothing to fit on the diagonal.
- **It preserves the correlation matrix exactly** (max elementwise diff 5e-16):
  `Corr[i,j] = V0[i,j]/(σ_i σ_j)` is invariant under `V0 -> S·V0·S`. So the
  model's entire predictive content -- everything the diffuse data actually tests
  -- lives in the **correlation matrix**, which the parameters shape and
  renormalization leaves untouched.

This means the "B-factor agreement" of the raw pre-renorm covariance `diag(V0)`
is *not* a property of the model; it is a property of the discarded intermediate
`V0`. The right diagnostic across parameters is the correlation matrix and its
conditioning, not the raw diagonal.

## What actually varies with λ (correlation matrix + conditioning)

Sweeping λ at fixed `cutoff=15`, `resolution=4.0`. The "raw-B corr" and
"dynamic range" columns below describe the **discarded `V0`**, not the model
(whose B-factors are always exact); they are kept only to show how violent the
renormalization surgery is at small λ:

| λ    | diffuse corr | cond(H raw) | (V0 raw-B corr) | (V0 dyn. range) |
|------|--------------|-------------|-----------------|-----------------|
| 0.16 | **0.388**    | 6.6e14      | 0.215           | 9.4e4           |
| 0.30 | 0.214        | 1.8e8       | 0.244           | 7.9e2           |
| 0.50 | 0.188        | 2.2e5       | 0.388           | 1.1e2           |
| 1.0  | 0.134        | 3.3e3       | 0.640           | 2.6e1           |
| 2.0  | 0.056        | 2.1e2       | **0.728**       | 9.7             |

The best diffuse correlation (0.39 at λ=0.16) coincides exactly with the *worst*
conditioning (cond H = 6.6e14, at the edge of double precision). This is not a
tradeoff between two physical objectives -- the diagonal is imposed either way.
It is a warning: the 0.39 sits precisely where the pseudoinverse is dominated by
`rcond`-truncation noise (see the cutoff section above, and the two-tier test
below).

## Conditioning is the hidden variable behind the whole tradeoff

The condition number of the raw Hessian (largest / smallest-non-rigid
eigenvalue) spans twelve orders of magnitude across the λ sweep and moves
monotonically with everything else (`cutoff=15`, real-space `kpoints=0`):

| λ    | cond(H raw) | cond(V0) | cond(V renorm) | diffuse | raw-B corr |
|------|-------------|----------|----------------|---------|------------|
| 0.16 | 6.6e14      | 2.0e8    | 2.6e6          | 0.388   | 0.215      |
| 0.30 | 1.8e8       | 7.5e4    | 5.5e4          | 0.214   | 0.244      |
| 0.50 | 2.2e5       | 7.2e3    | 7.6e3          | 0.188   | 0.388      |
| 1.0  | 3.3e3       | 7.0e2    | 6.5e2          | 0.134   | 0.640      |
| 2.0  | 2.1e2       | 8.6e1    | 8.8e1          | 0.056   | 0.728      |

- Small λ -> near-singular H (6.6e14, at the edge of double precision); large λ
  -> trivially invertible (2e2). Best diffuse correlation coincides exactly with
  worst conditioning.
- `cond(V0)` is much smaller than `cond(H)` (2.0e8 vs 6.6e14 at λ=0.16) because
  the default path `numpy.linalg.pinv(H)` *truncates* singular values below
  `rcond = 1e-15 x largest`. V0 is a regularized inverse, not a true one: a
  faithful inverse would give V0 a top eigenvalue ~1/2.16e-24 = 4.6e23, but the
  truncation caps it at 3.9e22. The softest surviving H mode (2.16e-24) sits
  just above the cut (1.42e-24) -- a knife's edge, which is exactly why the
  cutoff sweep was rugged.
- Renormalization (`V -> S V S`, S = diag(renorm_fac)) helps the covariance
  conditioning at small λ (2.0e8 -> 2.6e6) but is nearly inert at large λ where
  S is already ~uniform.

## Small λ = near-decoupled crystallographic molecules (the real structure)

The physical reading of "starved off-diagonals + extra soft modes": at small λ
the Hessian describes a set of near-decoupled domains connected only by
threadlike springs. Measured on 4WOR:

- At λ=0.16 there are **9 near-zero eigenvalues** vs 3 at λ>=0.3 -- six *extra*
  soft modes, i.e. the near-free relative motion of soft-coupled blocks.
- Thresholding the atom-atom coupling graph at λ=0.16: keeping only edges
  stronger than 1e-4 x max splits the network into **exactly 4 clusters of 137
  atoms** (548 = 4 x 137). 4WOR is **P4_1 with 4 molecules** in the P1-expanded
  cell -- the four "domains" are the four symmetry mates, held together only by
  springs weaker than 1e-4 of the strongest. At λ=0.5 the fragmentation
  threshold moves to 1e-2; by λ=2.0 the network is robustly connected at all
  scales.

So the small-λ diffuse signal is dominated by each molecule's *internal*
vibrations (physically the leading term in protein diffuse scattering), which is
why it agrees with data despite the crystal being numerically near-disconnected.
The inter-molecular coupling -- including crystal contacts -- is essentially
absent.

## Two-tier domain-coupled ENM (`model=ANMEXP_DOMAIN`) -- implemented and tested

Rather than tuning a single global λ (which forces one stiffness on both intra-
and inter-molecular springs), the model is split into two tiers on the CA network:

1. **Intramolecular pairs** (same P4_1 molecule) keep the ANMEXP form,
   `Γ_intra = exp(-r/λ)` (weight `(1/r)exp(-r/2λ)`).
2. **Inter-molecular pairs** (different molecules, across all 27 cells, includes
   crystal contacts) get a logistic-threshold ANM spring,
   `Γ_inter = k_inter / (1 + exp((r - thresh)/w))`, refined center `thresh`,
   fixed width `w`, and tier ratio `k_inter`.

Molecule membership is a parameter-independent `molid` array (4 blocks of 137,
from the P4_1 expansion); the intra/inter split is a constant mask. Params:
`anmexp.lambda`, `inter.thresh`, `inter.k`, `inter.width`. Selected via
`model=ANMEXP_DOMAIN` in both `calcH` (kpoints=0) and `calcDk` (kpoints=1); the
default `ANMEXP`/`EXPCUT` paths are unchanged.

**Conditioning objective achieved.** At λ=0.16, thresh=10, k=1 the two-tier model
drops cond(H) from 6.4e14 to ~1e3 and restores the near-zero mode count from 9 to
3 -- the six spurious soft modes are lifted cleanly off the `rcond` floor. The
inversion is now honest (no truncation noise).

**But the diffuse correlation does not survive the fix -- which is the point.**
With honest, well-conditioned inter-molecular coupling the anisotropic diffuse
correlation collapses, and it is *frozen* against the coupling parameters:

| configuration                         | diffuse corr        |
|---------------------------------------|---------------------|
| plain ANMEXP (rcond-edge inversion)   | **0.39**            |
| ANMEXP_DOMAIN, k_inter = 0 (exact block decoupling) | 0.047 |
| ANMEXP_DOMAIN, k_inter = 1e-4 … 1.0   | **−0.023** (flat)   |

- The correlation is **discontinuous at k=0** and then **constant to 4 significant
  figures across four orders of magnitude of k_inter** (1e-4 → 1.0), and nearly
  constant across λ (−0.0229 at λ=0.16 → −0.0101 at λ=1.0) and across thresh
  (−0.018 at 6 Å → −0.034 at 15 Å). The flatness in k_inter is the
  renormalization-invariance of an overall scale showing through: once every
  inter pair carries *some* honest spring, the correlation matrix is essentially
  set, and scaling it up is (nearly) inert.
- **Even exact, clean domain decoupling (k=0) only reaches 0.047**, far below
  0.39. So the 0.39 is not "the internal vibrations of cleanly isolated
  molecules" either. It is specifically the near-singular soft modes surviving
  the `rcond` cut in the plain-ANMEXP inversion.

**Conclusion.** The 0.39 anisotropic correlation of the single-tier ANMEXP model
lives entirely in the `rcond`-truncation regime of a near-singular Hessian; it is
a numerical artifact, not physical inter-domain signal. Giving the model honest,
well-conditioned coupling -- the correct thing to do numerically -- removes it.
A physically meaningful diffuse model will need signal that is robust to
conditioning (e.g. richer inter-molecular contact geometry, all-atom rather than
CA, or a different covariance construction), not a spring stiffness tuned at the
edge of double precision.

## Verified

- End-to-end `calcenm.sh 4wor.pdb 0.16` gives an anisotropic correlation of
  **0.36105** at `resolution=4.0`.
- The diffuse calculation dropped from ~50 s to ~14 s at 4 A; also verified at
  1.6 A (307520 reflections).
- Cutoff/rcond diagnosis and the λ sweep table above were measured directly on
  4WOR (Hessian norms, eigenvalue spectrum, pinv perturbation test, rcond control
  sweep, and the λ sweep).
- Conditioning table, the pinv-truncation explanation (cond(V0) << cond(H)), and
  the domain-decoupling result (9 vs 3 near-zero modes; 4x137 clustering at
  λ=0.16 matching the 4 P4_1 molecules) were all measured directly on the dumped
  Hessians/covariances at `cutoff=15`, `resolution=4.0`.
- Renormalization preserves the correlation matrix exactly (max elementwise diff
  5e-16) and forces `diag(V)` to the experimental B-factors exactly -- verified on
  the dumped V0/V for 4WOR.
- Two-tier `ANMEXP_DOMAIN` results (cond 6.4e14 -> ~1e3, 9 -> 3 near-zero modes at
  k=1; diffuse corr 0.047 at k=0, flat −0.023 for k=1e-4…1.0, thresh sweep 6–15 Å,
  λ sweep 0.16–1.0) measured end-to-end via `calcenm.sh` at `resolution=4.0`.
