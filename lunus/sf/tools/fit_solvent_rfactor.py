#!/usr/bin/env python
"""
Fit the bulk-solvent scales against deposited amplitudes, and report R by shell.

    PYTHONPATH=<repo root> python tools/fit_solvent_rfactor.py \\
        examples/compare_gemmi/7FPV.pdb examples/compare_gemmi/7FPV-sf.cif

This is the validation gate named in docs/solvent-design.md: every other check
on the solvent model compares it against itself, against gemmi, or against a
constructed fixture. This one compares it against measurements, which is the
only test that can find a mask model that is self-consistently wrong.

WHAT IS FITTED, AND WHY THAT IS NOT CHEATING. k_sol, b_sol and the overall
scale are refined HERE, in the harness, while the shipped SolventModel defaults
stay fixed at 0.35 / 46 / 1. That split is deliberate and is argued in the
design note: a published refined R comes from refined solvent scales plus
anisotropic scaling, so comparing fixed defaults against it measures the
missing scaling rather than the mask. Fitting three parameters against 70,000
observations is standard practice and is what makes the residual attributable
to the model.

--scale-bins replaces the single overall scale with one per resolution bin,
which is what mmtbx.f_model fits and is most of why its R is lower on the same
model. `--scale-bins log` uses mmtbx's own bins (cctbx's log_binning: fine at
low resolution, wide at high, as many as the data support up to 30);
`--scale-bins N` uses N equal-count bins, which is what every binned number
recorded before 2026-11-09 was measured with. Across the 40 sampleworks
entries the swap test (lunus.sf F_calc and mask, mmtbx scaling) put two thirds
of the R gap in the scaling model and none in the structure factors. It is off
by default so the numbers in the design note stay reproducible. The cost is
that the lowest-resolution bin overlaps the solvent term, so a binned fit
WITHOUT solvent recovers some of what the mask would have, and the drop in R
from switching the solvent on is smaller.

WHAT THE ANSWER MEANS, AND WHAT IT DOES NOT.

  * The target is NOT the deposited R = 0.126 / R-free = 0.145. That came from
    a fully refined model with refined individual anisotropic ADPs. Two
    independent reasons this harness cannot reach it are printed in the caveats
    at the end of every run, and the larger one is not the solvent model.
  * What IS diagnostic: R should fall substantially when the solvent term is
    switched on, the fall should be concentrated below ~5 A, and the fitted
    k_sol / b_sol should land near the conventional 0.35 e/A^3 and 46 A^2.
    Scales that fit well but land somewhere unphysical mean the mask is
    absorbing an error rather than modelling the solvent.

THE CALIBRATION TARGET IS MEASURED, NOT ASSUMED. The cutoff is calibrated so
that our threshold mask calls the same fraction of the cell solvent as gemmi's
geometric mask on the same model (--radii-set, default Refmac). The alternative
-- picking a target from the deposited solvent content, 42.44% for 7FPV -- is
WRONG in a way worth stating, because the deposited figure counts all solvent
while the bulk mask covers only the disordered remainder: the ordered waters
and ligands in the model occupy the rest. gemmi's mask on the same coordinates
already makes that subtraction, which is why it is the target.

FOR A MODEL WITHOUT HYDROGENS THERE IS NO CALIBRATION, since 2026-11-10:
--mask-level auto (the default) cuts the sphere field at 1.0 sphere overlaps,
where the calibration lands anyway (median 0.93-0.95): with no search and no
gemmi target it gains at 2.6-3.0 A, ties at 1.8-2.6 A and is within 0.001 of
calibration on every SW40 entry (slightly worse on more of them than better;
mask_level_rule has the numbers). Models with
hydrogens are still calibrated as below, because their extra spheres push
the right level to 1.1-1.5 with the hydrogen fraction; so are the atoms and
density sources, --target-occupancy, and a sphere field built with any of
--sphere-offset, --sphere-edge, --sphere-radii, --mask-blur or --taper-frac
away from its default. --mask-level calibrate calibrates
every model; a number cuts at that level. See docs/solvent-design.md, "A
fixed level instead of a calibration".

THE TARGET IS MEASURED ON A FIXED REFERENCE GRID (--target-spacing, default
0.7 A), not on the working grid, since 2026-11-10. gemmi's occupancy on a
grid steps with the voxel size where its shrink stencil gains neighbours, at
r_shrink/sqrt(k) (0.8 and 0.57 A for the Refmac radii), and drifts slowly in
between (7FPV cut to 3 A: 0.19 on 1 A voxels, 0.26-0.28 between 0.8 and
0.57 A, 0.29 below; 3I7M: 0.297 on d_min/3 voxels, 0.316 on d_min/4), so a
target measured on the working grid jumped whenever the grid crossed a step,
and the two grid changes of 2026-11-10 had worked almost entirely through
it: with the target held, a finer grid reproduces the coarser one's R to four
decimals, and a coarse d_min/3 grid with the target on a 0.7 A reference
reaches the same R as the capped grid. 0.7 A sits inside the Refmac plateau
(0.57, 0.8), the regime of mmtbx's own fixed 0.6 A step; the default was
tuned for --radii-set Refmac, and the run prints the actual reference spacing
and warns when it lands near a step.
Measured on 40 benchmark entries (binned R-free gap to mmtbx, median): at
2.6-3.0 A a 1.0 A reference gives +0.0270, 0.7 A +0.0105, 0.5 A +0.0115,
and the target is 0.88, 0.98 and 0.99 of mmtbx's own mask occupancy; at
1.8-2.2 A 1.0 A gives +0.0146 against +0.0035 for the other two. --target-
spacing same measures it on the working grid, as every run before did.

--mask gemmi runs the entire fit on gemmi's geometric mask instead of ours.
Comparing the two R-factors separates "the mask model is wrong" from
"everything downstream of the mask is wrong", which no single number can.
That mask is built on the WORKING grid and keeps the step dependence on the
voxel size that the reference-grid target removes from ours: on 1 A voxels
(d_min/3 at 3 A) gemmi's shrink stencil has no neighbours and the mask is
too large (7FPV at 3 A: occupancy 0.19, R-free 0.1895, against 0.26 and
0.1634 on 0.7 A voxels). Beyond ~2.1 A run --mask gemmi with --max-voxel 0.7;
the run warns when the voxels are no finer than the shrink radius.

--write-mtz PATH writes the scaled model as synthetic data: FP = |k_total
(F_protein + k_mask F_mask)| on the Fobs scale, SIGFP the observed sigmas,
FreeR_flag the free set as read, in the CCP4 convention (test set 0, work
bins 1..N-1: deposited CCP4 bins kept, status letters and 0/1 columns
rewritten, the same reflections free) and
PHIC the model's phase, on exactly the reflections the fit used, with the fit
in the MTZ history. The model is the binned row's (--also-scale-bins, log
bins by default since 2026-11-11), or the main fit's when that row does not
run. --mtz-complete adds FMODEL/PHIFMODEL on every reflection to d_min, the
unmeasured ones scaled with their nearest fitted reflection's bin, and
extends FreeR_flag to the reflections the file never flagged at the observed
fraction (extend_free_flags; 20 or fewer all go to the work set); a file
without a test set gets no FreeR_flag column at all; FP/SIGFP stay on the
measured reflections, because iotbx refuses an FP whose SIGFP is missing.
--mtz-components and --mtz-observed add F_protein/F_mask and FOBS/SIGFOBS.

--mask-source spheres|atoms|density is what the threshold is cut from. The
default since 2026-11-10 is SPHERES: every atom a flat-top soft sphere of
half its OPLS united-atom Lennard-Jones sigma (lunus.sf.mask_spheres), with
no further smoothing; on the benchmark it reaches gemmi's geometric mask at
2.6-3.0 A (median binned R-free gap to mmtbx -0.0003, against +0.0046 for
the atom-presence field and -0.0015 for gemmi) and beats the field at 1.8-
2.6 A (-0.0017 against +0.0017) and on SW40 (-0.0018 against -0.0005, 38 of
39 better); see docs/solvent-design.md, "Soft spheres of the Lennard-Jones
radius". The rest of this paragraph is the atoms source, the default for
part of the same day.
The default, atoms, is an ATOM-PRESENCE FIELD: every atom placed as a
unit-weight Gaussian at one B (--mask-source-b, default 20 A^2), from the
same coordinates, blur and symmetry expansion as the density, so the mask no
longer follows the electron count or the B of each atom, as a geometric mask
does not. On 20 entries at 2.6-3.0 A the binned R-free gap to mmtbx fell from
+0.0105 to +0.0046 (every entry better; gemmi's geometric mask on the same
grid reaches -0.0015), on 20 at 1.8-2.6 A from +0.0062 to +0.0017, on SW40
(0.9-2.5 A) from +0.0010 to -0.0005 (35 of 39 better), and on 7FPV cut to
3 A from 0.1921 to 0.1845 against mmtbx's 0.1811. The
alternatives measured on the same 20 low-resolution entries and rejected: a
wider or narrower --mask-blur (50 / 150 / 200 A^2: 0, +0.004, +0.009), a
differentiable closing after the threshold (every setting +0.003 to +0.011;
solvent_torch.close_mask, kept), hydrogens left out of the source (+0.001),
and a uniform-B source still weighted by electrons (B 10-60: -0.002 to
-0.006, half the gain); afterwards, widths scaled per element by the van der
Waals radius squared (+0.0044, within noise of the uniform +0.0046). F_protein is always the density's; the field is a
second placement with uniform-B kernels, see solvent_torch.mask_source_weights
and SolventModel.source_b. 'density' is every run before.

--mask-blur B smooths the mask's source with exp(-B s^2/4) BEFORE
thresholding, which is the threshold mask's analogue of the probe radius a
geometric mask is built with. It defaults to SolventModel's own value, so a plain run reports the
SHIPPED configuration; --mask-blur 0 disables it and reproduces what the model
did before this existed, which is where the unphysical scales come from (with
--protein-blur 0 as well: the placement blur below is the same smoothing, and
stays in the density the threshold sees).

THE GRID is gemmi's and xtraj's rate 1.5, voxels of d_min/3, the grid every
number recorded before 2026-11-10 was measured on. --rate overrides it with a
plain oversampling (spacing d_min/(2 rate)); --max-voxel caps the voxel edge
(spacing = min(d_min/3, cap)). Both were tried as defaults on 2026-11-10 (a
plain rate 2, then a 0.7 A cap) because a finer grid closed ~60% of the
low-resolution gap to mmtbx; that gain turned out to come entirely through
the calibration target, gemmi's mask occupancy, which STEPS with the voxel
size (see "THE TARGET IS MEASURED ON A FIXED REFERENCE GRID" below). With the
target on its own reference grid, the d_min/3 grid gives the same R as the
capped one in every resolution band (40 benchmark entries, 2026-11-10: no
entry moves by more than 0.0011; THE THRESHOLD MASK ONLY, see --mask gemmi
below) with 16% less torch time at 2.6-3.0 A (median per-entry ratio; 0.43x
the voxels), 7% less at 2.2-2.6 A and the same grid below 2.2 A, so the
default went back to d_min/3 the same day. F_protein
does not move with the grid once the protein blur is on, and the mask's
boundary on 1 A voxels at 3 A costs nothing measurable; mmtbx builds its mask
on a fixed ~0.6 A step. (A plain rate 2 had also cost ~0.0003 and 20-30%
time below 2.4 A.)

--protein-blur B places the PROTEIN density with B added to every atom and
removes it again from F_protein, exactly (exp(+B s^2/4) in reciprocal space).
The answer is unchanged in exact arithmetic; what changes is that an atom at
B = 1-2 A^2 is a Gaussian narrower than the voxel and aliases on the grid
(2YSK, 2P1N in the torchref1000 run, on the d_min/3 grid of the time:
|F_protein| wrong by R 0.11-0.19 against direct summation, R-free 0.04-0.05
above mmtbx). The default, auto,
is cell_utils.recommended_blur, the rule xtraj's torch engine uses: zero
whenever the model's sharpest atom already clears the grid's own target of
57 h^2 with h the voxel edge -- on the default d_min/3 grid 6 A^2 at d_min
1 A, 25 at 2 A and 57 at 3 A (28 at 3 A with --max-voxel 0.7, where the cap
binds; 32 on the d_min/4 grid of --rate 2) -- so a model is touched whenever
its sharpest atom is below that. The threshold
mask's own smoothing (--mask-blur) is a total and is reduced by the same
amount, so the mask the fit sees is the shipped one up to the aliasing this
removes; when the placement blur exceeds --mask-blur the mask is smoother
than asked for and the run says so.

--reference prints what this dataset can actually support, computed with cctbx
and mmtbx and no lunus code at all: the R that mmtbx's own bulk solvent reaches
on the same model, and the R that the isotropic-ADP approximation costs on its
own. Both are needed to read the numbers above them, and neither is obtainable
from inside this pipeline.
"""

import argparse
import contextlib
import io
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import torch


# ---------------------------------------------------------------- observations

def read_observations(path, d_min=None, crystal_symmetry=None, log=sys.stdout,
                      info=None):
    """
    Deposited structure factors -> (miller_array of F, free-set boolean mask).

    The reading path here is verified and fiddly enough to be worth spelling
    out, because most of its steps are easy to skip and none of them fails
    loudly if you do:

      1. A deposited file can hold SEVERAL data blocks. 1VME has six: the
         merged amplitudes refinement used, then five blocks of unmerged
         per-dataset intensities that carry a cell but no space group. The
         refinement data is the FIRST block holding an observation array;
         taking "the first intensity array" instead picked an unmerged
         dataset and fell over on its missing symmetry.
      2. The observations may be INTENSITIES (7FPV: anomalous, 142,372
         Bijvoet-separate, with sigmas) or AMPLITUDES (1VME); when a block,
         or an MTZ, holds both, the intensities are used. Intensities go
         through french_wilson(), which converts I to F properly, i.e.
         without simply discarding the negative intensities a weak reflection
         legitimately produces; it rejects a couple of hundred and is chatty
         on stdout even with log=None, hence the redirect. Amplitudes are
         used as they are. Either way merge_equivalents() then
         average_bijvoet_mates() reduces to one observation per unique
         reflection.
      3. A block without symmetry takes it from the MODEL (crystal_symmetry),
         which is what every refinement program does.
      4. The R-free set is encoded one of two ways in mmCIF (an MTZ carries
         an integer flag column under one of several names, found by
         iotbx's word list). _refln.status is a string
         per reflection, 'f' for free, 'o' for work, 'x' for unobserved.
         pdbx_r_free_flag is an integer whose convention varies: 7FPV uses
         CCP4 bins 0-19 with the test set in bin 0 (3,716 reflections, 5.0%,
         matching the 5.010% in its REMARK 3); other entries use 0/1 with
         either value meaning free. cctbx's own scorer decides which value is
         the test set, since reading the column as a boolean would silently
         put 95% of 7FPV in the test set.
    """
    from cctbx.array_family import flex
    from iotbx.reflection_file_reader import any_reflection_file
    from iotbx.reflection_file_utils import get_r_free_flags_scores

    arrays = any_reflection_file(file_name=path).as_miller_arrays(
        crystal_symmetry=crystal_symmetry, force_symmetry=True)

    def block(a):
        # mmCIF arrays are labelled "<data block>,_refln.<item>,..." and a
        # deposited file can hold several blocks (point 1 above). MTZ labels
        # carry no block, so every MTZ array counts as one block -- otherwise
        # an MTZ's free-flag column is never matched to its amplitudes.
        labels = a.info().label_string()
        return labels.split(",")[0] if "_refln" in labels else ""

    def is_flag_array(a):
        # iotbx's word list (free, test, cross, status, flag), applied
        # case-insensitively to every label but the mmCIF block name. Not
        # iotbx's looks_like_r_free_flags_info itself: that reads only the
        # LAST label of an array with at most two, and 1VME's status column
        # arrives as "r1vmesf,_refln.status,wavelength_id=1". Covers
        # _refln.status and _refln.pdbx_r_free_flag in mmCIF, and
        # FreeR_flag, R-free-flags, FREE and r_free_flags in MTZ.
        if a.is_xray_intensity_array() or a.is_xray_amplitude_array():
            return False                      # an observation column named F_test is not a flag
        if not (a.is_integer_array() or a.is_string_array()):
            return False                      # flags are integers or status letters
        labels = a.info().labels
        if block(a):
            labels = labels[1:]
        return any(word in label.lower() for label in labels
                   for word in ("free", "test", "cross", "status", "flag"))

    obs = [a for a in arrays
           if a.is_xray_intensity_array() or a.is_xray_amplitude_array()]
    if not obs:
        raise SystemExit("no intensity or amplitude array in %s" % path)
    first = block(obs[0])
    in_block = [a for a in obs if block(a) == first]
    intensities = [a for a in in_block if a.is_xray_intensity_array()]
    I_or_F = intensities[0] if intensities else in_block[0]
    if len({block(a) for a in obs}) > 1:
        print("%s holds %d data blocks; using the first with observations, "
              "%s" % (path, len({block(a) for a in arrays}), first), file=log)
    if I_or_F.space_group_info() is None:
        raise SystemExit("%s carries no space group; pass the model's" % path)

    # An AMPLITUDE deposited as exactly zero is excluded, because the
    # depositors' own refinements excluded it: it is either a placeholder for
    # an unobserved reflection (sigma 0, status '<' or 'h'; 2GOY: 24,664 of
    # 110,066, the whole 2.62-2.49 A shell) or a weak measurement truncated to
    # F = 0 that an F > 0 cutoff removed (sigma > 0; on 6KZ9 and 5KRF the
    # REMARK 3 reflection counts and R match ours only without them). Kept,
    # mmtbx finds a bin with nothing in it and raises "min() argument is an
    # empty array", and our R counts |0 - F_c|. They go BEFORE the merge, so
    # that a zero deposited for one Bijvoet mate or symmetry equivalent is
    # dropped rather than averaged into its partner. Intensities are left
    # alone: French-Wilson is what turns a weak or negative intensity into an
    # amplitude, and it applies its own rejection (I/sigma below -4, and
    # sigma <= 0 with I <= 0, see cctbx.french_wilson), which is what keeps
    # the information in weak and negative intensities.
    zero_note = ""
    if I_or_F.is_xray_amplitude_array():
        nonzero = I_or_F.data() != 0
        n_zero = nonzero.count(False)
        if n_zero:
            I_or_F = I_or_F.select(nonzero)
            zero_note = ("%s: %d amplitude(s) deposited as exactly zero dropped, "
                         "as the depositors' refinement did" % (path, n_zero))
            print(zero_note, file=log)

    with contextlib.redirect_stdout(io.StringIO()):
        F = I_or_F.merge_equivalents().array().average_bijvoet_mates()
        if F.is_xray_intensity_array():
            F = F.french_wilson(log=None)
    if d_min is not None:
        F = F.resolution_filter(d_min=d_min)

    flags = [a for a in arrays if block(a) == first and is_flag_array(a)]
    free_mask = np.zeros(F.size(), dtype=bool)
    as_read = None
    if flags:
        raw = flags[0]
        status = isinstance(raw.data(), flex.std_string)
        if status:
            observed = flex.bool([v != "x" for v in raw.data()])
            free = raw.customized_copy(
                data=flex.bool([v == "f" for v in raw.data()]))
            free = free.select(observed)
            test_value = None
        else:
            test_value = get_r_free_flags_scores(
                miller_arrays=[raw], test_flag_value=None).test_flag_values[0]
            free = raw.customized_copy(data=(raw.data() == test_value))
            as_read = raw
        free = free.merge_equivalents().array().average_bijvoet_mates()
        # The flags are WRITTEN in one convention, CCP4's: test set 0, work
        # bins 1..N-1, which CCP4 programs (FREE 0 by default) and iotbx's
        # scorer read alike. Deposited CCP4 bins with 0 the test set are
        # kept as they are; anything else -- status letters, a 0/1 column
        # with 1 free (which CCP4 reads INVERTED), any other test value --
        # is rewritten from the free set read above, so the same reflections
        # stay free. Which reflections are free is unchanged either way.
        deposited_ccp4 = (not status and test_value == 0
                          and flex.max(raw.data()) >= 4)    # r_free_utils.looks_like_ccp4_flags
        if deposited_ccp4:
            # Kept exactly when equivalents agree, which they do in deposited
            # mmCIF (I+ and I- on one row) and in CCP4 freerflag output (one
            # flag per unique hkl). Where two equivalents carry different
            # WORK bins (3 and 7 on two Friedel mates), the exact merge
            # refuses; replacing them with one fixed bin would pile every
            # such reflection into it, so the bins are redrawn from the free
            # set below instead. A test/work mismatch has already raised in
            # the merge of free above, so the free set is never in question.
            from libtbx.utils import Sorry
            try:
                as_read = as_read.merge_equivalents().array()
                as_read = as_read.as_non_anomalous_array().merge_equivalents().array()
            except (RuntimeError, Sorry):    # Sorry for a Friedel mismatch
                deposited_ccp4 = False
        if not deposited_ccp4:
            # The work bins are drawn at random, seeded, with Python's
            # generator handed back as it was -- and drawn AFTER the merge:
            # drawn per row, two Friedel mates on separate rows almost never
            # agree, and merging them would send nearly every work
            # reflection to bin 1.
            from cctbx.r_free_utils import export_r_free_flags_for_ccp4
            state = random.getstate()
            random.seed(0)
            try:
                ccp4 = export_r_free_flags_for_ccp4(free.data(), True)
            finally:
                random.setstate(state)
            as_read = free.customized_copy(data=ccp4)
            test_value = 0
        F, free = F.common_sets(free)
        free_mask = np.array(free.data(), dtype=bool)
    else:
        print("no R-free flags found: R-free will not be reported", file=log)

    F, free_mask, note = drop_unfittable_reflections(F, free_mask, log=log, name=path)
    if info is not None:
        # For callers that silence log but still need to say what was
        # dropped (mmtbx_fit, for a run of that method alone).
        info["dropped_note"] = "\n".join(n for n in (zero_note, note) if n)
        # The free set as integer CCP4 flags (the file's own bins where it
        # deposited CCP4 bins, otherwise rewritten above), in F's order, with
        # the test value (always 0); None without flags. --write-mtz writes
        # these rather than the boolean mask.
        # (common_set returns as_read's entries in F's order; F was cut from
        # a common set with it, so every reflection of F has one)
        # No test reflection among the fitted ones means no free set to
        # write: a column of work values would read as all-free in CCP4.
        # free_flags_file keeps every reflection the file flags, fitted or
        # not, so that --mtz-complete draws flags only for reflections the
        # file never flagged (French-Wilson rejections and dropped zeros
        # keep the flags they were deposited with).
        has_test = as_read is not None and bool(free_mask.any())
        info["free_flags"] = (as_read.common_set(F), test_value) if has_test else None
        info["free_flags_file"] = as_read if has_test else None
    return F, free_mask


def drop_unfittable_reflections(F, free_mask, log=sys.stdout, name="observations"):
    """Remove from a miller array (and the matching free-set mask) the
    reflections no scaling can fit: (0,0,0), which carries no measurable
    amplitude and would sit alone in the lowest-resolution shell with an
    infinite d, and the systematically absent reflections of the space
    group, on which a model with that symmetry has F exactly zero. Deposited
    files do carry those (2ZFX: 42 odd 0k0 in P 21 21 2 at a median |F| of
    23 against 184; 1VME: 26; 23 of the 1,000 torchref entries, at most 0.3%
    of an entry's reflections), refinement programs drop them on input, and
    mmtbx's anisotropic scaler asserts on the zero (MMTBX_ASSERT(fm_abs > 0),
    the 2ZFX and 8P0O failures of the torchref1000 run, 2026-11-10). Both
    methods read through here, so both see the same reflections. (With
    --no-expand-symmetry and a model that does not have its declared
    symmetry the model's F there is not exactly zero; dropping them is
    still what every program does.)

    Amplitudes deposited as exactly zero are a different matter -- excluded
    because the depositors' refinements excluded them -- and
    read_observations removes them from the raw array before merging, where
    a zero mate cannot be averaged into a real one.

    Returns (F, free_mask, note): note is the printed line, or "" when
    nothing was absent. A miller array's select() needs a flex.bool, not a
    Python list -- the list raised ArgumentError on every file holding
    (0,0,0) (reported on 3DHK and 2ZG0 in the torchref set)."""
    from cctbx.array_family import flex

    origin = (np.array(F.indices()).reshape(-1, 3) == 0).all(axis=1)
    absent = np.array(F.sys_absent_flags().data(), dtype=bool)
    n_absent = int(absent.sum())
    note = ""
    if n_absent:
        note = ("%s: %d systematically absent reflection(s) of %s dropped, as "
                "refinement programs do" % (name, n_absent, F.space_group_info()))
        print(note, file=log)
    keep = ~(origin | absent)
    if not keep.all():
        F = F.select(flex.bool(keep))
        free_mask = free_mask[keep]
    return F, free_mask, note


# ------------------------------------------------------------------- the model

def read_model(path, log=sys.stdout, special_position_weights=True, select="all"):
    """Coordinates, isotropic-equivalent B, elements and symmetry weights, with
    the crystal symmetry taken from the file's own CRYST1 rather than the
    command line -- this tool is pointed at a deposited structure, which knows
    its own cell. The reader is used with its defaults: cctbx reads an atom
    with occupancy below 1 as a general position, which is what makes the
    half occupancy stated for an atom on a 2-fold in pre-cctbx depositions
    come out as one full atom. See model_arrays for what occ holds and what
    special_position_weights=False does.

    With special_position_weights=False the reader's snapping is off as well
    (min_distance_sym_equiv=0): a full-occupancy atom near a symmetry element
    stays where the file put it, with its ANISOU as deposited rather than
    symmetrized, and counts as a general position, so a complete cell is
    taken exactly as given, every copy its own. How many atoms the default
    reader would have moved is printed, so the flag used by mistake on one
    asymmetric unit is visible.

    select is an iotbx atom selection ("not element H"); only the atoms it
    keeps are returned, see select_atoms. "all" keeps everything."""
    from iotbx.pdb import hierarchy

    pdb_in = hierarchy.input(file_name=path, sort_atoms=False)
    if special_position_weights:
        xrs = pdb_in.input.xray_structure_simple()
    else:
        xrs = pdb_in.input.xray_structure_simple(min_distance_sym_equiv=0.0)
        # A diagnostic only: the default read can refuse a complete cell
        # whose symmetry copies crowd a symmetry element ("special position
        # not well defined"), which is the kind of model this path is for,
        # so its failure must not take the read with it.
        try:
            n_would = special_position_count(pdb_in.input.xray_structure_simple())
        except RuntimeError as e:
            print("%s: cctbx's default read, which moves atoms onto special "
                  "positions, fails on this file (%s); all atoms left where "
                  "the file has them (special_position_weights=False)"
                  % (path, str(e).strip()), file=log)
        else:
            print("%s: %d atom(s) cctbx would put on special positions left "
                  "where the file has them, as general positions "
                  "(special_position_weights=False)" % (path, n_would), file=log)
    xrs = select_atoms(pdb_in, xrs, select, log, name=path)
    return (xrs,) + model_arrays(xrs, log, name=path,
                                 special_position_weights=special_position_weights)


def select_atoms(pdb_in, xrs, select, log=sys.stdout, name="model"):
    """xrs restricted to the atoms an iotbx selection keeps. "all" (or None)
    returns it unchanged.

    The mask is taken in FILE order (atom_order.selection_in_file_order),
    which is the order xray_structure_simple's scatterers are in. The
    hierarchy's own order groups altlocs within a residue, so on a file whose
    altlocs are interleaved atom by atom a mask from atom_selection_cache()
    applied straight to the scatterers lands on the wrong atoms. xrs must be
    the structure built from this same pdb_in, with no atoms dropped yet."""
    if select in (None, "all"):
        return xrs
    from scitbx.array_family import flex

    from lunus.sf.atom_order import selection_in_file_order

    keep = selection_in_file_order(pdb_in, select)
    if len(keep) != xrs.scatterers().size():
        raise ValueError("%s: %d atoms in the file, %d scatterers"
                         % (name, len(keep), xrs.scatterers().size()))
    if not keep.any():
        raise SystemExit("--select %r keeps no atoms of %s" % (select, name))
    print("%s: %d of %d atoms kept (--select %r)"
          % (name, keep.sum(), len(keep), select), file=log)
    return xrs.select(flex.bool(keep.tolist()))


def special_position_count(xrs):
    """Scatterers whose site symmetry fixes them: multiplicity below order_z.
    One with no site symmetry applied (multiplicity 0) is not counted."""
    order_z = xrs.space_group().order_z()
    return sum(1 for sc in xrs.scatterers() if 0 < sc.multiplicity() < order_z)


def use_special_position_weights(no_special_position_weights, no_expand_symmetry):
    """The run() rule: cctbx's symmetry weights only when the model is one
    asymmetric unit that the grid expansion completes. A model placed once
    (--no-expand-symmetry) or expanded as a fold (--no-special-position-weights)
    keeps its deposited occupancies."""
    return not (no_special_position_weights or no_expand_symmetry)


def model_arrays(xrs, log=sys.stdout, name="model", special_position_weights=True):
    """Per-atom arrays from an xray.structure, for build_density: elements,
    isotropic-equivalent B, occ, fractional coordinates, the ANISOU count,
    U_cart and the anisotropic mask. Converts xrs to isotropic in place.

    occ is NOT the deposited occupancy. It is cctbx's scatterer weight,
    occupancy * multiplicity / order_z, the weight cctbx's own direct
    summation sums with over every operation of the space group. build_density
    places each atom once and then symmetry-expands the grid under every
    operation, so an atom on a special position lands on itself
    order_z / multiplicity times, and the weight is what makes that sum count
    it once. For an atom on a general position it equals the occupancy.

    special_position_weights=False returns the deposited occupancies instead.
    That is the right input when the model is NOT expanded (a complete cell
    placed once, build_density(..., expand_symmetry=False)), and when a
    complete cell with its space group in CRYST1 is expanded on purpose as
    a fold, every copy of the asymmetric unit being its own: there each atom
    must keep its own occupancy, and the expansion scales every atom by the
    same order_z."""
    from cctbx import adptbx

    # BEFORE convert_to_isotropic, which discards it. u_star is in the
    # fractional reciprocal basis; the kernel measures displacements in
    # Cartesian angstroms, and the two coincide only for a cubic cell with
    # a = 1 -- so this conversion is invisible on a toy and wrong on a crystal.
    is_aniso = np.array(xrs.use_u_aniso(), dtype=bool)
    unit_cell = xrs.unit_cell()
    u_cart = np.zeros((xrs.scatterers().size(), 3, 3))
    for i, sc in enumerate(xrs.scatterers()):
        if is_aniso[i]:
            c = adptbx.u_star_as_u_cart(unit_cell, sc.u_star)
            u_cart[i] = [[c[0], c[3], c[4]],
                         [c[3], c[1], c[5]],
                         [c[4], c[5], c[2]]]

    # A deposited ANISOU is not always positive definite (7P6M has 7 atoms
    # with a slightly negative eigenvalue). The real-space kernel cannot
    # place a Gaussian that grows along one axis, and raises; cctbx just
    # evaluates the reciprocal-space Debye-Waller factor. Clamp the offending
    # eigenvalues to a small floor -- a change far below what the data can
    # see -- and say how many atoms it touched.
    U_FLOOR = 1e-3  # A^2, B ~ 0.08
    if is_aniso.any():
        w, v = np.linalg.eigh(u_cart[is_aniso])
        bad = (w < U_FLOOR).any(axis=1)
        if bad.any():
            w_min = float(w[bad].min())
            w = np.maximum(w, U_FLOOR)
            fixed = np.einsum("nij,nj,nkj->nik", v, w, v)
            idx = np.flatnonzero(is_aniso)[bad]
            u_cart[idx] = fixed[bad]
            print("%d ANISOU tensor(s) not positive definite; eigenvalues "
                  "clamped to %.0e A^2 (lowest was %.4f)"
                  % (int(bad.sum()), U_FLOOR, w_min), file=log)

    n_aniso = int(is_aniso.sum())
    xrs.convert_to_isotropic()

    elements = [s.scattering_type for s in xrs.scatterers()]
    b_per_atom = np.array(xrs.extract_u_iso_or_u_equiv()) * (8.0 * np.pi ** 2)
    raw_occ = np.array(xrs.scatterers().extract_occupancies())
    n_special = special_position_count(xrs)
    if special_position_weights:
        # multiplicity is 0 until site symmetry has been applied
        # (add_scatterer and xray_structure_simple do; appending to
        # scatterers() does not), and weight() is then 0 for every atom: an
        # empty density, not an error. The deposited occupancies below do
        # not depend on it.
        n_unset = sum(1 for sc in xrs.scatterers() if sc.multiplicity() == 0)
        if n_unset:
            raise ValueError("%s: %d scatterer(s) have no site symmetry applied "
                             "(multiplicity 0), so their weight is undefined"
                             % (name, n_unset))
        occ = np.array([sc.weight() for sc in xrs.scatterers()])
    else:
        occ = raw_occ.copy()
        if n_special:
            print("%s: deposited occupancies kept for the %d atom(s) on special "
                  "positions (special_position_weights=False)" % (name, n_special),
                  file=log)
    frac = np.array(xrs.sites_frac()).reshape(-1, 3)
    # Isotropic scatterers get u_iso * I, which the anisotropic kernel
    # reproduces exactly -- but is_aniso keeps them on the fast path.
    u_cart[~is_aniso] = (np.array(xrs.extract_u_iso_or_u_equiv())[~is_aniso]
                         [:, None, None] * np.eye(3))

    print("%s: %d atoms, B %.1f-%.1f, occupancy %.2f-%.2f, %d on special "
          "positions, %d with ANISOU"
          % (name, len(elements), b_per_atom.min(), b_per_atom.max(),
             raw_occ.min(), raw_occ.max(), n_special, n_aniso), file=log)
    return elements, b_per_atom, occ, frac, n_aniso, u_cart, is_aniso


def neutral_type(sym):
    """cctbx's scattering type without its charge suffix: 'O1-' -> 'O',
    'Zn2+' -> 'Zn', 'C' -> 'C'."""
    import re

    return re.sub(r"[0-9]*[+-]$", "", sym)


def scattering_table(present, log=sys.stdout):
    """IT92 coefficients for the scattering types cctbx actually assigned.

    A deposited model is read with cctbx, which types a charged atom by its
    ion -- 'O1-' for a carboxylate oxygen, 'Zn2+', 'Cl1-' -- and the default
    IT92_COEFFS table knows only neutral elements, so 7TX0 raised KeyError
    on 'O1-'. cctbx's own IT92 table carries the ions, and it is what
    mmtbx.f_model scatters off, so taking the coefficients from there keeps
    the two methods on identical form factors. A type cctbx has no entry
    for ('N1+' is one) falls back to the neutral atom, which is what a
    refinement program would do too, and says so.
    """
    from lunus.sf.elements import it92_coefficients

    table, fallback = {}, []
    for sym in present:
        try:
            table[sym] = it92_coefficients([sym], source="cctbx")[sym]
        except (RuntimeError, ValueError):
            neutral = neutral_type(sym)
            table[sym] = it92_coefficients([neutral], source="cctbx")[neutral]
            fallback.append("%s -> %s" % (sym, neutral))
    if fallback:
        print("no IT92 entry for charged type(s); using the neutral atom: %s"
              % ", ".join(fallback), file=log)
    return table


# ------------------------------------------------------------------- timing

def device_sync(device):
    """Wait for the device's queued work, so that a wall-clock reading after
    it measures the work and not the time to enqueue it. No-op on CPU."""
    d = str(device)
    if d.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(d)
    elif d.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


@contextlib.contextmanager
def timed(timing, phase, device="cpu"):
    """Add the wall time of the block to timing[phase] (seconds), with a
    device synchronisation at both ends. timing is a plain dict; None
    disables the timer, so library functions can take an optional one."""
    if timing is None:
        yield
        return
    device_sync(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        device_sync(device)
        timing[phase] = timing.get(phase, 0.0) + time.perf_counter() - t0


def print_timing(timing, log=sys.stdout):
    """The per-phase table run() prints: every phase that was timed, in the
    order it was first entered, then the remainder and the total."""
    total = timing.get("total", 0.0)
    accounted = sum(v for k, v in timing.items() if k != "total")
    print("\ntiming (s, wall; device work waited for at each boundary)", file=log)
    for phase, seconds in timing.items():
        if phase != "total":
            print("  %-26s %8.2f" % (phase, seconds), file=log)
    print("  %-26s %8.2f" % ("other", max(0.0, total - accounted)), file=log)
    print("  %-26s %8.2f" % ("total", total), file=log)


def build_density(xrs, elements, b_per_atom, occ, frac, grid, device, dtype,
                  expand_symmetry, log=sys.stdout, u_cart=None,
                  is_aniso=None, cutoff=0.01, timing=None, blur=0.0):
    """Place the density, then symmetry-expand on the GRID.

    blur (A^2) is added to every atom's B when it is placed; the caller takes
    it out of F again with compute_fcalc(..., blur=blur). See protein_blur().

    expand_symmetry is True here and False for a trajectory, and the difference
    is a property of the input model rather than a preference: a deposited PDB
    is one asymmetric unit whose other copies genuinely need generating, while
    an MD box already contains them. Getting it backwards does not raise --
    it symmetry-averages a model that was already complete. occ is taken as
    given: with the expansion it must be the symmetry weight read_model
    returns, occupancy * multiplicity / order_z, which is what lets the
    expansion count an atom on a special position once; without it, or for
    a complete cell expanded as a fold, the deposited occupancy
    (read_model(..., special_position_weights=False)).
    """
    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.density_torch import compute_density
    from lunus.sf.kernel_torch import (build_atom_kernels_aniso_torch,
                                       build_atom_kernels_torch)
    from lunus.sf.symmetry_torch import build_grid_ops_from_cctbx, symmetrize_sum

    cell = xrs.unit_cell().parameters()
    M_np = orth_matrix(*cell)
    present = sorted(set(elements))
    coeffs = scattering_table(present, log)
    atom_A, atom_lam, offsets, atom_r, taper_w, e2i = build_atom_kernels_torch(
        elements, present, coeffs, b_per_atom, blur, grid, M_np,
        cutoff=cutoff, device=device, dtype=dtype)

    atom_L6 = aniso_mask = None
    if u_cart is not None:
        # The anisotropic kernels REPLACE atom_A and the radii for every atom,
        # including the isotropic ones -- for which U = u_iso*I gives exactly
        # the isotropic values back, so the two paths stay consistent. Only
        # atom_lam and the dispatch mask distinguish them.
        atom_A, atom_L6, offsets, atom_r, taper_w, _ = \
            build_atom_kernels_aniso_torch(
                elements, present, coeffs, u_cart, blur, grid, M_np,
                cutoff=cutoff, device=device, dtype=dtype)
        aniso_mask = torch.tensor(is_aniso, device=device)
        print("anisotropic ADPs: %d of %d atoms on the tensor kernel"
              % (int(is_aniso.sum()), len(elements)), file=log)

    frac_t = torch.tensor(frac, dtype=dtype, device=device)
    element_idx = torch.tensor([e2i[e] for e in elements], dtype=torch.long,
                               device=device)
    occ_t = torch.tensor(occ, dtype=dtype, device=device)
    orth = torch.tensor(M_np, dtype=dtype, device=device)

    with timed(timing, "density", device):
        density = compute_density(frac_t, element_idx, occ_t, atom_A, atom_lam,
                                  offsets, atom_r, grid, orth, taper_w,
                                  compile_core=False, atom_L6=atom_L6,
                                  aniso_mask=aniso_mask)
    if expand_symmetry:
        with timed(timing, "symmetry expansion", device):
            grid_ops = build_grid_ops_from_cctbx(xrs.space_group(), grid)
            density = symmetrize_sum(density, grid_ops)
        print("symmetry-expanded on the grid: %d operations beyond the identity"
              % len(grid_ops), file=log)
    return density, orth, float(np.linalg.det(M_np))


MAX_VOXEL_DEFAULT = None  # A; no cap: plain d_min/3. See grid_rate() and the module docstring


def grid_rate(d_min, rate=None, max_voxel=MAX_VOXEL_DEFAULT):
    """The oversampling rate the grid is built with, and a line saying why.

    The default is gemmi's and xtraj's rate 1.5, voxels of d_min/3, the grid
    every number recorded before 2026-11-10 was measured on. max_voxel, when
    given, caps the voxel edge: spacing = min(d_min/3, max_voxel). Expressed
    as a rate so that grid_shape_for_resolution, and every caller that reads
    --rate, see one quantity. An explicit rate overrides both."""
    if rate is not None:
        if rate <= 0.0:
            raise ValueError("--rate must be > 0, got %g" % rate)
        return float(rate), "--rate %g: voxels of d_min/(2 rate) = %.3f A" % (rate, d_min / (2 * rate))
    natural = d_min / 3.0
    if max_voxel is None:
        return 1.5, "voxels of d_min/3 = %.3f A (rate 1.5, the default)" % natural
    if max_voxel <= 0.0:
        raise ValueError("--max-voxel must be > 0 A, got %g" % max_voxel)
    if natural <= max_voxel * (1.0 + 1e-9):      # at the cap counts as under it
        return 1.5, ("voxels of d_min/3 = %.3f A (rate 1.5; finer than the "
                     "--max-voxel cap of %g A)" % (natural, max_voxel))
    rate = d_min / (2.0 * max_voxel)
    return rate, ("voxels capped at --max-voxel %g A (d_min/3 would be %.3f A; "
                  "rate %.2f)" % (max_voxel, natural, rate))


def sharpest_b(b_per_atom, u_cart=None, is_aniso=None, occ=None):
    """The smallest B the density will be placed with: the isotropic B of the
    sharpest isotropic atom, or the smallest principal B of an anisotropic
    one, which is what sets how finely the grid must sample it. Atoms with
    zero occupancy place no density and do not count (unless every atom is
    such, when all are kept rather than returning nothing)."""
    # b becomes the B each atom is PLACED with: its isotropic B, or for an
    # atom on the tensor kernel its smallest principal B.
    b = np.array(b_per_atom, dtype=float)
    if u_cart is not None and is_aniso is not None and np.any(is_aniso):
        is_aniso = np.asarray(is_aniso, dtype=bool)
        b[is_aniso] = (np.linalg.eigvalsh(np.asarray(u_cart)[is_aniso]).min(axis=1)
                       * (8.0 * np.pi ** 2))
    keep = np.ones(b.shape, dtype=bool)
    if occ is not None:
        scatters = np.asarray(occ, dtype=float) > 0.0
        if scatters.any():
            keep = scatters
    return float(b[keep].min())


def positive_float_arg(option):
    """argparse type: a float > 0, named in the error."""
    def parse(text):
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError("%s expects a number, got %r" % (option, text))
        if not math.isfinite(value) or value <= 0.0:
            raise argparse.ArgumentTypeError("%s must be a finite number > 0, got %g"
                                             % (option, value))
        return value
    return parse


def keyword_or_number_arg(keyword, option, unit, allow_zero, keywords=None):
    """An argparse type: the keyword (any case, returned lower-case) or a
    finite number in unit, >= 0 when allow_zero and > 0 otherwise. keywords,
    when given, is the full tuple of accepted keywords (keyword among them)."""
    keywords = keywords or (keyword,)

    def parse(text):
        if text.lower() in keywords:
            return text.lower()
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "expected %s or a number in %s, got %r"
                % (" or ".join(repr(k) for k in keywords), unit, text))
        if not math.isfinite(value):
            raise argparse.ArgumentTypeError(
                "%s must be a finite number in %s, got %r" % (option, unit, text))
        if value < 0.0 or (value == 0.0 and not allow_zero):
            raise argparse.ArgumentTypeError(
                "%s must be %s %s" % (option, ">= 0" if allow_zero else "> 0", unit))
        return value
    return parse


# --protein-blur: "auto", or B >= 0 in A^2 (0 disables it)
protein_blur_arg = keyword_or_number_arg("auto", "--protein-blur", "A^2", allow_zero=True)

# --mask-level: "auto", "calibrate" (to gemmi's occupancy), or a level > 0 in
# sphere overlaps. auto is MASK_LEVEL_FIXED for a model without hydrogens and
# calibration for one with them (mask_level_rule says why).
MASK_LEVEL_DEFAULT = "auto"
MASK_LEVEL_FIXED = 1.0


mask_level_arg = keyword_or_number_arg("calibrate", "--mask-level", "sphere overlaps",
                                       allow_zero=False, keywords=("auto", "calibrate"))


def mask_level_rule(setting, mask, mask_source, elements, target_occupancy,
                    nondefault_field=()):
    """--mask-level resolved: (level or None to calibrate, the reason).

    A fixed level is in sphere overlaps, so it only means anything for the
    sphere field. 'auto' cuts that field at MASK_LEVEL_FIXED when the model
    has no hydrogens and calibrates it to gemmi's occupancy when it has any.
    Measured 2026-11-10 against calibration on every entry, the binned
    R-free gap to mmtbx, medians: 1.0 is -0.0015 / -0.0021 / -0.0014 on the
    models without hydrogens of the 2.6-3.0 A, 1.8-2.6 A and SW40 sets
    (calibrated -0.0003 / -0.0022 / -0.0016). Per entry that is a gain at
    low resolution (13 better, 3 worse), a wash at 1.8-2.6 A (9 / 7) and a
    slight loss on SW40 (7 / 14, median +0.0001, none by more than 0.001).
    0.9 is the best level on SW40 (-0.0017) but loses at low resolution
    (6 entries worse by more than 0.001 against 1); 1.1 is worse on SW40
    and at 1.8-2.6 A. The hydrogen test runs on the model as selected, so
    --select "not element H" makes a model with hydrogens one without. A
    model with explicit hydrogens has its sp3 carbons bare plus a sphere per
    hydrogen, which packs the field deeper, and its calibrated level runs
    from 1.12 to 1.46 with the hydrogen fraction; at 1.0 the four such
    entries at 1.8-3.0 A lost 0.006-0.011. Typing them united-atom with the
    hydrogens left out brings their level back to ~0.93 and recovers the
    low-resolution loss, but costs SW40's hydrogen entries 0.0009, so they
    keep the calibration. An explicit target (--target-occupancy) is a
    request to calibrate, and auto honours it. So is any sphere-field setting
    away from the one the level was measured with (nondefault_field, the
    options' names): an offset, edge, radius table, extra smoothing or taper
    moves where the field crosses 1.0, which calibration compensates for and
    a fixed level does not."""
    from lunus.sf.mask_spheres import is_hydrogen

    if setting == "calibrate":
        return None, "--mask-level calibrate"
    if setting != "auto":
        return float(setting), "--mask-level %g" % setting
    if mask == "gemmi" or mask_source != "spheres":
        return None, "auto: a fixed level is only defined for the sphere field"
    if target_occupancy is not None:
        return None, "auto: --target-occupancy given"
    if nondefault_field:
        return None, ("auto: %s differ from the settings the fixed level was "
                      "measured with, so it is calibrated" % ", ".join(nondefault_field))
    n_h = sum(is_hydrogen(e) for e in elements)
    if n_h:
        return None, ("auto: the model has %d hydrogens, whose spheres move the "
                      "level, so it is calibrated" % n_h)
    return MASK_LEVEL_FIXED, "auto: no hydrogens in the model"


def protein_blur(setting, b_min, grid, orth_np, d_min):
    """--protein-blur resolved to a number: 'auto' is recommended_blur's
    rule (zero when b_min already clears the grid's target), anything else
    is taken as B in A^2 and must not be negative: a negative blur places
    the atoms sharper than the model says and sharpens F_protein's error."""
    from lunus.sf.cell_utils import recommended_blur

    if str(setting).lower() == "auto":
        return recommended_blur(b_min, grid, orth_np, d_min=d_min)
    value = float(setting)
    if value < 0.0:
        raise ValueError("protein blur must be >= 0 A^2, got %g" % value)
    return value


def blur_density(density, orth, blur_b):
    """density * exp(-B s^2 / 4), applied in reciprocal space on the grid.

    THE THRESHOLD MASK'S MISSING PROBE RADIUS. A geometric mask is built from
    vdW radius + probe and then shrunk, and those two steps are what make its
    boundary smooth and simply connected. A density threshold has no equivalent:
    it follows the density's own boundary, including every crevice between
    neighbouring atoms, so the mask carries far more high-resolution structure
    than a geometric one -- measured on 7FPV, 1.5x the |F_mask| in every shell
    beyond 2 A, essentially uncorrelated with gemmi's.

    Smoothing the density first is the cheapest thing that plays the probe's
    role, and it is exact and differentiable on a periodic grid: one FFT pair.
    B is in the same A^2 convention as everything else here, so the Gaussian
    sigma is sqrt(B / (8 pi^2)) -- B = 100 is sigma = 1.13 A, close to a water
    probe radius, which is where the fitted scales come right.
    """
    if blur_b == 0.0:
        return density
    Nu, Nv, Nw = density.shape
    # s^2 for every grid frequency, through the full reciprocal metric rather
    # than 1/a^2 + 1/b^2 + 1/c^2 -- those agree only for an orthogonal cell,
    # and silently do not for a monoclinic or triclinic one.
    Mstar = torch.linalg.inv(orth).T
    fu = torch.fft.fftfreq(Nu, d=1.0 / Nu, device=density.device)
    fv = torch.fft.fftfreq(Nv, d=1.0 / Nv, device=density.device)
    fw = torch.fft.fftfreq(Nw, d=1.0 / Nw, device=density.device)
    H = torch.stack(torch.meshgrid(fu, fv, fw, indexing="ij"), dim=-1)
    s_cart = H.to(Mstar.dtype) @ Mstar.T
    s2 = (s_cart * s_cart).sum(-1).to(density.dtype)
    G = torch.fft.fftn(density) * torch.exp(-0.25 * blur_b * s2)
    return torch.fft.ifftn(G).real


def gemmi_mask_occupancy(pdb_path, grid, radii_set):
    """The occupancy of gemmi's GEOMETRIC mask on the same model and grid.

    This is the calibration target. It is a measurement of the same quantity
    our threshold mask produces -- the fraction of the cell left to the bulk
    once the model's own atoms, ordered waters and ligands included, are
    accounted for -- computed by the conventional method rather than ours.
    """
    arr = np.array(gemmi_mask_grid(pdb_path, radii_set, size=grid), copy=False)
    return float(arr.mean()), arr


def gemmi_mask_grid(pdb_path, radii_set, size=None, spacing=None):
    """gemmi's geometric mask on the model, as a gemmi FloatGrid of the given
    size, or of the given spacing (rounded up to a symmetry-compatible size
    by gemmi) when no size is given."""
    import gemmi

    if size is None and spacing is None:
        raise ValueError("gemmi_mask_grid needs a grid size or a spacing")
    st = gemmi.read_structure(pdb_path)
    st.setup_entities()
    g = gemmi.FloatGrid()
    if size is not None:
        g.set_size(*size)
    g.set_unit_cell(st.cell)
    g.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    if size is None:
        g.set_size_from_spacing(float(spacing), gemmi.GridSizeRounding.Up)
    masker = gemmi.SolventMasker(getattr(gemmi.AtomicRadiiSet, radii_set))
    masker.put_mask_on_float_grid(g, st[0])
    return g


TARGET_SPACING_DEFAULT = 0.7   # A; see gemmi_mask_occupancy_at_spacing()


# --target-spacing: "same" (the working grid), or a spacing in A > 0
target_spacing_arg = keyword_or_number_arg("same", "--target-spacing", "A", allow_zero=False)


def gemmi_mask_occupancy_at_spacing(pdb_path, spacing, radii_set):
    """gemmi's geometric-mask occupancy on a REFERENCE grid of the given
    spacing (rounded up to a symmetry-compatible size by gemmi), independent
    of the grid the fit runs on.

    THE TARGET MUST NOT MOVE WITH THE WORKING GRID. gemmi's occupancy on a
    grid depends on the voxel size, and not smoothly: it STEPS where the
    shrink stencil gains neighbours, at voxel edges of r_shrink/sqrt(k)
    (Refmac radii: r_shrink 0.8 A, so steps at 0.8 and 0.57 A), and falls
    slowly between the steps. On 7FPV it is 0.19 on voxels coarser than
    0.8 A, 0.24-0.28 between 0.8 and 0.57, 0.29 below (3I7M: 0.297 on
    d_min/3 voxels, 0.316 on d_min/4). A threshold calibrated to the working
    grid's value therefore jumped whenever the grid crossed a step, and the
    grid changes of 2026-11-10 acted through it: holding the target, a
    finer grid reproduced the coarser one's R to four decimals. Measuring
    the target on one fixed spacing makes the calibration a property of the
    model, the radii set, and the cell and space group (through gemmi's
    rounding of the spacing to a symmetry-compatible size) alone. The 0.7 A
    default sits inside the Refmac plateau (0.57, 0.8), the regime of
    mmtbx's own fixed 0.6 A step with its 0.9 A shrink; the plateaus differ
    for the other radii sets, and run() says when the actual spacing lands
    near a step.
    Returns (occupancy, grid shape)."""
    g = gemmi_mask_grid(pdb_path, radii_set, spacing=spacing)
    return float(np.array(g, copy=False).mean()), (g.nu, g.nv, g.nw)


# gemmi's shrink radius per AtomicRadiiSet (gemmi/solmask.hpp), which sets
# where its mask occupancy steps with the voxel size: at r_shrink/sqrt(k).
SHRINK_RADIUS = {"Refmac": 0.8, "Cctbx": 0.9, "VanDerWaals": 1.1}


def shrink_step_distance(orth_np, grid, r_shrink, margin=0.05):
    """The voxel edge or face diagonal that lies within `margin` of gemmi's
    shrink radius on this grid, or None.

    gemmi's occupancy steps when its shrink stencil gains a neighbour, i.e.
    when a voxel EDGE, or the FACE DIAGONAL of two edges, crosses r_shrink
    (measured on 7FPV: edges .55/.56/.69 A with a face diagonal of 0.784 A
    jump by 0.013 against .68/.56/.69 A, whose diagonals all exceed 0.8).
    The lengths are taken through the orthogonalisation matrix, so an oblique
    cell's diagonals are the true ones; the shorter of the two diagonals of
    each face is the one the stencil reaches first. The body diagonal would
    be the next step, below ~0.46 A for Refmac, and is not checked."""
    M = np.asarray(orth_np, dtype=float)
    v = [M[:, i] / float(grid[i]) for i in range(3)]
    lengths = [np.linalg.norm(vi) for vi in v]
    for i in range(3):
        for j in range(i + 1, 3):
            lengths.append(min(np.linalg.norm(v[i] + v[j]), np.linalg.norm(v[i] - v[j])))
    near = [L for L in lengths if abs(L - r_shrink) <= margin * r_shrink]
    return float(min(near, key=lambda L: abs(L - r_shrink))) if near else None


# --------------------------------------------------------------------- scaling

UNIQUE_IJ = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _scaled_amplitudes(F_calc, inv_d2, s_cart, log_k, u_params, aniso):
    """|F_calc| after the overall scale.

    Isotropic: k * exp(-B s^2 / 4), the same convention as everything else in
    the pipeline.

    Anisotropic: k * exp(-s.B_cart.s / 4), with B_cart a symmetric (3,3) in the
    CARTESIAN reciprocal frame and s = Mstar h in A^-1. Note this is not the
    h.U.h of f_total, deliberately. In the integer-h basis the diagonal
    components differ by the ratio of the cell edges squared -- on 7FPV that is
    70x between a and c -- so an optimizer stepping them together either does
    nothing to one or overflows exp() on another, which is exactly what it did.
    In this frame every component is a B in A^2, the isotropic answer is
    B_cart = B*I, and the problem is well scaled. Converting back to U for
    f_total is U = Mstar^T B_cart Mstar / 4.
    """
    mag = torch.abs(F_calc)
    if aniso:
        B = torch.zeros(3, 3, dtype=inv_d2.dtype, device=inv_d2.device)
        for value, (i, j) in zip(u_params, UNIQUE_IJ):
            B = B + value * _unit_sym(i, j, inv_d2)
        exponent = -0.25 * ((s_cart @ B) * s_cart).sum(-1)
    else:
        exponent = -0.25 * u_params[0] * inv_d2
    # A line search may propose a B that overflows exp() before it is rejected;
    # an inf here becomes a nan gradient and poisons the whole fit rather than
    # producing the large loss that would have made the step be rejected.
    return torch.exp(log_k) * torch.exp(exponent.clamp(max=60.0)) * mag


def _unit_sym(i, j, like):
    E = torch.zeros(3, 3, dtype=like.dtype, device=like.device)
    E[i, j] = 1.0
    E[j, i] = 1.0
    return E


def _f_calc(F_protein, F_mask, inv_d2, k_sol, b_sol):
    """F_protein + k_sol * exp(-b_sol s^2 / 4) * F_mask -- f_total's inner
    bracket, with the outer scale left to _scaled_amplitudes so that the fit
    can treat the two independently."""
    if F_mask is None:
        return F_protein
    weight = k_sol * torch.exp(-0.25 * b_sol * inv_d2)
    return F_protein + weight.to(F_protein.dtype) * F_mask


def optimal_log_k(F_obs, model_amplitudes):
    """The overall scale that minimises sum (Fo - k*g)^2, in closed form.

    Solved rather than refined because it is exactly linear, which keeps it out
    of the search and makes the coarse scan below a scan over two parameters
    instead of four.
    """
    num = (F_obs * model_amplitudes).sum()
    den = (model_amplitudes * model_amplitudes).sum()
    return torch.log(torch.clamp(num / den, min=1e-12))


def r_factor(F_obs, F_model):
    """R = sum||Fo| - |Fc|| / sum|Fo|, the crystallographic convention."""
    return float((F_obs - F_model).abs().sum() / F_obs.abs().sum())


def resolution_bins(inv_d2, n_bins):
    """Equal-count resolution bins: an index 0..n_bins-1 per reflection, 0 the
    lowest resolution. Free reflections are binned with the rest, so that a
    scale fitted on the work set applies to them unchanged."""
    order = torch.argsort(inv_d2)
    bins = torch.empty_like(order)
    bins[order] = (torch.arange(len(order), device=order.device) * n_bins
                   // len(order))
    return bins


def log_resolution_bins(d, n_lowest=100, min_per_bin=50, max_bins=30,
                        eps=1e-4, device="cpu"):
    """mmtbx's resolution bins: an index per reflection, 0 the lowest
    resolution, from d spacings in A.

    A port of cctbx.miller.set.log_binning (Urzhumtsev et al. 2009), which is
    what mmtbx.f_model bins its scales by, with the same defaults: the first
    bin holds the n_lowest lowest-resolution reflections, every later bin is a
    step of equal width in ln(1/d^2) set by the second n_lowest, the step is
    widened if that would make more than max_bins, a last bin under a quarter
    of the one before is merged into it, and any bin under min_per_bin
    reflections is merged into the one before. Fine at low resolution, where
    the solvent term and the scale change fastest; few and wide at high
    resolution; and fewer bins when there are fewer reflections, which
    equal-count bins (resolution_bins) do not do. Fewer than 2*n_lowest
    reflections is one bin.

    Pass the d spacings cctbx itself computed for these reflections
    (F_obs_array.d_spacings()) to get mmtbx's bins exactly: limits fall on
    data values by construction, so recomputing d some other way can move a
    reflection that sits on a limit. Free reflections are binned with the
    rest, as in resolution_bins.

    One deliberate difference from cctbx: if reflections n_lowest to
    2*n_lowest share one d spacing the bin step is zero, and cctbx's loop
    never ends; this raises ValueError instead. Merged data do not do that."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if 2 * n_lowest >= n:
        return torch.zeros(n, dtype=torch.long, device=device)
    assert max_bins > 1
    d_sorted = np.sort(d)[::-1]               # lowest resolution first

    # Scalar math, not numpy: only these four ln(1/d^2) are ever used, and
    # the limits sit on data values, so a reflection on a limit lands in the
    # bin cctbx puts it in only if the roundoff is libm's, as in cctbx --
    # numpy's vectorised log/exp can differ in the last place.
    def lnss(x):
        return math.log(1.0 / (x * x))

    def d_of(x):
        return math.sqrt(1.0 / math.exp(x))

    s0, s1, s2, top = (lnss(float(d_sorted[i]))
                       for i in (0, n_lowest, 2 * n_lowest, n - 1))
    if s2 == s1:
        raise ValueError("log_resolution_bins: reflections %d to %d share one "
                         "d spacing, so the bin step is zero"
                         % (n_lowest, 2 * n_lowest))

    def limits_for(step, cap=None):
        # cap: stop as soon as there are more limits than max_bins. The
        # first call's result is then discarded in favour of the rescaled
        # step, so a step that is tiny but not zero cannot loop for as long
        # as cctbx's would.
        limits = [d_of(s0)]
        x = s1
        while x <= top:
            if cap is not None and len(limits) > cap:
                return limits
            limits.append(d_of(x))
            x += step
        limits.append(d_of(min(top, x)))
        return limits

    limits = limits_for(s2 - s1, cap=max_bins)
    if len(limits) > max_bins:
        limits = limits_for((top - s1) / (max_bins - 1))
    pairs = []
    for i in range(len(limits)):
        if i < len(limits) - 2:
            pairs.append((limits[i], limits[i + 1]))
        elif i == len(limits) - 1:
            pairs.append((limits[i - 1], limits[i]))
    selections = []
    for i, (d_max, d_min) in enumerate(pairs):
        if i == 0:
            sel = (d <= d_max + eps) & (d > d_min)
        elif i == len(pairs) - 1:
            sel = (d <= d_max) & (d > d_min - eps)
        else:
            sel = (d <= d_max) & (d > d_min)
        selections.append(sel)
    last, before = selections[-1], selections[-2]
    if last.sum() < before.sum() / 4:
        selections = selections[:-2] + [before | last]
    merged = []
    for sel in selections:
        if sel.sum() < min_per_bin and merged:
            merged[-1] = merged[-1] | sel
        else:
            merged.append(sel)
    bins = np.full(n, -1, dtype=np.int64)
    for i, sel in enumerate(merged):
        bins[sel] = i
    if (bins < 0).any():
        raise ValueError("log_resolution_bins left %d reflections unbinned"
                         % int((bins < 0).sum()))
    return torch.as_tensor(bins, device=device)


def scale_bins_arg(text):
    """argparse type for a bin specification: "log" for mmtbx's bins, or an
    integer N >= 0 for N equal-count bins (0 = no bins)."""
    if text == "log":
        return text
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected 'log' or a whole number of bins, got %r" % text)
    if n < 0:
        raise argparse.ArgumentTypeError("number of bins must be >= 0")
    return n


def assign_scale_bins(spec, inv_d2, d):
    """The bin index per reflection for a scale_bins_arg value, or None for 0.
    inv_d2 is the torch tensor the fit uses; d the cctbx d spacings of the
    same reflections, in the same order."""
    if spec == "log":
        return log_resolution_bins(d, device=inv_d2.device)
    if spec:
        return resolution_bins(inv_d2, spec)
    return None


def mmtbx_moving_average(x):
    """mmtbx.bulk_solvent.scaler.moving_average, ported: five passes of a
    three-point mean applied only at local extrema, ends padded by
    repetition. Takes and returns a list of floats."""
    x_ = [x[0]] + list(x) + [x[-1]]
    for _ in range(5):
        result = x_[:]
        extremum = [i for i in range(1, len(result) - 1)
                    if ((result[i - 1] < result[i] and result[i + 1] < result[i]) or
                        (result[i - 1] > result[i] and result[i + 1] > result[i]))]
        for i in extremum:
            result[i] = (x_[i - 1] + x_[i] + x_[i + 1]) / 3.0
        x_ = result[:]
    return result[1:-1]


def solvent_curve_rule(k_bins, d_min_bins, smooth=True, d_cut=3.0):
    """mmtbx's regularisation of a per-bin solvent scale (scaler.smooth and
    the non-negative grid it searches): negatives to zero; from the first
    bin that is zero AND whose high-resolution edge is below d_cut, zero in
    every bin from there on; and the moving average at extrema. The solvent
    term has no signal at high resolution and a free bin fits noise
    (unregularised, 36 of 40 SW40 entries went negative there).

    Values under 0.005, half of mmtbx's grid step, count as zero.

    One deliberate difference in order: mmtbx smooths first and truncates
    after, which works for its grid search because that lands on exact
    zeros in runs. A least-squares bin lands on noise around zero instead,
    and smoothing first would average a clamped zero with its neighbours and
    re-admit the noise; so here the cut is found on the clamped values and
    the smoothing is applied to the bins that survive it. k_bins is a
    tensor, low resolution first; d_min_bins the bins' high-resolution edges
    in A."""
    # Values below half of mmtbx's grid step (0.01) are zero, as the grid
    # would have them. This is what makes the cut fire on a numerical zero:
    # a least-squares bin in a region with no solvent lands at +-1e-6 with a
    # sign the optimizer decides, and without the snap the cut would depend
    # on that sign. It also keeps a 0.001 bin next to a zero from being a
    # strict extremum that the moving average lifts off the floor. The edge
    # therefore sits at 0.005: a bin at 0.0051 is kept and smoothed, one at
    # 0.0049 is a zero, which is the same jump mmtbx's grid has at its step.
    k = [float(v) if float(v) >= 0.005 else 0.0 for v in k_bins]
    keep = solvent_cut_index(k, d_min_bins, d_cut)
    head = k[:keep]
    if smooth and len(head) > 2:
        head = mmtbx_moving_average(head)      # a mean of >= 0 values is >= 0
    out = head + [0.0] * (len(k) - keep)
    return torch.tensor(out, dtype=k_bins.dtype, device=k_bins.device)


def solvent_cut_index(k_bins, d_min_bins, d_cut=3.0):
    """Index of the first bin that is non-positive AND has its high-resolution
    edge below d_cut; bins from there on carry no solvent. len(k_bins) when
    there is none."""
    for i, (v, d) in enumerate(zip(k_bins, d_min_bins)):
        if float(v) <= 0.0 and d < d_cut:
            return i
    return len(k_bins)


def exponential_from_curve(k_bins, bins, inv_d2, d_cut=4.0):
    """(k_sol, b_sol, n): the exponential k exp(-b s^2/4) closest, in least
    squares over the reflections at d >= d_cut, to the per-bin curve they
    see; n is how many reflections that was. The same quantity mmtbx reports
    as k_sol/b_sol from its k_mask curve (f_model.k_sol_b_sol_from_k_mask:
    reflections at d >= 4 A, k in [0, 0.6], b in [0, 150]; mmtbx searches
    locally around its own starting guess, this is a global grid at the same
    bounds, so the two can differ in the second digit). A diagnostic, not a
    model parameter. With fewer than 2 reflections at d >= d_cut there is
    nothing to fit: n is 0 and the result is (max of the curve, 0)."""
    s2 = inv_d2.detach().to(torch.float64).cpu()
    k_refl = k_bins.detach().to(torch.float64).cpu()[bins.cpu()]
    sel = s2 <= 1.0 / d_cut ** 2
    n = int(sel.sum())
    if n < 2:
        return min(0.6, max(0.0, float(k_bins.max()))), 0.0, 0
    k, s2 = k_refl[sel], s2[sel]
    k_grid = torch.arange(0.0, 0.6001, 0.005, dtype=torch.float64)
    b_grid = torch.arange(0.0, 150.001, 1.0, dtype=torch.float64)
    # sum over reflections of (k_g e^{-b s2/4} - k)^2, expanded so that the
    # per-reflection work is two reductions rather than a (121, 151, n) grid
    e = torch.exp(-0.25 * b_grid[:, None] * s2[None, :])           # (151, n)
    ee = (e * e).sum(1)                                             # (151,)
    ek = (e * k[None, :]).sum(1)                                    # (151,)
    cost = (k_grid[:, None] ** 2) * ee[None, :] - 2.0 * k_grid[:, None] * ek[None, :]
    i, j = divmod(int(torch.argmin(cost)), len(b_grid))
    return float(k_grid[i]), float(b_grid[j]), n


def describe_bins(spec, bins):
    """"10 log bins" or "20 equal-count bins", for printed output."""
    n = int(bins.max()) + 1
    return "%d %s bin%s" % (n, "log" if spec == "log" else "equal-count",
                           "" if n == 1 else "s")


def solvent_scheme(fit):
    """"per bin" or "exponential": which solvent scale a fit_scales result
    used, for printed output and the result dict."""
    return "per bin" if fit.get("k_mask_bins") is not None else "exponential"


def solvent_weight(fit, inv_d2):
    """The factor on F_mask the fit applies, per reflection: k_mask[bin] for
    a per-bin solvent fit, k_sol exp(-b_sol s^2/4) otherwise, zero for a fit
    without solvent. model_amplitudes and the shell table's solvent column
    both take it from here, so they cannot disagree about which solvent
    model was fitted -- the k_sol/b_sol of a per-bin fit are a diagnostic
    the model never used."""
    if fit.get("k_mask_bins") is not None:
        return fit["k_mask_bins"][fit["bins"]]
    if not fit["fit_solvent"]:
        return torch.zeros_like(inv_d2)
    return fit["k_sol"] * torch.exp(-0.25 * fit["b_sol"] * inv_d2)


def scan_grids(fit_solvent, like):
    """The (k_sol, b_sol, B_overall) grids of fit_scales' coarse scan, on
    like's dtype and device. One place, so that the equivalence tests scan
    the grid the fit scans. Without the solvent the first two are the single
    node 0."""
    dtype, device = like.dtype, like.device
    b_grid = torch.linspace(-20.0, 80.0, 26, dtype=dtype, device=device)
    if fit_solvent:
        k_grid = torch.linspace(0.0, 1.2, 25, dtype=dtype, device=device)
        # Deliberately wider than any physical b_sol. A scan that stops at 200
        # returns 199 for a mask carrying too much high-resolution content,
        # which reads as a fit rather than as the pinned boundary it is.
        bs_grid = torch.linspace(0.0, 400.0, 41, dtype=dtype, device=device)
    else:
        k_grid = torch.zeros(1, dtype=dtype, device=device)
        bs_grid = torch.zeros(1, dtype=dtype, device=device)
    return k_grid, bs_grid, b_grid


def coarse_scan(Fo, Fp, Fm, s2, k_grid, bs_grid, b_grid):
    """Stage 1 of fit_scales: the node of the (k_sol, b_sol, B_overall) grid
    with the smallest least-squares residual, the overall scale solved
    exactly at every node. Returns (k_sol, b_sol, B_overall, log_k) as
    floats, the same node coarse_scan_reference finds.

    Batched: for each b_sol the amplitudes |F_p + k_sol w(b_sol) F_m| are
    formed for every k_sol at once, (n_k, n_refl), and the two sums the
    residual needs at every (k_sol, B_overall) node,

        num = sum_h Fo e^{-B s^2/4} |F_c|,   den = sum_h e^{-B s^2/4 * 2} |F_c|^2,

    are two matrix products against the (n_B, n_refl) table of overall-B
    factors. The residual at the exact scale k = num/den is
    sum Fo^2 - num^2/den, which is the loop's sum (Fo - k g)^2 expanded, so
    nothing is fitted differently. What changes is the shape of the work:
    the loop ran 26,650 nodes as separate small kernels and read each
    residual back as a Python float, which on a GPU is a device
    synchronisation per node and was ~4 s of every entry's wall time there
    (measured on the torchref1000 pilot, 2026-11-10); here it is 41 passes
    of a few batched operations and one synchronisation at the end. The
    minimum is taken in the loop's own order (k_sol outermost, B_overall
    innermost) so a tie resolves to the same node."""
    with torch.no_grad():
        n_k, n_bs, n_b = len(k_grid), len(bs_grid), len(b_grid)
        E = torch.exp(-0.25 * b_grid[:, None] * s2[None, :])        # (n_b, n)
        E2 = E * E
        Fo_sq = (Fo * Fo).sum()
        num = torch.empty(n_k, n_bs, n_b, dtype=s2.dtype, device=s2.device)
        den = torch.empty_like(num)
        Fm_row = None if Fm is None else Fm[None, :]
        for j in range(n_bs):
            # _f_calc broadcasts: (n_k, 1) k_sol against (1, n) reflections
            # gives the (n_k, n) table; without a mask it is the (1, n) row,
            # which the assignments below broadcast over n_k (then 1).
            mag = torch.abs(_f_calc(Fp[None, :], Fm_row, s2[None, :],
                                    k_grid[:, None], bs_grid[j]))
            num[:, j, :] = (mag * Fo[None, :]) @ E.T                 # (n_k, n_b)
            den[:, j, :] = (mag * mag) @ E2.T
        k = torch.clamp(num / den, min=1e-12)
        # sum (Fo - k g)^2 at the clamped k, exactly as the loop evaluated it
        resid = Fo_sq - 2.0 * k * num + k * k * den
        idx = torch.argmin(resid.reshape(-1))                        # first minimum, loop order
        i = idx // (n_bs * n_b)
        j = idx // n_b % n_bs
        m = idx % n_b
        # One read-back for all four, the only synchronisation in the scan.
        return tuple(torch.stack([k_grid[i], bs_grid[j], b_grid[m],
                                  torch.log(k.reshape(-1)[idx])]).tolist())


def coarse_scan_reference(Fo, Fp, Fm, s2, k_grid, bs_grid, b_grid):
    """The scan as a Python loop over nodes, kept as the reference the
    vectorised coarse_scan is tested against. Not used by the fit."""
    best = None
    with torch.no_grad():
        for k_sol in k_grid:
            for b_sol in bs_grid:
                mag = torch.abs(_f_calc(Fp, Fm, s2, k_sol, b_sol))
                for b_ov in b_grid:
                    g = torch.exp(-0.25 * b_ov * s2) * mag
                    log_k = optimal_log_k(Fo, g)
                    resid = float(((Fo - torch.exp(log_k) * g) ** 2).sum())
                    if best is None or resid < best[0]:
                        best = (resid, float(k_sol), float(b_sol),
                                float(b_ov), float(log_k))
    return best[1:]


def fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work, aniso,
               fit_solvent=True, bins=None, solvent_bins=False, log=sys.stdout):
    """
    Least squares on amplitudes over the work set. Returns a parameter dict.

    Two stages, because the objective is not convex in (k_sol, b_sol) and a
    gradient method started at the conventional values can and does settle into
    a local minimum that depends on where it began:

      1. A coarse scan over (k_sol, b_sol), with the overall scale solved
         exactly at every node and the overall B scanned alongside
         (coarse_scan: batched, one synchronisation; the two structure
         factor sets are computed once, before the scan).
      2. LBFGS on every parameter at once from the best node.

      3. Only with bins (resolution_bins or log_resolution_bins): the
         single overall scale becomes one scale per bin, refined by LBFGS
         jointly with everything else from where stage 2 finished. This is
         mmtbx's binned k_isotropic, and it is what lets the model follow a
         falloff that no Gaussian B can -- 8VQ1, room-temperature XFEL data,
         goes from R-work 0.198 to 0.138 with 20 equal-count bins, against
         mmtbx's 0.137. k_sol and b_sol stay single numbers, so
         they remain the diagnostic they are without bins.

      4. Only with bins AND solvent_bins=True: the solvent scale becomes one
         value per bin as well, k_mask[bin] in place of k_sol exp(-b_sol s^2/4),
         which is what mmtbx fits (a free k_mask per bin). Least squares
         jointly with the bin scales, started from the exponential of stage
         3, B_cart held where stage 3 left it, and regularised by mmtbx's own
         rule (solvent_curve_rule): no negative values, a moving average at
         local extrema, and zero from the first zero below 3 A outward. The
         surviving bins are refit from the regularised curve and the bin
         scales are then solved in closed form. (In practice the second pass
         returns to the least-squares optimum of the bins it frees, so its
         effect is to refit after the cut has zeroed bins that carried
         weight.) The curve is a step per bin; mmtbx additionally chooses,
         per bin, between the step and a linear interpolation to the next
         bin, which this does not do. Measured on SW40 (stage 0,
         2026-11-09) this closes the lunus-vs-mmtbx R-free gap from a median
         +0.0020 to +0.0011, which is where mmtbx's complete scaling on the
         same structure factors lands; the exponential's shape -- too little
         solvent at 6-4 A, too much at 3-2 A -- was the whole remaining
         scaling difference. The k_sol/b_sol returned by such a fit are an
         exponential fitted to the curve at d >= 4 A, as mmtbx reports them,
         not parameters the model used.

    Fitting on amplitudes rather than intensities matches the quantity R is
    computed from. The fit is unweighted -- weighting by 1/sigma^2 would tilt
    it toward the strong low-resolution terms, which is where the solvent lives
    and therefore exactly where an unweighted comparison is more honest.
    """
    Fo = F_obs[work]
    Fp = F_protein[work]
    Fm = None if F_mask is None else F_mask[work]
    s2 = inv_d2[work]
    s = s_cart[work]

    k_sol0, b_sol0, b_ov0, log_k0 = coarse_scan(Fo, Fp, Fm, s2,
                                                *scan_grids(fit_solvent, inv_d2))
    print("  coarse scan: k_sol %.3f, b_sol %.1f, B_overall %.1f"
          % (k_sol0, b_sol0, b_ov0), file=log)

    dtype, device = inv_d2.dtype, inv_d2.device
    log_k = torch.tensor(log_k0, dtype=dtype, device=device, requires_grad=True)
    k_sol = torch.tensor(k_sol0, dtype=dtype, device=device,
                         requires_grad=fit_solvent)
    b_sol = torch.tensor(b_sol0, dtype=dtype, device=device,
                         requires_grad=fit_solvent)
    if aniso:
        # Initialised at the isotropic answer, B_cart = B*I, so the refinement
        # starts where the isotropic fit finished and the off-diagonals start
        # at exactly zero -- which is where orthorhombic symmetry requires them
        # to stay. See the note this prints at the end.
        u_params = [torch.tensor(b_ov0 if i == j else 0.0, dtype=dtype,
                                 device=device, requires_grad=True)
                    for i, j in UNIQUE_IJ]
    else:
        u_params = [torch.tensor(b_ov0, dtype=dtype, device=device,
                                 requires_grad=True)]

    params = [log_k] + u_params + ([k_sol, b_sol] if fit_solvent else [])
    opt = torch.optim.LBFGS(params, max_iter=200, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        F_c = _f_calc(Fp, Fm, s2, k_sol, b_sol)
        model = _scaled_amplitudes(F_c, s2, s, log_k, u_params, aniso)
        loss = ((Fo - model) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    log_k_bins = None
    if bins is not None:
        # The overall scale is absorbed into the bins, and the isotropic part
        # of B is then degenerate with them: left free, the pair drifts
        # together (B_overall -34 against bin scales falling to 0.01 on 8VQ1)
        # and the printed B means nothing. So the isotropic part is held at
        # the single-scale answer and only the anisotropy refines with the
        # bins; an isotropic fit has no B left to refine at all.
        b = bins[work]
        n_bins = int(bins.max()) + 1
        log_k_bins = torch.full((n_bins,), float(log_k.detach()), dtype=dtype,
                                device=device, requires_grad=True)
        n_diag = 3 if aniso else 1
        b_iso0 = sum(float(p) for p in u_params[:n_diag]) / n_diag

        def held(u):
            shift = sum(u[:n_diag]) / n_diag - b_iso0
            return [p - shift for p in u[:n_diag]] + list(u[n_diag:])

        params = [log_k_bins] + (u_params if aniso else []) + (
            [k_sol, b_sol] if fit_solvent else [])
        opt = torch.optim.LBFGS(params, max_iter=300,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            F_c = _f_calc(Fp, Fm, s2, k_sol, b_sol)
            model = _scaled_amplitudes(F_c, s2, s, log_k_bins[b],
                                       held(u_params), aniso)
            loss = ((Fo - model) ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            u_params = [p.clone() for p in held(u_params)]
        k = torch.exp(log_k_bins.detach())
        print("  %d resolution bins: scale %.3f-%.3f (low to high resolution: "
              "%s)" % (n_bins, float(k.min()), float(k.max()),
                       " ".join("%.2f" % float(v) for v in k)), file=log)
        log_k_bins = log_k_bins.detach()

    k_mask_bins, n_fit = None, 0
    k_sol_out, b_sol_out = k_sol.detach(), b_sol.detach()
    if bins is not None and fit_solvent and solvent_bins:
        # Bin edges and centres from every reflection in the bin, not the
        # work subset: a bin is never empty that way, and it is what
        # mmtbx's ss_bin_values are.
        d_min_bins = [float(1.0 / torch.sqrt(inv_d2[bins == i].max()))
                      for i in range(n_bins)]
        s2_bins = [float(inv_d2[bins == i].mean()) for i in range(n_bins)]
        k_mask_bins, log_k_bins = _fit_solvent_bins(
            Fo, Fp, Fm, s2, s, b, n_bins, log_k_bins, u_params, aniso,
            k_sol_out, b_sol_out, d_min_bins, s2_bins)
        ks, bs, n_fit = exponential_from_curve(k_mask_bins, bins, inv_d2)
        k_sol_out = torch.tensor(ks, dtype=dtype, device=device)
        b_sol_out = torch.tensor(bs, dtype=dtype, device=device)
        print("  solvent scale per bin (low to high resolution): %s"
              % " ".join("%.2f" % float(v) for v in k_mask_bins), file=log)
        if n_fit:
            print("  exponential closest to it over the %d reflections at d >= 4 A: "
                  "k_sol %.3f, b_sol %.1f (reported as k_sol/b_sol; not used by "
                  "the model)" % (n_fit, ks, bs), file=log)
        else:
            print("  fewer than 2 reflections at d >= 4 A to fit an exponential to; "
                  "k_sol is reported as the curve's maximum and b_sol as 0", file=log)

    return {
        "log_k": log_k.detach(),
        "log_k_bins": log_k_bins,
        "bins": bins,
        "u_params": [p.detach() for p in u_params],
        "aniso": aniso,
        "k_sol": k_sol_out,
        "b_sol": b_sol_out,
        "fit_solvent": fit_solvent,
        "k_mask_bins": k_mask_bins,
        "n_refl_exp_fit": n_fit if k_mask_bins is not None else None,
    }


# THE PER-BIN SOLVENT SCALE IS BOUNDED AT 1.0. mmtbx's initial per-bin search
# runs on a grid from 0 to 1.0 (bulk_solvent/scaler.py, k_mask_grid_search);
# its later cycles are not strictly bounded but only accept a curve that lowers
# R. A free least-squares bin can run away: on 4Y0B (data to 1.83 A, in the
# torchref1000 run of 2026-11-10, once the soft-sphere mask had given F_mask
# real high-resolution content) the high-resolution bins found a degenerate
# solution in which the solvent stood in for the protein, solvent scales 2,
# 5.6, 16, 49, 147 with the bin scales going to zero, binned R-free 0.507
# against mmtbx's 0.324. Bounded, the last bin sits AT the bound (curve
# ... 0.21 0.24 0.35 1.00) and R-free is 0.331: the bound binds, it does not
# make the curve fall to zero. The solve is exact per bin (_fit_solvent_bins).
# On the 20 + 20 + 39 entries of the low-, mid-resolution and SW40 sets no
# other entry moves by more than 0.0009 and no median moves.
SOLVENT_BIN_MAX = 1.0
# Measured and rejected, kept for a later look: zeroing every bin whose
# high-resolution edge is below this many A (with a 0.6 bound) fixed 4Y0B
# equally but made 9 of those 79 entries worse by more than 0.001 and none
# better; it discards real solvent signal beyond 3 A. mmtbx itself cuts only
# from the first zero bin below 3 A, which solvent_curve_rule does. None = off.
SOLVENT_BIN_FORCE_CUT = None


def _fit_solvent_bins(Fo, Fp, Fm, s2, s, b, n_bins, log_k_bins, u_params, aniso,
                      k_sol, b_sol, d_min_bins, s2_bins):
    """Stage 4 of fit_scales, on the work set: one solvent scale per bin.
    d_min_bins and s2_bins are each bin's high-resolution edge (A) and mean
    s^2, over every reflection in it. Returns (k_mask_bins, log_k_bins),
    both detached. A bin with no work reflections (possible with many
    equal-count bins on little data) keeps its stage-3 scale and starts the
    solvent at the exponential's value at the bin centre."""
    dtype, device = s2.dtype, s2.device
    u = [p.detach() for p in u_params]
    in_bin = [b == i for i in range(n_bins)]
    with torch.no_grad():
        km0 = k_sol * torch.exp(-0.25 * b_sol * s2)
        km = torch.stack([
            km0[sel].mean() if bool(sel.any())
            else k_sol * torch.exp(-0.25 * b_sol * s2_bins[i])
            for i, sel in enumerate(in_bin)])

    def solve_lbfgs(active, km_start, lk_start):
        # the unbounded joint solve (SOLVENT_BIN_MAX = None), as before
        km_p = km_start.clone().requires_grad_(True)
        lk_p = lk_start.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([km_p, lk_p], max_iter=300,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-10, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            km_eff = torch.where(active, km_p, torch.zeros_like(km_p))
            F_c = Fp + km_eff[b].to(Fp.dtype) * Fm
            model = _scaled_amplitudes(F_c, s2, s, lk_p[b], u, aniso)
            loss = ((Fo - model) ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return km_p.detach(), lk_p.detach()

    # THE BOUNDED SOLVE IS EXACT, PER BIN. With the anisotropy u held fixed
    # the model in bin i is exp(lk_i) * A * |Fp + k_i Fm| with A per
    # reflection, so the bins are independent and lk_i has a closed form:
    # what is left is a 1-D minimisation of the residual over k_i in
    # [0, SOLVENT_BIN_MAX], done as mmtbx's initial search does it (a 0.01
    # grid) followed by golden-section refinement in the bracketing cell.
    # Not a clamp inside an optimizer: that was tried first and a bin that
    # overshot below 0 got no gradient back, came out as 0, and fired the
    # curve rule's cut on 4Y0B -- the right answer for the wrong reason.
    with torch.no_grad():
        A = _scaled_amplitudes(torch.ones_like(Fp), s2, s,
                               torch.zeros((), dtype=dtype, device=device), u, aniso)

    def bin_residual(k, sel):
        # residual sum of squares at the closed-form scale, for a tensor of k
        g = A[sel][None, :] * (Fp[sel][None, :] + k[:, None].to(Fp.dtype) * Fm[sel][None, :]).abs()
        fo = Fo[sel][None, :]
        num = (fo * g).sum(1)
        den = (g * g).sum(1).clamp(min=1e-300)
        return (fo * fo).sum(1) - num * num / den

    def solve_bounded(active, km_start, lk_start):
        km_out = torch.zeros_like(km_start)
        k_max = float(SOLVENT_BIN_MAX)
        grid = torch.arange(0.0, k_max + 1e-9, 0.01, dtype=dtype, device=device)
        with torch.no_grad():
            for i, sel in enumerate(in_bin):
                if not bool(active[i]):
                    continue
                if not bool(sel.any()):
                    # no work reflections: keep the starting value (the
                    # exponential's at the bin centre), as the unbounded
                    # solve did; a 0 here would fire the curve rule's cut
                    km_out[i] = km_start[i].clamp(0.0, k_max)
                    continue
                r = bin_residual(grid, sel)
                j0 = int(torch.argmin(r))
                lo, hi = float(grid[max(j0 - 1, 0)]), float(grid[min(j0 + 1, len(grid) - 1)])
                gr = 0.5 * (5 ** 0.5 - 1)
                a, c = lo, hi
                x1, x2 = c - gr * (c - a), a + gr * (c - a)
                f1, f2 = (float(v) for v in bin_residual(
                    torch.tensor([x1, x2], dtype=dtype, device=device), sel))
                for _ in range(40):
                    if f1 <= f2:
                        c, x2, f2 = x2, x1, f1
                        x1 = c - gr * (c - a)
                        f1 = float(bin_residual(torch.tensor([x1], dtype=dtype, device=device), sel)[0])
                    else:
                        a, x1, f1 = x1, x2, f2
                        x2 = a + gr * (c - a)
                        f2 = float(bin_residual(torch.tensor([x2], dtype=dtype, device=device), sel)[0])
                best = min((float(r[j0]), float(grid[j0])), (f1, x1), (f2, x2))
                km_out[i] = best[1]
        return km_out, lk_start

    def solve(active, km_start, lk_start):
        if SOLVENT_BIN_MAX is None:
            return solve_lbfgs(active, km_start, lk_start)
        return solve_bounded(active, km_start, lk_start)

    active = torch.ones(n_bins, dtype=torch.bool, device=device)
    if SOLVENT_BIN_FORCE_CUT is not None:
        active = torch.tensor([float(d) >= SOLVENT_BIN_FORCE_CUT for d in d_min_bins],
                              dtype=torch.bool, device=device)
    km, lk = solve(active, km, log_k_bins)
    km = solvent_curve_rule(km, d_min_bins)
    # Second pass: every bin before the cut stays free, including one the
    # first pass clamped to zero above 3 A, so that it can come back positive
    # once its neighbours are regularised (mmtbx's grid allows the same);
    # bins beyond the cut are held at zero.
    cut = solvent_cut_index(km, d_min_bins)
    active = (torch.arange(n_bins, device=device) < cut) & active
    if cut > 0:
        km, lk = solve(active, km, lk)
        km = solvent_curve_rule(torch.where(active, km, torch.zeros_like(km)), d_min_bins)
    # Bin scales in closed form with the curve fixed: exactly linear per bin.
    with torch.no_grad():
        F_c = Fp + km[b].to(Fp.dtype) * Fm
        zero = torch.zeros((), dtype=dtype, device=device)
        g = _scaled_amplitudes(F_c, s2, s, zero, u, aniso)
        lk = torch.stack([optimal_log_k(Fo[sel], g[sel]) if bool(sel.any())
                          else lk[i] for i, sel in enumerate(in_bin)])
    return km, lk


def model_f_c(fit, F_protein, F_mask, inv_d2):
    """F_c = F_protein + k_mask F_mask with the fit's solvent, before the
    overall scale; F_protein alone for a fit without solvent."""
    if fit["fit_solvent"] and F_mask is not None:
        return F_protein + solvent_weight(fit, inv_d2).to(F_protein.dtype) * F_mask
    return F_protein


def model_log_k(fit):
    """The fit's log overall scale, per reflection for a binned fit."""
    if fit.get("log_k_bins") is not None:
        return fit["log_k_bins"][fit["bins"]]
    return fit["log_k"]


def model_amplitudes(fit, F_protein, F_mask, inv_d2, s_cart):
    with torch.no_grad():
        return _scaled_amplitudes(model_f_c(fit, F_protein, F_mask, inv_d2), inv_d2,
                                  s_cart, model_log_k(fit), fit["u_params"], fit["aniso"])


def model_scale(fit, inv_d2, s_cart):
    """The overall scale model_amplitudes puts on F_c, per reflection: real
    and positive, so F_model keeps F_c's phase."""
    return _scaled_amplitudes(torch.ones_like(inv_d2), inv_d2, s_cart, model_log_k(fit),
                              fit["u_params"], fit["aniso"])


def model_structure_factors(fit, F_protein, F_mask, inv_d2, s_cart):
    """Complex F_model = k_total (F_protein + k_mask F_mask) on the Fobs
    scale: model_amplitudes is its modulus, and its phase is F_c's."""
    with torch.no_grad():
        F_c = model_f_c(fit, F_protein, F_mask, inv_d2)
        return model_scale(fit, inv_d2, s_cart).to(F_c.dtype) * F_c


def nearest_bins(d_new, d_fit, bins_fit):
    """A scale bin for reflections the fit never saw: that of the fitted
    reflection nearest in d. Inside the fitted range that is the bin whose
    range holds it; beyond it, the first or last bin, so a complete set's
    lowest- and highest-resolution reflections take the end bins' scales."""
    if len(d_fit) == 1:
        return np.full(len(d_new), bins_fit[0])
    order = np.argsort(d_fit)
    d_sorted, b_sorted = d_fit[order], bins_fit[order]
    pos = np.clip(np.searchsorted(d_sorted, d_new), 1, len(d_sorted) - 1)
    left, right = d_sorted[pos - 1], d_sorted[pos]
    pick = np.where(np.abs(d_new - left) <= np.abs(right - d_new), pos - 1, pos)
    return b_sorted[pick]


def extend_free_flags(flags, test_value, complete_set):
    """flags (an integer miller array, test set test_value) extended to every
    reflection of complete_set it lacks, keeping every value it has.

    Up to 20 new reflections all go to the work set, as cctbx does for a
    set too small to draw from; more are drawn at the flags' own test
    fraction (cctbx's generate_r_free_flags, lattice symmetry respected,
    which draws from Python's generator; per reflection at a fraction of
    0.5 or more, which it refuses).
    A work reflection takes a work value of the file's own, at random: a
    CCP4 bin 1..N-1, or the one work value of a 0/1 column. Not cctbx's
    extend_flags(preserve_input_values=True), which writes a small
    all-work draw on CCP4 flags as bin 0 -- the TEST set. Seeded; the
    caller restores the generators."""
    from cctbx.array_family import flex

    missing = complete_set.lone_set(flags)
    n = missing.indices().size()
    if n == 0:
        return flags, 0
    values = list(flags.data())
    work_values = sorted(set(values) - {test_value}) or [1 if test_value == 0 else 0]
    fraction = values.count(test_value) / float(len(values))
    if n <= 20:
        is_test = [False] * n
    elif fraction < 0.5:
        is_test = list(missing.generate_r_free_flags(
            fraction=fraction, max_free=None, use_lattice_symmetry=True).data())
    else:
        # cctbx refuses a fraction of 0.5 or more (Sorry); a set flagged
        # that way is drawn reflection by reflection at its own fraction
        is_test = [random.random() < fraction for _ in range(n)]
    new = flex.int([test_value if t else random.choice(work_values) for t in is_test])
    return flags.concatenate(missing.array(data=new)), n


def complete_free_flags(free_flags, free_flags_file, complete_set):
    """The flags to write with a complete set: every flag the file has
    inside it (free_flags_file, or the fitted reflections' free_flags pair
    without one), extended by extend_free_flags to the reflections it never
    flagged. Returns (flags, number drawn). The draw is seeded so that the
    same run writes the same file, with Python's generator -- the only one
    it draws from -- handed back as it was."""
    flags, test_value = free_flags
    base = flags if free_flags_file is None else free_flags_file.common_set(complete_set)
    state = random.getstate()
    random.seed(0)
    try:
        return extend_free_flags(base, test_value, complete_set)
    finally:
        random.setstate(state)


def write_model_mtz(path, F_obs_array, free_flags, F_model, history,
                    complete=None, components=None, observed=False):
    """The scaled model as synthetic data.

    F_obs_array is the observations the fit used (indices, symmetry,
    sigmas); F_model the complex model on those reflections, in that order.
    Columns: FP and SIGFP (|F_model| with the observed sigmas), PHIC
    (F_model's phase), FreeR_flag (free_flags, an integer miller array, or
    None for none; with complete, already extended to it by
    complete_free_flags). complete, a dict with the complete miller set
    ('set') and the model on it ('F_model'), adds FMODEL/PHIFMODEL over
    every reflection to d_min: FP/SIGFP stay on the observed reflections,
    because iotbx refuses an FP whose SIGFP is missing. components, a dict of complex arrays
    'F_protein' and 'F_mask' on the same reflections as the model written
    (the complete set if given), adds FPROTEIN/PHIFPROTEIN and
    FMASK/PHIFMASK, unscaled. observed adds FOBS/SIGFOBS. history is a list
    of lines for the MTZ history."""
    from cctbx import miller
    from cctbx.array_family import flex

    def bare(a):
        # a miller set of a's indices: complete_set() returns a set, the
        # observations an array, and the columns are built the same way on both
        return miller.set(a, a.indices(), a.anomalous_flag())

    def phases(a, values):
        return bare(a).array(data=flex.double(np.degrees(np.angle(values)).tolist()))

    def cplx(a, values):
        return bare(a).array(
            data=flex.complex_double(np.asarray(values, dtype=complex).tolist()))

    F_model = np.asarray(F_model)
    fp = F_obs_array.customized_copy(data=flex.double(np.abs(F_model).tolist()))
    ds = fp.as_mtz_dataset(column_root_label="FP")
    ds.add_miller_array(phases(F_obs_array, F_model), column_root_label="PHIC",
                        column_types="P")
    if free_flags is not None:
        ds.add_miller_array(free_flags.set_observation_type(None),
                            column_root_label="FreeR_flag")
    target = F_obs_array if complete is None else complete["set"]
    if complete is not None:
        ds.add_miller_array(cplx(target, complete["F_model"]), column_root_label="FMODEL")
    if components is not None:
        ds.add_miller_array(cplx(target, components["F_protein"]),
                            column_root_label="FPROTEIN")
        ds.add_miller_array(cplx(target, components["F_mask"]), column_root_label="FMASK")
    if observed:
        ds.add_miller_array(F_obs_array, column_root_label="FOBS")
    mtz = ds.mtz_object()
    # MTZ history lines hold 80 characters; the tool's own lines are kept
    # shorter, and only a long file name in the last one is cut
    mtz.add_history(flex.std_string([line[:80] for line in history]))
    mtz.write(file_name=path)


# ----------------------------------------------------------------- diagnostics

def compare_masks(ours, gemmi_grid, F_ours, F_gemmi, d, log=sys.stdout):
    """How our threshold mask differs from the geometric one, in the two places
    where it shows: the voxels, and the transform by resolution shell.

    Worth reporting even when the fit looks fine, because the two disagree in a
    way that a single overall agreement number hides -- they match where the
    solvent term is large and are uncorrelated where it is small.
    """
    gm = torch.tensor(np.asarray(gemmi_grid), dtype=ours.dtype,
                      device=ours.device)
    agreement = float(((ours > 0.5) == (gm > 0.5)).to(ours.dtype).mean())
    print("  binarised voxel agreement with gemmi's mask: %.4f" % agreement,
          file=log)

    a_ours = np.abs(F_ours.detach().cpu().numpy())
    a_gem = np.abs(F_gemmi.detach().cpu().numpy())
    edges = np.percentile(d, np.linspace(0, 100, 6))
    print("  %-16s %8s %10s %10s %8s"
          % ("d range (A)", "n", "|Fm| ours", "/ gemmi", "corr"), file=log)
    for lo, hi in zip(edges[:-1][::-1], edges[1:][::-1]):
        sel = (d >= lo) & (d <= hi)
        corr = np.corrcoef(a_ours[sel], a_gem[sel])[0, 1]
        print("  %6.2f - %-7.2f %8d %10.1f %10.3f %8.4f"
              % (hi, lo, sel.sum(), a_ours[sel].mean(),
                 a_ours[sel].mean() / a_gem[sel].mean(), corr), file=log)
    return agreement


def mmtbx_fit(pdb, cif, d_min=None, isotropic=False, select="all"):
    """R-work, R-free, k_sol and b_sol from mmtbx.f_model on the deposited
    model, with no lunus.sf code involved. select restricts the model to an
    iotbx atom selection, through the same select_atoms the torch path uses,
    so the two methods see identical atoms.

    This is mmtbx's own bulk solvent and anisotropic scaling, run on the
    coordinates and ADPs exactly as deposited -- no refinement -- so it is the
    R that a correct solvent model and a correct scattering model reach on
    this data. isotropic=True first flattens every ADP to its isotropic
    equivalent, which is what lunus.sf computes without --aniso-adp.

    The observations go through read_observations, i.e. the same French-Wilson
    amplitudes and the same free set the torch fit sees, so the two methods
    are compared on identical data.
    """
    import contextlib
    import io as _io

    import mmtbx.f_model
    from cctbx.array_family import flex
    from iotbx.pdb import hierarchy

    pdb_in = hierarchy.input(file_name=pdb, sort_atoms=False)
    xrs = select_atoms(pdb_in, pdb_in.input.xray_structure_simple(), select,
                       log=_io.StringIO(), name=pdb)
    # The reader's chatter goes to a sink, but what it dropped is returned,
    # so that a run of this method alone can say why its reflection count is
    # short of the file's.
    info = {}
    F_obs, free_np = read_observations(
        cif, d_min=d_min, crystal_symmetry=xrs.crystal_symmetry(),
        log=_io.StringIO(), info=info)
    free_flags = F_obs.customized_copy(
        data=flex.bool(free_np.tolist())).set_observation_type(None)

    if isotropic:
        xrs = xrs.deep_copy_scatterers()
        xrs.convert_to_isotropic()

    fmodel = mmtbx.f_model.manager(
        f_obs=F_obs, r_free_flags=free_flags, xray_structure=xrs)
    with contextlib.redirect_stdout(_io.StringIO()):
        fmodel.update_all_scales(remove_outliers=False)
    k_sol, b_sol = fmodel.k_sol_b_sol_from_k_mask()
    # mmtbx's own mask occupancy (its default mask: fixed ~0.6 A step,
    # solvent radius 1.1, shrink 0.9), the quantity our calibration target
    # is the gemmi analogue of. The manager keeps the contact_surface_fraction
    # of the asu_mask it built for F_mask, so it is not built a second time.
    # It stays None until the manager has computed F_mask itself, so pass
    # None on rather than fail the whole fit over a reported number.
    mask_occupancy = fmodel.mask_manager.solvent_content_via_mask
    if mask_occupancy is not None:
        mask_occupancy = float(mask_occupancy)
    return {"R-work": float(fmodel.r_work()), "R-free": float(fmodel.r_free()),
            "k_sol": float(k_sol), "b_sol": float(b_sol),
            "mask_occupancy": mask_occupancy,
            "n_work": int(F_obs.size() - free_np.sum()),
            "n_free": int(free_np.sum()),
            "n_atoms": int(xrs.scatterers().size()),
            "dropped_note": info["dropped_note"]}


def external_reference(pdb, cif, d_min, log=sys.stdout, select="all"):
    """What this dataset supports, measured without any lunus code.

    Two numbers, and the harness's output cannot be read without them:

      * mmtbx.f_model's own bulk solvent and scaling on the SAME model. That is
        the R a correct solvent model reaches here, and it is the honest
        target -- not the deposited R, which additionally had every coordinate
        and ADP refined against this data.
      * the same, on the model converted to isotropic ADPs, which is what
        lunus.sf represents without --aniso-adp. The gap between the two rows
        is the cost of the anisotropic kernel being switched off, and on a
        1.04 A structure it is large.
    """
    print("\nexternal reference (cctbx/mmtbx; no lunus.sf code involved)",
          file=log)
    print("  %-34s %8s %8s %8s %8s"
          % ("", "R-work", "R-free", "k_sol", "b_sol"), file=log)
    for name, isotropic in (("deposited model, anisotropic ADPs", False),
                            ("same model, isotropic ADPs", True)):
        r = mmtbx_fit(pdb, cif, d_min, isotropic=isotropic, select=select)
        print("  %-34s %8.4f %8.4f %8.3f %8.1f"
              % (name, r["R-work"], r["R-free"], r["k_sol"], r["b_sol"]),
              file=log)
    print("  Which row is the target depends on --aniso-adp: without it the",
          file=log)
    print("  second, with it the first. The gap between them is the",
          file=log)
    print("  anisotropic ADPs.", file=log)


# --------------------------------------------------------------------- report

def shell_table(d, F_obs, model_no_solvent, model_solvent, work, free,
                n_shells, solvent_share, log=sys.stdout):
    """R by resolution shell, lowest resolution first.

    THE LAST COLUMN IS THE ONE THAT SAYS WHERE THE SOLVENT ACTS. It is
    |k_sol exp(-b_sol s^2/4) F_mask| / |F_protein| (|k_mask[bin] F_mask| /
    |F_protein| for a per-bin solvent fit), the size of the solvent term
    itself, and it dies away by ~4 A as bulk solvent must.

    Read it before reading dR, because dR alone is misleading here: the
    no-solvent column is not a clean baseline but a fit DISTORTED by the
    missing term. With no solvent to absorb the low-resolution deficit the
    overall B compensates, which throws off every shell including the ones
    solvent never reaches -- so dR is large at high resolution while the
    solvent contribution there is nil. That is the scaling recovering, not
    bulk solvent working at 1 A.
    """
    edges = np.percentile(d, np.linspace(0, 100, n_shells + 1))
    print("\n  %-17s %7s  %-15s  %-15s  %8s %8s"
          % ("d range (A)", "n", "R-work  no/with", "R-free  no/with",
             "dR-work", "solvent"), file=log)
    for lo, hi in zip(edges[:-1][::-1], edges[1:][::-1]):
        sel = (d >= lo) & (d <= hi)
        rows = []
        for mask in (sel & work, sel & free):
            if mask.sum() < 2:
                rows.append((float("nan"), float("nan")))
                continue
            rows.append((r_factor(F_obs[mask], model_no_solvent[mask]),
                         r_factor(F_obs[mask], model_solvent[mask])))
        (rw0, rw1), (rf0, rf1) = rows
        print("  %7.2f - %-7.2f %7d  %6.4f / %6.4f  %6.4f / %6.4f  %+8.4f %7.1f%%"
              % (hi, lo, int(sel.sum()), rw0, rw1, rf0, rf1, rw1 - rw0,
                 100.0 * float(solvent_share[sel].mean())), file=log)


def write_mtz_from_run(args, result, F_obs_array, free_flags, fit, binned_row,
                       F_protein, F_mask, inv_d2, s_cart, density, mask,
                       volume, orth, blur, free_flags_file=None):
    """--write-mtz: the model of `fit` (the binned row's when binned_row,
    else the main fit's) written by write_model_mtz, with the fit in the
    history. The mask level and source come from result, the dtype and
    device from inv_d2. Returns the path written."""
    dtype, device = inv_d2.dtype, inv_d2.device
    mask_level, mask_source = result["mask_level"], result["mask_source"]
    from lunus.sf.solvent_torch import f_solvent
    from lunus.sf.structure_factor_torch import compute_fcalc, reciprocal_inv_d2

    if binned_row:
        which = "binned row (%s, solvent %s)" % (result["bin_scheme"],
                                                 result["solvent_scheme binned"])
        rw, rf = result["R-work binned"], result["R-free binned"]
        k_sol, b_sol = result["k_sol binned"], result["b_sol binned"]
    else:
        reason = ("--scale-bins makes the main fit binned" if args.scale_bins
                  else "the binned row is off (--also-scale-bins 0)")
        which = "main fit (%s, solvent %s)" % (result["main_bin_scheme"] or "single scale",
                                               result["solvent_scheme"])
        rw, rf = result["R-work"], result["R-free"]
        k_sol, b_sol = result["k_sol"], result["b_sol"]
        print("\n--write-mtz: writing the main fit's model, since %s" % reason)
    F_model = model_structure_factors(fit, F_protein, F_mask, inv_d2, s_cart)

    complete = components = None
    comp_obs = {"F_protein": F_protein, "F_mask": F_mask}
    if args.mtz_complete:
        full = F_obs_array.complete_set(d_min=F_obs_array.d_min())
        hkl_all = torch.tensor(np.array(full.indices()), dtype=torch.long, device=device)
        inv_d2_all = reciprocal_inv_d2(orth, hkl_all).to(dtype)
        s_all = hkl_all.to(dtype) @ torch.linalg.inv(orth)
        with torch.no_grad():
            Fp_all = compute_fcalc(density, volume, hkl_all, orth, blur=blur)
            Fm_all = f_solvent(mask, volume, hkl_all, orth)
        fit_all = dict(fit)
        if fit.get("bins") is not None:
            d_all = np.array(full.d_spacings().data())
            fit_all["bins"] = torch.as_tensor(
                nearest_bins(d_all, np.array(F_obs_array.d_spacings().data()),
                             fit["bins"].cpu().numpy()),
                dtype=fit["bins"].dtype, device=fit["bins"].device)
        Fmod_all = model_structure_factors(fit_all, Fp_all, Fm_all, inv_d2_all, s_all)
        # the measured reflections keep exactly the values written as FP
        from cctbx import miller
        pairs = miller.match_indices(full.indices(), F_obs_array.indices()).pairs()
        i_all = torch.as_tensor(np.array(pairs.column(0), dtype=np.int64), device=device)
        i_obs = torch.as_tensor(np.array(pairs.column(1), dtype=np.int64), device=device)
        for big, small in ((Fmod_all, F_model), (Fp_all, F_protein), (Fm_all, F_mask)):
            big[i_all] = small[i_obs].to(big.dtype)
        complete = {"set": full, "F_model": Fmod_all.cpu().numpy()}
        comp_obs = {"F_protein": Fp_all, "F_mask": Fm_all}
    if args.mtz_components:
        components = {k: v.detach().cpu().numpy() for k, v in comp_obs.items()}

    kb = (torch.exp(fit["log_k_bins"]).tolist() if fit.get("log_k_bins") is not None
          else [float(torch.exp(fit["log_k"]))])
    km = fit["k_mask_bins"].tolist() if fit.get("k_mask_bins") is not None else None
    level = ("%g sphere overlaps" % mask_level if mask_level is not None
             else "gemmi mask" if math.isnan(result["mask_cutoff"])
             else "calibrated, cutoff %.4g" % result["mask_cutoff"])

    def wrap(label, values, fmt="%.5g"):
        lines, line = [], label
        for v in values:
            item = " " + fmt % v
            if len(line) + len(item) > 78:
                # MTZ readers strip leading blanks, so a continuation names
                # its list rather than relying on indentation
                lines.append(line)
                line = label.split(" (")[0].split(" per ")[0] + " (cont.):"
            line += item
        return lines + [line]

    history = ["lunus.sf fit_solvent_rfactor --write-mtz: scaled model, synthetic data",
               "model %s" % which,
               "FP = |k_total (F_protein + k_mask F_mask)| on the Fobs scale; PHIC its phase",
               "R-work %.4f  R-free %.4f  (%d work, %d free)"
               % (rw, rf, result["n_work"], result["n_free"]),
               "k_sol %.4f  b_sol %.2f%s"
               % (k_sol, b_sol, "" if km is None
                  else "  (nearest exponential; the model is k_mask)")]
    history += wrap("scale B (A^2, %s):" % ("cartesian B11 B22 B33 B12 B13 B23"
                                            if fit["aniso"] else "isotropic"),
                    [float(u) for u in fit["u_params"]], fmt="%.3f")
    history += ["mask: %s source, %s; protein blur %.2f A^2"
               % (mask_source, level, result["protein_blur"])]
    history += wrap("k_total per bin, low to high res:" if len(kb) > 1
                    else "k_total (single scale):", kb)
    if km is not None:
        history += wrap("k_mask per bin, low to high res:", km)
    flags, n_new = (None if free_flags is None else free_flags[0]), 0
    if complete is not None:
        if free_flags is not None:
            flags, n_new = complete_free_flags(free_flags, free_flags_file, complete["set"])
        history.append("FMODEL/PHIFMODEL: complete set to %.3f A, %d reflections"
                       % (F_obs_array.d_min(), complete["set"].size()))
        if n_new:
            history.append("FreeR_flag drawn for %d reflections the file never flagged"
                           % n_new)
    history.append("from %s and %s" % (args.pdb.split("/")[-1], args.cif.split("/")[-1]))
    write_model_mtz(args.write_mtz, F_obs_array, flags,
                    F_model.cpu().numpy(), history, complete=complete,
                    components=components, observed=args.mtz_observed)
    print("\nwrote %s: %s model on %d measured reflections%s"
          % (args.write_mtz, which, F_obs_array.size(),
             "" if complete is None
             else ", FMODEL on %d to %.2f A" % (complete["set"].size(), F_obs_array.d_min())))
    return args.write_mtz


def build_parser():
    try:
        from lunus.sf.solvent_torch import SOURCE_B_DEFAULT
    except ImportError:            # --help without the repo on PYTHONPATH
        SOURCE_B_DEFAULT = 20.0

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb")
    p.add_argument("cif", help="deposited structure factors (mmCIF or MTZ)")
    p.add_argument("--d-min", type=float, default=None,
                   help="truncate the observations (default: use them all)")
    p.add_argument("--max-voxel", type=float, default=MAX_VOXEL_DEFAULT, metavar="A",
                   help="cap the voxel edge: voxels of d_min/3 but no coarser "
                        "than this, in A (default: no cap). With the "
                        "calibration target on its own reference grid a cap "
                        "no longer changes R; it was the default for part of "
                        "2026-11-10 at 0.7 A. See the module docstring")
    p.add_argument("--rate", type=float, default=None,
                   help="override the grid rule with a plain oversampling: "
                        "spacing d_min/(2*rate); the default is 1.5 (d_min/3)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--mask", default="threshold", choices=["threshold", "gemmi"],
                   help="ours, or gemmi's geometric mask through the same fit")
    p.add_argument("--radii-set", default="Refmac",
                   choices=["Refmac", "Cctbx", "VanDerWaals"],
                   help="gemmi radii set, for the calibration target and --mask gemmi")
    p.add_argument("--target-spacing", type=target_spacing_arg,
                   default=TARGET_SPACING_DEFAULT, metavar="A",
                   help="grid spacing the calibration target (gemmi's mask "
                        "occupancy) is measured on, independent of the working "
                        "grid (default %g A, tuned for --radii-set Refmac: "
                        "inside the plateau between the steps of gemmi's "
                        "occupancy at 0.8 and 0.57 A); 'same' measures it on "
                        "the working grid, as every run before 2026-11-10 did"
                        % TARGET_SPACING_DEFAULT)
    p.add_argument("--target-occupancy", type=float, default=None,
                   help="override the gemmi-measured calibration target")
    p.add_argument("--mask-level", type=mask_level_arg, default=MASK_LEVEL_DEFAULT,
                   metavar="auto|calibrate|L",
                   help="where the sphere field is cut: 'calibrate' finds the "
                        "cutoff at which the mask's occupancy matches gemmi's "
                        "geometric mask on the reference grid; a number L cuts "
                        "at L sphere overlaps directly, with no gemmi target and "
                        "no search (--mask-source spheres only); 'auto' "
                        "(default) is %g for a model without hydrogens and "
                        "calibrate for one with them, for another source, or "
                        "for a sphere field with non-default settings"
                        % MASK_LEVEL_FIXED)
    p.add_argument("--taper-frac", type=float, default=0.5,
                   help="taper width as a fraction of the SOLVENT MASK cutoff")
    p.add_argument("--density-cutoff", type=float, default=1e-4, metavar="E",
                   help="atomic density below which an atom's contribution is "
                        "dropped when placing density on the grid, e/A^3. Not the solvent mask "
                        "threshold above -- this is the same quantity xtraj "
                        "calls gemmi_cutoff, and shares its default. Pass 0.01 "
                        "for the old one. See docs/solvent-design.md, "
                        "'What the density cutoff costs'")
    p.add_argument("--protein-blur", default="auto", type=protein_blur_arg,
                   metavar="B",
                   help="extra B (A^2) the protein density is placed with and "
                        "F_protein is corrected for, so sharp atoms do not "
                        "alias on the grid. auto (default): "
                        "cell_utils.recommended_blur, as xtraj; 0 disables it")
    p.add_argument("--mask-source", default="spheres", choices=["spheres", "atoms", "density"],
                   help="what the threshold mask is cut from: 'spheres' (default "
                        "since 2026-11-10), a flat-top soft sphere per atom of half "
                        "its Lennard-Jones sigma (lunus.sf.mask_spheres: OPLS "
                        "united-atom radii by residue and atom name, element "
                        "fallback), with no further smoothing unless --mask-blur "
                        "is given; 'atoms', an atom-presence field in which every "
                        "atom is a unit-weight Gaussian at --mask-source-b (the "
                        "default for part of 2026-11-10); or 'density', the "
                        "electron density itself, as every run before")
    p.add_argument("--sphere-edge", type=positive_float_arg("--sphere-edge"), default=None,
                   metavar="W", help="edge width of the soft spheres, A (default 0.5)")
    p.add_argument("--sphere-offset", type=float, default=0.0, metavar="A",
                   help="added to every sphere's radius (sigma/2 + offset); the "
                        "geometric mask's probe-minus-shrink, swept (default 0)")
    p.add_argument("--sphere-radii", default="ua", choices=["ua", "element"],
                   help="'ua': OPLS united-atom Lennard-Jones radii (sigma/2) by residue "
                        "and atom name; 'element': the element fallback for every "
                        "atom (to separate the typing from the shape)")
    p.add_argument("--mask-source-b", type=positive_float_arg("--mask-source-b"),
                   default=SOURCE_B_DEFAULT, metavar="B",
                   help="the B every atom is placed with in the 'atoms' mask "
                        "source (default %g A^2). What matters is the TOTAL "
                        "width, this plus --mask-blur: totals of 70-110 A^2 "
                        "measure alike at low resolution and 120 is marginally "
                        "better on SW40" % SOURCE_B_DEFAULT)
    p.add_argument("--mask-blur", type=float, default=None, metavar="B",
                   help="smooth the mask's source by exp(-B s^2/4) before "
                        "thresholding (default: 0 for --mask-source spheres, "
                        "whose soft edge is the smoothing; SolventModel's own "
                        "100 A^2 for atoms and density)")
    p.add_argument("--aniso-adp", action="store_true",
                   help="use each atom's ANISOTROPIC DISPLACEMENT PARAMETER "
                        "instead of its isotropic equivalent. Distinct from "
                        "--aniso, which is the overall SCALE tensor: that is a "
                        "fitted correction to |F|, this is the atomic model. "
                        "The gap --reference quantifies is this one -- on 7FPV "
                        "mmtbx reaches 0.130/0.151 with ADPs and 0.179/0.194 "
                        "without.")
    p.add_argument("--reference", action="store_true",
                   help="also report what mmtbx reaches on the same model")
    p.add_argument("--aniso", action="store_true",
                   help="fit a full anisotropic scale tensor instead of one B")
    p.add_argument("--scale-bins", type=scale_bins_arg, default=0,
                   metavar="log|N",
                   help="refine one overall scale per resolution bin instead "
                        "of a single one, as mmtbx does. 'log' uses mmtbx's "
                        "own logarithmic bins; N uses N equal-count bins (how "
                        "binned numbers before 2026-11-09 were measured, "
                        "with N = 20). Default 0, off")
    p.add_argument("--also-scale-bins", type=scale_bins_arg, default="log",
                   metavar="log|N",
                   help="after the main fit, fit again with scale bins ('log' "
                        "or N, as --scale-bins) and "
                        "report that R alongside it, leaving the main fit and "
                        "everything printed from it unchanged. This is how "
                        "pdb_rfactor.py shows how much of a gap to mmtbx is "
                        "the scaling model, and the model --write-mtz writes. "
                        "Default log (mmtbx's bins) since 2026-11-11, as "
                        "pdb_rfactor already ran it; 0 turns it off")
    p.add_argument("--write-mtz", default=None, metavar="PATH",
                   help="write the scaled model as synthetic data: FP (the "
                        "model's amplitude on the Fobs scale), SIGFP (the "
                        "observed sigmas), FreeR_flag (the free set as read, in "
                        "the CCP4 convention: test 0, work bins 1..N-1) and "
                        "PHIC (the model's phase), on the reflections the fit "
                        "used, with the fit in the MTZ history. The model is "
                        "the binned row's (--also-scale-bins); without that "
                        "row, the main fit's")
    p.add_argument("--mtz-complete", action="store_true",
                   help="with --write-mtz, also write FMODEL/PHIFMODEL on "
                        "every reflection to d_min, with the fitted scales "
                        "carried to the unmeasured ones (the nearest fitted "
                        "reflection's bin) and FreeR_flag extended to them at "
                        "the observed fraction; FP/SIGFP stay on the measured "
                        "reflections")
    p.add_argument("--mtz-components", action="store_true",
                   help="with --write-mtz, also write the unscaled F_protein "
                        "and F_mask (FPROTEIN/PHIFPROTEIN, FMASK/PHIFMASK)")
    p.add_argument("--mtz-observed", action="store_true",
                   help="with --write-mtz, also write the observations the "
                        "fit used (FOBS/SIGFOBS)")
    p.add_argument("--solvent-bins", default="auto", choices=["auto", "on", "off"],
                   help="with scale bins, fit one solvent scale per bin "
                        "(mmtbx's k_mask) instead of the single exponential "
                        "k_sol exp(-b_sol s^2/4). 'auto' (default) turns it on "
                        "for log bins only, which is also the only binning "
                        "mmtbx does it on: equal-count bins put every "
                        "reflection to about 3 A in ONE bin, and one flat "
                        "solvent scale over that range is worse than the "
                        "exponential (7FPV, 20 bins: R-free +0.0035). 'off' "
                        "reproduces binned numbers recorded before 2026-11-10")
    p.add_argument("--no-expand-symmetry", action="store_true",
                   help="the model already contains every symmetry copy")
    p.add_argument("--no-special-position-weights", action="store_true",
                   help="take the model exactly as deposited: occupancies "
                        "rather than cctbx's symmetry weights (occupancy x "
                        "multiplicity / order_z), and no atom moved onto a "
                        "symmetry element or its ANISOU symmetrized. For a "
                        "complete cell with its space group in CRYST1 that is "
                        "expanded as a fold, every copy of the asymmetric unit "
                        "being its own. Implied by --no-expand-symmetry")
    p.add_argument("--select", default="all", metavar="SEL",
                   help="an iotbx atom selection; only the atoms it keeps "
                        "are used, by this fit and by --reference alike, so "
                        "both methods see the same model (\"not element H\" "
                        "drops deposited hydrogens). The gemmi calibration "
                        "target and --mask gemmi are measured on the file as "
                        "deposited; gemmi's masker ignores hydrogens itself, "
                        "so for that selection the two agree. Default: all")
    p.add_argument("--shells", type=int, default=10)
    return p


def run(args):
    """The whole fit, from a parsed namespace. Returns the overall numbers
    (R-work/R-free with and without solvent, k_sol, b_sol, k_overall) as a
    dict so that tools/pdb_rfactor.py can tabulate them; everything else is
    printed as it goes."""
    timing = {}
    t_start = time.perf_counter()
    from lunus.sf.cell_utils import grid_shape_for_resolution, orth_matrix
    from lunus.sf.solvent_torch import (
        MASK_BLUR_DEFAULT, SolventMaskWarning, calibrate_cutoff, f_solvent,
        mask_occupancy, mask_source_weights, shell_voxels, solvent_mask,
    )
    explicit_level = args.mask_level not in ("calibrate", "auto")
    if explicit_level and (args.mask == "gemmi" or args.mask_source != "spheres"):
        # a level is in sphere overlaps, a unit only the sphere field has
        raise ValueError("--mask-level %g needs --mask threshold and "
                         "--mask-source spheres; use --mask-level calibrate "
                         "for the other sources" % args.mask_level)
    if explicit_level and args.target_occupancy is not None:
        raise ValueError("--target-occupancy is a calibration target, and "
                         "--mask-level %g calibrates nothing; pass one or the "
                         "other" % args.mask_level)
    mtz_extras = [name for name, on in (("--mtz-complete", args.mtz_complete),
                                        ("--mtz-components", args.mtz_components),
                                        ("--mtz-observed", args.mtz_observed)) if on]
    if mtz_extras and not args.write_mtz:
        raise ValueError("%s add columns to the file --write-mtz PATH writes; "
                         "pass --write-mtz too" % " and ".join(mtz_extras))
    if args.write_mtz:
        # checked before the fit, not after it
        if os.path.isdir(args.write_mtz):
            raise ValueError("--write-mtz %s is a folder; name the file to write"
                             % args.write_mtz)
        folder = os.path.dirname(os.path.abspath(args.write_mtz))
        if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
            raise ValueError("--write-mtz %s: %s is not a writable folder"
                             % (args.write_mtz, folder))
    if args.mask_blur is None:
        # the sphere field's 0.5 A edge is all the smoothing it wants
        args.mask_blur = 0.0 if args.mask_source == "spheres" else MASK_BLUR_DEFAULT
    from lunus.sf.structure_factor_torch import compute_fcalc, reciprocal_inv_d2
    from lunus.sf.symmetry_torch import adjust_grid_for_symmetry

    device = args.device
    dtype = getattr(torch, args.dtype)

    print("=" * 78)
    print("bulk solvent against deposited amplitudes")
    print("=" * 78)

    with timed(timing, "read model"):
        xrs, elements, b_per_atom, occ, frac, n_aniso, u_cart, is_aniso = \
            read_model(args.pdb, special_position_weights=use_special_position_weights(
                args.no_special_position_weights, args.no_expand_symmetry),
                select=args.select, log=sys.stdout)
    # log=sys.stdout on both readers: the stdout of the call, not the one
    # bound at import, so the model summary and the dropped-reflection notes
    # land where everything else run() prints does (print_timing below, and
    # fit_scales's sink, for the same reason).
    obs_info = {}
    with timed(timing, "read observations"):
        F_obs_array, free_np = read_observations(
            args.cif, d_min=args.d_min, crystal_symmetry=xrs.crystal_symmetry(),
            log=sys.stdout, info=obs_info)
    d_np = np.array(F_obs_array.d_spacings().data())
    hkl_np = np.array(F_obs_array.indices())
    fobs_np = np.array(F_obs_array.data())
    print("%s: %d amplitudes, %.2f-%.2f A, %d free (%.1f%%)"
          % (args.cif, len(fobs_np), d_np.max(), d_np.min(), free_np.sum(),
             100.0 * free_np.mean()))

    cell = xrs.unit_cell().parameters()
    d_min_grid = args.d_min if args.d_min is not None else float(d_np.min())
    rate, grid_rule = grid_rate(d_min_grid, args.rate, args.max_voxel)
    print("grid: " + grid_rule)
    grid = adjust_grid_for_symmetry(
        grid_shape_for_resolution(cell[0], cell[1], cell[2], d_min_grid,
                                  rate=rate),
        [np.array(o.r().as_double()).reshape(3, 3) for o in xrs.space_group()],
        [np.array(o.t().as_double()) for o in xrs.space_group()])
    print("cell %.3f %.3f %.3f %.1f %.1f %.1f, %s, grid %s (%.1f M voxels), %s"
          % (cell + (xrs.space_group().type().lookup_symbol(), grid,
                     np.prod(grid) / 1e6, args.dtype)))

    b_sharpest = sharpest_b(b_per_atom, u_cart if args.aniso_adp else None, is_aniso,
                            occ=occ)
    blur_is_auto = str(args.protein_blur).lower() == "auto"
    orth_np = orth_matrix(*cell)
    blur = protein_blur(args.protein_blur, b_sharpest, grid, orth_np, d_min_grid)
    if blur > 0.0:
        print("protein density placed with B + %.1f A^2 (sharpest atom B %.1f, "
              "%s) and F_protein corrected by exp(+B s^2/4)"
              % (blur, b_sharpest, "--protein-blur auto: cell_utils.recommended_blur"
                 if blur_is_auto else "--protein-blur"))
    elif blur_is_auto:
        print("protein density placed as is (sharpest atom B %.1f clears the "
              "grid's blur target)" % b_sharpest)
    else:
        print("protein density placed as is (--protein-blur 0; sharpest atom "
              "B %.1f)" % b_sharpest)

    density, orth, volume = build_density(
        xrs, elements, b_per_atom, occ, frac, grid, device, dtype,
        expand_symmetry=not args.no_expand_symmetry,
        u_cart=u_cart if args.aniso_adp else None, is_aniso=is_aniso,
        cutoff=args.density_cutoff, timing=timing, blur=blur)

    # THE MASK'S SOURCE: an atom-presence field, every atom a unit-weight
    # Gaussian at one B, placed from the same coordinates with the same
    # protein blur and symmetry expansion; F_protein stays the density's.
    # solvent_torch.mask_source_weights says why and what it measured.
    # --mask gemmi thresholds nothing, so no field is placed for it.
    mask_density = density
    mask_source = args.mask_source if args.mask != "gemmi" else "none (--mask gemmi)"
    # the B already placed into the mask's field: the protein blur, except
    # for the sphere field, which is placed without it
    source_blur = blur
    sphere_edge = None
    if args.mask == "gemmi":
        print("mask source: none (--mask gemmi uses that mask as it is)")
    elif args.mask_source == "atoms":
        coeffs = scattering_table(sorted(set(elements)), log=io.StringIO())
        m_occ = np.asarray(occ, dtype=float) * mask_source_weights(elements, coeffs)
        m_b = np.full(len(elements), float(args.mask_source_b))
        with timed(timing, "mask source field", device):
            mask_density, _, _ = build_density(
                xrs, elements, m_b,
                m_occ, frac, grid, device, dtype,
                expand_symmetry=not args.no_expand_symmetry,
                u_cart=None, is_aniso=np.zeros(len(elements), dtype=bool),
                cutoff=args.density_cutoff, blur=blur, log=io.StringIO())
        print("mask source: atom-presence field, %d atoms of unit weight at "
              "B = %.1f A^2 (--mask-source atoms)" % (len(elements), args.mask_source_b))
    elif mask_source == "spheres":
        from lunus.sf import mask_spheres
        from lunus.sf.density_torch import compute_density
        from lunus.sf.symmetry_torch import build_grid_ops_from_cctbx, symmetrize_sum

        labels = [sc.label for sc in xrs.scatterers()]
        radii = mask_spheres.sphere_radii(labels, elements, offset=args.sphere_offset,
                                          by_element=args.sphere_radii == "element")
        sphere_edge = (args.sphere_edge if args.sphere_edge is not None
                       else mask_spheres.EDGE_WIDTH_DEFAULT)
        source_blur = 0.0
        with timed(timing, "mask source field", device):
            sA, slam, soff, srad, staper, sidx = mask_spheres.sphere_kernels(
                radii, grid, orth.cpu().numpy(), width=sphere_edge, device=device, dtype=dtype)
            mask_density = compute_density(
                torch.tensor(np.asarray(frac), dtype=dtype, device=device), sidx,
                torch.tensor(np.asarray(occ), dtype=dtype, device=device),
                sA, slam, soff, srad, grid, orth, staper, compile_core=False)
            if not args.no_expand_symmetry:
                mask_density = symmetrize_sum(
                    mask_density, build_grid_ops_from_cctbx(xrs.space_group(), grid))
        vals, counts = np.unique(np.round(radii, 3), return_counts=True)
        print("mask source: soft spheres (--mask-source spheres, %s radii, offset %.2f, "
              "edge %.2f A): %d atoms, radii %s"
              % (args.sphere_radii, args.sphere_offset, sphere_edge, len(radii),
                 ", ".join("%.2f A x%d" % (v, c) for v, c in zip(vals, counts))))
    else:
        print("mask source: the electron density (--mask-source density)")

    hkl = torch.tensor(hkl_np, dtype=torch.long, device=device)
    F_obs = torch.tensor(fobs_np, dtype=dtype, device=device)
    work = ~free_np
    inv_d2 = reciprocal_inv_d2(orth, hkl).to(dtype)
    # s = Mstar h in A^-1, with Mstar = inv(orth).T, so s_cart = hkl @ inv(orth).
    # The same quantity reciprocal_inv_d2 squares and sums; kept as a vector
    # here because the anisotropic scale needs its direction, not just |s|.
    s_cart = hkl.to(dtype) @ torch.linalg.inv(orth)

    with torch.no_grad(), timed(timing, "F_protein (fft)", device):
        F_protein = compute_fcalc(density, volume, hkl, orth, blur=blur)

    # ---- the mask
    print("\nsolvent mask")
    # the sphere-field settings MASK_LEVEL_FIXED was measured with; auto
    # calibrates when any differs
    nondefault_field = [name for name, changed in (
        ("--sphere-offset", args.sphere_offset != 0.0),
        ("--sphere-edge", args.sphere_edge not in (None, mask_spheres.EDGE_WIDTH_DEFAULT)
         if mask_source == "spheres" else False),
        ("--sphere-radii", args.sphere_radii != "ua"),
        ("--mask-blur", args.mask_blur != 0.0),
        ("--taper-frac", args.taper_frac != 0.5)) if changed]
    mask_level, level_reason = mask_level_rule(
        args.mask_level, args.mask, args.mask_source, elements, args.target_occupancy,
        nondefault_field=nondefault_field)
    fixed_level = mask_level is not None
    if args.mask != "gemmi":
        print("  mask level: %s (%s)"
              % ("%g sphere overlaps" % mask_level if fixed_level else "calibrated",
                 level_reason))
    with timed(timing, "gemmi mask (cpu)"):
        occ_working, gemmi_grid = gemmi_mask_occupancy(args.pdb, grid, args.radii_set)
        if fixed_level:
            # the cut is a fixed level of the sphere field: nothing to match,
            # so the reference-grid mask is not built at all
            target, target_source = None, "none (fixed level %g)" % mask_level
        elif args.target_occupancy is not None:
            target, target_source = args.target_occupancy, "--target-occupancy"
        elif args.mask == "gemmi":
            # --mask gemmi calibrates nothing: the mask in use is this one,
            # so there is no reference-grid target to build or to warn about;
            # the "target" reported is simply that mask's occupancy
            target, target_source = occ_working, "none (--mask gemmi)"
        elif args.target_spacing == "same":
            target, target_source = occ_working, "gemmi, working grid"
        else:
            target, target_grid = gemmi_mask_occupancy_at_spacing(
                args.pdb, args.target_spacing, args.radii_set)
            target_source = "gemmi, %g A reference grid" % args.target_spacing
    print("  gemmi %s geometric mask on the same model and grid: occupancy %.4f"
          % (args.radii_set, occ_working))
    if fixed_level:
        print("  calibration target: none (the sphere field is cut at %g)"
              % mask_level)
    elif args.target_occupancy is not None:
        print("  calibration target: occupancy %.4f (--target-occupancy)" % target)
    elif args.mask == "gemmi":
        print("  calibration target: none (--mask gemmi uses that mask as it is)")
    elif args.target_spacing == "same":
        print("  calibration target: that occupancy (--target-spacing same)")
    else:
        actual = max(cell[i] / target_grid[i] for i in range(3))
        print("  calibration target: occupancy %.4f of the same mask on a %g A "
              "reference grid %s (actual spacing up to %.3f A), independent "
              "of the working grid"
              % (target, args.target_spacing, target_grid, actual))
        r_shrink = SHRINK_RADIUS.get(args.radii_set)
        near = (shrink_step_distance(orth_np, target_grid, r_shrink)
                if r_shrink is not None else None)
        if near is not None:
            warnings.warn(
                "a voxel edge or face diagonal of the reference grid (%.3f A) "
                "is within 5%% of gemmi's %s shrink radius (%.2f A), where its "
                "mask occupancy steps as the shrink stencil gains a neighbour; "
                "the target is then sensitive to the cell's rounding of "
                "--target-spacing. Choose a spacing whose edges and diagonals "
                "keep clear of %.2f A."
                % (near, args.radii_set, r_shrink, r_shrink),
                SolventMaskWarning, stacklevel=2)
    if args.mask == "gemmi":
        # whatever the target line said: the mask in use is gemmi's on the
        # working grid, which keeps the step dependence on the voxel size
        r_shrink = SHRINK_RADIUS.get(args.radii_set)
        coarsest = max(cell[i] / grid[i] for i in range(3))
        if r_shrink is not None and coarsest >= r_shrink:
            warnings.warn(
                "the working grid's voxels are no finer than gemmi's %s shrink "
                "radius (%.2f A) along at least one axis (up to %.2f A), where "
                "gemmi's mask loses its shrink step and its occupancy drops "
                "below the plateau the threshold mask is calibrated on (7FPV at "
                "3 A: 0.19 against 0.26); for --mask gemmi beyond ~2.1 A use "
                "--max-voxel 0.7" % (args.radii_set, r_shrink, coarsest),
                SolventMaskWarning, stacklevel=2)
    if args.select != "all":
        print("  (gemmi's mask is built from the file as deposited, not from "
              "--select %r; its masker ignores hydrogens)" % args.select)

    if args.mask == "gemmi":
        mask = torch.tensor(np.array(gemmi_grid), dtype=dtype, device=device)
        print("  using gemmi's mask directly (--mask gemmi)")
        # the occupancy of the mask in use, which is gemmi's on the working
        # grid, not the target's reference-grid (or overridden) value
        occupancy, shell, cutoff = occ_working, 0, float("nan")
    else:
        with torch.no_grad(), timed(timing, "threshold mask", device):
            # the density already carries the protein blur, which is the
            # same operation as this smoothing (SolventModel.mask_source
            # says why), so --mask-blur is a total and only the remainder
            # is applied: the mask the fit sees is the shipped one up to
            # the aliasing the protein blur removes. Nothing here can
            # sharpen a density placed with more than --mask-blur.
            # the sphere field is placed without the protein blur, so it
            # takes the whole --mask-blur (source_blur 0)
            source = blur_density(mask_density, orth,
                                  max(args.mask_blur - source_blur, 0.0))
            if source_blur > args.mask_blur:
                warnings.warn(
                    "the protein blur (%.1f A^2) exceeds --mask-blur (%.1f), "
                    "so the mask is smoother than asked for; lower "
                    "--protein-blur or accept the wider effective probe"
                    % (blur, args.mask_blur), SolventMaskWarning, stacklevel=2)
            if mask_source == "spheres":
                if args.mask_blur:
                    print("  sphere field smoothed by B = %.1f A^2 before "
                          "thresholding (it carries no protein blur)" % args.mask_blur)
            elif args.mask_blur:
                sigma = np.sqrt(args.mask_blur / (8.0 * np.pi ** 2))
                if blur <= 0.0:
                    already = ""
                elif blur < args.mask_blur:
                    already = " (%.1f of it already in the placed density)" % blur
                else:
                    already = (" -- but the placed density already carries "
                               "%.1f A^2, and that is what the threshold sees"
                               % blur)
                print("  density smoothed by B = %.1f A^2 (sigma %.2f A) "
                      "before thresholding%s" % (args.mask_blur, sigma, already))
            elif blur > 0.0:
                print("  --mask-blur 0, but the placed density carries the "
                      "protein blur B = %.1f A^2, so the threshold sees that "
                      "smoothing (--protein-blur 0 removes it)" % blur)
            if fixed_level:
                cutoff = float(mask_level)
            else:
                cutoff = calibrate_cutoff(source, target,
                                          taper_width_frac=args.taper_frac)
            mask = solvent_mask(source, cutoff, args.taper_frac * cutoff)
        occupancy, shell = float(mask_occupancy(mask)), shell_voxels(mask)
        cutoff = float(cutoff)
        print("  threshold mask %s: cutoff %.4g %s (%.3f of peak %.3f)"
              % ("at a fixed level (--mask-level)" if fixed_level
                 else "calibrated to it", cutoff,
                 {"atoms": "atoms/A^3", "spheres": "sphere overlaps"}.get(
                     args.mask_source, "e/A^3"),
                 cutoff / float(source.max()), float(source.max())))
        if fixed_level and not 0.05 < occupancy < 0.95:
            warnings.warn(
                "the mask level %g calls %.1f%% of the cell solvent, which no "
                "protein crystal has; the level is wrong for this model, or the "
                "model is not what the sphere field expects"
                % (mask_level, 100.0 * occupancy),
                SolventMaskWarning, stacklevel=2)
        print("  occupancy %.4f, taper shell %d voxels (%.2f%% of the grid)"
              % (occupancy, shell, 100.0 * shell / np.prod(grid)))

    with torch.no_grad(), timed(timing, "F_mask (fft)", device):
        F_mask = f_solvent(mask, volume, hkl, orth)
    with torch.no_grad():
        low = d_np > 5.0
        print("  |F_mask|/|F_protein|: %.4f overall, %.4f beyond 5 A"
              % (float(F_mask.abs().mean() / F_protein.abs().mean()),
                 float(F_mask[low].abs().mean() / F_protein[low].abs().mean())))
        if args.mask != "gemmi":
            with timed(timing, "gemmi F_mask + compare", device):
                F_gemmi = f_solvent(
                    torch.tensor(np.array(gemmi_grid), dtype=dtype, device=device),
                    volume, hkl, orth)
                compare_masks(mask, gemmi_grid, F_mask, F_gemmi, d_np)

    # ---- the fits
    print("\nfitting %d work reflections (%d free held out)"
          % (work.sum(), free_np.sum()))
    bins = assign_scale_bins(args.scale_bins, inv_d2, d_np)
    if bins is not None:
        print("overall scale per resolution bin: %s"
              % describe_bins(args.scale_bins, bins))

    def per_bin_solvent(spec):
        """The --solvent-bins rule: on for log bins unless told otherwise."""
        return {"on": True, "off": False}.get(args.solvent_bins, spec == "log")
    print("no solvent:")
    with timed(timing, "fit, no solvent", device):
        fit0 = fit_scales(F_obs, F_protein, None, inv_d2, s_cart, work, args.aniso,
                          fit_solvent=False, bins=bins)
    print("with solvent:")
    with timed(timing, "fit, with solvent", device):
        fit1 = fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work, args.aniso,
                          fit_solvent=True, bins=bins,
                          solvent_bins=per_bin_solvent(args.scale_bins))

    m0 = model_amplitudes(fit0, F_protein, None, inv_d2, s_cart)
    m1 = model_amplitudes(fit1, F_protein, F_mask, inv_d2, s_cart)

    print("\nfitted parameters")
    print("  %-22s %14s %14s" % ("", "no solvent", "with solvent"))
    print("  %-22s %14.4f %14.4f"
          % ("k_overall", float(torch.exp(fit0["log_k"])),
             float(torch.exp(fit1["log_k"]))))
    if args.aniso:
        names = ["B11", "B22", "B33", "B12", "B13", "B23"]
        for i, name in enumerate(names):
            print("  %-22s %14.2f %14.2f"
                  % (name + " (A^2, cart)", float(fit0["u_params"][i]),
                     float(fit1["u_params"][i])))
    else:
        print("  %-22s %14.2f %14.2f"
              % ("B_overall (A^2)", float(fit0["u_params"][0]),
                 float(fit1["u_params"][0])))
    print("  %-22s %14s %14.4f"
          % ("k_sol (e/A^3)", "-", float(fit1["k_sol"])))
    print("  %-22s %14s %14.4f"
          % ("b_sol (A^2)", "-", float(fit1["b_sol"])))
    print("  %-22s %14s %14s"
          % ("conventional", "-", "0.35 / 46"))

    print("\noverall")
    result = {"n_atoms": len(elements), "protein_blur": float(blur),
              "rate": float(rate),
              "max_voxel": None if args.max_voxel is None else float(args.max_voxel),
              "grid_rule": grid_rule,     # says whether the cap or --rate set the grid
              "target_occupancy": None if target is None else float(target),
              "mask_level": mask_level,
              "mask_level_rule": level_reason,
              "mask_source": mask_source,
              "mask_source_b": float(args.mask_source_b) if mask_source == "atoms" else None,
              "sphere_edge": sphere_edge,
              "target_source": target_source,
              "target_spacing": None if fixed_level else args.target_spacing,
              "gemmi_occupancy_working_grid": float(occ_working),
              "grid": tuple(int(n) for n in grid),
              "b_sharpest": b_sharpest, "mask_cutoff": cutoff,
              "mask_occupancy": occupancy,
              "k_sol": float(fit1["k_sol"]), "b_sol": float(fit1["b_sol"]),
              "k_overall": float(torch.exp(fit1["log_k"])),
              "n_work": int(work.sum()), "n_free": int(free_np.sum()),
              "main_scale_bins": 0 if bins is None else int(bins.max()) + 1,
              "main_bin_scheme": (None if bins is None
                                  else describe_bins(args.scale_bins, bins)),
              "solvent_scheme": solvent_scheme(fit1)}
    for name, sel in (("R-work", work), ("R-free", free_np)):
        if sel.sum() < 2:
            result[name + " no solvent"] = result[name] = float("nan")
            continue
        r0, r1 = r_factor(F_obs[sel], m0[sel]), r_factor(F_obs[sel], m1[sel])
        result[name + " no solvent"], result[name] = r0, r1
        print("  %-8s no solvent %.4f   with solvent %.4f   change %+.4f (%+.1f%%)"
              % (name, r0, r1, r1 - r0, 100.0 * (r1 - r0) / r0))

    fitb = None
    if args.also_scale_bins and not args.scale_bins:
        # Reuses F_protein and F_mask, so this costs a scale fit, not a
        # structure-factor calculation. Reported, and the model --write-mtz
        # writes, but never used for anything else below: the single-scale
        # fit stays the measure of the model.
        bins_b = assign_scale_bins(args.also_scale_bins, inv_d2, d_np)
        n = int(bins_b.max()) + 1
        # fit_scales's own log defaults to the stdout bound at import, which
        # redirect_stdout does not reach; hand it a sink explicitly so the
        # curve is printed once, below, and not twice.
        with contextlib.redirect_stdout(io.StringIO()), \
                timed(timing, "fit, binned row", device):
            fitb = fit_scales(F_obs, F_protein, F_mask, inv_d2, s_cart, work,
                              args.aniso, fit_solvent=True, bins=bins_b,
                              solvent_bins=per_bin_solvent(args.also_scale_bins),
                              log=io.StringIO())
        mb = model_amplitudes(fitb, F_protein, F_mask, inv_d2, s_cart)
        result.update({"scale_bins": n,
                       "bin_scheme": describe_bins(args.also_scale_bins,
                                                   bins_b),
                       "k_sol binned": float(fitb["k_sol"]),
                       "b_sol binned": float(fitb["b_sol"])})
        for name, sel in (("R-work", work), ("R-free", free_np)):
            rb = (r_factor(F_obs[sel], mb[sel]) if sel.sum() >= 2
                  else float("nan"))
            result[name + " binned"] = rb
        # The scheme on its own line, so that the line below keeps the format
        # log parsers written before log bins existed already match.
        result["solvent_scheme binned"] = solvent_scheme(fitb)
        print("  binned row: %s, solvent %s" % (result["bin_scheme"],
                                                result["solvent_scheme binned"]))
        kb = torch.exp(fitb["log_k_bins"])
        print("  binned row scales (low to high resolution): %s"
              % " ".join("%.2f" % float(v) for v in kb))
        if fitb.get("k_mask_bins") is not None:
            print("  solvent scale per bin (low to high resolution): %s"
                  % " ".join("%.2f" % float(v) for v in fitb["k_mask_bins"]))
            if fitb["n_refl_exp_fit"]:
                print("  k_sol/b_sol below are the exponential closest to that curve "
                      "over the %d reflections at d >= 4 A, as mmtbx reports them"
                      % fitb["n_refl_exp_fit"])
            else:
                print("  fewer than 2 reflections at d >= 4 A: k_sol below is the "
                      "curve's maximum and b_sol 0, not a fit")
        print("  with %d scale bins instead (--also-scale-bins): R-work %.4f, "
              "R-free %.4f, k_sol %.3f, b_sol %.1f"
              % (n, result["R-work binned"],
                 result["R-free binned"],
                 result["k_sol binned"], result["b_sol binned"]))

    with torch.no_grad():
        # solvent_weight, not k_sol/b_sol: for a per-bin fit those are the
        # exponential nearest the curve, and the column must show the
        # solvent the model applied.
        contribution = solvent_weight(fit1, inv_d2) * F_mask.abs()
        solvent_share = (contribution / F_protein.abs()).cpu().numpy()
    with timed(timing, "tables", device):
        shell_table(d_np, F_obs, m0, m1, work, free_np, args.shells, solvent_share)

    # Equal-count shells put four fifths of a 1.04 A dataset above 2 A, which
    # is precisely where bulk solvent does nothing. The acceptance criterion is
    # about the low-resolution end, so it gets bands of its own.
    print("\nlow resolution, where the acceptance criterion lives")
    print("  %-16s %7s  %-15s  %8s %8s"
          % ("d range (A)", "n", "R-work  no/with", "dR-work", "solvent"))
    bands = [(np.inf, 8.0), (8.0, 5.0), (5.0, 4.0), (4.0, 3.0), (3.0, 2.0)]
    for hi, lo in bands:
        sel = (d_np < hi) & (d_np >= lo) & work
        if sel.sum() < 2:
            continue
        r0, r1 = r_factor(F_obs[sel], m0[sel]), r_factor(F_obs[sel], m1[sel])
        print("  %6.1f - %-7.1f %7d  %6.4f / %6.4f  %+8.4f %7.1f%%"
              % (min(hi, d_np.max()), lo, int(sel.sum()), r0, r1, r1 - r0,
                 100.0 * float(solvent_share[sel].mean())))

    # after the tables, so that a failed write does not lose them
    if args.write_mtz:
        with timed(timing, "write mtz", device):
            write_mtz_from_run(args, result, F_obs_array, obs_info.get("free_flags"),
                               fit1 if fitb is None else fitb, fitb is not None,
                               F_protein, F_mask, inv_d2, s_cart,
                               density, mask, volume, orth, blur,
                               free_flags_file=obs_info.get("free_flags_file"))

    if args.reference:
        with timed(timing, "mmtbx reference"):
            external_reference(args.pdb, args.cif, args.d_min, select=args.select)

    print("\ncaveats, in descending order of how much they cost")
    if args.aniso_adp:
        print("  * %d of %d atoms use their full ANISOTROPIC ADP. The largest"
              % (n_aniso, len(elements)))
        print("    approximation in this pipeline is therefore NOT in play, and")
        print("    the target is the deposited-model row of --reference.")
    else:
        print("  * %d of %d atoms were refined with ANISOTROPIC ADPs and are used"
              % (n_aniso, len(elements)))
        print("    here as their isotropic equivalents. This inflates R at HIGH")
        print("    resolution and is unrelated to the solvent model. It is the")
        print("    largest single reason the overall R here is not the deposited")
        print("    one -- pass --aniso-adp to remove it.")
    print("  * No positional, ADP or occupancy refinement happens here at all:")
    print("    the deposited coordinates are used exactly as given, against")
    print("    which only 3-4 scale parameters are fitted.")
    if bins is not None:
        print("  * The overall scale is fitted per resolution bin (%d bins), so"
              % (int(bins.max()) + 1))
        print("    k_overall above is the single-scale value it started from.")
        print("    The lowest bin overlaps the solvent term and absorbs part")
        print("    of it, which shrinks the no-solvent/with-solvent change.")
        if result["solvent_scheme"] == "per bin":
            print("  * The solvent scale is one value per bin as well (mmtbx's")
            print("    k_mask); the k_sol/b_sol printed for this fit are the")
            print("    exponential closest to that curve at d >= 4 A, as mmtbx")
            print("    reports them. --solvent-bins off restores the exponential.")
    print("  * The fit is unweighted least squares on amplitudes.")
    if args.aniso:
        print("  * The off-diagonal B_cart components are unconstrained. In an")
        print("    ORTHORHOMBIC space group symmetry requires them to be zero,")
        print("    so their fitted magnitudes above are a direct check on the")
        print("    u_aniso code path rather than a modelling result.")
    print("\n  The diagnostic quantity is the CHANGE from solvent and where it")
    print("  falls, not the absolute R. See docs/solvent-design.md, Validation.")
    device_sync(device)
    timing["total"] = time.perf_counter() - t_start
    print_timing(timing, log=sys.stdout)     # the stdout of the call, not of the import
    result["timing"] = dict(timing)
    return result


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
