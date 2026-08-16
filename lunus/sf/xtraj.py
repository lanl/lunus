#!/usr/bin/env python
#
# LIBTBX_SET_DISPATCHER_NAME lunus.xtraj
#
# Read a MD trajectory and output structure factor statistics including diffuse
#
# Michael Wall, Los Alamos National Laboratory
#
# Version 0.1a, July 2018
# Version 0.2a, October 2019
#
# This script depends on CCTBX. Launch using mpirun for parallel execution.

from __future__ import print_function
from iotbx.pdb import hierarchy
from cctbx.array_family import flex
import cProfile
import pstats
import mmtbx.utils
import mmtbx.model
from cctbx import maptbx
import copy
import mdtraj as md
import time
import numpy as np
import scipy.optimize
import os
from libtbx.utils import Keep
from cctbx import crystal
from cctbx import miller
import cctbx.sgtbx
import subprocess
import pickle
import h5py
import math
from scipy.interpolate import CubicSpline
import gemmi

def mpi_enabled():
  return 'OMPI_COMM_WORLD_SIZE' in os.environ.keys()


from cctbx import sgtbx, crystal

def force_exact_reflections(input_array, processed_array, fill_value=0.0):
    """
    Takes a processed (e.g., Laue-averaged) array and maps its data 
    back onto the exact unmerged indices of the original input array.
    
    Args:
        input_array (cctbx.miller.array): The original raw data array.
        processed_array (cctbx.miller.array): The merged/processed array.
        fill_value (float): The value to use if a reflection was dropped 
                            during processing.
                            
    Returns:
        cctbx.miller.array: An array identical to input_array in symmetry, 
                            indices, and order, but containing the processed data.
    """
    
    # 1. Temporarily cast the input array into the processed array's symmetry.
    # This ensures that when we map to the ASU, we hit the exact same wedge 
    # of reciprocal space that the processed array lives in.
    temp_array = input_array.customized_copy(
        crystal_symmetry=processed_array.crystal_symmetry()
    )
    
    # 2. Map the original indices to the target ASU.
    # Because your data is non-anomalous, this naturally collapses Friedel pairs 
    # (e.g., mapping -h,-k,-l to h,k,l) to match the processed array.
    mapped_input = temp_array.map_to_asu()
    
    # 3. Create a fast Python lookup dictionary from the processed array.
    # flex.miller_index behaves as an iterable of tuples, which make perfect dict keys.
    processed_dict = dict(zip(processed_array.indices(), processed_array.data()))
    
    # 4. Iterate through the mapped original indices and look up the new values.
    new_data = flex.double()
    for hkl in mapped_input.indices():
        val = processed_dict.get(hkl, fill_value) 
        new_data.append(val)
        
    # 5. Inject the extracted data back into a copy of the ORIGINAL array framework.
    # This guarantees the returned array has the EXACT indices, original symmetry, 
    # and sort order that you started with.
    final_array = input_array.customized_copy(data=new_data)
    
    return final_array

def symmetry_average(miller_array, space_group_symbol):
    
    # 1. Parse the user's specific space group
    sg_info = sgtbx.space_group_info(space_group_symbol)
    original_sg = sg_info.group()
    
    # 2. Ask cctbx for the Patterson group (Laue symmetry + Lattice centering)
    patterson_sg = original_sg.build_derived_patterson_group()
    
    # 3. Create the target crystal symmetry using the Patterson group
    target_symm = crystal.symmetry(
        unit_cell=miller_array.unit_cell(),
        space_group=patterson_sg 
    )
    
    # 4. Merge as before
    array_with_laue = miller_array.customized_copy(crystal_symmetry=target_symm)
    merged_obj = array_with_laue.merge_equivalents()
    
    return force_exact_reflections(miller_array,merged_obj.array())

def to_aniso(miller_array, apply_symmetry_str="P1", mask_value=np.nan):
    """
    Calculates and subtracts the isotropic component from a cctbx miller_array.

    The isotropic component is calculated by averaging the data in spherical shells
    in reciprocal space. The shell thickness is defined by the length of the reciprocal
    unit cell diagonal (d* of the 1,1,1 reflection). A not-a-knot cubic spline is then
    used to interpolate the background for each specific reflection, which is
    subtracted directly from the input array.

    Args:
        miller_array (cctbx.miller.array): The non-anomalous Miller array to process.
                                           Must contain data of type double.
        apply_symmetry_str (str): The space group symbol to use for downstream averaging.
        mask_value (float): The value used to denote unmeasured/invalid data. 
                            Defaults to np.nan.

    Returns:
        cctbx.miller.array: A new array, modified to contain only the anisotropic 
                            values, merged according to the target symmetry.

    Raises:
        ValueError: If there are fewer than 4 populated resolution shells, making
                    cubic spline interpolation impossible.
    """
    # 1. Extract unit cell and determine reciprocal cell diagonal (shell thickness)
    uc = miller_array.unit_cell()
    shell_thickness = math.sqrt(uc.d_star_sq((1, 1, 1)))

    # Extract d* values (magnitudes in reciprocal space)
    d_star_sq_flex = miller_array.d_star_sq().data()
    d_star_flex = flex.sqrt(d_star_sq_flex)

    # Convert to numpy for vectorized binning
    d_star_np = d_star_flex.as_numpy_array()
    
    # Use .copy() to ensure we don't accidentally mutate the bound C++ memory block
    data_np = miller_array.data().as_numpy_array().copy()

    # 1. Normalize the distances against the shell thickness (rscale equivalent)
    normalized_d_star = d_star_np / shell_thickness

    # 2. Replicate the C rounding logic: (size_t)(val + 0.5)
    bin_indices = np.floor(normalized_d_star + 0.5).astype(int)

    bin_centers = []
    bin_averages = []

    # Get all unique 'r' values generated
    unique_r_bins = np.unique(bin_indices)

    # 3. Average the data
    for r in unique_r_bins:
        # Find all reflections assigned to this radius 'r'
        mask = (bin_indices == r)
        bin_data = data_np[mask]

        # Filter out the mask value (handles both NaN and specific floats)
        if np.isnan(mask_value):
            valid_data = bin_data[~np.isnan(bin_data)]
        else:
            valid_data = bin_data[bin_data != mask_value]

        # Only process shells that contain valid data
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            
            # The center of the bin is exactly r * shell_thickness
            center = r * shell_thickness
            bin_centers.append(center)
            bin_averages.append(mean_val)

    if len(bin_centers) < 4:
        raise ValueError(
            f"Only found {len(bin_centers)} populated shells. "
            "A minimum of 4 is required for cubic spline interpolation."
        )

    # 4. Fit the interpolating function
    spline = CubicSpline(bin_centers, bin_averages, bc_type='not-a-knot', extrapolate=True)

    # 5. Evaluate the background
    isotropic_bg_np = spline(d_star_np)

    # 6. Safe Subtraction: Create a boolean mask of valid data points
    # This prevents the mask_value flag itself from being subtracted from
    if np.isnan(mask_value):
        valid_mask = ~np.isnan(data_np)
    else:
        valid_mask = (data_np != mask_value)

    # Apply the subtraction exclusively to valid reflections
    data_np[valid_mask] -= isotropic_bg_np[valid_mask]
    
    # 7. Convert back to flex and assign
    data_flex = flex.double(data_np)
    miller_array = miller_array.customized_copy(data=data_flex)

    # 8. Apply downstream symmetry merging
    # Note: Assumes symmetry_average() is defined elsewhere in your module
    miller_array = symmetry_average(miller_array, apply_symmetry_str)

    return miller_array

def calc_msd(x):
  d = np.zeros(this_sites_frac.shape)
  msd = 0
  for k in range(3):
    d[:,k] = (this_sites_frac[:,k] - ref_sites_frac[:,k] + x[k] + 0.5)%1.0 - 0.5
    msd += np.sum(d[:,k] * d[:,k])
    return msd

def write_nexus_metadata(h5_file, miller_array):
  """Helper to map cctbx symmetry to NeXus groups."""
  # 1. Create Entry
  entry = h5_file.require_group("entry")
  entry.attrs["NX_class"] = "NXentry"

  # 2. Create Sample/Crystal Group
  sample = entry.require_group("sample")
  sample.attrs["NX_class"] = "NXsample"

  # Extract unit cell and space group
  uc = miller_array.crystal_symmetry().unit_cell().parameters()
  sg = str(miller_array.crystal_symmetry().space_group_info())

  # Write attributes
  dset_uc = sample.create_dataset("unit_cell", data=uc)
  dset_uc.attrs["units"] = "angstrom"
  dset_sg = sample.create_dataset("unit_cell_group", data=sg)
  # 3. Write Reflection Data
  refl = entry.require_group("reflections")
  refl.attrs["NX_class"] = "NXdata"
  
  # Convert flex.miller_index to numpy (N, 3)
  indices = miller_array.indices().as_vec3_double().as_double().as_numpy_array().reshape(-1, 3)
  refl.create_dataset("hkl", data=indices)
  
  refl.create_dataset("diffuse", data=miller_array.data().as_numpy_array())
                        
if __name__=="__main__":
  import sys

  args = sys.argv[1:]

# selection                                                              

  try:
    idx = [a.find("selection")==0 for a in args].index(True)
  except ValueError:
    selection_text = "all"
  else:
    selection_text = args.pop(idx).split("=")[1]

# d_min

  try:
    idx = [a.find("d_min")==0 for a in args].index(True)
  except ValueError:
    d_min = 0.9
  else:
    d_min = float(args.pop(idx).split("=")[1])

# d_max

  try:
    idx = [a.find("d_max")==0 for a in args].index(True)
  except ValueError:
    d_max = 999
  else:
    d_max = float(args.pop(idx).split("=")[1])

# gemmi_cutoff

  try:
    idx = [a.find("gemmi_cutoff")==0 for a in args].index(True)
  except ValueError:
    gemmi_cutoff = 0.01
  else:
    gemmi_cutoff = float(args.pop(idx).split("=")[1])


# nsteps (use in lieu of "last" parameter)

  try:
    idx = [a.find("nsteps")==0 for a in args].index(True)
  except ValueError:
    nsteps = 0
  else:
    nsteps = int(args.pop(idx).split("=")[1])

# stride

  try:
    idx = [a.find("stride")==0 for a in args].index(True)
  except ValueError:
    stride = 1
  else:
    stride = int(args.pop(idx).split("=")[1])

# first frame number (numbering starts at 0)

  try:
    idx = [a.find("first")==0 for a in args].index(True)
  except ValueError:
    first = 0
  else:
    first = int(args.pop(idx).split("=")[1])

# last frame number

  try:
    idx = [a.find("last")==0 for a in args].index(True)
  except ValueError:
    last = 0
  else:
    last = int(args.pop(idx).split("=")[1])

# chunk size (number of frames) for breaking up the trajectory

  try:
    idx = [a.find("chunk")==0 for a in args].index(True)
  except ValueError:
    chunksize = None
  else:
    chunksize = int(args.pop(idx).split("=")[1])

# topology file (typically a .pdb file)

  try:
    idx = [a.find("top")==0 for a in args].index(True)
  except ValueError:
    top_file = "top.pdb"
  else:
    top_file = args.pop(idx).split("=")[1]

# trajectory file (mpirun works with .xtc but not .pdb)

  try:
    idx = [a.find("traj")==0 for a in args].index(True)
  except ValueError:
    traj_file = "traj.xtc"
  else:
    traj_file = args.pop(idx).split("=")[1]

# density_traj (does nothing right now)

  try:
    idx = [a.find("density_traj")==0 for a in args].index(True)
  except ValueError:
    dens_file = None
  else:
    dens_file = args.pop(idx).split("=")[1]

# diffuse

  try:
    idx = [a.find("diffuse")==0 for a in args].index(True)
  except ValueError:
    diffuse_file = "diffuse.hkl"
  else:
    diffuse_file = args.pop(idx).split("=")[1]

# fcalc

  try:
    idx = [a.find("fcalc")==0 for a in args].index(True)
  except ValueError:
    fcalc_file = "fcalc.mtz"
  else:
    fcalc_file = args.pop(idx).split("=")[1]

# icalc

  try:
    idx = [a.find("icalc")==0 for a in args].index(True)
  except ValueError:
    icalc_file = "icalc.mtz"
  else:
    icalc_file = args.pop(idx).split("=")[1]

# Diffuse data file

  try:
    idx = [a.find("ID_file")==0 for a in args].index(True)
  except ValueError:
    diffuse_data_file = None
  else:
    diffuse_data_file = args.pop(idx).split("=")[1]

# filtered frames file

  try:
    idx = [a.find("filtered_file")==0 for a in args].index(True)
  except ValueError:
    filtered_file = None
  else:
    filtered_file = args.pop(idx).split("=")[1]
# density map

#  try:
#    idx = [a.find("density")==0 for a in args].index(True)
#  except ValueError:
#    density_file = "density.ccp4"
#  else:
#    density_file = args.pop(idx).split("=")[1]

# partial_sum (don't divide by nsteps at the end)

  try:
    idx = [a.find("partial_sum")==0 for a in args].index(True)
  except ValueError:
    partial_sum_mode = False
  else:
    partial_sum_str = args.pop(idx).split("=")[1]
    if partial_sum_str == "True":
      partial_sum_mode = True
    else:
      partial_sum_mode = False

# translational fit (align using fractional coordinates)

  try:
    idx = [a.find("fit")==0 for a in args].index(True)
  except ValueError:
    translational_fit = False
  else:
    fit_str = args.pop(idx).split("=")[1]
    if fit_str == "True":
      translational_fit = True
    else:
      translational_fit = False

# Scattering table

  try:
    idx = [a.find("scattering_table")==0 for a in args].index(True)
  except ValueError:
    scattering_table = 'n_gaussian'
  else:
    scattering_table = args.pop(idx).split("=")[1]

    # Unit cell, replaces the one in the top file

  try:
    idx = [a.find("unit_cell")==0 for a in args].index(True)
  except ValueError:
    unit_cell_str = None
  else:
    unit_cell_str = args.pop(idx).split("=")[1]

# Space group, replaces the one in the top file

  try:
    idx = [a.find("space_group")==0 for a in args].index(True)
  except ValueError:
    space_group_str = None
  else:
    space_group_str = args.pop(idx).split("=")[1]

# Space group to use for applying symmetry after calculation

  try:
    idx = [a.find("apply_symmetry")==0 for a in args].index(True)
  except ValueError:
    apply_symmetry_str = "P1"
  else:
    apply_symmetry_str = args.pop(idx).split("=")[1]

# Apply B factor in structure calculations, then reverse after calc

  try:
    idx = [a.find("apply_bfac")==0 for a in args].index(True)
  except ValueError:
    apply_bfac = True
  else:
    apply_bfac_str = args.pop(idx).split("=")[1]
    if apply_bfac_str == "False":
      apply_bfac = False
    else:
      apply_bfac = True

# Calculate f_000 and print it

  try:
    idx = [a.find("calc_f000")==0 for a in args].index(True)
  except ValueError:
    calc_f000 = False
  else:
    calc_f000_str = args.pop(idx).split("=")[1]
    if calc_f000_str == "False":
      calc_f000 = False
    else:
      calc_f000 = True
      
# Do optimization with respect to data

  try:
    idx = [a.find("do_opt")==0 for a in args].index(True)
  except ValueError:
    do_opt = False
  else:
    do_opt_str = args.pop(idx).split("=")[1]
    if do_opt_str == "False":
      do_opt = False
    else:
      do_opt = True

# Calculate correlations using anisotropic component 

  try:
    idx = [a.find("corr_aniso")==0 for a in args].index(True)
  except ValueError:
    corr_aniso = False
  else:
    corr_aniso_str = args.pop(idx).split("=")[1]
    if corr_aniso_str == "True":
      corr_aniso = True
    else:
      corr_aniso = False

  try:
    idx = [a.find("checkpoint")==0 for a in args].index(True)
  except ValueError:
    checkpoint_opt = False
  else:
    checkpoint_opt_str = args.pop(idx).split("=")[1]
    if checkpoint_opt_str == "False":
      checkpoint_opt = False
    else:
      checkpoint_opt = True

# Use topology file B factoris in structure calculations

  try:
    idx = [a.find("use_top_bfacs")==0 for a in args].index(True)
  except ValueError:
    use_top_bfacs = False
  else:
    use_top_bfacs_str = args.pop(idx).split("=")[1]
    if use_top_bfacs_str == "True":
      use_top_bfacs =True 
    else:
      use_top_bfacs = False

# Set nsteps if needed
  
  if (nsteps == 0):
    nsteps = last - first + 1
  elif (last != 0):
    print("Please specify nsteps or last, but not both.")
    raise ValueError()

  last = first + nsteps - 1

# Method (engine) for calculating structure factors

  try:
    idx = [a.find("engine")==0 for a in args].index(True)
  except ValueError:
    engine = "cctbx"
  else:
    engine = args.pop(idx).split("=")[1]

# Device for the torch engine (e.g. "cpu", "cuda", "cuda:0")

  try:
    idx = [a.find("torch_device")==0 for a in args].index(True)
  except ValueError:
    torch_device_str = "cpu"
  else:
    torch_device_str = args.pop(idx).split("=")[1]

# Max atoms processed at once per element group, for the torch engine.
# Lower this if you're still running out of memory; raise it if you have
# memory to spare and want fewer, larger batched calls.

  try:
    idx = [a.find("torch_max_atoms_per_batch")==0 for a in args].index(True)
  except ValueError:
    torch_max_atoms_per_batch = 10000
  else:
    torch_max_atoms_per_batch = int(args.pop(idx).split("=")[1])

# Density taper width (Angstrom) for the torch engine, spanning the last
# taper_width of the cutoff radius, replacing gemmi's hard cutoff (see
# density_torch.py for why: hard cutoffs make autograd gradients
# essentially grid-noise-dominated near the boundary). Leave unset for
# an automatic default of ~2x the average grid spacing.

  try:
    idx = [a.find("torch_taper_width")==0 for a in args].index(True)
  except ValueError:
    torch_taper_width_override = None
  else:
    torch_taper_width_override = float(args.pop(idx).split("=")[1])

# torch.compile the density splat's inner blocks (default True). Fusing them
# is worth ~2.9x on the splat -- the per-voxel work is memory-bound, so run
# eagerly it costs one pass over an (n_atoms, n_voxels) array per elementary
# operation, roughly twenty of them, where fused it is a couple. Costs a
# one-off compile of ~1s on the first frame, so set torch_compile=False for a
# single-frame run, if your torch has no working compiler backend, or to
# isolate a suspected compile-related numerical difference.

  try:
    idx = [a.find("torch_compile")==0 for a in args].index(True)
  except ValueError:
    torch_compile = True
  else:
    torch_compile = args.pop(idx).split("=")[1] == "True"

# float32 matmul precision, passed to torch.set_float32_matmul_precision().
#
#   highest  full float32 (torch's default, and ours)
#   high     TF32 on NVIDIA tensor cores: ~10 bits of mantissa instead of 24
#   medium   bfloat16: ~8 bits of mantissa
#
# The splat's inner product is a matmul, so this can matter on a GPU with
# tensor cores -- newer CUDA torch prints a startup hint recommending "high".
#
# It is OFF by default because it is a NUMERICS change, not a free speedup:
# TF32 keeps float32's exponent range but truncates the mantissa, so it is a
# deliberate accuracy-for-speed trade rather than something to enable silently
# in a program whose output is compared against a reference to six figures.
# Try it, measure both the time and the agreement, then decide.

# Run under cProfile and print the hot functions at the end (default False).
# Worth reaching for when the wall time is not where you expect: on a GPU the
# splat can drop to a few percent of per-frame time, leaving the cost in
# host-side work that no torch tuning will touch.

  try:
    idx = [a.find("profile")==0 for a in args].index(True)
  except ValueError:
    profile = False
  else:
    profile = args.pop(idx).split("=")[1] == "True"

# Time the frame loop by phase and report per-frame averages at the end
# (default False). Two halves:
#
#   host    trajectory read, coords->numpy, set_sites_cart, xrs.select,
#           sites_frac+occ, element idx, accumulate
#   device  splat, symmetrize, fft+extract, the transfers, into cctbx
#
# plus the loop's wall time and the residual the phases do not account for.
# The device half only applies to engine=torch; the host half is the same
# code for every engine, so the report is printed whatever the engine.
#
# This exists because cProfile cannot answer the question. It charges per
# PYTHON call, so the splat's ~90 batches/frame of eager torch ops look
# enormous while the FFT, one call doing heavy GPU work, looks free. Worse,
# CUDA kernels are asynchronous: without an explicit synchronize, a timer
# measures queue submission rather than execution, and the cost lands on
# whichever later call happens to block. The device timers synchronize on both
# sides of each phase, so they measure the phase; the host timers need no
# synchronize, since they queue no device work and every device phase has
# already synchronized on exit.

  try:
    idx = [a.find("torch_timing")==0 for a in args].index(True)
  except ValueError:
    torch_timing = False
  else:
    torch_timing = args.pop(idx).split("=")[1] == "True"

# Diagnostics for the splat's in-situ cost, which is ~4.6x what the same call
# costs repeated back-to-back on the same tensors (see docs/performance.md).
# The two differ in more than one way, so these separate them:
#
#   torch_splat_stats=True   count the work per frame -- atoms, chunks, and
#                            atom-voxel pairs both ideal and as actually
#                            batched. Answers "more work or slower work?"
#                            against bench_splat.py's pair count. Free.
#   torch_reuse_grid=True    splat into ONE preallocated grid buffer for every
#                            frame instead of allocating a fresh one per call,
#                            which is what the back-to-back repeat does.
#   torch_free_early=True    drop the previous frame's density and F tensors
#                            before the next splat, so the allocator is not
#                            holding two grids while it runs.
#
# The last two are probes, so both default to False: the point is to compare
# runs with and without, not to change the pipeline before the measurement.
# Whether to APPLY the space group's operations to the model density
# (default False).
#
# space_group= does two jobs that are usually the same job and here are not:
# it names the crystal's symmetry -- which the output reflection files should
# carry -- and it decides whether the density gets expanded by that symmetry.
# Those come apart for a model that ALREADY contains the full cell contents,
# which is the normal case for this program: an MD box that is an integer
# supercell of the crystal cell folds into a complete unit cell all by itself.
#
# Applying the operations to a complete model does not add the missing copies,
# because none are missing. It symmetry-AVERAGES the ones that are there --
# symmetrize_sum over the orbit is `order` times the mean over the orbit -- and
# that averaging removes exactly the deviations from crystallographic symmetry
# that the diffuse signal is made of. Measured on the 7FPV trajectory, 4
# frames: applying P2(1)2(1)2(1) raises mean Icalc 15.1x (~ order^2, the Bragg
# term adding coherently) while diffuse rises only 4.4x, so diffuse/Icalc falls
# from 0.0387 to 0.0113 -- a 3.4x suppression of the observable this program
# exists to compute.
#
# Hence the default: do not expand. Set expand_symmetry=True when the model is
# ONE asymmetric unit and the other copies genuinely need generating, or when
# symmetry averaging is what you want.
  try:
    idx = [a.find("expand_symmetry")==0 for a in args].index(True)
  except ValueError:
    expand_symmetry = False
  else:
    expand_symmetry = args.pop(idx).split("=")[1] == "True"

  try:
    idx = [a.find("torch_splat_stats")==0 for a in args].index(True)
  except ValueError:
    torch_splat_stats = False
  else:
    torch_splat_stats = args.pop(idx).split("=")[1] == "True"

  try:
    idx = [a.find("torch_reuse_grid")==0 for a in args].index(True)
  except ValueError:
    torch_reuse_grid = False
  else:
    torch_reuse_grid = args.pop(idx).split("=")[1] == "True"

  try:
    idx = [a.find("torch_free_early")==0 for a in args].index(True)
  except ValueError:
    torch_free_early = False
  else:
    torch_free_early = args.pop(idx).split("=")[1] == "True"

#   torch_profile_frames=N   record N frames with torch.profiler (CPU + CUDA
#                            activities), write a Chrome trace and print the
#                            per-operator table. Frame 0 is skipped and frame
#                            1 is a profiler warmup, so recording starts at
#                            frame 2 and N=3 is usually enough. Rank 0 only.
#
# This is the instrument for the splat's rate, once work volume has been ruled
# out: it separates one kernel running slower from the same kernels with gaps
# between them from more kernels being launched, which no wall-clock timer can.
# It perturbs what it measures, so the phase table from a profiled run is not a
# performance measurement -- take timings from an unprofiled run.
  try:
    idx = [a.find("torch_profile_frames")==0 for a in args].index(True)
  except ValueError:
    torch_profile_frames = 0
  else:
    torch_profile_frames = int(args.pop(idx).split("=")[1])

  try:
    idx = [a.find("torch_matmul_precision")==0 for a in args].index(True)
  except ValueError:
    torch_matmul_precision = "highest"
  else:
    torch_matmul_precision = args.pop(idx).split("=")[1]
    if torch_matmul_precision not in ("highest", "high", "medium"):
      raise ValueError(
        "torch_matmul_precision must be highest, high or medium; got {0!r}"
        .format(torch_matmul_precision)
      )

# Dump the first frame's real-space density grid to density_gemmi.npy /
# density_torch.npy for tools/compare_density.py (default False). Off by
# default because it is a whole uncompressed float32 grid -- 221 MB for a
# 300x300x144 run -- written on every invocation, which is how the scratch
# directory grew to ~466 MB. Turn it on only for an actual density comparison.

  try:
    idx = [a.find("save_density")==0 for a in args].index(True)
  except ValueError:
    save_density = False
  else:
    save_density = args.pop(idx).split("=")[1] == "True"

# Max atom-voxel PAIRS per batch for the torch engine -- the quantity that
# actually sets intermediate tensor size, and the main tuning knob now that
# batching is budgeted by pairs rather than by atom count. It is a speed knob
# as much as a memory one: the default keeps each working buffer around 16 MB
# so it stays cache-resident, which measured faster than both larger and
# smaller values. Re-tune with bench_splat.py on a different machine.
# Leave unset to use density_torch.splat_density's default.

  try:
    idx = [a.find("torch_max_pairs_per_batch")==0 for a in args].index(True)
  except ValueError:
    torch_max_pairs_per_batch = None
  else:
    torch_max_pairs_per_batch = int(args.pop(idx).split("=")[1])

# torch threads per MPI rank on CPU (default 1, only applied when mpi_size > 1).
# This is the single most important setting for multi-rank CPU throughput with
# engine=torch: gemmi's density calculator is single-threaded and beats the
# torch splat ~2.3x per core, so torch only wins by spreading over cores. Aim
# for mpi_size * torch_num_threads ~= physical cores -- e.g. 3 ranks x 6
# threads rather than 18 ranks x 1 thread. Ignored for a single rank (torch
# then uses its own default) and on GPU.

  try:
    idx = [a.find("torch_num_threads")==0 for a in args].index(True)
  except ValueError:
    torch_num_threads = None
  else:
    torch_num_threads = int(args.pop(idx).split("=")[1])

# CCTBX Method for calculating structure factors

  try:
    idx = [a.find("cctbx_method")==0 for a in args].index(True)
  except ValueError:
    cctbx_method = "fft"
  else:
    cctbx_method = args.pop(idx).split("=")[1]

# Calculate difference with respect to reference (for optimization)

  try:
    idx = [a.find("diff_mode")==0 for a in args].index(True)
  except ValueError:
    diff_mode = False
  else:
    diff_str = args.pop(idx).split("=")[1]
    if diff_str == "True":
      diff_mode = True
    else:
      diff_mode = False
      
# Reference fcalc for diff mode

  try:
    idx = [a.find("fcalc_ref")==0 for a in args].index(True)
  except ValueError:
    fcalc_ref_file = "fcalc_ref.mtz"
  else:
    fcalc_ref_file = args.pop(idx).split("=")[1]
    
# Reference icalc for diff mode

  try:
    idx = [a.find("icalc_ref")==0 for a in args].index(True)
  except ValueError:
    icalc_ref_file = "icalc_ref.mtz"
  else:
    icalc_ref_file = args.pop(idx).split("=")[1]

# Number of frames in reference

  try:
    idx = [a.find("nref")==0 for a in args].index(True)
  except ValueError:
    nsteps_ref = 1
  else:
    nsteps_ref = int(args.pop(idx).split("=")[1])

# Initialize MPI

  if mpi_enabled():
    import mpi4py
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
  else:
    mpi_comm = None
    mpi_rank = 0
    mpi_size = 1

  if use_top_bfacs:
    if apply_bfac:
      if mpi_rank == 0:
        print("Setting apply_bfac = False as use_top_bfacs = True")
      apply_bfac = False
      
# If in diff_mode, read the reference .mtz files

  profiler = cProfile.Profile()
  if profile:
    profiler.enable()
  #profiler.enable()

  if diff_mode:
    if mpi_rank == 0:      
      from iotbx.reflection_file_reader import any_reflection_file
      hkl_in = any_reflection_file(file_name=fcalc_ref_file)
      miller_arrays = hkl_in.as_miller_arrays()
      avg_fcalc_ref = miller_arrays[0]
      hkl_in = any_reflection_file(file_name=icalc_ref_file)
      miller_arrays = hkl_in.as_miller_arrays()
      avg_icalc_ref = miller_arrays[0]

# If there's a diffuse data file, read it

  if diffuse_data_file is not None:
    if mpi_rank == 0:      
      from iotbx.reflection_file_reader import any_reflection_file
      hkl_in = any_reflection_file(file_name=diffuse_data_file)
      miller_arrays = hkl_in.as_miller_arrays()
      diffuse_expt = miller_arrays[0].as_non_anomalous_array()
      if corr_aniso:
        diffuse_expt = to_aniso(diffuse_expt,apply_symmetry_str)
    else:
      diffuse_expt = None
    # Serial runs have no communicator, and rank 0 already holds the value.
    if mpi_enabled():
      diffuse_expt = mpi_comm.bcast(diffuse_expt,root=0)

# read .pdb file. It's used as a template, so don't sort it.

  if mpi_rank == 0:
    pdb_in = hierarchy.input(file_name=top_file,sort_atoms=False)

  if mpi_enabled():
    if mpi_rank == 0:
      pdb_bytes = pickle.dumps(pdb_in, protocol=4)
    else:
      pdb_bytes = None
    pdb_bytes = mpi_comm.bcast(pdb_bytes,root=0)
    if mpi_rank != 0:
      pdb_in = pickle.loads(pdb_bytes)

# MEW use cctbx.xray.structure.customized_copy() here to change the unit cell and space group as needed
  symm = pdb_in.input.crystal_symmetry()
  if unit_cell_str is None:
    unit_cell = symm.unit_cell()
  else:
    unit_cell = unit_cell_str
  if space_group_str is None:
    space_group_info = symm.space_group_info()
  else:
    space_group_info = cctbx.sgtbx.space_group_info(symbol=space_group_str)

  xrs = pdb_in.input.xray_structure_simple(crystal_symmetry=crystal.symmetry(unit_cell=unit_cell,space_group_info=space_group_info))

  space_group_str = str(space_group_info).replace(" ","")
  selection_cache = pdb_in.hierarchy.atom_selection_cache()
  selection = selection_cache.selection(selection_text)
  xrs.convert_to_isotropic()
  if apply_bfac:
    xrs.set_b_iso(20.0*d_min*d_min)
  else:
    if not use_top_bfacs:
      xrs.set_b_iso(0.0)
  xrs.set_occupancies(1.0)
  xrs_sel = xrs.select(selection)
  xrs_sel.scattering_type_registry(table=scattering_table)

  # ---- phase timing (torch_timing=True) --------------------------------
  #
  # Wall-clock accounting for the per-frame pipeline, shared by the host-side
  # phases below and the device phases in the engine == "torch" branch. The
  # two differ only in whether they synchronize the device before reading the
  # clock: host phases queue no device work, so they do not need to, and
  # every device phase synchronizes on exit, so nothing is left pending when
  # a host phase starts.
  #
  # The timers are engine-independent -- everything from the trajectory read
  # to the running sums is the same code for gemmi and cctbx -- so the report
  # is printed whenever torch_timing is on, whatever the engine.

  import collections
  import contextlib

  _phase_totals = collections.OrderedDict()
  _phase_calls = collections.Counter()
  _phase_series = {}

  def _record_phase(name, dt):
    _phase_totals[name] = _phase_totals.get(name, 0.0) + dt
    _phase_calls[name] += 1
    # Per-call durations, not just the total. A one-time cost paid on frame 0
    # and a genuine per-frame cost are indistinguishable in an average, and
    # differ by the frame count -- which is how a 10-frame comparison can show
    # an effect a 251-frame one cannot see.
    if torch_splat_stats:
      _phase_series.setdefault(name, []).append(dt)

  @contextlib.contextmanager
  def host_phase(name):
    if not torch_timing:
      yield
      return
    t0 = time.time()
    try:
      yield
    finally:
      _record_phase(name, time.time() - t0)

  # Set by the torch setup below. Every other engine reads xrs frame by frame.
  torch_bypass_xrs = False
  # Per-frame splat work counters, filled when torch_splat_stats=True.
  torch_splat_stats_log = []
  # torch.profiler handle, set by the torch setup when torch_profile_frames>0.
  torch_profiler = None

  def report_phases(n_frames, loop_wall=None):
    if not torch_timing or not _phase_totals:
      return
    total = sum(_phase_totals.values())
    print("\n=== {0} engine phase timing (rank 0), {1} frames ===".format(
      engine, n_frames))
    # The call count matters for reading the table: most phases run once per
    # frame, but the trajectory read runs once per CHUNK, so its ms/frame
    # figure moves with chunk= and is not comparable to the others without it.
    print("  {0:<22} {1:>10} {2:>12} {3:>8} {4:>7}".format(
      "phase", "total s", "ms/frame", "share", "calls"))
    for name, secs in _phase_totals.items():
      print("  {0:<22} {1:>10.2f} {2:>12.1f} {3:>7.1f}% {4:>7d}".format(
        name, secs, 1e3 * secs / max(n_frames, 1), 100.0 * secs / total,
        _phase_calls[name]))
    print("  {0:<22} {1:>10.2f} {2:>12.1f}".format(
      "TOTAL (timed)", total, 1e3 * total / max(n_frames, 1)))
    if loop_wall is not None:
      # What the timers did not catch. A large residual here means the frame
      # loop is spending time somewhere still unbracketed -- which is exactly
      # the condition these timers exist to detect, so report it rather than
      # leaving it to be inferred from the run's total wall time.
      resid = loop_wall - total
      print("  {0:<22} {1:>10.2f} {2:>12.1f}".format(
        "frame loop wall", loop_wall, 1e3 * loop_wall / max(n_frames, 1)))
      print("  {0:<22} {1:>10.2f} {2:>12.1f} {3:>7.1f}%".format(
        "unaccounted", resid, 1e3 * resid / max(n_frames, 1),
        100.0 * resid / loop_wall if loop_wall > 0 else 0.0))
    print("  Timers cover the frame loop only: setup, the MPI reduction and "
          "the output files are outside them.")

    if torch_splat_stats_log:
      # Compare pairs_actual against bench_splat.py's pair count on the same
      # structure and grid. If they agree, the splat's in-situ cost is not
      # extra work and the difference is execution rate. pairs_actual /
      # pairs_ideal is the chunk padding, which is the one part of the work
      # volume that the coordinates can move.
      ideal = [s["pairs_ideal"] for s in torch_splat_stats_log]
      actual = [s["pairs_actual"] for s in torch_splat_stats_log]
      chunks = [s["chunks"] for s in torch_splat_stats_log]
      print("\n  splat work over {0} frames ({1:,} atoms):".format(
        len(actual), torch_splat_stats_log[0]["atoms"]))
      print("    {0:<16} {1:>18} {2:>18}".format("", "min", "max"))
      for name, vals in (("pairs ideal", ideal), ("pairs actual", actual),
                         ("chunks", chunks)):
        print("    {0:<16} {1:>18,} {2:>18,}".format(name, min(vals), max(vals)))
      print("    padding (actual/ideal): {0:.4f} - {1:.4f}".format(
        min(a / i for a, i in zip(actual, ideal)),
        max(a / i for a, i in zip(actual, ideal))))

    series = _phase_series.get("splat")
    if series and len(series) >= 4:
      # Frame 0 separately: it carries the one-time warmup, and any cost that
      # a setup-time change moves OUT of the loop shows up here and nowhere
      # else. Then the first and last tenth of the remainder, for the trend --
      # a run that speeds up as it goes is not throttling.
      first = series[0]
      rest = sorted(series[1:])
      chron = series[1:]
      k = max(1, len(chron) // 10)
      print("\n  splat per frame (ms): frame 0 {0:.1f}, then min {1:.1f}, "
            "median {2:.1f}, max {3:.1f}".format(
              1e3 * first, 1e3 * rest[0], 1e3 * rest[len(rest) // 2],
              1e3 * rest[-1]))
      print("    first {0} of the rest {1:.1f}, last {0} {2:.1f}".format(
        k, 1e3 * sum(chron[:k]) / k, 1e3 * sum(chron[-k:]) / k))

  if engine == "torch":
    import torch

    # Before any tensor work, so every matmul in the run sees the same setting.
    if torch_matmul_precision != "highest":
      torch.set_float32_matmul_precision(torch_matmul_precision)

    def _torch_sync(dev):
      """Block until queued device work has finished.

      CUDA and MPS kernels are launched asynchronously, so an unsynchronized
      timer measures how long it took to ENQUEUE the work, not to do it.
      """
      dev = str(dev)
      if dev.startswith("cuda"):
        torch.cuda.synchronize()
      elif dev.startswith("mps"):
        torch.mps.synchronize()

    @contextlib.contextmanager
    def torch_phase(name, device):
      if not torch_timing:
        yield
        return
      _torch_sync(device)
      t0 = time.time()
      try:
        yield
      finally:
        _torch_sync(device)
        _record_phase(name, time.time() - t0)

    from lunus.sf.elements import it92_coefficients
    from lunus.sf.cell_utils import orth_matrix, grid_shape_for_resolution
    from lunus.sf.kernel_torch import build_element_kernels_torch, build_atom_kernels_torch
    from lunus.sf.structure_factor_torch import structure_factors_one_config, compute_fcalc
    from lunus.sf.density_torch import splat_density
    from lunus.sf.symmetry_torch import (
      build_grid_ops_from_cctbx, adjust_grid_for_symmetry, symmetrize_sum,
    )

    # left empty when unset so splat_density's own (measured) default applies
    torch_pairs_kwarg = ({} if torch_max_pairs_per_batch is None
                         else {"max_pairs_per_batch": torch_max_pairs_per_batch})

    torch_b_iso = 20.0 * d_min * d_min if apply_bfac else 0.0
    torch_dtype = torch.float32  # confirmed via the resolution-shell comparison that float64
                                  # wasn't the fix -- also required for MPS (Apple GPU), which
                                  # doesn't support float64 at all
    torch_device = torch_device_str

    # How many CPUs this process may ACTUALLY use, which in a container is not
    # what the machine reports. torch sizes its thread pool from the host's
    # core count, so a pod limited to 3 CPUs gets 104 threads; they spin-wait,
    # exhaust the cgroup's CPU quota within ~12 ms of every 100 ms period, and
    # the kernel then freezes the whole process until the period boundary.
    #
    # Measured here, 10 frames, CUDA, 104 threads on 3 CPUs: splat 287.9
    # ms/frame median. With OMP_NUM_THREADS=2: 66.3 -- against 62.6 ms for the
    # same work in tools/bench_splat.py. The 4.6x "the splat is slow in the
    # loop" mystery was this, start to finish. See docs/performance.md.
    # The CFS QUOTA is the only thing that knows. nproc, os.cpu_count() and
    # os.sched_getaffinity() all report the host's cores -- 104 on the pod
    # this was diagnosed on -- because a CPU limit is enforced by accounting,
    # not by restricting which cores the process may run on. Affinity is used
    # here only as an additional cap, never as the answer.
    def _cpu_quota():
      try:
        with open("/sys/fs/cgroup/cpu.max") as fh:      # cgroup v2
          quota, period = fh.read().split()
          if quota != "max":
            return max(1, int(float(quota) / float(period)))
      except (OSError, ValueError):
        pass
      try:                                              # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
          quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
          period = int(fh.read())
        if quota > 0:
          return max(1, quota // period)
      except (OSError, ValueError):
        pass
      return None                                       # no limit, or unreadable

    # Leave the CPU multi-rank case to the more specific logic below.
    if not (torch_device == "cpu" and mpi_size > 1):
      _quota = _cpu_quota()
      try:
        _affinity = len(os.sched_getaffinity(0))        # Linux only
      except AttributeError:
        _affinity = None

      if torch_num_threads is not None:
        _target, _why = torch_num_threads, "torch_num_threads"
      elif _quota is not None:
        _target = _quota if _affinity is None else min(_quota, _affinity)
        _why = "cgroup CPU quota"
      elif torch_device != "cpu":
        # No discoverable quota, and the host only dispatches on a device run
        # -- there is no torch CPU work in the frame loop to spread. One
        # thread cannot oversubscribe a limit we cannot see.
        _target, _why = 1, "device run with no discoverable CPU quota"
      else:
        _target, _why = None, None                      # real CPU work: leave it

      if _target is not None and _target != torch.get_num_threads():
        _was = torch.get_num_threads()
        torch.set_num_threads(_target)
        if mpi_rank == 0:
          print("torch engine: torch.set_num_threads({0}), was {1} ({2})"
                .format(_target, _was, _why))
          if _quota is not None and _was > _target:
            print("  Threads beyond the quota spin-wait, spend it faster than "
                  "useful work does, and get the whole process throttled -- "
                  "measured 4.3x on the splat. This does NOT cover numpy/BLAS, "
                  "which reads the environment before this program starts: "
                  "export OMP_NUM_THREADS={0} as well.".format(_target))
          elif _quota is None:
            print("  The host only dispatches on a device run, so there is no "
                  "torch CPU work to spread. torch_num_threads=N to override.")

    if torch_device == "cpu" and mpi_size > 1:
      # Without this, EVERY rank's PyTorch process independently tries to
      # use all available cores for its own internal threading (the
      # splat's elementwise math, index_add_, FFT) -- with mpi_size ranks
      # doing that simultaneously, the result is severe CPU oversubscription
      # and contention, often making a multi-rank CPU run much SLOWER than
      # a single process, especially on CPU-limited/shared/containerized
      # nodes.
      #
      # Note on what to expect from multi-rank CPU runs with this engine:
      # gemmi's DensityCalculatorX is single-threaded (measured: 1.00 cores
      # busy) and about 2.3x faster than the torch splat PER CORE -- 1.44 s
      # against 3.37 s on the 135k-atom case. torch only wins on wall time by
      # spreading over cores (0.87 s on 6 threads), which one-thread-per-rank
      # denies it. Rebalancing does NOT rescue this: measured over 10 frames,
      # 10 ranks x 1 thread gave 7.4 s, 6 x 3 gave 7.2 s, 3 x 6 gave 8.0 s, and
      # 2 x 9 gave 21.4 s (too few ranks and each one serializes several
      # frames), against gemmi at 3.1 s. Expect gemmi to win multi-rank CPU
      # runs; the torch engine earns its keep through differentiability, and on
      # GPU or in a single multi-threaded process.
      torch_threads = torch_num_threads if torch_num_threads is not None else 1
      torch.set_num_threads(torch_threads)
      if mpi_rank == 0:
        print("torch engine: mpi_size =", mpi_size, "on CPU, set torch.set_num_threads(",
              torch_threads, ") per rank to avoid oversubscription (torch_num_threads=N "
              "to change; keep mpi_size*N near the physical core count)")

    torch_elements_present = sorted(set(s.scattering_type for s in xrs_sel.scatterers()))
    # Read the scattering coefficients for exactly the elements in this
    # structure. They come from gemmi or cctbx at run time rather than from a
    # table stored here, so any element those libraries know is supported
    # without editing anything.
    try:
      IT92_COEFFS = it92_coefficients(torch_elements_present)
    except (KeyError, ImportError) as e:
      raise type(e)("engine=torch: {0}".format(e))

    torch_a, torch_b_uc, torch_c, torch_alpha, torch_beta, torch_gamma = xrs_sel.unit_cell().parameters()
    torch_M_np = orth_matrix(torch_a, torch_b_uc, torch_c, torch_alpha, torch_beta, torch_gamma)
    torch_orth_matrix = torch.tensor(torch_M_np, dtype=torch_dtype, device=torch_device)
    torch_cell_volume = xrs_sel.unit_cell().volume()
    torch_grid_shape = grid_shape_for_resolution(torch_a, torch_b_uc, torch_c, d_min, rate=1.5)

    # SYMMETRY: match gemmi exactly -- splat only the atoms we were given,
    # then symmetry-expand the DENSITY GRID (gemmi's
    # put_model_density_on_grid() = add_model_density_to_grid() +
    # symmetrize_sum()). Expanding the density is far cheaper than expanding
    # the ATOMS and re-splatting every symmetry copy, which is
    # O(n_atoms * box_volume) Gaussian evaluations per copy; symmetrizing the
    # finished grid is O(n_grid) integer index arithmetic per operation, and
    # for signed-permutation operations (which is all of them for most space
    # groups) it reduces to axis permutes/flips/rolls with no index tensors
    # materialized at all. It is also fully differentiable -- see
    # symmetry_torch.py and validate_symmetry.py, which checks parity against
    # gemmi's own symmetrize_sum() rather than against a reading of its source.
    #
    # For a P1 space group build_grid_ops_from_cctbx returns an empty list and
    # nothing is applied, which is again what gemmi does (its symmetrize()
    # returns immediately for P1).
    #
    # The grid may need adjusting first: grid_shape_for_resolution() rounds
    # each axis up independently, which does not guarantee that a symmetry
    # operation maps the grid onto itself (a 4-fold in the ab plane needs
    # Nu == Nv; a 3/4 screw translation needs Nw divisible by 4). gemmi
    # applies the equivalent constraints when sizing its own grid.
    torch_grid_shape_raw = torch_grid_shape
    torch_sym_rots = [np.array(op.r().as_double()).reshape(3, 3) for op in xrs_sel.space_group()]
    torch_sym_trans = [np.array(op.t().as_double()) for op in xrs_sel.space_group()]
    torch_grid_shape = adjust_grid_for_symmetry(
      torch_grid_shape_raw, torch_sym_rots, torch_sym_trans
    )
    torch_grid_ops = (
      build_grid_ops_from_cctbx(xrs_sel.space_group(), torch_grid_shape)
      if expand_symmetry else [])

    if mpi_rank == 0:
      if torch_grid_shape != torch_grid_shape_raw:
        print("torch engine: grid adjusted for symmetry compatibility,",
              torch_grid_shape_raw, "->", torch_grid_shape)
      print("torch engine: space group =", xrs_sel.space_group().type().lookup_symbol(),
            ", order =", xrs_sel.space_group().order_z(),
            ", density symmetry ops applied (excl. identity) =", len(torch_grid_ops),
            "" if expand_symmetry else
            "(expand_symmetry=False: the model is taken to hold the full cell "
            "contents already; set True to expand an asymmetric unit, or to "
            "symmetry-average)")
      if not torch_grid_ops:
        print("torch engine: P1 (or no non-identity ops) -- density used as splatted, "
              "no symmetry expansion, matching gemmi")

    if use_top_bfacs:
      # extract_u_iso_or_u_equiv() must be called on the xray_structure, NOT on
      # xrs_sel.scatterers(): the scatterer-array overload requires a unit_cell
      # argument (to convert u_star -> u_equiv) and raises Boost.Python
      # ArgumentError without one. This matches the gemmi branch's working call.
      torch_u_isos_setup = np.array(xrs_sel.extract_u_iso_or_u_equiv())
      torch_b_per_atom_setup = torch_u_isos_setup * (8.0 * np.pi ** 2)  # U -> B, same as the gemmi branch
      torch_elements_per_atom_setup = [s.scattering_type for s in xrs_sel.scatterers()]

      torch_atom_A, torch_atom_lam, torch_elem_offsets, torch_atom_radius_ang, torch_taper_width, torch_element_to_idx = build_atom_kernels_torch(
        torch_elements_per_atom_setup, torch_elements_present, IT92_COEFFS, torch_b_per_atom_setup, blur=0.0,
        grid_shape=torch_grid_shape, orth_matrix_np=torch_M_np,
        cutoff=gemmi_cutoff, taper_width=torch_taper_width_override,
        device=torch_device, dtype=torch_dtype,
      )
      if mpi_rank == 0:
        print("torch engine: use_top_bfacs=True, per-atom B range =",
              torch_b_per_atom_setup.min(), "-", torch_b_per_atom_setup.max(),
              ", n_atoms (as splatted, before density symmetry expansion) =",
              len(torch_elements_per_atom_setup))
    else:
      torch_elem_A, torch_elem_lam, torch_elem_offsets, torch_elem_radius_ang, torch_taper_width, torch_element_to_idx = build_element_kernels_torch(
        torch_elements_present, IT92_COEFFS, torch_b_iso, blur=0.0,
        grid_shape=torch_grid_shape, orth_matrix_np=torch_M_np,
        cutoff=gemmi_cutoff, taper_width=torch_taper_width_override,
        device=torch_device, dtype=torch_dtype,
      )

    torch_miller_set = miller.build_set(
      crystal_symmetry=xrs_sel.crystal_symmetry(),
      anomalous_flag=False,
      d_min=d_min,
    )
    torch_hkl_np = np.array(torch_miller_set.indices())
    torch_hkl = torch.tensor(torch_hkl_np, dtype=torch.long, device=torch_device)

    # ---- frame-invariant inputs, hoisted out of the frame loop -----------
    #
    # The per-frame cctbx round trip -- push the frame's coordinates into the
    # xray.structure with set_sites_cart, select, then read sites_frac,
    # occupancies and scattering types straight back out -- measured ~97
    # ms/frame beyond the flex conversion, for information that either does not
    # change between frames (selection, occupancies, elements) or is one 3x3
    # matmul from what mdtraj already handed over (fractional coordinates).
    # The torch path needs nothing else from xrs, so it skips the round trip.
    #
    # translational_fit is the exception: it fits against xrs.sites_frac() and
    # writes the result back with set_sites_frac(), so it needs the structure
    # maintained frame by frame.
    torch_bypass_xrs = not translational_fit

    if torch_bypass_xrs:
      # cctbx computes frac = M_frac * cart on column vectors; these are rows.
      torch_frac_matrix_np = np.array(
        xrs_sel.unit_cell().fractionalization_matrix()).reshape((3, 3))

      # A bypass that disagreed with cctbx would move every atom and change
      # every structure factor, quietly. Check the matrix against sites_frac()
      # itself, on the topology coordinates xrs is holding right now.
      _chk_cart = xrs_sel.sites_cart().as_double().as_numpy_array().reshape((-1, 3))
      _chk_ref = xrs_sel.sites_frac().as_double().as_numpy_array().reshape((-1, 3))
      _chk_dev = np.max(np.abs(_chk_cart @ torch_frac_matrix_np.T - _chk_ref))
      assert _chk_dev < 1e-12, (
        "fractionalization matrix disagrees with cctbx sites_frac() by %g; "
        "the frame-loop bypass would change the answer" % _chk_dev)

      # The selection as row indices into the trajectory's atom axis. None
      # means "all atoms", which lets the frame loop skip the gather entirely.
      _sel_np = selection.as_numpy_array()
      torch_sel_idx = None if _sel_np.all() else np.nonzero(_sel_np)[0]

      torch_occ_np = np.array(xrs_sel.scatterers().extract_occupancies())
      torch_element_idx_np = np.array(
        [torch_element_to_idx[s.scattering_type] for s in xrs_sel.scatterers()],
        dtype=np.int64,
      )

      if mpi_rank == 0:
        print("torch engine: bypassing the per-frame xray.structure round trip"
              " (%d of %d atoms selected, fractionalization agrees with cctbx "
              "to %.2g)" % (len(torch_occ_np), len(_sel_np), _chk_dev))
    elif mpi_rank == 0:
      print("torch engine: translational_fit=True, so the per-frame "
            "xray.structure round trip is kept")

    # ---- splat probes (see the flag comments near the top) ---------------
    #
    # Bound before the loop so the free-early probe has something to drop on
    # frame 0 and the reuse probe has a buffer to splat into.
    torch_density = None
    F_t = None
    torch_grid_buf = None
    if torch_reuse_grid:
      torch_grid_buf = torch.zeros(
        torch_grid_shape[0] * torch_grid_shape[1] * torch_grid_shape[2],
        device=torch_device, dtype=torch_dtype,
      )
      if mpi_rank == 0:
        print("torch engine: splatting into one reused %.1f MB grid buffer "
              "rather than a fresh allocation per frame"
              % (torch_grid_buf.numel() * torch_grid_buf.element_size() / 1e6))
    if torch_free_early and mpi_rank == 0:
      print("torch engine: dropping the previous frame's density and F before "
            "each splat")

    if torch_profile_frames > 0 and mpi_rank == 0:
      if profile:
        raise ValueError(
          "torch_profile_frames and profile=True cannot be combined: cProfile "
          "charges per Python call and would misattribute the torch work it is "
          "wrapped around. Run one or the other.")

      from torch.profiler import ProfilerActivity, schedule as _profiler_schedule

      _trace_path = "torch_trace_rank{0}.json".format(mpi_rank)

      def _on_trace_ready(prof):
        prof.export_chrome_trace(_trace_path)
        # Sort by device self time where the build reports it -- the name
        # changed across torch versions, and CPU-only runs have neither.
        for key in ("self_device_time_total", "self_cuda_time_total",
                    "self_cpu_time_total"):
          try:
            table = prof.key_averages().table(sort_by=key, row_limit=25)
          except (AssertionError, KeyError, ValueError, RuntimeError):
            continue
          print("\n=== torch.profiler, sorted by {0} ===".format(key))
          print(table)
          break
        print("torch profiler: Chrome trace written to", _trace_path,
              "-- load it in chrome://tracing or https://ui.perfetto.dev to see "
              "kernel timings and the gaps between them")

      _activities = [ProfilerActivity.CPU]
      if str(torch_device).startswith("cuda"):
        _activities.append(ProfilerActivity.CUDA)

      torch_profiler = torch.profiler.profile(
        activities=_activities,
        # Frame 0 carries one-time warmup and frame 1 warms the profiler
        # itself, so recording starts at frame 2.
        schedule=_profiler_schedule(
          wait=1, warmup=1, active=torch_profile_frames, repeat=1),
        on_trace_ready=_on_trace_ready,
        record_shapes=True,
      )
      print("torch engine: profiling {0} frames (from frame 2); the phase "
            "table from this run is NOT a performance measurement"
            .format(torch_profile_frames))

    # Compile ONCE on rank 0, then let the others start. Without this every
    # rank compiles the same kernels itself, and N concurrent compilations
    # saturate the machine: measured on 10 ranks with a cold cache, frame times
    # staggered in ~1 s steps up to 17.0 s (total 17.5 s), against 7.4 s with
    # no stagger once rank 0 warms the cache first. It is not file-lock
    # contention -- per-rank TORCHINDUCTOR_CACHE_DIRs reproduce the stagger
    # exactly. The penalty only shows up on a COLD cache, which is why it looks
    # intermittent: a second run of the same job reuses the on-disk cache.
    if torch_compile and mpi_size > 1:
      from lunus.sf.density_torch import warmup_compile
      if mpi_rank == 0:
        _t_warm = time.time()
        warmup_compile(
          torch_elem_offsets, torch_grid_shape, torch_orth_matrix, torch_taper_width,
          max_atoms_per_batch=torch_max_atoms_per_batch, **torch_pairs_kwarg,
        )
        print("torch engine: populated the torch.compile cache on rank 0 in "
              "%.1f s before releasing the other %d ranks (each would otherwise "
              "compile the same kernels itself)" % (time.time() - _t_warm, mpi_size - 1))
      mpi_comm.Barrier()

    if mpi_rank == 0:
      print("torch engine: elements =", torch_elements_present,
            ", grid =", torch_grid_shape,
            ", b_iso =", ("per-atom (use_top_bfacs)" if use_top_bfacs else torch_b_iso),
            ", device =", torch_device, ", max_atoms_per_batch =", torch_max_atoms_per_batch,
            ", max_pairs_per_batch =",
            ("default" if torch_max_pairs_per_batch is None else torch_max_pairs_per_batch),
            ", taper_width =", torch_taper_width, ", torch.compile =", torch_compile,
            ", matmul_precision =", torch_matmul_precision)

  if (mpi_rank == 0):
#    pdbtmp = xrs_sel.as_pdb_file()
#    with open("reference.pdb","w") as fo:
#      fo.write(pdbtmp)
    if engine == "sfall":
      sfall_script = \
"""
sfall xyzin $1 hklin $2 hklout $3 <<EOF
mode sfcalc xyzin hklin
symm {space_group}
labin FP=F SIGFP=SIGF
RESOLUTION {d_min}
NOSCALE
VDWR 3.0
end
EOF
"""
      print("Writing the following as run_sfall.sh:")
      print(sfall_script.format(d_min=d_min,space_group=space_group_str))
      with open("run_sfall.sh","w") as fo:
        fo.write(sfall_script.format(d_min=d_min,space_group=space_group_str))
    if calc_f000:
      f_000 = mmtbx.utils.f_000(xray_structure=xrs_sel,mean_solvent_density=0.0)
      volume = xrs_sel.unit_cell().volume()
      print("f_000 = %g, volume = %g" % (f_000.f_000,volume))

  if engine == "sfall":
    fcalc = xrs_sel.structure_factors(d_min=d_min,algorithm=cctbx_method).f_calc()
    mtz_dataset = fcalc.as_mtz_dataset('FWT')
    famp = abs(fcalc)
    famp.set_observation_type_xray_amplitude()
    famp.set_sigmas(sigmas=flex.double(fcalc.data().size(),1))
    #famp_with_sigmas = miller_set.array(data=famp.data(),sigmas=sigmas)
    mtz_dataset.add_miller_array(famp,'F')
    mtz_dataset.mtz_object().write(file_name="reference_{rank:03d}.mtz".format(rank=mpi_rank))

# read the MD trajectory and extract coordinates

  assert nsteps >= mpi_size, "nsteps < mpi_size"

  if (chunksize is None):
    nchunks = mpi_size
    chunksize = int(nsteps/nchunks)
  else:
    nchunks = int(nsteps/chunksize)
    if nchunks < mpi_size:
      nchunks = mpi_size
      chunksize = int(nsteps/nchunks)
      if mpi_rank == 0:
          print("nchunks < mp_size. Resetting chunksize = ",chunksize)
  
  skiplist = np.zeros((mpi_size), dtype=int)
  chunklist = np.zeros((mpi_size), dtype=int)
  nchunklist = np.zeros((mpi_size), dtype=int)
  chunks_per_rank = int(nchunks/mpi_size)
  extra_chunks = nchunks % mpi_size
  extra_frames = nsteps - chunksize*nchunks
  if extra_frames > chunksize:
      if mpi_rank == 0:
          print("extra_frames = {0}, chunksize = {1}, fixing".format(extra_frames, chunksize))
      extra_chunks = extra_chunks + int(extra_frames/chunksize)
      extra_frames = extra_frames - int(extra_frames/chunksize)*chunksize
      if extra_chunks > mpi_size:
        chunks_per_rank = chunks_per_rank + int(extra_chunks/mpi_size)
        extra_chunks = extra_chunks - int(extra_chunks/mpi_size)*mpi_size
  # The scheme below reserves the LAST rank to mop up the frames that do not
  # divide evenly, spreading the whole chunks over the other mpi_size - 1
  # ranks. That has no meaning on a single rank -- and divided by zero -- so
  # single-rank runs get their own, simpler assignment.
  #
  # Chunking still matters with one rank: it is what bounds peak memory, which
  # is the reason to pass chunk= in the first place. So rank 0 runs every
  # chunk itself rather than being handed one chunk of every frame.
  if mpi_size == 1:
    n_full_chunks = nsteps // chunksize
    chunks_per_rank = n_full_chunks + (1 if nsteps % chunksize else 0)
    extra_chunks = 0
    extra_frames = 0          # consumed below by the final, shorter chunk
    chunklist[0] = chunksize
    nchunklist[0] = chunks_per_rank
    skiplist[0] = first
    if mpi_rank == 0:
      print("nchunks = {0}, chunks_per_rank = {1}, chunksize = {2}, frames = {3}"
            .format(chunks_per_rank, chunks_per_rank, chunksize, nsteps))
      print("Will use ", nsteps, " frames on a single worker in ",
            chunks_per_rank, " chunks of at most ", chunksize, " frames")
    ct = nsteps
  else:
    if nchunks != mpi_size and extra_frames != 0:
      chunks_per_rank = int(nchunks/(mpi_size-1))
      extra_chunks = nchunks % (mpi_size - 1)
    ct = 0
    if mpi_rank == 0:
      print("nchunks = {0}, extra_chunks = {1}, chunks_per_rank = {2}, extra_frames = {3}".format(nchunks,extra_chunks,chunks_per_rank,extra_frames))
    for i in range(mpi_size):
      if (i == 0):
        skiplist[i] = first
      else:
        skiplist[i] = skiplist[i-1] + chunklist[i-1]*nchunklist[i-1]
      if extra_frames == 0:
        chunklist[i] = chunksize
        nchunklist[i] = chunks_per_rank
        if i < extra_chunks:
          nchunklist[i] = nchunklist[i] + 1
      else:
        if nchunks == mpi_size:
          chunklist[i] = chunksize
          if i < extra_frames:
            chunklist[i] = chunklist[i] + 1
          nchunklist[i] = 1
        else:
          if i == mpi_size-1:
            chunklist[i] = nsteps-ct
            nchunklist[i] = 1
          else:
            chunklist[i] = chunksize
            nchunklist[i] = chunks_per_rank
          if i < extra_chunks:
            nchunklist[i] = nchunklist[i] + 1
      ct = ct + chunklist[i]*nchunklist[i]

  if (mpi_rank == 0):
    stime = time.time()
  if (mpi_rank == 0 and mpi_size > 1):
    print("Will use ",ct," frames distributed over ",mpi_size," workers")
    if (mpi_size == nchunks):
      print("Each worker will handle ",chunks_per_rank," chunks of ",chunksize," frames with one extra frame in the first ",extra_frames," workers")
    else:
      print("Each worker will handle ",chunks_per_rank," chunks of ",chunksize," frames with one extra chunk in the first ",extra_chunks," workers.")
      if extra_frames != 0:
        print("As an exception the last worker will handle ",nchunklist[mpi_size-1]," chunks with ",chunklist[mpi_size-1]," frames")

  ct = 0
  sig_fcalc = None
  sig_icalc = None
  sig_fcalc_acc = None      # numpy running sums; see the accumulation below
  sig_icalc_acc = None
  sig_indices_ref = None

  if (skiplist[mpi_rank] <= last):
    skip_calc = False
  else:
    skip_calc = True

  if mpi_rank == 0:
    work_rank = mpi_size - 1
  else:
    work_rank = mpi_rank - 1
    
  ti = md.iterload(traj_file,chunk=chunklist[work_rank],top=top_file,skip=skiplist[work_rank])

# Each MPI rank works with its own trajectory chunk t
  
  chunk_ct = 0
  fcalc_list = None
  
  itime = time.time()

  if torch_profiler is not None:
    torch_profiler.start()

  # Driven with next() rather than `for tt in ti:` so that the time spent
  # inside mdtraj's iterload generator -- the actual file read and unit
  # conversion -- lands in a timer of its own. Under `for`, that work is
  # charged to the loop header, where neither cProfile (one call, all chunks)
  # nor a phase timer can see it. Semantics are otherwise unchanged.
  while True:

    _t_read = time.time()
    try:
      tt = next(ti)
    except StopIteration:
      break
    if torch_timing:
      _record_phase("traj read", time.time() - _t_read)

    mtime = time.time()
      
    t = tt
#    print "rank =",mpi_rank," skip = ",skiplist[mpi_rank]," chunk = ",chunklist[mpi_rank]," start time = ",t.time[0]," coords of first atom = ",t.xyz[0][0]

#    if mpi_enabled():
#      mpi_comm.Barrier()                                                                          
    na = len(t.xyz[0])

  # np.around() is needed here to avoid small changes in the coordinates
  #   that accumulate large errors in structure factor calculations. The factor
  #   of 10 is needed as the units from the trajectory are nm, whereas cctbx
  #   needs Angstrom units.

    with host_phase("coords->numpy"):
      tsites = np.around(np.array(t.xyz*10.,dtype=np.float64),3)

  # ***The following code needs modification to prevent the bcast here, as
  #   it will create a barrier that prevents execution when the number
  #   of steps is not equal to a multiple of the number of ranks
    
    if (translational_fit and chunk_ct == 0):

      if (mpi_rank == 0):

        # Get the fractional coords of the reference structure alpha carbons, for translational fit. 
        # MEW Note: only uses all c-alpha atoms in the structure and only does translational fit for now

        sites_frac = xrs.sites_frac().as_double().as_numpy_array().reshape((na,3))
        sel_indices = t.topology.select('name CA')
        ref_sites_frac = np.take(sites_frac,sel_indices,axis=0)

      else:
        ref_sites_frac = None
        sel_indices = None

    # Broadcast arrays used for translational fit

      if mpi_enabled():
        ref_sites_frac = mpi_comm.bcast(ref_sites_frac,root=0)
        sel_indices = mpi_comm.bcast(sel_indices,root=0)

  # calculate fcalc, diffuse intensity, and (if requested) density trajectory

    if mpi_rank == 0:
      stime = time.time()
      if chunk_ct == 0:
        print("Number of atoms in topology file = ",na)

    map_data = []
    num_elems = len(t)

    if (num_elems <= 0 or skip_calc):
      num_elems = 0
      xrs_sel = xrs.select(selection)
      xrs_sel.scattering_type_registry(table=scattering_table)
      if sig_fcalc is None:
        sig_fcalc = xrs_sel.structure_factors(d_min=d_min,algorithm=cctbx_method).f_calc() * 0.0
        sig_fcalc = sig_fcalc.resolution_filter(d_min=d_min,d_max=d_max)
      if sig_icalc is None:
        sig_icalc = abs(sig_fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
      print("WARNING: Worker ",work_rank," is idle")

    else:

      for i in range(num_elems):

        # overwrite crystal structure coords with trajectory coords

        # Skipped entirely under the torch bypass, which reads the frame's
        # coordinates from tsites directly and never touches xrs.
        #
        # Split rather than timed as one phase: these are the two largest
        # host costs in the frame, and they have different fixes. The flex
        # conversion is numpy->cctbx marshalling that every engine pays; the
        # setter is cctbx's per-scatterer cartesian->fractional pass.
        if not torch_bypass_xrs:
          with host_phase("coords->flex"):
            # flex.vec3_double(ndarray) converts row by row: 85 ms for 135,834
            # atoms, measured, which made this the largest host cost in the
            # frame. Handing it a flex.double built from the flattened array
            # instead hits the buffer path -- 0.5 ms, bit-identical output.
            # tsites is C-contiguous, so ravel() is a view.
            tmp = flex.vec3_double(flex.double(tsites[i].ravel()))
          with host_phase("set_sites_cart"):
            xrs.set_sites_cart(tmp)

    # perform translational fit with respect to the alpha carbons in the topology file

        if (translational_fit):
          sites_frac = xrs.sites_frac().as_double().as_numpy_array().reshape((na,3))
          x0 = [0.0,0.0,0.0]
          otime1 = time.time()
          this_sites_frac = np.take(sites_frac,sel_indices,axis=0)
          res = scipy.optimize.minimize(calc_msd,x0,method='Powell',jac=None,options={'disp': False,'maxiter': 10000})
          for j in range(3):
            sites_frac[:,j] +=res.x[j]        
            otime2 = time.time()
            # Same per-row conversion as above; sites_frac is (na, 3) C-contiguous.
            xrs.set_sites_frac(flex.vec3_double(flex.double(sites_frac.ravel())))
    #        print ("Time to optimize = ",otime2-otime1)

    # select the atoms for the structure factor calculation
    #   (under the torch bypass the selection was applied once, at setup)
        if not torch_bypass_xrs:
          with host_phase("xrs.select"):
            xrs_sel = xrs.select(selection)
        if engine == "sfall":
          _t_engine = time.time()
          pdbtmp = xrs_sel.as_pdb_file()
          pdbnam_tmp = "tmp_{rank:03d}.pdb".format(rank=mpi_rank)
          fcalcnam_tmp = "tmp_{rank:03d}.mtz".format(rank=mpi_rank)
          lognam = "sfall_{rank:03d}.log".format(rank=mpi_rank)
          with open(pdbnam_tmp,"w") as fo:
            fo.write(pdbtmp)
          with open(lognam,"w") as fo:
            subprocess.run(["bash","run_sfall.sh",pdbnam_tmp,"reference_{rank:03d}.mtz".format(rank=mpi_rank),fcalcnam_tmp],stdout=fo)
          from iotbx.reflection_file_reader import any_reflection_file
          hkl_in = any_reflection_file(file_name=fcalcnam_tmp)
          miller_arrays = hkl_in.as_miller_arrays()
          fcalc = miller_arrays[1]
          if torch_timing:
            _record_phase("sfall calc", time.time() - _t_engine)
        elif engine == "gemmi":
          # One timestamp pair rather than a `with` around the whole branch,
          # which would reindent it for no gain. The branch is a single phase
          # from the report's point of view either way.
          _t_engine = time.time()
          st = gemmi.Structure()
          st.cell = gemmi.UnitCell(*xrs_sel.unit_cell().parameters())
          # gemmi expands the model by whatever symmetry the structure
          # declares, so expand_symmetry=False is expressed by declaring P1.
          # The reflection files still carry the true space group, which comes
          # from the miller set rather than from here.
          st.spacegroup_hm = (
            xrs_sel.space_group().type().lookup_symbol()
            if expand_symmetry else "P 1")
          
          model = gemmi.Model("1")
          chain = gemmi.Chain("A")
          res = gemmi.Residue()
          res.name = "UNK"
          res.seqid = gemmi.SeqId("1")
          
          # Extract flat arrays for fast iteration
          coords = xrs_sel.sites_cart()
          u_isos = xrs_sel.extract_u_iso_or_u_equiv()
          occs = xrs_sel.scatterers().extract_occupancies()
          elements = [s.scattering_type for s in xrs_sel.scatterers()]
          
          u_to_b = 8.0 * np.pi**2

          if mpi_rank == 0 and chunk_ct == 0 and i == 0:
            cctbx_frac0 = np.array(xrs_sel.sites_frac()[0])
            gemmi_frac0 = st.cell.fractionalize(gemmi.Position(*coords[0]))
            gemmi_frac0 = np.array([gemmi_frac0.x, gemmi_frac0.y, gemmi_frac0.z])
            print("DIAGNOSTIC atom 0: element =", elements[0], ", occ =", occs[0],
                  ", u_iso =", u_isos[0], ", cartesian =", tuple(coords[0]))
            print("  cctbx sites_frac()      =", cctbx_frac0)
            print("  gemmi cell.fractionalize=", gemmi_frac0)
            print("  max abs diff            =", np.max(np.abs(cctbx_frac0 - gemmi_frac0)))

    
          # zip() executes in C, minimizing Python loop overhead
          for (x, y, z), u, occ, el in zip(coords, u_isos, occs, elements):
            atom = gemmi.Atom()
            atom.name = el
            atom.element = gemmi.Element(el)
            atom.pos = gemmi.Position(x, y, z)
            atom.b_iso = u * u_to_b
            atom.occ = occ
            res.add_atom(atom)
        
          chain.add_residue(res)
          model.add_chain(chain)
          st.add_model(model)

          calc = gemmi.DensityCalculatorX()
          calc.d_min = d_min
          calc.cutoff = gemmi_cutoff
          calc.grid.unit_cell = st.cell
          calc.grid.spacegroup = st.find_spacegroup()
          calc.put_model_density_on_grid(st[0])
          real_space_array = np.array(calc.grid, copy=False)
          if mpi_rank == 0 and chunk_ct == 0 and i == 0:
            print("DIAGNOSTIC gemmi grid dtype =", real_space_array.dtype,
                  ", shape =", real_space_array.shape,
                  ", sum =", real_space_array.sum())
            if save_density:
              np.save("density_gemmi.npy", np.asarray(real_space_array))
              print("DIAGNOSTIC saved density_gemmi.npy")
        
          # 2. Use SciPy's Inverse FFT to compute the positive exponent (+2pi i h x)
          # Setting workers=-1 tells SciPy to use all available CPU cores.
          # ifftn automatically normalizes the sum by (1 / N_voxels).
          
          complex_grid = scipy.fft.ifftn(real_space_array, workers=-1, overwrite_x=True)

          # 3. Multiply by unit cell Volume to complete the absolute scaling
          complex_grid *= st.cell.volume
          grid = calc.grid

          # Get the integer grid dimensions before the FFT transforms it
          Nu, Nv, Nw = grid.nu, grid.nv, grid.nw
          if mpi_rank == 0 and chunk_ct == 0 and i == 0:
            print("gemmi engine: grid =", (Nu, Nv, Nw), ", cutoff =", gemmi_cutoff,
                  ", blur =", calc.blur, ", d_min =", d_min)

          # --- 6. Generate Miller Indices (HKL) up to d_min ---
          # We use cctbx here just to easily get the mathematically correct list of HKLs 
          # for the asymmetric unit at this resolution limit.
          miller_set = miller.build_set(
            crystal_symmetry=xrs_sel.crystal_symmetry(),
            anomalous_flag=False, # We want unique reflections, assuming Friedel's law
            d_min=d_min
          )
          hkls_flex = miller_set.indices()
          hkls_np = np.array(hkls_flex)
          
          # --- 7. Vectorized Extraction from Gemmi Complex Grid ---
          h = hkls_np[:, 0]
          k = hkls_np[:, 1]
          l = hkls_np[:, 2]
          # Since we used half_l=False, the full reciprocal space grid is available.
          # We just apply grid wrapping (modulo) for all dimensions based on periodic boundary conditions.
          u = h % Nu
          v = k % Nv
          w = l % Nw

          # Extract the complex values directly from the NumPy view of the grid
          # Since we used half_l=False, the full reciprocal space grid is available.
          # Apply the complex conjugate to the reflections where we used the Friedel mate
        
          complex_array = np.array(getattr(complex_grid, 'array', complex_grid), copy=False)
          extracted_sf = complex_array[u, v, w]

          extracted_sf = np.conjugate(extracted_sf)
          d_spacings_np = np.array(miller_set.d_spacings().data())
        
          # The un-blur math is mathematically: exp( B / (4 * d^2) )
          inv_d2 = 1.0 / (d_spacings_np ** 2)
          unblur_multiplier = np.exp(calc.blur * 0.25 * inv_d2)
        
          # Apply the multiplier across all reflections instantly
          extracted_sf *= unblur_multiplier

          #Fix type
          extracted_sf = extracted_sf.astype(np.complex128)

          # --- 8. Construct CCTBX Miller Array ---
    
          # Safely convert the complex numpy array into a cctbx flex.complex_double array.     
          # np.ascontiguousarray ensures the memory layout aligns cleanly for the C++ backend.
          sf_flex = flex.complex_double(np.ascontiguousarray(extracted_sf, dtype=np.complex128))
    
          # Combine the original miller_set and the flex array to build the final object
          fcalc = miller.array(miller_set=miller_set, data=sf_flex)
          
          # Optional: Attach labels so CCTBX knows what this data represents
          fcalc.set_info(miller.array_info(labels=["FWT", "PHIFWT"]))
          fcalc = fcalc.resolution_filter(d_min=d_min,d_max=d_max)
          if torch_timing:
            _record_phase("gemmi calc", time.time() - _t_engine)
        elif engine == "torch":
          if torch_bypass_xrs:
            # Everything cctbx was being asked for, without cctbx: the frame's
            # coordinates fractionalized in one matmul, and the occupancies and
            # element indices computed once at setup.
            with host_phase("frac coords"):
              cart_np = tsites[i] if torch_sel_idx is None else tsites[i][torch_sel_idx]
              coords_frac_np = cart_np @ torch_frac_matrix_np.T
            occ_np = torch_occ_np
            element_idx_np = torch_element_idx_np
          else:
            with host_phase("sites_frac+occ"):
              coords_frac_np = xrs_sel.sites_frac().as_double().as_numpy_array().reshape((-1, 3))
              occ_np = np.array(xrs_sel.scatterers().extract_occupancies())
            # Timed separately from the coordinates because, unlike them, this is
            # a Python-level pass over every scatterer twice -- and unlike them it
            # does not change from frame to frame.
            with host_phase("element idx"):
              elements_this_frame = [s.scattering_type for s in xrs_sel.scatterers()]
              element_idx_np = np.array(
                [torch_element_to_idx[e] for e in elements_this_frame], dtype=np.int64
              )

          with torch_phase("host->device", torch_device):
            frac_t = torch.tensor(coords_frac_np, dtype=torch_dtype, device=torch_device)
            occ_t = torch.tensor(occ_np, dtype=torch_dtype, device=torch_device)
            element_idx_t = torch.tensor(element_idx_np, dtype=torch.long, device=torch_device)

          if use_top_bfacs:
            assert torch_atom_A.shape[0] == frac_t.shape[0], (
              f"per-frame atom count ({frac_t.shape[0]}) doesn't match the "
              f"setup-time count ({torch_atom_A.shape[0]}) that the per-atom B-factor "
              "kernels were built from -- the selection/atom count should be stable "
              "frame to frame; do not trust results until this is resolved."
            )
            atom_A_t = torch_atom_A
            atom_lam_t = torch_atom_lam
            atom_radius_ang_t = torch_atom_radius_ang
          else:
            atom_A_t = torch_elem_A[element_idx_t]
            atom_lam_t = torch_elem_lam[element_idx_t]
            atom_radius_ang_t = torch_elem_radius_ang[element_idx_t]

          # No gradients needed here -- xtraj.py is a forward-only diffuse
          # calculation, not a training loop. The same structure_factors_one_config
          # call works with requires_grad=True instead, outside no_grad(), in a
          # script that actually backpropagates (see example_usage.py).
          with torch.no_grad():
            # Probe: with the previous frame's grid and F still alive, the
            # allocator cannot hand the splat back the block it just used --
            # unlike the back-to-back repeat it is being compared against.
            if torch_free_early:
              torch_density = None
              F_t = None
            _splat_stats = {} if torch_splat_stats else None
            with torch_phase("splat", torch_device):
              torch_density = splat_density(
                frac_t, element_idx_t, occ_t,
                atom_A_t, atom_lam_t, torch_elem_offsets, atom_radius_ang_t,
                torch_grid_shape, torch_orth_matrix, torch_taper_width,
                max_atoms_per_batch=torch_max_atoms_per_batch,
                compile_core=torch_compile,
                out=torch_grid_buf, stats=_splat_stats,
                **torch_pairs_kwarg,
              )
            if _splat_stats is not None:
              torch_splat_stats_log.append(_splat_stats)


            # Symmetry-expand the density grid, exactly as gemmi's
            # put_model_density_on_grid() does via symmetrize_sum(). No-op
            # (empty torch_grid_ops) for P1.
            with torch_phase("symmetrize", torch_device):
              torch_density = symmetrize_sum(torch_density, torch_grid_ops)
            if mpi_rank == 0 and chunk_ct == 0 and i == 0:
              print("DIAGNOSTIC torch_density.device =", torch_density.device,
                    ", dtype =", torch_density.dtype)
              dev_type = torch_density.device.type
              if dev_type == "mps":
                print("DIAGNOSTIC MPS current_allocated_memory =",
                      torch.mps.current_allocated_memory(), "bytes")
              elif dev_type == "cuda":
                print("DIAGNOSTIC CUDA memory_allocated =",
                      torch.cuda.memory_allocated(torch_density.device), "bytes")
              else:
                print("DIAGNOSTIC running on CPU (device.type =", dev_type, ")")
              print("DIAGNOSTIC torch grid shape =", tuple(torch_density.shape),
                    ", dtype =", torch_density.dtype, ", sum =", torch_density.sum().item())

              if save_density:
                np.save("density_torch.npy", torch_density.cpu().numpy())
                print("DIAGNOSTIC saved density_torch.npy")
            with torch_phase("fft+extract", torch_device):
              F_t = compute_fcalc(
                torch_density, torch_cell_volume, torch_hkl, torch_orth_matrix, blur=0.0,
              )

          with torch_phase("device->host", torch_device):
            extracted_sf = F_t.cpu().numpy().astype(np.complex128)
          with torch_phase("into cctbx", torch_device):
            sf_flex = flex.complex_double(np.ascontiguousarray(extracted_sf, dtype=np.complex128))
            fcalc = miller.array(miller_set=torch_miller_set, data=sf_flex)
            fcalc.set_info(miller.array_info(labels=["FWT", "PHIFWT"]))
            fcalc = fcalc.resolution_filter(d_min=d_min,d_max=d_max)
        else:
          with host_phase("cctbx calc"):
            # Same distinction as the other two engines: cctbx generates the
            # symmetry copies from the structure's own space group, so a model
            # that already holds them is calculated in P1.
            #
            # The reflections must still be the CRYSTAL's, though. Left to
            # itself, structure_factors() on a P1 structure returns P1's
            # indices -- 11,965 rather than 3,388 here, labelled P1 -- so the
            # Miller set is built at the true symmetry and the P1 structure is
            # sampled at exactly those indices. That is what the torch and
            # gemmi branches do, and it is NOT the same as merging equivalents
            # afterwards, which would average the symmetry-related reflections
            # and reintroduce the averaging expand_symmetry=False exists to
            # avoid.
            if expand_symmetry:
              xrs_sel.scattering_type_registry(table=scattering_table)
              fcalc = xrs_sel.structure_factors(d_min=d_min,algorithm=cctbx_method).f_calc()
            else:
              p1_symmetry = crystal.symmetry(
                unit_cell=xrs_sel.unit_cell(),
                space_group_info=cctbx.sgtbx.space_group_info(symbol="P 1"))
              xrs_calc = xrs_sel.customized_copy(crystal_symmetry=p1_symmetry)
              xrs_calc.scattering_type_registry(table=scattering_table)
              # structure_factors_from_scatterers asserts that the structure
              # and the Miller set share a space group, so the set is built at
              # the crystal's symmetry -- which fixes WHICH reflections -- and
              # then relabelled P1 to satisfy that assertion. The result is
              # relabelled back. Only the labels move; no index is merged,
              # mapped or averaged.
              cctbx_ms = miller.build_set(
                crystal_symmetry=xrs_sel.crystal_symmetry(),
                anomalous_flag=False, d_min=d_min)
              fcalc = cctbx_ms.customized_copy(
                crystal_symmetry=p1_symmetry
              ).structure_factors_from_scatterers(
                xray_structure=xrs_calc, algorithm=cctbx_method).f_calc()
              fcalc = fcalc.customized_copy(
                crystal_symmetry=xrs_sel.crystal_symmetry())
            fcalc = fcalc.resolution_filter(d_min=d_min,d_max=d_max)

        if do_opt:
          diffuse_expt_common,fcalc_common = diffuse_expt.common_sets(fcalc.as_non_anomalous_array())
          icalc_common = abs(fcalc_common).set_observation_type_xray_amplitude().f_as_f_sq()
          fcalc_common_data = np.array(fcalc_common.data())
          icalc_common_data = np.array(icalc_common.data())
          if fcalc_list is None:
            fcalc_list = np.empty((chunklist[work_rank]*nchunklist[work_rank],fcalc_common_data.size),dtype=fcalc_common_data.dtype)
            icalc_list = np.empty((chunklist[work_rank]*nchunklist[work_rank],icalc_common_data.size),dtype=icalc_common_data.dtype)
            sig_fcalc = fcalc_common
            sig_icalc = icalc_common
          fcalc_list[ct] = fcalc_common_data
          icalc_list[ct] = icalc_common_data
    # Commented out some density trajectory code
    #    if not (dens_file is None):
    #      this_map = fcalc.fft_map(d_min=d_min, d_max=d_max, resolution_factor = 0.5)
    #      real_map_np = this_map.real_map_unpadded().as_numpy_array()
    #      map_data.append(real_map_np)
        else:
          # Running sums are accumulated as numpy arrays rather than by adding
          # miller arrays frame by frame.
          #
          # Every frame shares one Miller set -- same cell, same d_min -- so
          # miller.__add__ repeats identical index matching on every frame.
          # Measured at 53 ms per call, twice per frame: 26 s of a 251-frame
          # run, about 15% of it, against ~8% for the actual GPU calculation.
          # |F|^2 is likewise formed directly instead of going through
          # abs() -> set_observation_type -> f_as_f_sq(), which agrees to
          # 2e-18 relative (pure float rounding).
          #
          # sig_fcalc/sig_icalc are still kept as the FIRST frame's miller
          # arrays: downstream code uses them as containers, overwriting
          # .data() with the reduced totals, so the objects are needed even
          # though their values are not.
          with host_phase("accumulate"):
            fcalc_np = fcalc.data().as_numpy_array()
            if sig_fcalc is None:
              sig_fcalc = fcalc
              sig_icalc = abs(fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
              sig_fcalc_acc = fcalc_np.copy()
              sig_icalc_acc = sig_icalc.data().as_numpy_array().copy()
              sig_indices_ref = fcalc.indices()
            else:
              # Adding the data arrays elementwise is only equivalent to adding
              # the miller arrays if the Miller sets agree; check rather than
              # assume, since a silent mismatch would misalign every reflection.
              assert fcalc.indices().all_eq(sig_indices_ref), (
                "Miller indices changed between frames; the running sums assume "
                "a single fixed Miller set")
              sig_fcalc_acc += fcalc_np
              sig_icalc_acc += fcalc_np.real**2 + fcalc_np.imag**2
        ct = ct + 1
        # One profiler step per FRAME, not per chunk: the schedule's
        # wait/warmup/active counts are frames.
        if torch_profiler is not None:
          torch_profiler.step()

    chunk_ct = chunk_ct + 1

    print("Worker ",work_rank,"on MPI rank ",mpi_rank," processed chunk ",chunk_ct," of ",nchunklist[work_rank]," with ",chunklist[work_rank]," frames in ",time.time()-mtime," seconds")

    if (chunk_ct >= nchunklist[work_rank]):
      break

  loop_wall = time.time() - itime

  if torch_profiler is not None:
    # on_trace_ready has normally already fired, at the end of the active
    # window; this releases the profiler for the (shorter) case where the run
    # ended first.
    torch_profiler.stop()

# Frames this rank actually processed. It can be fewer than the PLANNED
# chunklist*nchunklist: a final short chunk when chunk= does not divide the
# frame count, or a trajectory that ends before the plan says it should.
#
# fcalc_list/icalc_list were sized from the planned count with np.empty(), so
# any row that was never written holds uninitialised memory. They are consumed
# with np.sum(..., axis=0) and `for x in range(len(fcalc_list))`, which would
# fold that garbage straight into the result -- silently, since np.empty gives
# plausible-looking floats. Trim to what was filled.

  n_frames_this = ct
  if mpi_rank == 0:
    report_phases(n_frames_this, loop_wall)
  if fcalc_list is not None and n_frames_this < len(fcalc_list):
    fcalc_list = fcalc_list[:n_frames_this]
    icalc_list = icalc_list[:n_frames_this]


    
# Commented out some density trajectory code
#  if not (dens_file is None):
#    map_grid = np.concatenate(map_data)
#    Ni = map_data[0].shape[0]
#    Nj = map_data[0].shape[1]
#    Nk = map_data[0].shape[2]
#    map_grid_3D = np.reshape(map_grid,(len(tsites),Ni,Nj,Nk))
#    np.save(dens_file,map_grid_3D)

  if profile and mpi_rank == 0:
    profiler.disable()
    print("\n=== cProfile: top 30 by cumulative time (rank 0) ===")
    pstats.Stats(profiler).sort_stats('cumtime').print_stats(30)
    print("\n=== cProfile: top 30 by total time in the function itself ===")
    pstats.Stats(profiler).sort_stats('tottime').print_stats(30)

  #profiler.disable()

  #stats = pstats.Stats(profiler)

  # Sort by 'tottime' or 'cumtime', and print the top 20 slowest functions
  #stats.sort_stats('tottime').print_stats(50)

  print("Worker ",work_rank," is done with individual calculations")
  sys.stdout.flush()

  if mpi_enabled():
    mpi_comm.Barrier()

  if (mpi_rank == 0):
    mtime = time.time()
    print("TIMING: Calculate individual statistics = ",mtime-itime)

# If optimization is on, calculate sig_fcalc and sig_icalc
  if do_opt:
    if apply_bfac:
      miller_set = sig_fcalc.set()
      dwf_array = miller_set.debye_waller_factors(b_iso=20.0*d_min*d_min)
      dwf_data_np = np.array(dwf_array.data())
      # fcalc_list /= dwf_data_np[np.newaxis,:]
      # icalc_list /= dwf_data_np[np.newaxis,:]
      for x in range(len(fcalc_list)):
        fcalc_list[x] /= dwf_data_np
        icalc_list[x] /= dwf_data_np * dwf_data_np
    #At this point fcalc_list and icalc_list can be used for optimization
    #Still need to calculate the sums across all MPI ranks, however.
    sig_fcalc_np = np.sum(fcalc_list,axis=0)
    sig_icalc_np = np.sum(icalc_list,axis=0)
  else:
    # Accumulated above as numpy, so no conversion is needed here. Fall back to
    # the miller arrays for the case where no frame was processed on this rank.
    if sig_fcalc_acc is not None:
      sig_fcalc_np = sig_fcalc_acc
      sig_icalc_np = sig_icalc_acc
    else:
      sig_fcalc_np = sig_fcalc.data().as_numpy_array()
      sig_icalc_np = sig_icalc.data().as_numpy_array()
    
# perform reduction of sig_fcalc, sig_icalc, and ct

  # if mpi_rank == 0:
  #   tot_sig_fcalc_np = np.zeros_like(sig_fcalc_np)
  #   tot_sig_icalc_np = np.zeros_like(sig_icalc_np)
  # else:
  #   tot_sig_fcalc_np = None
  #   tot_sig_icalc_np = None
  tot_sig_fcalc_np = np.zeros_like(sig_fcalc_np)
  tot_sig_icalc_np = np.zeros_like(sig_icalc_np)

  if mpi_enabled():
    mpi_comm.Barrier()                                                        

  if mpi_enabled():
    mpi_comm.Allreduce(sig_fcalc_np,tot_sig_fcalc_np,op=MPI.SUM)
    mpi_comm.Allreduce(sig_icalc_np,tot_sig_icalc_np,op=MPI.SUM)
    ct = mpi_comm.allreduce(ct,op=MPI.SUM)
  else:
    tot_sig_fcalc_np = sig_fcalc_np
    tot_sig_icalc_np = sig_icalc_np

# compute averages

  if (mpi_rank == 0):
    sig_fcalc_data = sig_fcalc.data()
    sig_icalc_data = sig_icalc.data()
    for x in range(sig_fcalc_data.size()):
      sig_fcalc_data[x] = tot_sig_fcalc_np[x]
      sig_icalc_data[x] = tot_sig_icalc_np[x]
    avg_fcalc = sig_fcalc / float(ct)
    avg_icalc = sig_icalc / float(ct)
    if apply_bfac and not do_opt:
      miller_set = avg_fcalc.set()
      dwf_array = miller_set.debye_waller_factors(b_iso=20.0*d_min*d_min)
      dwf_data = dwf_array.data()
      avg_fcalc_data = avg_fcalc.data()
      avg_icalc_data = avg_icalc.data()
      for x in range(0,avg_fcalc_data.size()):
        avg_fcalc_data[x] /= dwf_data[x]
        avg_icalc_data[x] /= dwf_data[x] * dwf_data[x]
    # Calculate difference if requested
    if diff_mode:
      avg_fcalc_ref_data = avg_fcalc_ref.data()
      avg_icalc_ref_data = avg_icalc_ref.data()
      for x in range(0,avg_fcalc_data.size()):
        avg_fcalc_data[x] = (avg_fcalc_ref_data[x] * float(nsteps_ref) - avg_fcalc_data[x] * float(ct)) / float(nsteps_ref - ct)
        avg_icalc_data[x] = (avg_icalc_ref_data[x] * float(nsteps_ref) - avg_icalc_data[x] * float(ct)) / float(nsteps_ref - ct)
    sq_avg_fcalc = abs(avg_fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
    sq_avg_fcalc_data = sq_avg_fcalc.data()
    diffuse_array=avg_icalc*1.0
    diffuse_data = diffuse_array.data()
    for x in range(0,diffuse_data.size()):
      diffuse_data[x]=diffuse_data[x]-sq_avg_fcalc_data[x]
    etime = time.time()
    print("TIMING: Reduction = ",etime-mtime)
    print("TIMING: Total diffuse calculation = ",etime-stime)

# Compute the correlation with the data, if available

    if diffuse_data_file is not None:
      print("Calculating Correlation")
      if do_opt:
        #Common sets already have been extracted in this case
        diffuse_array_common = diffuse_array
      else:
        diffuse_expt_common, diffuse_array_common = diffuse_expt.common_sets(diffuse_array.as_non_anomalous_array())
      if corr_aniso:
        diffuse_array_common = to_aniso(diffuse_array_common,apply_symmetry_str)
      C = np.corrcoef(np.array([diffuse_expt_common.data(),diffuse_array_common.data()]))
      print("Pearson correlation between diffuse simulation and data = ",C[0,1])
      Camp = np.corrcoef(np.sqrt(np.abs(np.array([diffuse_expt_common.data(),diffuse_array_common.data()]))))
      print("   Correlation between amplitudes = ",Camp[0,1])

# write fcalc

    if not partial_sum_mode:
      avg_fcalc.as_mtz_dataset('FWT').mtz_object().write(file_name=fcalc_file)
    else:
      sig_fcalc.as_mtz_dataset('FWTsum').mtz_object().write(file_name=fcalc_file)

# write density map

#    if not partial_sum_mode:
#      symmetry_flags = maptbx.use_space_group_symmetry
#      dmap = avg_fcalc.fft_map(d_min=d_min,d_max=d_max,resolution_factor=0.5,symmetry_flags=symmetry_flags)
#      dmap.apply_volume_scaling()
#      dmap = avg_fcalc.fft_map(f_000=f_000.f_000)
#      dmap.as_ccp4_map(file_name=density_file)

# write icalc

    print("Average Icalc:")
    count=0
    for hkl,intensity in avg_icalc:
      print("%4d %4d %4d   %10.2f" %(hkl+tuple((intensity,))))
      count+=1
      if count>10: break
    if not partial_sum_mode:
      avg_icalc.as_mtz_dataset('Iavg').mtz_object().write(file_name=icalc_file)
    else:
      sig_icalc.as_mtz_dataset('Isum').mtz_object().write(file_name=icalc_file)

# write diffuse

    print("Diffuse:")
    count=0
    for hkl,intensity in diffuse_array:
      print("%4d %4d %4d   %10.2f" %(hkl+tuple((intensity,))))
      count+=1
      if count>10: break
    if(diffuse_file.endswith(".mtz")):
      if not partial_sum_mode:
        diffuse_array.as_mtz_dataset('ID').mtz_object().write(file_name=diffuse_file)
      else:
        diffuse_array.as_mtz_dataset('IDpart').mtz_object().write(file_name=diffuse_file)
    else:
      f=open(diffuse_file,'w')
      for hkl,intensity in diffuse_array:
        print("%4d %4d %4d   %10.2f" %(hkl+tuple((intensity,))),file=f)
      f.close()

#Perform optimization      
  if do_opt:
    
    if mpi_rank == 0:
      stime = time.time()
      print("Doing optimization using ",ct," initial frames")
    else:
      diffuse_array_common = None

    # Serial runs have no communicator; rank 0 set diffuse_array_common above.
    if mpi_enabled():
      diffuse_array_common = mpi_comm.bcast(diffuse_array_common,root=0)

    #Initialize correlations array
    w = np.ones(ct)
    ct_nonzero = ct
    first_this = (skiplist[work_rank]-first)
    # Frames actually processed, not the planned chunklist*nchunklist, so the
    # slices below line up with fcalc_list/icalc_list when the final chunk was
    # short.
    last_this = first_this + n_frames_this - 1
    #Get the slice for the section handled by this rank
    C_all_this = np.zeros(ct)
    C_this = C_all_this[first_this:last_this+1]
    w_indices = None
    if filtered_file is not None:
      w_indices = np.loadtxt(filtered_file,dtype=int)
      w[w_indices] = 0
      ct_nonzero=np.count_nonzero(w)
      if mpi_rank == 0:
        print("Filtered frames from ",filtered_file,", ",ct_nonzero," out of ",ct," remaining")
    w_this = w[first_this:last_this+1]
    keep_optimizing = True
    h5_mpi=False
    
    if checkpoint_opt:
      if mpi_rank == 0:
        print("Writing checkpoint file on rank 0")
        with h5py.File("checkpoint.nxs", "w") as f:
          f.attrs["NX_class"] = "NXroot"
          
          write_nexus_metadata(f, diffuse_expt_common)
            
      # Wait for Rank 0 to finish metadata definitions
      if mpi_enabled():
        mpi_comm.Barrier()
      # --- Parallel Numpy Writing (All Ranks) ---
      if h5_mpi:
        with h5py.File("checkpoint.nxs", "a", driver="mpio", comm=mpi_comm) as f:
          data_shape = (last-first+1,fcalc_list.shape[1],2)
          dset = f.create_dataset("entry/data/fcalc_list", data_shape, dtype=np.float32)
                  
          dset[first_this:last_this+1, :, :] = np.stack((np.real(fcalc_list),np.imag(fcalc_list)),axis=-1).astype(np.float32)

          data_shape = (last-first+1)
          dset = f.create_dataset("entry/data/weights",data_shape,dtype=np.float32)
          dset[first_this:last_this+1] = w_this.astype(np.float32)
      else:
        for i in range(mpi_size):
          if work_rank == i:
            if work_rank == 0:          
              with h5py.File("checkpoint.nxs", "a") as f:
                print("Worker 0 creating the datasets")
                data_shape = (last-first+1,fcalc_list.shape[1],2)
                dset = f.create_dataset("entry/data/fcalc_list", data_shape, dtype=np.float32)
                data_shape = (last-first+1)
                dset = f.create_dataset("entry/data/weights",data_shape,dtype=np.float32)
            print("Worker ",work_rank," writing fcalc_list with ",fcalc_list.nbytes/2/1024/1024," MB")  
            with h5py.File("checkpoint.nxs", "a") as f:            
              dset = f["entry/data/fcalc_list"]
              dset[first_this:last_this+1, :, :] = np.stack((np.real(fcalc_list),np.imag(fcalc_list)),axis=-1).astype(np.float32)
              dset = f["entry/data/weights"]
              dset[first_this:last_this+1] = w_this.astype(np.float32)
          if mpi_enabled():
            mpi_comm.Barrier()
      if mpi_rank == 0:
        print("Nexus file successfully written with cctbx metadata and parallel arrays.")          

    with open("deleted_frames.txt","w") as f:
      if (w_indices is not None):
        for x in range(len(w_indices)):
          f.write("{0}\n".format(w_indices[x]))
    for x in range(len(fcalc_list)):
        if w_this[x] == 0:
          try:
            sig_fcalc_np = sig_fcalc_np - fcalc_list[x]
          except TypeError:
            print("Couldn't calculate fcalc difference on worker ",work_rank," with len(C_this), ct_nonzero, x = ",len(C_this),ct_nonzero,x)
            print("Types of tot_sig_fcalc_np, fcalc_list[x] = ",type(tot_sig_fcalc_np),type(fcalc_list[x]))
          try:            
            sig_icalc_np = sig_icalc_np - icalc_list[x]
          except:
            print("Couldn't calculate icalc difference on worker ",work_rank," with len(C_this), ct_nonzero, x = ",len(C_this),ct_nonzero,x)
            print("Types of tot_sig_icalc_np, icalc_list[x] = ",type(tot_sig_icalc_np),type(icalc_list[x]))
    if mpi_enabled():
      mpi_comm.Allreduce(sig_fcalc_np,tot_sig_fcalc_np,op=MPI.SUM)
      mpi_comm.Allreduce(sig_icalc_np,tot_sig_icalc_np,op=MPI.SUM)
    else:
      tot_sig_fcalc_np = sig_fcalc_np
      tot_sig_icalc_np = sig_icalc_np

    diffuse_this = np.ascontiguousarray((ct_nonzero*tot_sig_icalc_np - tot_sig_fcalc_np * tot_sig_fcalc_np.conjugate()).real)
    if corr_aniso:
      flex_diffuse_this = flex.double(diffuse_this)
      diffuse_array_common = diffuse_array_common.customized_copy(data = flex_diffuse_this)
      diffuse_array_common = to_aniso(diffuse_array_common,apply_symmetry_str)
      diffuse_this = np.array(diffuse_array_common.data())
    diffuse_expt_np = np.array(diffuse_expt_common.data())
    C_ref = np.corrcoef(np.array([diffuse_expt_np,diffuse_this]))[0,1]
    if mpi_rank == 0:
      print("Initial correlation after filtering = ",C_ref)
      
    while keep_optimizing:
      C_this[:] = 0
#      print("Worker = ",work_rank,"ct = ",ct,"len(C_this) = ",len(C_this),first_this,last_this)
    #Calculation the correlations leaving out each frame
      ct_nonzero = ct_nonzero - 1
      for x in range(len(C_this)):
        if w_this[x] != 0:
          try:
            sig_fcalc_this = tot_sig_fcalc_np - fcalc_list[x]
          except TypeError:
            print("Couldn't calculate fcalc difference on worker ",work_rank," with len(C_this), ct_nonzero, x = ",len(C_this),ct_nonzero,x)
            print("Types of tot_sig_fcalc_np, fcalc_list[x] = ",type(tot_sig_fcalc_np),type(fcalc_list[x]))
          try:            
            sig_icalc_this = tot_sig_icalc_np - icalc_list[x]
          except:
            print("Couldn't calculate icalc difference on worker ",work_rank," with len(C_this), ct_nonzero, x = ",len(C_this),ct_nonzero,x)
            print("Types of tot_sig_icalc_np, icalc_list[x] = ",type(tot_sig_icalc_np),type(icalc_list[x]))
          diffuse_this = np.ascontiguousarray((ct_nonzero*sig_icalc_this - sig_fcalc_this * sig_fcalc_this.conjugate()).real)
          if corr_aniso:
            flex_diffuse_this = flex.double(diffuse_this)
            diffuse_array_common = diffuse_array_common.customized_copy(data = flex_diffuse_this)
            diffuse_array_common = to_aniso(diffuse_array_common,apply_symmetry_str)
            diffuse_this = np.array(diffuse_array_common.data())
          C_this[x] = np.corrcoef(np.array([diffuse_expt_np,diffuse_this]))[0,1]
      C_all = np.zeros(ct)
      if mpi_enabled():
        mpi_comm.Allreduce(C_all_this,C_all,op=MPI.SUM)
      else:
        # One rank owns every frame, so its partial array IS the total.
        C_all = C_all_this
      if np.max(C_all) > C_ref:
        C_ref = np.max(C_all)
        maxind = np.argmax(C_all)
#        print("DEBUG: ",work_rank,maxind,first_this,last_this)
        if maxind >= first_this and maxind <= last_this:
          which_rank = mpi_rank
        else:
          which_rank = 0
        if mpi_enabled():
          which_rank = mpi_comm.allreduce(which_rank,MPI.SUM)
#        print("which_rank = ",which_rank)
        if which_rank == mpi_rank:          
          tot_sig_fcalc_np = tot_sig_fcalc_np - fcalc_list[maxind - first_this]
          tot_sig_icalc_np = tot_sig_icalc_np - icalc_list[maxind - first_this]
        else:
          tot_sig_fcalc_np = None
          tot_sig_icalc_np = None
        # Serial: which_rank == mpi_rank == 0, so the branch above already
        # subtracted the frame and both arrays are current.
        if mpi_enabled():
          tot_sig_fcalc_np = mpi_comm.bcast(tot_sig_fcalc_np,root=which_rank)
          tot_sig_icalc_np = mpi_comm.bcast(tot_sig_icalc_np,root=which_rank)
        w[maxind] = 0
        if mpi_rank == 0:
          print("Max correlation, index = ",C_ref,maxind)
          with open("deleted_frames.txt","a") as f:
            f.write("{0}\n".format(maxind))
      else:
        keep_optimizing = False
    if (mpi_rank == 0):
      etime = time.time()
      print("TIMING: Optimization took ",etime-stime," secs")
      print("Final correlation is ",C_ref)
      print("Total ",ct_nonzero+1," frames remaining (see selected_frames.ndx)")
      keep_idx = np.where(w!=0)
      print(keep_idx[0])
      with open("selected_frames.ndx","w") as f:
        f.write("[ selected frames ]\n")
        for x in range(len(keep_idx[0])):
          f.write("{0}\n".format(keep_idx[0][x]+1))
      # print("Diffuse Expt:")
      # count=0
      # for hkl,intensity in diffuse_expt_common:
      #   print("%4d %4d %4d   %10.2f" %(hkl+tuple((intensity,))))
      #   count+=1
      #   if count>10: break
      # print("Diffuse Array:")
      # count=0
      # for hkl,intensity in diffuse_array_common:
      #   print("%4d %4d %4d   %10.2f" %(hkl+tuple((intensity,))))
      #   count+=1
      #   if count>10: break
