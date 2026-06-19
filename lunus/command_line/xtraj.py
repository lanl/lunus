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
import cctbx.sgtbx
import subprocess
import pickle
import h5py
import math
from scipy.interpolate import CubicSpline

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

def to_aniso(miller_array,apply_symmetry_str="P1"):
    """
    Calculates and subtracts the isotropic component from a cctbx miller_array in place.
    
    The isotropic component is calculated by averaging the data in spherical shells 
    in reciprocal space. The shell thickness is defined by the length of the reciprocal 
    unit cell diagonal (d* of the 1,1,1 reflection). A natural cubic spline is then 
    used to interpolate the background for each specific reflection, which is 
    subtracted directly from the input array.
    
    Args:
        miller_array (cctbx.miller.array): The non-anomalous Miller array to process.
                                           Must contain data of type double.
                                           
    Returns:
        cctbx.miller.array: The same input array, modified in place to contain 
                            only the anisotropic values.
                            
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
    data_np = miller_array.data().as_numpy_array()

    # 2. Determine spherical shell bins
    max_d_star = np.max(d_star_np)
    num_bins = int(np.ceil(max_d_star / shell_thickness))
    
    # Handle edge case where the array might be entirely empty
    if num_bins == 0:
        return miller_array
        
    bin_edges = np.linspace(0, num_bins * shell_thickness, num_bins + 1)
    bin_indices = np.digitize(d_star_np, bin_edges)

    bin_centers = []
    bin_averages = []

    # 3. Calculate mean values for each populated shell
    for i in range(1, len(bin_edges)):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_data = data_np[mask]
            mean_val = np.mean(bin_data)
            
            center = (bin_edges[i-1] + bin_edges[i]) / 2.0
            bin_centers.append(center)
            bin_averages.append(mean_val)

    bin_centers = np.array(bin_centers)
    bin_averages = np.array(bin_averages)

    # 4. Fit the interpolating function
    if len(bin_centers) < 4:
        raise ValueError(
            f"Only found {len(bin_centers)} populated shells. "
            "A minimum of 4 is required for cubic spline interpolation."
        )
        
    spline = CubicSpline(bin_centers, bin_averages, bc_type='natural')

    # 5. Evaluate and subtract in place
    isotropic_bg_np = spline(d_star_np)
    #isotropic_bg_flex = flex.double(isotropic_bg_np)

    # The -= operator directly mutates the underlying C++ memory block
    data_np -= isotropic_bg_np
    data_flex = flex.double(data_np)
    miller_array = miller_array.customized_copy(data = data_flex)
    
    miller_array = symmetry_average(miller_array,apply_symmetry_str)

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
    fcalc = xrs_sel.structure_factors(d_min=d_min).f_calc()
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
  
  for tt in ti:
    
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
      if sig_fcalc is None:
        sig_fcalc = xrs_sel.structure_factors(d_min=d_min).f_calc() * 0.0
      if sig_icalc is None:
        sig_icalc = abs(sig_fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
      print("WARNING: Worker ",work_rank," is idle")

    else:

      for i in range(num_elems):

        # overwrite crystal structure coords with trajectory coords

        tmp = flex.vec3_double(tsites[i,:,:])
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
            xrs.set_sites_frac(flex.vec3_double(sites_frac))
    #        print ("Time to optimize = ",otime2-otime1)

    # select the atoms for the structure factor calculation

        xrs_sel = xrs.select(selection)
        if engine == "sfall":
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
        else:
          xrs_sel.scattering_type_registry(table=scattering_table)
          fcalc = xrs_sel.structure_factors(d_min=d_min).f_calc()

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
    #      this_map = fcalc.fft_map(d_min=d_min, resolution_factor = 0.5)
    #      real_map_np = this_map.real_map_unpadded().as_numpy_array()
    #      map_data.append(real_map_np)
        else:
          if sig_fcalc is None:
            sig_fcalc = fcalc
            sig_icalc = abs(fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
          else:
            sig_fcalc = sig_fcalc + fcalc
            sig_icalc = sig_icalc + abs(fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
        ct = ct + 1

    chunk_ct = chunk_ct + 1

    print("Worker ",work_rank," processed chunk ",chunk_ct," of ",nchunklist[work_rank]," with ",chunklist[work_rank]," frames in ",time.time()-mtime," seconds")

    if (chunk_ct >= nchunklist[work_rank]):
      break


    
# Commented out some density trajectory code
#  if not (dens_file is None):
#    map_grid = np.concatenate(map_data)
#    Ni = map_data[0].shape[0]
#    Nj = map_data[0].shape[1]
#    Nk = map_data[0].shape[2]
#    map_grid_3D = np.reshape(map_grid,(len(tsites),Ni,Nj,Nk))
#    np.save(dens_file,map_grid_3D)                           

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
#      dmap = avg_fcalc.fft_map(d_min=d_min,resolution_factor=0.5,symmetry_flags=symmetry_flags)
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

    diffuse_array_common = mpi_comm.bcast(diffuse_array_common,root=0)

    #Initialize correlations array
    w = np.ones(ct)
    ct_nonzero = ct
    first_this = (skiplist[work_rank]-first)
    last_this = first_this + chunklist[work_rank]*nchunklist[work_rank]-1
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
      mpi_comm.Allreduce(C_all_this,C_all,op=MPI.SUM)
      if np.max(C_all) > C_ref:
        C_ref = np.max(C_all)
        maxind = np.argmax(C_all)
#        print("DEBUG: ",work_rank,maxind,first_this,last_this)
        if maxind >= first_this and maxind <= last_this:
          which_rank = mpi_rank
        else:
          which_rank = 0
        which_rank = mpi_comm.allreduce(which_rank,MPI.SUM)
#        print("which_rank = ",which_rank)
        if which_rank == mpi_rank:          
          tot_sig_fcalc_np = tot_sig_fcalc_np - fcalc_list[maxind - first_this]
          tot_sig_icalc_np = tot_sig_icalc_np - icalc_list[maxind - first_this]
        else:
          tot_sig_fcalc_np = None
          tot_sig_icalc_np = None
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
