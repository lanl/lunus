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

from iotbx.pdb import hierarchy
from cctbx.array_family import flex
import mmtbx.utils
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
import future

def mpi_enabled():
  return 'OMPI_COMM_WORLD_SIZE' in os.environ.keys()

def calc_msd(x):
  d = np.zeros(this_sites_frac.shape)
  msd = 0
  for k in range(3):
    d[:,k] = (this_sites_frac[:,k] - ref_sites_frac[:,k] + x[k] + 0.5)%1.0 - 0.5
    msd += np.sum(d[:,k] * d[:,k])
    return msd

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
    d_min = 1.6
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

# Set nsteps if needed
  
  if (nsteps == 0):
    nsteps = last - first + 1
  elif (last != 0):
    print("Please specify nsteps or last, but not both.")
    raise ValueError()

  last = first + nsteps - 1


# Initialize MPI

  if mpi_enabled():
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
  else:
    mpi_comm = None
    mpi_rank = 0
    mpi_size = 1

# read .pdb file. It's used as a template, so don't sort it.

  if mpi_rank == 0:
    pdb_in = hierarchy.input(file_name=top_file,sort_atoms=False)

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
  else:
    pdb_in = None
    xrs = None
    
  if mpi_enabled():
    pdb_in = mpi_comm.bcast(pdb_in,root=0)
    xrs = mpi_comm.bcast(xrs,root=0)
    
  selection_cache = pdb_in.hierarchy.atom_selection_cache()
  selection = selection_cache.selection(selection_text)
  xrs.convert_to_isotropic()
  xrs.set_b_iso(0.0)
  xrs.set_occupancies(1.0)
  xrs_sel = xrs.select(selection)
  fcalc = xrs_sel.structure_factors(d_min=d_min).f_calc()
  f_000 = mmtbx.utils.f_000(xray_structure=xrs_sel)
  volume = xrs_sel.unit_cell().volume()
  print("f_000 = %g, volume = %g" % (f_000.f_000,volume))
  if (mpi_rank == 0):
    fcalc.as_mtz_dataset('FWT').mtz_object().write(file_name="reference.mtz")

# read the MD trajectory and extract coordinates

  if (chunksize is None):
    nchunks = mpi_size
    chunksize = int(nsteps/nchunks)
  else:
    nchunks = int(nsteps/chunksize)
  chunklist = np.zeros((mpi_size), dtype=np.int)
  nchunklist = np.zeros((mpi_size),dtype=np.int)
  skiplist = np.zeros((mpi_size), dtype=np.int)
  nchunksize = nchunks/mpi_size
  leftover = nchunks % mpi_size
  for i in range(mpi_size):
    chunklist[i] = chunksize
    nchunklist[i] = nchunksize
    if (i < leftover):
      nchunklist[i] += 1
    if (i == 0):
      skiplist[i] = first
    else:
      skiplist[i] = skiplist[i-1] + chunklist[i-1] * nchunklist[i-1]

  if (mpi_rank == 0):               
    stime = time.time()

  ct = 0
  sig_fcalc = None
  sig_icalc = None

  if (skiplist[mpi_rank] <= last):
    skip_calc = False
  else:
    skip_calc = True

  ti = md.iterload(traj_file,chunk=chunklist[mpi_rank],top=top_file,skip=skiplist[mpi_rank])

# Each MPI rank works with its own trajectory chunk t

  chunk_ct = 0

  itime = time.time()
  
  for tt in ti:         
    t = tt
#    print "rank =",mpi_rank," skip = ",skiplist[mpi_rank]," chunk = ",chunklist[mpi_rank]," start time = ",t.time[0]," coords of first atom = ",t.xyz[0][0]

#    if mpi_enabled():
#      mpi_comm.Barrier()                                                                          
    if (mpi_rank == 0): 
      mtime = time.time()                                                        
      print("TIMING: md.iterload = ",mtime-stime)

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
      print("WARNING: Rank ",mpi_rank," is idle")

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

        fcalc = xrs_sel.structure_factors(d_min=d_min).f_calc()

    # Commented out some density trajectory code
    #    if not (dens_file is None):
    #      this_map = fcalc.fft_map(d_min=d_min, resolution_factor = 0.5)
    #      real_map_np = this_map.real_map_unpadded().as_numpy_array()
    #      map_data.append(real_map_np)

        if sig_fcalc is None:
          sig_fcalc = fcalc
          sig_icalc = abs(fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
        else:
          sig_fcalc = sig_fcalc + fcalc
          sig_icalc = sig_icalc + abs(fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
        ct = ct + 1

    chunk_ct = chunk_ct + 1

    print("Rank ",mpi_rank," processed chunk ",chunk_ct," of ",nchunklist[mpi_rank])

    if (chunk_ct >= nchunklist[mpi_rank]):
      break


    
# Commented out some density trajectory code
#  if not (dens_file is None):
#    map_grid = np.concatenate(map_data)
#    Ni = map_data[0].shape[0]
#    Nj = map_data[0].shape[1]
#    Nk = map_data[0].shape[2]
#    map_grid_3D = np.reshape(map_grid,(len(tsites),Ni,Nj,Nk))
#    np.save(dens_file,map_grid_3D)                           

  print("Rank ",mpi_rank," is done with individual calculations")
  sys.stdout.flush()

  if mpi_enabled():
    mpi_comm.Barrier()

  if (mpi_rank == 0):
    mtime = time.time()
    print("TIMING: Calculate individual statistics = ",mtime-itime)

# perform reduction of sig_fcalc, sig_icalc, and ct

  sig_fcalc_np = sig_fcalc.data().as_numpy_array()
  sig_icalc_np = sig_icalc.data().as_numpy_array()

  if mpi_rank == 0:
    tot_sig_fcalc_np = np.zeros_like(sig_fcalc_np)
    tot_sig_icalc_np = np.zeros_like(sig_icalc_np)
  else:
    tot_sig_fcalc_np = None
    tot_sig_icalc_np = None

  if mpi_enabled():
    mpi_comm.Barrier()                                                        

  if mpi_enabled():
    mpi_comm.Reduce(sig_fcalc_np,tot_sig_fcalc_np,op=MPI.SUM,root=0)
    mpi_comm.Reduce(sig_icalc_np,tot_sig_icalc_np,op=MPI.SUM,root=0)
    ct = mpi_comm.reduce(ct,op=MPI.SUM,root=0)
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
    sq_avg_fcalc = abs(avg_fcalc).set_observation_type_xray_amplitude().f_as_f_sq()
    avg_icalc = sig_icalc / float(ct)
    diffuse_array=avg_icalc*1.0
    sq_avg_fcalc_data = sq_avg_fcalc.data()
    diffuse_data = diffuse_array.data()
#    print ("diffuse_data[0] = ",diffuse_data[0])
    for x in range(0,diffuse_data.size()):
      diffuse_data[x]=diffuse_data[x]-sq_avg_fcalc_data[x]
    etime = time.time()
    print("TIMING: Reduction = ",etime-mtime)
    print("TIMING: Total diffuse calculation = ",etime-stime)

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
    f=open(diffuse_file,'w')
    for hkl,intensity in diffuse_array:
      print("%4d %4d %4d   %10.2f" %(hkl+tuple((intensity,))),file=f)
    f.close()
