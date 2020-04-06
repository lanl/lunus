#!/usr/bin/env python
#
# LIBTBX_SET_DISPATCHER_NAME lunus.process

from __future__ import print_function
import time
from time import time
import numpy as np
import glob, subprocess, shlex
import lunus
import copy, os
from dials.array_family import flex
import future,six
import dxtbx
from dxtbx.imageset import ImageSetFactory
from dxtbx.model.experiment_list import Experiment, ExperimentList, ExperimentListFactory
from dxtbx.serialize import dump
from scitbx.matrix import col
from scitbx import matrix
import sys



def mpi_enabled():
  return 'OMPI_COMM_WORLD_SIZE' in os.environ.keys()
#  return False

def mpi_init():
  global mpi_comm
  global MPI
  from mpi4py import MPI as MPI
  mpi_comm = MPI.COMM_WORLD

def get_mpi_rank():
  return mpi_comm.Get_rank() if mpi_enabled() else 0


def get_mpi_size():
  return mpi_comm.Get_size() if mpi_enabled() else 1

def get_experiment_params(experiments):

  beam = experiments[0].beam
  detector = experiments[0].detector

  beam_direction = col(beam.get_sample_to_source_direction())
  wavelength = beam.get_wavelength()

  beam_params = "\nbeam_vec={0},{1},{2}\nwavelength={3}".format(-beam_direction[0],-beam_direction[1],-beam_direction[2],wavelength)

  experiment_params = []

  panel_ct = 0

  for panel in detector: 
    pixel_size_mm = panel.get_pixel_size()[0]
    beam_mm_x = panel.get_beam_centre(beam.get_s0())[0]
    beam_mm_y = panel.get_beam_centre(beam.get_s0())[1]
    distance_mm = panel.get_distance()
    fast_vec = col(panel.get_fast_axis())
    slow_vec = col(panel.get_slow_axis())
    origin_vec = col(panel.get_origin())
    normal_vec = col(panel.get_normal())
    
    fast_vec_params = "\nfast_vec={0},{1},{2}".format(fast_vec[0],fast_vec[1],fast_vec[2])
    slow_vec_params = "\nslow_vec={0},{1},{2}".format(slow_vec[0],slow_vec[1],slow_vec[2])
    origin_vec_params = "\norigin_vec={0},{1},{2}".format(origin_vec[0],origin_vec[1],origin_vec[2])
    normal_vec_params = "\nnormal_vec={0},{1},{2}".format(normal_vec[0],normal_vec[1],normal_vec[2])

    more_params = "\npixel_size_mm={0}\nbeam_mm_x={1}\nbeam_mm_y={2}\ndistance_mm={3}".format(pixel_size_mm,beam_mm_x,beam_mm_y,distance_mm)

    experiment_params.append(beam_params+fast_vec_params+slow_vec_params+origin_vec_params+normal_vec_params+more_params)

#    print "panel_ct = ",panel_ct,origin_vec_params

    panel_ct += 1

  return(experiment_params)

def get_experiment_xvectors(experiments):

  beam = experiments[0].beam
  detector = experiments[0].detector

  x = []

  for panel in detector: 
    lab_coordinates = flex.vec3_double()
    pixels = flex.vec2_double(panel.get_image_size())
    mms = panel.pixel_to_millimeter(pixels)
    lab_coordinates.extend(panel.get_lab_coord(mms))

    # generate s1 vectors
    s1 = lab_coordinates.each_normalize() * (1/beam.get_wavelength())
    # Generate x vectors
    x.append((s1 - beam.get_s0()).as_double())

  return(x)

def mpi_bcast(d):
  if mpi_enabled():
    db = mpi_comm.bcast(d,root=0)
  else:
    db = d
    
  return db

def mpi_barrier():
  if mpi_enabled():
    mpi_comm.Barrier()

def mpi_reduce_p(p):

  if mpi_enabled():
#    if get_mpi_rank() == 0:
#      print("LUNUS.PROCESS: Convertinf flex arrays to numpy arrays")
#      sys.stdout.flush()

    l = p.get_lattice().as_numpy_array()
    c = p.get_counts().as_numpy_array()

    if get_mpi_rank() == 0:
      lt = np.zeros_like(l)
      ct = np.zeros_like(c)
    else: 
      lt = None
      ct = None


    mpi_comm.Reduce(l,lt,op=MPI.SUM,root=0)
    mpi_comm.Reduce(c,ct,op=MPI.SUM,root=0)
    
    if get_mpi_rank() == 0:
#      print("LUNUS.PROCESS: Converting numpy arrays to flex arrays")
      p.set_lattice(flex.double(lt))
      p.set_counts(flex.int(ct))

  return p

def process_one_glob():

    global fresh_lattice

    imnum=0

    tte = 0.0
    ttr = 0.0

    metrolist = glob.glob(metro_glob)
    metrolist.sort()

    filelist=glob.glob(image_glob)
    filelist.sort()

    if get_mpi_rank() == 0:
      print("Analyzing {0} images".format(len(filelist)))

    if (subtract_background_images==True):
      bkglist=glob.glob(bkg_glob)
      bkglist.sort()
      if get_mpi_rank() == 0:
        if (len(filelist) != len(bkglist) and len(bkglist) != 1):
          raise(ValueError,"Must either be one background or as many as there are images.")
      if (len(bkglist) == 1):
        if get_mpi_rank() == 0:
          bkg = dxtbx.load(bkglist[0])
        else:
          bkg = None
        bkg = mpi_bcast(bkg)
        bkg_data = bkg.get_raw_data()

    if (rotation_series):
      if get_mpi_rank() == 0:
        experiments = ExperimentListFactory.from_json_file(metrolist[0], check_format=False)
        experiment_params = get_experiment_params(experiments)
        x = get_experiment_xvectors(experiments)
        crystal_reference = copy.deepcopy(experiments.crystals()[0])
      else:
        experiments = None
        experiment_params = None
        x = None
        crystal_reference = None
      if get_mpi_rank() == 0:
        print("Broadcasting rotation series common data")
        sys.stdout.flush()
      tbc = time()
      experiments = mpi_bcast(experiments)
      experiment_params = mpi_bcast(experiment_params)
      x = mpi_bcast(x)
      crystal_reference = mpi_bcast(crystal_reference)
      tbc = time() - tbc
      if get_mpi_rank() == 0:
        print("LUNUS.PROCESS: Took {0} seconds to broadcast rotation series common data".format(tbc))
        sys.stdout.flush()
    # prepend image 0 to each range.

    i_iter = list(range(get_mpi_rank(),len(filelist),get_mpi_size()))

#    print("Rank {0}, i_iter = {1}".format(get_mpi_rank(),i_iter))

    i_iter.insert(0,0)
    
    tmode = 0.0
    tscale = 0.0
    tmap = 0.0
    tmask = 0.0
    tcorrection = 0.0
    tsetup = 0.0
    
    for i in i_iter:
      if fresh_lattice:
        if get_mpi_rank() == 0:
          print("Reference image ")
          sys.stdout.flush()
      else:
        print("{0} ".format(i))
        sys.stdout.flush()

      if fresh_lattice:
        if i != 0:
          raise(ValueError,"Image number must be 0 first time through")

      if (not rotation_series):
        if (get_mpi_rank() == 0 or not fresh_lattice):
          experiments = ExperimentListFactory.from_json_file(metrolist[i], check_format=False)
          experiment_params = get_experiment_params(experiments)
          x = get_experiment_xvectors(experiments)
          crystal_reference = copy.deepcopy(experiments.crystals()[0])
        else:
          experiments = None
          experiment_params = None
          x = None
          crystal_reference = None
        if fresh_lattice:
#          experiments = mpi_bcast(experiments)
          experiment_params = mpi_bcast(experiment_params)
          x = mpi_bcast(x)
          crystal_reference = mpi_bcast(crystal_reference)

      imgname=filelist[i]

      if (debug_level>0):
        print("Rank {0} processing {1}".format(get_mpi_rank(),imgname))

      bt = time()

      if (get_mpi_rank() == 0 or not fresh_lattice):
        img = dxtbx.load(imgname)
        data = img.get_raw_data()
        scan = img.get_scan()
        gonio = img.get_goniometer()
      else:
        data = None
        scan = None
        gonio = None
      if fresh_lattice:
        if (debug_level>0):
          print("Broadcasting reference image info")
        data = mpi_bcast(data)
        scan = mpi_bcast(scan)
        gonio = mpi_bcast(gonio)

      et = time()
      ttr += et - bt

      if (debug_level>0):
        print("Took {0} secs to read {1}".format(et-bt,imgname))

#      print("min of data = ",flex.min(data))
#      print("max of data = ",flex.max(data))

      if (debug_level>0):
        print("Rank {0} entering p.set_image()".format(get_mpi_rank()))

      if isinstance(data,tuple):
        for pidx in range(len(data)):
          p.set_image(pidx,data[pidx])
      else:
        p.set_image(data)

      if (debug_level>0):
        print("Rank {0} entering p.LunusSetparamsim()".format(get_mpi_rank()))

      for pidx in range(len(experiment_params)):
        deck_and_extras = deck+experiment_params[pidx]
        p.LunusSetparamsim(pidx,deck_and_extras)

      if (subtract_background_images==True):
        if (len(bkglist) != 1):
          if (get_mpi_rank() == 0 or not fresh_lattice):
            bkg = dxtbx.load(bkglist[i])
          else:
            bkg = None
          if fresh_lattice:
            bkg = mpi_bcast(bkg)
          bkg_data = bkg.get_raw_data()
        if isinstance(bkg_data,tuple):
          for pidx in range(len(bkg_data)):
            p.set_background(pidx,bkg_data[pidx])
        else:
#          print "setting background"
#          print "max(bkg_data) = ",np.amax(bkg_data.as_numpy_array())
          p.set_background(bkg_data)

        p.LunusBkgsubim()

      crystal = copy.deepcopy(crystal_reference)

      if (rotation_series):
        if (debug_level>0):
          print("Rank {0} applying rotation".format(get_mpi_rank()))
        start_angle, delta_angle = scan.get_oscillation()      
        axis = gonio.get_rotation_axis()
        crystal.rotate_around_origin(axis, start_angle + (delta_angle/2), deg=True)

      A_matrix = matrix.sqr(crystal.get_A()).inverse()
      At = np.asarray(A_matrix.transpose()).reshape((3,3))
      At_flex = A_matrix.transpose().as_flex_double_matrix()

      p.set_amatrix(At_flex)

      if (fresh_lattice==True):
        for pidx in range(len(x)):
          p.set_xvectors(pidx,x[pidx])
        p.set_image_ref()

        if get_mpi_rank() == 0:
          p.print_image_params()
      
        p.LunusProcimlt(0)

        fresh_lattice = False
      else:
#        print("LUNUS.PROCESS: Rank {0} STARTING processing image number {1}".format(get_mpi_rank(),imnum))
#        sys.stdout.flush()
        bt = time()
        p.LunusProcimlt(1)
        et = time()
        te = et - bt
        print("LUNUS.PROCESS: Rank {0} FINISHED processing image number {1}".format(get_mpi_rank(),i))
        
        tte += te

        timers = p.get_lattice_timers()

        tmode += timers[0]
        tscale += timers[1]
        tmap += timers[2]
        tmask += timers[3]
        tcorrection += timers[4]
        tsetup += timers[5]

        imnum = imnum +1

    print()

    mpi_barrier()

    if (get_mpi_rank() == 0):
      print("LUNUS.PROCESS: Reads took {0} seconds".format(ttr))
      print("LUNUS.PROCESS: Individual processing took {0} seconds".format(tte))
      print("LUNUS.PROCESS: Setup took {0} seconds".format(tsetup))
      print("LUNUS.PROCESS: Masking and thresholding took {0} seconds".format(tmask))
      print("LUNUS.PROCESS: Solid angle and polarization correction took {0} seconds".format(tcorrection))
      print("LUNUS.PROCESS: Mode filtering took {0} seconds".format(tmode))
      print("LUNUS.PROCESS: Mapping took {0} seconds".format(tmap))

if __name__=="__main__":

  args = sys.argv[1:] 
  usage = ["experiments=<experiments.json file, for metrology info>",
           "images=<glob for images, used for rotation series>"]

 # Output .vtk file
  try:
    idx = [a.find("vtk")==0 for a in args].index(True)
  except ValueError:
    vtk_file = "diffuse.vtk"
  else:
    vtk_file = args.pop(idx).split("=")[1]

 # Output .hkl file
  try:
    idx = [a.find("hkl")==0 for a in args].index(True)
  except ValueError:
    hkl_file = "diffuse.hkl"
  else:
    hkl_file = args.pop(idx).split("=")[1]

 # Output .cube file
  try:
    idx = [a.find("cube")==0 for a in args].index(True)
  except ValueError:
    cube_file = None
  else:
    cube_file = args.pop(idx).split("=")[1]

 # Output .lat file
  try:
    idx = [a.find("lat")==0 for a in args].index(True)
  except ValueError:
    lat_file = "diffuse.lat"
  else:
    lat_file = args.pop(idx).split("=")[1]

 # Input lunus input deck
  try:
    idx = [a.find("params")==0 for a in args].index(True)
  except ValueError:
    raise(ValueError,"Processing parameters must be specified using params=")
  else:
    deck_file = args.pop(idx).split("=")[1]

 # rotation series mode
  try:
    idx = [a.find("rotation_series")==0 for a in args].index(True)
  except ValueError:
    rotation_series=True
  else:
    rotation_series_str = args.pop(idx).split("=")[1]
    if (rotation_series_str == "False"):
      rotation_series=False

 # debug level
  try:
    idx = [a.find("debug_level")==0 for a in args].index(True)
  except ValueError:
    debug_level=0
  else:
    debug_level = int(args.pop(idx).split("=")[1])

 # Input json
  keep_going = True
  metro_glob_list = []
  while keep_going:
    try:
      metroidx = [(a.find("metrology")==0 or (a.find("experiments")==0)) for a in args].index(True)
    except ValueError:
      keep_going = False
    else:
      metro_glob_list.append(args.pop(metroidx).split("=")[1])
  if (len(metro_glob_list) == 0):
    raise(ValueError,"Experiments .json file must be specified using experiments=")
  
 # Image input glob
  keep_going = True
  image_glob_list = []
  while (keep_going):
    try:
      imageglobidx = [(a.find("image_glob")==0 or (a.find("images")==0)) for a in args].index(True)
    except ValueError:
      keep_going = False
    else:
      image_glob_list.append(args.pop(imageglobidx).split("=")[1])

 # Background image input glob
  keep_going = True
  bkg_glob_list = []
  subtract_background_images = False
  while (keep_going):
    try:
      bkgglobidx = [(a.find("bkg_glob")==0 or (a.find("backgrounds")==0)) for a in args].index(True)
    except ValueError:
      keep_going = False
    else:
      bkg_glob_list.append(args.pop(bkgglobidx).split("=")[1])
      subtract_background_images = True

  if (subtract_background_images):
    bkg_glob = bkg_glob_list[0]

  if mpi_enabled():
    mpi_init()
    if (debug_level>0):
      print("Rank {0} reporting for duty".format(get_mpi_rank()))
      sys.stdout.flush()
  if get_mpi_rank() == 0:
    if (len(metro_glob_list) != len(image_glob_list) or (subtract_background_images and len(metro_glob_list) != len(bkg_glob_list))):
      raise(ValueError,"Must specify same number of experiments, images, and backgrounds")


# Get a sample experiments file and use it to initialize the processor class

  metro_glob = metro_glob_list[0]

  metrolist = glob.glob(metro_glob)
#  print("type(metrolist) = ",type(metrolist))
#  print(metrolist)
  metrolist.sort()

  metro = metrolist[0]

  if get_mpi_rank() == 0:
    experiments = ExperimentListFactory.from_json_file(metro, check_format=False)
    experiment_params = get_experiment_params(experiments)
  else:
    experiments = None
    experiment_params = None

  experiment_params = mpi_bcast(experiment_params)

#  print "Rank = ",get_mpi_rank()," type(experiment_params) = ",type(experiment_params)

  p = lunus.Process(len(experiment_params))

# Get the input deck and initialize the lattice

  if get_mpi_rank() == 0:
    with open(deck_file) as f:
      deck = f.read()
  else:
    deck = None

  deck = mpi_bcast(deck)

  deck_and_extras = deck+experiment_params[0]

#  print deck_and_extras

  p.LunusSetparamslt(deck_and_extras)

  fresh_lattice = True

  for i in range(len(metro_glob_list)):
    if get_mpi_rank() == 0:
      print("Image set ",i+1,":",end=" ")
      sys.stdout.flush()

    metro_glob = metro_glob_list[i]
    image_glob = image_glob_list[i]
    if (subtract_background_images):
      bkg_glob = bkg_glob_list[i]

#      p.LunusSetmetim()

# Temporary, set default metrology based on beam center and distance

    process_one_glob()

  if get_mpi_rank() == 0:
    print("LUNUS.PROCESS: Done processing individual globs")
    sys.stdout.flush()

  bt = time()
  p = mpi_reduce_p(p)
  et = time()
  tred = et - bt

  if get_mpi_rank() == 0:
    print("Time spent in reduction (sec): ",tred)
    p.divide_by_counts()

    if (not vtk_file is None):
      p.write_as_vtk(vtk_file)
    if (not hkl_file is None):
      p.write_as_hkl(hkl_file)
    if (not cube_file is None):
      p.write_as_vtk(cube_file)
    if (not lat_file is None):
      p.write_as_lat(lat_file)
