from time import clock, time
import numpy as np
import glob, subprocess, shlex

if __name__=="__main__":
  import sys

  args = sys.argv[1:] 
  usage = ["metrology=<experiments.json file, for metrology info>",
           "image_glob=<glob for images, used for rotation series>",
           "index_glob=<glob for individual .json files from stills",
           "xvectors_dir=<directory for xvectors>",
           "amatrix_dir_prefix=<directory prefix for amatrix info"]

 # Read command line arguments

  output_format="binary"

 # Input json
  try:
    metroidx = [a.find("metrology")==0 for a in args].index(True)
  except ValueError:
    raise ValueError,"Metrology .json file must be specified using metrology=."
  else:
    metro = args.pop(metroidx).split("=")[1]
 # Image input glob
  try:
    imageglobidx = [a.find("image_glob")==0 for a in args].index(True)
  except ValueError:
    rotation_series = False
  else:
    image_glob = args.pop(imageglobidx).split("=")[1]
    rotation_series = True
 # Image input glob
  try:
    indexglobidx = [a.find("index_glob")==0 for a in args].index(True)
  except ValueError:
    stills_process = False
  else:
    index_glob = args.pop(indexglobidx).split("=")[1]
    stills_process = True
  # Output xvectors directory
  try:
    xvectorsdiridx = [a.find("xvectors_dir")==0 for a in args].index(True)
  except ValueError:
    xvectors_dir = "tmpdir_common"
  else:
    xvectors_dir = args.pop(xvectorsdiridx).split("=")[1]
  # Output json files directory
  try:
    jsondiridx = [a.find("json_dir")==0 for a in args].index(True)
  except ValueError:
    json_dir = "tmpdir_common"
  else:
    json_dir = args.pop(jsondiridx).split("=")[1]
    output_format="json"
  # Output amatrix directory
  try:
    amatrixdiridx = [a.find("amatrix_dir")==0 for a in args].index(True)
  except ValueError:
    amatrix_dir_prefix = "tmpdir_"
  else:
    amatrix_dir_prefix = args.pop(amatrixdiridx).split("=")[1]

  if (not (rotation_series or stills_process)):
    raise ValueError,"Must specify image_glob (for rotation series) or index_glob (for stills)."

  import copy, os

  import dxtbx
  from dxtbx.model.experiment_list import ExperimentListFactory
  from dials.array_family import flex
   
  experiments = ExperimentListFactory.from_json_file(metro, check_format=False)
  beam = experiments[0].beam
  detector = experiments[0].detector

  print "s0 from experiment[0]: ",beam.get_s0()
  print "wavelength from experiment[0]: ",beam.get_wavelength()
 
  lab_coordinates = flex.vec3_double()
  for panel in detector: 
    pixels = flex.vec2_double(panel.get_image_size())
    mms = panel.pixel_to_millimeter(pixels)
    lab_coordinates.extend(panel.get_lab_coord(mms))

    # generate s1 vectors
  s1 = lab_coordinates.each_normalize() * (1/beam.get_wavelength())
    # Generate x vectors
  x = np.asarray(s1 - beam.get_s0())
    
  DATAsize = np.asarray(detector[0].get_image_size())

#  np.save(xvectors,x)
#  np.save(datasize,DATAsize)

  if (not os.path.isdir(xvectors_dir)):
    command = 'mkdir {0}'.format(xvectors_dir)
    call_params = shlex.split(command)
    subprocess.call(call_params)

  np.save(xvectors_dir+"/x_vectors.npy",x)

#  np.save(xvectors_dir+"/DATAsize.npy",DATAsize)

  x.astype('float32').tofile(xvectors_dir+"/xvectors.bin")

  if rotation_series:
    from dxtbx.imageset import ImageSetFactory
    from dxtbx.model.experiment_list import Experiment, ExperimentList
    from dxtbx.serialize import dump
    imnum=1
    filelist=glob.glob(image_glob)
    filelist.sort()
    for imgname in filelist:
      img = dxtbx.load(imgname)
      detector = img.get_detector()
      beam = img.get_beam()
      scan = img.get_scan()
      gonio = img.get_goniometer()

      print "s0 from ",imgname,": ",beam.get_s0()
      print "wavelength from ",imgname,": ",beam.get_wavelength()

      crystal = copy.deepcopy(experiments.crystals()[0])
      axis = gonio.get_rotation_axis()
      start_angle, delta_angle = scan.get_oscillation()
      crystal.rotate_around_origin(axis, start_angle + (delta_angle/2), deg=True)
      if (output_format == "json"):
        exp_list = ExperimentList()
        exp_list.append(Experiment(imageset=ImageSetFactory.make_imageset(list([imgname])),
                                   beam=beam,
                                   detector=detector,
                                   goniometer=gonio,
                                   scan=scan,
                                   crystal=crystal))
        dump.experiment_list(exp_list,json_dir+"/experiments_for_lunus_{0:05d}.json".format(imnum))
      else:
        from scitbx import matrix
        A_matrix = matrix.sqr(crystal.get_A()).inverse()
        At = np.asarray(A_matrix.transpose()).reshape((3,3))
        print At

        workdir=amatrix_dir_prefix+"{0}".format(imnum)
        if (not os.path.isdir(workdir)):
          command = 'mkdir {}'.format(workdir)
          call_params = shlex.split(command)
          subprocess.call(call_params)
#        np.save(workdir+"/At.npy",At)
        At.astype('float32').tofile(workdir+"/At.bin")
      imnum = imnum +1

  if stills_process:
    imnum=1
    filelist=glob.glob(index_glob)
    filelist.sort()
    for jsonname in filelist:
      experiments = ExperimentListFactory.from_json_file(jsonname, check_format=False)
      detector = experiments[0].detector
      beam = experiments[0].beam
      crystal = experiments.crystals()[0]

      from scitbx import matrix
      A_matrix = matrix.sqr(crystal.get_A()).inverse()
      At = np.asarray(A_matrix.transpose()).reshape((3,3))

      if (output_format != "json"):
        workdir=amatrix_dir_prefix+"{0}".format(imnum)
        if (not os.path.isdir(workdir)):
          command = 'mkdir {}'.format(workdir)
          call_params = shlex.split(command)
          subprocess.call(call_params)
#        np.save(workdir+"/At.npy",At)
        At.astype('float32').tofile(workdir+"/At.bin")
      imnum = imnum +1
