#!/usr/bin/env python
#
# LIBTBX_SET_DISPATCHER_NAME lunus.stills_process

from __future__ import absolute_import, division
from dials.command_line.stills_process import Script
from dials.command_line.stills_process import Processor as SP_Processor

from dials.command_line import stills_process
from libtbx.phil import parse
from copy import deepcopy
lunus_phil_str = '''
  lunus {
    d_min = 2.5
      .type = float
      .help = Limiting resolution of diffuse intensity map.
    input_deck = lunus_input_params.sh
      .type = string
      .help = Lunus input deck.
  }
'''
lunus_defaults = """
  mp {
    method = mpi
      .type = choice
  }
"""


stills_process.phil_scope = parse(stills_process.control_phil_str +
                                  stills_process.dials_phil_str +
                                  lunus_phil_str,
                                  process_includes=True).fetch(
                            parse(stills_process.program_defaults_phil_str)).fetch(
                            parse(lunus_defaults))

class Processor(SP_Processor):


  ncalls = 0;
  ref_data = None
#  ref_img = DiffuseImage("placeholder.file")

  def process_datablock(self, tag, datablock):

    if not self.params.output.composite_output:
      self.setup_filenames(tag)
    self.tag = tag

    if self.params.output.datablock_filename:
      from dxtbx.datablock import DataBlockDumper
      dump = DataBlockDumper(datablock)
      dump.as_json(self.params.output.datablock_filename)

    # Do the processing
    try:
      observed = self.find_spots(datablock)
    except Exception as e:
      print "Error spotfinding", tag, str(e)
      if not self.params.dispatch.squash_errors: raise
      return
    try:
      experiments, indexed = self.index(datablock, observed)
    except Exception as e:
      print "Couldn't index", tag, str(e)
      if not self.params.dispatch.squash_errors: raise
      return
    try:
      experiments, indexed = self.refine(experiments, indexed)
    except Exception as e:
      print "Error refining", tag, str(e)
      if not self.params.dispatch.squash_errors: raise
      return

    from diffuse_scattering.sematura import DiffuseExperiment, DiffuseImage
    def from_experiments(self,experiments):
        exp_xtal = experiments.crystals()[0]

        ### define useful attributes
        self.crystal = exp_xtal
        uc = self.crystal.get_unit_cell()
        uc_nospace = str(uc).replace(" ", "")
        uc_nospace_noparen = uc_nospace[1:-1]
        self.unit_cell = uc_nospace_noparen
        self.space_group = self.crystal.get_space_group()
        self.laue_group = self.space_group.laue_group_type()
        # self.a_matrix = crystal.get_A()
        self.experiments = experiments
    DiffuseExperiment.from_experiments = from_experiments

    if (self.params.mp.method == 'mpi'):
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank() # each process in MPI has a unique id, 0-indexed
    else:
      rank = 0
    test_exp = DiffuseExperiment()
    test_exp.from_experiments(experiments)

    img_set = test_exp.experiments.imagesets()
    imgs = img_set[0]
    file_list = imgs.paths()
    img_file = file_list[0]
    test_img = DiffuseImage(img_file)
    test_img.set_general_variables()
    test_img.remove_bragg_peaks(radial=True)
    if self.ncalls == 0: 
      if rank == 0:
        self.ref_data = deepcopy(test_img.lunus_data_scitbx)
#        print "ref_data is",self.ref_data
#      else:
#        self.ref_data = None
      if (self.params.mp.method == 'mpi'):
#      from mpi4py import MPI
#      comm = MPI.COMM_WORLD
#        print "Barrier, rank = ",rank
#        print "Broadcast, rank = ",rank
        self.ref_data = comm.bcast(self.ref_data,root=0)
        comm.barrier()
#        print "Broadcast done, rank = ",rank

    if self.ref_data == None:
      print "ref_data = None for Rank = ",rank
    test_img.scale_factor_from_images(self.ref_data)
#    test_img.scale_factor()
    test_img.crystal_geometry(test_exp.crystal)

    test_img.setup_diffuse_lattice(self.params.lunus.d_min)
    test_img.integrate_diffuse()
    self.ncalls = self.ncalls + 1
#    print "RANK, NCALLS = ",rank, self.ncalls

stills_process.Processor = Processor


if __name__ == '__main__':
  from dials.util import halraiser
  try:
    script = Script()
    script.run()
  except Exception as e:
    halraiser(e)
