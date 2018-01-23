#!/usr/bin/env python
#
# LIBTBX_SET_DISPATCHER_NAME lunus.stills_process

from __future__ import absolute_import, division
from dials.command_line.stills_process import Script
from dials.command_line.stills_process import Processor as SP_Processor

from dials.command_line import stills_process
from libtbx.phil import parse
from diffuse_scattering.sematura import DiffuseExperiment, DiffuseImage
from copy import deepcopy
import numpy as np

sematura_phil_str = '''
  lunus {
    d_min = 1.4
      .type = float
      .help = Limiting resolution of diffuse intensity map.
  }
'''
sematura_defaults = """
  mp {
    method = mpi
      .type = choice
  }
"""


stills_process.phil_scope = parse(stills_process.control_phil_str +
                                  stills_process.dials_phil_str +
                                  sematura_phil_str,
                                  process_includes=True).fetch(
                            parse(stills_process.program_defaults_phil_str)).fetch(
                            parse(sematura_defaults))

class Processor(SP_Processor):


  ncalls = 0;
  ref_data = None
  lt = None
  ct = None
  mean_lattice = None
  ref_img = None

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
    test_img.crystal_geometry(test_exp.crystal)
    phil_cell=self.params.indexing.known_symmetry.unit_cell.parameters()
# Set the unit cell to the .phil input, to make all the lattices identical
    test_img.cella = phil_cell[0]
    test_img.cellb = phil_cell[1]
    test_img.cellc = phil_cell[2]
    test_img.alpha = phil_cell[3]
    test_img.beta = phil_cell[4]
    test_img.gamma = phil_cell[5]
    test_img.setup_diffuse_lattice(self.params.lunus.d_min)
    if self.ncalls == 0: 
      if rank == 0:
        self.ref_data = deepcopy(test_img.lunus_data_scitbx)
        self.ref_img = deepcopy(test_img)
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
    if (self.ncalls == 0):
      self.lt = np.zeros(test_img.latsize, dtype=np.float32)
      self.ct = np.zeros(test_img.latsize, dtype=np.float32)
    test_img.integrate_diffuse()
    self.lt = np.add(self.lt,test_img.lt)
    self.ct = np.add(self.ct,test_img.ct)
    self.ncalls = self.ncalls + 1
#    print "RANK, NCALLS = ",rank, self.ncalls

  def array_to_vtk(self):

        from os import getcwd
        mean_lattice_vtkname = (getcwd()+"/arrays/"+self.ref_img.id+"_mean.vtk")
        vtkfile = open(mean_lattice_vtkname,"w")

        array_values = self.mean_lattice.flatten()

        a_recip = 1./self.ref_img.cella
        b_recip = 1./self.ref_img.cellb
        c_recip = 1./self.ref_img.cellc

        print >>vtkfile,"# vtk DataFile Version 2.0"
        print >>vtkfile,"lattice_type=PR;unit_cell={0},{1},{2},{3},{4},{5};space_group={6};".format(self.ref_img.cella,self.ref_img.cellb,self.ref_img.cellc,self.ref_img.alpha,self.ref_img.beta,self.ref_img.gamma, self.ref_img.unit_cell)
        print >>vtkfile,"ASCII"
        print >>vtkfile,"DATASET STRUCTURED_POINTS"
        print >>vtkfile,"DIMENSIONS %d %d %d"%(self.ref_img.latxdim,self.ref_img.latydim,self.ref_img.latzdim)
        print >>vtkfile,"SPACING %f %f %f"%(a_recip,b_recip,c_recip)
        print >>vtkfile,"ORIGIN %f %f %f"%(-self.ref_img.i_0*a_recip,-self.ref_img.j_0*b_recip,-self.ref_img.k_0*c_recip)
        print >>vtkfile,"POINT_DATA %d"%(self.ref_img.latsize)
        print >>vtkfile,"SCALARS volume_scalars float 1"
        print >>vtkfile,"LOOKUP_TABLE default\n"

        index = 0
        for k in range(0,self.ref_img.latzdim):
            for j in range(0,self.ref_img.latydim):
                for i in range(0,self.ref_img.latxdim):
                    print >>vtkfile,array_values[index],
                    index += 1
                print >>vtkfile,""

        vtkfile.close()
        return

  def finalize(self):
    ''' Perform any final operations '''
    if self.params.output.composite_output:
      # Dump composite files to disk
      if len(self.all_indexed_experiments) > 0 and self.params.output.refined_experiments_filename:
        from dxtbx.model.experiment_list import ExperimentListDumper
        dump = ExperimentListDumper(self.all_indexed_experiments)
        dump.as_json(self.params.output.refined_experiments_filename)

      if len(self.all_indexed_reflections) > 0 and self.params.output.indexed_filename:
        self.save_reflections(self.all_indexed_reflections, self.params.output.indexed_filename)

      if len(self.all_integrated_experiments) > 0 and self.params.output.integrated_experiments_filename:
        from dxtbx.model.experiment_list import ExperimentListDumper
        dump = ExperimentListDumper(self.all_integrated_experiments)
        dump.as_json(self.params.output.integrated_experiments_filename)

      if len(self.all_integrated_reflections) > 0 and self.params.output.integrated_filename:
        self.save_reflections(self.all_integrated_reflections, self.params.output.integrated_filename)

      # Create a tar archive of the integration dictionary pickles
      if len(self.all_int_pickles) > 0 and self.params.output.integration_pickle:
        import tarfile, StringIO, time, cPickle as pickle
        tar_template_integration_pickle = self.params.output.integration_pickle.replace('%d', '%s')
        outfile = os.path.join(self.params.output.output_dir, tar_template_integration_pickle%('x',self.composite_tag)) + ".tar"
        tar = tarfile.TarFile(outfile,"w")
        for i, (fname, d) in enumerate(zip(self.all_int_pickle_filenames, self.all_int_pickles)):
          string = StringIO.StringIO(pickle.dumps(d, protocol=2))
          info = tarfile.TarInfo(name=fname)
          info.size=len(string.buf)
          info.mtime = time.time()
          tar.addfile(tarinfo=info, fileobj=string)
        tar.close()

    print "Entering MERGE step."

    if (self.params.mp.method == 'mpi'):
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank() # each process in MPI has a unique id, 0-indexed
    else:
      rank = 0
    if (self.params.mp.method == 'mpi'):
      sumlt = np.zeros(self.lt.size,dtype=np.float32)
      sumct = np.zeros(self.ct.size,dtype=np.float32)
      comm.Reduce(self.lt, sumlt, op=MPI.SUM, root=0)
      comm.Reduce(self.ct, sumct, op=MPI.SUM, root=0)
    else:
      sumlt = self.lt
      sumct = self.ct
    if (rank == 0):
      sum_lt_masked = np.ma.array(sumlt,mask=sumct==0)
      sum_ct_masked = np.ma.array(sumct,mask=sumct==0)
      mean_lt = np.ma.divide(sum_lt_masked,sum_ct_masked)
      mean_lt.set_fill_value(-32768)
      self.mean_lattice = mean_lt.filled()
      from os import getcwd
      print "mean_lattice id = ",self.ref_img.id
      mean_lattice_npzname = (getcwd()+"/arrays/"+self.ref_img.id+"_mean.npz")
      np.savez(mean_lattice_npzname, mean_lt=self.mean_lattice)
      self.array_to_vtk()      

stills_process.Processor = Processor


if __name__ == '__main__':
  from dials.util import halraiser
  try:
    script = Script()
    script.run()
  except Exception as e:
    halraiser(e)
