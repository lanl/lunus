# Build dynamical matrix for ANM and compute covariance matrix 
#    according to Riccardi et al, Biophys J 2010
# Michael Wall, 1/7/2016

import sys
from iotbx.pdb import hierarchy
import iotbx.cif
from cctbx.array_family import flex
from cctbx.eltbx import xray_scattering
import numpy as np
from multiprocessing import Pool
import copy

def get_input_dict(args):
  # return dictionary of keywords and their values
  input_dict={}
  for arg in args:
    spl=arg.split("=")
    if len(spl)==2:
      input_dict[spl[0]]=spl[1]
  return input_dict

if __name__=="__main__":
    
    args = sys.argv[1:] 
    input_dict = get_input_dict(args)
    for a in input_dict:
        if (a == "input.pdb"):
          pdbinname = input_dict[a]
        if (a == "output.pdb"):
          pdboutname = input_dict[a]
        if (a == "input.V"):
          Vinname = input_dict[a]
        if (a == "output.V"):
          Voutname = input_dict[a]

# Read structure file
    pdb_in = hierarchy.input(file_name=pdbinname)
# Wher there are alternate conformations, select the A conformation and delete the rest
    for model in pdb_in.hierarchy.models():
      for chain in model.chains():
        for rg in chain.residue_groups():
          if (len(rg.atom_groups())>1):
            for ag in rg.atom_groups():
              if (ag.altloc != "A" and ag.altloc != ""):
                print('  removing alternative conformation "%s" for residue "%s"' % (ag.altloc,ag.resname))
                for atom in ag.atoms():
                  ag.remove_atom(atom=atom)
    ftmp=open(pdbinname+"_tmp.pdb","w")
    ftmp.write(pdb_in.hierarchy.as_pdb_string(crystal_symmetry=pdb_in.input.crystal_symmetry()))
    ftmp.close()
    pdb_in = hierarchy.input(file_name=pdbinname+"_tmp.pdb")
    struct = pdb_in.input.xray_structure_simple().expand_to_p1(append_number_to_labels=True)
# Get scatterers
    scat = struct.scatterers()
# Reorder the scatterers and place back in the structure
    struct.erase_scatterers()
    nc=4
    for i in range(nc):
      for j in range(int(len(scat)/nc)):
        struct.add_scatterer(scat[j*nc+i])
    fout=open(pdboutname,"w")
    fout.write(struct.as_pdb_file())
    fout.close
# Get scatterers in new order
    scat = struct.scatterers()
#    print scat[0].label,scat[1].label
# Select alpha carbons
    selca = struct.atom_names_selection(atom_names=['CA'])
# Select backbone nitrogens
#    global seln
    seln = struct.atom_names_selection(atom_names=['N'])
#    np.savetxt("diffuse.dat",D.reshape((nh*nk,nl)).real)
    Vca = np.loadtxt(Vinname)
    na = len(scat)
    nm = 3*na
    nca = 0
    for i in range(na):
      if selca[i]:
        nca = nca + 1
    if (3*nca != len(Vca)):
      raise ValueError("Number of modes = 3 * N(CA) must be equal to size of SVD")
    Vaa = np.zeros((nm,len(Vca)))
    thisV = np.zeros(nm)
    idxca = np.zeros(na)
    ica = -1
    for i in range(na):
      if seln[i]:
        ica = ica + 1
      idxca[i]=ica
    for i in range(len(Vca)):
      for j in range(na):
        jj = 3*j
        jidxca=3*idxca[j]
        for k in range(3):
          Vaa[jj+k][i]=Vca[jidxca+k][i]
      Vaa[:,i]=Vaa[:,i]/np.sqrt(np.dot(Vaa[:,i],Vaa[:,i]))
    np.savetxt(Voutname,Vaa)
        
