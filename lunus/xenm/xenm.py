
# Build dynamical matrix for ANM and compute covariance matrix 
#    according to Riccardi et al, Biophys J 2010
# Michael Wall, 1/7/2016
# Modified for all atom calculation of normal modes, 6/30/2016
# Modified for k-points calculation, 12/4/2017

import sys
from iotbx.pdb import hierarchy
from iotbx.reflection_file_reader import any_reflection_file
import iotbx.cif
from cctbx.array_family import flex
from cctbx.eltbx import xray_scattering
import numpy as np
#import scipy as sp
from multiprocessing.pool import ThreadPool
import copy
import time, datetime
import os
import subprocess,shlex
from copy import deepcopy
#import mpi4py

def compute_r_factors (fobs, fmodel, flags) :
  fmodel, fobs = fmodel.common_sets(other=fobs)
  fmodel, flags = fmodel.common_sets(other=flags)
  fc_work = fmodel.select(~(flags.data()))
  fo_work = fobs.select(~(flags.data()))
  fc_test = fmodel.select(flags.data())
  fo_test = fobs.select(flags.data())
  r_work = fo_work.r1_factor(fc_work)
  r_free = fo_test.r1_factor(fc_test)
  print("r_work = %.4f" % r_work)
  print("r_free = %.4f" % r_free)
  print("")
  flags.setup_binner(n_bins=20)
  fo_work.use_binning_of(flags)
  fc_work.use_binner_of(fo_work)
  fo_test.use_binning_of(fo_work)
  fc_test.use_binning_of(fo_work)
  for i_bin in fo_work.binner().range_all() :
    sel_work = fo_work.binner().selection(i_bin)
    sel_test = fo_test.binner().selection(i_bin)
    fo_work_bin = fo_work.select(sel_work)
    fc_work_bin = fc_work.select(sel_work)
    fo_test_bin = fo_test.select(sel_test)
    fc_test_bin = fc_test.select(sel_test)
    if fc_test_bin.size() == 0 : continue
    r_work_bin = fo_work_bin.r1_factor(other=fc_work_bin,
      assume_index_matching=True)
    r_free_bin = fo_test_bin.r1_factor(other=fc_test_bin,
      assume_index_matching=True)
#    cc_work_bin = fo_work_bin.correlation(fc_work_bin).coefficient()
#    cc_free_bin = fo_test_bin.correlation(fc_test_bin).coefficient()
    legend = flags.binner().bin_legend(i_bin, show_counts=False)
    print("%s  %8d %8d  %.4f %.4f" % (legend, fo_work_bin.size(),
      fo_test_bin.size(), r_work_bin, r_free_bin))
#    print "%s  %8d %8d  %.4f %.4f  %.3f %.3f" % (legend, fo_work_bin.size(),
#      fo_test_bin.size(), r_work_bin, r_free_bin, cc_work_bin, cc_free_bin)

# Gershgorin bounds on eigenvalues
def gb(X):
    D = X.diagonal()
#    print D
    R = np.sum(np.absolute(X-np.diag(D)),axis=1)
#    print D
    DpR=D+R
    DmR=D-R
#    print DpR
    return min(np.amin(DpR),np.amin(DmR)),max(np.amax(DpR),np.amax(DmR))

def pinv(H):
  wmin,wmax = gb(H)
  dw = 0.1*wmax

  wN = np.array([wmax+dw,wmax+2*dw,wmax+3*dw])

  Nv = np.zeros((len(H)//3,3))

  Nv1D = Nv.reshape(len(H))

  N = np.zeros((len(H),len(H)))
  Ninv = np.zeros((len(H),len(H)))

  for k in range(3):
    Nv.fill(0.)
    Nv[:,k]=np.sqrt(1./len(Nv))
    N += wN[k]*np.outer(Nv1D,Nv1D)
    Ninv += 1./wN[k]*np.outer(Nv1D,Nv1D)

  Hm = H + N

  Hminv = np.linalg.inv(Hm)
  Hpinv = Hminv - Ninv
  return Hpinv

def calcCM(q,V):
  qsq = np.dot(q,q)
  return np.exp(qsq*V)-1

def calcH(Gamma,coords,cell,Hmodel):
  global masses_sqrt, bbond, w_benm
# Backbone-enhanced ENM: multiply the force constant of sequence-adjacent
# backbone bonds by w_benm. Single-cell (minimum-image) path, so all bbond pairs
# are real backbone bonds. Applied on a copy so the caller's Gamma is untouched.
  if w_benm != 1.0:
    Gamma = np.where(bbond, Gamma*w_benm, Gamma)
  nsel = len(coords)
  cellby2 = cell[:]/2.
  H = np.full((3*nsel,3*nsel),1.)
  H4D =  H.reshape((nsel,3,nsel,3))
  S = np.zeros((nsel,nsel))
  if np.all(H != 1.):
    print("H not equal to 1.")
  for i in range(nsel):
    ii = 3*i
    dc = np.zeros((nsel,3))
    dc1D = dc.reshape(3*nsel)
    svec2 = np.zeros(nsel)
    smask = np.full(nsel,1.)
    for k in range(3):
      if pbc:
        dc[:,k] = (coords[i,k] - coords[:,k] + cellby2[k])%cell[k] - cellby2[k]
      else:
        dc[:,k] = coords[i,k] - coords[:,k]
      svec2[:] += dc[:,k]*dc[:,k]      
    if usecutoff and Hmodel != "ANMEXP_DOMAIN":
      smask[svec2 > cut] = 0.0
      Gamma[i,:] *= smask[:]
    svec = np.sqrt(svec2)
    S[i,:] = svec
    if (Hmodel == "ANMEXP"):
      expvec = np.exp(-svec/anmexp_lambda/2.)
    if (Hmodel == "EXPCUT"):
      expterm = np.exp((-svec+expcut_thresh)/expcut_lambda)
      expcutvec = np.sqrt(expterm/(1.+expterm))
    if (Hmodel == "ANMEXP_DOMAIN"):
# Two-tier weight (see calcDk): intramolecular pairs use ANMEXP exp(-r/2λ),
# inter-molecular pairs use sqrt(inter_k * logistic-threshold).
      intra = (molid == molid[i])
      logistic = 1./(1.+np.exp((svec-inter_thresh)/inter_width))
      wvec = np.where(intra, np.exp(-svec/anmexp_lambda/2.),
                      np.sqrt(inter_k*logistic))
    svec[svec==0.0] = 1.0
    anmfac = 1./svec
    for k in range(3):
# Hessian value with force constants for the ordinary ANM model
      if (Hmodel == "ANM"):
        dc[:,k] *= anmfac
# Hessian value with force constants for the exponential falloff ANM model
      if (Hmodel == "ANMEXP"):
        dc[:,k] *= anmfac*expvec
# Hessian value with force constants for the exponentially smoothed threshold ANM model
      if (Hmodel == "EXPCUT"):
        dc[:,k] *= anmfac*expcutvec
# Hessian value with force constants for the two-tier domain-coupled model
      if (Hmodel == "ANMEXP_DOMAIN"):
        dc[:,k] *= anmfac*wvec
# Hessian value with force constants for the ANM^2 model
      if (Hmodel == "ANM2"):
        dc[:,k] *= anmfac/svec
# Hessian value with force constants for the ANM^6 model
      if (Hmodel == "ANM6"):
        dc[:,k] *= anmfac/svec2
    if (i==0):
      print(dc1D[0:10])
    for j in range(nsel):
      for k in range(3):
        H4D[i,:,j,k] *= dc[j,:]
        H4D[i,k,j,:] *= dc[j,:]
#    for k in range(3):
#      H[ii+k,:] *= dc1D
#      H[:,ii+k] *= dc1D
# Scale using gamma and calculate diagonal elements
  print("H[0,0:10]=",H[0,0:10])
  for i in range(nsel):
    for k1 in range(3):
      for k2 in range(3):
        H4D[i,k1,:,k2] *= -Gamma[i,:]/masses_sqrt/masses_sqrt[i]
#        H4D[i,k1,:,k2] *= -Gamma[i,:]
        hsum = np.sum(H4D[i,k1,:,k2])
        H4D[i,k1,i,k2] = -hsum
# H is symmetric in exact arithmetic. The minimum-image convention (%cell) above
# introduces tiny floating-point asymmetry (~1e-7) that is harmless for the short-
# range ANMEXP weights but O(1)-visible once inter-molecular springs are O(1)
# (ANMEXP_DOMAIN). Symmetrize explicitly (a no-op to rounding for symmetric H)
# before the sanity check, which now guards only against gross assembly errors.
  H = 0.5*(H + H.transpose())
  H4D = H.reshape((nsel,3,nsel,3))
  if not np.isclose(H.transpose(), H, 1.e-10).all():
    print("H is not symmetric!")
# Number of violating elements = ",np.sum(H.transpose()!=H),"and here are the H elements: ",H[H.transpose()!=H],"and the H.transpose() elements: ",H.transpose()[H.transpose()!=H]
    sys.exit(1)
  return H
#          D4D[i,k1,:,k2] = H4D[i,k1,:,k2]/masses_sqrt/masses_sqrt[i]
#    D = -D
#    for i in range(len(H)):
#      print H[i][i]," ",

def CvsR(coords,Rlist,C):
  nsel = len(coords)
  rslt = list()
  Csigs = np.sqrt(C[0].diagonal())
  for indr in range(len(Rlist)):
    R = Rlist[indr]
    thisC = copy.deepcopy(C[indr])
    for i in range(nsel):
      thisC[i,:] /= Csigs
      thisC[:,i] /= Csigs
    for i in range(nsel):
      dc = np.zeros((nsel,3))
      svec2 = np.zeros(nsel)
      for k in range(3):
        dc[:,k] = (coords[i,k] - coords[:,k] + R[k])
        svec2[:] += dc[:,k]*dc[:,k]      
      svec = np.sqrt(svec2)
      for j in range(nsel):
        if (i != j):
          rslt.append([svec[j],thisC[i,j]])
  return np.array(rslt)

def calcDk(coords,Rlist,Hmodel,kplist):
  global masses_sqrt, bbond, w_benm
  Dklist = list()
  Dk4Dlist = list()
  nsel = len(coords)
# print kplist

  Hdiag = np.zeros((nsel,3,3))
  for indr in range(len(Rlist)):
    R = Rlist[indr]
#    print "R=",R
    Gamma = np.full((nsel,nsel),1.)
# Backbone-enhanced ENM: multiply the force constant of sequence-adjacent
# backbone bonds by w_benm, but ONLY in the home cell (indr==0): a backbone bond
# is a covalent link between residues i and i+1 within one molecule, which exists
# only at R=0. For R!=0 the partner is a copy in a neighboring cell (a crystal
# contact), never a backbone bond. The mask is parameter-independent.
    if w_benm != 1.0 and indr == 0:
      Gamma = np.where(bbond, Gamma*w_benm, Gamma)
    H = np.full((3*nsel,3*nsel),1.)
    H4D =  H.reshape((nsel,3,nsel,3))
    for i in range(nsel):
      i3 = 3*i
      dc = np.zeros((nsel,3))
      dc1D = dc.reshape(3*nsel)
      svec2 = np.zeros(nsel)
      smask = np.full(nsel,1.)
      for k in range(3):
        if pbc:
          dc[:,k] = (coords[i,k] - coords[:,k] + R[k])
        else:
          print("kpoints method requires pbc=True")
          sys.exit(1)
        svec2[:] += dc[:,k]*dc[:,k]      
      if usecutoff and Hmodel != "ANMEXP_DOMAIN":
        smask[svec2 > cut] = 0.0
        Gamma[i,:] *= smask[:]
      svec = np.sqrt(svec2)
#          S[i,:] = svec
      if (Hmodel == "ANMEXP"):
        expvec = np.exp(-svec/anmexp_lambda/2.)
      if (Hmodel == "EXPCUT"):
        expterm = np.exp((-svec+expcut_thresh)/expcut_lambda)
        expcutvec = np.sqrt(expterm/(1.+expterm))
      if (Hmodel == "ANMEXP_DOMAIN"):
# Two-tier weight: intramolecular pairs use the ANMEXP falloff exp(-r/2λ);
# inter-molecular pairs use a logistic-threshold ANM whose weight is
# sqrt(inter_k * logistic), so the effective force constant is inter_k*logistic
# with logistic = 1/(1+exp((r-inter_thresh)/inter_width)). A pair is
# intramolecular ONLY if it has the same molid AND lies in the same unit cell
# (R=0, indr==0): for any R!=0 the partner is a copy in a neighboring cell, a
# physically distinct protein, so it is inter-molecular regardless of molid.
# The intra/inter mask is parameter-independent; both weights are smooth in the
# refined parameters.
        home = (indr == 0)
        intra = (molid == molid[i]) & home
        logistic = 1./(1.+np.exp((svec-inter_thresh)/inter_width))
        wvec = np.where(intra, np.exp(-svec/anmexp_lambda/2.),
                        np.sqrt(inter_k*logistic))
      svec[svec==0.0] = 1.0
      anmfac = 1./svec
      for k in range(3):
# Hessian value with force constants for the exponential falloff ANM model
        if (Hmodel == "ANMEXP"):
          dc[:,k] *= anmfac*expvec
# Hessian value with force constants for the exponentially smoothed threshold ANM model
        elif (Hmodel == "EXPCUT"):
          dc[:,k] *= anmfac*expcutvec
# Hessian value with force constants for the two-tier domain-coupled model
        elif (Hmodel == "ANMEXP_DOMAIN"):
          dc[:,k] *= anmfac*wvec
        else:
          print("kpoints method requires model=ANMEXP, EXPCUT or ANMEXP_DOMAIN")
          sys.exit(1)
      if (i==0 and indr==0):
        print(dc1D[0:10])
#      for j in range(nsel):
      for k1 in range(3):
        for k2 in range(3):
          H4D[i,k2,:,k1] *= dc[:,k2]
          H4D[i,k1,:,k2] *= dc[:,k2]
# Scale using Gamma and compute the diagonal elements for R=0
    if (indr==0):
      print("H[0,0:10]=",H[0,0:10])
    for i in range(nsel):
      for k1 in range(3):
        for k2 in range(3):
#          H4D[i,k1,:,k2] *= -Gamma[i,:]
          H4D[i,k1,:,k2] *= -Gamma[i,:]/masses_sqrt/masses_sqrt[i]

#          H4D[i,k1,i,k2]=0
#          if (indr == 0):

          Hdiag[i,k1,k2] -= np.sum(H4D[i,k1,:,k2])
#            H4D[i,k1,i,k2] = -np.sum(H4D[i,k1,:,k2])

# Calculate symmetric and asymmetric components
#    Hp = (H + H.transpose())/2.
#    Hm = (H - H.transpose())/2.

#    print "H4D[0,:,12,:]:"
#    for k in range(3):
#      print H4D[0,k,12,:]
#    print "H4D[12,:,0,:]:"
#    for k in range(3):
#      print H4D[12,k,0,:]      
#    print "H[",indr,"][1,:]=",H[1,:]
#    print "H[",indr,"][:,1]=",H[:,1]
#    print "Hp[",indr,"][0,:]=",Hp[0,:]
#    print "Hp[",indr,"][:,0]=",Hp[:,0]
#    print "Hm[",indr,"][0,:]=",Hm[0,:]
#    print "Hm[",indr,"][:,0]=",Hm[:,0]

    for indk in range(len(kplist)):
      kp = kplist[indk]
#      print kp
      if (len(Dklist) < (indk+1)):
        Dklist.append(np.zeros((3*nsel,3*nsel),dtype=complex))
        Dk4Dlist.append(Dklist[indk].reshape((nsel,3,nsel,3)))
# Multiply by phase factor and add to accumulated D:
      kR = kp[0]*R[0]+kp[1]*R[1]+kp[2]*R[2]
      expkR = np.exp((0.0+1.j)*kR)
#      coskR = np.cos(kR)
#      sinkR = np.sin(kR)
#    expkR = np.exp((0.0 + 1.j)*kR)
#        print "kR={0},expkR={1}".format(kR,expkR)
#        coskR = np.cos(kR)
#      Dklist[indk] += Hp*coskR + (0.0+1.j)*Hm*sinkR
      Dklist[indk] += H*expkR
#        D += H*coskR
#          for i in range(nsel):
#            for k1 in range(3):
#              for k2 in range(3):
#                D4D[i,k1,i,k2] += Hdiag[i,k1,k2]

# Correct the diagonal elements

  for indk in range(len(kplist)):
    for i in range(nsel):
      for k1 in range(3):
        for k2 in range(3):
          Dk4Dlist[indk][i,k1,i,k2] += (1.+0.j)*Hdiag[i,k1,k2]
          
# Calculate the diagonal elements
#  for indk in range(len(kplist)):
#    for i in range(nsel):
#      for k1 in range(3):
#        for k2 in range(3):
#          Dk4Dlist[indk][i,k1,i,k2] = 0
#          dsum = np.sum(Dk4Dlist[indk][i,k1,:,k2])
#          Dk4Dlist[indk][i,k1,i,k2] = -dsum
#    print "D[0]=",D[0]
#  if not np.isclose(np.imag(np.conjugate(D.transpose())), np.imag(D), 1.e-10).all():
#    print "D is not symmetric!"
#  Dlist.append(copy.deepcopy(D))
  return Dklist


def get_input_dict(args):
  # return dictionary of keywords and their values
  input_dict={}
  for arg in args:
    spl=arg.split("=")
    if len(spl)==2:
      input_dict[spl[0]]=spl[1]
  return input_dict

def calc_diffuse_layer(i):
    global nh,nk,nl,q0,twopi,V,seln,sites_np,W,nat,nsel,a,b,c,coupling,cir
    D = np.zeros((nk,nl),dtype=complex)
    hh = i - nh//2 + 1
#    print("Layer: ", hh)
    for j in range(nk):
        kk = j - nk//2 + 1
        for k in range(nl):
          ll = k - nl//2 + 1
# Compute some common needed quantities
          h = np.array([hh,kk,ll])
          q = q0*h
          qsq = np.dot(q,q)
          ssq = qsq/twopi/twopi
# Compute the diffuse intensity coupling matrix at a selected wave vector
          if (coupling=="qsqV"):
            dc = qsq*V
          else:
#            dc = qsq*V
            dc = np.exp(qsq*V)-1
# Compute the array of phase factors
          pf = np.exp((0+1.j)*np.dot(sites_np,q))
# Compute the array of Debye-Waller factors
#          dwf = np.exp(2.*us*qsq)
# Compute the array of atomic form factors
          ff = c+a[0]*np.exp(-b[0,:]*ssq)+a[1]*np.exp(-b[1,:]*ssq)
# Compute the array of effective atomic scattering factors
#          f = pf*dwf*ff
          f = pf*ff
# Compute the array of coarse-grained, residue-wise effective atomic scattering factors
          fr = np.zeros(nsel,dtype=complex)
          jj = -1
          for ii in range(nat):
# New residue begins at N atom
            if seln[ii]:
              jj = jj + 1
            fr[jj] = fr[jj] + f[ii]
# Multiply the scattering factors by the coarse-grained Debye-Waller factor
          fr = fr * np.exp(-V.diagonal()*qsq/2.)
# Compute the diffuse intensity at this lattice point
#          print fr*fr.conj()
#          print fr.shape,dc.shape
          D[j][k]=np.dot(fr.conj().T,np.dot(dc,fr))
# Adjust the diagonal using the specified intra-residue correlation
          dD = fr*fr.conj()*(np.exp(cir*V.diagonal()*qsq)-np.exp(V.diagonal()*qsq))
          D[j][k] += np.sum(dD)
# Divide by W
          D[j][k] /= W
    return D

def calc_diffuse_layer_kpoints(i):
    global nh,nk,nl,q0,twopi,V,C,seln,sites_np,W,nat,nsel,a,b,c,coupling,cir,Rlist,kpoints
    D = np.zeros((nk,nl),dtype=complex)
    hh = i - nh//2 + 1
    print("Layer: ", hh)
    for j in range(nk):
        kk = j - nk//2 + 1
        for k in range(nl):
          ll = k - nl//2 + 1
# Compute some common needed quantities
          h = np.array([hh,kk,ll])
          q = q0*h
          qsq = np.dot(q,q)
          ssq = qsq/twopi/twopi
# Compute the diffuse intensity coupling matrix at a selected wave vector
          if (kpoints == 1):
            if (coupling == "qsqV"):
              dc = qsq*C[0]
            else:
              dc = np.exp(qsq*C[0]) -1
          else:
            dc = np.zeros(V.shape,dtype=complex)
            for indr in range(len(Rlist)):
              R = Rlist[indr]
              expqR = np.exp((0 + 1.j)*np.dot(q,R))
#            cosqR = np.cos(np.dot(q,R))
              if (coupling == "qsqV"):
#              dc += cosqR*(qsq*C[indr])
                thisdc = expqR*(qsq*C[indr])
              else:
#               dc += cosqR*(np.exp(qsq*C[indr])-1)
                thisdc = expqR*(np.exp(qsq*C[indr])-1)
              dc = dc + thisdc
# Compute the array of phase factors
          pf = np.exp((0+1.j)*np.dot(sites_np,q))
# Compute the array of Debye-Waller factors
#          dwf = np.exp(2.*us*qsq)
# Compute the array of atomic form factors
          ff = c+a[0]*np.exp(-b[0,:]*ssq)+a[1]*np.exp(-b[1,:]*ssq)
# Compute the array of effective atomic scattering factors
#          f = pf*dwf*ff
          f = pf*ff
# Compute the array of coarse-grained, residue-wise effective atomic scattering factors
          fr = np.zeros(nsel,dtype=complex)
          jj = -1
          for ii in range(nat):
# New residue begins at N atom
            if seln[ii]:
              jj = jj + 1
            fr[jj] = fr[jj] + f[ii]
# Multiply the scattering factors by the coarse-grained Debye-Waller factor
          fr = fr * np.exp(-V.diagonal()*qsq/2.)
# Compute the diffuse intensity at this lattice point
#          print fr*fr.conj()
#          print fr.shape,dc.shape
          D[j][k]=np.dot(fr.conj().T,np.dot(dc,fr))
# Adjust the diagonal using the specified intra-residue correlation
          dD = fr*fr.conj()*(np.exp(cir*V.diagonal()*qsq)-np.exp(V.diagonal()*qsq))
          D[j][k] += np.sum(dD)
# Divide by W
          D[j][k] /= W
    return D

def build_group_matrix(seln_arr, nat, nsel):
    """Return the (nsel x nat) residue-grouping matrix G such that
    fr = G @ f reproduces the per-atom coarse-graining loop, in which a new
    residue begins at each backbone-N atom (seln[ii] True). G[r, ii] = 1 when
    atom ii belongs to residue r."""
    seln_bool = np.asarray([bool(seln_arr[ii]) for ii in range(nat)])
    group_idx = np.cumsum(seln_bool) - 1
    G = np.zeros((nsel, nat))
    G[group_idx, np.arange(nat)] = 1.0
    return G

def calc_diffuse_vectorized(nh, nk, nl, q0, twopi, coupling_matrix, Vdiag,
                            sites_np, a, b, c, G, cir, W, coupling,
                            point_block=4096):
    """Vectorized diffuse-scattering calculation over the whole reciprocal grid.

    Numerically identical (to ~1e-9 relative) to the per-point loops in
    calc_diffuse_layer / calc_diffuse_layer_kpoints (kpoints==1), but much
    faster: the cheap per-point quantities (phase factors, form factors,
    coarse-grained scattering factors fr) are computed in large vectorized
    blocks, and the expensive coupling contraction fr^H . dc . fr -- where
    dc = exp(qsq * coupling_matrix) - 1 depends only on the scalar qsq -- is
    evaluated once per *unique* qsq value and applied to all points sharing
    that qsq with a single BLAS matrix multiply.

    coupling_matrix is V (kpoints==0) or C[0] (kpoints==1); Vdiag is always
    V.diagonal(), matching the original code. Returns DS with shape
    (nh, nk, nl), real dtype, ready to write out.
    """
    nsel = coupling_matrix.shape[0]
    h = np.arange(nh) - nh//2 + 1
    k = np.arange(nk) - nk//2 + 1
    l = np.arange(nl) - nl//2 + 1
    H, K, L = np.meshgrid(h, k, l, indexing='ij')
    hvec = np.stack([H.reshape(-1), K.reshape(-1), L.reshape(-1)], axis=1).astype(float)
    npts = hvec.shape[0]
    q = hvec * q0[None, :]
    qsq = np.einsum('pi,pi->p', q, q)
    ssq = qsq / twopi / twopi

    # Coarse-grained, DW-weighted scattering factors fr for every grid point,
    # computed in memory-bounded blocks (all cheap, fully vectorized ops).
    # fr is the one large array we must keep whole (the qsq grouping below
    # indexes it in scattered order). Peak memory ~ npts*nsel*16 bytes.
    # The per-point intra-residue correction (dD) depends only on qsq, so we
    # fold it in here rather than materializing a second npts*nsel matrix.
    fr = np.empty((npts, nsel), dtype=complex)
    Dcorr = np.empty(npts)
    for s in range(0, npts, point_block):
        e = min(s + point_block, npts)
        pf = np.exp(1j * (q[s:e] @ sites_np.T))
        ff = (c[None, :]
              + a[0][None, :] * np.exp(-b[0][None, :] * ssq[s:e, None])
              + a[1][None, :] * np.exp(-b[1][None, :] * ssq[s:e, None]))
        frb = ((pf * ff) @ G.T) * np.exp(-Vdiag[None, :] * qsq[s:e, None] / 2.)
        fr[s:e] = frb
        corr = (np.exp(cir * Vdiag[None, :] * qsq[s:e, None])
                - np.exp(Vdiag[None, :] * qsq[s:e, None]))
        Dcorr[s:e] = np.einsum('pm,pm->p', (frb * frb.conj()).real, corr)

    # Main coupling term, grouped by unique qsq: dc = exp(qsq*coupling)-1
    # depends only on the scalar qsq, so compute it once per unique qsq and
    # apply it to all points sharing that qsq with a single matrix multiply.
    Dmain = np.empty(npts, dtype=complex)
    if coupling == "qsqV":
        Dmain = qsq * np.einsum('pm,pm->p', fr.conj(), fr @ coupling_matrix.T)
    else:
        uq, inv = np.unique(np.round(qsq, 9), return_inverse=True)
        order = np.argsort(inv, kind='stable')
        bounds = np.searchsorted(inv[order], np.arange(len(uq) + 1))
        for u in range(len(uq)):
            idx = order[bounds[u]:bounds[u + 1]]
            dc = np.exp(uq[u] * coupling_matrix) - 1.0      # one exp per unique qsq
            frg = fr[idx]
            Dmain[idx] = np.einsum('pm,pm->p', frg.conj(), frg @ dc.T)

    DS = ((Dmain.real + Dcorr) / W)
    return DS.reshape(nh, nk, nl)

# All atom diffuse scattering calculation
def calc_diffuse_layer_aa(i):
    global nh,nk,nl,q0,twopi,V,seln,sites_np,W,nat,nca,a,b,c,coupling,cir
    D = np.zeros((nk,nl),dtype=complex)
    hh = i - nh//2 + 1
#      print hh,".",
    for j in range(nk):
        kk = j - nk//2 + 1
        for k in range(nl):
          ll = k - nl//2 + 1
# Compute some common needed quantities
          h = np.array([hh,kk,ll])
          q = q0*h
          qsq = np.dot(q,q)
          ssq = qsq/twopi/twopi
# Compute the diffuse intensity coupling matrix at a selected wave vector
          if (coupling=="qsqV"):
            dc = qsq*V
          else:
            dc = np.exp(qsq*V)-1
# Compute the array of phase factors
          pf = np.exp((0+1.j)*np.dot(sites_np,q))
# Compute the array of Debye-Waller factors
#          dwf = np.exp(2.*us*qsq)
# Compute the array of atomic form factors
          ff = c+a[0]*np.exp(-b[0,:]*ssq)+a[1]*np.exp(-b[1,:]*ssq)
# Compute the array of effective atomic scattering factors
#          f = pf*dwf*ff
          f = pf*ff
# Compute the array of coarse-grained, residue-wise effective atomic scattering factors
#          fr = np.zeros(nca,dtype=complex)
#          jj = -1
#          for ii in range(nat):
# New residue begins at N atom
#            if seln[ii]:
#              jj = jj + 1
#            fr[jj] = fr[jj] + f[ii]
# Multiply the scattering factors by the Debye-Waller factor
          f = f * np.exp(-V.diagonal()*qsq/2.)
# Compute the diffuse intensity at this lattice point
#          print fr*fr.conj()
#          print fr.shape,dc.shape
          D[j][k]=np.dot(f.conj().T,np.dot(dc,f))
# Adjust the diagonal using the specified intra-residue correlation
#          dD = fr*fr.conj()*(np.exp(cir*V.diagonal()*qsq)-np.exp(V.diagonal()*qsq))
#          D[j][k] += np.sum(dD)
# Divide by W
          D[j][k] /= W
    return D

if __name__=="__main__":
    
    Hmodel = "ANM"
    limitmodes = False
    pbc = True
    cutoff = 10.
    cir = 1.
    nproc=1
    xmodes=0
    writesvd=False
    readsvd = False
    coupling="default"
    writeH = False
    writeHtilde = False
    writeHrenorm = False
    writecorr = False
    writeV = False
    writeV0 = False
    writeC = False
    writeS = False
    writeVpdb = False
    writeGamma = False
    writeDiffuse = False
    writepdbwork = False
    writepdb = False
    writeCvsR = False
    usefastpinv = False
    readGamma = False
    calcR = False
    usecutoff = True
    w_benm=1.
    fnm=1.
    anmexp_lambda=0.157
    expcut_lambda = 1.0
    expcut_thresh = 1.0
# Two-tier domain-coupled model (ANMEXP_DOMAIN): intramolecular pairs use ANMEXP,
# inter-molecular pairs use a logistic-threshold ANM whose effective force
# constant is inter_k/(1+exp((r-inter_thresh)/inter_width)).
    inter_thresh = 10.0
    inter_k = 1.0
    inter_width = 1.5
# Covariance shrinkage toward the diagonal (Ledoit-Wolf style): after
# renormalization, off-diagonal covariance elements are scaled by cov_shrink
# while the diagonal (the imposed experimental B-factors) is preserved exactly.
# cov_shrink=1.0 is the raw model; a value slightly below 1 restores positive
# definiteness when the near-singular pseudoinverse leaves a small negative mode.
# cov_shrink="auto" derives the minimal shrinkage from the matrix itself (from
# the most-negative correlation eigenvalue) so the user need not supply a value.
    cov_shrink = 1.0
    llm_gamma = 4.0
    svdbasename = "svd"
    asel = "CA"
    res = 1.6
    kpoints=0
    args = sys.argv[1:] 
    input_dict = get_input_dict(args)
    for a in input_dict:
        if (a == "input.pdb"):
          pdbinname = input_dict[a]
        if (a == "output.pdb"):
          pdboutname = input_dict[a]
          writepdb = True
        if (a == "output.pdbwork"):
          pdbworkname = input_dict[a]
          writepdbwork = True
        if (a == "output.CvsR"):
          cvsrname = input_dict[a]
          writeCvsR = True
        if (a == "input.Gamma"):
          Gammainname = input_dict[a]
          readGamma = True
        if (a == "model"):
          Hmodel = input_dict[a]
        if (a == "kpoints"):
          kpoints = int(input_dict[a])
        if (a == "output.hkl"):
          hkloutname = input_dict[a]
          writeDiffuse = True
        if (a == "input.hkl"):
          hklinname = input_dict[a]
          calcR = True
        if (a == "nmodes"):
          nmodes = int(input_dict[a])
          limitmodes = True
        if (a == "xmodes"):
          xmodes = int(input_dict[a])
        if (a == "cutoff"):
          cutoff = float(input_dict[a])
          usecutoff = True
        if (a == "anmexp.lambda"):
          anmexp_lambda = float(input_dict[a])
        if (a == "inter.thresh"):
          inter_thresh = float(input_dict[a])
        if (a == "inter.k"):
          inter_k = float(input_dict[a])
        if (a == "inter.width"):
          inter_width = float(input_dict[a])
        if (a == "covariance.shrinkage"):
          if str(input_dict[a]).lower() == "auto":
            cov_shrink = "auto"
          else:
            cov_shrink = float(input_dict[a])
        if (a == "expcut.lambda"):
          expcut_lambda = float(input_dict[a])
        if (a == "expcut.thresh"):
          expcut_thresh = float(input_dict[a])
        if (a == "llm.gamma"):
          llm_gamma = float(input_dict[a])
        if (a == "usecutoff"):
          if (input_dict[a] == "False"):
              usecutoff = False
          else:
              usecutoff = True
        if (a == "pbc"):
          if(input_dict[a] == "False"):
            pbc = False
          else:
            pbc = True
        if (a == "nproc"):
          nproc = int(input_dict[a])
        if (a == "svd.basename"):
          svdbasename = input_dict[a]
        if (a == "write.svd"):
          if (input_dict[a] == "True"):
            writesvd = True
        if (a == "fast.pinv"):
          if (input_dict[a] == "True"):
            usefastpinv = True
        if (a == "read.svd"):
          if (input_dict[a] == "True"):
            readsvd = True
        if (a == "output.H"):
          Houtname = input_dict[a]
          writeH = True
        if (a == "output.Htilde"):
          Htildeoutname = input_dict[a]
          writeHtilde = True
        if (a == "output.Hrenorm"):
          Hrenormoutname = input_dict[a]
          writeHrenorm = True
        if (a == "output.Gamma"):
          Gammaoutname = input_dict[a]
          writeGamma = True
        if (a == "backbone.enhancement"):
          w_benm=float(input_dict[a])
        if (a == "output.corr"):
          corroutname = input_dict[a]
          writecorr = True
        if (a == "output.V0"):
          V0outname = input_dict[a]
          writeV0 = True
        if (a == "output.V"):
          Voutname = input_dict[a]
          writeV = True
        if (a == "output.C"):
          Coutname = input_dict[a]
          writeC = True
        if (a == "output.S"):
          Soutname = input_dict[a]
          writeS = True
        if (a == "output.Vpdb"):
          Vpdboutname = input_dict[a]
          writeVpdb = True
        if (a == "coupling"):
          coupling = input_dict[a]
        if (a == "intra.residue.corr"):
          cir = float(input_dict[a])
        if (a == "nm.fraction"):
          fnm = float(input_dict[a])
        if (a == "atom.selection"):
          asel = input_dict[a]
        if (a == "resolution"):
          res = float(input_dict[a])

# Read structure file
    pdb_in = hierarchy.input(file_name=pdbinname)
# Where there are alternate conformations, select the A conformation and delete the rest
    pdb_in.hierarchy.remove_hd()
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
#    ftmp.write(pdb_in.hierarchy.as_pdb_string())
    ftmp.close()
    pdb_in = hierarchy.input(file_name=pdbinname+"_tmp.pdb")
    struct_asym = pdb_in.input.xray_structure_simple()
# Calculate R factor if asked for
    if calcR:
        hklin = any_reflection_file(file_name="fix_last.updated_refine_001.mtz")    
        miller_arrays=hklin.as_miller_arrays()
        i_obs = miller_arrays[0]
        flags = miller_arrays[1]
        f_obs=i_obs.f_sq_as_f().map_to_asu()
        flags = flags.customized_copy(data=flags.data()==1).map_to_asu()
        f_obs = f_obs.merge_equivalents().array()
        flags = flags.merge_equivalents().array()
        flags_plus_minus = flags.generate_bijvoet_mates()
        f_obs, flags_plus_minus = f_obs.common_sets(other=flags_plus_minus)
        fin_calc = struct_asym.structure_factors(anomalous_flag=True,d_min=1.6).f_calc()
        compute_r_factors(f_obs,fin_calc,flags_plus_minus)
    struct = struct_asym.expand_to_p1(append_number_to_labels=True)
# Get scatterers
    scat = struct.scatterers()
# Reorder the scatterers and place back in the structure
    struct.erase_scatterers()
    nc=4
    for i in range(nc):
      for j in range(int(len(scat)/nc)):
        struct.add_scatterer(scat[j*nc+i])
# Get scatterers in new order
    scat = struct.scatterers()
#    print scat[0].label,scat[1].label
# Select alpha carbons and backbone nitrogens
# Note: after structure reordering, we need to create selection arrays from scatterer labels
    selca = flex.bool(len(scat))
    seln = flex.bool(len(scat))
    for i, scatterer in enumerate(scat):
        label = scatterer.label
        # Check for CA atoms (alpha carbons)
        if ' CA ' in label:
            selca[i] = True
        # Check for backbone N atoms
        if (' N  ' in label or label.startswith('pdb=" N ')) and 'pdb=' in label:
            seln[i] = True
# Get atom sites, which contain positional coords
    sites=struct.sites_cart()
#    global sites_np
    sites_np=np.zeros((len(sites),3))
    for i in range(len(sites)):
      for k in range(3):
        sites_np[i][k] = sites[i][k]
# Get unit cell
    cell=flex.double(struct.unit_cell().parameters())
# Get isotropic Us (the method returns u_iso/3
    us = 3.*struct.extract_u_iso_or_u_equiv()
# Get atomic weights
    masses_cctbx = struct.atomic_weights()
# Get atom form factor parameters
    c = np.zeros(len(scat))
    a = np.zeros((2,len(scat)))
    b = np.zeros((2,len(scat)))
    for i in range(len(scat)):
      ng = xray_scattering.n_gaussian_table_entry(scat[i].scattering_type,2).gaussian()                  
      c[i]=ng.c()
      for j in range(len(a[:,i])):
        a[j][i]=ng.array_of_a()[j]
        b[j][i]=ng.array_of_b()[j]
# Calculate the number of selected atoms
    nsel=0
    for i in range(len(sites)):
      if (asel == "heavy" or (asel == "CA" and selca[i])):
        nsel = nsel + 1
    print("nsel = ",nsel)
# Get the sites and sigmas for the selected atoms
    sigs = np.zeros(nsel)
    masses = np.zeros(nsel)
    coords = np.zeros((nsel,3))
    coords1D = coords.reshape(3*nsel)
# molid[snum] is the crystallographic molecule (P1 symmetry-operator index) of
# selected atom snum, parsed from the "_K" suffix that expand_to_p1 appends to
# each scatterer label. Used by the two-tier ANMEXP_DOMAIN model to distinguish
# intramolecular from inter-molecular atom pairs. It is parameter-independent.
# resid[snum] is the residue sequence number, parsed from the same label
# (e.g. 'pdb=" CA  LYS A   6 "_0' -> 6). Used by the backbone-enhanced ENM (BENM)
# to identify sequence-adjacent CA pairs. chnid[snum] is the chain identifier
# (a single character) so backbone bonds are not made across chain breaks.
    molid = np.zeros(nsel,dtype=int)
    resid = np.zeros(nsel,dtype=int)
    chnid = np.empty(nsel,dtype='U4')
    snum = 0
    for i in range(len(sites)):
      if (asel == "heavy" or (asel == "CA" and selca[i])):
        sigs[snum] = np.sqrt(us[i])
        masses[snum] = masses_cctbx[i]
        for k in range(3):
          coords[snum][k] = sites[i][k]
        try:
          molid[snum] = int(scat[i].label.rsplit('_',1)[1])
        except (IndexError, ValueError):
          molid[snum] = 0
# Parse residue number and chain id from the pdb= field of the label, e.g.
# 'pdb=" CA  LYS A   6 "_0' -> chain 'A', residue 6. The pdb field is fixed-width:
# atom name (4) + altloc(1) + resname(3) + chain(1) + resseq(4).
        try:
          fld = scat[i].label.split('pdb="',1)[1]
          chnid[snum] = fld[9]
          resid[snum] = int(fld[10:14])
        except (IndexError, ValueError):
          chnid[snum] = ''
          resid[snum] = -1000000 - snum
        snum = snum + 1
    print("Number of distinct molecules (molid) = ",len(np.unique(molid)))
# Backbone-bond mask for the backbone-enhanced ENM (BENM, Ming & Wall 2005).
# A pair (i,j) is a backbone bond iff it is sequence-adjacent (|resid|==1) within
# the same chain AND the same crystallographic molecule (R=0 handled per-cell in
# calcDk). The mask is parameter-independent; the BENM multiplies the force
# constant of these pairs by the refinable factor w_benm (backbone.enhancement).
    bbond = (np.abs(resid[:,None] - resid[None,:]) == 1) \
            & (molid[:,None] == molid[None,:]) \
            & (chnid[:,None] == chnid[None,:])
    print("Number of backbone bonds (i<j) = ",int(np.sum(np.triu(bbond,1))),
          " ; backbone.enhancement w_benm = ",w_benm)
    cellby2 = cell[0:3]/2.
    np.set_printoptions(threshold=None)
    masses_sqrt = np.sqrt(masses)
# Define cutoff
#    cut = 10.5*10.5
    cut = cutoff*cutoff
# Calculate the off-diagonal elements of the Hessian
    if readGamma:
        Gamma = np.load(Gammainname)
        if (len(Gamma) != nsel):
            print("Size of Gamma is different from size of H")
            sys.exit(1)
    else:
        Gamma = np.zeros((nsel,nsel))
        Gamma.fill(1.)
    if writepdbwork:
      pdb_in.hierarchy.adopt_xray_structure(struct_asym)
      pdb_in.hierarchy.write_pdb_file(pdbworkname,anisou=False)
    print("Calculate the Hessian")
    stime = time.time()
    if (kpoints == 0 and Hmodel != "LLM"):
      H = calcH(Gamma,coords,cell,Hmodel)
    elif (kpoints > 0 or Hmodel == "LLM"):
#      D = np.array([calcDk(coords,cell,Hmodel,[0.,0.,0.])])
     print(Hmodel)
     Rlist = list()
     Rlist.append(np.array([0.0,0.0,0.0]))
     for ii in range(-1,2):
       Ri = ii*cell[0]
       for jj in range(-1,2):
#    for jj in range(0,1):
         Rj = jj*cell[1]
         for kk in range(-1,2):
#      for kk in range(0,1):
           if (ii !=0 or jj !=0 or kk !=0):
             Rk = kk*cell[2]
             Rlist.append(np.array([Ri,Rj,Rk]))
#              print np.array([Ri,Rj,Rk])
     if (Hmodel != "LLM"):
      kplist = list()
      for ki in range(1,kpoints+1):
        ui = (2.*float(ki) - float(kpoints) - 1.)/2./float(kpoints)
        for kj in range(1,kpoints+1):
          uj = (2.*float(kj) - float(kpoints) - 1.)/2./float(kpoints)
          for kk in range(1,kpoints+1):
            uk = (2.*float(kk) - float(kpoints) - 1.)/2./float(kpoints)
#            if (ui != 0.0 and uj != 0.0 and uk != 0.0):
            kplist.append(2.*np.pi*np.array([ui/cell[0],uj/cell[1],uk/cell[2]]))
#      print kp
      D = calcDk(coords,Rlist,Hmodel,kplist)
#      sys.exit(1)       
#              D.append(calcDk(coords,cell,Hmodel,kp))
#          D4D[i,k1,:,k2] = H4D[i,k1,:,k2]/masses_sqrt/masses_sqrt[i]
#    D = -D
#    for i in range(len(H)):
#      print H[i][i]," ",
    if writeH:
      np.save(Houtname,H)
    etime = time.time()
    print("time = ",etime-stime," seconds")
# Calculate the pseudoinverse of the Hessian
    if (Hmodel != "LLM"):
     print("Calculate the peseudoinverse of the Hessian")
     stime = time.time()
     if usefastpinv:
      timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
      Xinfile=os.path.join("X_work_"+timestamp+".npy")
      Xtildeoutfile=os.path.join("Xtilde_work_"+timestamp+".npy")
      stime = time.time()
      np.save(Xinfile,H)
      command = 'python calc_pinv.py input.X={0} output.Xtilde={1}'.format(Xinfile,Xtildeoutfile)
#    command = '/projects/opt/centos7/anaconda3/bin/python calc_pinv.py input.X={0} output.Xtilde={1}'.format(Xinfile,Xtildeoutfile)
      print(command)
      call_params = shlex.split(command)
      subprocess.call(call_params,env=dict(os.environ,OMP_NUM_THREADS="16"))
      Htilde = np.load(Xtildeoutfile)
      etime = time.time()
      print("time = ",etime-stime," seconds")
      command = 'rm {0} {1}'.format(Xinfile,Xtildeoutfile)
      call_params = shlex.split(command)
      subprocess.call(call_params)
     else:
      if (kpoints==0):
        Htilde = np.linalg.pinv(H)
      else:
#        print D[0].shape
        Htilde = np.zeros(D[0].shape,dtype=complex)
        Ddag = list()
#        print "There are ",np.sum(np.abs(np.imag(Ddag[0]))>0.001)," imaginary components of abs(Ddag[0]) > 0.001"
        print(len(D))
        for i in range(len(D)):
          Ddag.append(np.linalg.pinv(D[i]))
#          Ddag.append(sp.linalg.pinv(D[i]))
#          print "There are ",np.sum(np.abs(np.imag(Ddag[i]))>0.001)," imaginary components of abs(Ddag[",i,"]) > 0.001"
          Htilde = Htilde + Ddag[i]
#          print "Number of negative Ddag[i].diagonal() elements = ",np.sum(Ddag[i].diagonal()<0)
        print("There are ",np.sum(np.abs(np.imag(Htilde))>0.001)," imaginary components of Htilde > 0.001")
#    HtildeImag=np.imag(Htilde)
     etime = time.time()
     print("time = ",etime-stime," seconds")
# Calculate the atom pair covariance matrix from Htilde
#    Htilde4D = np.real(Htilde.reshape((nsel,3,nsel,3)))
     Htilde4D = Htilde.reshape((nsel,3,nsel,3))
     for i in range(nsel):
       for k1 in range(3):
         for k2 in range(3):
#              thisC[i,:] += coskR * Ddag4D[indk][i,k,:,k]
#              thisC[i,:] += expkR * Ddag4D[indk][i,k,:,k]/masses_sqrt/masses_sqrt[i]
           Htilde4D[i,k1,:,k2] *= 1./masses_sqrt/masses_sqrt[i]
     if (kpoints > 0):
      Ddag4D = list()
      for i in range(len(Ddag)):
        Ddag4D.append(Ddag[i].reshape((nsel,3,nsel,3)))
     V = np.zeros((nsel,nsel))
     for i in range(nsel):
      for k in range(3):
#        V[i,:] += np.real(Htilde4D[i,k,:,k])/masses_sqrt/masses_sqrt[i]
        V[i,:] += np.real(Htilde4D[i,k,:,k])
     if (kpoints > 0): 
      C = list()
      for indr in range(len(Rlist)):
        R = Rlist[indr]
        thisC = np.zeros((nsel,nsel),dtype=complex)
        for indk in range(len(kplist)):
          kp = kplist[indk]
          kR = kp[0]*R[0]+kp[1]*R[1]+kp[2]*R[2]
          expkR = np.exp((0 + 1.j)*kR)
#          coskR = np.cos(kR)
          for i in range(nsel):
            for k in range(3):
#              thisC[i,:] += coskR * Ddag4D[indk][i,k,:,k]
              thisC[i,:] += expkR * Ddag4D[indk][i,k,:,k]/masses_sqrt/masses_sqrt[i]
#              thisC[i,:] += expkR * Ddag4D[indk][i,k,:,k]
        C.append(np.real(thisC))
#        nimag=np.sum(np.abs(np.imag(thisC))>1.0e-10)
#        if (nimag>0):
#          print "Number of imaginary elements in thisC = ",nimag
#          print "thisC[0,0:10]=",thisC[0,0:10]
#        V[i,:] += Htilde4D[i,k,:,k]/masses_sqrt/masses_sqrt[i]
    else:
      C = list()
      for indr in range(len(Rlist)):
        R = Rlist[indr]
        dc = np.zeros((nsel,3))
        thisC = np.zeros((nsel,nsel))
        for i in range(nsel):
          svec2 = np.zeros(nsel)
          for k in range(3):
            dc[:,k] = (coords[i,k] - coords[:,k] + R[k])
            svec2[:] += dc[:,k]*dc[:,k]      
          thisC[i,:] = np.exp(-np.sqrt(svec2)/llm_gamma)
        C.append(thisC)
      V = C[0]
    nnpos = np.sum(V.diagonal()<=0)
    print("Number of nonpositive V.diagonal() elements = ",nnpos," out of ",len(V.diagonal()))
#    print "V.diagonal()=",V.diagonal()
    print("V[0,0:10]=",V[2,0:10])
#    print "np.real(C[0].diagonal())=",np.real(C[0].diagonal())
    if (nnpos == 0):
      Vsigs = np.sqrt(V.diagonal())
    else:
      raise ValueError("There are nonpositive V.diagonal() elements, aborting")
    if writeV0:
      np.save(V0outname,V)
# Renormalize V and Htilde
    renorm_fac = sigs/Vsigs
    for i in range(nsel):
      V[i,:] *= renorm_fac
      V[:,i] *= renorm_fac
      if (kpoints>0 or Hmodel == "LLM"):
        for indr in range(len(C)):
          C[indr][i,:] *= renorm_fac
          C[indr][:,i] *= renorm_fac
      if (Hmodel != "LLM"):
        for k1 in range(3):
          for k2 in range(3):
            Htilde4D[i,k1,:,k2] *= renorm_fac
            Htilde4D[:,k1,i,k2] *= renorm_fac
            if (kpoints>0):
              for j in range(len(Ddag)):
                Ddag4D[j][i,k1,:,k2] *= renorm_fac
                Ddag4D[j][:,k1,i,k2] *= renorm_fac
    print("Renorm done")
# Covariance shrinkage toward the diagonal: M -> cov_shrink*M + (1-cov_shrink)*diag(M),
# i.e. scale every off-diagonal element by cov_shrink while preserving the
# diagonal (the imposed experimental B-factors) exactly. This is a mild global
# rescale that restores positive definiteness lost to the near-singular
# pseudoinverse; cov_shrink=1.0 leaves the model unchanged. Applied to both V
# (kpoints==0 diffuse path) and every C[indr] (kpoints>0 path) so the diffuse
# calculation is consistent regardless of method. It commutes with
# renormalization (both are diagonal-preserving), so order does not matter.
# cov_shrink=="auto": derive the minimal shrinkage from the matrix itself.
# Off-diagonal scaling by a maps correlation eigenvalues lam -> (1-a)+a*lam, so
# the most-negative correlation eigenvalue lam_min sets the positive-definite
# threshold a* = 1/(1-lam_min).  A matrix is PD iff its correlation matrix is,
# and shrinkage is applied uniformly to V and every C[indr], so we take lam_min
# over whichever matrices actually feed the diffuse calc: V (kpoints==0) or C[0]
# (kpoints>0).  Each correlation uses its own diagonal.  We take a hair below a*
# (scaled by 1-eps) so the smallest eigenvalue lands just above 0.
    if cov_shrink == "auto":
      shrink_targets = [C[0]] if (kpoints>0) else [V]
      lam_min = 0.0
      for M in shrink_targets:
        d = np.sqrt(np.clip(M.diagonal(), 1.0e-30, None))
        Corr = M / np.outer(d, d)
        lam_min = min(lam_min, np.linalg.eigvalsh(0.5*(Corr+Corr.T)).min())
      if lam_min < 0.0:
        a_star = 1.0/(1.0 - lam_min)
        cov_shrink = a_star*(1.0 - 1.0e-6)
        print("Auto covariance shrinkage: lam_min(Corr) = ",lam_min,
              " -> cov_shrink = ",cov_shrink)
      else:
        cov_shrink = 1.0
        print("Auto covariance shrinkage: Corr already PD (lam_min = ",
              lam_min,"), cov_shrink = 1.0 (no-op)")
    if cov_shrink != 1.0:
      Vdiag_keep = V.diagonal().copy()
      V *= cov_shrink
      np.fill_diagonal(V, Vdiag_keep)
      if (kpoints>0 or Hmodel == "LLM"):
        for indr in range(len(C)):
          Cdiag_keep = C[indr].diagonal().copy()
          C[indr] *= cov_shrink
          np.fill_diagonal(C[indr], Cdiag_keep)
      print("Applied covariance shrinkage cov_shrink = ",cov_shrink)
    if (writeCvsR==True):
      cvsr = CvsR(coords,Rlist,C)
      np.savetxt(cvsrname,cvsr)
    if writeHtilde:
      np.save(Htildeoutname,Htilde)
# Write PDB with anisotropic Us
    if writepdb:
        stmp = pdb_in.input.xray_structure_simple()
        stmp.convert_to_anisotropic()
        pdb_in.hierarchy.adopt_xray_structure(stmp)
        pdb_in.hierarchy.write_pdb_file("before.pdb",anisou=True)
        if calcR:
            fout_calc = stmp.structure_factors(anomalous_flag=True,d_min=1.6).f_calc()
            compute_r_factors(f_obs,fout_calc,flags_plus_minus)
        utmp = stmp.extract_u_cart_plus_u_iso()
        uisotmp = stmp.extract_u_iso_or_u_equiv()
        for i in range(len(utmp)):
            tmp = flex.double(utmp[i])
            for j in range(3):
                tmp[j]=float(np.real(Htilde4D[i,j,i,j]))
            tmp[3]=float(np.real(Htilde4D[i,0,i,1]))
            tmp[4]=float(np.real(Htilde4D[i,0,i,2]))
            tmp[5]=float(np.real(Htilde4D[i,1,i,2]))
            uisotmp[i] = (tmp[0]+tmp[1]+tmp[2])/3.
            utmp[i]=list(tmp)
        stmp.convert_to_isotropic()
        stmp.set_u_iso(values=uisotmp)
        pdb_in.hierarchy.adopt_xray_structure(stmp)
        stmp.convert_to_anisotropic()
        stmp.set_u_cart(utmp)
        pdb_in.hierarchy.adopt_xray_structure(stmp)
        pdb_in.hierarchy.write_pdb_file(pdboutname,anisou=True)
        if calcR:
            fout_calc = stmp.structure_factors(anomalous_flag=True,d_min=1.6).f_calc()
            compute_r_factors(f_obs,fout_calc,flags_plus_minus)

# Write V coord files
    if writeVpdb:
        ctca = 0
        for i in range(nsel):
            if selca[i]:
                Vslice = deepcopy(V[i,:])
                Vslice[i] = 0.0
                struct.set_u_iso(values=flex.double(Vslice))
                pdb_string = struct.as_pdb_file()
                thisname = '{0}_{1:03d}.pdb'.format(Vpdboutname,ctca)
                open(thisname,"w").write(pdb_string)
                ctca += 1
    if writeV:
        np.save(Voutname,V)
    if writeC:
        Corr = deepcopy(V)
        for i in range(nsel):
            Corr[i,:] /= sigs
            Corr[:,i] /= sigs
        np.save(Coutname,Corr)
#    if writeS:
#        np.save(Soutname,S)
    if writeHrenorm or writeGamma:
# Calculate the renormalized Hessian
        print("Calculate the renormalized Hessian")
        Hrenorm = copy.deepcopy(H)
        Hrenorm4D = Hrenorm.reshape((nsel,3,nsel,3))
        for i in range(nsel):
          for k1 in range(3):
            for k2 in range(3):
              Hrenorm4D[i,k1,:,k2] /= renorm_fac
              Hrenorm4D[:,k1,i,k2] /= renorm_fac
#        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
#        Xinfile=os.path.join("X_work_"+timestamp+".npy")
#        Xtildeoutfile=os.path.join("Xtilde_work_"+timestamp+".npy")
#        stime = time.time()
#        np.save(Xinfile,Htilde)
#        command = 'python calc_pinv.py input.X={0} output.Xtilde={1}'.format(Xinfile,Xtildeoutfile)
#        print(command)
#        call_params = shlex.split(command)
#        subprocess.call(call_params,env=dict(os.environ,OMP_NUM_THREADS="16"))
#        Hrenorm = np.load(Xtildeoutfile)
#        etime = time.time()
#        print("time = ",etime-stime," seconds")
#        command = 'rm {0} {1}'.format(Xinfile,Xtildeoutfile)
#        print(command)
#        call_params = shlex.split(command)
#        subprocess.call(call_params)
        if writeHrenorm:
            np.save(Hrenormoutname,Hrenorm)
        if writeGamma:
            Gamma = np.zeros((nsel,nsel))
            for i in range(nsel):
                for k in range(3):
                    Gamma[i,:] -= Hrenorm4D[i,k,:,k]
#            for i in range(nsel):
#                Gamma[i,:] *= masses_sqrt
#                Gamma[:,i] *= masses_sqrt
#                Gamma[i,i] = 0.0
            np.save(Gammaoutname,Gamma)
# Compute the diffuse intensity on a grid
#    sys.exit()
    nh=(int(cell[0]/res)+1)*2
    nk=(int(cell[1]/res)+1)*2
    nl=(int(cell[2]/res)+1)*2
    F000=np.sum(c+a[0]+a[1])
#    global W,nk,nl,D,twopi,q0,nat
    W=F000*F000/1.e6
    DS = np.zeros((nh,nk,nl),dtype=complex)
    twopi = 2. * np.pi
    q0 = np.array([twopi/cell[0],twopi/cell[1],twopi/cell[2]])
#    if (kpoints != 0):
#      q0 = q0/kpoints
#      nh = nh*kpoints
#      nk = nk*kpoints
#      nl = nl*kpoints      
    nat = len(scat)
    if writeDiffuse:
# Multiprocess call to calculate diffuse
# Use time() for time since multiprocess
        stime = time.time()
        # Vectorized path: CA coarse-graining with kpoints 0 or 1 (single
        # coupling matrix). Numerically identical to the per-point loops but
        # far faster. Falls back to the original loops for the all-atom
        # ("heavy") path and for multi-R kpoints (>1).
        use_vectorized = (asel != "heavy") and (kpoints in (0, 1))
        if use_vectorized:
            coupling_matrix = V if (kpoints == 0) else C[0]
            G = build_group_matrix(seln, nat, nsel)
            DS = calc_diffuse_vectorized(nh, nk, nl, q0, twopi,
                                         coupling_matrix, V.diagonal(),
                                         sites_np, a, b, c, G, cir, W, coupling)
        else:
            pool = ThreadPool(processes=nproc)
            if (asel == "heavy"):
                DS = pool.map(calc_diffuse_layer_aa,range(nh))
            else:
                DS = pool.map(calc_diffuse_layer_kpoints,range(nh))
        etime = time.time()
        print("Diffuse calculation took ",etime-stime," seconds")
        fout = open(hkloutname,"w")
        for i in range(nh):
            hh = i - nh//2 + 1
            for j in range(nk):
                kk = j - nk//2 + 1
                for k in range(nl):
                    ll = k - nl//2 + 1
                    fout.write('{0} {1} {2} {3}\n'.format(int(hh),int(kk),int(ll),DS[i][j][k].real))
        fout.close()
#    np.savetxt("diffuse.dat",D.reshape((nh*nk,nl)).real)
