import numpy as np
import scipy.sparse.linalg
import time
import sys

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

  wN = np.array([wmax+dw,wmax+2*dw,wmax+3*dw],dtype=np.complex)

  Nv = np.zeros((len(H)//3,3),dtype=np.complex)

  Nv1D = Nv.reshape(len(H))

  N = np.zeros((len(H),len(H)),dtype=np.complex)
  Ninv = np.zeros((len(H),len(H)),dtype=np.complex)

  for k in range(3):
    Nv.fill(0.)
    Nv[:,k]=np.sqrt(1./len(Nv))
    N += wN[k]*np.outer(Nv1D,Nv1D)
    Ninv += 1./wN[k]*np.outer(Nv1D,Nv1D)

  Hm = H + N

  Hminv = np.linalg.inv(Hm)
  Hpinv = Hminv - Ninv
  return Hpinv

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
        if (a == "input.X"):
            Xname = input_dict[a]
        if (a == "output.Xtilde"):
            Xtildename = input_dict[a]

    X = np.load(Xname)
    Xtilde = pinv(X)

    np.save(Xtildename,Xtilde)
