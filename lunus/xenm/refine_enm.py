import numpy as np
import scipy.optimize
import subprocess, shlex
import os

fiter = 0

def calc_corr(x):
    global fiter
    fiter = fiter + 1
    if (x[0]>0):
        f = open("this_correl.txt","w")
        # Find calcenm.sh relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calcenm_script = os.path.join(script_dir, 'scripts', 'calcenm.sh')
        command = '{0} 4wor.pdb {1}'.format(calcenm_script, x[0])
        call_params = shlex.split(command)
        subprocess.call(call_params,stdout=f)
        f.close()
        f = open("this_correl.txt")
        flines = f.readlines()
        clist = flines[-1].split(' ')
        f.close()
    else:
        clist = [0,0]
    if (np.isnan(float(clist[1]))):
        print("Changing value from nan to 0")
        clist[1] = "0"
    print("Function iteration = ",fiter,", lambda = ",x[0],", overall correlation = ",clist[0],", anisotropic correlation (target) = ",clist[1])
    return -float(clist[1])

if __name__=="__main__":
    x0 = [0.16]
    res = scipy.optimize.minimize(calc_corr,x0,method='Powell',jac=None,options={'disp': True,'maxiter': 10000})
    lam = res.x[0]
    print("Refined lambda = ",lam)
