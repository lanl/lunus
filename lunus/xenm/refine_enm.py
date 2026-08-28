import numpy as np
import scipy.optimize
import subprocess, shlex
import os

# Refine ENM parameters against experimental diffuse scattering by maximizing the
# anisotropic correlation reported by scripts/calcenm.sh.
#
# Two modes:
#   MODEL=ANMEXP (default)  -- refine the single intramolecular decay length
#                              lambda (1 parameter).
#   MODEL=ANMEXP_DOMAIN     -- refine the two-tier domain-coupled model over
#                              [lambda, inter.thresh, inter.k] (3 parameters):
#                              intramolecular ANMEXP decay length, inter-molecular
#                              logistic contact center, and the intra/inter tier
#                              strength ratio.
#
# The model is selected with the MODEL environment variable, which is passed
# straight through to calcenm.sh; for ANMEXP_DOMAIN the thresh/k parameters are
# passed to calcenm.sh via the INTER_THRESH / INTER_K environment variables.

fiter = 0

MODEL = os.environ.get("MODEL", "ANMEXP")
PDB = os.environ.get("PDB", "4wor.pdb")

script_dir = os.path.dirname(os.path.abspath(__file__))
calcenm_script = os.path.join(script_dir, 'scripts', 'calcenm.sh')


def run_calcenm(lam, thresh=None, k=None):
    """Run one calcenm.sh evaluation and return the anisotropic correlation."""
    env = dict(os.environ)
    env["MODEL"] = MODEL
    if MODEL == "ANMEXP_DOMAIN":
        env["INTER_THRESH"] = repr(float(thresh))
        env["INTER_K"] = repr(float(k))
    f = open("this_correl.txt", "w")
    command = '{0} {1} {2}'.format(calcenm_script, PDB, lam)
    call_params = shlex.split(command)
    subprocess.call(call_params, stdout=f, env=env)
    f.close()
    with open("this_correl.txt") as f:
        flines = f.readlines()
    clist = flines[-1].split(' ')
    if (np.isnan(float(clist[1]))):
        print("Changing value from nan to 0")
        clist[1] = "0"
    return float(clist[0]), float(clist[1])


def calc_corr(x):
    global fiter
    fiter = fiter + 1
    lam = x[0]
    if MODEL == "ANMEXP_DOMAIN":
        thresh, k = x[1], x[2]
        # Parameters must stay physical: positive lambda, positive contact range,
        # non-negative tier strength. Penalize excursions so Powell backs off.
        if (lam <= 0 or thresh <= 0 or k < 0):
            return 0.0
        overall, aniso = run_calcenm(lam, thresh, k)
        print("Function iteration = ", fiter, ", lambda = ", lam,
              ", inter.thresh = ", thresh, ", inter.k = ", k,
              ", overall correlation = ", overall,
              ", anisotropic correlation (target) = ", aniso)
    else:
        if (lam <= 0):
            return 0.0
        overall, aniso = run_calcenm(lam)
        print("Function iteration = ", fiter, ", lambda = ", lam,
              ", overall correlation = ", overall,
              ", anisotropic correlation (target) = ", aniso)
    return -aniso


if __name__ == "__main__":
    if MODEL == "ANMEXP_DOMAIN":
        x0 = [0.16, 10.0, 1.0]
    else:
        x0 = [0.16]
    res = scipy.optimize.minimize(calc_corr, x0, method='Powell', jac=None,
                                  options={'disp': True, 'maxiter': 10000})
    print("Refined parameters = ", res.x)
    if MODEL == "ANMEXP_DOMAIN":
        print("Refined lambda = ", res.x[0],
              " inter.thresh = ", res.x[1], " inter.k = ", res.x[2])
    else:
        print("Refined lambda = ", res.x[0])
    print("Best anisotropic correlation = ", -res.fun)
