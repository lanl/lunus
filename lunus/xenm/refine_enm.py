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
#
# Rigid-body term: set RIGID_BODY=True to co-refine the rigid-body
# translational variance rigid.u alongside lambda ([lambda, rigid.u], 2 params).
# rigid.u is passed to calcenm.sh via the RIGID_U environment variable. This is
# orthogonal to the MODEL selection (works with ANMEXP or ANMEXP_DOMAIN); when on,
# it appends rigid.u as the last refined parameter.

fiter = 0

MODEL = os.environ.get("MODEL", "ANMEXP")
PDB = os.environ.get("PDB", "4wor.pdb")
REFINE_RIGID = (os.environ.get("RIGID_BODY", "False") == "True")

script_dir = os.path.dirname(os.path.abspath(__file__))
calcenm_script = os.path.join(script_dir, 'scripts', 'calcenm.sh')


def run_calcenm(lam, thresh=None, k=None, rigid_u=None):
    """Run one calcenm.sh evaluation and return the anisotropic correlation."""
    env = dict(os.environ)
    env["MODEL"] = MODEL
    if MODEL == "ANMEXP_DOMAIN":
        env["INTER_THRESH"] = repr(float(thresh))
        env["INTER_K"] = repr(float(k))
    if REFINE_RIGID:
        env["RIGID_BODY"] = "True"
        env["RIGID_U"] = repr(float(rigid_u))
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
    # rigid.u, when refined, is always the last element of x.
    rigid_u = x[-1] if REFINE_RIGID else None
    if REFINE_RIGID and rigid_u < 0:
        return 0.0
    if MODEL == "ANMEXP_DOMAIN":
        thresh, k = x[1], x[2]
        # Parameters must stay physical: positive lambda, positive contact range,
        # non-negative tier strength. Penalize excursions so Powell backs off.
        if (lam <= 0 or thresh <= 0 or k < 0):
            return 0.0
        overall, aniso = run_calcenm(lam, thresh, k, rigid_u)
        print("Function iteration = ", fiter, ", lambda = ", lam,
              ", inter.thresh = ", thresh, ", inter.k = ", k,
              ", rigid.u = ", rigid_u,
              ", overall correlation = ", overall,
              ", anisotropic correlation (target) = ", aniso)
    else:
        if (lam <= 0):
            return 0.0
        overall, aniso = run_calcenm(lam, rigid_u=rigid_u)
        print("Function iteration = ", fiter, ", lambda = ", lam,
              ", rigid.u = ", rigid_u,
              ", overall correlation = ", overall,
              ", anisotropic correlation (target) = ", aniso)
    return -aniso


if __name__ == "__main__":
    if MODEL == "ANMEXP_DOMAIN":
        x0 = [0.16, 10.0, 1.0]
    else:
        x0 = [0.16]
    if REFINE_RIGID:
        x0 = x0 + [0.4]
    res = scipy.optimize.minimize(calc_corr, x0, method='Powell', jac=None,
                                  options={'disp': True, 'maxiter': 10000})
    print("Refined parameters = ", res.x)
    xr = np.atleast_1d(res.x)
    if MODEL == "ANMEXP_DOMAIN":
        print("Refined lambda = ", xr[0],
              " inter.thresh = ", xr[1], " inter.k = ", xr[2])
    else:
        print("Refined lambda = ", xr[0])
    if REFINE_RIGID:
        print("Refined rigid.u = ", xr[-1])
    print("Best anisotropic correlation = ", -res.fun)
