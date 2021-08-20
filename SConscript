   # Authors: Mike Wall & Nick Sauter
   # Date: 6/15/2017
   # Lattice methods wrapped by Alex Wolff (9/22/2017).





import libtbx.load_env
import os, sys
op = os.path
Import("env_base", "env_etc")

env_etc.lunus_dist = libtbx.env.dist_path("lunus")

env_etc.lunus_include = env_etc.lunus_dist
env_etc.lunus_common_includes = [
  env_etc.libtbx_include,
  env_etc.scitbx_include,
  env_etc.lunus_include,
  env_etc.boost_adaptbx_include,
  env_etc.boost_include,
  op.dirname(libtbx.env.find_in_repositories(
    relative_path="tbxx", optional=False))
]

CPPP = os.path.join(env_etc.lunus_include,"c","include")

env_lunus = env_base.Clone(SHLINKFLAGS=env_etc.shlinkflags)

replacement_ccflags = []
for f in env_lunus["CCFLAGS"]:
 if (f not in ["-ffast-math"]):
   replacement_ccflags.append(f)
env_lunus.Replace(CCFLAGS = replacement_ccflags)

#replacement_ccflags = []
#for f in env_lunus["SHCCFLAGS"]:
# if (f not in ["-ffast-math"]):
#   replacement_ccflags.append(f)
#env_lunus.Replace(SHCCFLAGS = replacement_ccflags)

#replacement_cflags = []
#for f in env_lunus["SHCFLAGS"]:
# if (f not in ["-ffast-math"]):
#   replacement_cflags.append(f)
#env_lunus.Replace(CFLAGS = replacement_cflags)

#for f in ["-g","-DUSE_OPENMP","-I/opt/local/include/libomp","-Xpreprocessor"#,"-fopenmp"]:
#  replacement_ccflags.append(f)

#env_lunus.Prepend(CCFLAGS=["-g","-DUSE_OPENMP","-I/opt/local/include/libomp","-Xpreprocessor","-fopenmp"])
#env_lunus.Prepend(SHCCFLAGS=["-g","-O2","-fopenmp","-DUSE_OPENMP","-Xpreprocessor"])
env_lunus.Prepend(CCFLAGS=["-Xpreprocessor"])
if (env_etc.have_openmp):
  env_lunus.Prepend(CCFLAGS=["-DUSE_OPENMP"])
  env_lunus.Prepend(LIBS=["gomp"])
if (env_etc.enable_cuda):
  env_lunus.Prepend(CCFLAGS=["-DUSE_CUDA","-DLUNUS_NUM_JBLOCKS=4"])
#env_lunus.Prepend(CCFLAGS=["-g","-O2","-fopenmp","-DUSE_OPENMP","-Xpreprocessor"])
#env_lunus.Prepend(CCFLAGS=["-g","-O3","-fopenmp","-DUSE_OPENMP","-DUSE_OFFLOAD","-foffload=nvptx-none","-foffload=-lm","-foffload=-fPIC","-Xpreprocessor"])
#env_lunus.Prepend(LIBS=["gomp"])
#env_lunus.Prepend(LIBS=["gomp","cudart"])
#env_lunus.Prepend(LIBPATH=["/opt/local/lib/libomp","/opt/cudatoolkit/10.0/lib64"])
#env_lunus.Prepend(LIBPATH=["/opt/cudatoolkit/10.0/lib64"])

correct_prefix = "#"+os.path.basename(env_etc.lunus_dist)

env_lunus.StaticLibrary(target='#lib/lunus',
  source = [os.path.join(correct_prefix,"c","lib",srcfile) for srcfile in [
  "linitim.c",
  "lcalcsim.c",
  "lfreeim.c",
  "lmodeim.c",
  "lofstim.c",
  "lpunchim.c",
  "lnormim.c",
  "lpolarim.c",
  "lscaleim.c",
  "lavgrcf.c",
  "lavgrim.c",
  "lwindim.c",
  "lthrshim.c",
  "linitlt.c",
  "lfreelt.c",
  "lreadlt.c",
  "lreadvtk.c",
  "lgettag.c",
  "lparsecelllt.c",
  "lmattools.c",
  "lwritevtk.c",
  "lwritehkl.c",
  "lwritelt.c",
  "lsetmetim.c",
  "lsymlt.c",
  "lsymlib.c",
  "lanisolt.c",
  "lsvtools.c",
  "lspline.c",
  "lsetparamsim.c",
  "lsetparamslt.c",
  "ltime.c",
  "lprocimlt.c",
  "lcloneim.c",
  "lcfim.c",
  "lwritecube.c",
  "lbkgsubim.c",
  "lslistim.c",
  "lmulcfim.c"]],
  CPPPATH=[CPPP] )

if (not env_etc.no_boost_python):

  Import("env_no_includes_boost_python_ext")
  env_lunus = env_no_includes_boost_python_ext.Clone()

  env_etc.include_registry.append(
    env=env_lunus,
    paths=env_etc.lunus_common_includes + [env_etc.python_include])


  env_lunus.Append(LIBS=env_etc.libm+["scitbx_boost_python","boost_python","cctbx"])

  env_etc.enable_more_warnings(env=env_lunus)
  env_lunus.SConscript("lunus/SConscript",exports={ 'env' : env_lunus })
  if sys.platform.startswith('linux') and env_etc.enable_cuda:
    env_lunus.SConscript("lunus/cuda/SConscript",exports={ 'env' : env_lunus })

