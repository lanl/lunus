   # Authors: Mike Wall & Nick Sauter
   # Date: 6/15/2017
   # Lattice methods wrapped by Alex Wolff (9/22/2017).





import libtbx.load_env
import os
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

#env_lunus.Prepend(CCFLAGS=["-g","-DUSE_OPENMP","-I/opt/local/include/libomp","-Xpreprocessor","-fopenmp"])
env_lunus.Prepend(CCFLAGS=["-g","-O3","-fopenmp","-DUSE_OPENMP","-DUSE_OFFLOAD","-Xpreprocessor"])
#env_lunus.Prepend(CCFLAGS=["-g","-O3","-fopenmp","-DUSE_OPENMP","-Xpreprocessor"])
env_lunus.Prepend(LIBS=["gomp"])
#env_lunus.Prepend(LIBS=["gomp","cudart"])
#env_lunus.Prepend(LIBPATH=["/opt/local/lib/libomp","/opt/cudatoolkit/10.0/lib64"])
#env_lunus.Prepend(LIBPATH=["/opt/cudatoolkit/10.0/lib64"])

env_lunus.StaticLibrary(target='#lib/lunus',
  source = [os.path.join(env_etc.lunus_dist,"c","lib","linitim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lcalcsim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lfreeim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lmodeim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lpunchim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lnormim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lpolarim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lscaleim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lavgrim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lwindim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lthrshim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","linitlt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lfreelt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lreadlt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lreadvtk.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lgettag.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lparsecelllt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lmattools.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lwritevtk.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lwritehkl.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lwritelt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lsetmetim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lsymlt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lsymlib.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lanisolt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lsvtools.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lspline.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lsetparamsim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lsetparamslt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","ltime.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lprocimlt.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lcloneim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lcfim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lwritecube.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lbkgsubim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lslistim.c"),
      os.path.join(env_etc.lunus_dist,"c","lib","lmulcfim.c")],
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

