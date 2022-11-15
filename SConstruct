   # Author: Mike Wall
   # Date: 11/14/2022

import os, sys
op = os.path

env = Environment()

compiler_string = env['CC']

if compiler_string[:3] == 'gcc':
  env.compiler = 'gnu'
elif compiler_string[:5] == 'clang':
  env.compiler = 'clang'

env.enable_openmp = False
env.enable_cuda = False
env.enable_kokkos = False

for key, value in ARGLIST:
  if key == 'enable-openmp':
    if bool(value):
       env.enable_openmp = True
  if key == 'enable-cuda':
    if bool(value):
       env.enable_cuda = True
  if key == 'enable-kokkos':
    if bool(value):
       env.enable_kokkos = True

env.lunus_include = Dir(os.path.join("c","include"))

CPPP = env.lunus_include

if (env.enable_openmp):
  env.Prepend(CCFLAGS=["-fopenmp", "-DUSE_OPENMP"])
  env.Prepend(SHCXXFLAGS=["-fopenmp", "-DUSE_OPENMP"])

if sys.platform.startswith('linux') and env.enable_kokkos:
  kokkos_flags = ["-DUSE_KOKKOS","-DLUNUS_NUM_JBLOCKS=16","-DLUNUS_NUM_IBLOCKS=8"]
  env.Prepend(CCFLAGS=kokkos_flags)
  env.Prepend(SHCXXFLAGS=kokkos_flags)
else:
  if (env.enable_cuda):
    env.Prepend(CCFLAGS=["-DUSE_CUDA","-DLUNUS_NUM_JBLOCKS=16","-DLUNUS_NUM_IBLOCKS=8"])

import glob
glob_str = os.path.join("c","lib","*.c")
srcfile_list = glob.glob(glob_str)
print(srcfile_list)
env.StaticLibrary(target='c/lib/lunus', source = [srcfile for srcfile in srcfile_list],CPPPATH=[CPPP,'c/lib'] )

if sys.platform.startswith('linux') and env.enable_kokkos:
  env.SConscript("lunus/kokkos/SConscript",exports={ 'env' : env_lunus })
  lunus_program_libs = ['lunus','lunus_kokkos','m']
else:
  if sys.platform.startswith('linux') and env_etc.enable_cuda:
    env.SConscript("lunus/cuda/SConscript",exports={ 'env' : env_lunus })
    lunus_program_libs = ['lunus','lunus_cuda','m']
  else:
    lunus_program_libs = ['lunus','m']

glob_str = os.path.join("c","src","*.c")
srcfile_list = glob.glob(glob_str)
for f in [srcfile for srcfile in srcfile_list]:
    env.Program('c/bin/{}'.format("lunus."+os.path.basename(f).split('.')[0]),f,LIBS=lunus_program_libs,CPPPATH=[CPPP,'c/lib'],LIBPATH='c/lib')


