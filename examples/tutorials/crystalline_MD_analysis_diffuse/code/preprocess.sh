#!/bin/bash

f=$1

if [ -z "$2" ]; then
  echo "Must supply a space group using one of the following numerical codes:"
  lunus.symlt
  exit 1
fi
#############################
# PREPROCESS DIFFUSE DATA #
#############################

# truncate data if resolution range is provided  
if [ ! -z "$3" ] && [ ! -z "$4" ]; then
  lunus.cullreslt $f ${f%.lat}_culled.lat $3 $4
fi
# symmetrize
lunus.symlt ${f%.lat}_culled.lat ${f%.lat}_culled_sym.lat $2
# calculated experimental data anisotropic component
if [ ! -z "$5" ]; then
  lunus.anisolt ${f%.lat}_culled_sym.lat ${f%.lat}_culled_sym_aniso.lat
else
  lunus.anisolt ${f%.lat}_culled_sym.lat ${f%.lat}_culled_sym_aniso.lat $5
fi


