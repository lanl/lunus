#!/bin/bash
module load intel/17.0.6 impi/5.1.1.109

export WORKDIR=${PWD}
cd raw
cctbx.python ${LUNUS_HOME}/scripts/setup_lunus_from_dials.py metrology=refined_experiments.json image_glob=*.img json_dir=tmpdir_common
cd ..
ls -1 raw/tmpdir_common/*.json > jsonlist.inp
sed 's@\${work_dir}@'${WORKDIR}'@g' lunus_params_unique.sh > lunus_input_params.sh
ts=`date +%s`
lunus lunus_input_params.sh
te=`date +%s`
dt_lunus=`echo "$te-$ts" | bc`
echo "TIMING: Lunus = $dt_lunus sec"
