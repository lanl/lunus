#!/bin/bash -c
#module load gcc/9.1.0
#module load cuda/9.2.148
source ../../../../build/setpaths.sh
libtbx.python ../../scripts/test_cctbx_lunus.py input=../data/TestImageCBF_00001.cbf output=TestCCTBXLunus_00001.cbf reference=../ref/TestCCTBXLunusRef_00001.cbf
if [ $? == 0 ]; then
    echo "Test Passed"
else
    echo "Test Failed"
fi
mv TestCCTBXLunus_00001.cbf ../proc/TestCCTBXLunus_00001.cbf
