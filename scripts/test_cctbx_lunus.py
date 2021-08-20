from __future__ import print_function
import future,six
import lunus
import dxtbx
from dxtbx.format.FormatCBFMini import FormatCBFMini
from dials.array_family import flex
from lunus import LunusDIFFIMAGE
import os
import numpy as np

import sys

args = sys.argv[1:] 

# Input image file
try:
    idx = [(a.find("input")==0) for a in args].index(True)
except ValueError:
    raise(ValueError,"Input image file must be specified using input=.")
else:
    inp_name = args.pop(idx).split("=")[1]
# Output image file
try:
    idx = [(a.find("output")==0) for a in args].index(True)
except ValueError:
    raise(ValueError,"Output image file must be specified using output=.")
else:
    out_name = args.pop(idx).split("=")[1]

# Reference image file
try:
    idx = [(a.find("reference")==0) for a in args].index(True)
except ValueError:
    ref_name = None
else:
    ref_name = args.pop(idx).split("=")[1]

img = dxtbx.load(inp_name)

detector = img.get_detector()
beam = img.get_beam()
gonio = img.get_goniometer()
scan = img.get_scan()

A = LunusDIFFIMAGE()
data = img.get_raw_data()
data_np = data.as_numpy_array()
hist=np.histogram(data,range=(0,50),bins=50,density=False)
print(hist[0:50])
print(data[0])
A.set_image(data)
deck = '''
#lunus input deck
punchim_xmin=1203
punchim_ymin=1250
punchim_xmax=2459
punchim_ymax=1314
windim_xmin=100
windim_ymin=100
windim_xmax=2362
windim_ymax=2426
thrshim_min=0
thrshim_max=30000
modeim_bin_size=1
modeim_kernel_width=20

'''
A.LunusSetparamsim(deck)
A.LunusPunchim()
A.LunusWindim()
A.LunusThrshim()
A.LunusModeim()
print("Mode filter finished.")
data2 = A.get_image();
if not ref_name is None:
    ref = dxtbx.load(ref_name)
    ref_data_np = ref.get_raw_data().as_numpy_array().flatten()
    data2_np = data2.as_numpy_array().flatten()
#    data2_np_rand = np.random.random_sample((len(data2_np),))
    cc = np.corrcoef(ref_data_np,data2_np)[0,1]
#dxtbx.format.FormatCBFMini.FormatCBFMini.as_file(detector,beam,gonio,scan,data,path,header_convention="GENERIC_MINI",det_type="GENERIC")
FormatCBFMini.as_file(detector,beam,gonio,scan,data2,out_name)
if np.abs(cc - 1.0) < 0.00001:
    sys.exit(0)
else: 
    sys.exit(1)
