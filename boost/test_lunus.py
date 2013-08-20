#!/Library/Frameworks/Python.framework/Versions/Current/bin/python
# Copyright Ralf W. Grosse-Kunstleve 2006. Distributed under the Boost
# Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Using the doctest module here to ensure that the results are as expected.
import sys
sys.path.append('/Users/mewall/Common/packages/lunus/boost/bin/darwin-4.5.4/debug/')
import numpy as np
#import re
#from xfel.cxi.display_spots import run_one_index_core
#from cctbx.array_family import flex
#from labelit.command_line.imagefiles import QuickImage
from lunus import *

l = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
mm = [[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]
m=[item for sublist in mm for item in sublist]
print m
r = list(l)
print r
tohkl(m,l,r)
print r
print l

