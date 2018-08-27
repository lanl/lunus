from __future__ import division
print "in lunus __init__"
import boost.python
ext = boost.python.import_ext("lunus_ext")
from lunus_ext import *

class LunusDriver(object):

    def __init__(self,filename):
        import shlex
        with open(filename) as f:
            for line in shlex.split(f.read()):
                var, _, value = line.partition('=')
                if (var == "jsonlist_name"):
                    self.jsonlist_name = value
                if (var == "punchim_xmin"):
                    self.punchim_xmin = int(value)
                if (var == "punchim_xmax"):
                    self.punchim_xmax = int(value)
                if (var == "punchim_ymin"):
                    self.punchim_ymin = int(value)
                if (var == "punchim_ymax"):
                    self.punchim_ymax = int(value)
                if (var == "windim_xmin"):
                    self.windim_xmin = int(value)
                if (var == "windim_xmax"):
                    self.windim_xmax = int(value)
                if (var == "windim_ymin"):
                    self.windim_ymin = int(value)
                if (var == "windim_ymax"):
                    self.windim_ymax = int(value)
                if (var == "integration_image_type"):
                    self.integration_image_type = value
                if (var == "filterhkl"):
                    if (value == "True"):
                        self.filterhkl = True
                    else:
                        self.filterhkl = False
                if (var == "scale_inner_radius"):
                    self.scale_inner_radius = int(value)
                if (var == "scale_outer_radius"):
                    self.scale_outer_radius = int(value)
                if (var == "points_per_hkl"):
                    self.pphkl = int(value)
                if (var == "resolution" || var == "d_min"):
                    self.d_min = float(value)
                if (var == "xvectors_path"):
                    self.xvectors_path = value
                if (var == "thrshim_min"):
                    self.threshim_min = value
                if (var == "thrshim_max"):
                    self.threshim_max = value
                if (var == "modeim_bin_size"):
                    self.modeim_bin_size = int(value)
                if (var == "modeim_kernel_width"):
                    self.modeim_kernel_width = int(value)
                if (var == "polarim_polarization" || var == "polarization"):
                    self.polarim_polarization = float(value)
                if (var == "distance_mm"):
                    self.distance_mm = float(value)
                if (var == "correction_factor_scale" || var == "overall_scale_factor"):
                    self.correction_factor_scale = float(value)
