"""
Report the FFT grid dimensions phenix/cctbx uses for structure factor calculation.
"""
import iotbx.pdb

pdb_inp = iotbx.pdb.input(file_name='P1test_B20.pdb')
xrs = pdb_inp.xray_structure_simple()
xrs.scattering_type_registry(table='it1992')

fc = xrs.structure_factors(d_min=2.0, algorithm='fft').f_calc()
cell = xrs.unit_cell()
a, b, c = cell.parameters()[:3]

for grf in (1./3, 1./4, 1./5):
    fft_map = fc.fft_map(resolution_factor=grf)
    nr = fft_map.n_real()
    pixel_a = a / nr[0]
    pixel_b = b / nr[1]
    pixel_c = c / nr[2]
    print("grid_resolution_factor=%.3f: grid=%dx%dx%d  pixel=%.3fx%.3fx%.3f A" % (
          grf, nr[0], nr[1], nr[2], pixel_a, pixel_b, pixel_c))
