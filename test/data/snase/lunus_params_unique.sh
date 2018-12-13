#json input file list
jsonlist_name=jsonlist.inp
# integration
xvectors_path=${work_dir}/raw/tmpdir_common/xvectors.bin
diffuse_lattice_dir=${work_dir}/lattices
diffuse_lattice_prefix=snc_lunus_json
lunus_image_dir=${work_dir}/proc
resolution=1.6
points_per_hkl=1
overall_scale_factor=1.
filterhkl=True
writevtk=True
#image mask
thrshim_max=10000
thrshim_min=1
punchim_xmax=556
punchim_xmin=510
punchim_ymax=517
punchim_ymin=473
windim_xmax=984
windim_xmin=40
windim_ymax=984
windim_ymin=40
#mode filter
modeim_bin_size=1
modeim_kernel_width=15
#polarization correction
distance_mm=57.42
polarim_offset=0.0
polarim_polarization=0.93
#image scale factor parameters, for merge
scale_inner_radius=100
scale_outer_radius=500
