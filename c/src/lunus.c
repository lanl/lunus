/* LUNUS.C - Diffuse scattering analysis and modeling.
   
   Author: Mike Wall  
   Date: 6/13/2017
   Version: 1.
   
   "lunus <input_deck>"

   Apply lunus diffuse scattering methods according to instructions in input_deck
   
   */

#include<lunus.h>

int main(int argc, char *argv[])
{
  FILE
        *deckin,
	*imagein,
        *lunusout,
        *scaleout;
  
  char
    inputdeck[10000],
    *imageinpath,
    *lunusoutpath,
    *scaleoutpath,
    *deck,    
    *image_prefix,
    *image_suffix,
    *lunus_image_prefix,
    *scale_image_prefix,
    *lunus_image_dir,
    *raw_image_dir,
    *json_dir,
    *amatrix_format,
    *amatrix_path,
    *xvectors_path,
    *writevtk_str,
    *integration_image_type,
    *lattice_dir,
    *diffuse_lattice_prefix,
    *unit_cell,
    *spacegroup,
    *imagelist_name = NULL,
    *jsonlist_name = NULL,
    *imagelist[20000],
    *jsonlist[20000],
    error_msg[LINESIZE];

  void *buf;

  int
    writevtk,
    do_integrate = 1,
    filterhkl = 1;

  struct xyzcoords *xvectors_cctbx = NULL,*xvectors = NULL, *Hlist, *dHlist;

  IJKCOORDS_DATA *ilist, *jlist, *klist;

  struct xyzmatrix *amatrix,at[20000];

  size_t
    index,
    i,
    j,
    num_images;

  IJKCOORDS_DATA
    i0, j0, k0;

  float
    normim_tilt_x=0.0,
    normim_tilt_y=0.0,
    distance_mm=0.0,
    polarim_offset=0.0,
    polarim_polarization=1.0,
    correction_factor_scale=1.0,
    resolution;

  int
    modeim_bin_size=1,
    modeim_kernel_width,
    thrshim_min,
    thrshim_max,
    punchim_xmax,
    punchim_xmin,
    punchim_ymax,
    punchim_ymin,
    windim_xmax,
    windim_xmin,
    windim_ymax,
    windim_ymin,
    scale_inner_radius=0,
    scale_outer_radius=0,
    pphkl = 1;

  DIFFIMAGE 
    *imdiff, *imdiff_corrected = NULL, *imdiff_scale = NULL, *imdiff_scale_ref = NULL;

  LAT3D
    *lat;

  LATTICE_DATA_TYPE
    *latct = NULL;

  size_t 
    str_length,
    num_read;

  MPIVARS
    *mpiv;


  // Initialize MPI


  mpiv = (MPIVARS *)malloc(sizeof(MPIVARS));
  mpiv->argc = argc;
  mpiv->argv = argv;

  linitMPI(mpiv);

/*
 * Set input line defaults:
 */

/*
 * Read information from input line:
 */
	switch(argc) {
		case 2:
		  strcpy(inputdeck,argv[1]);
			break;
		default:
			printf("\n Usage: lunus <input deck>\n\n");
			lfinalMPI(mpiv);
			exit(0);
	}

#ifdef DEBUG
	//	printf("LUNUS: reading input deck\n");
#endif
	// Read input deck into buffer

	num_read = lreadbuf((void **)&deck,inputdeck);

#ifdef DEBUG
	//	printf("Length of input deck = %ld\n",num_read);
#endif
	// Parse input deck

	raw_image_dir=lgettag(deck,"\nraw_image_dir");

	if ((json_dir=lgettag(deck,"\njson_dir")) == NULL) {
	  json_dir = (char *)malloc(strlen(".")+1);
	  strcpy(json_dir,".");
	}

	lunus_image_dir=lgettag(deck,"\nlunus_image_dir");

	image_prefix=lgettag(deck,"\nimage_prefix");

	if ((image_suffix=lgettag(deck,"\nimage_suffix")) == NULL) {
	  image_suffix = (char *)malloc(strlen("img"+1));
	  strcpy(image_suffix,"img");
	}
	  

	lunus_image_prefix=lgettag(deck,"\nlunus_image_prefix");

	scale_image_prefix=lgettag(deck,"\nscale_image_prefix");

	if (lgettag(deck,"\npunchim_xmax") == NULL) {
	  punchim_xmax = -1;
	} else {
	  punchim_xmax = atoi(lgettag(deck,"\npunchim_xmax"));
	}

	if (lgettag(deck,"\npunchim_xmin") == NULL) {
	  punchim_xmin = -1;
	} else {
	  punchim_xmin = atoi(lgettag(deck,"\npunchim_xmin"));
	}

	if (lgettag(deck,"\npunchim_ymax") == NULL) {
	  punchim_ymax = -1;
	} else {
	  punchim_ymax = atoi(lgettag(deck,"\npunchim_ymax"));
	}

	if (lgettag(deck,"\npunchim_ymin") == NULL) {
	  punchim_ymin = -1;
	} else {
	  punchim_ymin = atoi(lgettag(deck,"\npunchim_ymin"));
	}

	if (lgettag(deck,"\nwindim_xmax") == NULL) {
	  windim_xmax = -1;
	} else {
	  windim_xmax = atoi(lgettag(deck,"\nwindim_xmax"));
	}

	if (lgettag(deck,"\nwindim_xmin") == NULL) {
	  windim_xmin = -1;
	} else {
	  windim_xmin = atoi(lgettag(deck,"\nwindim_xmin"));
	}

	if (lgettag(deck,"\nwindim_ymax") == NULL) {
	  windim_ymax = -1;
	} else {
	  windim_ymax = atoi(lgettag(deck,"\nwindim_ymax"));
	}

	if (lgettag(deck,"\nwindim_ymin") == NULL) {
	  windim_ymin = -1;
	} else {
	  windim_ymin = atoi(lgettag(deck,"\nwindim_ymin"));
	}

	if ((writevtk_str=lgettag(deck,"\nwritevtk")) == NULL) {
	  writevtk_str = (char *)malloc(strlen("False"+1));
	  strcpy(writevtk_str,"False");
	}

	if (strcmp(writevtk_str,"True")==0) {
	  writevtk=1;
	} else {
	  writevtk=0;
	}

	imagelist_name=lgettag(deck,"\nimagelist_name");

	jsonlist_name=lgettag(deck,"\njsonlist_name");

	if ((integration_image_type=lgettag(deck,"\nintegration_image_type")) == NULL) {
	  integration_image_type = (char *)malloc(strlen("raw"+1));
	  strcpy(integration_image_type,"raw");
	}

	if (lgettag(deck,"\nfilterhkl") == NULL) {
	  filterhkl = 1;
	} else if (strcmp(lgettag(deck,"\nfilterhkl"),"False")==0) {
	  filterhkl=0;
	} 
	//else {
	  //	  printf("LUNUS: filterhkl=False is only value recognized, setting filterhkl = 1.\n");
	//	}

	if (lgettag(deck,"\nscale_inner_radius") == NULL) {
	  scale_inner_radius = -1;
	} else {
	  scale_inner_radius = atoi(lgettag(deck,"\nscale_inner_radius"));
	}

	if (lgettag(deck,"\nscale_outer_radius") == NULL) {
	  scale_outer_radius = -1;
	} else {
	  scale_outer_radius = atoi(lgettag(deck,"\nscale_outer_radius"));
	}

	if (lgettag(deck,"\npphkl") == NULL) {
	  pphkl = 1;
	} else {
	  pphkl = atoi(lgettag(deck,"\npphkl"));
	}

	if (lgettag(deck,"\npoints_per_hkl") != NULL) {
	  pphkl = atoi(lgettag(deck,"\npoints_per_hkl"));
	}

	unit_cell=lgettag(deck,"\nunit_cell");

	spacegroup=lgettag(deck,"\nspacegroup");

	if (lgettag(deck,"\nresolution") == NULL) {
	  resolution = -1.;
	} else {
	  resolution = atof(lgettag(deck,"\nresolution"));
	}

	if ((xvectors_path=lgettag(deck,"\nxvectors_path")) == NULL) {
	  if (do_integrate!=0) {
	    perror("Must provide xvectors_path for integration\n");
	    exit(1);
	  }
	}

	amatrix_format=lgettag(deck,"\namatrix_format");

	if ((lattice_dir=lgettag(deck,"\nlattice_dir")) == NULL) {
	  lattice_dir = (char *)malloc(strlen(".")+1);
	  strcpy(lattice_dir,".");
	}

	if (lgettag(deck,"\ndiffuse_lattice_dir") != NULL) {
	  lattice_dir = lgettag(deck,"\ndiffuse_lattice_dir");
	}

	if ((diffuse_lattice_prefix=lgettag(deck,"\ndiffuse_lattice_prefix")) == NULL) {
	  diffuse_lattice_prefix = (char *)malloc(strlen("diffuse_lunus")+1);
	  strcpy(diffuse_lattice_prefix,"diffuse_lunus");
	}

	if(lgettag(deck,"\nthrshim_max") == NULL) {
	  thrshim_max = -1;
	} else {
	  thrshim_max = atoi(lgettag(deck,"\nthrshim_max"));
	}

	if(lgettag(deck,"\nthrshim_min") == NULL) {
	  thrshim_min = -1;
	} else {
	  thrshim_min = atoi(lgettag(deck,"\nthrshim_min"));
	}

	if(lgettag(deck,"\nmodeim_bin_size") == NULL) {
	  modeim_bin_size = -1;
	} else {
	  modeim_bin_size = atoi(lgettag(deck,"\nmodeim_bin_size"));
	}

	if(lgettag(deck,"\nmodeim_kernel_width") == NULL) {
	  modeim_kernel_width = -1;
	} else {
	  modeim_kernel_width = atoi(lgettag(deck,"\nmodeim_kernel_width"));
	}

	if(lgettag(deck,"\nnum_images") == NULL) {
	  num_images = -1;
	} else {
	  num_images = atoi(lgettag(deck,"\nnum_images"));
	}

	if(lgettag(deck,"\npolarim_offset") == NULL) {
	  polarim_offset = 0.0;
	} else {
	  polarim_offset = atof(lgettag(deck,"\npolarim_offset"));
	}

	if(lgettag(deck,"\npolarim_polarization") == NULL) {
	  polarim_polarization = 1.;
	} else {
	  polarim_polarization = atof(lgettag(deck,"\npolarim_polarization"));
	}

	if (lgettag(deck,"\ndistance_mm") == NULL) {
	  distance_mm = -1.;
	} else {
	  distance_mm = atof(lgettag(deck,"\ndistance_mm"));
	}

	if (lgettag(deck,"\ncorrection_factor_scale") == NULL) {
	  correction_factor_scale = 1.;
	} else {
	  correction_factor_scale = atof(lgettag(deck,"\ncorrection_factor_scale"));
	}

	if (lgettag(deck,"\noverall_scale_factor") != NULL) {
	  correction_factor_scale = atof(lgettag(deck,"\noverall_scale_factor"));
	}
	  
	// Print input deck values

	if (mpiv->my_id == 0) {
	  if (raw_image_dir != NULL) printf("raw_image_dir=%s\n",raw_image_dir);
	  
	  if (lunus_image_dir != NULL) printf("lunus_image_dir=%s\n",lunus_image_dir);
	  
	  if (image_prefix != NULL) printf("image_prefix=%s\n",image_prefix);
	  
	  if (image_suffix != NULL) printf("image_suffix=%s\n",image_suffix);
	  
	  if (lunus_image_prefix != NULL) printf("lunus_image_prefix=%s\n",lunus_image_prefix);
	  
	  if (scale_image_prefix != NULL) printf("scale_image_prefix=%s\n",scale_image_prefix);

	  if (imagelist_name != NULL) printf("imagelist_name=%s\n",imagelist_name);

	  if (jsonlist_name != NULL) printf("jsonlist_name=%s\n",jsonlist_name);
	  
	  printf("integration_image_type=%s\n",integration_image_type);

	  printf("lattice_dir=%s\n",lattice_dir);
	  
	  printf("diffuse_lattice_prefix=%s\n",diffuse_lattice_prefix);

	  printf("punchim_xmax=%d\n",punchim_xmax);
	  
	  printf("punchim_xmin=%d\n",punchim_xmin);
	  
	  printf("punchim_ymax=%d\n",punchim_ymax);
	  
	  printf("punchim_ymin=%d\n",punchim_ymin);
	  
	  printf("windim_xmax=%d\n",windim_xmax);
	  
	  printf("windim_xmin=%d\n",windim_xmin);
	  
	  printf("windim_ymax=%d\n",windim_ymax);
	  
	  printf("windim_ymin=%d\n",windim_ymin);
	  
	  printf("thrshim_max=%d\n",thrshim_max);
	  
	  printf("thrshim_min=%d\n",thrshim_min);
	  
	  printf("modeim_kernel_width=%d\n",modeim_kernel_width);
	  
	  printf("modeim_bin_size=%d\n",modeim_bin_size);
	  
	  printf("num_images=%ld\n",num_images);
	  
	  printf("polarim_offset=%f\n",polarim_offset);
	
	  printf("polarim_polarization=%f\n",polarim_polarization);

	  printf("resolution=%f\n",resolution);

	  if (filterhkl == 1) {
	    printf("filterhkl=True\n");
	  } else {
	    printf("filterhkl=False\n");
	  }

	  if (distance_mm>0.0) {
	    printf("distance_mm=%f\n",distance_mm);
	  } else {
	    printf("distance_mm=(obtained from image header)\n");
	  }

	  printf("correction_factor_scale=%f\n",correction_factor_scale);

	}
	/*
	 * Initialize diffraction image:
	 */


	if ((imdiff = linitim()) == NULL) {
	  perror("Couldn't initialize diffraction image.\n\n");
	  exit(0);
	}

	// Define parameters from input deck

	imdiff->punchim_upper.c = punchim_xmax;
	imdiff->punchim_lower.c = punchim_xmin;
	imdiff->punchim_upper.r = punchim_ymax;
	imdiff->punchim_lower.r = punchim_ymin;

	imdiff->window_upper.c = windim_xmax;
	imdiff->window_lower.c = windim_xmin;
	imdiff->window_upper.r = windim_ymax;
	imdiff->window_lower.r = windim_ymin;

	imdiff->upper_threshold = thrshim_max;
	imdiff->lower_threshold = thrshim_min;

	imdiff->polarization = polarim_polarization;
	imdiff->polarization_offset = polarim_offset;

	imdiff->cassette.x = normim_tilt_x;
	imdiff->cassette.y = normim_tilt_y;
	imdiff->cassette.z = 0.0;

	imdiff->mode_height = modeim_kernel_width - 1;
	imdiff->mode_width = modeim_kernel_width - 1;
	imdiff->mode_binsize = modeim_bin_size;

	imdiff->mask_inner_radius = scale_inner_radius;
	imdiff->mask_outer_radius = scale_outer_radius;

	// Initialize other images

	imdiff_corrected = linitim();
	imdiff_scale = linitim();
	imdiff_scale_ref = linitim();

	// Process all of the images

	/*	int n;

	n = num_images/mpiv->num_procs;
	if (num_images % mpiv->num_procs != 0) {
	  n++;
	} 

	int ib = mpiv->my_id*n+1;
	int ie = (mpiv->my_id+1)*n;
	*/
	int ct=0;

	if (mpiv->my_id == 0) {

	if (imagelist_name == NULL && jsonlist_name == NULL) {

	  if (raw_image_dir == NULL || image_prefix == NULL || image_suffix == NULL) {
	    perror("Can't generate image list due to NULL value of one or more filename components.\n");
	    exit(1);
	  }

	  printf("No imagelist or jsonlist provided. Generating image list using loop scheme.\n");

	  size_t ii=0;

	  for (i=0;i<num_images;i++) {

	    // Read amatrix

	    if ((readAmatrix(&at[ii],amatrix_format,i) == -1)) {
	      printf("Missing amatrix file %s. Skipping frame %d.\n",amatrix_path,i);
	    } else {
	      
	      
	      //	printf("i=%d, imagelist[0] = %s\n",i,imagelist[0]);
	      
	      //	exit(1);
	      
	      //	for (i=ib;i<=ie&&i<=num_images;i++) {
	      
	      str_length = snprintf(NULL,0,"%s/%s_%05d.%s",raw_image_dir,image_prefix,i+1,image_suffix);
	      
	      imagelist[ii] = (char *)malloc(str_length+1);
	      
	      sprintf(imagelist[ii],"%s/%s_%05d.%s",raw_image_dir,image_prefix,i+1,image_suffix);
	      
	      ii++;
	    }
	    
	  }
	  num_images = ii;

	} else if (jsonlist_name != NULL) {

	  FILE *f;

	  if ((f = fopen(jsonlist_name,"r")) == NULL) {
	    printf("Can't open %s.\n",jsonlist_name);
	    exit(1);
	  }
	  
	  size_t bufsize = LINESIZE;
	  char *buf;

	  i = 0;

	  buf = (char *)malloc(LINESIZE+1);

	  int chars_read;

	  while ((chars_read = getline(&buf,&bufsize,f)) != -1) {

	    buf[chars_read-1]=0;

	    char *json_name;

	    str_length = snprintf(NULL,0,"%s/%s",json_dir,buf);

	    json_name = (char *)malloc(str_length+1);

	    sprintf(json_name,"%s/%s",json_dir,buf);
	    
	    if ((imagelist[i] = readExptJSON(&at[i],json_name)) == NULL) {
	      printf("Skipping %s, unable to read\n",json_name);
	    } else {
	      //	      printf("%s\n",imagelist[i]);
	      i++;
	    }

	  }

	  num_images = i;

	} else {

	  FILE *f;

	  if ((f = fopen(imagelist_name,"r")) == NULL) {
	    printf("Can't open %s.\n",imagelist_name);
	    exit(1);
	  }
	  
	  size_t bufsize = LINESIZE;
	  char *buf;

	  i = 0;

	  buf = (char *)malloc(LINESIZE+1);

	  int chars_read;

	  while ((chars_read = getline(&buf,&bufsize,f)) != -1) {

	    // Read amatrix

	    if ((readAmatrix(&at[i],amatrix_format,i+1) == -1)) {
	      printf("Missing amatrix file %s. Skipping frame %d.\n",amatrix_path,i);
	    } else {
	      
	      
	      buf[chars_read-1]=0;
	      
	      str_length = snprintf(NULL,0,"%s/%s",raw_image_dir,buf);
	      
	      imagelist[i] = (char *)malloc(str_length+1);
	      
	      sprintf(imagelist[i],"%s/%s",raw_image_dir,buf);
	      
	      //	    printf("%s\n",imagelist[i]);
	      
	      i++;
	    }
	    
	  }
	  num_images = i;
	}
	}
	lbcastBufMPI((void *)&num_images,sizeof(size_t),0,mpiv);
	int il_sz[num_images];
	if (mpiv->my_id == 0) {
	  for (i=0;i<num_images;i++) {
	    il_sz[i] = strlen(imagelist[i])+1;
	  }
	}
	lbcastBufMPI((void *)&il_sz,sizeof(int)*num_images,0,mpiv);
	for (i=0;i<num_images;i++) {
	  if (mpiv->my_id != 0) {
	    imagelist[i] = (char *)malloc(il_sz[i]);
	  }
	  lbcastBufMPI((void *)imagelist[i],il_sz[i],0,mpiv);
	}

	lbcastBufMPI((void *)&at,sizeof(struct xyzmatrix)*num_images,0,mpiv);	

	// Define the integration lattices and associated variables
        //      if performing integration

	//	printf("entering unit cell\n");

	if (do_integrate!=0) {
	  lat = linitlt();
	  if (unit_cell == NULL) {
	    float a,b,c,alpha,beta,gamma,adotb,adotc,bdotc;
	    struct xyzmatrix a0;
	    a0 = at[0];
	    a = sqrtf(a0.xx*a0.xx+a0.xy*a0.xy+a0.xz*a0.xz);
	    b = sqrtf(a0.yx*a0.yx+a0.yy*a0.yy+a0.yz*a0.yz);
	    c = sqrtf(a0.zx*a0.zx+a0.zy*a0.zy+a0.zz*a0.zz);
	    adotb = a0.xx*a0.yx + a0.xy*a0.yy + a0.xz*a0.yz;
	    adotc = a0.xx*a0.zx + a0.xy*a0.zy + a0.xz*a0.zz;
	    bdotc = a0.yx*a0.zx + a0.yy*a0.zy + a0.yz*a0.zz;
	    alpha = acosf(bdotc/b/c)*180./PI;
	    beta = acosf(adotc/a/c)*180./PI;
	    gamma = acosf(adotb/a/b)*180./PI;
	    str_length = snprintf(NULL,0,"%f,%f,%f,%f,%f,%f",a,b,c,alpha,beta,gamma);
	    unit_cell = (char *)malloc(str_length+1);
	    sprintf(unit_cell,"%f,%f,%f,%f,%f,%f",a,b,c,alpha,beta,gamma);
	    printf("Calculated unit cell from first amatrix=%s\n",unit_cell);
	  }
	  strcpy(lat->cell_str,unit_cell);
	  lparsecelllt(lat);
	  lat->cell.a *= pphkl;
	  lat->cell.b *= pphkl;
	  lat->cell.c *= pphkl;
	  lat->xvoxels = ((int)(lat->cell.a/resolution)+1)*2;
	  lat->yvoxels = ((int)(lat->cell.b/resolution)+1)*2;
	  lat->zvoxels = ((int)(lat->cell.c/resolution)+1)*2;
	  i0 = (IJKCOORDS_DATA)(lat->xvoxels/2. - 1.);
	  j0 = (IJKCOORDS_DATA)(lat->yvoxels/2. - 1.);
	  k0 = (IJKCOORDS_DATA)(lat->zvoxels/2. - 1.);
	  lat->xscale = 1./lat->cell.a;
	  lat->yscale = 1./lat->cell.b;
	  lat->zscale = 1./lat->cell.c;
	  lat->xbound.min = -i0*lat->xscale;
	  lat->ybound.min = -j0*lat->yscale;
	  lat->zbound.min = -k0*lat->zscale;
	  lat->xbound.max = lat->xbound.min + ((float)lat->xvoxels-1)*lat->xscale;
	  lat->ybound.max = lat->ybound.min + ((float)lat->yvoxels-1)*lat->yscale;
	  lat->zbound.max = lat->zbound.min + ((float)lat->zvoxels-1)*lat->zscale;
	  lat->origin.i = (IJKCOORDS_DATA)(-lat->xbound.min/lat->xscale + .5);
	  lat->origin.j = (IJKCOORDS_DATA)(-lat->ybound.min/lat->yscale + .5);
	  lat->origin.k = (IJKCOORDS_DATA)(-lat->zbound.min/lat->zscale + .5);
	  lat->xyvoxels = lat->xvoxels * lat->yvoxels;
	  lat->lattice_length = lat->xyvoxels*lat->zvoxels;
	  //	  if (lat->lattice != NULL) free(lat->lattice);
	  lat->lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,sizeof(LATTICE_DATA_TYPE));
	  //	  if (latct != NULL) free(latct);
	  latct = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,sizeof(LATTICE_DATA_TYPE));
	}

	for (i=mpiv->my_id+1;i<=num_images;i=i+mpiv->num_procs) {

	  //	  str_length = snprintf(NULL,0,"%s/%s_%05d.%s",raw_image_dir,image_prefix,i,image_suffix);

	  //	  imageinpath = (char *)malloc(str_length+1);

	  //	  sprintf(imageinpath,"%s/%s_%05d.%s",raw_image_dir,image_prefix,i,image_suffix);

#ifdef DEBUG
	  //	  printf("imageinpath = %s\n",imageinpath);
#endif



	  /*
	   * Read diffraction image:
	   */
	  
	  if ( (imagein = fopen(imagelist[i-1],"rb")) == NULL ) {
	    printf("Can't open %s.",imagelist[i-1]);
	    exit(0);
	  }

	  imdiff->infile = imagein;
	  if (lreadim(imdiff) != 0) {
	    perror(imdiff->error_msg);
	    exit(0);
	  }

	  fclose(imagein);

	  // Define image parameters from input deck
	  
	  imdiff->punchim_upper.c = punchim_xmax;
	  imdiff->punchim_lower.c = punchim_xmin;
	  imdiff->punchim_upper.r = punchim_ymax;
	  imdiff->punchim_lower.r = punchim_ymin;
	  
	  imdiff->window_upper.c = windim_xmax;
	  imdiff->window_lower.c = windim_xmin;
	  imdiff->window_upper.r = windim_ymax;
	  imdiff->window_lower.r = windim_ymin;
	  
	  imdiff->upper_threshold = thrshim_max;
	  imdiff->lower_threshold = thrshim_min;
	  
	  imdiff->polarization = polarim_polarization;
	  imdiff->polarization_offset = polarim_offset;

	  if (distance_mm>0.0) {
	    imdiff->distance_mm = distance_mm;
	  }
	  
	  imdiff->cassette.x = normim_tilt_x;
	  imdiff->cassette.y = normim_tilt_y;
	  imdiff->cassette.z = 0.0;
	  
	  imdiff->mode_height = modeim_kernel_width - 1;
	  imdiff->mode_width = modeim_kernel_width - 1;
	  imdiff->mode_binsize = modeim_bin_size;
	  
	  imdiff->mask_inner_radius = scale_inner_radius;
	  imdiff->mask_outer_radius = scale_outer_radius;
	  
	  // Apply masks

	  lpunchim(imdiff);
	  lwindim(imdiff);
	  lthrshim(imdiff);

	  // Mode filter to create image to be used for scaling

	  lcloneim(imdiff_scale,imdiff);

	  lmodeim(imdiff_scale);

	  // Write mode filtered image

	  /*
	  str_length = snprintf(NULL,0,"%s/%s_%05d.%s",lunus_image_dir,scale_image_prefix,i,image_suffix);

	  scaleoutpath = (char *)malloc(str_length+1);

	  sprintf(scaleoutpath,"%s/%s_%05d.%s",lunus_image_dir,scale_image_prefix,i,image_suffix);

	  if ( (scaleout = fopen(scaleoutpath,"wb")) == NULL ) {
	    printf("Can't open %s.",scaleoutpath);
	    exit(1);
	  }

	  imdiff_scale->outfile = scaleout;
	  if(lwriteim(imdiff_scale) != 0) {
	    perror(imdiff_scale->error_msg);
	    exit(1);
	  }

	  fclose(scaleout);
	  */

	  // Calculate correction factor

	  imdiff->correction[0]=correction_factor_scale;
	  lcfim(imdiff);	  

	  // Write masked and corrected image

	  lcloneim(imdiff_corrected,imdiff);
	  if (lmulcfim(imdiff_corrected) != 0) {
	    perror(imdiff_corrected->error_msg);
	    exit(1);
	  }

	  /*

	  str_length = snprintf(NULL,0,"%s/%s_%05d.%s",lunus_image_dir,lunus_image_prefix,i,image_suffix);

	  lunusoutpath = (char *)malloc(str_length+1);

	  sprintf(lunusoutpath,"%s/%s_%05d.%s",lunus_image_dir,lunus_image_prefix,i,image_suffix);

	  if ( (lunusout = fopen(lunusoutpath,"wb")) == NULL ) {
	    printf("Can't open %s.",lunusoutpath);
	    exit(1);
	  }

	  imdiff_corrected->outfile = lunusout;
	  if(lwriteim(imdiff_corrected) != 0) {
	    perror(imdiff_corrected->error_msg);
	    exit(1);
	  }

	  fclose(lunusout);
	  */

	  //	  imdiff->correction[0]=1.;
	  //	  lcfim(imdiff);

	  if (do_integrate!=0) {
	    
	    // Set up common variables on the first pass

	    if (ct == 0) {
	      // Reference image for scaling
	      lcloneim(imdiff_scale_ref,imdiff_scale);
	      //	      lbarrierMPI(mpiv);
	      //	      printf("Barrier 1 passed\n");
	      // Use the rank 0 image
	      lbcastImageMPI(imdiff_scale_ref->image,imdiff_scale_ref->image_length,0,mpiv);
	      //	      lbarrierMPI(mpiv);
	      //	      printf("Barrier 2 passed\n");
	      // Read the xvectors on rank 0
	      if (mpiv->my_id == 0) {
		num_read = lreadbuf((void **)&xvectors_cctbx,xvectors_path);
		if (num_read != 3*imdiff->image_length*sizeof(float)) {
		  perror("LUNUS: Number of xvectors differs from number of pixels in image.\n");
		  exit(1);
		}
#ifdef DEBUG		
		printf("SAMPLES\n");
		for (j=50000;j<50010;j++) {
		  printf("(%f, %f, %f): %d\n",xvectors_cctbx[j].x,xvectors_cctbx[j].y,xvectors_cctbx[j].z, imdiff_corrected->image[j]);
		}
		printf("\n");
#endif		
	      // Reorder the xvectors (transpose)

		index = 0;

		if (xvectors != NULL) free(xvectors);

		xvectors = (struct xyzcoords *)malloc(num_read);		

		size_t k;

		for (j=0; j<imdiff->vpixels; j++) {
		  for (k=0; k<imdiff->hpixels; k++) {
		    // The following conditional is needed to prevent a segfault-inducing Intel 18.X optimization error:
		    //		    if (index == 0) printf("");
		    //		    xvectors[k*imdiff->vpixels + j] = xvectors_cctbx[index];
		    xvectors[index] = xvectors_cctbx[k*imdiff->vpixels + j];
		    index++;
		  }
		}
	      }

	      // Broadcast the xvectors to other ranks
	      lbcastBufMPI((void *)&num_read,sizeof(size_t),0,mpiv);
	      if (mpiv->my_id != 0) {
		if (xvectors != NULL) free(xvectors);
		xvectors = (struct xyzcoords *)malloc(num_read);
	      }
	      lbcastBufMPI((void *)xvectors,num_read,0,mpiv);
	      //	      Hlist = (struct xyzcoords *)malloc(num_read);
	      //	      dHlist = (struct xyzcoords *)malloc(num_read);
	      //	      ilist = (IJKCOORDS_DATA *)malloc(num_read);
	      //	      jlist = (IJKCOORDS_DATA *)malloc(num_read);
	      //	      klist = (IJKCOORDS_DATA *)malloc(num_read);
	    }
	  
	    // Calculate the image scale factor

	    lscaleim(imdiff_scale_ref,imdiff_scale);
	    float this_scale_factor = imdiff_scale_ref->rfile[0];
	    printf("Image %d scale factor, error = %f, %f\n",i,imdiff_scale_ref->rfile[0],imdiff_scale_ref->rfile[1]);


	    // Calculate the rotated and scaled xvectors, yielding Miller indices

#ifdef DEBUG
	    /*	    if (i == 1) {
	    for (j = 50000;j<50010;j++) {
	      printf("(%f %f %f) ",xvectors[j].x,xvectors[j].y,xvectors[j].z);
	    }
	    printf("\n");
	    for (j=50000;j<50010;j++) {	      
	      printf("%d ",imdiff_scale->image[j]);
	    }
	    printf("\n");
	    }
	    */
#endif

	    // Collect the image data into the lattice

	    size_t data_added=0;
	    struct xyzcoords H, dH;
	    IJKCOORDS_DATA ii,jj,kk;
	    index = 0;
	    for (j=0; j<imdiff->image_length; j++) {
	      H = lmatvecmul(at[i-1], xvectors[j]);
#ifdef DEBUG
	      if (j<10) {
		printf("Image %d, H[%d] = (%f, %f, %f)\n",i,j,H.x,H.y,H.z);
	      }
#endif
	      dH.x = fabs(H.x - roundf(H.x));
	      dH.y = fabs(H.y - roundf(H.y));
	      dH.z = fabs(H.z - roundf(H.z));
	      if (filterhkl==0 || dH.x>=0.25 || dH.y>=0.25 || dH.z>=0.25) {
		ii = (IJKCOORDS_DATA)roundf(H.x*(float)pphkl) + i0;
		jj = (IJKCOORDS_DATA)roundf(H.y*(float)pphkl) + j0;
		kk = (IJKCOORDS_DATA)roundf(H.z*(float)pphkl) + k0;
		if (ii>=0 && ii<lat->xvoxels && jj>=0 && jj<lat->yvoxels &&
		    kk>=0 && kk<lat->zvoxels && imdiff_scale->image[index]>0 &&
		    imdiff_scale->image[index]!=imdiff->ignore_tag) {
		  size_t latidx = kk*lat->xyvoxels + jj*lat->xvoxels + ii;
		  if (strcmp(integration_image_type,"raw")==0) {
		    lat->lattice[latidx] += 
		      (LATTICE_DATA_TYPE)imdiff->image[index]
		      * imdiff->correction[index]
		      * this_scale_factor;
		  }
		  if (strcmp(integration_image_type,"corrected")==0) {
		    lat->lattice[latidx] += 
		      (LATTICE_DATA_TYPE)imdiff_corrected->image[index]
		      * this_scale_factor;
		  }
		  if (strcmp(integration_image_type,"scale")==0) {
		    lat->lattice[latidx] += 
		      (LATTICE_DATA_TYPE)imdiff_scale->image[index]
		      * imdiff->correction[index]
		      * this_scale_factor;
		  }
		  latct[latidx] += 1.;
		  data_added += 1;
		}
	      }
	      index++;
	    }
#ifdef DEBUG
	    printf("data_added = %ld\n",data_added);
#endif
	  }
	  ct++;
	}

#ifdef DEBUG
	if (do_integrate != 0) {
	  // Count number of data points in the lattice
	  int num_data_points=0;
	  float sum_data_points=0.0;
	  
	  for (j=0;j<lat->lattice_length;j++) {
	    if (latct[j] != 0) {
	      if (lat->lattice[j]/latct[j] < 32767.) {
		num_data_points++;
		sum_data_points += lat->lattice[j]/latct[j];
	      }
	    }
	  }
	  printf("num_data_points=%d, mean_data_points=%f\n",num_data_points,sum_data_points/(float)num_data_points);
	}
#endif
	if (do_integrate != 0) {

	  // Merge the data and counts

	  LATTICE_DATA_TYPE *latsum, *latctsum;
	  
	  latsum = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,sizeof(LATTICE_DATA_TYPE));
	  latctsum = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,sizeof(LATTICE_DATA_TYPE));
	  
	  lreduceSumLatticeMPI(lat->lattice,latsum,lat->lattice_length,0,mpiv);
	  lreduceSumLatticeMPI(latct,latctsum,lat->lattice_length,0,mpiv);
	  
	  // Calculate the mean on the root rank and output the result
	  
	  if (mpiv->my_id == 0) {
	    for (j=0; j<lat->lattice_length; j++) {
	      if (latctsum[j] != 0) {
		lat->lattice[j] = latsum[j]/latctsum[j];
		if isnan(lat->lattice[j]) lat->lattice[j] = lat->mask_tag;
	      } else {
		lat->lattice[j] = lat->mask_tag;
	      }
	    }
	    
	    // output the result
	    
	    str_length = snprintf(NULL,0,"%s/%s.lat",lattice_dir,diffuse_lattice_prefix);
	    
	    char *latticeoutpath;
	    
	    latticeoutpath = (char *)malloc(str_length+1);
	    
	    sprintf(latticeoutpath,"%s/%s.lat",lattice_dir,diffuse_lattice_prefix);
	    
	    //	  printf("imageinpath = %s\n",imageinpath);
	    
	    FILE *latticeout;
	    
	    if ( (latticeout = fopen(latticeoutpath,"wb")) == NULL ) {
	      printf("Can't open %s.\n",latticeoutpath);
	      exit(1);
	    }
	    lat->outfile=latticeout;
	    lwritelt(lat);
	    fclose(latticeout);
	    // Write .vtk if requested

	    if (writevtk != 0) {

	      str_length = snprintf(NULL,0,"%s/%s.vtk",lattice_dir,diffuse_lattice_prefix);
	    
	      char *vtkoutpath;
	      
	      vtkoutpath = (char *)malloc(str_length+1);
	      
	      sprintf(vtkoutpath,"%s/%s.vtk",lattice_dir,diffuse_lattice_prefix);
	      
	      //	  printf("imageinpath = %s\n",imageinpath);
	      
	      FILE *vtkout;
	      
	      if ( (vtkout = fopen(vtkoutpath,"w")) == NULL ) {
		printf("Can't open %s.\n",vtkoutpath);
		exit(1);
	      }
	      
	      lat->outfile=vtkout;
	      if (spacegroup != NULL) {
		strcpy(lat->space_group_str,spacegroup);
	      } else {
		strcpy(lat->space_group_str,"Unknown");
	      }
	      lwritevtk(lat);
	    }
	  }
	}
	lfinalMPI(mpiv);

}

