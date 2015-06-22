/* GENLAT.C - Create a 3D lattice based upon an input list of
   diffraction images and index files.
   
   Author: Mike Wall
   Date: 2/22/95
   Version: 1.

   Usage:

   "genlat <output file > <inner radius cutoff> <outer radius cutoff>
   <x-origin> <y-origin> <z-origin> <minrange>"

   A list in the form of:

   <index file> <image file> <scale factor>

   must be supplied at standard input, terminated by '.' in the
   first character of the last line.

*/

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
    *infile,
    *outfile,
    *imagein;
  
  char
    *info_string,
    *space_ptr,
    input_line[LINESIZE],
    error_msg[LINESIZE];
  
  size_t
    *ct,
    num_values,
    num_wrote,
    num_read,
    map_index,
    index,
    i,
    j,
    k;

  int
    hh,
    kk,
    ll;
  
  DIFFIMAGE 
    *imdiff;

  LAT3D
    *lat;

  struct voxel
    *map3D,
    *voxel_data;

  struct xyzcoords
    dist,
    pos;

  float
    cell_a,
    cell_b,
    cell_c,
    cell_alpha,
    cell_beta,
    cell_gamma,
    distance,
    x_beam,
    y_beam,
    minrange,
    q_squared,
    inner_radius,
    inner_radius_sq,
    outer_radius,
    outer_radius_sq,
    scale;

  struct ijkcoords
    origin;

  /*
   * Set input line defaults:
   */
  
  scale=1;
  outfile = stdout;
  inner_radius= DEFAULT_INNER_RADIUS_LT;
  outer_radius = DEFAULT_OUTER_RADIUS_LT;
  inner_radius_sq = inner_radius*inner_radius;
  outer_radius_sq = DEFAULT_BOUND_MAX*DEFAULT_BOUND_MAX;
  origin.i = origin.j = origin.k = DEFAULT_LATTICE_ORIGIN;
  x_beam = DEFAULT_X_BEAM;
  y_beam = DEFAULT_Y_BEAM;
  distance = DEFAULT_DISTANCE_MM;
  switch(argc) {    
  case 17:
    cell_gamma = atof(argv[16]);
  case 16:
    cell_beta = atof(argv[15]);
  case 15:
    cell_alpha = atof(argv[14]);
  case 14:
    cell_c = atof(argv[13]);
  case 13:
    cell_b = atof(argv[12]);
  case 12:
    cell_a = atof(argv[11]);
  case 11:
    distance = atof(argv[10]);
  case 10:
    y_beam = atof(argv[9]);
  case 9:
    x_beam = atof(argv[8]);
  case 8:
    minrange = atof(argv[7]);
  case 7:
    origin.k = atol(argv[6]);
  case 6:
    origin.j = atol(argv[5]);
  case 5:
    origin.i = atol(argv[4]);
  case 4:
    outer_radius = atof(argv[3]);
    outer_radius_sq = outer_radius*outer_radius;
  case 3:
    inner_radius = atof(argv[2]);
    inner_radius_sq = inner_radius*inner_radius;
  case 2:
    if (strcmp(argv[1],"-") == 0) {
      outfile = stdout;
    }
    else {
      if ( (outfile = fopen(argv[1],"wb")) == NULL ) {
	printf("\nCan't open %s.\n\n",argv[1]);
	exit(0);
      }
    }
    break;
  default:
    printf("\n Usage: genlat <output file> <inner radius> "
	   "<outer radius> <x-origin> <y-origin> <z-origin>"
	   " <minrange> <x beam> <y beam> <distance>\n\n"
           " A list in the form of:\n\n"
           "   <index file> <image file> <scale factor>\n\n"
           " must be supplied at standard input, terminated by '.' in the\n"
           " first character of the last line.\n\n");
    exit(0);
  }

    /*
     * Initialize diffraction image:
     */

    if ((imdiff = linitim()) == NULL) {
      perror("Couldn't initialize diffraction image.\n\n");
      exit(0);
    }


    /*
     * Initialize lattice:
     */

    if ((lat = linitlt()) == NULL) {
      perror("Couldn't initialize lattice.n\n");
      exit(0);
    }

  /*
   * Loop through input lines until '.' is encountered.
   */

    int imnum = 1;

  while ((*gets(input_line) != '.') && (input_line != NULL)) {
    printf("%s\n",input_line);
    info_string = input_line;
    space_ptr = strchr(info_string,' ');
    *space_ptr = 0;
    if ( (infile = fopen(info_string,"r")) == NULL ) {
      printf("\nCan't open index file %s.\n\n",info_string);
      exit(0);
    }
    info_string = space_ptr+1;
    space_ptr = strchr(info_string,' ');
    *space_ptr = 0;
    if ( (imagein = fopen(info_string,"rb")) == NULL ) {
      printf("\nCan't open image %s.\n\n",info_string);
      exit(0);
    }
    info_string = space_ptr+1;
    if (info_string != "") {
      scale = atof(info_string);
    } else {
      scale = 1.;
    }


    /*
     * Read diffraction image:
     */
    
    imdiff->infile = imagein;
    if (lreadim(imdiff) != 0) {
      perror(imdiff->error_msg);
      goto CloseShop;
    }

    printf("Read the diffraction image\n");

  
    /*
     * Allocate memory:
     */
  
    map3D = (struct voxel *)malloc(sizeof(struct voxel)*imdiff->image_length);
    if (!map3D) {
      printf("\n***Unable to allocate map3D.\n");
      goto CloseShop;
    }


  
  
    /*
     * Set main defaults:
     */
  
    lat->map3D = map3D;
    lat->minrange.x = lat->minrange.y = lat->minrange.z = minrange;
    lat->origin = origin;
    lat->xbound.min = - lat->origin.i*lat->xscale;               
    lat->xbound.max = (lat->xvoxels - lat->origin.i - 1)*lat->xscale;
    lat->ybound.min = - lat->origin.j*lat->yscale; 
    lat->ybound.max = (lat->yvoxels - lat->origin.j - 1)*lat->yscale;
    lat->zbound.min = - lat->origin.k*lat->zscale; 
    lat->zbound.max = (lat->zvoxels - lat->origin.k - 1)*lat->zscale;
    lat->cell.a = cell_a;
    lat->cell.b = cell_b;
    lat->cell.c = cell_c;
    lat->cell.alpha = cell_alpha;
    lat->cell.beta = cell_beta;
    lat->cell.gamma = cell_gamma;
    //  imdiff->beam_mm.x = x_beam;
    //  imdiff->beam_mm.y = y_beam;
    //  imdiff->distance_mm = distance;

    /*
     * Allocate ct:
     */
  
    ct = (size_t *)calloc(lat->lattice_length,sizeof(size_t));
    if (!ct) {
      printf("\nNot enough room to allocate counting table ct.\n\n");
      exit(0);
    }

    /*
     * Read input file which contains the orientation matrix:
     */

    printf("Reading the input file...");
    
    imdiff->infile = infile;
    lgetmat(imdiff);

    printf("done\n");
    
    printf("U matrix\n\n");
    printf("%f %f %f\n",imdiff->u.xx,imdiff->u.xy,imdiff->u.xz);
    printf("%f %f %f\n",imdiff->u.yx,imdiff->u.yy,imdiff->u.yz);
    printf("%f %f %f\n",imdiff->u.zx,imdiff->u.zy,imdiff->u.zz);


    /*
     * Step through the image and generate the map:
     */

    printf("Stepping through the image...");
    
    map_index = 0;
    for (i=0; i<imdiff->vpixels; i++) {
      imdiff->pos.r = i; 
      for (j=0; j<imdiff->hpixels; j++) {
	imdiff->pos.c = j;
	imdiff->map3D = &map3D[map_index];
	lgensv(imdiff);
	map_index++;
      }
    }
    
    printf("done\n");

    /*
     * Scale the values to the input line scale:
     */
    
    for(map_index=0; map_index<imdiff->image_length; map_index++) {
      if (map3D[map_index].value != (float)imdiff->lattice_ignore_tag)
	map3D[map_index].value *= scale;
    }
    
    /*
     * Step through the map and fill the lattice:
     */

    for(map_index=0; map_index<imdiff->image_length; map_index++) {
      voxel_data = &map3D[map_index];
      if (map_index == 1000000)	printf("%ld : %f, %f, %f, %f\n",map_index,voxel_data->pos.x,voxel_data->pos.y,voxel_data->pos.z,voxel_data->value);/***/
      q_squared = (voxel_data->pos.x*voxel_data->pos.x +
		   voxel_data->pos.y*voxel_data->pos.y +
		   voxel_data->pos.z*voxel_data->pos.z);
      if ((inner_radius_sq < q_squared) && (outer_radius_sq >
					    q_squared)) {
	if (voxel_data->pos.x < 0) {
	  hh = (int)(voxel_data->pos.x - .5);
	} else {
	  hh = (int)(voxel_data->pos.x + .5);
	}
	if (voxel_data->pos.y < 0) {
	  kk = (int)(voxel_data->pos.y - .5);
	} else {
	  kk = (int)(voxel_data->pos.y + .5);
	}
	if (voxel_data->pos.z < 0) {
	  ll = (int)(voxel_data->pos.z - .5);
	} else {
	  ll = (int)(voxel_data->pos.z + .5);
	}
	i = (size_t) (voxel_data->pos.x + lat->origin.i + .5); 
	j = (size_t) (voxel_data->pos.y + lat->origin.j + .5); 
	k = (size_t) (voxel_data->pos.z + lat->origin.k + .5); 
	dist.x = (float)fabsf(voxel_data->pos.x - (float)hh);
	dist.y = (float)fabsf(voxel_data->pos.y - (float)kk);
	dist.z = (float)fabsf(voxel_data->pos.z - (float)ll);
	if ((voxel_data->value != lat->mask_tag) && 
	    (voxel_data->value != 0) &&
	    (dist.x > lat->minrange.x) &&
	    (dist.y > lat->minrange.y) &&
	    (dist.z > lat->minrange.z)) {
	  index = k*lat->xyvoxels + j*lat->xvoxels + i;
	  if (index >= lat->lattice_length) {
	    printf("\nTried to index lattice outside of range.\n");
	    goto CloseShop;
	  }
	  if (voxel_data->value < 0) {
	    printf("%d,%f\n",(int)map_index,(float)voxel_data->value);
	  }		  
	  if (ct[index] == 0) {
	    //	    lat->lattice[index] = (LATTICE_DATA_TYPE)imnum;
	    lat->lattice[index] = (LATTICE_DATA_TYPE)voxel_data->value;
	    ct[index]++;
	  } else {
	    //	    lat->lattice[index] = (LATTICE_DATA_TYPE)imnum;
	    lat->lattice[index] = 
	      	      (LATTICE_DATA_TYPE)(((float)ct[index]*lat->lattice[index] + 
	      				   voxel_data->value)
	      				  /(float)(ct[index]+1.));
	    ct[index]++;
	  }
	}
       }
    }

    /* 
     * Close input files:
     */
    
    fclose(infile);
    fclose(imagein);
    imnum++;
  }

  /*
   * Write lattice to output file:
   */
  
  lat->outfile = outfile;
  if (lwritelt(lat) != 0) {
    perror("Couldn't write lattice.\n\n");
    exit(0);
  }
  
 CloseShop:
  
  /*
   * Free allocated memory:
   */
  
  lfreeim(imdiff);
  lfreelt(lat);
  free((struct voxel *)map3D);
  free((size_t *)ct);
  
  /*
   * Close files:
   */
  
  fclose(outfile);
}


