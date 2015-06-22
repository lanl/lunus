/* LAVGRLT.C - Calculate the average voxel value vs. radius for an 
              input lattice
   
   Author: Mike Wall
   Date: 3/1/95
   Version: 1.
   
   */

#include<mwmask.h>

int lavgrlt(LAT3D *lat)
{
  size_t
    *ct,
    i,
    j,
    k,
    r,
    index = 0;

  int
    return_value = 0;
  
  struct ijkcoords rvec;
  
  float
    rscale;

  struct xyzcoords
    rfloat;

  /*
   * Allocate counting array:
   */

  ct = (size_t *)calloc(MAX_RFILE_LENGTH,sizeof(size_t));
  if (!ct) {
    sprintf(lat->error_msg,"\nLAVGRLT: Couldn't allocate counting "
	    "array ct.\n\n");
    return_value = 1;
    goto CloseShop;
  }

  rscale = (lat->xscale*lat->xscale + lat->yscale*lat->yscale +
		 lat->zscale*lat->zscale);
  lat->rfile_length = 0;
  for(k = 0; k < lat->zvoxels; k++) {
    for(j = 0; j < lat->yvoxels; j++) {
      for (i = 0; i < lat->xvoxels; i++) {
	rvec.i = i - lat->origin.i;
	rvec.j = j - lat->origin.j;
	rvec.k = k - lat->origin.k;
	rfloat.x = lat->xscale * rvec.i;
	rfloat.y = lat->yscale * rvec.j;
	rfloat.z = lat->zscale * rvec.k;
	r = (size_t)(sqrtf((rfloat.x*rfloat.x + rfloat.y*rfloat.y + 
		       rfloat.z*rfloat.z) / rscale)+.5);
	if (r > lat->rfile_length) lat->rfile_length = r;
	if (lat->lattice[index] != lat->mask_tag) {
	  if (ct[r] == 0) {
	    lat->rfile[r] = lat->lattice[index];
	    ct[r] = 1;
	  } else {
	    lat->rfile[r] = ((float)ct[r]*lat->rfile[r] +
			     lat->lattice[index])/(float)(ct[r]+1);
	    ct[r]++;
	  }
	}
	index++;
      }
    }
  }
  free((size_t *)ct);
  CloseShop:
  return(return_value);
}





