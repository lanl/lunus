/* LMINRLT.C - Calculate the minimum voxel value vs. radius for an 
              input lattice
   
   Author: Mike Wall
   Date: 2/23/2017
   Version: 1.
   
   */

#include<mwmask.h>

int lminrlt(LAT3D *lat)
{
  size_t
    *ct,
    i,
    j,
    k,
    r,
    lat_index;

  int
    return_value = 0;
  
  struct ijkcoords index,rvec;
  
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

  lat->rfile_length = 0;
  lat->index.i=lat->origin.i+1;
  lat->index.j=lat->origin.j+1;
  lat->index.k=lat->origin.k+1;
  rscale=lssqrFromIndex(lat);
  lat_index = 0;
  for(index.k = 0; index.k < lat->zvoxels; index.k++) {
    for(index.j = 0; index.j < lat->yvoxels; index.j++) {
      for (index.i = 0; index.i < lat->xvoxels; index.i++) {
	lat->index = index;
	
	r = (size_t)(sqrtf(lssqrFromIndex(lat) / rscale)+.5);
	if (lat->lattice[lat_index] != lat->mask_tag) {
	  //	  if (lat->lattice[index]<0) printf("%d,%f\n",(int)index,lat->lattice[index]);
	  if (r >= lat->rfile_length) lat->rfile_length = r + 1;
	  if (ct[r] == 0) {
	    lat->rfile[r] = lat->lattice[lat_index];
	    ct[r] = 1;
	  } else {
	    if (lat->rfile[r] > lat->lattice[lat_index]) {
	      lat->rfile[r] = lat->lattice[lat_index];
	    }
	  }
	}
	lat_index++;
      }
    }
  }
  // If there are no counts at a given radius, mask the value

  for (r=0;r<lat->rfile_length;r++) {
    if (ct[r]==0) {
      lat->rfile[r]=lat->mask_tag;
    }
  }

  free((size_t *)ct);
  CloseShop:
  return(return_value);
}





