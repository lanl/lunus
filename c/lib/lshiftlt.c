/* LSHIFTLT.C - Translate the intensity in a lattice by specified coordinate shift.
   
   Author: Mike Wall
   Date: 7/17/2014
   Version: 1.
   
   */

#include<mwmask.h>

int lshiftlt(LAT3D *lat,struct ijkcoords t)
{
  size_t
    index,
    ct,
    *lat_index;
  
  int
    return_value = 0;

  struct ijkcoords 
    i,j,k,
    index1,index2;

  LATTICE_DATA_TYPE
    *lattice;

  /*
   * Allocate lattice:
   */

  lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,
					sizeof(LATTICE_DATA_TYPE));
  if (!lattice) {
    sprintf(lat->error_msg,"\nLROTLT:  Couldn't allocate lattice.\n\n");
    return_value = 1;
    goto CloseShop;
  }

  for(index1.k = 0; index1.k < lat->zvoxels; index1.k++) {
    for(index1.j = 0; index1.j < lat->yvoxels; index1.j++) {
      for (index1.i = 0; index1.i < lat->xvoxels; index1.i++) {
	index = index1.k*lat->xyvoxels+index1.j*lat->xvoxels+index1.i;
	lattice[index]=lat->mask_tag;
      }
    }
  }
  for(index1.k = 0; index1.k < lat->zvoxels; index1.k++) {
    for(index1.j = 0; index1.j < lat->yvoxels; index1.j++) {
      for (index1.i = 0; index1.i < lat->xvoxels; index1.i++) {
	index2 = lijksum(index1,t);
	if (index2.i >= 0 && index2.i < lat->xvoxels && index2.j >= 0 && index2.j < lat->yvoxels && 
	    index2.k >= 0 && index2.k < lat->zvoxels) {
	  lattice[index2.k*lat->xyvoxels+index2.j*lat->xvoxels+index2.i] =
	    lat->lattice[index1.k*lat->xyvoxels+index1.j*lat->xvoxels+index1.i];
	}
      }
    }
  }

  /*
   * Copy lattice to input lattice:
   */

  for(index = 0; index < lat->lattice_length; index++)
    {
      lat->lattice[index] = lattice[index];
    }
  CloseShop:
  free((LATTICE_DATA_TYPE *)lattice);
  return(return_value);
}








