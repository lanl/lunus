/* LSUMLT.C - Calculate the voxel-by-voxel sum of two lattices.
   
   Author: Mike Wall
   Date: 4/11/95
   Version: 1.
   
   */

#include<mwmask.h>

int lsumlt(LAT3D *lat1, LAT3D *lat2)
{
  size_t
    i,
    j,
    k,
    index = 0;

  int
    return_value = 0;
  
  for(k = 0; k < lat1->zvoxels; k++) {
    for(j = 0; j < lat1->yvoxels; j++) {
      for (i = 0; i < lat1->xvoxels; i++) {
	if ((lat1->lattice[index] != lat1->mask_tag) &&
	    (lat2->lattice[index] != lat1->mask_tag)) {
	  lat1->lattice[index] += lat2->lattice[index];
	}
	else {
	  lat1->lattice[index] = lat1->mask_tag;
	}
	index++;
      }
    }
  }
  CloseShop:
  return(return_value);
}





