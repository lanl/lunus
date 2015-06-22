/* LREADLT.C - Read a lattice from a file.

   Author: Mike Wall
   Date: 9/27/95
   Version: 1.

*/

#include<mwmask.h>

int lreadlt(LAT3D *lat)
{

  size_t
    num_read;

  int
    return_value = 0;

  /*
   * Read in 3D lattice descriptor:
   */

  num_read = fread(&lat->xvoxels, sizeof(long), 1, lat->infile);
  num_read = fread(&lat->yvoxels, sizeof(long), 1, lat->infile);
  num_read = fread(&lat->zvoxels, sizeof(long), 1, lat->infile);
  num_read = fread(&lat->xbound, sizeof(struct bounds), 1, lat->infile);
  num_read = fread(&lat->ybound, sizeof(struct bounds), 1, lat->infile);
  num_read = fread(&lat->zbound, sizeof(struct bounds), 1, lat->infile);


  
  lat->xscale = (float)((lat->xbound.max - lat->xbound.min)/ 
		         (float)(lat->xvoxels - 1));               
  lat->yscale = (float)((lat->ybound.max - lat->ybound.min)/ 
		         (float)(lat->yvoxels - 1));               
  lat->zscale = (float)((lat->zbound.max - lat->zbound.min)/ 
		         (float)(lat->zvoxels - 1));               

  lat->origin.i =  (long)(-lat->xbound.min / lat->xscale + .5);
  lat->origin.j =  (long)(-lat->ybound.min / lat->yscale + .5);
  lat->origin.k =  (long)(-lat->zbound.min / lat->zscale + .5);
  lat->xyvoxels = lat->xvoxels * lat->yvoxels;
  lat->lattice_length = lat->xyvoxels*lat->zvoxels;

  /*
   * Allocate memory for 3D lattice:
   */
  
  if (lat->lattice) free((LATTICE_DATA_TYPE *)lat->lattice);
  lat->lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,
					     sizeof(LATTICE_DATA_TYPE));
  if (!lat->lattice) {
    printf("\nCouldn't allocate 3D lattice.\n\n");
    return_value = 1;
    goto CloseShop;
  }
  
  /*
   * Read in lattice:
   */
  
  num_read = fread(lat->lattice, sizeof(LATTICE_DATA_TYPE), 
		   lat->lattice_length,lat->infile);
  int i;
  for (i=0;i<lat->lattice_length;i++) {
    if (lat->lattice[i] != lat->mask_tag && lat->lattice[i] < 0) {
      printf("%d,%f\n",i,lat->lattice[i]);
    }
  }
  if (num_read != lat->lattice_length) {
    printf("/nCouldn't read all of the lattice from input file.\n\n");
    return_value = 2;
    goto CloseShop;
  }
  CloseShop:
  return(return_value);
}
