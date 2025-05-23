/* LWRITELT.C - Write a lattice to a file.

   Author: Mike Wall
   Date: 9/27/95
   Version: 1.

*/

#include<mwmask.h>

int lwritelt(LAT3D *lat)
{

  size_t
    num_wrote;

  int
    return_value = 0;

/*
 * Write lattice to output file:
 */

  num_wrote = fwrite(&lat->xvoxels, sizeof(uint32_t), 1, lat->outfile);
  num_wrote = fwrite(&lat->yvoxels, sizeof(uint32_t), 1, lat->outfile);
  num_wrote = fwrite(&lat->zvoxels, sizeof(uint32_t), 1, lat->outfile);
  num_wrote = fwrite(&lat->xbound, sizeof(struct bounds), 1, lat->outfile);
  num_wrote = fwrite(&lat->ybound, sizeof(struct bounds), 1, lat->outfile);
  num_wrote = fwrite(&lat->zbound, sizeof(struct bounds), 1, lat->outfile);
  /*
  int i;
  for (i=0;i<lat->lattice_length;i++) {
    if (lat->lattice[i] != lat->mask_tag && lat->lattice[i] < 0) {
      printf("%d,%f\n",i,lat->lattice[i]);
    }
  }
  */
  num_wrote = fwrite(lat->lattice, sizeof(LATTICE_DATA_TYPE), 
			lat->lattice_length, lat->outfile);

  if (num_wrote != lat->lattice_length) {
    printf("/nCouldn't write all of the lattice to output file.\n\n");
    return_value = 1;
    goto CloseShop;
  }

  CloseShop:

  return(return_value);
}



