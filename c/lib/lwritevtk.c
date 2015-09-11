/* LWRITEVTK.C - Write a lattice to a .vtk file.

   Author: Mike Wall
   Date: 10/1/12
   Version: 1.

*/

#include<mwmask.h>

int lwritevtk(LAT3D *lat)
{

  int
    return_value = 0;

/*
 * Write lattice to output file:
 */

  fprintf(lat->outfile,"# vtk DataFile Version 2.0\n");
  fprintf(lat->outfile,"Input file is %s\n",lat->filename);
  fprintf(lat->outfile,"ASCII\n");
  fprintf(lat->outfile,"DATASET STRUCTURED_POINTS\n");
  fprintf(lat->outfile,"DIMENSIONS %d %d %d\n",(int)lat->xvoxels,(int)lat->yvoxels,(int)lat->zvoxels);
  fprintf(lat->outfile,"SPACING %f %f %f\n",lat->xscale,lat->yscale,lat->zscale);
  fprintf(lat->outfile,"ORIGIN %f %f %f\n",lat->xbound.min,lat->ybound.min,lat->zbound.min);
  fprintf(lat->outfile,"POINT_DATA %d\n",(int)lat->lattice_length);
  fprintf(lat->outfile,"SCALARS volume_scalars float 1\n");
  fprintf(lat->outfile,"LOOKUP_TABLE default\n");

  int index = 0,ct=0;
  int i,j,k;

  for (k=0;k<lat->zvoxels;k++) {
    for (j=0;j<lat->yvoxels;j++) {
      for (i=0;i<lat->xvoxels;i++) {
	fprintf(lat->outfile, "%g ",lat->lattice[index]);
	index++;
      }
      fprintf(lat->outfile,"\n");
    }
  }
  

  if (index != lat->lattice_length) {
    printf("/nCouldn't write all of the lattice to output file.\n\n");
    return_value = 1;
    goto CloseShop;
  }

  CloseShop:

  return(return_value);
}



