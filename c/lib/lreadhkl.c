/* LREADHKL.C - Read a lattice from a .hkl file, given a template.

   Author: Mike Wall
   Date: 1/18/13
   Version: 1.

*/

#include<mwmask.h>

int lreadhkl(LAT3D *lat,LAT3D *tmpl)
{

  int
    return_value = 0;
  
  int 
    len_inl = 10000;

  char
    *token, *inl;

  inl = (char *)malloc(len_inl*sizeof(char));

/*
 * Read the header:
 */

  int header_done = 0;


  lat->xvoxels = tmpl->xvoxels;
  lat->yvoxels = tmpl->yvoxels;
  lat->zvoxels = tmpl->zvoxels;
  lat->lattice_length = tmpl->lattice_length;
  lat->lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,
					     sizeof(LATTICE_DATA_TYPE));
  lat->xyvoxels = tmpl->xyvoxels;
  lat->xscale = tmpl->xscale;
  lat->yscale = tmpl->yscale;
  lat->zscale = tmpl->zscale;
  lat->xbound.min = tmpl->xbound.min;
  lat->ybound.min = tmpl->ybound.min;
  lat->zbound.min = tmpl->zbound.min;
  lat->xbound.max = lat->xbound.min + (lat->xvoxels-1)*lat->xscale;
  lat->ybound.max = lat->ybound.min + (lat->yvoxels-1)*lat->yscale;
  lat->zbound.max = lat->zbound.min + (lat->zvoxels-1)*lat->zscale;  
  /*
  lat->origin.i = (IJKCOORDS_DATA)(-lat->xbound.min*lat->xscale+.49);
  lat->origin.j = (IJKCOORDS_DATA)(-lat->ybound.min*lat->yscale+.49);
  lat->origin.k = (IJKCOORDS_DATA)(-lat->zbound.min*lat->zscale+.49);
  */
  lat->origin.i = tmpl->origin.i;
  lat->origin.j = tmpl->origin.j;
  lat->origin.k = tmpl->origin.k;
  printf("origin: %d %d %d\n",lat->origin.i,lat->origin.j,lat->origin.k);

  size_t index = 0,ct=0;
  int i,j,k;
  float I,sigI;

  printf("%d %d %d %ld\n",lat->xvoxels,lat->yvoxels,lat->zvoxels,lat->xvoxels*lat->yvoxels*lat->zvoxels);

  printf("Initializing the hkl file data...\n");

  index=0;

  for (k=0;k<lat->zvoxels;k++) {
    for (j=0;j<lat->yvoxels;j++) {
      for (i=0;i<lat->xvoxels;i++) {
	lat->lattice[index] = lat->mask_tag;
	index++;
      }
    }
  }

  printf("Reading the hkl file data...\n");

  while (fscanf(lat->infile,"%d %d %d %f",&i,&j,&k,&I)!=EOF) {
//  while (fscanf(lat->infile,"%d %d %d %f %f",&i,&j,&k,&I,&sigI)!=EOF) {
    //    printf("%d %d %d\n",k+lat->origin.k,j+lat->origin.j,i+lat->origin.i);
    int kidx = k+lat->origin.k;
    int jidx = j+lat->origin.j;
    int iidx = i+lat->origin.i;
    if ((iidx>=0 && iidx < lat->xvoxels) && (jidx>=0 && jidx < lat->yvoxels) && (kidx>=0 && kidx < lat->zvoxels)) {
      lat->lattice[(k+lat->origin.k)*lat->xyvoxels+(j+lat->origin.j)*lat->xvoxels+i+lat->origin.i] = I;
    }
  }

  printf("...done\n");

  if (index != lat->lattice_length) {
    printf("\nCouldn't read all of the lattice from the input file.\n\n");
    return_value = 1;
    goto CloseShop;
  }

  CloseShop:
  free(inl);
  return(return_value);
}



