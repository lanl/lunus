/* LDECILT.C - Decimate a lattice.
   
   Author: Mike Wall
   Date: 1/18/2017
   Version: 1.
   
   */

#include<mwmask.h>

int ldecilt(LAT3D *lat)
{
  size_t
    i,
    j,
    k,
    dmap_index,
    map_index = 0,
    lat_index = 0;

  long dfac;

  long 
    c,
    r,
    s;

  long
    return_value = 0;

  long dlen,nrc;

  LATTICE_DATA_TYPE
    *ddata;

  dfac = (long)lat->decimation_factor;

  printf("Decimating lattice using factor %ld\n",dfac);

  if (!((lat->xvoxels % dfac) == 0 && (lat->yvoxels % dfac) == 0 && (lat->zvoxels % dfac) == 0)) {
    printf("\nLDECIMAP: Map dimensions not a multiple of %d\n\n",dfac);
    return_value = 4;
    goto CloseShop;
  }

  /*
   * Allocate decimated map
   */

  dlen = lat->lattice_length/dfac/dfac/dfac;

  ddata = (LATTICE_DATA_TYPE *)calloc((size_t)dlen,sizeof(LATTICE_DATA_TYPE));

  /*
   * Calculate the decimated data
   */  

  float *stencil;
  int dfac_is_even;
  int stencil_size;

  if (dfac % 2 == 0) {
    dfac_is_even=1;
  } else {
    dfac_is_even=0;
  }

  if (dfac_is_even) {
    stencil_size = dfac+1;
  } else {
    stencil_size = dfac;
  }

  stencil = (float *)malloc(sizeof(float)*stencil_size);

  float stencil_weight=1./(float)dfac/(float)dfac/(float)dfac;

  for (i=0;i<stencil_size;i++) {
    if (dfac_is_even && ((i==0) || (i==(stencil_size-1)))) {
      stencil[i]=0.5;
    } else {
      stencil[i]=1.0;
    }
  }

  int stencil_ofst = (stencil_size-1)/2;

  for (k = 1; k < lat->zvoxels; k=k+dfac) {
    for (j = 0; j < lat->yvoxels; j=j+dfac) {
      for (i = 0; i < lat->xvoxels; i=i+dfac) {
	long kk,jj,ii;
	dlat_index = ((k*lat->xyvoxels/dfac+j*lat->xvoxels)/dfac+i)/dfac;
	for (kk=0;kk<stencil_size;kk++) {
	  long kidx = (k-stencil_ofst+kk+lat->zvoxels)%lat->zvoxels;
	  //	  printf("sidx = %d\n",sidx);
	  for (jj=0;jj<stencil_size;jj++) {
	    long jidx = (j-stencil_ofst+jj+lat->yvoxels)%lat->yvoxels;
	    for (ii=0;ii<stencil_size;ii++) {
	      long iidx = (i-stencil_ofst+ii+lat->xvoxels)%lat->xvoxels;
	      lat_index = kidx*lat->xyvoxels+jidx*lat->xvoxels+iidx;
	      ddata[dlat_index]+=lat->lattice[lat_index]*stencil[kk]*stencil[jj]*stencil[ii]*stencil_weight;
	    }
	  }
	} 
      }
    }
  }

  // Replace original map with decimated map
  
  lat->lattice_length=dlen;
  lat->lattice=(LATTICE_DATA_TYPE *)realloc(lat->lattice,sizeof(LATTICE_DATA_TYPE)*dlen);
  memcpy((void *)lat->lattice,(void *)ddata,dlen*sizeof(LATTICE_DATA_TYPE));
  lat->xvoxels /= dfac;
  lat->yvoxels /= dfac;
  lat->zvoxels /= dfac;
  lat->xyvoxels /= dfac*dfac;
  lat->xscale *= dfac;
  lat->yscale *= dfac;
  lat->zscale *= dfac;

  lat->xbound.
  lat->xbound.max = lat->xbound.min + ((float)lat->xvoxels-1)*lat->xscale;
  lat->ybound.max = lat->ybound.min + ((float)lat->yvoxels-1)*lat->yscale;
  lat->zbound.max = lat->zbound.min + ((float)lat->zvoxels-1)*lat->zscale;

  lat->origin.i = (IJKCOORDS_DATA)(-lat->xbound.min/lat->xscale + .5);
  lat->origin.j = (IJKCOORDS_DATA)(-lat->ybound.min/lat->yscale + .5);
  lat->origin.k = (IJKCOORDS_DATA)(-lat->zbound.min/lat->zscale + .5);

     
  CloseShop:
  return(return_value);
}





