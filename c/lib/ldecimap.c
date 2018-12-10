/* LDECIMAP.C - Decimate a map.
   
   Author: Mike Wall
   Date: 5/4/2015
   Version: 1.
   
   */

#include<mwmask.h>

int ldecimap(CCP4MAP *map)
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

  MAP_DATA_TYPE
    *ddata;

  /* 
   * Check map properties:
   */

  // Only works if map mode = 2 (float data type)

  if (map->mode != 2) {
    printf("\nLDECIMAP: Unrecognized CCP4 map MODE = %d\n\n",map->mode);
    return_value = 3;
    goto CloseShop;
  }

  dfac = (long)map->decimation_factor;

  printf("Decimating map using factor %ld\n",dfac);

  if (!((map->nr % dfac) == 0 && (map->ns % dfac) == 0 && (map->nc % dfac) == 0)) {
    printf("\nLDECIMAP: Map dimensions not a multiple of %d\n\n",dfac);
    return_value = 4;
    goto CloseShop;
  }

  /*
   * Allocate decimated map
   */

  dlen = map->map_length/dfac/dfac/dfac;

  ddata = (MAP_DATA_TYPE *)calloc((size_t)dlen,sizeof(MAP_DATA_TYPE));

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

  nrc=map->section_length;
  for (s = 0; s < map->ns; s=s+dfac) {
    for (r = 0; r < map->nr; r=r+dfac) {
      for (c = 0; c < map->nc; c=c+dfac) {
	long ss,rr,cc;
	dmap_index = ((s*nrc/dfac+r*map->nc)/dfac+c)/dfac;
	for (ss=0;ss<stencil_size;ss++) {
	  long sidx = (s-stencil_ofst+ss+map->ns)%map->ns;
	  //	  printf("sidx = %d\n",sidx);
	  for (rr=0;rr<stencil_size;rr++) {
	    long ridx = (r-stencil_ofst+rr+map->nr)%map->nr;
	    for (cc=0;cc<stencil_size;cc++) {
	      long cidx = (c-stencil_ofst+cc+map->nc)%map->nc;
	      map_index = sidx*nrc+ridx*map->nc+cidx;
	      ddata[dmap_index]+=map->data[map_index]*stencil[ss]*stencil[rr]*stencil[cc]*stencil_weight;
	    }
	  }
	} 
      }
    }
  }

  // Replace original map with decimated map
  
  map->map_length=dlen;
  map->data_buf=(void *)realloc(map->data_buf,sizeof(MAP_DATA_TYPE)*dlen);
  map->data = (MAP_DATA_TYPE *)map->data_buf;
  memcpy(map->data_buf,(void *)ddata,dlen*sizeof(MAP_DATA_TYPE));
  map->nr /= dfac;
  map->ns /= dfac;
  map->nc /= dfac;
  map->nx /= dfac;
  map->ny /= dfac;
  map->nz /= dfac;
  map->section_length /= dfac*dfac;
     
  CloseShop:
  return(return_value);
}





