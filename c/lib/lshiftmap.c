/* LSHIFTMAP.C - Translate a map.
   
   Author: Mike Wall
   Date: 5/8/2015
   Version: 1.
   
   */

#include<mwmask.h>

int lshiftmap(CCP4MAP *map)
{
  size_t
    i,
    j,
    k,
    smap_index,
    map_index = 0,
    lat_index = 0;

  long dfac=1; // decimation factor, keep for now to allow for the possibility of translating with decimation

  long 
    c,
    r,
    s;

  long
    return_value = 0;

  long dlen,nrc;

  MAP_DATA_TYPE
    *sdata;

  /* 
   * Check map properties:
   */

  // Only works if map mode = 2 (float data type)

  if (map->mode != 2) {
    printf("\nLDECIMAP: Unrecognized CCP4 map MODE = %d\n\n",map->mode);
    return_value = 3;
    goto CloseShop;
  }

  printf("Translating map by %f %f %f\n",map->shift.x,map->shift.y,map->shift.z);

  /*
   * Allocate shifted map
   */

  dlen = map->map_length/dfac/dfac/dfac;

  sdata = (MAP_DATA_TYPE *)calloc((size_t)dlen,sizeof(MAP_DATA_TYPE));

  /*
   * Calculate the decimated data
   */  

  float *stencilx,*stencily,*stencilz;
  float *stencils,*stencilr,*stencilc;
  int dfac_is_even;
  int stencil_size;

  stencil_size = 2;
  stencilx = (float *)malloc(sizeof(float)*stencil_size);
  stencily = (float *)malloc(sizeof(float)*stencil_size);
  stencilz = (float *)malloc(sizeof(float)*stencil_size);
  stencilc = (float *)malloc(sizeof(float)*stencil_size);
  stencilr = (float *)malloc(sizeof(float)*stencil_size);
  stencils = (float *)malloc(sizeof(float)*stencil_size);

  struct ijkcoords wholeshift;
  struct xyzcoords fracshift;
  struct xyzcoords normshift;

  int wss,wsr,wsc;

  printf("xlen,ylen,zeln = %f,%f,%f\n",map->xlen,map->ylen,map->zlen);

  normshift.x = map->shift.x/(map->xlen/(float)map->nx);
  normshift.y = map->shift.y/(map->ylen/(float)map->ny);
  normshift.z = map->shift.z/(map->zlen/(float)map->nz);

  printf("normshift.x,y,z=%f,%f,%f\n",normshift.x,normshift.y,normshift.z);

  wholeshift.i = (int)normshift.x;
  wholeshift.j = (int)normshift.y;
  wholeshift.k = (int)normshift.z;

  if (normshift.x<0) {
    wholeshift.i--;
  }
  if (normshift.y<0) {
    wholeshift.j--;
  }
  if (normshift.z<0) {
    wholeshift.k--;
  }

  fracshift.x = normshift.x - (float)wholeshift.i;
  fracshift.y = normshift.y - (float)wholeshift.j;
  fracshift.z = normshift.z - (float)wholeshift.k;

  printf("fractional shift: %f %f %f\n",fracshift.x,fracshift.y,fracshift.z);

  stencilx[0] = 1.-fracshift.x;
  stencilx[1] = fracshift.x;
  stencily[0] = 1.-fracshift.x;
  stencily[1] = fracshift.y;
  stencilz[0] = 1.-fracshift.z;
  stencilz[1] = fracshift.z;

  if (map->mapc == 3 && map->mapr == 1 && map->maps == 2) {
    wsc = wholeshift.k;
    wsr = wholeshift.i;
    wss = wholeshift.j;
    for (i=0;i<=1;i++) {
      stencils[i]=stencily[i];
      stencilr[i]=stencilx[i];
      stencilc[i]=stencilz[i];
    }
  } else if (map->mapc == 3 && map->mapr == 2 && map->maps == 1) {
    wsc = wholeshift.k;
    wsr = wholeshift.j;
    wss = wholeshift.i;
    for (i=0;i<=1;i++) {
      stencils[i]=stencilx[i];
      stencilr[i]=stencily[i];
      stencilc[i]=stencilz[i];
    }
  }    
  

  int stencil_ofst = (stencil_size-1)/2;

  printf("Applying stencil\n");

  printf("Whole offsets: %d %d %d\n",wss,wsr,wsc);

  nrc=map->section_length;
  for (s = 0; s < map->ns; s=s+dfac) {
    for (r = 0; r < map->nr; r=r+dfac) {
      for (c = 0; c < map->nc; c=c+dfac) {
	long ss,rr,cc;
	smap_index = ((s*nrc/dfac+r*map->nc)/dfac+c)/dfac;
	for (ss=0;ss<stencil_size;ss++) {
	  long sidx = (s-stencil_ofst+wss+ss+map->ns)%map->ns;
	  //	  printf("sidx = %d\n",sidx);
	  for (rr=0;rr<stencil_size;rr++) {
	    long ridx = (r-stencil_ofst+wsr+rr+map->nr)%map->nr;
	    for (cc=0;cc<stencil_size;cc++) {
	      long cidx = (c-stencil_ofst+wsc+cc+map->nc)%map->nc;
	      map_index = sidx*nrc+ridx*map->nc+cidx;
	      sdata[smap_index]+=map->data[map_index]*stencils[ss]*stencilr[rr]*stencilc[cc];
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
  memcpy(map->data_buf,(void *)sdata,dlen*sizeof(MAP_DATA_TYPE));
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





