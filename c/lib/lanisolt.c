/* LANISOLT.C - Extract the anisotropic component of a lattice.
   
   Author: Mike Wall
   Date: 9/27/2016
   Version: 1.
   
   */

#include<mwmask.h>

void lanisolt(LAT3D *lat)
{

  size_t *ct,r,lat_index;

  float rf,rscale;

  struct ijkcoords index;

  RFILE_DATA_TYPE *rfile,*rfile2;

  rfile = (RFILE_DATA_TYPE *)calloc(MAX_RFILE_LENGTH,sizeof(RFILE_DATA_TYPE));
  rfile2 = (RFILE_DATA_TYPE *)calloc(MAX_RFILE_LENGTH,sizeof(RFILE_DATA_TYPE));
  ct = (size_t *)calloc(MAX_RFILE_LENGTH,sizeof(size_t));
  if (!ct) {
    sprintf(lat->error_msg,"\nLANISOLT: Couldn't allocate counting "
	    "array ct.\n\n");
    //    return_value = 1;
    //    goto CloseShop;
  }

  lat->rfile_length = 0;
  lat->index.i=lat->origin.i+1;
  lat->index.j=lat->origin.j+1;
  lat->index.k=lat->origin.k+1;
  rscale=lssqrFromIndex(lat);
  //  rscale = (lat->xscale*lat->xscale + lat->yscale*lat->yscale +
  //	    lat->zscale*lat->zscale);
  // calculate isotropic component
  lat_index = 0;
  for(index.k = 0; index.k < lat->zvoxels; index.k++) {
    for(index.j = 0; index.j < lat->yvoxels; index.j++) {
      for (index.i = 0; index.i < lat->xvoxels; index.i++) {
	lat->index = index;
	
	r = (size_t)(sqrtf(lssqrFromIndex(lat) / rscale)+.5);
	if (isnan(lat->lattice[lat_index])) {
	  printf("Lattice value is nan at (%d,%d,%d). Aborting\n",index.i,index.j,index.k);
	  exit(1);
	}
	if (lat->lattice[lat_index] != lat->mask_tag) {
	  //	  if (lat->lattice[index]<0) printf("%d,%f\n",(int)index,lat->lattice[index]);
	  if (r >= lat->rfile_length) lat->rfile_length = r + 1;
	  if (ct[r] == 0) {
	    rfile[r] = lat->lattice[lat_index];
	    rfile2[r] = lat->lattice[lat_index]*lat->lattice[lat_index];
	    ct[r] = 1;
	  } else {
	    rfile[r] += lat->lattice[lat_index];
	    rfile2[r] += lat->lattice[lat_index]*lat->lattice[lat_index];
	    if (isnan(rfile[r])) {
	      perror("LANISOLT: rfile[r] is nan. Aborting\n");
	      exit(1);
	    }
	    ct[r]++;
	  }
	}
	lat_index++;
      }
    }
  }

  // If there are no counts at a given radius, mask the value

  int num_zero_counts=0;

  for (r=0;r<lat->rfile_length;r++) {
    if (ct[r]==0) {
      lat->rfile[r]=lat->mask_tag;
      num_zero_counts += 1;
    } else {
      lat->rfile[r] = rfile[r]/(RFILE_DATA_TYPE)ct[r];
      if (lat->procmode == 1) {
	lat->rfile[r] -= sqrtf(rfile2[r]/(RFILE_DATA_TYPE)ct[r]-lat->rfile[r]*lat->rfile[r]);
      }
    }	
  }

  free(rfile);
  free(rfile2);

  //  printf("LANISOLT: num_zero_counts = %d\n",num_zero_counts);

  // subtract isotropic component

  // First, compute the spline from the rfile

  float *tau, *c;
  int n,ibcbeg,ibcend;
  int l,jd,k;
  k = 4;
  jd = 0;
  int i,j;

  tau = (float *)malloc(sizeof(float)*lat->rfile_length);
  c = (float *)malloc(sizeof(float)*4*lat->rfile_length);
  //  n = (int)lat->rfile_length;
  ibcbeg = 0;
  ibcend = 0;

  // 


  // populate variables for spline

  n=0;

  for (i=0;i<lat->rfile_length;i++) {
    if (lat->rfile[i] != lat->mask_tag) {
      tau[n]=(float)i;
      c[4*n] = lat->rfile[i];
      n++;
    }
  }

  lspline(tau,c,&n,&ibcbeg,&ibcend);

  /*  for (i=0;i<n-1;i++) {
    int jd,k;
    k = 4;
    jd = 0;
    float x;
    x = (float)tau[i]+0.5;
    printf("(%f,%f)",c[4*i],lspleval(tau,c,&n,&k,&x,&jd));
  }
  printf("\n");
  exit(0);*/

  //  size_t maxr=40;
  lat_index = 0;
  for(index.k = 0; index.k < lat->zvoxels; index.k++) {
    for(index.j = 0; index.j < lat->yvoxels; index.j++) {
      for (index.i = 0; index.i < lat->xvoxels; index.i++) {
	lat->index = index;
	rf = sqrtf(lssqrFromIndex(lat) / rscale);
	r = (size_t)rf;
	if ((r < lat->rfile_length) && 
	    (lat->lattice[lat_index] != lat->mask_tag)) {
	  //    lat->lattice[index] -= lat->rfile[r];
	  //	  if (r <= maxr) {
	  l = (int)r + 1;
	  if (rf>= tau[0] && rf <= tau[n-2]) {
	    lat->lattice[lat_index] -= lspleval(tau,c,&l,&k,&rf,&jd);
	  } else {
	    lat->lattice[lat_index] = lat->mask_tag;
	  }
	  //	    lat->lattice[lat_index] -= (lat->rfile[r] + 
	  //				  (rf - (float)r)*(lat->rfile[r+1] - 
	  //						   lat->rfile[r]));
	    //	  } else {
	    //	    lat->lattice[lat_index] = lat->mask_tag;
	    //	  }
	}

	lat_index++;
      }
    }
  }
  free((size_t *)ct);
}

