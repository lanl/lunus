/* LPROCIMLT.C - Process an image and accumulate the data on a target lattice.

   Author: Mike Wall
   Date: 4/23/2019
   Version: 1.

*/

#include<mwmask.h>
#include <string.h>

int lprocimlt(LAT3D *lat) 
{

  DIFFIMAGE *imdiff_corrected, *imdiff_scale, *imdiff_scale_ref;

  IJKCOORDS_DATA
    i0, j0, k0;

  DIFFIMAGE *imdiff_list = lat->imdiff, *imdiff;

  static int ct = 0;

  static DIFFIMAGE 
    *imdiff_corrected_list = NULL, 
    *imdiff_scale_list = NULL, 
    *imdiff_scale_ref_list = NULL;

  double start,stop;

  // Initialize other images

  if (imdiff_corrected_list == NULL) {
    imdiff_corrected_list = linitim(imdiff_list->num_panels);
  }
  if (imdiff_scale_list == NULL) {
    imdiff_scale_list = linitim(imdiff_list->num_panels);
  }
  if (imdiff_scale_ref_list == NULL) {
    imdiff_scale_ref_list = linitim(imdiff_list->num_panels);
  }

  // Apply masks

  lat->timer.mask = ltime();

  if (imdiff_list->num_panels == 1) {
    lpunchim(imdiff_list);
    lwindim(imdiff_list);
  } else {
    perror("LPROCIMLT: Multipanel image detected. Skipping lpunchim() and lwindim().\n");
    // ***Insert multipanel masking method here
  }
  
  lthrshim(imdiff_list);

  lat->timer.mask = ltime() - lat->timer.mask;

  // Mode filter to create image to be used for scaling

  lat->timer.mode = ltime();

  lcloneim(imdiff_scale_list,imdiff_list);

  lmodeim(imdiff_scale_list);

  lat->timer.mode = ltime() - lat->timer.mode;
  
  // Calculate correction factor
  
  lat->timer.correction = ltime();

  lcfim(imdiff_list);	  
  
  // Calculate corrected image
  
  lcloneim(imdiff_corrected_list,imdiff_list);
  if (lmulcfim(imdiff_corrected_list) != 0) {
    perror(imdiff_corrected->error_msg);
    exit(1);
  }
  
  lat->timer.correction = ltime() - lat->timer.correction;
	    
  // Set up common variables on the first pass

  if (lat->procmode == 0) {
    
    lat->timer.setup = ltime();

    // Reference image for scaling
    lcloneim(imdiff_scale_ref_list,imdiff_scale_list);

#ifdef DEBUG
    int j;
    int num_nz=0;
    size_t num_ign = 0;
    float sum_vals = 0.0;
    imdiff_scale_ref = imdiff_scale_ref_list;
    for (j=0; j<imdiff_scale_ref->image_length; j++) {	      
      if (imdiff_scale_ref->image[j] != imdiff_scale_ref->overload_tag && imdiff_scale_ref->image[j] != 0) {
	num_nz++;
	sum_vals += imdiff_scale_ref->image[j];
	//	      printf("image[%d]=%d,",j,imdiff_scale_ref->image[j]);
      } else num_ign ++;
      //	      if (num_nz>10) break;
    }
    printf("num_ign = %ld,avg = %g,",num_ign,sum_vals/(float)num_nz);
    lavgrim(imdiff_scale_ref);
    for (j=100; j<110;j++) {
      if (j>100 && j<=110)  printf("rf[%d]=%f,",j,imdiff_scale_ref->rfile[j]);
    }
    printf("\n");
#endif


    if (strstr(lat->cell_str,"None") != NULL) {
      float a,b,c,alpha,beta,gamma,adotb,adotc,bdotc;
      struct xyzmatrix a0;
      a0 = imdiff_list->amatrix;
      a = sqrtf(a0.xx*a0.xx+a0.xy*a0.xy+a0.xz*a0.xz);
      b = sqrtf(a0.yx*a0.yx+a0.yy*a0.yy+a0.yz*a0.yz);
      c = sqrtf(a0.zx*a0.zx+a0.zy*a0.zy+a0.zz*a0.zz);
      adotb = a0.xx*a0.yx + a0.xy*a0.yy + a0.xz*a0.yz;
      adotc = a0.xx*a0.zx + a0.xy*a0.zy + a0.xz*a0.zz;
      bdotc = a0.yx*a0.zx + a0.yy*a0.zy + a0.yz*a0.zz;
      alpha = acosf(bdotc/b/c)*180./PI;
      beta = acosf(adotc/a/c)*180./PI;
      gamma = acosf(adotb/a/b)*180./PI;
      //	    str_length = snprintf(NULL,0,"%f,%f,%f,%f,%f,%f",a,b,c,alpha,beta,gamma);
      //	    unit_cell = (char *)calloc(str_length+1,sizeof(char));
      sprintf(lat->cell_str,"%f,%f,%f,%f,%f,%f",a,b,c,alpha,beta,gamma);
    }
    lparsecelllt(lat);
    sprintf(lat->lattice_type_str,"P1");
    sprintf(lat->space_group_str,"P1");
    lat->cell.a *= lat->pphkl;
    lat->cell.b *= lat->pphkl;
    lat->cell.c *= lat->pphkl;
    lat->xvoxels = ((int)(lat->cell.a/lat->resolution.max)+1)*2;
    lat->yvoxels = ((int)(lat->cell.b/lat->resolution.max)+1)*2;
    lat->zvoxels = ((int)(lat->cell.c/lat->resolution.max)+1)*2;
    i0 = (IJKCOORDS_DATA)(lat->xvoxels/2. - 1.);
    j0 = (IJKCOORDS_DATA)(lat->yvoxels/2. - 1.);
    k0 = (IJKCOORDS_DATA)(lat->zvoxels/2. - 1.);
    lat->xscale = 1./lat->cell.a;
    lat->yscale = 1./lat->cell.b;
    lat->zscale = 1./lat->cell.c;
    lat->xbound.min = -i0*lat->xscale;
    lat->ybound.min = -j0*lat->yscale;
    lat->zbound.min = -k0*lat->zscale;
    lat->xbound.max = lat->xbound.min + ((float)lat->xvoxels-1)*lat->xscale;
    lat->ybound.max = lat->ybound.min + ((float)lat->yvoxels-1)*lat->yscale;
    lat->zbound.max = lat->zbound.min + ((float)lat->zvoxels-1)*lat->zscale;
    lat->origin.i = (IJKCOORDS_DATA)(-lat->xbound.min/lat->xscale + .5);
    lat->origin.j = (IJKCOORDS_DATA)(-lat->ybound.min/lat->yscale + .5);
    lat->origin.k = (IJKCOORDS_DATA)(-lat->zbound.min/lat->zscale + .5);
    lat->xyvoxels = lat->xvoxels * lat->yvoxels;
    lat->lattice_length = lat->xyvoxels*lat->zvoxels;
    ct = 0;
    if (lat->lattice != NULL) free(lat->lattice);
    lat->lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,sizeof(LATTICE_DATA_TYPE));
    if (lat->latct != NULL) free(lat->latct);
    lat->latct = (size_t *)calloc(lat->lattice_length,sizeof(size_t));

#ifdef DEBUG
      FILE *imageout;
      if ( (imageout = fopen("debug.img","wb")) == NULL ) {
	printf("\nCan't open %s.\n\n","debug.img");
	exit(1);
      }

      imdiff_corrected_list->outfile=imageout;

      //      lwriteim(imdiff_corrected_list);

#endif

    lat->timer.setup = ltime() - lat->timer.setup;
  
    return(0);
  } else if (lat->procmode == 2) {
    lfreeim(imdiff_corrected_list);
    lfreeim(imdiff_scale_list);
    lfreeim(imdiff_scale_ref_list);
    return(0);
  }
    
	  
  // Calculate the image scale factor

  start = ltime();

  lscaleim(imdiff_scale_ref_list,imdiff_scale_list);

  stop = ltime();
  lat->timer.scale = stop - start;

  float this_scale_factor = imdiff_scale_ref_list->rfile[0];
  float this_scale_error = imdiff_scale_ref_list->rfile[1];

  printf("%g %g\n",this_scale_factor,this_scale_error);
  fflush(stdout);

  // Collect the image data into the lattice

  int pidx;

  //  printf("number of panels = %d\n",imdiff_list->num_panels);

  start = ltime();

  for (pidx = 0; pidx < imdiff_list->num_panels; pidx++) {
    imdiff = &imdiff_list[pidx];
    imdiff_corrected = &imdiff_corrected_list[pidx];
    imdiff_scale = &imdiff_scale_list[pidx];
    imdiff_scale_ref = &imdiff_scale_ref_list[pidx];

    size_t data_added=0;
    struct xyzcoords H, dH;
    IJKCOORDS_DATA ii,jj,kk;
    size_t index = 0;
    size_t j;
    i0 = (IJKCOORDS_DATA)(lat->xvoxels/2. - 1.);
    j0 = (IJKCOORDS_DATA)(lat->yvoxels/2. - 1.);
    k0 = (IJKCOORDS_DATA)(lat->zvoxels/2. - 1.);

    int iii;

    //    printf("Image %d, slist[1001..1010] = ",pidx);
    //    for (iii=1001;iii<=1010;iii++) printf("(%f,%f,%f) ",imdiff->slist[iii].x,imdiff->slist[iii].y,imdiff->slist[iii].z);
    //    printf("\n");
  

    for (j=0; j<imdiff->image_length; j++) {
      H = lmatvecmul(imdiff->amatrix, imdiff->slist[j]);

#ifdef DEBUG
      if (j<10) {
	printf("Image H[%d] = (%f, %f, %f)\n",j,H.x,H.y,H.z);
      }
#endif

      dH.x = fabs(H.x - roundf(H.x));
      dH.y = fabs(H.y - roundf(H.y));
      dH.z = fabs(H.z - roundf(H.z));

      if (lat->filterhkl==0 || dH.x>=0.25 || dH.y>=0.25 || dH.z>=0.25) {
	ii = (IJKCOORDS_DATA)roundf(H.x*(float)lat->pphkl) + i0;
	jj = (IJKCOORDS_DATA)roundf(H.y*(float)lat->pphkl) + j0;
	kk = (IJKCOORDS_DATA)roundf(H.z*(float)lat->pphkl) + k0;
	if (ii>=0 && ii<lat->xvoxels && jj>=0 && jj<lat->yvoxels &&
	    kk>=0 && kk<lat->zvoxels && imdiff_scale->image[index]>0 &&
	    imdiff_scale->image[index]!=imdiff->ignore_tag) {
	  size_t latidx = kk*lat->xyvoxels + jj*lat->xvoxels + ii;
	  if (strcmp(lat->integration_image_type,"raw")==0) {
	    lat->lattice[latidx] += 
	      (LATTICE_DATA_TYPE)(imdiff->image[index]-imdiff->value_offset)
	      * imdiff->correction[index]
	      * this_scale_factor;
	  }
	  if (strcmp(lat->integration_image_type,"corrected")==0) {
	    lat->lattice[latidx] += 
	      (LATTICE_DATA_TYPE)(imdiff_corrected->image[index]-imdiff_corrected->value_offset)
	      * this_scale_factor;
	  }
	  if (strcmp(lat->integration_image_type,"scale")==0) {
	    lat->lattice[latidx] += 
	      (LATTICE_DATA_TYPE)(imdiff_scale->image[index]-imdiff_scale->value_offset)
	      * imdiff->correction[index]
	      * this_scale_factor;
	  }
	  lat->latct[latidx] += 1;
	  data_added += 1;
	}
      }
      index++;
    }
  }

  stop = ltime();
  lat->timer.map = stop - start;

  ct++;
  //  printf("Done processing image %d\n",ct);
}


