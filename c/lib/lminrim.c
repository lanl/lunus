/* LMINRIM.C - Calculate the minimum intensity as a function of radius for an
		input image.
   
   Author: Mike Wall   
   Date: 4/4/93
   Version: 1.
   
   */

#include<mwmask.h>

int lminrim(DIFFIMAGE *imdiff_in)
{
  size_t
    i,
    r,
    c,
    *n,
    radius,
    index = 0;

  struct xycoords rvec;

  RFILE_DATA_TYPE *rf;

  XYZCOORDS_DATA ssq,rr;

  float cos_two_theta;

  int pidx;

  DIFFIMAGE *imdiff;

  if (imdiff_in->slist == NULL) lslistim(imdiff_in);

  rf = (RFILE_DATA_TYPE *)malloc(MAX_RFILE_LENGTH*sizeof(RFILE_DATA_TYPE));

  for (i=0;i<MAX_RFILE_LENGTH;i++) rf[i]=imdiff_in->ignore_tag;
  imdiff_in->rfile_length = 0;

  for (pidx = 0; pidx < imdiff_in->num_panels; pidx++) {

    imdiff = &imdiff_in[pidx];
    index = 0;

    struct xyzcoords s;

    for(r = 0; r < imdiff->vpixels; r++) {
      for(c = 0; c < imdiff->hpixels; c++) {
	s = imdiff->slist[index];
	ssq = ldotvec(s,s);
	cos_two_theta = 1. - ssq * imdiff->wavelength * imdiff->wavelength / 2.;
	rr = imdiff->distance_mm / cos_two_theta; 
	rvec.x = imdiff->wavelength * s.x * rr;
	rvec.y = imdiff->wavelength * s.y * rr;
	//	printf("rvec = (%f,%f)\n",rvec.x,rvec.y);
	radius = (size_t)(sqrtf(rvec.x*rvec.x + rvec.y*rvec.y)/imdiff->pixel_size_mm+.5);
	if ((imdiff->image[index] != imdiff->overload_tag) &&
	    (imdiff->image[index] != imdiff->ignore_tag)) {
	  if (radius >= imdiff_in->rfile_length) 
	    imdiff_in->rfile_length = radius+1;
	  if (rf[radius] > (RFILE_DATA_TYPE)imdiff->image[index] || rf[radius] == imdiff->ignore_tag) {
	    rf[radius] = (RFILE_DATA_TYPE)imdiff->image[index];
	  }
	}
	index++;
      }
    }
  }
  for(i=0;i<imdiff_in->rfile_length;i++) {
    if (rf[i] != imdiff_in->ignore_tag) {
      imdiff_in->rfile[i] = rf[i];
#ifdef DEBUG
      if (i>100 && i<=110) printf("lminrim rf[%d] = %g,",i,rf[i]);
#endif
    } else {
      imdiff_in->rfile[i] = 0.0;
    }
  }
  free(rf);
}

int lminrim_old(DIFFIMAGE *imdiff)
{
	size_t
		radius,
		index = 0;

	size_t i;

	RCCOORDS_DATA
		r,
		c;

	struct xycoords rvec;

	//	printf("Starting lminrim\n");
	if (imdiff->rfile != NULL) {
	  free(imdiff->rfile);
	}
	imdiff->rfile = (RFILE_DATA_TYPE *)malloc(MAX_RFILE_LENGTH*sizeof(RFILE_DATA_TYPE));
	//	printf("Allocated\n");
	for (i=0;i<MAX_RFILE_LENGTH;i++) imdiff->rfile[i]=imdiff->ignore_tag;
	//	printf("Initialized\n");
	imdiff->rfile_length = 0;
	for(r = 0; r < imdiff->vpixels; r++) {
	  rvec.y = r*imdiff->pixel_size_mm - imdiff->beam_mm.y;
	  for(c = 0; c < imdiff->hpixels; c++) {
	      rvec.x = c*imdiff->pixel_size_mm - imdiff->beam_mm.x;
	      radius = (size_t)(sqrtf(rvec.x*rvec.x + rvec.y*rvec.y)/imdiff->pixel_size_mm+.5);
	      if (radius > imdiff->rfile_length) imdiff->rfile_length = radius+1;
	      if (imdiff->image[index] != imdiff->overload_tag && imdiff->image[index] != imdiff->mask_tag && imdiff->image[index] != imdiff->ignore_tag) {
		/*		if ((imdiff->image[index] != 
			(IMAGE_DATA_TYPE)imdiff->rfile_mask_tag) && 
			((imdiff->rfile[radius] == imdiff->rfile_mask_tag) ||
		    	(imdiff->rfile[radius] > 
			(IMAGE_DATA_TYPE)imdiff->image[index]))) {*/
		if (imdiff->rfile[radius] > 
		    (RFILE_DATA_TYPE)imdiff->image[index] || imdiff->rfile[radius] == imdiff->ignore_tag) {
		  imdiff->rfile[radius] = (RFILE_DATA_TYPE)imdiff->image[index];
	        }
	      }
	  index++;
	  }
	}
	//	printf("Finished lminrim\n");
}
