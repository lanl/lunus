/* LSUBRFIM.C - Subtract I(r) from a diffraction image.
   
   Author: Mike Wall
   Date: 4/4/94
   Version: 1.
   
   */

#include<mwmask.h>

int lsubrfim(DIFFIMAGE *imdiff)
{
	size_t
		r,
		c,
		radius,
		index = 0;

	struct rccoords rvec;

	int
		return_value;

	for(r = 0; r < imdiff->vpixels; r++) {
	  for(c = 0; c < imdiff->hpixels; c++) {
	      rvec.r = r - imdiff->origin.r;
	      rvec.c = c - imdiff->origin.c;
	      radius = (size_t)sqrtf((float)(rvec.r*rvec.r + rvec.c*rvec.c));
	      if (radius < imdiff->rfile_length) {
		index = r*imdiff->hpixels+c;
	        if ((imdiff->image[index] != imdiff->overload_tag) &&
		    (imdiff->image[index] != imdiff->ignore_tag)) {
		  imdiff->image[index] -= 
			(IMAGE_DATA_TYPE)(imdiff->rfile[radius]); 
	        }
	      }
	      else {
		return_value = 1;
		sprintf(imdiff->error_msg,"\nAttempt to read past end "
			"of rfile.\n\n");
	      }
	    index++;
	  }
	}
	return(return_value);
}


