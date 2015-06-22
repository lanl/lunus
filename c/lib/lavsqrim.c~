/* LAVSQRIM.C - Calculate the average intensity squared as a function of 
	radius for a diffraction image.

   Author: Mike Wall
   Version:1.0
   Date:4/7/94
	
*/

#include<mwmask.h>

int lavsqrim(DIFFIMAGE *imdiff)
{
	RCCOORDS_DATA
		r,
		c;
	size_t
		*n,
		radius,
		index = 0;

	struct rccoords rvec;

	n = (size_t *)calloc(MAX_RFILE_LENGTH, sizeof(size_t));
	imdiff->rfile_length = 0;
	for(r = 0; r < imdiff->vpixels; r++) {
	  for(c = 0; c < imdiff->hpixels; c++) {
	      rvec.r = r - imdiff->origin.r;
	      rvec.c = c - imdiff->origin.c;
	      radius = (size_t)sqrtf((float)(rvec.r*rvec.r) + 
			(float)(rvec.c*rvec.c));
	      if ((imdiff->image[index] != imdiff->overload_tag) &&
		  (imdiff->image[index] != imdiff->ignore_tag)) {
	        if (radius > imdiff->rfile_length) 
			imdiff->rfile_length = radius;
		imdiff->rfile[radius] = (RFILE_DATA_TYPE)(
			(float)imdiff->image[index]*
			(float)imdiff->image[index]+ 
			(float)n[radius]*(float)imdiff->rfile[radius]) /
				(RFILE_DATA_TYPE)(n[radius] + 1);
		n[radius]++;
	      }
	    index++;
	  }
	}
	free((size_t *)n);
}
