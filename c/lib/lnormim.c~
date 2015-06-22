/* LNORMIM.C - Correct for solid-angle normalization and 
              detector-face rotation in a diffraction image.
   
   Author: Mike Wall
   Date: 2/24/95
   Version: 1.
   
   */

#include<mwmask.h>

int lnormim(DIFFIMAGE *imdiff)
{
  size_t 
    index = 0;
  
  RCCOORDS_DATA
    r,
    c;
  
  struct xycoords
    rvec;
  
  int 
    return_value = 0;
  
  
  float 
    correction_factor,
    radius_squared,
    distance,
    distance_squared;
  
  distance = imdiff->distance_mm / imdiff->pixel_size_mm;
  distance_squared = distance*distance;
  for(r=0; r < imdiff->vpixels; r++) {
    rvec.y = (float)(r - imdiff->origin.r);
    for(c=0; c < imdiff->hpixels; c++) {
      if ((imdiff->image[index] != imdiff->overload_tag) &&
	  (imdiff->image[index] != imdiff->ignore_tag)) {
	rvec.x = (float)(c - imdiff->origin.c);
	radius_squared = ((rvec.x*rvec.x) + (rvec.y*rvec.y));
	correction_factor = 1. + radius_squared/distance_squared - 
	  2*PI/180.*(rvec.x*imdiff->cassette.y -
	     rvec.y*imdiff->cassette.x)/distance;
	imdiff->image[index] *= correction_factor * sqrtf(correction_factor);
      }
      index++;
    }
  }
  return(return_value);
}








