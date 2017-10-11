/* LPOLARIM.C - Perform polarization correction on a diffraction image.
   
   Author: Mike Wall
   Date: 4/26/94
   Version: 1.
   
   */

#include<mwmask.h>

int lpolarim(DIFFIMAGE *imdiff)
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
    radius,
    radius_squared,
    distance_pixels,
    arctan_argument,
    two_theta,
    cos_two_theta,
    sin_two_theta,
    cos_two_rho,
    two_rho_offset;

  two_rho_offset = 2.*PI/180.*imdiff->polarization_offset;
  for(r=0; r < imdiff->vpixels; r++) {
    // rvec.y = (float)(r - imdiff->origin.r);
    rvec.y = (float)(r*imdiff->pixel_size_mm-imdiff->beam_mm.y);
    for(c=0; c < imdiff->hpixels; c++) {
      if ((imdiff->image[index] != imdiff->overload_tag) &&
	  (imdiff->image[index] != imdiff->ignore_tag)) {
	// rvec.x = (float)(c - imdiff->origin.c);
  rvec.x = (float)(c*imdiff->pixel_size_mm-imdiff->beam_mm.x);
	radius_squared = ((rvec.x*rvec.x) + (rvec.y*rvec.y));
	radius = sqrtf(radius_squared);
	// distance_pixels= imdiff->distance_mm / imdiff->pixel_size_mm;
	arctan_argument = radius / imdiff->distance_mm;
	if (arctan_argument > POLARIZATION_CORRECTION_THRESHOLD) {
	  two_theta = atanf(arctan_argument);
	  cos_two_theta = cosf(two_theta);
	  sin_two_theta = sinf(two_theta);
	  cos_two_rho = cos(2*acosf(rvec.x / radius) - two_rho_offset);
	  imdiff->image[index] -= imdiff->value_offset;
	  imdiff->image[index]=(IMAGE_DATA_TYPE)((float)imdiff->image[index] *
			       2. / (1. + cos_two_theta*cos_two_theta -
			       imdiff->polarization*cos_two_rho*
			         sin_two_theta*sin_two_theta));
	  imdiff->image[index] += imdiff->value_offset;
	}
      }
      index++;
    }
  }
  return(return_value);
}
