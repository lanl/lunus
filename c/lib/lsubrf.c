/* LSUBRF.C - Subtract one rfile from another.
   
   Author: Mike Wall
   Date: 5/1/94
   Version: 1.
   
   */

#include<mwmask.h>

int lsubrf(DIFFIMAGE *imdiff1, DIFFIMAGE *imdiff2)
{
	size_t
		radius;

	int
		return_value = 0;

  for(radius = 0; radius < imdiff1->rfile_length; radius++) {
    if (imdiff1->rfile[radius] != imdiff1->rfile_mask_tag) {
      if ((imdiff2->rfile[radius] != imdiff2->rfile_mask_tag)) {
        imdiff1->rfile[radius]=imdiff1->rfile[radius]-imdiff2->rfile[radius];
      }
    }
    else {
      imdiff1->rfile[radius] = imdiff1->rfile_mask_tag;
    }
  }
  return(return_value);
}
