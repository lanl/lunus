/* LPUNCH.C - Mark all pixels inside a window in an image with an ignore tag.
   
   Author: Mike Wall  
   Date: 5/12/94
   Version: 1.
   
   */

#include<mwmask.h>

int lpunch(DIFFIMAGE *imdiff)
{
  long i;
  short
    lpunch_return,
    r,
    c;
  
  
  for (i=0; i<imdiff->mask_count; i++) {
    r = imdiff->pos.r+imdiff->mask[i].r;
    c = imdiff->pos.c+imdiff->mask[i].c;
    if (!((r < 0) || (r > imdiff->vpixels) || (c < 0) ||       
	  (c > imdiff->hpixels))) {
      imdiff->image[r*imdiff->hpixels + c] = 
	imdiff->punch_tag;
      lpunch_return = 0;
    } else lpunch_return = 1;
  }
  return(lpunch_return);
}
