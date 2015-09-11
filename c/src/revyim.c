/* REVYIM.C - Reverse the y-direction in a diffraction image.
   
   Author: Mike Wall
   Date: 2/17/2014
   Version: 1.
   
   "revyim <image in> <image out>"

   Input is image.  Output is 16-bit 
   image with y-direction reversed.

*/

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
    *imagein,
    *imageout;
  
  size_t
    i,
    num_wrote,
    num_read;

  DIFFIMAGE 
    *imdiff;

  struct rccoords
    origin;

  /*
   * Set input line defaults:
   */
  
  imagein = stdin;
  imageout = stdout;

  /*
   * Read information from input line:
   */
  switch(argc) {
  case 3:
    if (strcmp(argv[2], "-") == 0) {
      imageout = stdout;
    }
    else {
      if ( (imageout = fopen(argv[2],"wb")) == NULL ) {
	printf("Can't open %s.",argv[2]);
	exit(0);
      }
    }
  case 2:
    if (strcmp(argv[1], "-") == 0) {
      imagein = stdin;
    }
    else {
      if ( (imagein = fopen(argv[1],"rb")) == NULL ) {
	printf("Can't open %s.",argv[1]);
	exit(0);
      }
    }
    break;
  default:
    printf("\n Usage: revyim "
	   "<image in> <image out>\n\n");
    exit(0);
  }
 
  /*
   * Initialize diffraction image:
   */
    
  if ((imdiff = linitim()) == NULL) {
    perror("\nCouldn't initialize diffraction image.\n\n");
    exit(0);
  }

  /*
   * Read diffraction image:
   */

  imdiff->infile = imagein;
  if (lreadim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }

  lrevyim(imdiff);

  /*
   * Write the output image:
   */

  imdiff->outfile = imageout;
  if(lwriteim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }
   
 CloseShop:
  
  /*
   * Free allocated memory:
   */

  lfreeim(imdiff);

  /*
   * Close files:
   */
  
  fclose(imagein);
  fclose(imageout);
  
}
