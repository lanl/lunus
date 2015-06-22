/* NORMIM.C - Correct for solid-angle normalization and 
              detector-face rotation in a diffraction image.
   
   Author: Mike Wall
   Date: 2/24/95
   Version: 1.
   
   "normim <image in> <image out> <x origin> <y origin>
      <distance[mm]> <cassette rotx> <cassette roty>" 

   Input is diffraction image.  Output is normalized corrected image.

   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
	*imagein,
	*imageout;
  
  size_t
    i;

  DIFFIMAGE 
	*imdiff;

  struct rccoords
	origin;

  float 
    distance_mm;

  struct xyzcoords
    cassette;

/*
 * Set input line defaults:
 */

  origin.r = DEFAULT_IMAGE_ORIGIN;
  origin.c = DEFAULT_IMAGE_ORIGIN;
  distance_mm = DEFAULT_DISTANCE_MM;
  cassette.x = DEFAULT_CASSETTE_ROTX;
  cassette.y = DEFAULT_CASSETTE_ROTY;
  imagein = stdin;
  imageout = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
	  case 8:
	  cassette.y = (XYZCOORDS_DATA)atof(argv[7]);
	  case 7:
	  cassette.x = (XYZCOORDS_DATA)atof(argv[6]);
	  case 6:
	  distance_mm = atof(argv[5]);
	  case 5:
	  origin.r = (RCCOORDS_DATA)atol(argv[4]);
	  case 4:
	  origin.c = (RCCOORDS_DATA)atol(argv[3]);
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
	  printf("\n Usage: normim <image in> <image out> "
		 "<x origin> <y origin> <distance [mm]>\n\n");
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
   * Set the origin for the image:
   */
  
  imdiff->origin = origin;

  /*
   * Set sample-to-detector distance for the image
   */

  imdiff->distance_mm = distance_mm;
  
  /*
   * Read diffraction image:
   */
  
  imdiff->infile = imagein;
  if (lreadim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }
  
  /*
   * Normalize the diffraction image:
   */
  
  imdiff->cassette.x = cassette.x;
  imdiff->cassette.y = cassette.y;
  lnormim(imdiff);
  
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


