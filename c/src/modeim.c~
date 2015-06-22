/* MODEIM.C - Remove the peaks from an image by mode filtering using a
              specified mask size.
   
   Author: Mike Wall 
   Date: 2/2/95
   Version: 1.
   
   "modeim <image in> <image out> <mask size> <bin size>"

   Input is ascii coordinates file.  Output is 16-bit 
   image of specified size (1024 x 1024 default).

   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
	*imagein,
	*imageout;
  
  DIFFIMAGE 
	*imdiff;

  int
    mask_size,
    binsize;
/*
 * Set input line defaults:
 */
	
  mask_size = DEFAULT_MODE_MASK_SIZE;
  binsize = DEFAULT_MODE_BIN_SIZE;
  imagein = stdin;
  imageout = stdout;
  
/*
 * Read information from input line:
 */
	switch(argc) {
	  case 5:
	  binsize = atoi(argv[4]);
	  case 4:
	  mask_size = atoi(argv[3]);
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
	  printf("\n Usage: modeim "
		 "<image in> <image out> <mask size> <bin size>\n\n");
	  exit(0);
	}
  
/*
 * Initialize diffraction image:
 */
  
  if ((imdiff = linitim()) == NULL) {
    perror("Couldn't initialize diffraction image.\n\n");
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

  /*
   * Initialize mask parameters:
   */

  imdiff->mode_height = mask_size - 1;
  imdiff->mode_width = mask_size - 1;
  imdiff->mode_binsize = binsize;

/*
 * Mode image:
 */

  lmodeim(imdiff);

/*
 * Write the output image:
 */

  imdiff->outfile = imageout;
  if(lwriteim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }

CloseShop:
  
  lfreeim(imdiff);

/*
 * Close files:
 */
  
  fclose(imagein);
  fclose(imageout);
  
}

