/* AVSQRIM.C - Calculate the average squared pixel value vs. radius 
		for an input image.
   
   Author: Mike Wall   
   Date: 4/18/93
   Version: 1.
   
   "avsqrim <input image> <output rfile>"

   Input is TIFF TV6 image.  Output is a list of values of I^2(r).

   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
	*imagein,
	*outfile;
  
  char
    error_msg[LINESIZE];

  size_t
    num_wrote,
    num_read;

  DIFFIMAGE 
	*imdiff;

  struct rccoords origin;

/*
 * Set input line defaults:
 */
	
	imagein = stdin;
	outfile = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
		case 3:
			if ( (outfile = fopen(argv[2],"wb")) == NULL ) {
				printf("Can't open %s.",argv[2]);
				exit(0);
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
			printf("\n Usage: avsqrim <input image> "
				"<output rfile> <x origin> <y origin>\n\n");
			exit(0);
	}
  

/*
 * Initialize diffraction image:
 */

  if ((imdiff = linitim(1)) == NULL) {
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


  /*
   * Generate rfile:
   */
  
	lavsqrim(imdiff);  

/*
 * Write the output rfile:
 */

   imdiff->outfile = outfile;
   if(lwriterf(imdiff) != 0) {
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
  fclose(outfile);
  
}

