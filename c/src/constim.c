/* CONSTIM - Create an image with constant pixel value based on an input
		image format.
   
   Author: Mike Wall
   Date: 4/18/94
   Version: 1.
   
   "constim <image in> <constant> <image out>" 

   Input is image and constant.  Output is constant image in the
        format of the input image.

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

  IMAGE_DATA_TYPE
	image_value;

/*
 * Set input line defaults:
 */
	
	imagein = stdin;
	imageout = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
		case 4:
			if (strcmp(argv[3], "-") == 0) {
				imageout = stdout;
			}
			else {
			 if ( (imageout = fopen(argv[3],"wb")) == NULL ) {
				printf("Can't open %s.",argv[3]);
				exit(0);
			 }
			}
                case 3:
			image_value = atof(argv[2]);
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
			printf("\n Usage: constim <image in> <constant> "
			       "<image out>\n\n");
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
 * Set image to constant:
 */

   imdiff->image[0] = (IMAGE_DATA_TYPE)image_value;
   lconstim(imdiff);

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

