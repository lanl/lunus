/* RAW2IM.C - Convert a raw image to a SMV or CBF file, given a template
   
   Author: Mike Wall 
   Date: 1/15/16
   Version: 1.
   
   "raw2im <image in> <image out> <template>"

   Input is raw image and template image. Output is a new image using the raw data 

   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
	*imagein,
        *rawin,
	*imageout;
  
  DIFFIMAGE 
	*imdiff;

  int
    mask_size,
    binsize;
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
	    imagein = stdin;
	  }
	  else {
	    if ( (imagein = fopen(argv[3],"rb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[3]);
	      exit(0);
	    }
	  }
	  case 3:
	  if (strcmp(argv[2], "-") == 0) {
	    imageout = stdout;
	  }
	  else {
	    if ( (imageout = fopen(argv[2],"wb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[2]);
	      exit(0);
	    }
	  }
	  case 2:
	  if (strcmp(argv[1], "-") == 0) {
	    imagein = stdin;
	  }
	  else {
	    if ( (rawin = fopen(argv[1],"rb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[1]);
	      exit(0);
	    }
	  }
	  break;
	  default:
	  printf("\n Usage: raw2img "
		 "<image in> <image out> <template>\n\n");
	  exit(0);
	}
  
/*
 * Initialize diffraction image:
 */
  
  if ((imdiff = linitim(1)) == NULL) {
    perror("Couldn't initialize diffraction image.\n\n");
    exit(0);
  }

/*
 * Read template diffraction image:
 */

  imdiff->infile = imagein;
  if (lreadim(imdiff) != 0) {
    perror(imdiff->error_msg);
    goto CloseShop;
  }

  // Read raw image

  void *buf;
  size_t num_read;
  size_t imsz;

  imsz = imdiff->image_length*sizeof(IMAGE_DATA_TYPE);
  buf = (void *)malloc(imsz);
		      
	   

  num_read = fread(buf, sizeof(char), imsz, rawin);

  // Copy raw image to .img image

  memcpy(imdiff->image,buf,imsz);

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

