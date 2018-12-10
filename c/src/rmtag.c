/* ADDTAG.C - Remove tag info from .img header.
   
   Author: Mike Wall
   Date: 6/27/2018
   Version: 1.
   
   "rmtag <image in> <image out> <tag name>" 

   Input is input image, output image, tag name.  Output is input image with tag deleted

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
  char
    *buf;

  char *tag,*tagval;
  
  tag = (char *)malloc(1000);
  tagval = (char *)malloc(1000);
  

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
    strcpy(tag,argv[3]);
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
    printf("\n Usage: addtag <image in> "
	   "<image out> <tag name> <tag value> \n\n");
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

  /*
   * Change the header:
   */

  buf = imdiff->header;
  if (strstr(buf,tag)==NULL) {
    printf("Couldn't find tag\n");
    exit(1);
  }
  else {
    printf("Found tag\n");
    char *startpos,*endpos,*tagstr;
    int vallen;
    startpos = strstr(buf,tag);
    //    startpos = strstr(buf,tag)+strlen(tag)+1;
    endpos = strchr(startpos,0x3b);
    //    strcpy(startpos,endpos);
    vallen = endpos - startpos;
    //    printf("length of value string = %d\n",vallen);
    //    if (vallen != strlen(tagval)) {
    //      printf("Length of tagval doesn't match current length\n");
    //      exit(1);
    //    }
    memcpy(startpos-2,endpos,imdiff->header_length-vallen);
  }
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

