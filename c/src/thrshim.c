/* THRSHIM.C - Set all pixels above and below given thresholds equal to an 
		ignore_tag.
   
   Author: Mike Wall
   Date: 5/13/94
   Version: 1.
   
   "thrshim <image in> <lower> <upper> <image out>" 

   Input is image and upper and lower thresholds.  Output is input image 
   with all pixel values outside of thresholds converted to an ingore_tag.

   */

#include<mwmask.h>

/*
 * Parse a threshold argument as a decimal integer. Returns 0 and stores the
 * value on success. On failure prints a message naming the argument to
 * stderr and returns 1. Rejects an empty string, trailing characters
 * (atoi would silently read "abc" as 0 and "12abc" as 12), and values that
 * do not fit in IMAGE_DATA_TYPE (atoi's cast would silently wrap them).
 */
static int parse_threshold(const char *s, const char *which,
			   IMAGE_DATA_TYPE *value)
{
  char *end;
  long v;

  errno = 0;
  v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "thrshim: %s threshold '%s' is not an integer\n",
	    which, s);
    return 1;
  }
  if (errno == ERANGE || (long)(IMAGE_DATA_TYPE)v != v) {
    fprintf(stderr, "thrshim: %s threshold '%s' is out of range for "
	    "the image data type\n", which, s);
    return 1;
  }
  *value = (IMAGE_DATA_TYPE)v;
  return 0;
}

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
	upper_threshold,
	lower_threshold;

/*
 * Set input line defaults:
 */
	
	imagein = stdin;
	imageout = stdout;

/*
 * Read information from input line. The thresholds are checked first so
 * that a bad argument is rejected before any output file is created.
 */
	if (argc >= 4) {
		if (parse_threshold(argv[2], "lower", &lower_threshold) != 0 ||
		    parse_threshold(argv[3], "upper", &upper_threshold) != 0) {
			exit(1);
		}
	}
	switch(argc) {
		case 5:
			if (strcmp(argv[4], "-") == 0) {
				imageout = stdout;
			}
			else {
			 if ( (imageout = fopen(argv[4],"wb")) == NULL ) {
				printf("Can't open %s.",argv[4]);
				exit(0);
			 }
			}
                case 4:
			/* both thresholds were parsed and checked above;
			   fewer arguments would leave them unset */
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
			printf("\n Usage: thrshim <image in> <lower> <upper> "
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

   imdiff->lower_threshold = lower_threshold;
   imdiff->upper_threshold = upper_threshold;
   lthrshim(imdiff);

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

