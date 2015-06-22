/* SEE.C - Convert a 16-bit 1024x1024 TV6 tiff image to an 8-bit tiff image.
		Scale to window given at input command line.
   Author: Mike Wall
   Date: 2/25/94
   Version: 1.0
   Description:

		"see <input file> <output file> <lower threshold> 
			<upper threshold>"

		Input is a 16-bit 1024x1024 TV6 tiff image.  
		Output is an 8-bit tiff image.
*/

#define LINESIZE 120
#define DEFAULT_HEADER_LENGTH 4096
#define DEFAULT_FILE_SIZE 2101248
#define DEFAULT_LOWER_THRESHOLD 256
#define DEFAULT_UPPER_THRESHOLD 32767
#define MAX_VALUE 32767

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

union data_union {
	short two_byte;
	struct {
		char low_byte;
		char high_byte;
	} one_byte;
};

int main(int argc, char *argv[])
{
	FILE
	    *f_in,
	    *f_out;

	char 
	*filename;

	char
	*outstream,
	    *header;

	short
	upper_threshold = DEFAULT_UPPER_THRESHOLD,
	    lower_threshold = DEFAULT_LOWER_THRESHOLD,
		pix_value;

	size_t
	    filesize,
	    imagesize,
	    imagelength,
	    num_read,
	    num_wrote,
	    i,
	    header_length = DEFAULT_HEADER_LENGTH;

	float
		scale;

	short
	*instream;

	union data_union
	pix;

	filename = (char *)malloc(sizeof(char)*(LINESIZE+1));
	if (!filename) {
		printf("Can't allocate image arrays.  Exiting.\n");
		exit(0);
	}
	f_out = stdout;
	switch(argc) {
		case 5:
		strcpy(filename,argv[4]);
		if (strcmp(filename,"-") == 0) {
			f_out = stdout;
		}
		else
		if ( (f_out = fopen(filename,"wb")) == NULL ) {
			printf("Can't open %s.",filename);
			exit(0);
		}
		case 4:
		upper_threshold = atoi(argv[3]);
		case 3:
		lower_threshold = atoi(argv[2]);
		case 2:
		/*
 * Open input and output files:
 */

		strcpy(filename,argv[1]);
		if (strcmp(filename,"-") == 0) {
			f_in = stdin;
		}
		else
		if ( (f_in = fopen(filename,"rb")) == NULL ) {
			printf("Can't open %s.",filename);
			exit(0);
		}
		break;
	default:

		printf("\n     Usage: see <input file> "
		    "<lower threshold> "
		    "<upper threshold> <output file>\n\n");
		exit(0);
	}

	/*
 * Allocate memory for the header:
 */

	header = (char *)malloc(sizeof(char)*(header_length+1));
	if (!header) {
		printf("Can't allocate header.  Exiting.\n");
		exit(0);
	}

	/*
 * Read the input image:
 */

	filesize = DEFAULT_FILE_SIZE;
	imagesize = filesize-header_length;
	imagelength = imagesize/2;
	outstream = (char *)malloc(sizeof(char)*(imagelength+1));
	instream = (short *)malloc(sizeof(short)*(imagelength+1));
	if (!(outstream && instream)) {
		printf("Can't allocate image arrays.  Exiting.\n");
		exit(0);
	}
	num_read = fread(header,sizeof(char),header_length,f_in);
	/*
 * Modify header to indicate 8-bit tiff:
 */

	header[0x88] = 0x08;		/* Bits per pixel */
	header[0xDE] = 0x10;		/* Strip size change */
	header[0x178] = 0x08;		/* TV6 Significant bpp */

	num_read = fread(instream,sizeof(short),imagelength,f_in);
	if (ferror(f_in)) {
		printf("Error reading file %s.",filename);
	}


	/*
 * Generate output image:
 */

	scale = 255. / (float)(upper_threshold - lower_threshold);
	i = 0;
	while (i < imagelength) {
		pix.two_byte = instream[i];
		pix_value = (short)pix.one_byte.high_byte * 256 + 
			(short)pix.one_byte.low_byte;
		if ((pix_value < lower_threshold) || (pix_value >= 32766)) {
			outstream[i] = 0;
		}
		else if (pix_value > upper_threshold) {
			outstream[i] = 255;
		}
		else {
			outstream[i] = (char)((scale * (pix_value - 
				lower_threshold)));
		}
		i++;
	}


	/* 
 * Write TIFF header:
 */

	num_wrote = fwrite(header, sizeof(char), header_length, f_out);

	/*
 * Write the output image:
 */

	num_wrote = fwrite(outstream, sizeof(char), imagelength, f_out);

	/*
 * Free allocated memory:
 */

	free((char *)filename);
	free((short *)instream);
	free((char *)outstream);
	free((char *)header);
}
