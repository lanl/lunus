/* AVGR.C - Generate a table of average voxel values as a function of radius
		for an input 3D lattice.
   
   Author: Mike Wall
   Date: 3/28/93
   Version: 1.
   
   Usage:
   		"avgr <input file> <output file>"

   Input is a 3D lattice.  Output is a list of average value by radius, to be 
	labelled (0,1,2,3,...).

   */

#include<mwmask.h>

int main(argc, argv)
	short argc;
	char *argv[];
{
  FILE
	*infile,
	*outfile;

  char
    error_msg[LINESIZE];

  size_t
	i,
	j,
	k,
	xyvoxels,
	num_read,
	num_wrote;

  LAT3D 
	*lat;

  RFILE_DATA_TYPE *rfile;

  float 
	xscale,
	yscale,
	zscale;

/*
 * Set input line defaults:
 */
	
	infile = stdin;
	outfile = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
		case 3:
			if (strcmp(argv[2],"-") == 0) {
				outfile = stdout;
			}
			else {
			 if ((outfile = fopen(argv[2],"wb")) == NULL) {
				printf("\nCan't open %s.\n\n",argv[2]);
				exit(0);
			 }
			}
		case 2:
			if (strcmp(argv[1],"-") == 0) {
				infile = stdin;
			}
			else {
			 if ( (infile = fopen(argv[1],"rb")) == NULL ) {
				printf("\nCan't open %s.\n\n",argv[1]);
				exit(0);
			 }
			}
		case 1:
			break;
		default:
			printf("\n Usage: avgr <input file> "
				"<output file>\n\n");
			exit(0);
	}
  
/*
 * Allocate memory for lattice:
 */

  lat = (LAT3D *)malloc(sizeof(LAT3D));
  if (!lat) {
    printf("\n***Unable to allocate all memory.\n");
    goto CloseShop;
  }
  
/*
 * Set main defaults:
 */

	lat->infile = infile;
	lat->outfile = outfile;
	lat->mask_tag = DEFAULT_LATTICE_MASK_TAG;

/*
 * Read in 3D lattice descriptor:
 */

  num_read = fread(&lat->xvoxels, sizeof(long), 1, infile);
  num_read = fread(&lat->yvoxels, sizeof(long), 1, infile);
  num_read = fread(&lat->zvoxels, sizeof(long), 1, infile);
  num_read = fread(&lat->xbound, sizeof(struct bounds), 1, infile);
  num_read = fread(&lat->ybound, sizeof(struct bounds), 1, infile);
  num_read = fread(&lat->zbound, sizeof(struct bounds), 1, infile);
  xyvoxels = lat->xvoxels * lat->yvoxels;
  lat->lattice_length = xyvoxels * lat->zvoxels;
  lat->rfile_length = 0;

/*
 * Allocate memory for 3D lattice:
 */

  lat->lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,
				sizeof(LATTICE_DATA_TYPE));
  if (!lat->lattice) {
	printf("\nNot enough room to allocate 3D lattice.\n\n");
	exit(0);
  }

/*
 * Read in lattice:
 */

  num_read = fread(lat->lattice, sizeof(LATTICE_DATA_TYPE), lat->lattice_length,
						infile);
  if (num_read != lat->lattice_length) {
    printf("/nCouldn't read all of the lattice from input file.\n\n");
    exit(0);
  }

/*
 * Allocate memory for rfile:
 */

  lat->rfile = (LATTICE_DATA_TYPE *)calloc(MAX_RFILE_LENGTH, 
					sizeof(LATTICE_DATA_TYPE));
  if (!lat->lattice) {
  	printf("\nNot enough room to allocate 3D lattice.\n\n");
  	exit(0);
  }

/*
 * Generate average rfile:
 */

  lavgr(lat);

/*
 * Write rfile to output file:
 */

  num_wrote = fwrite(lat->rfile, sizeof(RFILE_DATA_TYPE), lat->rfile_length, 
				outfile);
CloseShop:
  
  /*
   * Free allocated memory:
   */

  free((LATTICE_DATA_TYPE *)lat->lattice);
  free((LATTICE_DATA_TYPE *)lat->rfile);
  free((LAT3D *)lat);

  /*
   * Close files:
   */
  
  fclose(infile);
  fclose(outfile);
}

