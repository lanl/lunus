/* LAT2VTK.C - Convert internal .lat format to standard .vtk format.
   
   Author: Mike Wall
   Date: 10/1/2012
   Version: 1.
   
   Usage:
   		"lat2vtk <input lattice> <output lattice>"

		Input is a lattice in internal format. 
			Output is a lattice in VTK format.  
   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
    *latticein,
    *latticeout;
  
  char
    error_msg[LINESIZE],filename[256];
  
  size_t
    i,
    j,
    k,
    num_read,
    num_wrote;
  
  LAT3D 
    *lat;
  
/*
 * Set input line defaults:
 */
	latticein = stdin;
	latticeout = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
	  case 3:
	  if (strcmp(argv[2],"-") == 0) {
	    latticeout = stdout;
	  }
	  else {
	    if ((latticeout = fopen(argv[2],"wb")) == NULL) {
	      printf("\nCan't open %s.\n\n",argv[2]);
	      exit(0);
	    }
	  }
	  case 2:
	  if (strcmp(argv[1],"-") == 0) {
	    latticein = stdin;
	  }
	  else {
	    strcpy(filename,argv[1]);
	    if ( (latticein = fopen(argv[1],"rb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[1]);
	      exit(0);
	    }
	  }
	  break;
	  default:
	  printf("\n Usage: lat2vtk <input lattice> <output lattice>\n\n");
	  exit(0);
	}
  
  /*
   * Initialize lattice:
   */

  if ((lat = linitlt()) == NULL) {
    perror("Couldn't initialize lattice.\n\n");
    exit(0);
  }
  
  /*
   * Read in lattice:
   */

  lat->filename = filename;
  lat->infile = latticein;
  if (lreadlt(lat) != 0) {
    perror("Couldn't read lattice.\n\n");
    exit(0);
  }

/*
 * Write lattice to output file:
 */

  printf("Writing file\n");

  lat->outfile = latticeout;
  if (lwritevtk(lat) != 0) {
    perror("Couldn't write lattice.\n\n");
    exit(0);
  }

  printf("Done writing file\n");

CloseShop:
  
  /*
   * Free allocated memory:
   */

  lfreelt(lat);

  /*
   * Close files:
   */
  
  fclose(latticein);
  fclose(latticeout);
}

