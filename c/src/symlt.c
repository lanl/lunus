/* SYMLT.C - Symmetrize a lattice according to input line instructions.
   
   Author: Mike Wall
   Date: 2/28/95
   Version: 1.
   
   Usage:
   		"symlt <input lattice> <output lattice> <symmetry_operation>"

		Input are lattice and symmetry operation
		specification.  Output is symmetrized lattice.
   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
	*latticein,
	*latticeout;

  char
    error_msg[LINESIZE];

  size_t
    symop;

  LAT3D 
	*lat;

  RFILE_DATA_TYPE *rfile;

  struct ijkcoords
    origin;

/*
 * Set input line defaults:
 */
	
	latticein = stdin;
	latticeout = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
    case 7: 
    origin.k = atol(argv[6]);
    case 6:
    origin.j = atol(argv[5]);
    case 5:
    origin.i = atol(argv[4]);
	  case 4:
	  symop = atol(argv[3]);
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
	    if ( (latticein = fopen(argv[1],"rb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[1]);
	      exit(0);
	    }
	  }
	  break;
	  default:
	  printf("\n Usage: symlt <input lattice> "
		 "<output lattice> <symmetry_operation>\n\n"
		 "  Symmetry Operations:\n"
		 "    0 = P 1\n"
		 "    1 = P 4\n"
		 "    2 = P 2 2 2\n\n"
		 "    3 = P m -3\n");
	      
	  exit(0);
	}
  
  /*
   * Initialize lattices:
   */

  if ((lat = linitlt()) == NULL) {
    perror("Couldn't initialize lattice.\n\n");
    exit(0);
  }
  
  /*
   * Read in lattice:
   */

  lat->infile = latticein;
  if (lreadlt(lat) != 0) {
    perror("Couldn't read lattice.\n\n");
    exit(0);
  }

  /*
   * Perform symmetry operation:
   */

  lat->symop_index = symop;
  if (argc==7) {
    lat->origin.i=origin.i; lat->origin.j=origin.j; lat->origin.k=origin.k;
  }
  lsymlt(lat);
  
  /*
   * Write lattice to output file:
   */
  
  lat->outfile = latticeout;
  if (lwritelt(lat) != 0) {
    perror("Couldn't write lattice.\n\n");
    exit(0);
  }

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


