/* ANISOLT.C - Calculate the anisotropic component of a lattice.
   
   Author: Mike Wall
   Date: 9/27/2016
   Version: 1.
   
   Usage:
   		"anisolt <input lattice> <output lattice> <unit_cell_string>"

   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
    *latticein,
    *latticeout;
  
  char
    cell_str[256],
    error_msg[LINESIZE];
  
  int
    procmode = 0;

  size_t
    i,
    j,
    k,
    num_wrote;
  
  LAT3D 
    *lat;
  
  /*
   * Set input line defaults:
   */
  
  latticein = stdin;
  latticeout = stdout;

  strcpy(cell_str,"None");
  
  /*
   * Read information from input line:
   */
  switch(argc) {
  case 5:
    procmode = atoi(argv[4]);
    case 4:
      strcpy(cell_str,argv[3]);
    case 3:
    if (strcmp(argv[2],"-") == 0) {
      latticeout = stdout;
    }
    else {
      if ( (latticeout = fopen(argv[2],"wb")) == NULL ) {
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
    printf("\n Usage: anisolt <input lattice> <output lattice> "
	   "<cell_str> <run mode>\n\n");
    exit(0);
  }
  
  /*
   * Initialize lattice:
   */
  
  if (((lat = linitlt()) == NULL)) {
    perror("Couldn't initialize lattice.\n\n");
    exit(0);
  }

  /*
   * Read in lattices:
   */

  lat->infile = latticein;
  if (lreadlt(lat) != 0) {
    perror("Couldn't read lattice.\n\n");
    exit(0);
  }

  // Calculate anisotropic component of lattices

  if (!(strcmp(cell_str,"None")==0)) {
    strcpy(lat->cell_str,cell_str);
  } else {
    lat->cell.a = 1./lat->xscale;
    lat->cell.b = 1./lat->yscale;
    lat->cell.c = 1./lat->zscale;
    lat->cell.alpha = lat->cell.beta = lat->cell.gamma = 90.0;
    sprintf(cell_str,"%f,%f,%f,90.0,90.0,90.0",(float)lat->cell.a,(float)lat->cell.b,(float)lat->cell.c);
    printf("%s\n",cell_str);
  }
  lparsecelllt(lat);
  lat->procmode = procmode;
  lanisolt(lat);

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


