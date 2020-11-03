/* AVGRLT.C - Calculate the average voxel value vs. radius for an 
              input lattice
   
   Author: Mike Wall
   Date: 3/1/95
   Version: 1.
   
   Usage:
   		"avgrlt <input lattice> <output rfile> <cell>"

		Input is a 3D lattice and voxel coordinates for the
		origin.  Output is an rfile.  
   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
    *latticein,
    *outfile;
  
  char
    cell_str[256],
    error_msg[LINESIZE];
  
  size_t
    i,
    j,
    k,
    num_wrote;
  
  LAT3D 
    *lat;
  
  RFILE_DATA_TYPE *rfile;
  
  /*
   * Set input line defaults:
   */
  
  latticein = stdin;
  outfile = stdout;
  
  strcpy(cell_str,"None");

  /*
   * Read information from input line:
   */
  switch(argc) {
    case 4:
      strcpy(cell_str,argv[3]);
    case 3:
    if ((outfile = fopen(argv[2],"wb")) == NULL) {
      printf("\nCan't open %s.\n\n",argv[2]);
      exit(0);
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
    printf("\n Usage: avgrlt <input lattice> "
	   "<output rfile> <unit cell string>\n\n");
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

  lat->infile = latticein;
  if (lreadlt(lat) != 0) {
    perror("Couldn't read lattice.\n\n");
    exit(0);
  }

/*
 * Generate the radially averaged image:
 */

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
  lavgrlt(lat);

/*
 * Write rfile to output file:
 */

  num_wrote = fwrite(lat->rfile, sizeof(RFILE_DATA_TYPE),
		     lat->rfile_length, outfile);
  if (num_wrote != lat->rfile_length) {
    printf("\nCouldn't write rfile.\n\n");
    goto CloseShop;
  }

CloseShop:
  
  /*
   * Free allocated memory:
   */

  lfreelt(lat);

  /*
   * Close files:
   */
  
  fclose(outfile);
  fclose(latticein);
}

