/* LREADVTK.C - Read a lattice from a .vtk file.

   Author: Mike Wall
   Date: 11/19/12
   Version: 1.

*/

#include<mwmask.h>

int lreadvtk(LAT3D *lat)
{

  int
    return_value = 0;
  
  int 
    len_inl = 100000;

  char
    *token, *inl,*inl1,*inl2;

  char
    *fgets_status;

/*
 * Read the header:
 */

  int header_done = 0;


  //  printf("Allocating input lines\n");
  inl = (char *)malloc(len_inl);
  inl1 = (char *)malloc(len_inl);
  inl2 = (char *)malloc(len_inl);

  // Read the first two lines

  //  printf("Reading the first two input lines\n");

  if (fgets(inl1,len_inl-1,lat->infile) == NULL) {
    perror("Couldn't read first line from input file\n");
    exit(1);
  }
  if (fgets(inl2,len_inl-1,lat->infile) == NULL) {
    perror("Couldn't read second line from input file\n");
    exit(1);
  }

  // Parse the second line 

  //  printf("Parsing the second line\n");

  if (strstr(inl2,"lattice_type=")!=NULL) {
    //    printf("reading lattice type...\n");
    //    printf("lattice_type=%s\n",lgettag(inl2,"lattice_type"));
    strcpy(lat->lattice_type_str,lgettag(inl2,"lattice_type"));
  }

  //  printf("Parsing the unit cell\n");

  if (strstr(inl2,"unit_cell=")!=NULL) {
    //    printf("reading unit cell...\n");
    strcpy(lat->cell_str,lgettag(inl2,"unit_cell"));
    // Parse unit cell

    lparsecelllt(lat); // parse the cell string and place results in lat->cell,
                     //    lat->astar, lat->bstar, and lat->cstar
    //        printf("unit_cell=%s\n",lat->cell_str);
    //    printf("a,b,c,alpha,beta,gamma=%f,%f,%f,%f,%f,%f\n",lat->cell.a,lat->cell.b,lat->cell.c,lat->cell.alpha,lat->cell.beta,lat->cell.gamma);
    //    printf("a = (%f,%f,%f), b = (%f,%f,%f), c = (%f,%f,%f)\n",lat->a.x,lat->a.y,lat->a.z,lat->b.x,lat->b.y,lat->b.z,lat->c.x,lat->c.y,lat->c.z);
    //    printf("astar = (%f,%f,%f), bstar = (%f,%f,%f), cstar = (%f,%f,%f)\n",lat->astar.x,lat->astar.y,lat->astar.z,lat->bstar.x,lat->bstar.y,lat->bstar.z,lat->cstar.x,lat->cstar.y,lat->cstar.z);
  }

  if (strstr(inl2,"space_group=")!=NULL) {
    //    printf("reading space group...\n");
    strcpy(lat->space_group_str,lgettag(inl2,"space_group"));
    //    printf("space_group=%s\n",lat->space_group_str);
  }

  // Parse the header, stopping when the LOOKUP_TABLE line is encountered

  printf("Reading the vtk file header...\n");

  int found_token = 1;
  int stop_header = 0;
  while (found_token == 1&& stop_header==0) {
    found_token = 0;
    if ((fgets_status=fgets(inl,len_inl-1,lat->infile)) != NULL) {      
      if ((token = strtok(inl," \n")) == NULL) {
	printf("Null...\n");
	token = inl;
      } else {
	printf("%s\n",token);
	if (strstr(token,"DIMENSIONS")!=NULL) {
	  found_token = 1;
	  lat->xvoxels = atoi(strtok(NULL," "));
	  lat->yvoxels = atoi(strtok(NULL," "));
	  lat->zvoxels = atoi(strtok(NULL," "));
	}
	if (strstr(token,"SPACING")!=NULL) {
	  found_token = 1;
	  lat->xscale = atof(strtok(NULL," "));
	  lat->yscale = atof(strtok(NULL," "));
	  lat->zscale = atof(strtok(NULL," "));
	}
	if (strstr(token,"ORIGIN")!=NULL) {
	  found_token = 1;
	  lat->xbound.min = atof(strtok(NULL," "));
	  lat->ybound.min = atof(strtok(NULL," "));
	  lat->zbound.min = atof(strtok(NULL," "));
	}
	if (strstr(token,"POINT_DATA")!=NULL) {
	  found_token = 1;
	  lat->lattice_length = atoi(strtok(NULL," "));
	}
	if (strstr(token,"SCALARS")!=NULL) {
	  found_token = 1;
	}
	if (strstr(token,"LOOKUP_TABLE")!=NULL) {
	  found_token = 1;
	  stop_header = 1;
	}
	if (strstr(token,"ASCII")!=NULL) {
	  found_token = 1;
	}
	if (strstr(token,"DATASET")!=NULL) {
	  found_token = 1;
	}
      }
    }
  }

  printf("...done\n");

  if (lat->lattice_length != lat->xvoxels*lat->yvoxels*lat->zvoxels) {
    perror("POINT_DATA isn't equal to the product of the dimensions\n");
    exit(1);
  }

  if (lat->lattice) free((LATTICE_DATA_TYPE *)lat->lattice);
  lat->lattice = (LATTICE_DATA_TYPE *)calloc(lat->lattice_length,
					     sizeof(LATTICE_DATA_TYPE)); 

  lat->xyvoxels = lat->xvoxels*lat->yvoxels;

  lat->xbound.max = lat->xbound.min + ((float)lat->xvoxels-1)*lat->xscale;
  lat->ybound.max = lat->ybound.min + ((float)lat->yvoxels-1)*lat->yscale;
  lat->zbound.max = lat->zbound.min + ((float)lat->zvoxels-1)*lat->zscale;

  lat->origin.i = (IJKCOORDS_DATA)(-lat->xbound.min/lat->xscale + .5);
  lat->origin.j = (IJKCOORDS_DATA)(-lat->ybound.min/lat->yscale + .5);
  lat->origin.k = (IJKCOORDS_DATA)(-lat->zbound.min/lat->zscale + .5);

  int index = 0,ct=0;
  int i,j,k;

  printf("size: %d %d %d\n",lat->xvoxels,lat->yvoxels,lat->zvoxels);
  printf("origin: %d %d %d\n",lat->origin.i,lat->origin.j,lat->origin.k);
  printf("(min,max) bounds: (%g,%g), (%g,%g), (%g,%g)\n",lat->xbound.min,lat->xbound.max,lat->ybound.min,lat->ybound.max,lat->zbound.min,lat->zbound.max);

  printf("Reading the vtk file data...\n");

  int linenum=0;

  printf("%s\n",inl);

  while (fgets(inl,len_inl-1,lat->infile) != NULL) {
    ++linenum;
    int ind_tmp = 0;
    token = strtok(inl," \n");
    while (token != NULL) {
      lat->lattice[index] = atof(token);
      index++;
      token = strtok(NULL," \n");
      ++ind_tmp;
    }
  }

  printf("linenum = %d\n",linenum);

  /*  
for (k=0;k<lat->zvoxels;k++) {
    for (j=0;j<lat->yvoxels;j++) {
      for (i=0;i<lat->xvoxels;i++) {
	found_token = 0;
	if ((token = strtok(NULL," \n")) == NULL) {
	  while (found_token == 0) {
	    if (fgets(inl,len_inl-1,lat->infile) == NULL) {
	      perror("Failed to complete reading of .vtk file\n");
	      printf("%d %d %d\n",i,j,k);
	      //	      exit(1);
	    }
	    if ((token = strtok(inl," \n")) != NULL) {
	      found_token = 1;
	    }
	  }
	} else {
	  found_token = 1;
	}
	lat->lattice[index] = atof(token);
	index++;
      }
    }
  }
  */  
  printf("...done\n");

  if (index != lat->lattice_length) {
    printf("\nCouldn't read all of the lattice from the input file, index = %d.\n\n",index);
    return_value = 1;
    goto CloseShop;
  }

  CloseShop:
  free(inl);
  return(return_value);
}



