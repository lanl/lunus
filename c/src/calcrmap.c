/* CALCRMAP.C - Calculate an R factor and other figures of merit from a map and reflections.
   
   Author: Mike Wall
   Date: 5/4/2015
   Version: 1.
   
   Usage:
   		"calcrmap <input reflections .hkl> <input ccp4 map>"

		Input is a .hkl reflections file and a map in CCP4 format. 
			Output is R factor and other figures of merit to stdout.  
   */

#include<mwmask.h>

int main(int argc, char *argv[])
{
  FILE
    *mapin,
    *hklin,
    *mapout;
  
  char
    error_msg[LINESIZE],filename[256];
  
  size_t
    fft_index,
    map_index,
    friedel_index,
    num_read,
    num_wrote;

  CCP4MAP
    *map;

  LAT3D 
    *lat;

  int
    h,
    k,
    l,
    nn[4],
    c,r,s,cc,rr,ss;
  
  float
    I,sigI,Imodel,
    F,sigF,
    freal,fimag;
/*
 * Set input line defaults:
 */
	hklin = stdin;
	mapout = stdout;

/*
 * Read information from input line:
 */
	switch(argc) {
	  case 4:
	    if (strcmp(argv[3],"-") == 0) {
	      mapout = stdout;
	    }
	  else {
	    if ((mapout = fopen(argv[3],"wb")) == NULL) {
	      printf("\nCan't open %s.\n\n",argv[3]);
	      exit(0);
	    }
	  }
	  case 3:
	    if ( (mapin = fopen(argv[2],"rb")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[2]);
	      exit(0);
	    }
	  case 2:
	  if (strcmp(argv[1],"-") == 0) {
	    hklin = stdin;
	  }
	  else {
	    if ( (hklin = fopen(argv[1],"r")) == NULL ) {
	      printf("\nCan't open %s.\n\n",argv[1]);
	      exit(0);
	    }
	  }
	  break;
	  default:
	  printf("\n Usage: calcrmap <input reflections .hkl> <input ccp4 map>\n\n");
	  exit(0);
	}
  
  /*
   * Initialize map:
   */

  if ((map = linitmap()) == NULL) {
    perror("Couldn't initialize map.\n\n");
    exit(0);
  }
  
  /*
   * Read in map:
   */

  map->filename = filename;
  map->infile = mapin;
  if (lreadmap(map) != 0) {
    perror("Couldn't read map.\n\n");
    exit(0);
  }

  /*
   * Prepare the map data for fft:
   */

  float *fft_data;
  fft_data = (float *)calloc(map->map_length*2+1,sizeof(float));
  fft_index = 1;
  map_index = 0;
  for (s=0;s<map->ns;s++) {
    for (r=0;r<map->nr;r++) {
      for (c=0;c<map->nc;c++) {
	fft_data[fft_index]=(float)map->data[map_index];
	fft_index += 2;
	map_index++;
      }
    }
  }

  nn[1] = map->ns;
  nn[2] = map->nr;
  nn[3] = map->nc;

  lfft(fft_data,nn,3,1);

  float ignore_tag=-1.;

  /*
   * Read the .hkl reflections file one line at a time, and replace fft amplitudes with reflections. 
   * Assume amplitudes instead of intensities in hkl file. Also assume only positive values of hkl.
   */

  float *Iobs,*sigmaIobs,*Fobs,*sigmaFobs,*Fcalc,*Icalc;
  Iobs = (float *)malloc((map->map_length)*sizeof(float));
  sigmaIobs = (float *)malloc((map->map_length)*sizeof(float));
  Fobs = (float *)malloc((map->map_length)*sizeof(float));
  sigmaFobs = (float *)malloc((map->map_length)*sizeof(float));
  Fcalc = (float *)malloc((map->map_length)*sizeof(float));
  Icalc = (float *)malloc((map->map_length)*sizeof(float));
  for (h=0;h<map->map_length;h++) {
    Iobs[h] = ignore_tag;
    Fobs[h] = ignore_tag;
    Icalc[h] = ignore_tag;
    Fcalc[h] = ignore_tag;
  }
    printf("Iobs[0]=%f\n",Iobs[0]);
    printf("Scaling the map\n");
  float xx_R=0.0, xy_R=0.0;
  float xx_goof=0.0,xy_goof=0.0;
  float xx_wR2_shelx=0.0,xy_wR2_shelx=0.0;
  float xx_wR2_ccp4=0.0,xy_wR2_ccp4=0.0;
  float w_ccp4,w_shelx;
  while (fscanf(hklin,"%d %d %d %f %f",&h,&k,&l,&I,&sigI)!=EOF) {
    //    printf("%d %d %d\n",k+lat->origin.k,j+lat->origin.j,i+lat->origin.i);
    if (map->mapc == 3 && map->mapr == 1 && map->maps == 2) {
      r = h; s = k; c = l;
    } else if (map->mapc == 3 && map->mapr == 2 && map->maps == 1) {
            r = k; s = h; c = l;
      //      r = k; s = l; c = h;
      //      r = l; s = k; c = h;
      //      r = l; s = h; c = k;
      //      r = h; s = l; c = k;
      //       r = h; s = k; c = l;
    } else {
      printf("\nUnrecognized CCP4 map x,y,z definitions (MAPC,MAPR,MAPS) = (%ld,%ld,%ld)\n\n",map->mapc,map->mapr,map->maps);
      exit(4);
    }
    if (s>-map->ns/2 && s <= map->ns/2 && r>-map->nr/2 && r <= map->nr/2 && c>-map->nc/2 && c <= map->nc/2) {
      // negative
      if (s < 0) {
	ss = map->ns + s;
      } else ss=s;
      if (r < 0) {
	rr = map->nr + r;
      } else rr = r;
      if (c < 0) {
	cc = map->nc + c;
      } else cc = c;
      map_index = ss*map->section_length+rr*map->nc+cc;
      if (map_index > map->map_length) {
	printf("map_index greater than map->map_length\n");
	exit(0);
      }
      fft_index = map_index*2+1;
      freal=fft_data[fft_index]; fimag=fft_data[fft_index+1];
      Icalc[map_index]=freal*freal+fimag*fimag;
      if (Icalc[map_index]>0) {
	Fcalc[map_index]=sqrtf(Icalc[map_index]);
      }
      Iobs[map_index] = I;
      sigmaIobs[map_index] = sigI;
      if (I>0) {
	Fobs[map_index] = sqrtf(I);
	sigmaFobs[map_index] = sigI/2./Fobs[map_index];
      }
      // accumulate statistics for calculation of overall scale factor
      if (Fobs[map_index] != ignore_tag) {
        float xx = Fcalc[map_index]*Fcalc[map_index];
	float xy = Fobs[map_index]*Fcalc[map_index];
	xx_R += xx;
	xy_R += xy;
	w_ccp4 = 1./sigmaFobs[map_index]/sigmaFobs[map_index];
	xx_wR2_ccp4 += xx*w_ccp4; xy_wR2_ccp4 += xy*w_ccp4;
	w_shelx = 1./sigI/sigI;
	xx_wR2_shelx += Icalc[map_index]*Icalc[map_index]*w_shelx; 
	xy_wR2_shelx += Iobs[map_index]*Icalc[map_index]*w_shelx;
      }
      //      float p=(2.*Imodel+I)/3.;
      //w_shelx = 1./(sigI*sigI+(0.1*p)*(0.1*p));      
      //      xx_goof += I*I*w_shelx; xy_goof += I*Imodel*w_shelx;
      //      friedel_index = (s*map->section_length+r*map->nc+c)*2+1;
      //      scaled_fft_data[friedel_index] = F/Fmodel*fft_data[friedel_index];
      //      scaled_fft_data[friedel_index+1] = F/Fmodel*fft_data[friedel_index+1];
    }
  }

  // Calculate scale factors

  float scale_R = xx_R/xy_R; // overall scale factor
  float scale_wR2_ccp4 = xx_wR2_ccp4/xy_wR2_ccp4; // overall scale factor
  float scale_wR2_shelx = xx_wR2_shelx/xy_wR2_shelx; // overall scale factor
  //  float scale_goof = xy_goof/xx_goof; // overall scale factor

  // calculate figures of merit

  //  printf(" Scale factors: %f %f %f\n",scale_R,scale_wR2_ccp4,scale_wR2_shelx);

  float R_num=0,R_denom=0,wR2_ccp4_num=0,wR2_ccp4_denom=0,wR2_shelx_num=0,wR2_shelx_denom=0;
  long ndat = 0;
  //  printf("Iobs[0],sigmaIobs[0]=%f,%f",Iobs[0],sigmaIobs[0]);
  for (h=0;h<map->map_length;h++) {
    if (Iobs[h] != ignore_tag) {
      ndat++;
      if (sigmaIobs[h]<=0) {
	printf("sigmaIobs <= 0 for index %d\n",h);
	exit(0);
      }
      w_shelx = 1./sigmaIobs[h]/sigmaIobs[h];
      wR2_shelx_num += w_shelx*powf((Iobs[h]-Icalc[h]/scale_wR2_shelx),2.);
      wR2_shelx_denom += w_shelx*powf(Iobs[h],2.);
      //      float p = (2.*Imodel/scale_wR2_shelx+I)/3.;
	    //      w_shelx = 1./(sigsqI+(0.1*p)*(0.1*p));
      if (Iobs[h] > 0) {
	w_ccp4 = 1./sigmaFobs[h]/sigmaFobs[h];
	R_num += fabs(Fobs[h]-Fcalc[h]/scale_R);
	R_denom += Fobs[h];
	wR2_ccp4_num += w_ccp4*powf((Fobs[h]-Fcalc[h]/scale_wR2_ccp4),2.);
	wR2_ccp4_denom += w_ccp4*powf(Fobs[h],2.);
      }
    }
  }
  float R = R_num/R_denom;
  float wR2_ccp4 = sqrtf(wR2_ccp4_num/wR2_ccp4_denom);
  float wR2_shelx = sqrtf(wR2_shelx_num/wR2_shelx_denom);
  float goof = sqrtf(wR2_shelx_num/((float)ndat-1.));

  printf("%11.9f %11.9f %11.9f %11.9f\n",R,wR2_ccp4,wR2_shelx,goof);


CloseShop:
  
  /*
   * Free allocated memory:
   */

  free(fft_data);

  //  lfreelt(lat);

  /*
   * Close files:
   */
  
  fclose(mapin);
  fclose(mapout);
}

