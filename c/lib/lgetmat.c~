/* LGETMAT.C - Extract the crystal orientation from a DENZO output
               file.

   Author: Mike Wall
   Date: 9/27/95, 10/4/2012
   Version: 1.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mwmask.h>

#define XPOS 45
#define XLEN 10

int lgetmat(DIFFIMAGE *imdiff)
{
  char *inl;
  long i;
  int xpos;
  float u[9];

  /*
   * Allocate memory:
   */
  
  inl = (char *)malloc(sizeof(char)*(LINESIZE+1));
  
  /*
   * Skip the first four lines in the file:
   */
  
  for (i=0;i<4;i++) fgets(inl, LINESIZE, imdiff->infile);

  /*
  for(i=0;i<=2;i++) {
    fgets(inl, LINESIZE, imdiff->infile);
    
    
    xpos=XPOS;
    inl[xpos+XLEN]=0;
    u[i] = atof(inl+xpos);
    xpos=xpos+XLEN+1;
    inl[xpos+XLEN]=0;
    u[3+i] = atof(inl+xpos);
    xpos=xpos+XLEN+1;
    inl[xpos+XLEN]=0;
    u[6+i] = atof(inl+xpos);
  }
  imdiff->u.xx = u[2];
  imdiff->u.xy = u[1];
  imdiff->u.xz = u[0];
  imdiff->u.yx = u[5];
  imdiff->u.yy = u[4];
  imdiff->u.yz = u[3];
  imdiff->u.zx = u[8];
  imdiff->u.zy = u[7];
  imdiff->u.zz = u[6];
  */
  
  float osc_start, osc_end, d1, d2, rotx, roty, rotz;

  printf("Reading the orientation info line\n");

  fscanf(imdiff->infile,"%f %f %f %f %f %f %f",&osc_start,&osc_end,&d1,&d2,&rotz,&roty,&rotx);

  printf("%f %f %f %f\n",osc_start,rotx,roty,rotz);
  
  roty = -roty;

  osc_start *= PI/180.;
  rotx *= PI/180.;
  roty *= PI/180.;
  rotz *= PI/180.;


  struct xyzmatrix F,S,U;

  U = lrotmat(rotx+osc_start,roty,rotz);
  F = lrotmat(0,0,PI/2);
  //  S = lrotmat(osc_start,0,0);
  //  U = lmatmul(U,S);
  U = lmatmul(U,F);

  /*
  imdiff->u.xx = U.zx;
  imdiff->u.xy = U.yx;
  imdiff->u.xz = U.xx;
  imdiff->u.yx = U.zy;
  imdiff->u.yy = U.yy;
  imdiff->u.yz = U.xy;
  imdiff->u.zx = U.zz;
  imdiff->u.zy = U.yz;
  imdiff->u.zz = U.xz;
  */

  imdiff->u.xx = U.xx;
  imdiff->u.xy = U.xy;
  imdiff->u.xz = U.xz;
  imdiff->u.yx = U.yx;
  imdiff->u.yy = U.yy;
  imdiff->u.yz = U.yz;
  imdiff->u.zx = U.zx;
  imdiff->u.zy = U.zy;
  imdiff->u.zz = U.zz;

/*for(i=0;i<=8;i++){printf ("%f\n",u[i]);}/***/

  /*
   * Free memory:
   */
  
  free((char *)inl);
  return(0);
}




