/* CALCU.C - Create an image with a pattern.
   
   Author: Mike Wall
   Date: 10/3/2012
   Version: 1.
   
   "calcu <rotx> <roty> <rotz> <film> <osc_start>"
   
   Input is three crystal rotations. Output is u matrix.

*/

#include<mwmask.h>

int main(int argc, char *argv[])
{

  struct xyzmatrix Rx, Ry, Rz, F, S, U;

  float rotx=0,roty=0,rotz=0, film=0, osc_start = 0;
/*
 * Read information from input line:
 */
  switch(argc) {
  case 6:
    osc_start= atof(argv[5])*PI/180.;
  case 5:
    film= atof(argv[4])*PI/180.;
  case 4:
    rotz = atof(argv[3])*PI/180.;
  case 3:
    roty = atof(argv[2])*PI/180.;
  case 2:
    rotx = atof(argv[1])*PI/180.;
    break;
  default:
    printf("\n Usage: calcu "
	   "<rotx> <roty> <rotz> \n\n");
    exit(0);
  }

  //Define Rx, Ry, and Rz:

  /*
  Rx.xx = 1.; Rx.xy = 0.; Rx.xz = 0.;
  Rx.yx = 0.; Rx.yy = cos(rotx); Rx.yz = sin(rotx);
  Rx.zx = 0.; Rx.zy = -sin(rotx); Rx.zz = cos(rotx);

  Ry.xx = cos(roty); Ry.xy = 0.; Ry.xz = -sin(roty);
  Ry.yx = 0.; Ry.yy = 1.; Ry.yz = 0.;
  Ry.zx = sin(roty); Ry.zy = 0.; Ry.zz = cos(roty);

  Rz.xx = cos(rotz); Rz.xy = sin(rotz); Rz.xz = 0.;
  Rz.yx = -sin(rotz); Rz.yy = cos(rotz); Rz.yz = 0.;
  Rz.zx = 0.; Rz.zy = 0.; Rz.zz = 1.;

  F.xx = cos(film); F.xy = sin(film); F.xz = 0.;
  F.yx = -sin(film); F.yy = cos(film); F.yz = 0.;
  F.zx = 0.; F.zy = 0.; F.zz = 1.;

  U = lmatmul(Rx,F);
  U = lmatmul(Ry,U);
  U = lmatmul(Rz,U);
  */

  U = lrotmat(rotx,roty,rotz);
  F = lrotmat(0,0,film);
  S = lrotmat(0,osc_start,0);
  U = lmatmul(U,F);
  U = lmatmul(U,S);

  /*
  printf("%f %f %f\n",U.xx,U.xy,U.xz);
  printf("%f %f %f\n",U.yx,U.yy,U.yz);
  printf("%f %f %f\n",U.zx,U.zy,U.zz);
  */
 
  printf("%f %f %f\n",-U.yz,-U.zz,-U.xz);
  printf("%f %f %f\n",-U.yx,-U.zx,-U.xx);
  printf("%f %f %f\n",U.yy,U.zy,U.xy);

}
