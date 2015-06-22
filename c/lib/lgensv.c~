/* LGENSV.C - Generate a scattering vector for a pixel in an image,
              based on the pixel position and the crystal orientation
              matrix. 

   Author: Mike Wall
   Date: 9/27/95
   Version: 1.

*/

#include<mwmask.h>

int lgensv(DIFFIMAGE *imdiff)
{
  size_t
    index;
  
  float 
    spindle_rad,
    cs,
    sn,
    y,
    z,
    c;
  
  struct xyzcoords
    q,
    qcass;

  /*
   * Generate initial coordinates based on experimental geometry:
   */

  q.x = imdiff->pixel_size_mm * (float)imdiff->pos.c - imdiff->beam_mm.y;
  q.y = -(imdiff->pixel_size_mm * (float)imdiff->pos.r - imdiff->beam_mm.x);
  q.z = imdiff->distance_mm;

  /*
   * Generate cassette displacements:
   */

  qcass.x =  q.y * imdiff->cassette.z * PI / 180.;
  qcass.y = - q.x * imdiff->cassette.z * PI / 180;
  qcass.z = (-q.y * imdiff->cassette.x - q.x * imdiff->cassette.y) * PI
    / 180.;

  /* 
   * Correct for cassette displacements:
   */

  q.x += qcass.x;
  q.y += qcass.y;
  q.z += qcass.z;

  /*
   * Get pixel value:
   */

  index = imdiff->pos.r*imdiff->hpixels + imdiff->pos.c;
  if ((imdiff->image[index] != imdiff->ignore_tag) &&
      (imdiff->image[index] != imdiff->mask_tag)){
    imdiff->map3D->value = (float) imdiff->image[index];
    if (imdiff->image[index] < 0||imdiff->map3D->value < 0) printf("%d,%f,%f\n",(int)index,(float)imdiff->image[index],imdiff->map3D->value);
  } else {
    imdiff->map3D->value = (float)imdiff->lattice_ignore_tag;
  }
  
  /*
   * Build scattering vector:
   */
  
  c = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
  
  q.x = q.x / c;
  q.y = q.y / c;
  q.z = q.z / c - 1;
  
  /*printf("Unit cell: %g, %g, %g\n",imdiff->cell.a,imdiff->cell.b,imdiff->cell.c);/***/
  
  /*printf("U-matrix: %g, %g, %g\n %g, %g, %g\n %g, %g,%g\n",imdiff->u.xx,
    imdiff->u.xy,
    imdiff->u.xz,
    imdiff->u.yx,
    imdiff->u.yy,
    imdiff->u.yz,
    imdiff->u.zx,
    imdiff->u.zy,
    imdiff->u.zz);/***/
  
  /*printf("Position: %d,%d  %g,%g,%g\n",imdiff->pos.r,imdiff->pos.c,imdiff->map3D->pos.x,imdiff->map3D->pos.y,imdiff->map3D->pos.z);/***/
  
  /*
   * Orient scattering vector using u-matrix, and normalize to wavelength:
   */

  imdiff->map3D->pos.x = (imdiff->u.xx*q.x + imdiff->u.xy*q.y +
			  imdiff->u.xz*q.z)/imdiff->wavelength;
  imdiff->map3D->pos.y = (imdiff->u.yx*q.x + imdiff->u.yy*q.y +
			  imdiff->u.yz*q.z)/imdiff->wavelength;
  imdiff->map3D->pos.z = (imdiff->u.zx*q.x + imdiff->u.zy*q.y +
			  imdiff->u.zz*q.z)/imdiff->wavelength;

  /*
   * Generate (h,k,l) for output:
   */

  imdiff->map3D->pos.x *= imdiff->cell.a;
  imdiff->map3D->pos.y *= imdiff->cell.b;
  imdiff->map3D->pos.z *= imdiff->cell.c;
  

/*  printf("(%d,%d) -> (%g, %g, %g)\n",imdiff->pos.r,imdiff->pos.c,imdiff->map3D->pos.x,imdiff->map3D->pos.y,imdiff->map3D->pos.z);/***/
}









