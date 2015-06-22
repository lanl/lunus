#include<mwmask.h>

struct xyzmatrix lrotmat(float rotx, float roty, float rotz)
{ 
  //Define Rx, Ry, and Rz:

  struct xyzmatrix R,Rx,Ry,Rz;

  Rx.xx = 1.; Rx.xy = 0.; Rx.xz = 0.;
  Rx.yx = 0.; Rx.yy = cos(rotx); Rx.yz = sin(rotx);
  Rx.zx = 0.; Rx.zy = -sin(rotx); Rx.zz = cos(rotx);

  Ry.xx = cos(roty); Ry.xy = 0.; Ry.xz = -sin(roty);
  Ry.yx = 0.; Ry.yy = 1.; Ry.yz = 0.;
  Ry.zx = sin(roty); Ry.zy = 0.; Ry.zz = cos(roty);

  Rz.xx = cos(rotz); Rz.xy = sin(rotz); Rz.xz = 0.;
  Rz.yx = -sin(rotz); Rz.yy = cos(rotz); Rz.yz = 0.;
  Rz.zx = 0.; Rz.zy = 0.; Rz.zz = 1.;

  R = lmatmul(Ry,Rx);
  R = lmatmul(Rz,R);

  return R;

}

struct xyzmatrix lmatmul(struct xyzmatrix a, struct xyzmatrix b)
{
  struct xyzmatrix c;

  c.xx = a.xx*b.xx + a.xy*b.yx + a.xz*b.zx;
  c.xy = a.xx*b.xy + a.xy*b.yy + a.xz*b.zy;
  c.xz = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;

  c.yx = a.yx*b.xx + a.yy*b.yx + a.yz*b.zx;
  c.yy = a.yx*b.xy + a.yy*b.yy + a.yz*b.zy;
  c.yz = a.yx*b.xz + a.yy*b.yz + a.yz*b.zz;

  c.zx = a.zx*b.xx + a.zy*b.yx + a.zz*b.zx;
  c.zy = a.zx*b.xy + a.zy*b.yy + a.zz*b.zy;
  c.zz = a.zx*b.xz + a.zy*b.yz + a.zz*b.zz;

  return c;
}
