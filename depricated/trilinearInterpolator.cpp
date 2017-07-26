#include "trilinearInterpolator.hpp"


TrilinearInterpolator::TrilinearInterpolator( uint8_t *data, const Discretization &dx, const Discretization &dy, const Discretization &dz ):
data(data), discX(&dx), discY(&dy), discZ(&dz){};

int TrilinearInterpolator::getIndex( double x, const Discretization &disc ) const
{
  return (x-disc.min)*(disc.N-1)/(disc.max-disc.min);
}

int TrilinearInterpolator::xyz2Flattened( int ix, int iy, int iz ) const
{
  return ix*discY->N*discZ->N + iy*discZ->N + iz;
}

double TrilinearInterpolator::get( int ix, int iy, int iz ) const
{
  if (( ix < 0 ) || ( ix >= discX->N ) || ( iy < 0 ) || ( iy >= discY->N ) || (iz<0) || (iz>=discZ->N))
  {
    return 0.0;
  }
  return data[xyz2Flattened(ix,iy,iz)];
}

int TrilinearInterpolator::indexX( double x ) const
{
  return getIndex(x,*discX);
}

int TrilinearInterpolator::indexY( double y ) const
{
  return getIndex(y,*discY);
}

int TrilinearInterpolator::indexZ( double z ) const
{
  return getIndex(z,*discZ);
}

double TrilinearInterpolator::getCrd( unsigned int indx, const Discretization &disc ) const
{
  return indx*(disc.max-disc.min)/(disc.N-1) + disc.min;
}

double TrilinearInterpolator::operator()( double x, double y, double z ) const
{
  int ix = indexX(x);
  int iy = indexY(y);
  int iz = indexZ(z);
  return get(ix,iy,iz);
}
