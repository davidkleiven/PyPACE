#ifndef TRILINEAR_INTERPOLATOR_H
#define TRILINEAR_INTERPOLATOR_H
#include <cstdint>

struct Discretization
{
  double min{0.0};
  double max{0.0};
  unsigned int N{1};
};

class TrilinearInterpolator
{
public:
  TrilinearInterpolator( uint8_t *data, const Discretization &dx, const Discretization &dy, const Discretization &dz );

  /** Returns the data point at position x,y,z.
  NOTE: At the moment there is no interpolation. It simply returns the value at the closest position */
  double operator()( double x, double y, double z ) const;

  /** Returns the data value at position ix,iy,iz */
  double get( int ix, int iy, int iz ) const;

  /** Convertes x,y,z index to flattened index */
  int xyz2Flattened( int ix, int iy, int iz ) const;
private:
  const Discretization *discX{nullptr};
  const Discretization *discY{nullptr};
  const Discretization *discZ{nullptr};
  uint8_t *data{nullptr};

  int getIndex( double x, const Discretization &disc ) const;
  int indexX( double x ) const;
  int indexY( double y ) const;
  int indexZ( double z ) const;

  double getCrd( unsigned int indx, const Discretization &disc ) const;
};

#endif
