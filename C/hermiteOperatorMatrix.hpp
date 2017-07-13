#ifndef HERMITE_OPERATOR_MATRIX_H
#define HERMITE_OPERATOR_MATRIX_H
#include <vector>
#include <memory>
#include "trilinearInterpolator.hpp"
#include "constrainedOperator.hpp"

class HermiteOperatorMatrix: public ConstrainedOperator
{
public:

  /**
  * Constructor:
  * weights - weights in the Gauss-Hermite integration scheme
  * points - evaluation points in the Gauss-Hermite integration scheme
  * inorder - integration order of the Gauss-Hermite integration scheme
  * nbasis - number of basis function in each direction
  * support - flattened version of a 3D array representing the support (C-ordering is assumed )
  * mask - flattened version of a 3D array representing the mask in the Fourier Domain (C-ordering is assumed)
  * Nsup - dimension of the support, the 3D version of the support array is assumed to have the shape (Nsup,Nsup,Nsup)
  * Nmask - dimension of the mask, the 3D version of the mask array is assumed to have the shape (Nmask,Nmask,Nmask)
  * sizeX(Y,Z) - the scaling parameter in each of the direcion, the dimensionless coordinate is given by x/scaleX
  *              in real space and k_x*scaleX in the Fourier domain
  */
  HermiteOperatorMatrix( double *weights, double *points, unsigned int intorder, unsigned int nbasis, uint8_t *support,
  uint8_t *mask, int Nsup, int Nmask, double sizeX, double sizeY, double sizeZ, double voxelsize );

  /** Evaluates the Hermite polynomials on all required points */
  void evalHermitte();

  /** Integrates the basis function in realspace over the points outside the support */
  double integrateOutsideSupport( int nx1, int ny1, int nz1, int nx2, int ny2, int nz2 ) const;

  /** Integrates the basis function in Fourier space over the points that are included in the mask */
  double integrateInsideMask( int nx1, int ny1, int nz1, int nx2, int ny2, int nz2 ) const;

  /** Evaluates the norm */
  double norm( int nx, int ny, int nz ) const;

  /** Evaluates the basis function at one of the predefined points */
  double basis( int ix, int iy, int iz, int nx, int ny, int nz ) const;

  /** Compute one matrix element */
  void matrixElement( int n, int m, double &realPart, double &imagPart );
private:
  std::vector< std::vector<double> > hermiteEval;
  double *matrixReal{nullptr};
  double *matrixImag{nullptr};
  double *weights{nullptr};
  double *points{nullptr};
  unsigned int integorder{0};

  // Scaling parameters in the different directions
  double scaleX{1.0};
  double scaleY{1.0};
  double scaleZ{1.0};
  double voxelsize{1.0};
  std::unique_ptr<TrilinearInterpolator> supInterp;
  std::unique_ptr<TrilinearInterpolator> maskInterp;
  Discretization discReal;
  Discretization discFourier;

  /** Evaluates the Hermitte polynomials at position x */
  void evalAllHermitteAtPosition( int nmax, double x, double result[] ) const;
};
#endif
