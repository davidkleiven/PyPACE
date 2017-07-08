#include "hermiteOperatorMatrix.hpp"
//#include <gsl/gsl_sf_hermite.h>
#include <cassert>
#include <cmath>
#include <complex>

using namespace std;

HermiteOperatorMatrix::HermiteOperatorMatrix( double *weights, double *points, unsigned int intorder, unsigned int nbasis, uint8_t *support,
uint8_t *mask, int Nsup, int Nmask, double scaleX, double scaleY, double scaleZ ):weights(weights), points(points),
integorder(intorder), nbasis(nbasis), support(support), mask(mask), Nsup(Nsup), Nmask(Nmask), scaleX(scaleX), scaleY(scaleY),
scaleZ(scaleZ)
{
  discReal.min = -Nsup/2.0;
  discReal.max = Nsup/2.0;
  discReal.N = Nsup;
  double pi = acos(-1.0);
  discFourier.min = -pi/2.0;
  discFourier.max = pi/2.0;
  discFourier.N = Nmask;
  supInterp = unique_ptr<TrilinearInterpolator>( new TrilinearInterpolator(support,discReal,discReal,discReal) );
  maskInterp = unique_ptr<TrilinearInterpolator>( new TrilinearInterpolator(mask,discFourier,discFourier,discFourier) );
}

void HermiteOperatorMatrix::evalHermitte()
{
  assert( weights != nullptr );
  assert( points != nullptr );

  for ( unsigned int i=0;i<integorder;i++ )
  {
    hermiteEval.push_back( vector<double>(nbasis) );
    evalAllHermitteAtPosition( nbasis, points[i], &hermiteEval[i][0] );
    //gsl_sf_hermite_phys_array( integorder, points[i], &hermiteEval[i][0] );
  }
}

void HermiteOperatorMatrix::flattened2xyz( int flattened, int &nx, int &ny, int &nz ) const
{
  nz = flattened%nbasis;
  ny = ( flattened/nbasis )%nbasis;
  nx = flattened/(nbasis*nbasis);
}

double HermiteOperatorMatrix::basis( int ix, int iy, int iz, int nx, int ny, int nz ) const
{
  return hermiteEval[ix][nx]*hermiteEval[iy][ny]*hermiteEval[iz][nz];
}

double HermiteOperatorMatrix::integrateOutsideSupport( int nx1, int ny1, int nz1, int nx2, int ny2, int nz2 ) const
{
  double integral = 0.0;
  for ( int ix=0;ix<integorder;ix++ )
  for ( int iy=0;iy<integorder;iy++ )
  for ( int iz=0;iz<integorder;iz++ )
  {
    integral += ( 1.0-(*supInterp)(points[ix]*scaleX,points[iy]*scaleY,points[iz]*scaleZ) )*weights[ix]*weights[iy]*weights[iz]*\
                basis(ix,iy,iz,nx1,ny1,nz1)*basis(ix,iy,iz,nx2,ny2,nz2);
  }
  return integral;
}

double HermiteOperatorMatrix::integrateInsideMask( int nx1, int ny1, int nz1, int nx2, int ny2, int nz2 ) const
{
  double integral = 0.0;
  for ( int ix=0;ix<integorder;ix++ )
  for ( int iy=0;iy<integorder;iy++ )
  for ( int iz=0;iz<integorder;iz++ )
  {
    integral += (*maskInterp)(points[ix]/scaleX,points[iy]/scaleY,points[iz]/scaleZ)*weights[ix]*weights[iy]*weights[iz]*\
                basis(ix,iy,iz,nx1,ny1,nz1)*basis(ix,iy,iz,nx2,ny2,nz2);
  }
  return integral;
}

double HermiteOperatorMatrix::norm( int nx, int ny, int nz ) const
{
  double integral = 0.0;
  for ( int ix=0;ix<integorder;ix++ )
  for ( int iy=0;iy<integorder;iy++ )
  for ( int iz=0;iz<integorder;iz++ )
  {
    integral += weights[ix]*weights[iy]*weights[iz]*pow( basis(ix,iy,iz,nx,ny,nz),2 );
  }
  return sqrt( integral );
}

void HermiteOperatorMatrix::matrixElement( int n, int m, double &realpart, double &imagpart )
{
  int nx1, ny1, nz1, nx2, ny2, nz2;
  flattened2xyz( n, nx1, ny1, nz1 );
  flattened2xyz( m, nx2, ny2, nz2 );
  evalHermitte();
  double norm1 = norm( nx1, ny1, nz1 );
  double norm2 = norm( nx2, ny2, nz2 );
  double realsp = integrateOutsideSupport( nx1, ny1, nz1, nx2, ny2, nz2 );
  double fourier = integrateInsideMask( nx1, ny1, nz1, nx2, ny2, nz2 );

  complex<double> realconstib( realsp/(norm1*norm2), 0.0 );
  complex<double> fouriercontrib( fourier/(norm1*norm2), 0.0 );
  complex<double> im(0.0,1.0);
  int N = nx2+ny2+nz2-nx1-ny1-nz1;
  complex<double> total = realconstib + pow(im,N)*fouriercontrib;

  realpart = 0.5*total.real();
  imagpart = 0.5*total.imag();
}

void HermiteOperatorMatrix::evalAllHermitteAtPosition( int nmax, double x, double result[] ) const
{
  result[0] = 1.0;
  result[1] = 2.0*x;
  for ( int i=1;i<nmax-1;i++ )
  {
    result[i+1] = 2.0*x*result[i] - 2.0*i*result[i-1];
  }
}
