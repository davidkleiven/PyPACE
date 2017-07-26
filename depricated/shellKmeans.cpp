#include "shellKmeans.hpp"
#include <omp.h>
#include <cmath>
#include <cassert>
#include <iostream>

using namespace std;

void ShellKmeans::computeDataMean()
{
  double mean = 0.0;

  #pragma omp parallel for
  for ( unsigned int i=0;i<N*N*N;i++ )
  {
    mean += data[i];
  }

  bias[0] = mean/(N*N*N);
}

void ShellKmeans::flattened2indx( unsigned int flattened, int &ix, int &iy, int &iz ) const
{
  iz = flattened%N;
  iy = flattened/N;
  ix = flattened/(N*N);
}

double ShellKmeans::getRadius( int ix, int iy, int iz ) const
{
  ix -= N/2;
  iy -= N/2;
  iz -= N/2;
  return sqrt( ix*ix + iy*iy + iz*iz );
}

void ShellKmeans::computeMeanRadius()
{
  double radius = 0.0;

  #pragma omp parallel for
  for ( unsigned int i=0;i<N*N*N;i++ )
  {
    int ix, iy, iz;
    flattened2indx( i, ix, iy, iz );
    radius += getRadius(ix,iy,iz);
  }
  bias[1] = radius/(N*N*N);
}

/*
unsigned char ShellKmeans::getClusterID( double value[2] ) const
{
  double closest = 1E30;
  unsigned char closestID = 255;
  value[0] -= bias[0];
  value[1] -= bias[1];
  for ( unsigned int i=0;i<nClusters;i++ )
  {
    double cluster[2] = {means[i]-bias[0], shellradii[i]-bias[1]};
    double dot = weightDot(value,cluster);
    double norm1 = sqrt( weightDot(value,value) );
    double norm2 = sqrt( weightDot(cluster,cluster) );
    double distance = dot/(norm1*norm2);

    // Not two similar vectors the dot product is 1
    if ( abs(distance-1.0) < abs(closest-1.0) )
    {
      closest = distance;
      closestID = i;
    }
  }
  return closestID;
}*/

unsigned char ShellKmeans::getClusterID( double value[2] ) const
{
  double closest = 1E30;
  unsigned char closestID = 255;
  double weight = weights[1];
  for ( unsigned int i=0;i<nClusters;i++ )
  {
    double meanDiff = value[0]-means[i];
    double rDiff = value[1]-shellradii[i];
    double distance = sqrt( meanDiff*meanDiff + weight*rDiff*rDiff );
    if ( distance < closest )
    {
      closest = distance;
      closestID = i;
    }
  }
  return closestID;
}

double ShellKmeans::weightDot( double x[2], double y[2] ) const
{
  double dot = 0.0;
  for ( unsigned int i=0;i<2;i++ )
  {
    dot += x[i]*weights[i]*y[i];
  }
  return dot;
}

ShellKmeans::converged_t ShellKmeans::step()
{
  ShellKmeans::converged_t flag = converged_t::CONVERGED;
  vector<double> newMeans(nClusters, 0.0);
  vector<double> newRadii(nClusters, 0.0);
  vector<unsigned int> counter(nClusters, 0);

  #pragma omp parallel for
  for ( unsigned int i=0;i<N*N*N;i++ )
  {
    int ix,iy,iz;
    flattened2indx(i,ix,iy,iz);
    double rad = getRadius(ix,iy,iz);
    double value[2] = {data[i],rad};
    unsigned char id = getClusterID( value );

    if ( clusters[i] != id )
    {
      flag = converged_t::NOTCONVERGED;
    }
    clusters[i] = id;
    newMeans[id] += data[i];
    newRadii[id] += rad;
    counter[id] += 1;
  }

  // Update the arrays
  for ( unsigned int i=0;i<nClusters;i++ )
  {
    means[i] = newMeans[i]/counter[i];
    shellradii[i] = shellradii[i]/counter[i];
  }
  return flag;
}

ShellKmeans::converged_t ShellKmeans::run()
{
  assert( data       != nullptr );
  assert( clusters   != nullptr );
  assert( weights    != nullptr );
  assert( means      != nullptr );
  assert( shellradii != nullptr );

  computeDataMean();
  computeMeanRadius();
  for ( unsigned int i=0;i<maxIter;i++ )
  {
    switch( step() )
    {
      case converged_t::CONVERGED:
        cout << "Shell kmeans converged in " << i << " iterations...\n";
        return converged_t::CONVERGED;
      case converged_t::NOTCONVERGED:
        break;
    }
  }
  return converged_t::NOTCONVERGED;
}
