#include "shellPartitioning.hpp"
#include <cassert>
#include <omp.h>
#include <cmath>
#include <iostream>

using namespace std;

void ShellPartitioner::flattened2dindx( unsigned int flattened, int &ix, int &iy, int &iz ) const
{
  iz = flattened%N;
  iy = flattened/N;
  ix = flattened/(N*N);
}

double ShellPartitioner::radius( int ix, int iy, int iz ) const
{
  ix -= N/2;
  iy -= N/2;
  iz -= N/2;
  return sqrt( ix*ix + iy*iy + iz*iz );
}

unsigned int ShellPartitioner::getShellIndx( double r ) const
{
  double closest = 1E30;
  unsigned int closestID = 255;
  for ( unsigned int i=0;i<Nshells;i++ )
  {
    if ( shellRadii[i] > r ) return i;
  }
  //assert( closestID < Nshells );
  return Nshells-1;
}

void ShellPartitioner::updateShellMeans()
{
  assert( shellMeans.size() == Nshells );
  assert( shellMeansSq.size() == Nshells );
  assert( counter.size() == Nshells );

  #pragma omp parallel for
  for ( unsigned int i=0;i<N*N*N;i++ )
  {
    int ix, iy, iz;
    flattened2dindx( i, ix, iy, iz );
    double r = radius( ix, iy, iz );
    unsigned int id = getShellIndx(r);
    shellMeans[id] += data[i];
    shellMeansSq[id] += data[i]*data[i];
    counter[id] += 1;
    clusters[i] = id;
  }
}

double ShellPartitioner::getCombinedStandardDeviation()
{
  // Initialize the means
  shellMeans.resize(Nshells);
  shellMeansSq.resize(Nshells);
  counter.resize(Nshells);
  std::fill( shellMeans.begin(), shellMeans.end(), 0.0 );
  std::fill( shellMeansSq.begin(), shellMeansSq.end(), 0.0 );
  std::fill( counter.begin(), counter.end(), 0 );
  updateShellMeans();

  double avgStdDev = 0.0;
  for ( unsigned int i=1;i<Nshells;i++ )
  {
    if ( counter[i] == 0 ) continue;

    avgStdDev += sqrt( shellMeansSq[i]/counter[i] - pow( shellMeans[i]/counter[i], 2) )/counter[i];
  }
  return avgStdDev;
}
