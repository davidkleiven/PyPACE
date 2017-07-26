#ifndef SHELL_PARTITIONING
#define SHELL_PARTITIONING
#include <vector>

class ShellPartitioner
{
public:
  ShellPartitioner(){};
  double *data{nullptr};
  unsigned char *clusters{nullptr};
  unsigned int N{0};
  double *shellRadii{nullptr};
  unsigned int Nshells{0};
  std::vector<double> shellMeans;
  std::vector<double> shellMeansSq;
  std::vector<unsigned int> counter;

  /** Convertes flattened index to x,y,z assuming C-ordered array */
  void flattened2dindx( unsigned int flattened, int &ix, int &iy, int &iz ) const;

  /** Updates the means and the means square of the shells */
  void updateShellMeans();

  /** Returns the average standard deviation of all the clusters */
  double getCombinedStandardDeviation();

  /** Returns the index of the shell */
  unsigned int getShellIndx( double r ) const;

  /** Returns the distance from the center of the object to the point */
  double radius( int ix, int iy, int iz ) const;

};
#endif
