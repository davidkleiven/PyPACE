#ifndef SHELL_KMEANS_H
#define SHELL_KMEANS_H
#include <vector>
#include <memory>

class ShellKmeans
{
public:
  ShellKmeans(){};
  enum class converged_t{ CONVERGED, NOTCONVERGED };

  // Pointers to Python array. These are assumed to be flattened version of higher dimensional arrays (C-order)
  double *data{nullptr};
  unsigned char *clusters{nullptr};
  double *weights{nullptr};
  double *means{nullptr};
  double *shellradii{nullptr};

  // The size of one dimension
  unsigned int N{1};
  unsigned int nClusters{0};
  unsigned int maxIter{1000};

  /** Computes the mean of the data passed and place it in bias[0] */
  void computeDataMean();

  /** Computes the mean radius */
  void computeMeanRadius();

  /** Converts the flattened index to unflattened indices assuming C-ordering */
  void flattened2indx( unsigned int flattened, int &ix, int &iy, int &iz ) const;

  /** Computes the distance from the center */
  double getRadius( int ix, int iy, int iz ) const;

  /** Returns the cluster ID using cosine distance */
  unsigned char getClusterID( double value[2] ) const;

  /** Computes the weighted dot product between x and y */
  double weightDot( double x[2], double y[2] ) const;

  /** Run one iteration of the algorithm */
  converged_t step();
  converged_t run();
private:
  double bias[2];
};
#endif
