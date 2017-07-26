#ifndef CONSTRAINED_OPERATOR_H
#define CONSTRAINED_OPERATOR_H
#include <cstdint>

class ConstrainedOperator
{
public:
  ConstrainedOperator( uint8_t *mask, uint8_t *support, int Nsup, int Nmask, int nbasis );

  /** Convertes between the flattened index of the matrix to xyz indices */
  void flattened2xyz( int flattened, int &ix, int &iy, int &iz ) const ;
protected:
  uint8_t *support{nullptr};
  uint8_t *mask{nullptr};
  int Nsup{0};
  int Nmask{0};
  unsigned int nbasis{1};
};
#endif
