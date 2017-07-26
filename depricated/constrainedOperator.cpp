#include "constrainedOperator.hpp"

ConstrainedOperator::ConstrainedOperator(uint8_t *mask, uint8_t *support, int Nsup, int Nmask, int nbasis): mask(mask), \
support(support), Nsup(Nsup), Nmask(Nmask), nbasis(nbasis){};

void ConstrainedOperator::flattened2xyz( int flattened, int &nx, int &ny, int &nz ) const
{
  nz = flattened%nbasis;
  ny = ( flattened/nbasis )%nbasis;
  nx = flattened/(nbasis*nbasis);
}
