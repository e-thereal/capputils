/*
 * dot.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DOT_HPP_
#define TBBLAS_DOT_HPP_

#include <tbblas/tensor_base.hpp>
#include <thrust/inner_product.h>

namespace tbblas {

template<class T, unsigned dim, bool device>
T dot(const tensor_base<T, dim, device>& dt1, const tensor_base<T, dim, device>& dt2) {
  for (unsigned i = 0; i < dim; ++i)
    assert(dt1.size()[i] == dt2.size()[i]);
  return thrust::inner_product(dt1.frbegin(), dt1.frend(), dt2.frbegin(), T(0)) * dt1.scalar() * dt2.scalar();
}

}

#endif /* TBBLAS_DOT_HPP_ */
