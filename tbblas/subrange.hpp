/*
 * subrange.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SUBRANGE_HPP_
#define TBBLAS_SUBRANGE_HPP_

#include <tbblas/tensor_base.hpp>
#include <tbblas/tensor_proxy.hpp>

namespace tbblas {

template<class T, unsigned dim, bool device>
tensor_proxy<typename tensor_base<T, dim, device>::const_iterator, dim>
    subrange(const tensor_base<T, dim, device>& dt, const size_t (&start)[dim], const size_t (&size)[dim])
{
  for (unsigned i = 0; i < dim; ++i) {
    assert(start[i] + size[i] <= dt.size()[i]);
  }

  size_t pitch[dim];
  size_t first = start[0];
  pitch[0] = 1;
  for (int k = 1; k < dim; ++k) {
    pitch[k] = pitch[k-1] * dt.size()[k-1];
    first += pitch[k] * start[k];
  }
  return tensor_proxy<typename tensor_base<T, dim, device>::const_iterator, dim>(
          dt.cbegin() + first, // first
          size,                // size
          pitch                // pitch
      );
}

template<class T, unsigned dim, bool device>
tensor_proxy<typename tensor_base<T, dim, device>::iterator, dim>
    subrange(tensor_base<T, dim, device>& dt, const size_t (&start)[dim], const size_t (&size)[dim])
{
  for (unsigned i = 0; i < dim; ++i) {
    assert(start[i] + size[i] <= dt.size()[i]);
  }

  size_t pitch[dim];
  size_t first = start[0];
  pitch[0] = 1;
  for (int k = 1; k < dim; ++k) {
    pitch[k] = pitch[k-1] * dt.size()[k-1];
    first += pitch[k] * start[k];
  }
  return tensor_proxy<typename tensor_base<T, dim, device>::iterator, dim>(
          dt.begin() + first,  // first
          size,                // size
          pitch                // pitch
      );
}

}

#endif /* SUBRANGE_HPP_ */
