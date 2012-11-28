/*
 * zeros.hpp
 *
 *  Created on: Sep 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_ZEROS_HPP_
#define TBBLAS_ZEROS_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>
#include <tbblas/sequence.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T, unsigned dim>
struct zeros_expression {
  typedef typename tensor<T, dim>::dim_t dim_t;
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = tensor<T, dim>::cuda_enabled;

  typedef thrust::constant_iterator<value_t> const_iterator;

  zeros_expression(const dim_t& size) {
    for (unsigned i = 0; i < dimCount; ++i)
      _size[i] = size[i];
  }

  inline const_iterator begin() const {
    return thrust::constant_iterator<value_t>(0);
  }

  inline const_iterator end() const {
    return thrust::constant_iterator<value_t>(0) + count();
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= _size[i];
    return count;
  }

private:
  dim_t _size;
};

template<class T, unsigned dim>
struct is_expression<zeros_expression<T, dim> > {
  static const bool value = true;
};

template<class T>
zeros_expression<T, 1> zeros(const size_t& x1) {
  typename zeros_expression<T, 1>::dim_t size;
  size[0] = x1;
  return zeros_expression<T,1>(size);
}

template<class T>
zeros_expression<T, 2> zeros(const size_t& x1, const size_t& x2) {
  typename zeros_expression<T, 2>::dim_t size;
  size[0] = x1;
  size[1] = x2;
  return zeros_expression<T,2>(size);
}

template<class T>
zeros_expression<T, 3> zeros(const size_t& x1, const size_t& x2, const size_t& x3) {
  typename zeros_expression<T, 3>::dim_t size;
  size[0] = x1;
  size[1] = x2;
  size[2] = x3;
  return zeros_expression<T,3>(size);
}

template<class T>
zeros_expression<T, 4> zeros(const size_t& x1, const size_t& x2, const size_t& x3, const size_t& x4) {
  typename zeros_expression<T, 4>::dim_t size;
  size[0] = x1;
  size[1] = x2;
  size[2] = x3;
  size[3] = x4;
  return zeros_expression<T,4>(size);
}

template<class T, unsigned dim>
zeros_expression<T, dim> zeros(const sequence<int, dim>& size) {
  return zeros_expression<T, dim>(size);
}

}

#endif /* TBBLAS_ZEROS_HPP_ */
