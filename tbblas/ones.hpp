/*
 * ones.hpp
 *
 *  Created on: Sep 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_ONES_HPP_
#define TBBLAS_ONES_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <thrust/iterator/constant_iterator.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T, unsigned dim>
struct ones_expression {
  typedef typename tensor<T, dim>::dim_t dim_t;
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = true;    // can be executed on the device

  typedef thrust::constant_iterator<value_t> const_iterator;

  ones_expression(const dim_t& size, const dim_t& fullsize) : _size(size), _fullsize(fullsize) { }

  inline const_iterator begin() const {
    return thrust::constant_iterator<value_t>(1);
  }

  inline const_iterator end() const {
    return thrust::constant_iterator<value_t>(1) + count();
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= _size[i];
    return count;
  }

private:
  dim_t _size, _fullsize;
};

template<class T, unsigned dim>
struct is_expression<ones_expression<T, dim> > {
  static const bool value = true;
};

template<class T>
ones_expression<T, 1> ones(const size_t& x1) {
  typename ones_expression<T, 1>::dim_t size;
  size[0] = x1;
  return ones_expression<T,1>(size, size);
}

template<class T>
ones_expression<T, 2> ones(const size_t& x1, const size_t& x2) {
  typename ones_expression<T, 2>::dim_t size;
  size[0] = x1;
  size[1] = x2;
  return ones_expression<T,2>(size, size);
}

template<class T>
ones_expression<T, 3> ones(const size_t& x1, const size_t& x2, const size_t& x3) {
  typename ones_expression<T, 3>::dim_t size;
  size[0] = x1;
  size[1] = x2;
  size[2] = x3;
  return ones_expression<T,3>(size, size);
}

template<class T>
ones_expression<T, 4> ones(const size_t& x1, const size_t& x2, const size_t& x3, const size_t& x4) {
  typename ones_expression<T, 4>::dim_t size;
  size[0] = x1;
  size[1] = x2;
  size[2] = x3;
  size[3] = x4;
  return ones_expression<T,4>(size, size);
}

template<class T, unsigned dim>
ones_expression<T, dim> ones(const sequence<int, dim>& size) {
  return ones_expression<T, dim>(size, size);
}

template<class T, unsigned dim>
ones_expression<T, dim> ones(const sequence<int, dim>& size, const sequence<int, dim>& fullsize) {
  return ones_expression<T, dim>(size, fullsize);
}

}

#endif /* TBBLAS_ONES_HPP_ */
