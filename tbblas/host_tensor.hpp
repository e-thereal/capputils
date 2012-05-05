/*
 * host_tensor.hpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_HOSTTENSOR_HPP_
#define TBBLAS_HOSTTENSOR_HPP_

#include "tbblas.hpp"

#include <cassert>

#include <thrust/iterator/reverse_iterator.h>
#include "forward_reverse_iterator.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <complex>

#include "tensor_base.hpp"
#include "tensor_proxy.hpp"

namespace tbblas {

//template<>
//void fft<float, 3, true>(const tensor_base<float, 3>& dt, const size_t (&size)[3],
//    thrust::device_vector<complex_type<float>::complex_t>& ftdata);
//
//template<>
//void ifft<float, 3, true>(thrust::device_vector<complex_type<float>::complex_t>& ftdata,
//    const size_t (&size)[3], tensor_base<float, 3>& dt);

template<class T, unsigned dim>
class host_tensor : public tensor_base<T, dim, false>
{
public:
  typedef host_tensor<T, dim> tensor_t;
  typedef tensor_base<T, dim, false> base_t;
  typedef typename base_t::dim_t dim_t;

public:
  host_tensor(const dim_t& size) : base_t(size) { }

  template<class Tensor>
  host_tensor(const tensor_copy<Tensor>& copy_op) : base_t(copy_op) { }

public:
  template<class Tensor>
  tensor_t& operator=(const tensor_operation<Tensor>& op) {
    apply(op);
    return *this;
  }

  tensor_t& operator+=(const base_t& tensor) {
    apply(tensor_element_plus<base_t>(*this, tensor));
    return *this;
  }

  tensor_t& operator-=(const base_t& tensor) {
    apply(tensor_element_plus<base_t>(*this, T(-1) * tensor));
    return *this;
  }

  tensor_t& operator+=(const T& scalar) {
    apply(tensor_scalar_plus<base_t>(*this, scalar));
    return *this;
  }
};

template<class T>
class host_tensor<T, 3> : public tensor_base<T, 3, false>
{
public:
  typedef host_tensor<T, 3> tensor_t;
  typedef tensor_base<T, 3, false> base_t;
  typedef typename base_t::dim_t dim_t;

public:
  host_tensor(const dim_t& size)
   : base_t(size)
  { }

  host_tensor(size_t width, size_t height, size_t depth)
   : base_t(width * height * depth)
  {
    this->_size[0] = width;
    this->_size[1] = height;
    this->_size[2] = depth;
  }

  template<class Tensor>
  host_tensor(const tensor_copy<Tensor>& copy_op) : base_t(copy_op) { }

public:
  template<class Tensor>
  tensor_t& operator=(const tensor_operation<Tensor>& op) {
    apply(op);
    return *this;
  }

  tensor_t& operator+=(const base_t& tensor) {
    apply(tensor_element_plus<base_t>(*this, tensor));
    return *this;
  }

  tensor_t& operator-=(const base_t& tensor) {
    apply(tensor_element_plus<base_t>(*this, T(-1) * tensor));
    return *this;
  }

  tensor_t& operator+=(const T& scalar) {
    apply(tensor_scalar_plus<base_t>(*this, scalar));
    return *this;
  }
};

/*** Proxy generating operations ***/

template<class T, unsigned dim>
host_tensor<T, dim> flip(const host_tensor<T, dim>& dt) {
  return flip_tensor(dt);
}

} // end tbblas

/*** Proxy generating operations ***/

template<class T, unsigned dim>
tbblas::host_tensor<T, dim> operator*(const tbblas::host_tensor<T, dim>& dt, const T& scalar) {
  tbblas::host_tensor<T, dim> tensor(dt);
  tensor.mult(scalar);
  return tensor;
}

template<class T, unsigned dim>
tbblas::host_tensor<T, dim> operator*(const T& scalar, const tbblas::host_tensor<T, dim>& dt) {
  tbblas::host_tensor<T, dim> tensor(dt);
  tensor.mult(scalar);
  return tensor;
}

template<class T, unsigned dim>
tbblas::host_tensor<T, dim> operator/(const tbblas::host_tensor<T, dim>& dt, const T& scalar) {
  tbblas::host_tensor<T, dim> tensor(dt);
  tensor.mult(T(1) / scalar);
  return tensor;
}


#endif /* TBBLAS_HOSTTENSOR_HPP_ */
