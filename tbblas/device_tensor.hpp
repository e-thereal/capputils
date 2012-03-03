/*
 * device_tensor.hpp
 *
 *  Created on: Feb 27, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICETENSOR_HPP_
#define TBBLAS_DEVICETENSOR_HPP_

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

#include <cufft.h>
#include <complex>

#include "tensor_base.hpp"
#include "tensor_proxy.hpp"

namespace tbblas {

template<>
void fft<float, 3, true>(const tensor_base<float, 3>& dt, const size_t (&size)[3],
    thrust::device_vector<complex_type<float>::complex_t>& ftdata);

template<>
void ifft<float, 3, true>(thrust::device_vector<complex_type<float>::complex_t>& ftdata,
    const size_t (&size)[3], tensor_base<float, 3>& dt);

template<class T, unsigned dim>
class device_tensor : public tensor_base<T, dim, true>
{
public:
  typedef device_tensor<T, dim> tensor_t;
  typedef tensor_base<T, dim, true> base_t;
  typedef typename base_t::dim_t dim_t;

public:
  device_tensor(const dim_t& size)
   : base_t(size)
  { }

public:
  tensor_t& operator=(const tensor_operation<base_t>& op) {
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
};

template<class T>
class device_tensor<T, 3> : public tensor_base<T, 3, true>
{
public:
  typedef device_tensor<T, 3> tensor_t;
  typedef tensor_base<T, 3, true> base_t;
  typedef typename base_t::dim_t dim_t;

public:
  device_tensor(const dim_t& size)
   : base_t(size)
  { }

  device_tensor(size_t width, size_t height, size_t depth)
   : base_t(width * height * depth)
  {
    this->_size[0] = width;
    this->_size[1] = height;
    this->_size[2] = depth;
  }

public:
  tensor_t& operator=(const tensor_operation<base_t>& op) {
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
};

/*** Proxy generating operations ***/

template<class T, unsigned dim>
device_tensor<T, dim> flip(const device_tensor<T, dim>& dt) {
  return flip_tensor(dt);
}

/*** Operation wrapper generating operations ***/

template<class T, unsigned dim>
tensor_convolution<tensor_base<T, dim, true> > conv(
    const tensor_base<T, dim, true>& dt1, const tensor_base<T, dim, true>& dt2)
{
  return tensor_convolution<tensor_base<T, dim, true> >(dt1, dt2);
}

} // end tbblas

/*** Proxy generating operations ***/

template<class T, unsigned dim>
tbblas::device_tensor<T, dim> operator*(const tbblas::device_tensor<T, dim>& dt, const T& scalar) {
  tbblas::device_tensor<T, dim> tensor(dt);
  tensor.mult(scalar);
  return tensor;
}

template<class T, unsigned dim>
tbblas::device_tensor<T, dim> operator*(const T& scalar, const tbblas::device_tensor<T, dim>& dt) {
  tbblas::device_tensor<T, dim> tensor(dt);
  tensor.mult(scalar);
  return tensor;
}

template<class T, unsigned dim>
tbblas::device_tensor<T, dim> operator/(const tbblas::device_tensor<T, dim>& dt, const T& scalar) {
  tbblas::device_tensor<T, dim> tensor(dt);
  tensor.mult(T(1) / scalar);
  return tensor;
}

#endif /* TBBLAS_DEVICETENSOR_HPP_ */
