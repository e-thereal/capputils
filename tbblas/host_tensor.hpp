/*
 * host_tensor.hpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_HOST_TENSOR_HPP_
#define TBBLAS_HOST_TENSOR_HPP_

#include "tbblas.hpp"

#include <cassert>

#include <thrust/iterator/reverse_iterator.h>
#include "forward_reverse_iterator.hpp"

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include "tensor_proxy.hpp"
#include "device_tensor.hpp"

namespace tbblas {

template<class T, unsigned dim>
class host_tensor_base;

template<class T, unsigned dim>
class host_tensor;

template<class T, unsigned dim>
struct tensor_element_plus : public tensor_operation<T, dim> {
  typedef device_tensor_base<T, dim> tensor_t;

  tensor_t tensor1, tensor2;

  tensor_element_plus(const tensor_t& tensor1, const tensor_t& tensor2)
   : tensor1(tensor1), tensor2(tensor2) { }
};

template<class T, unsigned dim>
struct tensor_convolution : public tensor_operation<T, dim> {
  typedef device_tensor_base<T, dim> tensor_t;

  tensor_t tensor1, tensor2;

  tensor_convolution(const tensor_t& tensor1, const tensor_t& tensor2)
   : tensor1(tensor1), tensor2(tensor2) { }
};

template<class T, unsigned dim>
void fft(const device_tensor_base<T, dim>& dt, const size_t (&size)[dim],
    thrust::device_vector<typename complex_type_trait<T>::complex_t>& ftdata)
{ }

template<class T, unsigned dim>
void ifft(thrust::device_vector<typename complex_type_trait<T>::complex_t>& ftdata,
    const size_t (&size)[dim], device_tensor_base<T, dim>& dt)
{ }

template<>
void fft<float, 3>(const device_tensor_base<float, 3>& dt, const size_t (&size)[3],
    thrust::device_vector<complex_type_trait<float>::complex_t>& ftdata);

template<>
void ifft<float, 3>(thrust::device_vector<complex_type_trait<float>::complex_t>& ftdata,
    const size_t (&size)[3], device_tensor_base<float, 3>& dt);

/*** TENSOR BASE ***/

template<class T, unsigned dim>
class device_tensor_base {
public:

  typedef device_tensor_base<T, dim> tensor_t;
  typedef thrust::device_vector<T> data_t;

  class apply_scaling : public thrust::unary_function<T, T> {
  private:
    T scalar;
  public:
    apply_scaling(const T& scalar) : scalar(scalar) { }

    __host__ __device__
    T operator()(const T& value) const {
      return value * scalar;
    }
  };

public:
  typedef typename data_t::iterator data_iterator;
  typedef typename data_t::const_iterator const_data_iterator;
  typedef typename tbblas::forward_reverse_iterator<const_data_iterator> const_reverse_iterator;

  typedef typename tbblas::forward_reverse_iterator<data_iterator> iterator;
  typedef typename thrust::transform_iterator<apply_scaling, const_reverse_iterator> const_iterator;
  typedef size_t dim_t[dim];
  typedef typename complex_type_trait<T>::complex_t complex_t;
  typedef thrust::device_vector<complex_t> cdata_t;

protected:
  boost::shared_ptr<data_t> _data;
  mutable boost::shared_ptr<cdata_t> _cdata;
  dim_t _size;
  mutable dim_t _ftsize;
  T _scalar;
  bool _flipped;

protected:
  device_tensor_base(const size_t& count)
     : _data(new data_t(count)),
       _cdata(new cdata_t()), _scalar(1), _flipped(false)
  { }

  device_tensor_base(const dim_t& size)
   : _cdata(new cdata_t()), _scalar(1), _flipped(false)
  {
    size_t count = 1;
    for (int i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

public:
  virtual ~device_tensor_base() { }

  const dim_t& size() const {
    return _size;
  }

  data_t& data() {
    return *_data;
  }

  const data_t& data() const {
    return *_data;
  }

  T scalar() const {
    return _scalar;
  }

  iterator frbegin() {
    if (_flipped)
      return make_forward_reverse_iterator(_data->end(), true);
    else
      return make_forward_reverse_iterator(_data->begin(), false);
  }

  iterator frend() {
    if (_flipped)
      return make_forward_reverse_iterator(_data->begin(), true);
    else
      return make_forward_reverse_iterator(_data->end(), false);
  }

  const_reverse_iterator frbegin() const {
    if (_flipped)
      return make_forward_reverse_iterator(_data->end(), true);
    else
      return make_forward_reverse_iterator(_data->begin(), false);
  }

  const_reverse_iterator frend() const {
    if (_flipped)
      return make_forward_reverse_iterator(_data->begin(), true);
    else
      return make_forward_reverse_iterator(_data->end(), false);
  }

  iterator begin() {
    assert(_scalar == (T)1);
    return frbegin();
  }

  iterator end() {
    assert(_scalar == (T)1);
    return frend();
  }

  const_iterator cbegin() const {
    return thrust::make_transform_iterator(frbegin(), apply_scaling(_scalar));
  }

  const_iterator cend() const {
    return thrust::make_transform_iterator(frend(), apply_scaling(_scalar));
  }

  const_iterator begin() const {
    return cbegin();
  }

  const_iterator end() const {
    return cend();
  }

  void mult(const T& scalar) {
    _scalar *= scalar;
  }

  void flip() {
    _flipped = !_flipped;
  }

  /*** apply operations ***/

  void apply(const tensor_operation<T, dim>& op) {
    const tensor_element_plus<T, dim>* add = dynamic_cast<const tensor_element_plus<T, dim>*>(&op);
    if (add)
      apply(*add);

    const tensor_convolution<T, dim>* convolve = dynamic_cast<const tensor_convolution<T, dim>*>(&op);
    if (convolve)
      apply(*convolve);
  }

  void apply(const tensor_element_plus<T, dim>& op) {
    if (op.tensor1._scalar == T(1) && op.tensor2._scalar == T(1)) {
      thrust::transform(op.tensor1.frbegin(), op.tensor1.frend(), op.tensor2.frbegin(),
          begin(), thrust::plus<T>());
    } else {
      thrust::transform(op.tensor1.cbegin(), op.tensor1.cend(), op.tensor2.cbegin(),
          begin(), thrust::plus<T>());
    }
  }

  struct complex_mult {
    __host__ __device__
    complex_t operator()(const complex_t& c1, const complex_t& c2) const {
      return complex_type_trait<T>::mult(c1, c2);
    }
  };

  void resizeFtData(const dim_t& size) const {
    size_t newSize = 1;
    for (int i = 0; i < dim; ++i) {
      newSize *= size[i];
      _ftsize[i] = size[i];
    }

    if (_cdata->size() != newSize) {
      _cdata->resize(newSize);
    }
  }

  void apply(const tensor_convolution<T, dim>& op) {
    // calculate the convolution and write the result to the tensor
    // reuse the fft vector
    const tensor_t &dt1 = op.tensor1, &dt2 = op.tensor2;

    dim_t ftsize;
    for (int i = 0; i < dim; ++i)
      ftsize[i] = max(dt1.size()[i], dt2.size()[i]);

    thrust::device_vector<complex_t>& ftdata = dt1.fft(ftsize);
    resizeFtData(ftsize);
    thrust::transform(ftdata.begin(), ftdata.end(), dt2.fft(ftsize).begin(),
        _cdata->begin(), complex_mult());
    ifft(*_cdata, ftsize);
  }

  /*** tensor operations ***/
  T sum() const {
    return thrust::reduce(data().begin(), data().end()) * _scalar;
  }

  // Calculate the fourier transformation of the tensor and return the complex sequence
  thrust::device_vector<complex_t>& fft(const dim_t& ftsize) const {
    resizeFtData(ftsize);
    tbblas::fft(*this, ftsize, *_cdata);
    return *_cdata;
  }

  // Calculate the inverse fourier transform of the given complex sequence and write the result to the tensor
  void ifft(const thrust::device_vector<complex_t>& fourier, const dim_t& ftsize) {
    tbblas::ifft(*_cdata, ftsize, *this);
  }
};

template<class T, unsigned dim>
class device_tensor : public device_tensor_base<T, dim>
{
public:
  typedef device_tensor<T, dim> tensor_t;
  typedef typename device_tensor_base<T, dim>::dim_t dim_t;

public:
  device_tensor(const dim_t& size)
   : device_tensor_base<T, dim>(size)
  { }

public:
  tensor_t& operator=(const tensor_operation<T, dim>& op) {
    apply(op);
    return *this;
  }

  tensor_t& operator+=(const tensor_t& tensor) {
    apply(tensor_element_plus<T, dim>(*this, tensor));
    return *this;
  }

  tensor_t& operator-=(const tensor_t& tensor) {
    apply(tensor_element_plus<T, dim>(*this, T(-1) * tensor));
    return *this;
  }
};

template<class T>
class device_tensor<T, 3> : public device_tensor_base<T, 3>
{
public:
  typedef device_tensor<T, 3> tensor_t;
  typedef typename device_tensor_base<T, 3>::dim_t dim_t;

public:
  device_tensor(const dim_t& size)
   : device_tensor_base<T, 3>(size)
  { }

  device_tensor(size_t width, size_t height, size_t depth)
   : device_tensor_base<T, 3>(width * height * depth)
  {
    this->_size[0] = width;
    this->_size[1] = height;
    this->_size[2] = depth;
  }

public:
  tensor_t& operator=(const tensor_operation<T, 3>& op) {
    apply(op);
    return *this;
  }

  tensor_t& operator+=(const tensor_t& tensor) {
    apply(tensor_element_plus<T, 3>(*this, tensor));
    return *this;
  }

  tensor_t& operator-=(const tensor_t& tensor) {
    apply(tensor_element_plus<T, 3>(*this, T(-1) * tensor));
    return *this;
  }
};

/*** Proxy generating operations ***/

template<class T, unsigned dim>
device_tensor<T, dim> flip(const device_tensor<T, dim>& dt) {
  device_tensor<T, dim> tensor(dt);
  tensor.flip();
  return tensor;
}

template<class T, unsigned dim>
device_tensor_proxy<typename device_tensor<T, dim>::const_iterator, dim>
    subrange(const device_tensor<T, dim>& dt, const size_t (&start)[dim], const size_t (&size)[dim])
{
  size_t pitch[dim];
  size_t first = start[0];
  pitch[0] = 1;
  for (int k = 1; k < dim; ++k) {
    pitch[k] = pitch[k-1] * dt.size()[k-1];
    first += pitch[k] * start[k];
  }
  return device_tensor_proxy<typename device_tensor<T, dim>::const_iterator, dim>(
          dt.cbegin() + first, // first
          size,                // size
          pitch                // pitch
      );
}

template<class T, unsigned dim>
device_tensor_proxy<typename device_tensor<T, dim>::iterator, dim>
    subrange(device_tensor<T, dim>& dt, const size_t (&start)[dim], const size_t (&size)[dim])
{
  size_t pitch[dim];
  size_t first = start[0];
  pitch[0] = 1;
  for (int k = 1; k < dim; ++k) {
    pitch[k] = pitch[k-1] * dt.size()[k-1];
    first += pitch[k] * start[k];
  }
  return device_tensor_proxy<typename device_tensor<T, dim>::iterator, dim>(
          dt.begin() + first,  // first
          size,                // size
          pitch                // pitch
      );
}

/*** Scalar-valued operations ***/

template<class T, unsigned dim>
T sum(const device_tensor<T, dim>& dt) {
  return dt.sum();
}

template<class T, unsigned dim>
T dot(const device_tensor<T, dim>& dt1, const device_tensor<T, dim>& dt2) {
  return thrust::inner_product(dt1.frbegin(), dt1.frend(), dt2.frbegin(), 0) * dt1.scalar() * dt2.scalar();
}

/*** Operation wrapper generating operations ***/

template<class T, unsigned dim>
tensor_convolution<T, dim> conv(const device_tensor<T, dim>& dt1, const device_tensor<T, dim>& dt2) {
  return tensor_convolution<T, dim>(dt1, dt2);
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

/*** Operation wrapper generating operations ***/

template<class T, unsigned dim>
tbblas::tensor_element_plus<T, dim> operator+(
    const tbblas::device_tensor<T, dim>& dt1, const tbblas::device_tensor<T, dim>& dt2)
{
  return tbblas::tensor_element_plus<T, dim>(dt1, dt2);
}

template<class T, unsigned dim>
tbblas::tensor_element_plus<T, dim> operator-(
    const tbblas::device_tensor<T, dim>& dt1, const tbblas::device_tensor<T, dim>& dt2)
{
  return tbblas::tensor_element_plus<T, dim>(dt1, T(-1) * dt2);
}


#endif /* TBBLAS_HOST_TENSOR_HPP_ */
