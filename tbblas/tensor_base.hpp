/*
 * tensor_base.hpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_TENSOR_BASE_HPP_
#define TBBLAS_TENSOR_BASE_HPP_

#include <cufft.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <cassert>
#include <iostream>

#include "tensor_proxy.hpp"

namespace tbblas {

enum ReuseFlags {
  ReuseFTNone = 0,
  ReuseFT1 = 1,
  ReuseFT2 = 2
};

template<class, unsigned, bool>
class tensor_base;

template<class T>
struct complex_type {
};

template<>
struct complex_type<float> {
  typedef cufftComplex complex_t;

  __host__ __device__
  static inline complex_t mult(const complex_t& c1, const complex_t& c2) {
    return cuCmulf(c1, c2);
  }
};

template<>
struct complex_type<double> {
  typedef cufftDoubleComplex complex_t;

  __host__ __device__
  static inline complex_t mult(const complex_t& c1, const complex_t& c2) {
    return cuCmul(c1, c2);
  }
};

template<class T, bool device = false>
struct vector_type {
  typedef thrust::host_vector<T> vector_t;
};

template<class T>
struct vector_type<T, true> {
  typedef thrust::device_vector<T> vector_t;
};

template<class T, unsigned dim, bool device>
void fft(const tensor_base<T, dim, device>& dt, const size_t (&size)[dim],
    typename vector_type<typename complex_type<T>::complex_t, device>::vector_t& ftdata)
{
  assert(0);
}

template<class T, unsigned dim, bool device>
void ifft(typename vector_type<typename complex_type<T>::complex_t, device>::vector_t& ftdata,
    const size_t (&size)[dim], tensor_base<T, dim, device>& dt)
{
  assert(0);
}

template<class Tensor>
struct tensor_operation {
  virtual ~tensor_operation() { }
};

template<class Tensor>
struct tensor_element_plus : public tensor_operation<Tensor> {
  Tensor tensor1, tensor2;

  tensor_element_plus(const Tensor& tensor1, const Tensor& tensor2)
   : tensor1(tensor1), tensor2(tensor2) { }
};

template<class Tensor>
struct tensor_scalar_plus : public tensor_operation<Tensor> {
  Tensor tensor;
  typename Tensor::value_t scalar;

  tensor_scalar_plus(const Tensor& tensor, const typename Tensor::value_t& scalar)
   : tensor(tensor), scalar(scalar) { }
};

template<class Tensor>
struct tensor_convolution : public tensor_operation<Tensor> {
  Tensor tensor1, tensor2;
  int reuseFlag;

  tensor_convolution(const Tensor& tensor1, const Tensor& tensor2, int reuseFlag = ReuseFTNone)
   : tensor1(tensor1), tensor2(tensor2), reuseFlag(reuseFlag) { }
};

template<class T, unsigned dim, bool device = true>
class tensor_base {
public:

  typedef tensor_base<T, dim, device> tensor_t;
  typedef typename vector_type<T, device>::vector_t data_t;
  typedef T value_t;
  const static unsigned dimCount = dim;

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
  typedef typename complex_type<T>::complex_t complex_t;
  typedef typename vector_type<complex_t, device>::vector_t cdata_t;

protected:
  boost::shared_ptr<data_t> _data;
  boost::shared_ptr<cdata_t> _cdata;
  dim_t _size;
  T _scalar;
  bool _flipped;

protected:
  tensor_base(const size_t& count)
     : _data(new data_t(count)),
       _cdata(new cdata_t()), _scalar(1), _flipped(false)
  { }

  tensor_base(const dim_t& size)
   : _cdata(new cdata_t()), _scalar(1), _flipped(false)
  {
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

public:
  virtual ~tensor_base() { }

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

  void apply(const tensor_operation<tensor_t>& op) {
    const tensor_element_plus<tensor_t>* add = dynamic_cast<const tensor_element_plus<tensor_t>*>(&op);
    if (add)
      apply(*add);

    const tensor_convolution<tensor_t>* convolve = dynamic_cast<const tensor_convolution<tensor_t>*>(&op);
    if (convolve)
      apply(*convolve);

    const tensor_scalar_plus<tensor_t>* add_scalar = dynamic_cast<const tensor_scalar_plus<tensor_t>*>(&op);
    if (add_scalar)
      apply(*add_scalar);
  }

  void apply(const tensor_element_plus<tensor_t>& op) {
    const tensor_t& dt1 = op.tensor1;
    const tensor_t& dt2 = op.tensor2;

    for (unsigned i = 0; i < dim; ++i) {
      assert(dt1.size()[i] == dt2.size()[i]);
      assert(size()[i] == dt1.size()[i]);
    }

    if (dt1._scalar == T(1) && dt2._scalar == T(1)) {
      thrust::transform(dt1.frbegin(), dt1.frend(), dt2.frbegin(),
          begin(), thrust::plus<T>());
    } else {
      thrust::transform(dt1.cbegin(), dt1.cend(), dt2.cbegin(),
          begin(), thrust::plus<T>());
    }
  }

  struct complex_mult {
    __host__ __device__
    complex_t operator()(const complex_t& c1, const complex_t& c2) const {
      return complex_type<T>::mult(c1, c2);
    }
  };

  void resizeFtData(const dim_t& size) const {
    size_t newSize = 1;
    for (int i = 0; i < dim; ++i) {
      newSize *= size[i];
    }

    if (_cdata->size() != newSize) {
//      std::cout << "New size: " << _cdata->size() << " -> " << newSize << std::endl;
//      boost::shared_ptr<cdata_t> cdata(new cdata_t(newSize));
//      _cdata = cdata;
      _cdata->resize(newSize);
    }
  }

  void apply(const tensor_convolution<tensor_t>& op) {
    // calculate the convolution and write the result to the tensor
    // reuse the fft vector
    const tensor_t &dt1 = op.tensor1, &dt2 = op.tensor2;

    for (unsigned i = 0; i < dim; ++i)
      assert(size()[i] == abs((int)dt1.size()[i] - (int)dt2.size()[i]) + 1);

    dim_t ftsize;
    for (int i = 0; i < dim; ++i)
      ftsize[i] = max(dt1.size()[i], dt2.size()[i]);

    cdata_t& ftdata1 = (op.reuseFlag & ReuseFT1) ? *dt1._cdata : dt1.fft(ftsize);
    cdata_t& ftdata2 = (op.reuseFlag & ReuseFT2) ? *dt2._cdata : dt2.fft(ftsize);
    resizeFtData(ftsize);
    thrust::transform(ftdata1.begin(), ftdata1.end(), ftdata2.begin(),
        _cdata->begin(), complex_mult());
    ifft(*_cdata, ftsize);
  }

  void apply(const tensor_scalar_plus<tensor_t>& op) {
    using namespace thrust::placeholders;

    const tensor_t& dt = op.tensor;
    const value_t& scalar = op.scalar;

    if (dt._scalar == T(1))
      thrust::transform(dt.frbegin(), dt.frend(), begin(), _1 + scalar);
    else
      thrust::transform(dt.cbegin(), dt.cend(), begin(), _1 + scalar);
  }

  /*** tensor operations ***/
  T sum() const {
    return thrust::reduce(data().begin(), data().end()) * _scalar;
  }

  // Calculate the fourier transformation of the tensor and return the complex sequence
  cdata_t& fft(const dim_t& ftsize) const {
    resizeFtData(ftsize);
    tbblas::fft(*this, ftsize, *_cdata);
    return *_cdata;
  }

  // Calculate the inverse fourier transform of the given complex sequence and write the result to the tensor
  void ifft(const cdata_t& fourier, const dim_t& ftsize) {
    tbblas::ifft(*_cdata, ftsize, *this);
  }
};

template<class Tensor>
Tensor flip_tensor(const Tensor& t) {
  Tensor tensor(t);
  tensor.flip();
  return tensor;
}

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

/*** Scalar-valued operations ***/

template<class T, unsigned dim, bool device>
T sum(const tensor_base<T, dim, device>& dt) {
  return dt.sum();
}

template<class T, unsigned dim, bool device>
T dot(const tensor_base<T, dim, device>& dt1, const tensor_base<T, dim, device>& dt2) {
  for (unsigned i = 0; i < dim; ++i)
    assert(dt1.size()[i] == dt2.size()[i]);
  return thrust::inner_product(dt1.frbegin(), dt1.frend(), dt2.frbegin(), 0) * dt1.scalar() * dt2.scalar();
}

/*** Operation wrapper generating operations ***/

template<class T, unsigned dim, bool device>
tensor_convolution<tensor_base<T, dim, device> > conv(
    const tensor_base<T, dim, device>& dt1, const tensor_base<T, dim, device>& dt2,
    int reuseFlag = ReuseFTNone)
{
  return tensor_convolution<tensor_base<T, dim, device> >(dt1, dt2, reuseFlag);
}

} // end tbblas

/*** Operation wrapper generating operations ***/

template<class T, unsigned dim, bool device>
tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> > operator+(
    const tbblas::tensor_base<T, dim, device>& dt1, const tbblas::tensor_base<T, dim, device>& dt2)
{
  return tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> >(dt1, dt2);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> > operator-(
    const tbblas::tensor_base<T, dim, device>& dt1, const tbblas::tensor_base<T, dim, device>& dt2)
{
  return tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> >(dt1, T(-1) * dt2);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> > operator+(
    const tbblas::tensor_base<T, dim, device>& dt, const T& scalar)
{
  return tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> >(dt, scalar);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> > operator+(
    const T& scalar, const tbblas::tensor_base<T, dim, device>& dt)
{
  return tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> >(dt, scalar);
}

#endif /* TENSOR_BASE_HPP_ */
