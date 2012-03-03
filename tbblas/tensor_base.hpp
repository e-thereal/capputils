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

#include "tensor_proxy.hpp"

namespace tbblas {

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
{ }

template<class T, unsigned dim, bool device>
void ifft(typename vector_type<typename complex_type<T>::complex_t, device>::vector_t& ftdata,
    const size_t (&size)[dim], tensor_base<T, dim, device>& dt)
{ }

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
struct tensor_convolution : public tensor_operation<Tensor> {
  Tensor tensor1, tensor2;

  tensor_convolution(const Tensor& tensor1, const Tensor& tensor2)
   : tensor1(tensor1), tensor2(tensor2) { }
};

template<class T, unsigned dim, bool device = true>
class tensor_base {
public:

  typedef tensor_base<T, dim, device> tensor_t;
  typedef typename vector_type<T, device>::vector_t data_t;

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
  //typedef thrust::device_vector<complex_t> cdata_t;

protected:
  boost::shared_ptr<data_t> _data;
  mutable boost::shared_ptr<cdata_t> _cdata;
  dim_t _size;
  mutable dim_t _ftsize;
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
    for (int i = 0; i < dim; ++i) {
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
  }

  void apply(const tensor_element_plus<tensor_t>& op) {
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
      return complex_type<T>::mult(c1, c2);
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

  void apply(const tensor_convolution<tensor_t>& op) {
    // calculate the convolution and write the result to the tensor
    // reuse the fft vector
    const tensor_t &dt1 = op.tensor1, &dt2 = op.tensor2;

    dim_t ftsize;
    for (int i = 0; i < dim; ++i)
      ftsize[i] = max(dt1.size()[i], dt2.size()[i]);

    cdata_t& ftdata = dt1.fft(ftsize);
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
  return thrust::inner_product(dt1.frbegin(), dt1.frend(), dt2.frbegin(), 0) * dt1.scalar() * dt2.scalar();
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

#endif /* TENSOR_BASE_HPP_ */
