/*
 * tensor_base.hpp
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_TENSOR_BASE_HPP_
#define TBBLAS_TENSOR_BASE_HPP_

#include <thrust/functional.h>

#include <tbblas/forward_reverse_iterator.hpp>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <boost/shared_ptr.hpp>

#include <tbblas/type_traits.hpp>

namespace tbblas {

template<class Tensor>
struct tensor_copy {
  Tensor tensor;

  tensor_copy(const Tensor& tensor) : tensor(tensor) { }
};

template<class Tensor, class Operation>
void apply_operation(Tensor& tensor, const Operation& operation) {
  assert(0);
}

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
  typedef typename complex_type<T>::type complex_t;
  typedef typename vector_type<complex_t, device>::vector_t cdata_t;

protected:
  boost::shared_ptr<data_t> _data;
  dim_t _size;
  T _scalar;
  bool _flipped;

public:
  tensor_base(size_t x1 = 1, size_t x2 = 1, size_t x3 = 1, size_t x4 = 1)
   : _scalar(1), _flipped(false)
  {
    const size_t size[] = {x1, x2, x3, x4};
    size_t count = 1;

    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }

    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

  tensor_base(const dim_t& size)
   : _scalar(1), _flipped(false)
  {
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count));
  }

  template<class T2, bool device2>
  tensor_base(const tensor_copy<tensor_base<T2, dim, device2> >& copy_op)
   : _scalar(1), _flipped(false)
  {
    const tensor_base<T2, dim, device2>& tensor = copy_op.tensor;
    const dim_t& size = tensor.size();

    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _data = boost::shared_ptr<data_t>(new data_t(count));
    thrust::copy(tensor.cbegin(), tensor.cend(), data().begin());
  }

public:
  virtual ~tensor_base() { }

  const dim_t& size() const {
    return _size;
  }

  size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      count *= _size[i];
    }
    return count;
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

  bool unit_scale() const {
    return _scalar == T(1);
  }

  bool flipped() const {
    return _flipped;
  }

  /*** apply operations ***/

  template<class Operation>
  tensor_t& operator=(const Operation& op) {
    apply_operation(*this, op);
    return *this;
  }
};

template<class Tensor>
void apply_operation(Tensor& tensor, const tensor_copy<Tensor>& copy_op)
{
  const Tensor& t = copy_op.tensor;
  const typename Tensor::dim_t& size = t.size();

  for (unsigned i = 0; i < Tensor::dimCount; ++i) {
    assert(tensor.size()[i] == size[i]);    // todo: throw exception instead
  }
  thrust::copy(t.cbegin(), t.cend(), tensor.begin());
}

template<class T, unsigned dim, bool device>
tensor_copy<tensor_base<T, dim, device> > copy(const tensor_base<T, dim, device>& tensor) {
  return tensor_copy<tensor_base<T, dim, device> >(tensor);
}

template<class T, unsigned dim, bool device>
tensor_base<T, dim, device> flip(const tensor_base<T, dim, device>& t) {
  tensor_base<T, dim, device> tensor(t);
  tensor.flip();
  return tensor;
}

}

template<class T, unsigned dim, bool device>
tbblas::tensor_base<T, dim, device> operator*(const tbblas::tensor_base<T, dim, device>& dt, const T& scalar) {
  tbblas::tensor_base<T, dim, device> tensor(dt);
  tensor.mult(scalar);
  return tensor;
}

template<class T, unsigned dim, bool device>
tbblas::tensor_base<T, dim, device> operator*(const T& scalar, const tbblas::tensor_base<T, dim, device>& dt) {
  tbblas::tensor_base<T, dim, device> tensor(dt);
  tensor.mult(scalar);
  return tensor;
}

template<class T, unsigned dim, bool device>
tbblas::tensor_base<T, dim, device> operator/(const tbblas::tensor_base<T, dim, device>& dt, const T& scalar) {
  tbblas::tensor_base<T, dim, device> tensor(dt);
  tensor.mult(T(1) / scalar);
  return tensor;
}

#endif /* TENSOR_BASE_HPP_ */
