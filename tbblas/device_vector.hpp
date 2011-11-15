/*
 * device_vector.hpp
 *
 *  Created on: Nov 7, 2011
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICE_VECTOR_HPP_
#define TBBLAS_DEVICE_VECTOR_HPP_

#include <cassert>

#include <boost/shared_ptr.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cublas.h>

namespace tbblas {

template<class T>
class device_vector;

template<class T>
class device_matrix;

template<class T>
struct sum_vector_operation {
  device_vector<T> v1, v2;

  sum_vector_operation<T>(const device_vector<T>& v1, const device_vector<T>& v2) : v1(v1), v2(v2) { }
};

// c = alpha * a + beta * b
template<class T>
struct axpby {
  T alpha, beta;

  axpby(const T& alpha, const T& beta) : alpha(alpha), beta(beta) { }

  __host__ __device__ T operator()(const T& x, const T& y) const {
    return alpha * x + beta * y;
  }
};

template<class T>
class device_vector {
  friend class device_matrix<T>;

private:
  size_t _length, _offset, _increment;
  T _scalar;
  boost::shared_ptr<thrust::device_vector<T> > _data;

public:
  device_vector(size_t length = 0)
   : _length(length), _offset(0), _increment(1), _scalar(1),
     _data(new thrust::device_vector<T>(length))
  {
  }

  device_vector(size_t length, size_t offset, size_t increment, T scalar,
      const boost::shared_ptr<thrust::device_vector<T> >& data)
   : _length(length), _offset(offset), _increment(increment),
     _scalar(scalar), _data(data)
  {
  }

  thrust::device_vector<T>& data() {
    return *_data;
  }

  const thrust::device_vector<T>& data() const {
    return *_data;
  }

  size_t size() const {
    return _length;
  }

  void resize(size_t size) {
    _length = size;
    _offset = 0;
    _increment = 1;
    _scalar = 1;
    _data = boost::shared_ptr<thrust::device_vector<T> >(new thrust::device_vector<T>(size));
  }

  typename thrust::device_vector<T>::reference operator()(size_t i) {
    assert(_scalar == 1);
    return data()[_offset + _increment * i];
  }

  T sum() const {
    // TODO: allow arbitrary increments
    assert(_increment == 1);
    return _scalar * thrust::reduce(data().begin() + _offset, data().begin() + _offset + _length);
  }

  T norm_1() const {
    return cublasSasum(_length, data().data().get() + _offset, _increment) * _scalar;
  }

  device_vector<T>& operator+=(const device_vector<T>& v) {
    assert(_length == v._length);
    // TODO: use templated axpy operation
    cublasSaxpy(_length, v._scalar / _scalar, v.data().data().get() + v._offset, v._increment,
        data().data().get() + _offset, _increment);
    return *this;
  }

  device_vector<T>& operator-=(const device_vector<T>& v) {
    return *this += -v;
  }

  sum_vector_operation<T> operator+(const device_vector<T>& v) const {
    return sum_vector_operation<T>(*this, v);
  }

  sum_vector_operation<T> operator-(const device_vector<T>& v) const {
    return sum_vector_operation<T>(*this, -v);
  }

  device_vector<T> operator-(void) const {
    device_vector<T> v(*this);
    v._scalar = -v._scalar;
    return v;
  }

  device_vector<T> mult(const T& scalar) const {
    device_vector<T> v(*this);
    v._scalar *= scalar;
    return v;
  }

  device_vector<T>& operator=(const sum_vector_operation<T>& op) {
    const device_vector<T>& v1 = op.v1;
    const device_vector<T>& v2 = op.v2;

    assert(_increment == 1);
    assert(v1._increment == 1);
    assert(v2._increment == 1);
    assert(_length == v1._length);
    assert(v1._length == v2._length);

    thrust::transform(v1.data().begin() + v1._offset, v1.data().begin() + v1._offset + _length,
        v2.data().begin() + v2._offset, data().begin() + _offset, axpby<T>(v1._scalar / _scalar, v2._scalar / _scalar));

    return *this;
  }
};

template<class T>
T sum(const device_vector<T>& v) {
  return v.sum();
}

template<class T>
T norm_1(const device_vector<T>& v) {
  return v.norm_1();
}

template<class T>
device_vector<T> operator*(const device_vector<T>& v, const T& scalar) {
  return v.mult(scalar);
}

template<class T>
device_vector<T> operator*(const T& scalar, const device_vector<T>& v) {
  return v.mult(scalar);
}

}


#endif /* TBBLAS_DEVICE_VECTOR_HPP_ */
