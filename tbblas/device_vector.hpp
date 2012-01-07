/*
 * device_vector.hpp
 *
 *  Created on: Nov 7, 2011
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICE_VECTOR_HPP_
#define TBBLAS_DEVICE_VECTOR_HPP_

#include "tbblas.hpp"

#include <cassert>

#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform.h>

#include <cublas.h>

namespace tbblas {

template<class T>
void axpy(int n, const T alpha, const T* x, int incx, T* y, int incy) {
  // TODO static assert
}

template<>
void axpy(int n, const float alpha, const float* x, int incx, float* y, int incy);

template<>
void axpy(int n, const double alpha, const double* x, int incx, double* y, int incy);

template<class T>
T asum(int n, const T* x, int incx) { }

template<>
float asum(int n, const float* x, int incx);

template<>
double asum(int n, const double* x, int incx);

template<class T>
T nrm2(int n, const T* x, int incx) { }

template<>
float nrm2(int n, const float* x, int incx);

template<>
double nrm2(int n, const double* x, int incx);

template<class T>
void swap(int n, T* x, int incx, T* y, int incy) { }

template<>
void swap(int n, float* x, int incx, float* y, int incy);

template<>
void swap(int n, double* x, int incx, double* y, int incy);

template<class T>
class device_vector;

template<class T>
class device_matrix;

template<class T>
struct sum_matrix_operation;

template<class T>
struct add_vector_operation {
  device_vector<T> v1, v2;

  add_vector_operation<T>(const device_vector<T>& v1, const device_vector<T>& v2) : v1(v1), v2(v2) { }
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

public:
  // convert a linear index to a row index
  template <typename T1>
  struct linear_index_to_row_index : public thrust::unary_function<T1,T1>
  {
      T1 C; // number of columns

      __host__ __device__
      linear_index_to_row_index(T1 _C) : C(_C) {}

      __host__ __device__
      T1 operator()(T1 i)
      {
          return i / C;
      }
  };

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
    return asum<T>(_length, data().data().get() + _offset, _increment) * _scalar;
  }

  void swap(device_vector<T>& v) {
    assert(size() == v.size());
    tbblas::swap<T>(size(), data().data().get() + _offset, _increment, v.data().data().get() + v._offset, v._increment);
  }

  device_vector<T>& operator+=(const device_vector<T>& v) {
    assert(_length == v._length);
    axpy<T>(_length, v._scalar / _scalar, v.data().data().get() + v._offset, v._increment,
        data().data().get() + _offset, _increment);
    return *this;
  }

  device_vector<T>& operator-=(const device_vector<T>& v) {
    return *this += -v;
  }

  add_vector_operation<T> operator+(const device_vector<T>& v) const {
    return add_vector_operation<T>(*this, v);
  }

  add_vector_operation<T> operator-(const device_vector<T>& v) const {
    return add_vector_operation<T>(*this, -v);
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

  device_vector<T>& operator=(const add_vector_operation<T>& op) {
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

  device_vector<T>& operator=(const sum_matrix_operation<T>& op) {
    const device_matrix<T>& m = op.m;

    // TODO: Loosen constraints
    assert(m._transpose == false);
    assert(m._scalar == _scalar);
    assert(_increment == 1);
    assert(m.size2() == size());

    if (m._leadingDimension == m.size1()) {
      thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(m.size1())),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(m.size1())) + (m.size1() * m.size2()),
            m.data().begin() + m._offset,
            thrust::make_discard_iterator(),
            data().begin() + _offset);
    } else {
      thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(m.size1())),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(m.size1())) + (m.size1() * m.size2()),
            m.begin(),
            thrust::make_discard_iterator(),
            data().begin() + _offset);
    }

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
