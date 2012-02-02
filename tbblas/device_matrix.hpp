/*
 * device_matrix.hpp
 *
 *  Created on: Nov 7, 2011
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICE_MATRIX_HPP_
#define TBBLAS_DEVICE_MATRIX_HPP_

#include "tbblas.hpp"

#include <cassert>

#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

#include <cublas.h>

#include "device_vector.hpp"

namespace tbblas {

template<class T>
class device_vector;

template<class T>
class device_matrix;

template<class T>
struct copy_matrix_operation {
  device_matrix<T> m;

  copy_matrix_operation(const device_matrix<T>& m) : m(m) { }
};

template<class T>
struct add_matrix_operation {
  device_matrix<T> m1, m2;

  add_matrix_operation(const device_matrix<T>& m1, const device_matrix<T>& m2) : m1(m1), m2(m2) { }
};

template<class T>
struct scalar_add_matrix_operation {
  device_matrix<T> m;
  T scalar;

  scalar_add_matrix_operation(const device_matrix<T>& m, const T& scalar) : m(m), scalar(scalar) { }
};

template<class T>
struct element_prod_matrix_operation {
  device_matrix<T> m1, m2;

  element_prod_matrix_operation(const device_matrix<T>& m1, const device_matrix<T>& m2) : m1(m1), m2(m2) { }
};

template<class T>
struct prod_matrix_operation {
  device_matrix<T> m1, m2;

  prod_matrix_operation(const device_matrix<T>& m1, const device_matrix<T>& m2) : m1(m1), m2(m2) { }
};

template<class T>
struct sum_matrix_operation {
  device_matrix<T> m;

  sum_matrix_operation(const device_matrix<T>& m) : m(m) { }
};

template<class T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T* A, int lda,
    const T* B, int ldb, T beta, T *C, int ldc)
{
  // TODO: static assert. Only specialized versions are allowed
}

template<>
void gemm(char transa, char transb, int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float *C, int ldc);

template<>
void gemm(char transa, char transb, int m, int n, int k, double alpha, const double* A, int lda,
    const double* B, int ldb, double beta, double *C, int ldc);

/**
 * \brief Calculates B = alpha * A + B
 */
template<class T>
void tbblas_geaxpy(int m, int n, T alpha, const T* A, int lda,
    T* B, int ldb)
{ }

template<>
void tbblas_geaxpy(int m, int n, float alpha, const float* A, int lda,
    float* B, int ldb);

// TODO: Introduce new matrix proxy for repmat operations
//       Can also be used with vectors to create a matrix
//       by repeatedly copying a column or row vector

template<class T>
class device_matrix {
  friend class device_vector<T>;

public:
  typedef typename thrust::device_vector<T>::iterator Iterator;
  typedef typename thrust::device_vector<T>::const_iterator const_Iterator;

  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  // TODO: Handle transpose case correctly
  struct subrange_functor : public thrust::unary_function<difference_type,difference_type> {
    difference_type width, pitch;

    subrange_functor(difference_type width, difference_type pitch)
        : width(width), pitch(pitch) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const
    { 
        return i + (i / width) * (pitch - width);
    }
  };

  typedef typename thrust::counting_iterator<difference_type>                     CountingIterator;
  typedef typename thrust::transform_iterator<subrange_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>       PermutationIterator;
  typedef typename thrust::permutation_iterator<const_Iterator, TransformIterator> const_PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;
  typedef const_PermutationIterator const_iterator;

private:
  T _scalar;
  boost::shared_ptr<thrust::device_vector<T> > _data;
  size_t _rowCount, _columnCount, _offset, _leadingDimension;
  bool _transpose;



public:
  device_matrix(size_t rowCount = 0, size_t columnCount = 0)
    : _rowCount(rowCount), _columnCount(columnCount), _offset(0),
      _leadingDimension(rowCount), _transpose(false), _scalar(1),
      _data(new thrust::device_vector<T>(rowCount * columnCount))
  {
  }

  /*** Basic matrix methods ***/

  thrust::device_vector<T>& data() {
    return *_data;
  }

  const thrust::device_vector<T>& data() const {
    return *_data;
  }

  size_t size1() const {
    return _rowCount;
  }

  size_t size2() const {
    return _columnCount;
  }

  size_t rowCount() const {
    return _rowCount;
  }

  size_t columnCount() const {
    return _columnCount;
  }

  iterator begin(void) {
    assert(!_transpose);
      return PermutationIterator(data().begin() + _offset,
        TransformIterator(CountingIterator(0), subrange_functor(_rowCount, _leadingDimension)));
  }

  iterator end(void) {
      return begin() + _rowCount * _columnCount;
  }

  const_iterator begin(void) const {
    assert(!_transpose);
      return const_PermutationIterator(data().begin() + _offset,
        TransformIterator(CountingIterator(0), subrange_functor(_rowCount, _leadingDimension)));
  }

  const_iterator end(void) const {
      return begin() + _rowCount * _columnCount;
  }

  /**
   * \brief Creates new storage container
   */
  void resize(size_t rowCount, size_t columnCount) {
    _rowCount = rowCount;
    _columnCount = columnCount;
    _offset = 0;
    _leadingDimension = rowCount;
    _transpose = false;
    _scalar = 1;
    _data = boost::shared_ptr<thrust::device_vector<T> >(new thrust::device_vector<T>(rowCount * columnCount));
  }

  /*** Matrix operations applied by the assignment operator ***/

  device_matrix<T>& operator=(const copy_matrix_operation<T>& op) {
    device_matrix<T>& m = op.m;
    assert(size1() == m.size1());
    assert(size2() == m.size2());

    // TODO: implement
    return *this;
  }

  device_matrix<T>& prod(const device_matrix<T>& m1, const device_matrix<T>& m2, bool increment) {
    assert(m1.size2() == m2.size1());
    assert(size1() == m1.size1());
    assert(size2() == m2.size2());

    // TODO: handle transposes correctly, need to reverse the order of multiplications
    //       and transpose the factor matrices
    assert(_transpose == false);

    const char trans1 = m1._transpose ? 't' : 'n';
    const char trans2 = m2._transpose ? 't' : 'n';

    gemm<T>(trans1, trans2, m1.size1(), m2.size2(), m1.size2(), m1._scalar * m2._scalar / _scalar,
        m1.data().data().get() + m1._offset, m1._leadingDimension,
        m2.data().data().get() + m2._offset, m2._leadingDimension,
        (increment ? 1.f : 0.f), data().data().get() + _offset, _leadingDimension);

    return *this;
  }

  device_matrix<T>& operator=(const prod_matrix_operation<T>& op) {
    return prod(op.m1, op.m2, false);
  }

  device_matrix<T>& operator+=(const prod_matrix_operation<T>& op) {
    return prod(op.m1, op.m2, true);
  }

  device_matrix<T>& operator=(const add_matrix_operation<T>& op) {
    const device_matrix<T>& m1 = op.m1;
    const device_matrix<T>& m2 = op.m2;

    assert(m1.size1() == m2.size1());
    assert(m1.size2() == m2.size2());
    assert(size1() == m1.size1());
    assert(size2() == m1.size2());

    // TODO: Loosen the contraint to m1._transpose == m2._transpose
    assert(_transpose == false);
    assert(m1._transpose == false);
    assert(m2._transpose == false);

    thrust::transform(m1.begin(), m1.end(), m2.begin(), begin(), axpby<T>(m1._scalar / _scalar, m2._scalar / _scalar));

    return *this;
  }

  device_matrix<T>& operator+=(const device_matrix<T>& dm) {
    assert(size1() == dm.size1());
    assert(size2() == dm.size2());
    assert(!_transpose);
    assert(!dm._transpose);

    //if (size1() == _leadingDimension && dm.size1() == dm._leadingDimension) {
    //  thrust::transform(data().begin() + _offset, data().begin() + _offset + (size1() * size2()),
    //      dm.data().begin() + dm._offset, data().begin() + _offset, axpby<T>(1, dm._scalar / _scalar));
    //} else {
      thrust::transform(begin(), end(), dm.begin(), begin(), axpby<T>(1, dm._scalar / _scalar));
    //  tbblas_geaxpy<T>(size1(), size2(), dm._scalar / _scalar, dm.data().data().get() + dm._offset, dm._leadingDimension,
    //      data().data().get() + _offset, _leadingDimension);
    //}
    return *this;
  }

  device_matrix<T>& operator-=(const device_matrix<T>& dm) {
    return *this += -dm;
  }

  // TODO: multiply out the scalar values
  device_matrix<T>& element_prod(const device_matrix<T>& m1, const device_matrix<T>& m2) {
    assert(size1() == m1.size1());
    assert(m1.size1() == m2.size1());
    assert(size2() == m1.size2());
    assert(m1.size2() == m2.size2());
    assert(_transpose == m1._transpose);
    assert(m1._transpose == m2._transpose);

    thrust::transform(m1.begin(), m1.end(), m2.begin(), begin(), thrust::multiplies<T>());
    _scalar = m1._scalar * m2._scalar / _scalar;
    return *this;
  }

  device_matrix<T>& operator=(const element_prod_matrix_operation<T>& op) {
    return element_prod(op.m1, op.m2);
  }

  device_matrix<T>& operator*=(const device_matrix<T>& m) {
    return element_prod(*this, m);
  }

  device_matrix<T>& scalar_add(const device_matrix<T>& m, const T& scalar) {
    assert(_scalar == 1);
    assert(m._scalar == 1);
    thrust::transform(m.begin(), m.end(), thrust::make_constant_iterator<T>(scalar), begin(), thrust::plus<T>());
    return *this;
  }

  device_matrix<T>& operator=(const scalar_add_matrix_operation<T>& op) {
    return scalar_add(op.m, op.scalar);
  }

  device_matrix<T>& operator+=(const T& scalar) {
    return scalar_add(*this, scalar);
  }

  device_matrix<T>& operator-=(const T& scalar) {
    return scalar_add(*this, -scalar);
  }

  /*** Direct calcuations ***/
  T norm_1() const {
    assert(_transpose || _leadingDimension == _rowCount);
    assert(!_transpose || _leadingDimension == _columnCount);
    return asum<T>(_rowCount * _columnCount, data().data().get() + _offset, 1) * _scalar;
  }

  T norm_2() const {
    assert(_transpose || _leadingDimension == _rowCount);
    assert(!_transpose || _leadingDimension == _columnCount);
    return nrm2<T>(_rowCount * _columnCount, data().data().get() + _offset, 1) * _scalar;
  }

  /*** Operations that create a simple proxy ***/

  device_matrix<T> operator-(void) const {
    device_matrix<T> dm(*this);
    dm._scalar = -dm._scalar;
    return dm;
  }

  device_matrix<T> mult(const T& scalar) const {
    device_matrix<T> dm(*this);
    dm._scalar = scalar * _scalar;

    return dm;
  }

  device_matrix<T> transpose() const {
    device_matrix<T> dm(*this);
    dm._rowCount = _columnCount;
    dm._columnCount = _rowCount;
    dm._transpose = !_transpose;

    return dm;
  }

  device_matrix<T> subrange(int start1, int stop1, int start2, int stop2) const {
    device_matrix<T> dm(*this);
    dm._rowCount = stop1 - start1;
    dm._columnCount = stop2 - start2;
    if (!_transpose) {
      dm._offset = _offset + _leadingDimension * start2 + start1;
    } else {
      dm._offset = _offset + _leadingDimension * start1 + start2;
    }

    return dm;
  }

  device_vector<T> row(int row) const {
    if (_transpose) {
      return device_vector<T>(size2(), _offset + row * _leadingDimension, 1, _scalar, _data);
    } else {
      return device_vector<T>(size2(), _offset + row, _leadingDimension, _scalar, _data);
    }
  }

  device_vector<T> column(int column) const {
    if (!_transpose) {
      return device_vector<T>(size1(), _offset + column * _leadingDimension, 1, _scalar, _data);
    } else {
      return device_vector<T>(size1(), _offset + column, _leadingDimension, _scalar, _data);
    }
  }

  /*** uBLAS interfaces ***/

  device_matrix<T>& operator=(const boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major>& m) {
    assert(m.size1() == size1());
    assert(m.size2() == size2());
    thrust::copy(m.data().begin(), m.data().end(), data().begin());
    return *this;
  }

  device_matrix<T>& operator=(const boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major>& m) {
    assert(m.size1() == size1());
    assert(m.size2() == size2());
    boost::numeric::ublas::matrix<T> mt = boost::numeric::ublas::trans(m);
    thrust::copy(mt.data().begin(), mt.data().end(), data().begin());
    return *this;
  }

  boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major> ublasRowMajor() {
    boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major> m(size2(), size1());
    thrust::copy(begin(), end(), m.data().begin());
    if (!_transpose)
      return _scalar * boost::numeric::ublas::trans(m);
    else
      return _scalar * m;
  }

  boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> ublasColumnMajor() {
    assert(_offset == 0);

    boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major> m(size1(), size2());
    thrust::copy(data().begin(), data().end(), m.data().begin());
    if (_transpose)
      return _scalar * boost::numeric::ublas::trans(m);
    else
      return _scalar * m;
  }

  operator boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major>() {
    return ublasRowMajor();
  }

  operator boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major>() {
    return ublasColumnMajor();
  }
};

template<class T>
T norm_1(const device_matrix<T>& dm) {
  return dm.norm_1();
}

template<class T>
T norm_2(const device_matrix<T>& m) {
  return m.norm_2();
}

template<class T>
device_matrix<T> trans(const device_matrix<T>& dm) {
  return dm.transpose();
}

template<class T>
device_matrix<T> subrange(const device_matrix<T>& m, int start1, int stop1, int start2, int stop2) {
  return m.subrange(start1, stop1, start2, stop2);
}

template<class T>
device_vector<T> row(const device_matrix<T>& m, int row) {
  return m.row(row);
}

template<class T>
device_vector<T> column(const device_matrix<T>& m, int column) {
  return m.column(column);
}

template<class T>
device_matrix<T> operator*(const device_matrix<T>& m, const T& scalar) {
  return m.mult(scalar);
}

template<class T>
device_matrix<T> operator*(const T& scalar, const device_matrix<T>& m) {
  return m.mult(scalar);
}

template<class T>
device_matrix<T> operator/(const device_matrix<T>& m, const T& scalar) {
  return m.mult(1/scalar);
}

template<class T>
copy_matrix_operation<T> copy(const device_matrix<T>& m) {
  return copy_matrix_operation<T>(m);
}

template<class T>
prod_matrix_operation<T> prod(const device_matrix<T>& m1, const device_matrix<T> m2) {
  return prod_matrix_operation<T>(m1, m2);
}

template<class T>
sum_matrix_operation<T> sum(const device_matrix<T>& m) {
  return sum_matrix_operation<T>(m);
}

template<class T>
add_matrix_operation<T> operator+(const device_matrix<T>& m1, const device_matrix<T>& m2) {
  return add_matrix_operation<T>(m1, m2);
}

template<class T>
add_matrix_operation<T> operator-(const device_matrix<T>& m1, const device_matrix<T>& m2) {
  return add_matrix_operation<T>(m1, -m2);
}

template<class T>
element_prod_matrix_operation<T> operator*(const device_matrix<T>& m1, const device_matrix<T>& m2) {
  return element_prod_matrix_operation<T>(m1, m2);
}

template<class T>
scalar_add_matrix_operation<T> operator+(const device_matrix<T>& m, const T& scalar) {
  return scalar_add_matrix_operation<T>(m, scalar);
}

template<class T>
scalar_add_matrix_operation<T> operator-(const device_matrix<T>& m, const T& scalar) {
  return scalar_add_matrix_operation<T>(m, -scalar);
}

template<class T>
scalar_add_matrix_operation<T> operator+(const T& scalar, const device_matrix<T>& m) {
  return scalar_add_matrix_operation<T>(m, scalar);
}

}


#endif /* TBBLAS_DEVICE_MATRIX_HPP_ */
