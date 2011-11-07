/*
 * device_matrix.hpp
 *
 *  Created on: Nov 7, 2011
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICE_MATRIX_HPP_
#define TBBLAS_DEVICE_MATRIX_HPP_

#include <cassert>

#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cublas.h>

namespace tbblas {

template<class T>
class device_matrix;

template<class T>
struct copy_matrix_operation {
  device_matrix<T> m;

  copy_matrix_operation(const device_matrix<T>& m) : m(m) { }
};

template<class T>
struct prod_matrix_operation {
  device_matrix<T> m1, m2;

  prod_matrix_operation(const device_matrix<T>& m1, const device_matrix<T>& m2) : m1(m1), m2(m2) { }
};

template<class T>
copy_matrix_operation<T> copy(const device_matrix<T>& m) {
  return copy_matrix_operation<T>(m);
}

template<class T>
prod_matrix_operation<T> prod(const device_matrix<T>& m1, const device_matrix<T> m2) {
  return prod_matrix_operation<T>(m1, m2);
}

template<class T>
void gemm(char transa, char transb, int m, int n, int k, T alpha, const T* A, int lda,
    const T* B, int ldb, T beta, T *C, int ldc)
{
  // TODO: static assert. Only specialized versions are allowed
}

template<>
void gemm(char transa, char transb, int m, int n, int k, float alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float *C, int ldc)
{
  cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}



template<class T>
class device_matrix {
  friend class device_vector<T>;

private:
  size_t _rowCount, _columnCount, _offset, _leadingDimension;
  bool _transpose;
  T _scalar;
  boost::shared_ptr<thrust::device_vector<T> > _data;

public:
  device_matrix(size_t rowCount, size_t columnCount)
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

  device_matrix<T>& operator+=(const device_matrix<T>& dm) {
    assert(size1() == dm.size1());
    assert(size2() == dm.size2());
    assert(!_transpose);
    assert(!dm._transpose);
    assert(_leadingDimension == size1());
    assert(dm._leadingDimension == dm.size1());

    thrust::transform(data().begin() + _offset, data().begin() + _offset + (size1() * size2()),
        dm.data().begin() + dm._offset, data().begin() + _offset, axpby<T>(1, dm._scalar / _scalar));

    return *this;
  }

  /*** Operations that create a simple proxy ***/

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

  boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major> ublas() {
    assert(_offset == 0);

    boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major> m(size2(), size1());
    thrust::copy(data().begin(), data().end(), m.data().begin());
    if (!_transpose)
      return _scalar * boost::numeric::ublas::trans(m);
    else
      return _scalar * m;
  }

  operator boost::numeric::ublas::matrix<T, boost::numeric::ublas::row_major>() {
    return ublas();
  }
};

template<class T>
device_matrix<T> trans(const device_matrix<T>& m) {
  return m.transpose();
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

}


#endif /* TBBLAS_DEVICE_MATRIX_HPP_ */
