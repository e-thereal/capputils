/*
 * math.hpp
 *
 *  Created on: Oct 2, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_MATH_HPP_
#define TBBLAS_MATH_HPP_

#include <tbblas/scalar_expression.hpp>
#include <tbblas/complex.hpp>

namespace tbblas {

/*** SQRT ***/

template<class T>
struct sqrt_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return ::sqrt(value);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  scalar_expression<Expression, sqrt_operation<typename Expression::value_t> >
>::type
sqrt(const Expression& expr) {
  return scalar_expression<Expression, sqrt_operation<typename Expression::value_t> >(expr,
      sqrt_operation<typename Expression::value_t>());
}

/*** LOG ***/

template<class T>
struct log_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return ::log(value);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  scalar_expression<Expression, log_operation<typename Expression::value_t> >
>::type
log(const Expression& expr) {
  return scalar_expression<Expression, log_operation<typename Expression::value_t> >(expr,
      log_operation<typename Expression::value_t>());
}

/*** floor ***/

template<class T>
struct floor_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return ::floor(value);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  scalar_expression<Expression, floor_operation<typename Expression::value_t> >
>::type
floor(const Expression& expr) {
  return scalar_expression<Expression, floor_operation<typename Expression::value_t> >(expr,
      floor_operation<typename Expression::value_t>());
}

/*** ABS ***/

template<class T>
struct abs_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return ::abs(value);
  }
};

template<class T>
struct abs_operation<complex<T> > {
  typedef T value_t;

  __host__ __device__
  T operator()(const complex<T>& value) const {
    return ::sqrt(value.real * value.real + value.img * value.img);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  scalar_expression<Expression, abs_operation<typename Expression::value_t> >
>::type
abs(const Expression& expr) {
  return scalar_expression<Expression, abs_operation<typename Expression::value_t> >(expr,
      abs_operation<typename Expression::value_t>());
}

}

#endif /* TBBLAS_MATH_HPP_ */
