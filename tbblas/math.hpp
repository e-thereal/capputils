/*
 * math.hpp
 *
 *  Created on: Oct 2, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_MATH_HPP_
#define TBBLAS_MATH_HPP_

#include <tbblas/unary_expression.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/min.hpp>
#include <tbblas/max.hpp>

namespace tbblas {

template<class T>
struct p_norm_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return T(0.5)/M_PI * ::exp(-0.5 * value * value);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, p_norm_operation<typename Expression::value_t> >
>::type
p_norm(const Expression& expr) {
  return unary_expression<Expression, p_norm_operation<typename Expression::value_t> >(expr,
      p_norm_operation<typename Expression::value_t>());
}

/*** ERFC ***/


template<class T>
struct erfc_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return ::erfc(value);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, erfc_operation<typename Expression::value_t> >
>::type
erfc(const Expression& expr) {
  return unary_expression<Expression, erfc_operation<typename Expression::value_t> >(expr,
      erfc_operation<typename Expression::value_t>());
}

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
  unary_expression<Expression, sqrt_operation<typename Expression::value_t> >
>::type
sqrt(const Expression& expr) {
  return unary_expression<Expression, sqrt_operation<typename Expression::value_t> >(expr,
      sqrt_operation<typename Expression::value_t>());
}

/*** SIGMOID ***/

template<class T>
struct sigm_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return T(1)/ (T(1) + ::exp(-value));
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, sigm_operation<typename Expression::value_t> >
>::type
sigm(const Expression& expr) {
  return unary_expression<Expression, sigm_operation<typename Expression::value_t> >(expr,
      sigm_operation<typename Expression::value_t>());
}

/*** EXP ***/

template<class T>
struct exp_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return ::exp(value);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, exp_operation<typename Expression::value_t> >
>::type
exp(const Expression& expr) {
  return unary_expression<Expression, exp_operation<typename Expression::value_t> >(expr,
      exp_operation<typename Expression::value_t>());
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
  unary_expression<Expression, log_operation<typename Expression::value_t> >
>::type
log(const Expression& expr) {
  return unary_expression<Expression, log_operation<typename Expression::value_t> >(expr,
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
  unary_expression<Expression, floor_operation<typename Expression::value_t> >
>::type
floor(const Expression& expr) {
  return unary_expression<Expression, floor_operation<typename Expression::value_t> >(expr,
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
  unary_expression<Expression, abs_operation<typename Expression::value_t> >
>::type
abs(const Expression& expr) {
  return unary_expression<Expression, abs_operation<typename Expression::value_t> >(expr,
      abs_operation<typename Expression::value_t>());
}

//template<class T>
//struct phase_operation {
//  typedef T value_t;
//
//  __host__ __device__
//  T operator()(const T& value) const {
//    return 0;
//  }
//};
//
//template<class T>
//struct phase_operation<complex<T> > {
//  typedef T value_t;
//
//  __host__ __device__
//  T operator()(const T& value) const {
//    return 0;
//  }
//};
//
//template<class Expression>
//inline typename boost::enable_if<is_expression<Expression>,
//  scalar_expression<Expression, phase_operation<typename Expression::value_t> >
//>::type
//phase(const Expression& expr) {
//  return scalar_expression<Expression, phase_operation<typename Expression::value_t> >(expr,
//      phase_operation<typename Expression::value_t>());
//}

template<class T>
struct conjugate_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    return value;
  }
};

template<class T>
struct conjugate_operation<complex<T> > {
  typedef complex<T> value_t;

  __host__ __device__
  value_t operator()(const value_t& value) const {
    return value_t(value.real, -value.img);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, conjugate_operation<typename Expression::value_t> >
>::type
conj(const Expression& expr) {
  return unary_expression<Expression, conjugate_operation<typename Expression::value_t> >(expr,
      conjugate_operation<typename Expression::value_t>());
}

}

#endif /* TBBLAS_MATH_HPP_ */
