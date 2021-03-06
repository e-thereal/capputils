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
#include <tbblas/dot.hpp>

#include <tbblas/assert.hpp>

namespace tbblas {

// wrapper around the boost version
double binomial(int n, int k);

template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename Expression1::value_t
  >::type
>::type
cor(const Expression1& expr1, const Expression2& expr2) {
  typedef typename Expression1::value_t value_t;

  tbblas_assert(expr1.count() == expr2.count());

  value_t mean1 = sum(expr1) / expr1.count();
  value_t mean2 = sum(expr2) / expr2.count();

  return dot(expr1 - mean1, expr2 - mean2) / ::sqrt(dot(expr1 - mean1, expr1 - mean1) * dot(expr2 - mean2, expr2 - mean2));
}

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

template<class T>
struct bernstein_operation {
  typedef T value_t;

  T coefficient, e1, e2;

  bernstein_operation(int k, int n) {
    e1 = k;
    e2 = n - k;
    coefficient = binomial(n, k);
  }

  __host__ __device__
  T operator()(const T& x) const {
    return coefficient * ::pow(x, e1) * ::pow(1.f - x, e2);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, bernstein_operation<typename Expression::value_t> >
>::type
bernstein(const Expression& expr, int k, int n) {
  return unary_expression<Expression, bernstein_operation<typename Expression::value_t> >(expr,
      bernstein_operation<typename Expression::value_t>(k, n));
}

template<class T>
struct pow_operation {
  typedef T value_t;

  T e;

  pow_operation(const T& e) : e (e) { }

  __host__ __device__
  T operator()(const T& x) const {
    return ::pow(x, e);
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, pow_operation<typename Expression::value_t> >
>::type
pow(const Expression& expr, const typename Expression::value_t& exponent) {
  return unary_expression<Expression, pow_operation<typename Expression::value_t> >(expr,
      pow_operation<typename Expression::value_t>(exponent));
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
