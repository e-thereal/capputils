/*
 * math.hpp
 *
 *  Created on: Dec 11, 2012
 *      Author: tombr
 */

#ifndef GMLCONVRBM_MATH_HPP_
#define GMLCONVRBM_MATH_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/unary_expression.hpp>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T>
struct nrelu_mean_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    T var = T(1) / (T(1) + ::exp(-value));
    return ::sqrt(0.5 * var / M_PI) * ::exp(-0.5 * value * value / var) + 0.5 * value * ::erfc(-value / ::sqrt(2.0 * var));
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, nrelu_mean_operation<typename Expression::value_t> >
>::type
nrelu_mean(const Expression& expr) {
  return unary_expression<Expression, nrelu_mean_operation<typename Expression::value_t> >(expr,
      nrelu_mean_operation<typename Expression::value_t>());
}

}

#endif /* GMLCONVRBM_MATH_HPP_ */
