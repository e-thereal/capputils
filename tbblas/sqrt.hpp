/*
 * sqrt.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SQRT_HPP_
#define TBBLAS_SQRT_HPP_

#include <tbblas/scalar_expression.hpp>

namespace tbblas {

template<class T>
struct sqrt_operation {
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

}

#endif /* TBBLAS_SQRT_HPP_ */
