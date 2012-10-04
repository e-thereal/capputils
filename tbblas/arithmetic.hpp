/*
 * arithmetic.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_ARITHMETIC_HPP_
#define TBBLAS_ARITHMETIC_HPP_

#include <tbblas/binary_expression.hpp>
#include <thrust/functional.h>

namespace tbblas {

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::plus<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator+(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::plus<typename Expression1::value_t> >(
      expr1, expr2, thrust::plus<typename Expression1::value_t>());
}

}

#endif /* TBBLAS_ARITHMETIC_HPP_ */
