/*
 * max.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_MAX_HPP_
#define TBBLAS_MAX_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/binary_expression.hpp>

#include <tbblas/detail/reduce.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/numeric/conversion/bounds.hpp>

namespace tbblas {

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  typename Expression::value_t
>::type
max(const Expression& expr) {
  return tbblas::detail::reduce(
      typename tbblas::detail::select_system<Expression::cuda_enabled>::system(),
      expr.begin(), expr.end(),
      boost::numeric::bounds<typename Expression::value_t>::lowest(),
      thrust::maximum<typename Expression::value_t>());
}

template<class T>
struct max_operation {
  typedef T value_t;

  __host__ __device__
  value_t operator()(const value_t& value1, const value_t& value2) const {
    return value1 < value2 ? value2 : value1;
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_last_operation<typename Expression::value_t, max_operation<typename Expression::value_t> > >
>::type
max(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_last_operation<typename Expression::value_t, max_operation<typename Expression::value_t> > >(
      expr, scalar_last_operation<typename Expression::value_t, max_operation<typename Expression::value_t> >(scalar, max_operation<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_first_operation<typename Expression::value_t, max_operation<typename Expression::value_t> > >
>::type
max(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_first_operation<typename Expression::value_t, max_operation<typename Expression::value_t> > >(
      expr, scalar_first_operation<typename Expression::value_t, max_operation<typename Expression::value_t> >(scalar, max_operation<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, max_operation<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
max(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, max_operation<typename Expression1::value_t> >(
      expr1, expr2, max_operation<typename Expression1::value_t>());
}

}

#endif /* TBBLAS_MAX_HPP_ */
