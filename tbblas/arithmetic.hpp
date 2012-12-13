/*
 * arithmetic.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_ARITHMETIC_HPP_
#define TBBLAS_ARITHMETIC_HPP_

#include <tbblas/unary_expression.hpp>
#include <tbblas/binary_expression.hpp>
#include <thrust/functional.h>

namespace tbblas {

/*** scalar operations ***/

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::plus<typename Expression::value_t> > >
>::type
operator+(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::plus<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::plus<typename Expression::value_t> >(scalar, thrust::plus<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::plus<typename Expression::value_t> > >
>::type
operator+(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::plus<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::plus<typename Expression::value_t> >(scalar, thrust::plus<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::minus<typename Expression::value_t> > >
>::type
operator-(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::minus<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::minus<typename Expression::value_t> >(scalar, thrust::minus<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::minus<typename Expression::value_t> > >
>::type
operator-(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::minus<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::minus<typename Expression::value_t> >(scalar, thrust::minus<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> > >
>::type
operator-(const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> >(-1, thrust::multiplies<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> > >
>::type
operator*(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> >(scalar, thrust::multiplies<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> > >
>::type
operator*(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::multiplies<typename Expression::value_t> >(scalar, thrust::multiplies<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::divides<typename Expression::value_t> > >
>::type
operator/(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::divides<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::divides<typename Expression::value_t> >(scalar, thrust::divides<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::divides<typename Expression::value_t> > >
>::type
operator/(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::divides<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::divides<typename Expression::value_t> >(scalar, thrust::divides<typename Expression::value_t>()));
}

/*** binary operations ***/

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

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::minus<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator-(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::minus<typename Expression1::value_t> >(
      expr1, expr2, thrust::minus<typename Expression1::value_t>());
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::multiplies<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator*(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::multiplies<typename Expression1::value_t> >(
      expr1, expr2, thrust::multiplies<typename Expression1::value_t>());
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::divides<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator/(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::divides<typename Expression1::value_t> >(
      expr1, expr2, thrust::divides<typename Expression1::value_t>());
}

}

#endif /* TBBLAS_ARITHMETIC_HPP_ */
