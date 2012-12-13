/*
 * comparisons.hpp
 *
 *  Created on: Oct 3, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_COMPARISONS_HPP_
#define TBBLAS_COMPARISONS_HPP_

#include <tbblas/unary_expression.hpp>
#include <tbblas/binary_expression.hpp>

#include <thrust/functional.h>

namespace tbblas {

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::equal_to<typename Expression::value_t> > >
>::type
operator==(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::equal_to<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::equal_to<typename Expression::value_t> >(scalar, thrust::equal_to<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::equal_to<typename Expression::value_t> > >
>::type
operator==(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::equal_to<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::equal_to<typename Expression::value_t> >(scalar, thrust::equal_to<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::equal_to<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator==(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::equal_to<typename Expression1::value_t> >(
      expr1, expr2, thrust::equal_to<typename Expression1::value_t>());
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::not_equal_to<typename Expression::value_t> > >
>::type
operator!=(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::not_equal_to<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::not_equal_to<typename Expression::value_t> >(scalar, thrust::not_equal_to<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::not_equal_to<typename Expression::value_t> > >
>::type
operator!=(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::not_equal_to<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::not_equal_to<typename Expression::value_t> >(scalar, thrust::not_equal_to<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::not_equal_to<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator!=(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::not_equal_to<typename Expression1::value_t> >(
      expr1, expr2, thrust::not_equal_to<typename Expression1::value_t>());
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater<typename Expression::value_t> > >
>::type
operator>(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::greater<typename Expression::value_t> >(scalar, thrust::greater<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater<typename Expression::value_t> > >
>::type
operator>(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::greater<typename Expression::value_t> >(scalar, thrust::greater<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::greater<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator>(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::greater<typename Expression1::value_t> >(
      expr1, expr2, thrust::greater<typename Expression1::value_t>());
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less<typename Expression::value_t> > >
>::type
operator<(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::less<typename Expression::value_t> >(scalar, thrust::less<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less<typename Expression::value_t> > >
>::type
operator<(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::less<typename Expression::value_t> >(scalar, thrust::less<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::less<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator<(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::less<typename Expression1::value_t> >(
      expr1, expr2, thrust::less<typename Expression1::value_t>());
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater_equal<typename Expression::value_t> > >
>::type
operator>=(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater_equal<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::greater_equal<typename Expression::value_t> >(scalar, thrust::greater_equal<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater_equal<typename Expression::value_t> > >
>::type
operator>=(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::greater_equal<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::greater_equal<typename Expression::value_t> >(scalar, thrust::greater_equal<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::greater_equal<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator>=(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::greater_equal<typename Expression1::value_t> >(
      expr1, expr2, thrust::greater_equal<typename Expression1::value_t>());
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less_equal<typename Expression::value_t> > >
>::type
operator<=(const Expression& expr, const typename Expression::value_t& scalar) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less_equal<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::less_equal<typename Expression::value_t> >(scalar, thrust::less_equal<typename Expression::value_t>()));
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less_equal<typename Expression::value_t> > >
>::type
operator<=(const typename Expression::value_t& scalar, const Expression& expr) {
  return unary_expression<Expression, scalar_operation<typename Expression::value_t, thrust::less_equal<typename Expression::value_t> > >(
      expr, scalar_operation<typename Expression::value_t, thrust::less_equal<typename Expression::value_t> >(scalar, thrust::less_equal<typename Expression::value_t>()));
}

template<class Expression1, class Expression2>
inline typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        binary_expression<Expression1, Expression2, thrust::less_equal<typename Expression1::value_t> >
      >::type
    >::type
  >::type
>::type
operator<=(const Expression1& expr1, const Expression2& expr2) {
  return binary_expression<Expression1, Expression2, thrust::less_equal<typename Expression1::value_t> >(
      expr1, expr2, thrust::less_equal<typename Expression1::value_t>());
}

}


#endif /* TBBLAS_COMPARISONS_HPP_ */
