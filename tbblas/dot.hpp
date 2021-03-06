/*
 * dot2.hpp
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DOT2_HPP_
#define TBBLAS_DOT2_HPP_

#include <tbblas/type_traits.hpp>

#include <tbblas/detail/inner_product.hpp>
#include <tbblas/assert.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace tbblas {

template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if<boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
      typename Expression1::value_t
    >::type
  >::type
>::type
dot(const Expression1& expr1, const Expression2& expr2) {
  tbblas_assert(expr1.count() == expr2.count());
  return tbblas::detail::inner_product(
      typename tbblas::detail::select_system<Expression1::cuda_enabled && Expression2::cuda_enabled>::system(),
      expr1.begin(), expr1.end(), expr2.begin(), typename Expression1::value_t());
}

}

#endif /* TBBLAS_DOT2_HPP_ */
