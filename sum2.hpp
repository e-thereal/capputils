/*
 * sum2.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SUM2_HPP_
#define TBBLAS_SUM2_HPP_

#include <tbblas/type_traits.hpp>

#include <thrust/reduce.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  typename Expression::value_t
>::type
sum(const Expression& expr) {
  return thrust::reduce(expr.begin(), expr.end(), typename Expression::value_t());
}

}


#endif /* TBBLAS_SUM2_HPP_ */
