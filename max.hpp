/*
 * max.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_MAX_HPP_
#define TBBLAS_MAX_HPP_

#include <tbblas/type_traits.hpp>

#include <thrust/reduce.h>

#include <boost/utility/enable_if.hpp>
#include <boost/numeric/conversion/bounds.hpp>

namespace tbblas {

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  typename Expression::value_t
>::type
max(const Expression& expr) {
  return thrust::reduce(expr.begin(), expr.end(),
      boost::numeric::bounds<typename Expression::value_t>::lowest(),
      thrust::maximum<typename Expression::value_t>());
}

}

#endif /* TBBLAS_MAX_HPP_ */
