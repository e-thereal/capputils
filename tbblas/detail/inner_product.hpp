/*
 * inner_product.hpp
 *
 *  Created on: Jul 31, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DETAIL_INNER_PRODUCT_HPP_
#define TBBLAS_DETAIL_INNER_PRODUCT_HPP_

#include <thrust/inner_product.h>
#include <thrust/system/cuda/execution_policy.h>

#include <tbblas/context.hpp>
#include <tbblas/detail/system.hpp>


namespace tbblas {

namespace detail {

template<typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType inner_product(generic_system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputType init) {
  return thrust::inner_product(first1, last1, first2, init);
}

template<typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType inner_product(device_system, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputType init) {
  return thrust::inner_product(thrust::cuda::par.on(tbblas::context::get().stream), first1, last1, first2, init);
}

}

}

#endif /* TBBLAS_DETAIL_INNER_PRODUCT_HPP_ */
