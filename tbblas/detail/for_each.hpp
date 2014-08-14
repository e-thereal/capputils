/*
 * for_each.hpp
 *
 *  Created on: Jul 31, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DETAIL_FOR_EACH_HPP_
#define TBBLAS_DETAIL_FOR_EACH_HPP_

#include <thrust/system/cuda/execution_policy.h>

#include <tbblas/context.hpp>
#include <tbblas/detail/system.hpp>

#include <thrust/for_each.h>

namespace tbblas {

namespace detail {

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(generic_system, InputIterator first, InputIterator last, UnaryFunction f) {
  return thrust::for_each(first, last, f);
}

template<typename InputIterator,
         typename UnaryFunction>
InputIterator for_each(device_system, InputIterator first, InputIterator last, UnaryFunction f) {
  return thrust::for_each(thrust::cuda::par.on(tbblas::context::get().stream), first, last, f);
}

}

}


#endif /* TBBLAS_DETAIL_FOR_EACH_HPP_ */
