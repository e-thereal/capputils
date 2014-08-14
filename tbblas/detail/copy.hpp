/*
 * copy.hpp
 *
 *  Created on: Jul 31, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DETAIL_COPY_HPP_
#define TBBLAS_DETAIL_COPY_HPP_

#include <thrust/copy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <tbblas/context.hpp>
#include <tbblas/detail/system.hpp>

namespace tbblas {

namespace detail {

template<class InputIterator,
         class OutputIterator>
OutputIterator copy(generic_system, InputIterator first, InputIterator last, OutputIterator result) {
  return thrust::copy(first, last, result);
}

template<class InputIterator,
         class OutputIterator>
OutputIterator copy(device_system, InputIterator first, InputIterator last, OutputIterator result) {
  return thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream), first, last, result);
}

}

}

#endif /* TBBLAS_DETAIL_COPY_HPP_ */
