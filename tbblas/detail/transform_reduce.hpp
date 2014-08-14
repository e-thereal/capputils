/*
 * transform_reduce.hpp
 *
 *  Created on: Jul 31, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DETAIL_TRANSFORM_REDUCE_HPP_
#define TBBLAS_DETAIL_TRANSFORM_REDUCE_HPP_

#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>

#include <tbblas/context.hpp>
#include <tbblas/detail/system.hpp>

namespace tbblas {

namespace detail {

template<typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(generic_system,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  return thrust::transform_reduce(first, last, unary_op, init, binary_op);
}

template<typename InputIterator,
         typename UnaryFunction,
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(device_system,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  return thrust::transform_reduce(thrust::cuda::par.on(tbblas::context::get().stream), first, last, unary_op, init, binary_op);
}

}

}


#endif /* TBBLAS_DETAIL_TRANSFORM_REDUCE_HPP_ */
