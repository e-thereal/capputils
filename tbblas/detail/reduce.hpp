/*
 * reduce.hpp
 *
 *  Created on: Jul 31, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DETAIL_REDUCE_HPP_
#define TBBLAS_DETAIL_REDUCE_HPP_

#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

#include <tbblas/context.hpp>
#include <tbblas/detail/system.hpp>

namespace tbblas {

namespace detail {

template<typename InputIterator, typename T, typename BinaryFunction>
T reduce(generic_system, InputIterator first, InputIterator last, T init, BinaryFunction binary_op) {
  return thrust::reduce(first, last, init, binary_op);
}

template<typename InputIterator, typename T, typename BinaryFunction>
T reduce(device_system, InputIterator first, InputIterator last, T init, BinaryFunction binary_op) {
  return thrust::reduce(thrust::cuda::par.on(tbblas::context::get().stream), first, last, init, binary_op);
}


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(generic_system,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  return thrust::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(device_system,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output)
{
  return thrust::reduce_by_key(
      thrust::cuda::par.on(tbblas::context::get().stream),
      keys_first, keys_last, values_first, keys_output, values_output);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(generic_system,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  return thrust::reduce_by_key(keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  reduce_by_key(device_system,
                InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op)
{
  return thrust::reduce_by_key(
      thrust::cuda::par.on(tbblas::context::get().stream),
      keys_first, keys_last, values_first, keys_output, values_output, binary_pred, binary_op);
}

}

}

#endif /* TBBLAS_DETAIL_REDUCE_HPP_ */
