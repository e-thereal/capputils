/*
 * swap.hpp
 *
 *  Created on: Jun 2, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_SWAP_HPP_
#define TBBLAS_SWAP_HPP_

#include <tbblas/tensor.hpp>
#include <thrust/swap.h>

namespace tbblas {

template<class Proxy>
typename boost::enable_if<is_proxy<Proxy>,
  void
>::type
swap(Proxy proxy1, Proxy proxy2) {
  assert(proxy1.size() == proxy2.size());
  thrust::swap_ranges(proxy1.begin(), proxy1.end(), proxy2.begin());
}

template<class Proxy, class Tensor>
typename boost::enable_if<is_proxy<Proxy>,
  typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Proxy::dimCount == Tensor::dimCount && Proxy::cuda_enabled == Tensor::cuda_enabled,
      void
    >::type
  >::type
>::type
swap(Proxy proxy1, Tensor& tensor2) {
  assert(proxy1.size() == tensor2.size());
  thrust::swap_ranges(proxy1.begin(), proxy1.end(), tensor2.begin());
}

template<class Tensor, class Proxy>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if<is_proxy<Proxy>,
    typename boost::enable_if_c<Proxy::dimCount == Tensor::dimCount && Proxy::cuda_enabled == Tensor::cuda_enabled,
      void
    >::type
  >::type
>::type
swap(Tensor& tensor1, Proxy proxy2) {
  assert(tensor1.size() == proxy2.size());
  thrust::swap_ranges(tensor1.begin(), tensor1.end(), proxy2.begin());
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  void
>::type
swap(Tensor& tensor1, Tensor& tensor2) {
  assert(tensor1.size() == tensor2.size());
  thrust::swap_ranges(tensor1.begin(), tensor1.end(), tensor2.begin());
}

}

#endif /* TBBLAS_SWAP_HPP_ */
