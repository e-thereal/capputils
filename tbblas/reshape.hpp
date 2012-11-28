/*
 * reshape.hpp
 *
 *  Created on: Nov 26, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_RESHAPE_HPP_
#define TBBLAS_RESHAPE_HPP_

#include <tbblas/proxy.hpp>

namespace tbblas {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  proxy<Tensor>
>::type
reshape(Tensor& tensor, const typename Tensor::dim_t& size) {
  return proxy<Tensor>(tensor, size);
}

}

#endif /* TBBLAS_RESHAPE_HPP_ */
