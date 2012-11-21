/*
 * row.hpp
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_ROW_HPP_
#define TBBLAS_ROW_HPP_

#include <tbblas/proxy.hpp>
#include <boost/utility/enable_if.hpp>
#include <cassert>

namespace tbblas {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::dimCount == 2,
    proxy<Tensor>
  >::type
>::type
row(Tensor& tensor, unsigned rowIdx) {
  assert(rowIdx < tensor.size()[0]);
  return tensor[seq((int)rowIdx, 0), seq(1, tensor.size()[1])];
}

}

#endif /* ROW_HPP_ */
