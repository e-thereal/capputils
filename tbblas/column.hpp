/*
 * column.hpp
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_COLUMN_HPP_
#define TBBLAS_COLUMN_HPP_

#include <tbblas/proxy.hpp>
#include <boost/utility/enable_if.hpp>
#include <cassert>

namespace tbblas {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::dimCount == 2,
    const proxy<Tensor>
  >::type
>::type
column(Tensor& tensor, unsigned columnIdx) {
  assert(columnIdx < tensor.size()[1]);
  return tensor[seq(0, (int)columnIdx), seq(tensor.size()[0], 1)];
}

}

#endif /* TBBLAS_COLUMN_HPP_ */
