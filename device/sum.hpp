/*
 * sum.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICE_SUM_HPP_
#define TBBLAS_DEVICE_SUM_HPP_

#include <tbblas/tensor_base.hpp>

#include <thrust/transform.h>

#include <iostream>

namespace tbblas {

template<class T, unsigned dim, bool device>
void sum(const tensor_base<T, dim + 1, device>& in, size_t iDim,
    tensor_base<T, dim, device>& out);

}


#endif /* TBBLAS_DEVICE_SUM_HPP_ */
