/*
 * copy.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_COPY_HPP_
#define TBBLAS_COPY_HPP_

namespace tbblas {

template<class Tensor>
struct tensor_copy : public tensor_operation<Tensor> {
  Tensor tensor;

  tensor_copy(const Tensor& tensor) : tensor(tensor) { }
};

//  template<class T2, bool device2>
//  void apply(const tensor_copy<tensor_base<T2, dim, device2> >& copy_op)
//  {
//    const tensor_base<T2, dim, device2>& tensor = copy_op.tensor;
//    const dim_t& size = tensor.size();
//
//    for (unsigned i = 0; i < dim; ++i) {
//      assert(_size[i] == size[i]);    // todo: throw exception instead
//    }
//    thrust::copy(tensor.cbegin(), tensor.cend(), begin());
//  }

//template<class T, unsigned dim, bool device>
//tensor_copy<tensor_base<T, dim, device> > copy(const tensor_base<T, dim, device>& tensor) {
//  return tensor_copy<tensor_base<T, dim, device> >(tensor);
//}

}


#endif /* TBBLAS_COPY_HPP_ */
