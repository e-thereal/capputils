/*
 * plus.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_PLUS_HPP_
#define TBBLAS_PLUS_HPP_

#include <thrust/fill.h>

namespace tbblas {

template<class Tensor>
struct tensor_element_plus {
  Tensor tensor1, tensor2;

  tensor_element_plus(const Tensor& tensor1, const Tensor& tensor2)
   : tensor1(tensor1), tensor2(tensor2) { }
};

template<class Tensor>
struct tensor_scalar_plus {
  Tensor tensor;
  typename Tensor::value_t scalar;

  tensor_scalar_plus(const Tensor& tensor, const typename Tensor::value_t& scalar)
   : tensor(tensor), scalar(scalar) { }
};

template<class Tensor>
void apply_operation(Tensor& tensor, const tensor_element_plus<Tensor>& op)
{
  const Tensor& dt1 = op.tensor1;
  const Tensor& dt2 = op.tensor2;

  for (unsigned i = 0; i < Tensor::dimCount; ++i) {
    assert(dt1.size()[i] == dt2.size()[i]);
    assert(tensor.size()[i] == dt1.size()[i]);
  }

  if (dt1.unit_scale() && dt2.unit_scale()) {
    thrust::transform(dt1.frbegin(), dt1.frend(), dt2.frbegin(),
        tensor.begin(), thrust::plus<typename Tensor::value_t>());
  } else {
    thrust::transform(dt1.cbegin(), dt1.cend(), dt2.cbegin(),
        tensor.begin(), thrust::plus<typename Tensor::value_t>());
  }
}

template<class Tensor>
void apply_operation(Tensor& tensor, const tensor_scalar_plus<Tensor>& op) {
  using namespace thrust::placeholders;

  const Tensor& dt = op.tensor;
  const typename Tensor::value_t& scalar = op.scalar;

  if (dt.unit_scale())
    thrust::transform(dt.frbegin(), dt.frend(), tensor.begin(), _1 + scalar);
  else
    thrust::transform(dt.cbegin(), dt.cend(), tensor.begin(), _1 + scalar);
}

} /* end namespace tbblas */

template<class T, unsigned dim, bool device>
tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> > operator+(
    const tbblas::tensor_base<T, dim, device>& dt1, const tbblas::tensor_base<T, dim, device>& dt2)
{
  return tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> >(dt1, dt2);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> > operator-(
    const tbblas::tensor_base<T, dim, device>& dt1, const tbblas::tensor_base<T, dim, device>& dt2)
{
  return tbblas::tensor_element_plus<tbblas::tensor_base<T, dim, device> >(dt1, T(-1) * dt2);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> > operator+(
    const tbblas::tensor_base<T, dim, device>& dt, const T& scalar)
{
  return tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> >(dt, scalar);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> > operator+(
    const T& scalar, const tbblas::tensor_base<T, dim, device>& dt)
{
  return tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> >(dt, scalar);
}

template<class T, unsigned dim, bool device>
tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> > operator-(
    const tbblas::tensor_base<T, dim, device>& dt, const T& scalar)
{
  return tbblas::tensor_scalar_plus<tbblas::tensor_base<T, dim, device> >(dt, -scalar);
}

#endif /* PLUS_HPP_ */
