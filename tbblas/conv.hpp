/*
 * conv.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_CONV_HPP_
#define TBBLAS_CONV_HPP_

#include <tbblas/tensor_base.hpp>

#include <tbblas/fft.hpp>
#include <thrust/transform.h>
#include <cmath>

namespace tbblas {

template<class Tensor>
struct tensor_convolution {
  Tensor tensor1, tensor2;

  tensor_convolution(const Tensor& tensor1, const Tensor& tensor2)
   : tensor1(tensor1), tensor2(tensor2) { }
};

template<class T>
struct complex_mult {
  typedef typename complex_type<T>::type complex_t;

  __host__ __device__
  complex_t operator()(const complex_t& c1, const complex_t& c2) const {
    return complex_type<T>::mult(c1, c2);
  }
};

template<class Tensor>
void apply_operation(Tensor& tensor, const tensor_convolution<Tensor>& op) {

  // calculate the convolution and write the result to the tensor
  // the fft vector is not reused by default
  const Tensor &dt1 = op.tensor1, &dt2 = op.tensor2;

  for (unsigned i = 0; i < Tensor::dimCount; ++i)
    assert(tensor.size()[i] == abs((int)dt1.size()[i] - (int)dt2.size()[i]) + 1);

  typename Tensor::dim_t ftsize;
  unsigned ftcount = 1;
  for (unsigned i = 0; i < Tensor::dimCount; ++i)
    ftcount *= (ftsize[i] = std::max(dt1.size()[i], dt2.size()[i]));

  typename Tensor::cdata_t cdata1(ftcount), cdata2(ftcount), cresult(ftcount);
  tbblas::fft(dt1, ftsize, cdata1);
  tbblas::fft(dt2, ftsize, cdata2);

  thrust::transform(cdata1.begin(), cdata1.end(), cdata2.begin(),
      cresult.begin(), complex_mult<typename Tensor::value_t>());

  tbblas::ifft(cresult, ftsize, tensor);
}

template<class T, unsigned dim, bool device>
tensor_convolution<tensor_base<T, dim, device> > conv(
    const tensor_base<T, dim, device>& dt1, const tensor_base<T, dim, device>& dt2)
{
  return tensor_convolution<tensor_base<T, dim, device> >(dt1, dt2);
}

}

#endif /* TBBLAS_CONV_HPP_ */
