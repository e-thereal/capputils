/*
 * ifft.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_IFFT_HPP_
#define TBBLAS_IFFT_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/divides.hpp>

#include <boost/utility/enable_if.hpp>

#include <cufft.h>

#include <cassert>

namespace tbblas {

template<class Value>
struct ifft_trait {
};

template<>
struct ifft_trait<float> {
  const static cufftType type = CUFFT_C2R;

  static cufftResult exec(const cufftHandle& plan, complex<float> *idata, float *odata) {
    return cufftExecC2R(plan, (cufftComplex*)idata, odata);
  }
};

template<>
struct ifft_trait<double> {
  const static cufftType type = CUFFT_Z2D;

  static cufftResult exec(const cufftHandle& plan, complex<double> *idata, double *odata) {
    return cufftExecZ2D(plan, (cufftDoubleComplex*)idata, odata);
  }
};

template<class Tensor>
struct ifft_operation
{
  typedef typename Tensor::dim_t dim_t;
  typedef typename Tensor::value_t complex_t;
  typedef typename complex_t::value_t value_t;

  typedef tensor<value_t, Tensor::dimCount, Tensor::cuda_enabled> tensor_t;
  typedef fft_plan<Tensor::dimCount> plan_t;

  ifft_operation(Tensor& tensor, const plan_t& plan) : _tensor(tensor), _plan(plan) { }

  void apply(tensor_t& output) const {
    assert(cudaThreadSynchronize() == cudaSuccess);
    assert(ifft_trait<value_t>::exec(_plan.create(_tensor.size(), ifft_trait<value_t>::type),
        _tensor.data().data().get(), output.data().data().get()) == CUFFT_SUCCESS);
    assert(cudaThreadSynchronize() == cudaSuccess);
    output = output / (value_t)output.count();
  }

  inline dim_t size() const {
    return _tensor.size();
  }

private:
  Tensor& _tensor;
  plan_t _plan;
};

template<class T>
struct is_operation<ifft_operation<T> > {
  static const bool value = true;
};

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if_c<Tensor::dimCount <= 3,
      typename boost::enable_if<is_complex<typename Tensor::value_t>,
        ifft_operation<Tensor>
      >::type
    >::type
  >::type
>::type
ifft(Tensor& tensor, const fft_plan<Tensor::dimCount>& plan = fft_plan<Tensor::dimCount>()) {
  return ifft_operation<Tensor>(tensor, plan);
}

}

#endif /* TBBLAS_IFFT_HPP_ */
