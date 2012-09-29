/*
 * fft2.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_FFT2_HPP_
#define TBBLAS_FFT2_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/sequence.hpp>
#include <tbblas/fft_plan.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/utility.hpp>
#include <thrust/fill.h>

#include <cufft.h>

#include <cassert>

namespace tbblas {

template<class Value>
struct fft_trait {
};

template<>
struct fft_trait<float> {
  const static cufftType type = CUFFT_R2C;

  static cufftResult exec(const cufftHandle& plan, float *idata, complex<float> *odata) {
    return cufftExecR2C(plan, idata, (cufftComplex*)odata);
  }
};

template<>
struct fft_trait<double> {
  const static cufftType type = CUFFT_D2Z;

  static cufftResult exec(const cufftHandle& plan, double *idata, complex<double> *odata) {
    return cufftExecD2Z(plan, idata, (cufftDoubleComplex*)odata);
  }
};

template<class Tensor>
struct fft_operation
{
  typedef typename Tensor::dim_t dim_t;
  typedef typename Tensor::value_t value_t;
  typedef complex<value_t> complex_t;
  typedef tensor<complex_t, Tensor::dimCount, Tensor::cuda_enabled> tensor_t;
  typedef fft_plan<Tensor::dimCount> plan_t;

  fft_operation(Tensor& tensor, const plan_t& plan) : tensor(tensor), plan(plan) { }

  void apply(tensor_t& output) const {
    fft_trait<value_t>::exec(plan.create(tensor.size(), fft_trait<value_t>::type),
        tensor.data().data().get(), output.data().data().get());
  }

  inline const dim_t& size() const {
    return tensor.size();
  }

private:
  Tensor& tensor;
  plan_t plan;
};

template<class T>
struct is_operation<fft_operation<T> > {
  static const bool value = true;
};

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if_c<Tensor::dimCount <= 3,
      fft_operation<Tensor>
    >::type
  >::type
>::type
fft(Tensor& tensor, const fft_plan<Tensor::dimCount>& plan = fft_plan<Tensor::dimCount>()) {
  return fft_operation<Tensor>(tensor, plan);
}

}

#include <tbblas/ifft.hpp>

#endif /* TBBLAS_FFT2_HPP_ */
