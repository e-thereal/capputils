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
  typedef complex<float> output_t;
  const static cufftType type = CUFFT_R2C;

  template<class dim_t>
  static dim_t output_size(dim_t inputSize) {
    inputSize[0] = inputSize[0] / 2 + 1;
    return inputSize;
  }

  static cufftResult exec(const cufftHandle& plan, float *idata, complex<float> *odata) {
    return cufftExecR2C(plan, idata, (cufftComplex*)odata);
  }
};

template<>
struct fft_trait<double> {
  typedef complex<double> output_t;
  const static cufftType type = CUFFT_D2Z;

  template<class dim_t>
  static dim_t output_size(dim_t inputSize) {
    inputSize[0] = inputSize[0] / 2 + 1;
    return inputSize;
  }

  static cufftResult exec(const cufftHandle& plan, double *idata, complex<double> *odata) {
    return cufftExecD2Z(plan, idata, (cufftDoubleComplex*)odata);
  }
};

template<>
struct fft_trait<complex<float> > {
  typedef complex<float> output_t;
  const static cufftType type = CUFFT_C2C;

  template<class dim_t>
  static dim_t output_size(dim_t inputSize) {
    return inputSize;
  }

  static cufftResult exec(const cufftHandle& plan, complex<float> *idata, complex<float> *odata) {
    return cufftExecC2C(plan, (cufftComplex*)idata, (cufftComplex*)odata, CUFFT_FORWARD);
  }
};

template<>
struct fft_trait<complex<double> > {
  typedef complex<double> output_t;
  const static cufftType type = CUFFT_Z2Z;

  template<class dim_t>
  static dim_t output_size(dim_t inputSize) {
    return inputSize;
  }

  static cufftResult exec(const cufftHandle& plan, complex<double> *idata, complex<double> *odata) {
    return cufftExecZ2Z(plan, (cufftDoubleComplex*)idata, (cufftDoubleComplex*)odata, CUFFT_FORWARD);
  }
};

template<class Tensor>
struct fft_operation
{
  typedef typename Tensor::dim_t dim_t;
  typedef typename Tensor::value_t input_t;
  typedef typename fft_trait<input_t>::output_t output_t;
  typedef tensor<output_t, Tensor::dimCount, Tensor::cuda_enabled> tensor_t;
  typedef fft_plan<Tensor::dimCount> plan_t;

  fft_operation(Tensor& tensor, const plan_t& plan)
   : _tensor(tensor), _dimension(Tensor::dimCount), _plan(plan),
     _size(fft_trait<input_t>::output_size(tensor.size())),
     _fullsize(tensor.size())
  { }

  fft_operation(Tensor& tensor, unsigned dimension, const plan_t& plan)
   : _tensor(tensor), _dimension(dimension), _plan(plan),
     _size(fft_trait<input_t>::output_size(tensor.size())),
     _fullsize(tensor.size())
  { }

  void apply(tensor_t& output) const {
    cufftResult result;

    assert(cudaThreadSynchronize() == cudaSuccess);

    if((result = fft_trait<input_t>::exec(_plan.create(_tensor.size(), fft_trait<input_t>::type, _dimension),
            _tensor.data().data().get(), output.data().data().get())) != CUFFT_SUCCESS)
    {
      std::cout << result << std::endl;
      assert(0);
    }

    assert(cudaThreadSynchronize() == cudaSuccess);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Tensor& _tensor;
  unsigned _dimension;    ///< dimension of the fft (default is tensor dimension, if specified, multiple ffts are processed)
  plan_t _plan;
  dim_t _size;
  dim_t _fullsize;
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

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    fft_operation<Tensor>
  >::type
>::type
fft(Tensor& tensor, unsigned dimension, const fft_plan<Tensor::dimCount>& plan = fft_plan<Tensor::dimCount>()) {
  assert(dimension <= Tensor::dimCount);
  assert(dimension <= 3);
  return fft_operation<Tensor>(tensor, dimension, plan);
}

}

#include <tbblas/ifft.hpp>

#endif /* TBBLAS_FFT2_HPP_ */
