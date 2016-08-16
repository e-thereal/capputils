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
#include <tbblas/fft_plan.hpp>
#include <tbblas/context.hpp>
#include <tbblas/assert.hpp>

#include <boost/utility/enable_if.hpp>

#include <cufft.h>

#include <sstream>
#include <stdexcept>

namespace tbblas {

template<class Value>
struct ifft_trait {
};

template<>
struct ifft_trait<float> {
  const static cufftType type = CUFFT_C2R;

  static cufftResult exec(const cufftHandle& plan, complex<float> *idata, float *odata) {
    cufftSetStream(plan, tbblas::context::get().stream);
    return cufftExecC2R(plan, (cufftComplex*)idata, odata);
  }
};

template<>
struct ifft_trait<double> {
  const static cufftType type = CUFFT_Z2D;

  static cufftResult exec(const cufftHandle& plan, complex<double> *idata, double *odata) {
    cufftSetStream(plan, tbblas::context::get().stream);
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

  ifft_operation(Tensor& tensor, const plan_t& plan)
   : _tensor(tensor), _dimension(Tensor::dimCount), _plan(plan) { }

  ifft_operation(Tensor& tensor, unsigned dimension, const plan_t& plan)
   : _tensor(tensor), _dimension(dimension), _plan(plan) { }

  void apply(tensor_t& output) const {
    cufftResult result = CUFFT_SUCCESS;

    size_t count = 1;
    for (unsigned i = 0; i < _dimension; ++i)
      count *= output.size()[i];

#ifndef TBBLAS_NO_BATCHED_FFT
    if ((result = ifft_trait<value_t>::exec(_plan.create(_tensor.fullsize(), ifft_trait<value_t>::type, _dimension),
        _tensor.data().data().get(), output.data().data().get())) != CUFFT_SUCCESS)
    {
      std::stringstream s;
      s << "Could not execute FFT plan. Error code: " << result;
      throw std::runtime_error(s.str().c_str());
    }
#else
    // determine the size of a sub tensor, the number of elements of a sub tensor and the number of sub tensors
    dim_t inSize = _tensor.size(), outSize = output.size();
    for (unsigned i = _dimension; i < tensor_t::dimCount; ++i) {
      inSize[i] = outSize[i] = 1;
    }

    size_t inCount = 1, outCount = 1;
    for (unsigned i = 0; i < _dimension; ++i) {
      inCount *= inSize[i];
      outCount *= outSize[i];
    }

    size_t batchCount = _tensor.count() / inCount;

    for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {
      if ((result = ifft_trait<value_t>::exec(_plan.create(outSize, ifft_trait<value_t>::type, _dimension),
            _tensor.data().data().get() + iBatch * inCount,
            output.data().data().get() + iBatch * outCount)
          ) != CUFFT_SUCCESS)
      {
        std::cout << result << std::endl;
        tbblas_assert(0);
      }
    }
#endif
    output = output / (value_t)count;
  }

  inline dim_t size() const {
    return _tensor.fullsize();
  }

  inline dim_t fullsize() const {
    return _tensor.fullsize();
  }

private:
  Tensor& _tensor;
  unsigned _dimension;
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

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if<is_complex<typename Tensor::value_t>,
      ifft_operation<Tensor>
    >::type
  >::type
>::type
ifft(Tensor& tensor, unsigned dimension, const fft_plan<Tensor::dimCount>& plan = fft_plan<Tensor::dimCount>()) {
  return ifft_operation<Tensor>(tensor, dimension, plan);
}

}

#endif /* TBBLAS_IFFT_HPP_ */
