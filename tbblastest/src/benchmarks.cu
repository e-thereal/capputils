/*
 * benchmarks.cu
 *
 *  Created on: Mar 18, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>

#include <boost/timer.hpp>

typedef float value_t;
typedef tbblas::complex<value_t> complex_t;
typedef tbblas::tensor<value_t, 4, true> tensor_t;
typedef tbblas::tensor<complex_t, 4, true> ctensor_t;

#define TIMER_LOOP(a) for(size_t iCycle = 0; iCycle < a; ++iCycle)
#define STOP(a) { \
    cudaStreamSynchronize(0); \
    std::cout << a << " (" << __LINE__ << "): " << _timer.elapsed() << std::endl; \
    _timer.restart(); \
}

void benchmarks() {
  using namespace tbblas;

  boost::timer _timer;

  tensor_t A(32, 32, 32, 64), B = A;
  ctensor_t cA(32, 32, 17, 64), cB = cA;
  STOP("Init")

  // Addition
  TIMER_LOOP(1000) B = A + B;
  STOP("Addition")
  TIMER_LOOP(1000) cB = cA + cB;
  STOP("Addition")

  // Multiplication
  // Complex conjugate
  // Complex conjugate + multiplication
  // fft, ifft
  // fft, ifft of a 'slice'
  // nrelu_mean


}

