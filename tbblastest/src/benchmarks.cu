/*
 * benchmarks.cu
 *
 *  Created on: Mar 18, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/filter.hpp>
#include <tbblas/filter2.hpp>
#include <tbblas/io.hpp>

#include <boost/timer.hpp>

#include "math.hpp"

typedef float value_t;
typedef tbblas::complex<value_t> complex_t;
typedef tbblas::tensor<value_t, 4, true> tensor_t;
typedef tbblas::tensor<complex_t, 4, true> ctensor_t;
typedef tbblas::random_tensor<value_t, 4, true, tbblas::uniform<value_t> > randu_t;

#define TIMER_LOOP(a) for(size_t iCycle = 0; iCycle < a; ++iCycle)
#define STOP(a) { \
    cudaStreamSynchronize(0); \
    std::cout << a << " (" << __LINE__ << "): " << _timer.elapsed() << std::endl; \
    _timer.restart(); \
}

void benchmarks() {
  using namespace tbblas;

  boost::timer _timer;

#if 0
  tensor<value_t, 3, true> A(4, 4, 1), A2 = zeros<value_t>(6, 6, 3), B, C, kernel(3, 3, 1);
  A = 1, 2, 3, 4,
      5, 4, 3, 2,
      1, 2, 5, 6,
      1, 6, 8, 3;

  A2[seq(1,1,1), A.size()] = A;

  kernel =
      -1, 2, -1,
      -2, 4, -2,
      -1, 2, -1;

  B = filter3d(A, kernel);
  C = filter(A2, flip(kernel));

  tbblas_print(A);
  tbblas_print(kernel);
  tbblas_print(B);
  tbblas_print(C);
//  tbblas_print(C[seq(1,1,1), B.size()]);
#endif

#if 1
  randu_t randu(32, 32, 32, 64), randKernel(5, 5, 5, 64);

  tensor_t A = randu, B = randu, C = randu, S, kernel = randKernel, T;
  ctensor_t cA = fft(A, 3), cB = fft(B, 3), cC = fft(C, 3), cS, cT;
  STOP("Init")

  // Addition
  TIMER_LOOP(1000) C = A + B;
  STOP("Addition")
  TIMER_LOOP(1000) cC = cA + cB;
  STOP("Addition")

  // Multiplication
  TIMER_LOOP(1000) C = A * B;
  STOP("Multiplication")
  TIMER_LOOP(1000) cC = cA * cB;
  STOP("Multiplication")

  // Complex conjugate
  TIMER_LOOP(1000) cC = conj(cA);
  STOP("Complex conjugate ")
  TIMER_LOOP(1000) cC = conj(cA) * cB;
  STOP("Times complex conjugate ")

  // fft, ifft
  fft_plan<4> plan, iplan;
  TIMER_LOOP(1000) cA = fft(A, 3);
  STOP("FFT")
  TIMER_LOOP(1000) cA = fft(A, 3, plan);
  STOP("FFT")
  TIMER_LOOP(1000) A = ifft(cA, 3);
  STOP("iFFT")
  TIMER_LOOP(1000) A = ifft(cA, 3, plan);
  STOP("iFFT")

  // Sum
  TIMER_LOOP(1000) cS = sum(cA, 3);
  STOP("sum")

  // fft, ifft of a 'slice'
  TIMER_LOOP(1000) S = ifft(cS, 3);
  STOP("iFFT(slice)")
  TIMER_LOOP(1000) S = ifft(cS, 3, plan);
  STOP("iFFT(slice)")
  TIMER_LOOP(1000) cS = fft(S, 3);
  STOP("FFT(slice)")
  TIMER_LOOP(1000) cS = fft(S, 3, plan);
  STOP("FFT(slice)")

  // nrelu_mean
  TIMER_LOOP(1000) T = S = nrelu_mean(S);
  STOP("nrelu_mean")

  // filtering
  TIMER_LOOP(1000) C = filter(A, kernel, 3);
  STOP("filter")
  TIMER_LOOP(1000) C = filter3d(A, kernel);
  STOP("filter")
#endif
}
