/*
 * benchmarks.cu
 *
 *  Created on: Mar 18, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/filter.hpp>
#include <tbblas/filter2.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/mask.hpp>

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

#if 0
  randu_t randu(32, 32, 32, 64), randKernel(5, 5, 5, 64);
  tensor_t A = randu, B, C1, C2, C3, kernel = randKernel;

  tensor_t::dim_t topleft = kernel.size() / 2, paddedSize = A.size() + kernel.size() - 1;
  topleft[3] = 0;
  paddedSize[3] = A.size()[3];

  tensor_t A2 = zeros<value_t>(paddedSize);
  A2[topleft, A.size()] = A;
  B = filter(A2, flip(flip(flip(kernel, 0), 1), 2), 3);
  C1 = B[topleft, A.size()];
  C2 = filter3d(A, kernel, naive());
  C3 = filter3d(A, kernel, optimized());
  tbblas_print(dot(C1 - C2, C1 - C2));
  tbblas_print(dot(C1 - C3, C1 - C3));
  tbblas_print(dot(C2 - C3, C2 - C3));
#endif

#if 1
  tensor<value_t, 3, true> A(4, 4, 1), A2 = zeros<value_t>(6, 6, 3), B, C, C2, kernel(3, 3, 1);
  A = 1, 2, 3, 4,
      5, 4, 3, 2,
      1, 2, 5, 6,
      1, 6, 8, 3;

  A2[seq(1,1,1), A.size()] = A;

  kernel =
      -1, 2, -1,
      -2, 4, -2,
      -1, 2, -1;

  B = filter3d(A, kernel, optimized());
  C = filter3d(A, kernel, naive());
  C2 = filter(A2, flip(kernel));

  tbblas_print(A);
  tbblas_print(kernel);
  tbblas_print(B);
  tbblas_print(C);
  tbblas_print(C2[seq(1,1,1), B.size()]);
  tbblas_print(dot(B - C, B - C));
  tbblas_print(repeat(A, seq(1,1,2)));
#endif

#if 0
  boost::timer _timer;

  const int iterations = 1000;

  randu_t randv(32, 32, 32, 64), randF(5, 5, 5, 64), randh(32, 32, 32, 1);

  tensor_t A = randv, B = randv, C = randv, C2, S = randh, kernel = randF, T;
  ctensor_t cA = fft(A, 3), cB = fft(B, 3), cC = fft(C, 3), cS, cT;
  STOP("Init")

  // Addition
  TIMER_LOOP(iterations) C = A + B;
  STOP("Addition")
  TIMER_LOOP(iterations) cC = cA + cB;
  STOP("Addition")

  // Multiplication
  TIMER_LOOP(iterations) C = A * B;
  STOP("Multiplication")
  TIMER_LOOP(iterations) cC = cA * cB;
  STOP("Multiplication")

  // Complex conjugate
  TIMER_LOOP(iterations) cC = conj(cA);
  STOP("Complex conjugate ")
  TIMER_LOOP(iterations) cC = conj(cA) * cB;
  STOP("Times complex conjugate ")

  // fft, ifft
  fft_plan<4> plan, iplan;
  TIMER_LOOP(iterations) cA = fft(A, 3);
  STOP("FFT")
  TIMER_LOOP(iterations) cA = fft(A, 3, plan);
  STOP("FFT")
  TIMER_LOOP(iterations) A = ifft(cA, 3);
  STOP("iFFT")
  TIMER_LOOP(iterations) A = ifft(cA, 3, plan);
  STOP("iFFT")

  // filtering
  TIMER_LOOP(iterations) C = filter(A, kernel, 3);
  STOP("filter")
  TIMER_LOOP(iterations) C = filter(A, flip(kernel), 3);
  STOP("filter")
//  TIMER_LOOP(iterations) C = filter3d(A, kernel, naive());
//  STOP("filter")
  TIMER_LOOP(iterations) C = filter3d(A, kernel, optimized());
  STOP("filter")

  // Sum
  TIMER_LOOP(iterations) cS = sum(cA, 3);
  STOP("sum")

  // fft, ifft of a 'slice'
  TIMER_LOOP(iterations) S = ifft(cS, 3);
  STOP("iFFT(slice)")
  TIMER_LOOP(iterations) S = ifft(cS, 3, plan);
  STOP("iFFT(slice)")
  TIMER_LOOP(iterations) cS = fft(S, 3);
  STOP("FFT(slice)")
  TIMER_LOOP(iterations) cS = fft(S, 3, plan);
  STOP("FFT(slice)")

  // nrelu_mean
  TIMER_LOOP(iterations) T = S = nrelu_mean(S);
  STOP("nrelu_mean")

  TIMER_LOOP(iterations) cC = conj(cB) * cA;
  STOP("inverse filter")
  TIMER_LOOP(iterations) cC = 0.01 * conj(cB) * cA;
  STOP("inverse filter")
  TIMER_LOOP(iterations) cC = cC + 0.01 * conj(cB) * cA;
  STOP("inverse filter")
  TIMER_LOOP(iterations) cC = cC + 0.01 * repeat(conj(cS), cA.size() / cS.size()) * cA;
  STOP("inverse filter")
  TIMER_LOOP(iterations) kernel = filter(A, C, kernel.size(), 3);
  STOP("inverse filter")
  TIMER_LOOP(iterations) kernel = filter(A, repeat(flip(S), A.size() / S.size()), kernel.size(), 3);
  STOP("inverse filter")

//  C = filter3d(A, kernel, naive());
//  C2 = filter3d(A, kernel, optimized());
//  tbblas_print(dot(C - C2, C - C2));

#endif
}
