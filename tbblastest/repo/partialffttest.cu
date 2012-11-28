/*
 * partialffttest.cu
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <tbblas/linalg.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/conv.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/flip.hpp>

#include <tbblas/expand.hpp>

#include <boost/timer.hpp>

using namespace tbblas;

typedef tensor<double, 2, true> matrix;
typedef tensor<complex<double>, 2, true> cmatrix;
typedef random_tensor<double, 2, true, uniform<double> > rmatrix;

#define STOP cudaThreadSynchronize(); \
    std::cout << __LINE__ << ": " << timer.elapsed() << std::endl; \
    timer.restart();

#define TIMER_LOOP(count) for(size_t iCycle = 0; iCycle < count; ++iCycle)

void partialffttest() {
  std::cout << "Partial FFT test.\n" << std::endl;
  boost::timer timer;
  size_t timerCycles = 1000;

  rmatrix urand(6, 3);
  matrix A = urand;
//  matrix A(6, 3);
//  A = 1, 2, 3,
//      4, 5, 6,
//      7, 8, 9,
//      10, 11, 12,
//      13, 14, 15,
//      16, 17, 18;
//  std::cout << "A = " << A << std::endl;

  cmatrix B = fft(A);
  cmatrix B1d = fft(A, 1);
//  std::cout << "B = " << real(B) << std::endl;
//  std::cout << "B1d = " << real(B1d) << std::endl;
//  std::cout << "B1d = " << real(fftexpand(B1d)) << std::endl;

  for (size_t i = 0; i < A.size()[1]; ++i) {
    matrix A1 = column(A, i);
    cmatrix B1 = fft(A1);
//    std::cout << "B" << i + 1 << " = " << real(B1) << std::endl;
  }

  matrix C = ifft(B);
  matrix C1d = ifft(B1d, 1);

  std::cout << "Diff1: " << dot(A - C, A - C) << std::endl;
  std::cout << "Diff2: " << dot(A - C1d, A - C1d) << std::endl;

  matrix kernel(2, 3);
  kernel =
      1, 1, 1,
      1, 1, 1;
  kernel = kernel;

  matrix D = conv(A, kernel);
//  std::cout << "D = " << D << std::endl;

  matrix pkernel(A.size());
  pkernel[seq(0, 0), kernel.size()] = kernel;

  cmatrix ckernel = fft(pkernel, 1);
  cmatrix cD = B1d * ckernel;
  cD = sum(cD, 1);
  matrix D1d = ifft(cD, 1);
//  std::cout << "D1d = " << D1d << std::endl;

  tensor<double, 3, true> h, h2, v, F, h3, h4, pv, pF, dsum;
  tensor<complex<double>, 3, true> cF, cv, cSum;

  typedef tensor<double, 3, true>::dim_t dim_t;

  deserialize("h.tensor", h);
  deserialize("h2.tensor", h2);
  deserialize("F.tensor", F);
  deserialize("v.tensor", v);

//  std::cout << "v = " << v[seq(0,0,0), seq(5,5,3)] << std::endl;
//  std::cout << "F = " << F[seq(0,0,0), seq(5,5,3)] << std::endl;

//  F = ones<double>(2,2,2);
//  F = 1, 2,
//      3, 4,
//      5, 6,
//      7, 8;
//  v.resize(seq(3,3,2), seq(3,3,2));
//  v = 1, 2, 3,
//      4, 5, 6,
//      7, 8, 9,
//      10, 11, 12,
//      13, 14, 15,
//      16, 17, 18;

  h3 = conv(flip(F), v);

  dim_t paddedSize = v.size();
//  paddedSize[0] = 64;
//  paddedSize[1] = 64;

  pF = zeros<double>(paddedSize);
  pF[seq(0,0,0), F.size()] = flip(flip(F), 2);
  pv = zeros<double>(paddedSize);
  pv[seq(0,0,0), v.size()] = v;
  cF = fft(pF, 2);
  cv = fft(pv, 2);
  cF = cF * cv;
  STOP
  TIMER_LOOP(timerCycles)
    cSum = sum(cF[seq(0,0,0), cF.size()], 2);
  STOP
  TIMER_LOOP(timerCycles)
    cSum = sum(cF, 2);
  STOP
  pF = ifft(cF, 2);
  dim_t hiddenSize = h3.size();
  hiddenSize[2] = paddedSize[2];
  h4 = pF[paddedSize - hiddenSize, hiddenSize];

//  std::cout << "v = " << v << std::endl;
//  std::cout << "F = " << F << std::endl;
//  std::cout << "h3 = " << h3 << std::endl;
//  std::cout << "h4 = " << h4 << std::endl;
  STOP
  TIMER_LOOP(timerCycles) dsum = sum(h4, 0);
  STOP
  TIMER_LOOP(timerCycles) dsum = sum(h4, 2);
  STOP

  TIMER_LOOP(timerCycles) sum(h4[seq(0,0,0), h4.size()]);
  STOP
//  std::cout << "h4 = " << h4 << std::endl;
  std::cout << "h3 - h4: " << dot(h3 - dsum, h3 - dsum) << std::endl;

//  std::cout << "h = " << h[seq(0,0,0), seq(5,5,1)] << std::endl;
//  std::cout << "h2 = " << h2[seq(0,0,0), seq(5,5,1)] << std::endl;
}
