/*
 * fftflip.cu
 *
 *  Created on: Nov 27, 2012
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/expand.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/io.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/real.hpp>
#include <tbblas/img.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>

void fftflip() {

  using namespace tbblas;

  tensor<double, 1, true> A(5), B;
  tensor<complex<double>, 1, true> cA, cB;
  random_tensor<double, 1, true, uniform<double> > urand(5);

  A = 1, 5, 3, 2, 6;
//  A = urand;
  tbblas_print(A);

//  B = flip(A);
  B = 1, 6, 2, 3, 5;
  tbblas_print(B);

  cA = fft(A);
  cB = fft(B);

  tbblas_print(abs(cA));
  tbblas_print(abs(cB));

//  tbblas_print(abs(fftexpand(cA)));
//  tbblas_print(abs(fftexpand(cB)));
//  tbblas_print(abs(fftshift(fftexpand(cA))));
//  tbblas_print(abs(fftshift(fftexpand(cB))));
}
