/*
 * swaptest.cu
 *
 *  Created on: Jun 2, 2015
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/swap.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <thrust/swap.h>

using namespace tbblas;

typedef tensor<float, 2, true> matrix_t;

void swaptest() {

  random_tensor2<float, 2, true, uniform<float> > urand(4,4);

  matrix_t A = 9 * urand(), B = 9 * urand();

  tbblas_print(A);
  tbblas_print(B);

  swap(A[seq(0,0), seq(3,2)], B[seq(1,1), seq(3,2)]);
//  thrust::swap_ranges(A.begin(), A.end(), B.begin());

  tbblas_print(A);
  tbblas_print(B);
}
