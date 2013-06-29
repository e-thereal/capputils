/*
 * slicetest.cu
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>
#include <tbblas/rearrange.hpp>

using namespace tbblas;

typedef tensor<float, 3> volume_t;

void rearrangetest() {
  volume_t A(4,4,2), B, C;
  A = 1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
      101, 102, 103, 104,
      105, 106, 107, 108,
      109, 110, 111, 112,
      113, 114, 115, 116;

  tbblas_print(A);
  B = rearrange(A, seq(2,2,1));
  tbblas_print(B);
  C = rearrange_r(B, seq(2,2,1));
  tbblas_print(C);
}
