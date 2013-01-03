/*
 * copytest.cu
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>

using namespace tbblas;

typedef tensor<double, 2> matrix;

void copytest() {
  matrix A(3,3), B;

  A = 4, 3, 1,
      5, 3, 1,
      5, 3, 5;

  tbblas_print(A);

  B = A;
  A = 1, 1, 1,
      3, 3, 3,
      5, 5, 5;

  tbblas_print(A);
  tbblas_print(B);
}
