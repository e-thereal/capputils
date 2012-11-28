/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/io.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/random.hpp>

#include <tbblas/linalg.hpp>

#include <iostream>

#include <boost/timer.hpp>

typedef tbblas::tensor<float, 2, false> matrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu_t;

void sumtest() {
  using namespace tbblas;
  
  matrix A(3, 4);
  matrix B, C, D;
  
  A = 1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12;
  
  tbblas_print(A);
  tbblas_print(sum(A)) << std::endl;

  B = sum(A, 0);
  tbblas_print(B);

  C = sum(A, 1);
  tbblas_print(C);
  tbblas_print(sum(C)) << std::endl;

  D = prod(C, B);
  tbblas_print(D);

  tbblas_print(row(A, 1));
  tbblas_print(column(A, 3));

  matrix E = prod(column(A, 1), row(A, 2));
  tbblas_print(E);

  std::cout << trans(A[seq(0,1), seq(2,2)]) << std::endl;
}
