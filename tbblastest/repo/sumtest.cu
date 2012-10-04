/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor_base.hpp>
#include <thrust/copy.h>

#include <tbblas/sum.hpp>
#include <tbblas/io.hpp>

#include <iostream>

typedef tbblas::tensor_base<float, 2, true> matrix;
typedef tbblas::tensor_base<float, 1, true> vector;
typedef tbblas::tensor_base<float, 0, true> scalar;

void sumtest() {
  using namespace tbblas;
  
  const float values1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  
  matrix A(3, 4);
  vector B(4), C(3);
  scalar D;
  
  thrust::copy(values1, values1 + A.count(), A.begin());
  
  C = sum(A, 1);
  B = sum(A, 0);
  D = sum(C, 0);
  
  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "C = " << C << std::endl;
  std::cout << "D = " << D << std::endl;
}
