/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor_base.hpp>
#include <thrust/copy.h>

#include <tbblas/plus.hpp>
#include <tbblas/io.hpp>

#include <iostream>

typedef tbblas::tensor_base<float, 2, true> tensor_t;

void copytest() {
  using namespace tbblas;
  
  const float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  
  tensor_t A(3, 4);
  
  thrust::copy(values, values + A.count(), A.begin());
  
  tensor_t B = A, C = copy(A), D(3, 4), E(3, 4);
  D = A;
  E = copy(A);
  
  std::cout << "A1 = " << A << std::endl;
  A = A + A;
  std::cout << "A2 = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "C = " << C << std::endl;
  std::cout << "D = " << D << std::endl;
  std::cout << "E = " << E << std::endl;
}
