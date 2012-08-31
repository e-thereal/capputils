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

int helloworld() {
  using namespace tbblas;
  
  const float values1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const float values2[] = {2, 3, 5, 1, 3, 2, 6, 7, 3, 1, 23, 2};
  
  tensor_t A(3, 4), B(3, 4), C(3, 4);
  
  thrust::copy(values1, values1 + A.count(), A.begin());
  thrust::copy(values2, values2 + B.count(), B.begin());
  
  C = (C = A + 2.f) + B;
  
  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "A + B = " << C << std::endl;
  
  // Copy tests
  
  std::cout << "Copy tests:\n" << std::endl;
  
  tensor_t D = A, E = copy(A), F(3, 4), G(3, 4);
  F = A;
  G = copy(A);
  
  std::cout << "A1 = " << A << std::endl;
  A = A + B;
  std::cout << "A2 = " << A << std::endl;
  std::cout << "D = " << D << std::endl;
  std::cout << "E = " << E << std::endl;
  std::cout << "F = " << F << std::endl;
  std::cout << "G = " << G << std::endl;
  
  return 0;
}
