/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <thrust/copy.h>

#include <tbblas/io.hpp>
#include <tbblas/conv.hpp>
#include <tbblas/flip.hpp>

#include <iostream>

typedef tbblas::tensor<float, 2, true> tensor_t;

void convtest() {
  using namespace tbblas;
  
  const float values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const float kernel[] = {0.25, -0.25, 0.25, -0.25};
  
  tensor_t A(3, 4), F(2, 2), B(2, 3);
  
  thrust::copy(values, values + A.count(), A.begin());
  thrust::copy(kernel, kernel + F.count(), F.begin());
  
  B = conv(A, flip(F));
  
  std::cout << "A = " << A << std::endl;
  std::cout << "F = " << F << std::endl;
  std::cout << "conv(A, F) = " << B << std::endl;
}
