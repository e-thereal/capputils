/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <thrust/copy.h>

#include <tbblas/entropy.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/io.hpp>

#include <iostream>

typedef tbblas::tensor<float, 2, true> matrix;

void entropytest() {
  using namespace tbblas;
  
  matrix A(3, 4);
  
  A = 1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12;

  std::cout << "A = " << A << std::endl;
  std::cout << "Entropy = " << entropy(A, sum(A)) << std::endl;
}
