/*
 * proxytests.cu
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>

#include <tbblas/linalg.hpp>

#include <thrust/sequence.h>

#include <iostream>

using namespace tbblas;

typedef tensor<double, 2, true> matrix;

void proxytests() {
  matrix M(5, 7);

  thrust::sequence(trans(M).begin(), trans(M).end());

  std::cout << "M = " << M << std::endl;
}
