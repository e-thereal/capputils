/*
 * proxytests.cu
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */
#include "tests.h"

#include <thrust/sequence.h>

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>

#include <tbblas/linalg.hpp>

#include <iostream>

typedef tbblas::tensor<double, 2, true> matrix;

void proxytests() {
//  using namespace tbblas;

  matrix M(5, 7);

//  thrust::sequence(tbblas::trans(M).begin(), tbblas::trans(M).end());

  std::cout << "M = " << M << std::endl;
}
