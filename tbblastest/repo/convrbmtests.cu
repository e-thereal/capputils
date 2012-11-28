/*
 * convrbmtests.cu
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/io.hpp>
#include <tbblas/serialize.hpp>

typedef tbblas::tensor<double, 2, false> matrix;

void convrbmtests() {
  using namespace tbblas;

  std::cout << "ConvRBM tests." << std::endl;

  matrix A(4, 3);
  A = 2, 4, 2,
      3, 2, 1,
      5, 4, 3,
      1, 3, 2;

  std::cout << "A = " << A << std::endl;

  serialize(A, "A.matrix");

  matrix B;
  deserialize("A.matrix", B);
  std::cout << "B = " << B << std::endl;

  random_tensor<double, 2, false, normal<double> > randu(4, 3);
  matrix C = randu;
  std::cout << "C = " << C << std::endl;
  //C = randu;
  //std::cout << "C = " << C << std::endl;
}
