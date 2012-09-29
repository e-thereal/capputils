/*
 * proxycopy.cu
 *
 *  Created on: Sep 25, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io2.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/multiplies.hpp>

#include <boost/timer.hpp>

#include <thrust/copy.h>
#include <thrust/sequence.h>

typedef tbblas::tensor<float, 2, true> matrix_t;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu_t;

void proxycopy() {
  using namespace tbblas;

  randu_t randu(4, 4);
  matrix_t A = zeros<float>(6, 6), B = randu, C(3,3), D(3,3);

  A[seq(2,2), seq(3,2)] = flip(B[seq(1,2), seq(3,2)], 0);
  C = 2 * B[seq(0,0), seq(3,3)];

  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  std::cout << "C = " << C << std::endl;
  std::cout << "B[1:3,1:3] = " << B[seq(0,0), seq(3,3)] << std::endl << std::endl;
}
