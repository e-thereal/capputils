/*
 * scalarexpressions.cu
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>

#include <tbblas/io.hpp>
#include <tbblas/math.hpp>
#include <tbblas/sum.hpp>

typedef tbblas::tensor<float, 2, true> matrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu;
typedef typename matrix::dim_t dim_t;

void scalarexpressions() {
  using namespace tbblas;

  matrix A = randu(4,4);
  A[seq(1,1)] = 9;
  A[seq(2,2)] = 36;

  matrix B = sqrt(A);
  std::cout << "A = " << A << std::endl;
  std::cout << "sqrt(A) = " << B << std::endl;

  matrix C = 10.f * randu(10, 10) + 5.f;
  std::cout << "Avg(C) = " << sum(C) / (float)C.count() << std::endl;
}
