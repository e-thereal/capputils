/*
 * proxybenchmarks.cu
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>

#include <boost/timer.hpp>

#include <thrust/copy.h>

typedef tbblas::tensor<float, 2, true> matrix_t;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu_t;

void proxybenchmarks() {
  using namespace tbblas;

  const int N = 10245;
  const int M = 1000;
  const int shifts = 8;
  const int reps = 1000;

  randu_t randu(M, M);

  matrix_t A = zeros<float>(N, N);
  matrix_t B = randu;
  tensor<float, 2, false> C(shifts, shifts), D(shifts, shifts);

  boost::timer timer;

  for (int i = 0; i < shifts; ++i) {
    for (int j = 0; j < shifts; ++j) {
      cudaThreadSynchronize();
      timer.restart();
      for (int k = 0; k < reps; ++k) {
        A[seq(i,j), seq(M,M)] = B;
      }
      cudaThreadSynchronize();
      C[seq(i,j)] = timer.elapsed();

      timer.restart();
      for (int k = 0; k < reps; ++k) {
        thrust::copy(B.begin(), B.end(), A[seq(i,j), seq(M,M)].begin());
      }
      cudaThreadSynchronize();
      D[seq(i,j)] = timer.elapsed();
    }
  }

  std::cout << "C = " << C << std::endl;
  std::cout << "D = " << D << std::endl;
}
