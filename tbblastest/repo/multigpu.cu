/*
 * multigpu.cu
 *
 *  Created on: Dec 13, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <omp.h>

void multigpu() {
  typedef float value_t;

  omp_set_dynamic(0);
  omp_set_num_threads(2);

  #pragma omp parallel
  {
    int tid2 = omp_get_thread_num();
    cudaSetDevice(tid2);
    random_tensor<double, 3, true, uniform<double> > randu(128, 128, 128);
    tensor<double, 3, true> A = randu, B;
    for (int i = 0; i < 2000 / omp_get_num_threads(); ++i)
      B = sigm(A) > randu;
  }
}
