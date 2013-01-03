/*
 * multigpu.cu
 *
 *  Created on: Dec 13, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/dot.hpp>
#include <omp.h>

using namespace tbblas;

__global__ void kernel() { }

void multigpu() {
  typedef float value_t;

  kernel<<<1,1>>>();

  const int gpuCount = 2;

  omp_set_dynamic(0);
  omp_set_num_threads(gpuCount);

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    cudaSetDevice(tid);

//    if (tid == 0) {
//      for (int i = 1; i < gpuCount; ++i)
//        cudaDeviceEnablePeerAccess(i, 0);
//    } else {
//      cudaDeviceEnablePeerAccess(0, 0);
//    }
//    #pragma omp barrier

//    random_tensor<double, 3, true, uniform<double> > randu(128, 128, 128);
//    tensor<double, 3, true> A = randu, B;
//    for (int i = tid; i < 2000; i += gpuCount)
//      B = sigm(A) > randu;

    fft_plan<3> plan, iplan;
    random_tensor<double, 3, true, uniform<double> > randu(128, 128, 4);
    tensor<double, 3, true> A = randu, B;
    tensor<complex<double>, 3, true> cA;
    #pragma omp barrier
    for (int i = 0; i < 20000 / gpuCount; ++i) {
      cA = fft(A, 2, plan);
      B = ifft(cA, 2, iplan);
      #pragma omp barrier
    }
    #pragma omp barrier
    #pragma omp critical
    std::cout << "Error: " << dot(A - B, A - B) << std::endl;
  }
}
