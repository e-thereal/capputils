/*
 * synctest.cu
 *
 *  Created on: Jul 22, 2014
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/io.hpp>

#include <tbblas/deeplearn/conv_rbm.hpp>
#include <tbblas/util.hpp>

#include <thrust/system/cuda/execution_policy.h>
//#include <thrust/execution_policy.h>

#include <tbblas/detail/copy.hpp>
#include <tbblas/prod.hpp>
#include <tbblas/sum.hpp>

#include <omp.h>
//#include <unistd.h>

//#include <boost/thread/thread.hpp>
//#include <boost/thread/barrier.hpp>

#include <tbblas/deeplearn/math.hpp>

#include <tbblas/new_context.hpp>
#include <tbblas/context_manager.hpp>
#include <tbblas/reshape.hpp>

void do_something(int tid) {
  using namespace tbblas;

  tbblas::new_context context;

  typedef tensor<float, 3, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  dim_t dim = seq(128, 128, 32);

  cudaSetDevice(0);

  switch (tid) {
  case 0:
    {
      std::cout << "Thread 1: " << boost::this_thread::get_id() << std::endl;
      tensor<float, 3, true> A(dim), B(dim), C(dim);
//      cudaStreamSynchronize(0);
//      #pragma omp barrier

      for (size_t i = 0; i < 1000; ++i) {
//        C = A + B;
        thrust::transform(thrust::cuda::par.on(tbblas::context::get().stream), A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<float>());
//        cudaStreamSynchronize(s);
//        #pragma omp barrier
      }
      synchronize();
      std::cout << "Done 1: " << tbblas::context::get().stream << std::endl;
    }
    break;

  case 1:
    {
      std::cout << "Thread 2: " << boost::this_thread::get_id() << std::endl;
      tensor<float, 3, true> A(dim), B(dim), C(dim);
      tensor<complex<float>, 3, true> D;
//      cudaStreamSynchronize(0);
//      #pragma omp barrier
      for (size_t i = 0; i < 100; ++i) {
//        C = tbblas::deeplearn::nrelu_mean(A + B);
//        D = fft(C);
//        sum(C);
        thrust::transform(thrust::cuda::par.on(tbblas::context::get().stream), A.begin(), A.end(), B.begin(), C.begin(), thrust::plus<float>());
//        thrust::transform(thrust::cuda::par.on(s), C.begin(), C.end(), C.begin(), tbblas::deeplearn::nrelu_mean_operation<float>());
//        cudaStreamSynchronize(s);
//        #pragma omp barrier
      }
      synchronize();
      std::cout << "Done 2: " << tbblas::context::get().stream << std::endl;
    }
    break;
  }
}

void synctest() {
  std::cout << "Hello" << std::endl;
  tbblas::new_context context;

  tbblas::tensor<float, 2, true> matA(3,3), matB(1,3), matC(3,1), res;

  matA = 2, 1, 4,
         1, 4, 5,
         3, 1, 5;

  matB = 1, 3, 4;

  matC = 4,
         2,
         1;

  tbblas_print(matA);
  tbblas_print(matB);
  tbblas_print(matC);
  tbblas_print(res = prod(matB, matC));
  tbblas_print(res = prod(matC, matB));

  tbblas_print(res = sum(matA,0));
  tbblas_print(res = sum(matA,1));
  tbblas_print(res = sum(matB,0));
  tbblas_print(res = sum(matB,1));
  tbblas_print(res = sum(matC,0));
  tbblas_print(res = sum(matC,1));

  tbblas::tensor<float, 1> flatA = tbblas::reshape(matA, 9);
  tbblas::synchronize();
  tbblas_print(flatA);

  return;
#if 0
  std::cout << "Current context: " << tbblas::context::get().stream << std::endl;

//  tbblas::new_context context;
  {
    tbblas::new_context context;
    std::cout << "Current context: " << tbblas::context::get().stream << std::endl;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(2);

  std::cout << "Current context: " << tbblas::context::get().stream << std::endl;

//  #pragma omp parallel
//  do_something(omp_get_thread_num());
//  do_something(0);
//  do_something(1);

//  if (fork()) {
//    do_something(0);
//  } else {
//    do_something(1);
//  }

//  boost::thread thread1(do_something, 0);
//  boost::thread thread2(do_something, 1);
//
//  thread1.join();
//  thread2.join();

  tbblas::tensor<float, 3> A(256, 256, 64);
  tbblas::tensor<float, 3, true> C(256,256,64), D(256,256,64), E(256,256,64);

  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  {
    tbblas::new_context context(s1);
    for (int i = 0; i < 5; ++i) {
      C = A;
      E = tbblas::deeplearn::nrelu_mean(D * D);
      E = tbblas::deeplearn::nrelu_mean(D * D);
      A = C;
      {
        tbblas::new_context context(s2);
        C = A;
        E = tbblas::deeplearn::nrelu_mean(D * D);
        E = tbblas::deeplearn::nrelu_mean(D * D);
        A = C;
      }
    }
  }

  cudaDeviceSynchronize();

  {
    tbblas::new_context context(s1);
    for (int i = 0; i < 5; ++i) {
      C = A;
      E = tbblas::deeplearn::nrelu_mean(D * D);
      E = tbblas::deeplearn::nrelu_mean(D * D);
      A = C;
      {
        tbblas::new_context context(s2);
        C = A;
        E = tbblas::deeplearn::nrelu_mean(D * D);
        E = tbblas::deeplearn::nrelu_mean(D * D);
        A = C;
      }
    }
  }

  cudaDeviceSynchronize();
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

//  #pragma omp parallel
//  {
//    tbblas::new_context context;
//    for (int i = 0; i < 5; ++i) {
//      #pragma omp critical
//      {
//        C = A;
//        E = tbblas::deeplearn::nrelu_mean(D * D);
//        E = tbblas::deeplearn::nrelu_mean(D * D);
//        A = C;
//      }
//      tbblas::synchronize();
//    }
//  }
#endif
}
