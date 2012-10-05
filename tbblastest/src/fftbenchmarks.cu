/*
 * fftbenchmarks.cu
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/fft.hpp>

#include <boost/timer.hpp>

typedef tbblas::tensor<float, 2, true> matrix;
typedef tbblas::tensor<tbblas::complex<float>, 2, true> cmatrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu;
typedef tbblas::fft_plan<2> plan_t;

void fftbenchmarks() {

  const int reps = 1000;
  double t1, t2;
  boost::timer timer;

  matrix A = randu(1024, 1024);
  cmatrix C = fft(A);

  cudaThreadSynchronize();
  timer.restart();

  for (int i = 0; i < reps; ++i) {
    C = fft(A);
  }
  cudaThreadSynchronize();
  std::cout << "Without plan cache: " << (t1 = timer.elapsed()) << "s." << std::endl;

  timer.restart();
  plan_t plan;
  for (int i = 0; i < reps; ++i) {
    C = fft(A, plan);
  }
  cudaThreadSynchronize();
  std::cout << "With plan cache: " << (t2 = timer.elapsed()) << "s." << std::endl;
  std::cout << "Speedup: " << t1 / t2 << "x" << std::endl;
}
