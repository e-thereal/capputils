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
#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>

#include <boost/timer.hpp>

typedef double value_t;
typedef tbblas::tensor<value_t, 4, true> volume;
typedef tbblas::tensor<tbblas::complex<value_t>, 4, true> cvolume;
typedef tbblas::random_tensor<value_t, 4, true, tbblas::uniform<value_t> > randu;
typedef tbblas::fft_plan<4> plan_t;

void fftbenchmarks() {
  const int reps = 1000;
  double t1, t2;
  boost::timer timer;

  tbblas::sequence<int, 3> seq1 = 2 * tbblas::seq(0,0,0);

  std::cout << "Creating R ... " << std::flush;
//  randu R(128, 128, 64);
  randu R(42, 54, 27, 8);
  std::cout << "DONE!" << std::endl;

  std::cout << "Drawing random samples ... " << std::flush;
  volume A = R, B = A;
  std::cout << "DONE!" << std::endl;
  cvolume C = fft(A, 3), D = C;
  volume E = ifft(C, 3);

  std::cout << "Read difference: " << tbblas::dot(A - B, A - B) << std::endl;
  std::cout << "Complex difference: " << dot(abs(C - D), abs(C - D)) << std::endl;
  std::cout << "Difference: " << dot(A - E, A - E) << std::endl;

  A.resize(tbblas::seq<4>(0));
  A.resize(tbblas::seq(0,0,0,0));
  C.resize(tbblas::seq(0,0,0,0), tbblas::seq(0,0,0,0));

  return;

  cudaThreadSynchronize();
  timer.restart();

  for (int i = 0; i < reps; ++i) {
    C = fft(A, 3);
  }
  cudaThreadSynchronize();
  std::cout << "Without plan cache: " << (t1 = timer.elapsed()) << "s." << std::endl;

  timer.restart();
  plan_t plan;
  for (int i = 0; i < reps; ++i) {
    C = fft(A, 3, plan);
  }
  cudaThreadSynchronize();
  std::cout << "With plan cache: " << (t2 = timer.elapsed()) << "s." << std::endl;
  std::cout << "Speedup: " << t1 / t2 << "x" << std::endl;
}
