/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/io.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/real.hpp>
#include <tbblas/img.hpp>
#include <tbblas/fft.hpp>

#include <iostream>

#include <boost/timer.hpp>


void sumtest() {
  using namespace tbblas;
  
  {
    typedef tbblas::tensor<tbblas::complex<double>, 3, true> tensor_t;
    typedef tbblas::random_tensor<double, 3, true, tbblas::uniform<double> > randu_t;
    randu_t randu(128, 128, 128);

    tensor_t A(128, 128, 128), sum1, sum2;
    real(A) = floor(1000 * randu);
    img(A) = floor(1000 * randu);
    cudaStreamSynchronize(0);

    boost::timer timer;
    for (size_t i = 0; i < 1000; ++i)
      sum1 = sum(proxy<tensor_t>(A), 2);
    cudaStreamSynchronize(0);
    std::cout << "Time1: " << timer.elapsed() << std::endl;

    timer.restart();
    for (size_t i = 0; i < 1000; ++i)
      sum2 = sum(A, 2);
    cudaStreamSynchronize(0);
    std::cout << "Time2: " << timer.elapsed() << std::endl;

    std::cout << "Error: " << dot(sum1 - sum2, sum1 - sum2) << std::endl;
  }

  {
    typedef tbblas::tensor<tbblas::complex<double>, 4, true> tensor_t;
    typedef tbblas::random_tensor<double, 4, true, tbblas::uniform<double> > randu_t;
    randu_t randu(64, 64, 64, 16);

    tensor_t A(64, 64, 64, 16), sum1, sum2;
    real(A) = floor(1000 * randu);
    img(A) = floor(1000 * randu);
    cudaStreamSynchronize(0);

    boost::timer timer;
    for (size_t i = 0; i < 100; ++i)
      sum1 = sum(proxy<tensor_t>(A), 3);
    cudaStreamSynchronize(0);
    std::cout << "Time1: " << timer.elapsed() << std::endl;
  
    timer.restart();
    for (size_t i = 0; i < 100; ++i)
      sum2 = sum(A, 3);
    cudaStreamSynchronize(0);
    std::cout << "Time2: " << timer.elapsed() << std::endl;

    std::cout << "Error: " << dot(sum1 - sum2, sum1 - sum2) << std::endl;
  }
}
