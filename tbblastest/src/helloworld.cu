/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <thrust/copy.h>

#include <tbblas/io.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/random.hpp>
#include <iostream>

#include <boost/timer.hpp>

#include <thrust/for_each.h>

typedef tbblas::tensor<float, 2, true> matrix_t;
typedef tbblas::tensor<float, 1, true> vector_t;

typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > randu_t;
typedef tbblas::random_tensor<float, 2, true, tbblas::normal<float> > randn_t;

void helloworld() {
  using namespace tbblas;
  using namespace thrust::placeholders;
  
  //const float values1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  //const float values2[] = {2, 3, 5, 1, 3, 2, 6, 7, 3, 1, 23, 2};
  
  randu_t randu(3, 4);
  randn_t randn(3, 4);

  matrix_t A = 4 * randu;
  matrix_t B = 0.5 * randn + 2;
  
  //thrust::copy(values1, values1 + A.count(), A.begin());
  //thrust::copy(values2, values2 + B.count(), B.begin());
  
  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  matrix_t C = ((2.f * A - B) * B) / 2.f;
  std::cout << "A + B = " << C << std::endl;

  matrix_t D = randu_t(3, 3);
  std::cout << "D = " << D << std::endl;

  matrix_t E = zeros<float>(5, 5);

  //subrange(E, seq(1u, 1u), seq(3u, 3u)) = D;

  //std::cout << "E = " << E << std::endl;





  //thrust::transform(A.begin(), A.end(), B.begin(), D.begin(), ((2.f * _1 - _2) * _2) / 2.f);
//  thrust::for_each(
//      thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), D.begin())),
//      thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), D.end())),
//      functor()
//  );
//  std::cout << "A + B = " << D << std::endl;
  
#if 0
  matrix_t A1(5000, 2000), A2(5000, 2000), A3(5000, 2000);
  matrix2_t B1(5000, 2000), B2(5000, 2000), B3(5000, 2000);
  cudaThreadSynchronize();
  
  std::cout << "New interface:" << std::endl;
  boost::timer timer;
  for (unsigned i = 0; i < 500; ++i)
    //A3 = ((2.f * A1 - A2) + A2) / 2.f;
    A3 = A1 * A2;
  cudaThreadSynchronize();
  std::cout << "tbblas time: " << timer.elapsed() << "s" << std::endl;
  
  timer.restart();
  for (unsigned i = 0; i < 500; ++i) {
    //thrust::transform(A1.begin(), A1.end(), A2.begin(), A3.begin(), ((2.f * _1 - _2) + _2) / 2.f);
  
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(A1.begin(), A2.begin(), A3.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(A1.end(), A2.end(), A3.end())),
        functor()
    );
  }
  
  cudaThreadSynchronize();
  std::cout << "thrust time: " << timer.elapsed() << "s" << std::endl;
  

  std::cout << "\nOld interface:" << std::endl;
  timer.restart();
  for (unsigned i = 0; i < 500; ++i)
    B3 = tbblas::copy((B3 = (B3 = 2.f * B1 - B2) + B2) / 2.f);
  cudaThreadSynchronize();
  std::cout << "tbblas time: " << timer.elapsed() << "s" << std::endl;
  
  timer.restart();
  for (unsigned i = 0; i < 500; ++i)
    thrust::transform(B1.cbegin(), B1.cend(), B2.cbegin(), B3.begin(), ((2.f * _1 - _2) + _2) / 2.f);
  cudaThreadSynchronize();
  std::cout << "thrust time: " << timer.elapsed() << "s" << std::endl;
#endif
}
