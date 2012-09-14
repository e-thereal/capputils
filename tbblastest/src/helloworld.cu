/*
 * tests.cu
 *
 *  Created on: Nov 4, 2011
 *      Author: tombr
 */
#include "tests.h"

#include <tbblas/tensor.hpp>
#include <thrust/copy.h>

#include <tbblas/io2.hpp>
#include <tbblas/plus2.hpp>
#include <tbblas/minus.hpp>
#include <tbblas/multiplies.hpp>
#include <tbblas/divides.hpp>
#include <iostream>

#include <boost/timer.hpp>

#include <tbblas/tensor_base.hpp>
#include <tbblas/plus.hpp>

#include <thrust/for_each.h>

typedef tbblas::tensor<float, 2, true> matrix_t;
typedef tbblas::tensor<float, 1, true> vector_t;

typedef tbblas::tensor_base<float, 2, true> matrix2_t;

template<bool b>
class Test {
public:
  static void print() {
    std::cout << "True" << std::endl;
  }
};

template<>
class Test<false> {
public:
  static void print() {
    std::cout << "False" << std::endl;
  }
};

template<class T1, class T2>
typename boost::enable_if_c<T1::dimCount == T2::dimCount, T1>::type
add_tensors(const T1& t1, const T2& t2) {
  return t1;
}

struct functor {
  template <class Tuple>
  __host__ __device__
  void operator()(const Tuple& t) const {
    //thrust::get<2>(t) = ((a * thrust::get<0>(t) - thrust::get<1>(t)) * thrust::get<1>(t)) / b;
    thrust::get<2>(t) = thrust::get<0>(t) * thrust::get<1>(t);
  }
  
  functor() : a(2.0f), b(2.0f) { }
  
private:
  const float a, b;
};

void helloworld() {
  using namespace tbblas;
  using namespace thrust::placeholders;
  
  const float values1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const float values2[] = {2, 3, 5, 1, 3, 2, 6, 7, 3, 1, 23, 2};
  
  matrix_t A(3, 4);
  matrix_t B(3, 4);
  
  thrust::copy(values1, values1 + A.count(), A.begin());
  thrust::copy(values2, values2 + B.count(), B.begin());
  
  std::cout << "A = " << A << std::endl;
  std::cout << "B = " << B << std::endl;
  matrix_t C = ((2.f * A - B) * B) / 2.f, D(3, 4);
  std::cout << "A + B = " << C << std::endl;
  //thrust::transform(A.begin(), A.end(), B.begin(), D.begin(), ((2.f * _1 - _2) * _2) / 2.f);
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), D.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end(), D.end())),
      functor()
  );
  std::cout << "A + B = " << D << std::endl;
  
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
  
#if 0
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
