/*
 * poolingtest_gpu.cu
 *
 *  Created on: Feb 16, 2015
 *      Author: tombr
 */


#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>
#include <tbblas/io.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/util.hpp>

#include <tbblas/deeplearn/max_pooling.hpp>
#include <tbblas/deeplearn/avg_pooling.hpp>

#include <boost/timer.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>

using namespace tbblas;
using namespace tbblas::deeplearn;

typedef tensor<float, 2, true> matrix_t;
typedef tensor<uint8_t, 2, true> imatrix_t;

typedef tensor<float, 4, true> tensor_t;
typedef tensor<complex<float>, 4, true> ctensor_t;
typedef tensor<uint8_t, 4, true> stensor_t;

#define TWO_D_POOLING

template<class T, class Enable = void>
class TestClass {
public:
  static void print() {
    std::cout << "Base version." << std::endl;
  }
};

template<class T>
class TestClass<T, typename boost::enable_if<boost::is_integral<T> >::type> {
public:
  static void print() {
    std::cout << "Integral" << std::endl;
  }
};

template<class T, class Enable = typename boost::enable_if<boost::is_integral<T> >::type>
class TestClass2
{

};

void poolingtest() {

  TestClass<std::string>::print();
  TestClass<int>::print();

  TestClass2<int> c;

  return;

#ifdef TWO_D_POOLING
  random_tensor2<float, 2, true, uniform<float> > randn(6, 6);

  matrix_t A = floor(20 * randn());
  tbblas_print(A);

  imatrix_t S = get_max_pooling_switches(A, seq(2,2));
  matrix_t B = max_pooling(A, S, seq(2,2));
  tbblas_print(B);
  tbblas_print((matrix_t)S);

  matrix_t C = unpooling(B, S, seq(2,2));
  tbblas_print(C);

  B = avg_pooling(A, seq(2,2));
  tbblas_print(B);
  C = unpooling(B, seq(2,2));
  tbblas_print(C);
#else

  boost::timer timer;
  const size_t reps = 4;

  DerivedTest<int> test(2);
  test.print();

//  random_tensor2<float, 4, true, uniform<float> > randn(84, 108, 53, 8);

  fft_plan<4> plan;

  for (int height = 105; height <= 109; ++height) {
    if (height == 105) {
      std::cout << "                ";
      for (int width = 81; width <= 85; ++width)
        std::cout << width << "   ";
      std::cout << std::endl;
    }
    std::cout << "Height = " << height << ":";

    for (int width = 81; width <= 85; ++width) {
      random_tensor2<float, 4, true, uniform<float> > randn(width, height, 53, 8);

      tensor_t A = floor(20 * randn());
      ctensor_t cA = fft(A, 3, plan);

      synchronize();
      timer.restart();

      for (size_t i = 0; i < reps; ++i) {
        cA = fft(A, 3, plan);
      }

      synchronize();
      std::cout << " " << timer.elapsed();
      timer.restart();
    }
    std::cout << std::endl;
  }

  {
    random_tensor2<float, 4, true, uniform<float> > randn(96, 128, 53, 8);

    tensor_t A = floor(20 * randn());
    ctensor_t cA = fft(A, 3, plan);

    synchronize();
    timer.restart();

    for (size_t i = 0; i < reps; ++i) {
      cA = fft(A, 3, plan);
    }

    synchronize();
    std::cout << "96x128" << timer.elapsed() << std::endl;
    timer.restart();
  }

  {
    random_tensor2<float, 4, true, uniform<float> > randn(96, 128, 53, 8);

    tensor_t A = floor(20 * randn());
    ctensor_t cA = fft(A, 3, plan);

    synchronize();
    timer.restart();

    for (size_t i = 0; i < reps; ++i) {
      cA = fft(A, 3, plan);
    }

    synchronize();
    std::cout << "96x128" << timer.elapsed() << std::endl;
    timer.restart();
  }
#endif
}
