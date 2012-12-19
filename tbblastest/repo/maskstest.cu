/*
 * maskstest.cu
 *
 *  Created on: Nov 30, 2012
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/gaussian.hpp>
#include <tbblas/mask.hpp>
#include <tbblas/io.hpp>
#include <tbblas/math.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/random.hpp>

#include <boost/timer.hpp>

#include <sstream>

using namespace tbblas;

typedef tensor<double, 2, true> matrix;

template<class T>
struct nrelu_mean_operation {
  typedef T value_t;

  __host__ __device__
  T operator()(const T& value) const {
    T var = T(1) / (T(1) + ::exp(-value));
    return ::sqrt(0.5 * var / M_PI) * ::exp(-0.5 * value * value / var) + 0.5 * value * ::erfc(-value / sqrt(2.0 * var));
  }
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  unary_expression<Expression, nrelu_mean_operation<typename Expression::value_t> >
>::type
nrelu_mean(const Expression& expr) {
  return unary_expression<Expression, nrelu_mean_operation<typename Expression::value_t> >(expr,
      nrelu_mean_operation<typename Expression::value_t>());
}

void maskstest() {
  tbblas_print(gaussian<double>(seq(5, 5), 1.0));
  tbblas_print(mask<double>(seq(6, 6), seq(2, 2)));
  tbblas_print(mask<double>(seq(6, 6), seq(3, 3)));

  matrix A = ones<double>(3,3);
  matrix B(3,3);
  B = 0, 3, -1,
      6, 3, 0,
      -5, 3, 8;

  typedef typename thrust::iterator_traits<typename thrust::device_vector<double>::iterator>::iterator_category tag;


  tbblas_print(A + 2.0 > 1.0);
  tbblas_print(min(A, B));
  tbblas_print(min(B, 2));

  matrix mu(1,9), e1, e2;
  mu = -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2;

  random_tensor<double, 2, true, normal<double> > h_noise(mu.size()), big_noise(256, 256);

  e1 = zeros<double>(mu.size());
  for (int i = 0; i < 1000; ++i) {
    e1 = e1 + max(0.0, mu + sqrt(sigm(mu)) * h_noise) / 1000.0;
  }
  e2 = sqrt(0.5 * sigm(mu) / M_PI) * exp(-0.5 * mu * mu / sigm(mu)) + 0.5 * mu * erfc(-mu / sqrt(2 * sigm(mu)));

  tbblas_print(mu);
  tbblas_print(e1);
  tbblas_print(e2);
  tbblas_print(nrelu_mean(mu));

  cudaDeviceSynchronize();

  tensor<double, 3, true> x = random_tensor<double, 3, true, normal<double> >(256, 256, 64), y;

  boost::timer timer;
  for (int i = 0; i < 1000; ++i)
    y = sqrt(0.5 * sigm(x) / M_PI) * exp(-0.5 * x * x / sigm(x)) + 0.5 * x * erfc(-x / sqrt(2 * sigm(x)));
  cudaDeviceSynchronize();
  std::cout << timer.elapsed() << std::endl;
  timer.restart();
  for (int i = 0; i < 1000; ++i)
    y = nrelu_mean(x);
  cudaDeviceSynchronize();
  std::cout << timer.elapsed() << std::endl;
}
