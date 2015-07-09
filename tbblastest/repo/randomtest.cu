/*
 * randomtest.cu
 *
 *  Created on: Nov 3, 2014
 *      Author: tombr
 */

#include "tests.h"

#define CURAND_CALL(x) do { curandStatus_t status; if((status = x)!=CURAND_STATUS_SUCCESS) { \
std::cout << "Error: " << status << "in file "__FILE__" at line" << __LINE__ << std::endl;\
exit(EXIT_FAILURE);}} while(0)

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>

#include <cstdio>
#include <cstdlib>

void randomtest() {
  using namespace tbblas;

  typedef tensor<float, 4, true> tensor_t;
  typedef random_tensor<float, 4, true, uniform<float> > rand_t;
  typedef random_tensor2<float, 4, true, uniform<float> > rand2_t;

  tensor_t T(160, 208, 64, 3);
//  rand_t R(T.size());
  rand2_t R2(T.size());

  for (int i = 0; i < 1000; ++i) {
//    T = R;
    T = R2();
    T = R2(false);
  }
//  getchar();

  random_tensor2<double, 2, true, uniform<double> > uniform(3,3);
  tbblas_print(uniform(false));
  tbblas_print(uniform());
  tbblas_print(uniform());
  tbblas_print(uniform(false));

  random_tensor2<double, 2, true, normal<double> > norm(3,3);
  tbblas_print(norm(false));
  tbblas_print(norm());
  tbblas_print(norm());
  tbblas_print(norm(false));
}
