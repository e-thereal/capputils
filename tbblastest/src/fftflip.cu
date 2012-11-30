/*
 * fftflip.cu
 *
 *  Created on: Nov 27, 2012
 *      Author: tombr
 */

#include "tests.h"

//#define TBBLAS_ALLOC_WARNING_ENABLED

#include <tbblas/tensor.hpp>
#include <tbblas/io.hpp>
#include <tbblas/gaussian.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/math.hpp>
#include <tbblas/random.hpp>

void fftflip() {
  using namespace tbblas;

  typedef double value_t;

  for (int i = 2; i <= 1024; i *= 2) {
    for (int j = 2; j <= 1024; j *= 2) {
      random_tensor<value_t, 3, true, uniform<value_t> > randu(i, j, 1);
      tensor<value_t, 3, true> v = randu;
      tensor<complex<value_t>, 3, true> cv;
      value_t check = sum(v);
      cv = fft(v);
      if (sum(v) != check) {
        std::cout << "v error at " << i << ", " << j << std::endl;
      }
      check = sum(abs(cv));
      v = ifft(cv);
      if (sum(abs(cv)) != check) {
        std::cout << "cv error at " << i << ", " << j << std::endl;
      }
    }
  }
}
