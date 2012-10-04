/*
 * convtest2.cu
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#include <tbblas/tensor.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/io.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/flip.hpp>
#include <tbblas/random.hpp>
#include <tbblas/dot.hpp>

#include <tbblas/binary_expression.hpp>
#include <tbblas/max.hpp>

#include <thrust/sequence.h>
#include <thrust/copy.h>

typedef tbblas::tensor<float, 2, true> matrix;
typedef tbblas::random_tensor<float, 2, true, tbblas::uniform<float> > urand;
typedef tbblas::tensor<tbblas::complex<float>, 2, true> cmatrix;
typedef typename matrix::dim_t dim_t;

void convtest2() {
  using namespace tbblas;

  matrix image = urand(5,5);
//  thrust::sequence(image.begin(), image.end());
//  image[seq(2,2)] = 3;
  std::cout << "image = " << image << std::endl;

  matrix pattern = image[seq(1,1), seq(3,3)];
  std::cout << "pattern = " << pattern << std::endl;

  matrix paddedKernel = zeros<float>(image.size());

  dim_t topleft = (image.size() - pattern.size() + 1u) / 2u;
  std::cout << "Top left = " << topleft << std::endl;
  paddedKernel[topleft, pattern.size()] = flip(pattern);
  std::cout << "pattern (padded) = " << paddedKernel << std::endl;

  matrix shiftedKernel = ifftshift(paddedKernel);
  std::cout << "pattern (shifted) = " << shiftedKernel << std::endl;

  cmatrix cimage = fft(image), ckernel = fft(shiftedKernel);
  cmatrix cfiltered = cimage * ckernel;
  matrix filtered = ifft(cfiltered);
  std::cout << "filtered = " << filtered << std::endl;
  std::cout << "maximum = " << max(filtered) << std::endl;

  filtered = filtered == ones<float>(image.size()) * max(filtered);
  std::cout << "image = " << filtered << std::endl;
}
