/*
 * trainertests.cu
 *
 *  Created on: Oct 17, 2013
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/tensor.hpp>
#include <tbblas/random.hpp>
#include <tbblas/math.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/io.hpp>
#include <tbblas/conv.hpp>
#include <tbblas/filter.hpp>
#include <tbblas/filter2.hpp>
#include <tbblas/dot.hpp>

#include <algorithm>

#include <convnet/nvmatrix.cuh>
#include <convnet/cudaconv2.cuh>

using namespace tbblas;

typedef tbblas::tensor<float, 4, true> tensor_t;
typedef tbblas::tensor<complex<float>, 4, true> ctensor_t;
typedef tbblas::random_tensor<float, 4, true, normal<float> > randn_t;

typedef tbblas::tensor<float, 2, true> matrix_t;
typedef tbblas::tensor<float, 3, true> volume_t;

typedef fft_plan<4> plan_t;

//#define TEST_CORRECTNESS

void trainertests(int filterCount, int channelCount, int reps, int convnetReps) {

#ifdef CONVNET
  const int imageCount = 128, stride = 4;
  const int imgSize = 128 * stride, filterSize = 9 * stride + 1;

  randn_t v_rand(imgSize,imgSize,1,channelCount), k_rand(filterSize,filterSize,1,channelCount);
  tensor_t A = floor(10 * v_rand), K = floor(10 * k_rand), B1, B2, C;

  tensor_t flipped = flip(K);

  for (int i = 0; i < reps; ++i) {
    B1 = filter3d(A, K, optimized());
    B2 = sum(B1, 3);
  }
  C.resize(B2.size(), B2.fullsize());

#ifdef TEST_CORRECTNESS
  NVMatrix a1(imageCount, A.count()), a2(A.count(), imageCount),
           k1(filterCount,K.count()), k2(K.count(),filterCount),
           b1, b2; // b(numFilter x output size, numImages)
#else
  NVMatrix a1(A.count(), imageCount), &a2 = a1,
           k1(K.count(), filterCount), &k2 = k1,
           b1; // b(numFilter x output size, numImages)
#endif

#ifdef TEST_CORRECTNESS
  for (int i = 0; i < imageCount; ++i)
    thrust::copy(A.begin(), A.end(), thrust::device_pointer_cast(a1.getDevData()) + A.count() * i);
  for (int i = 0; i < filterCount; ++i)
    thrust::copy(K.begin(), K.end(), thrust::device_pointer_cast(k1.getDevData()) + K.count() * i);
  a1.transpose(a2);
  k1.transpose(k2);
#endif

  /**
   * void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                 int numImgColors, int numGroups);

     void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
                 int numImgColors, int numGroups);

     void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                 int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                 int moduleStride, int numImgColors, int numGroups, int partialSum);
   */

  for (int i = 0; i < convnetReps; ++i) {
    convFilterActs(a2, k2, b1,
        A.size()[1], A.size()[1] / stride, A.size()[0] / stride, -(K.size()[1] - 1) / 2, stride,
        A.size()[3], 1);
    if (i == 0)
      std::cout << b1.getNumRows() << " x " << b1.getNumCols() << std::endl;

//    convImgActs(b1, k2, a2,
//        A.size()[1], A.size()[0], A.size()[1], -(K.size()[1]-1) / 2, stride,
//        A.size()[3], 1);
//    if (i == 0)
//      std::cout << a2.getNumRows() << " x " << a2.getNumCols() << std::endl;
//
//    convWeightActs(a2, b1, k2,
//        A.size()[1], A.size()[1], A.size()[0], K.size()[1], -(K.size()[1]-1) / 2,
//        stride, A.size()[3], 1, 0);
//    if (i == 0)
//      std::cout << k2.getNumRows() << " x " << k2.getNumCols() << std::endl;
  }

  std::cout << b1.getNumCols() << " x " << b1.getNumRows() << std::endl;

#ifdef TEST_CORRECTNESS
  if (convnetReps)
    b1.transpose(b2);
  for (int i = 0; i < filterCount; ++i) {
    thrust::copy(thrust::device_pointer_cast(b2.getDevData()) + i * C.count(),
        thrust::device_pointer_cast(b2.getDevData()) + (i + 1) * C.count(), C.begin());

    std::cout << "error: " << sqrt(dot(B2 - C, B2 - C) / B2.count()) << std::endl;
  }
#endif

  return;
#endif

  // Define variables
  tensor_t h(64, 64, 64, 32), h2(64, 64, 64, 32), v(64, 64, 64, 32);

  plan_t v_plan;
  ctensor_t ch_full = fft(v, 3, v_plan), cF = fft(v, 3, v_plan), cv = fft(v, 3, v_plan);

  // Waste some memory
  tensor_t f[filterCount];
  for (int i = 0; i < filterCount; ++i)
    f[i].resize(seq(64, 64, 64, channelCount), seq(64, 64, 64, channelCount));

  // Perform operations
  for (int i = 0; i < reps; ++i) {
//    h = max(0.0, h2 + sqrt(sigm(h2)) * h_noise);
    ch_full = conj(cF) * cv;
    ch_full = cF + cv;
    v = v + v;
  }
}
