/*
 * filter2.hpp
 *
 *  Created on: Mar 12, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_FILTER2_HPP_
#define TBBLAS_FILTER2_HPP_

#include <tbblas/tensor.hpp>

#include <tbblas/shift.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/zeros.hpp>

namespace tbblas {

struct optimized {};
struct naive {};

const int granularity = 8;

template<class T, class T2>
__global__ void filter3d_kernel(T2, const T* input, dim3 size, const T* kernel, dim3 kernelSize,
    T* output, dim3 outSize, unsigned batches)
{
  //dim_t outputTopleft = input.size() / 2 - output.size() / 2;

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  const int imageCount = size.x * size.y * size.z;
  const int kernelCount = kernelSize.x * kernelSize.y * kernelSize.z;

  for (int iBatch = 0; iBatch < batches; ++iBatch) {
    T sum = 0;
    for (int kz = 0, i = 0; kz < kernelSize.z; ++kz) {
      for (int ky = 0; ky < kernelSize.y; ++ky) {
        for (int kx = 0; kx < kernelSize.x; ++kx, ++i) {
          // calculate input image index
          const int ix = x + kx - kernelSize.x / 2;
          const int iy = y + ky - kernelSize.y / 2;
          const int iz = z + kz - kernelSize.z / 2;

          if (ix >= 0 && iy >= 0 && iz >= 0 && ix < size.x && iy < size.y && iz < size.z)
            sum += input[iBatch * imageCount + ix + size.x * (iy + size.y * iz)] * kernel[iBatch * kernelCount + i];
        }
      }
    }

    output[iBatch * imageCount + x + size.x * (y + size.y * z)] = sum;
  }
}

template<class T>
__global__ void filter3d_kernel(optimized, const T* input, dim3 size, const T* kernel, dim3 kernelSize,
    T* output, dim3 outSize, unsigned batches)
{
  //dim_t outputTopleft = input.size() / 2 - output.size() / 2;

  const int xOffset = blockIdx.x * blockDim.x;
  const int yOffset = blockIdx.y * blockDim.y;
  const int zOffset = blockIdx.z * blockDim.z;
  const int x = threadIdx.x + xOffset;
  const int y = threadIdx.y + yOffset;
  const int z = threadIdx.z + zOffset;

  const int imageCount = size.x * size.y * size.z;
  const int kernelCount = kernelSize.x * kernelSize.y * kernelSize.z;

  extern __shared__ T sharedBuffer[];

  // Read image data to shared memory
  const int haloSizeX = kernelSize.x / 2;
  const int haloSizeY = kernelSize.y / 2;
  const int haloSizeZ = kernelSize.z / 2;
  const int paddedHaloSizeX = ((haloSizeX + granularity - 1) / granularity) * granularity;
  const int readingWidth  = 2 * paddedHaloSizeX + blockDim.x;
  const int readingHeight = 2 * haloSizeY + blockDim.y;
  const int readingDepth  = 2 * haloSizeZ + blockDim.z;

  T* imageBuffer = &sharedBuffer[0];
  T* kernelBuffer = &sharedBuffer[readingWidth * readingHeight * readingDepth];

  for (int iBatch = 0; iBatch < batches; ++iBatch) {

    for (int bz = threadIdx.z; bz < readingDepth; bz += blockDim.z) {
      for (int by = threadIdx.y; by < readingHeight; by += blockDim.y) {
        for (int bx = threadIdx.x; bx < readingWidth; bx += blockDim.x) {
          const int ix = bx + xOffset - paddedHaloSizeX;
          const int iy = by + yOffset - haloSizeY;
          const int iz = bz + zOffset - haloSizeZ;

          if (ix >= 0 && ix < size.x && iy >= 0 && iy < size.y && iz >= 0 && iz < size.z)
            imageBuffer[bx + readingWidth * (by + readingHeight * bz)] = input[iBatch * imageCount + ix + size.x * (iy + size.y * iz)];
          else
            imageBuffer[bx + readingWidth * (by + readingHeight * bz)] = 0;
        }
      }
    }

    for (int kz = threadIdx.z; kz < kernelSize.z; kz += blockDim.z) {
      for (int ky = threadIdx.y; ky < kernelSize.y; ky += blockDim.y) {
        for (int kx = threadIdx.x; kx < kernelSize.x; kx += blockDim.x) {
          kernelBuffer[kx + kernelSize.x * (ky + kernelSize.y * kz)] = kernel[iBatch * kernelCount + kx + kernelSize.x * (ky + kernelSize.y * kz)];
        }
      }
    }

    __syncthreads();

    if (x < size.x && y < size.y && z < size.z) {

      T sum = 0;
      for (int kz = 0, i = 0; kz < kernelSize.z; ++kz) {
        for (int ky = 0; ky < kernelSize.y; ++ky) {
          for (int kx = 0; kx < kernelSize.x; ++kx, ++i) {
            const int bx = threadIdx.x + kx - haloSizeX + paddedHaloSizeX;
            const int by = threadIdx.y + ky;
            const int bz = threadIdx.z + kz;
            sum += imageBuffer[bx + readingWidth * (by + readingHeight * bz)] * kernelBuffer[i];
          }
        }
      }

      output[iBatch * imageCount + x + size.x * (y + size.y * z)] = sum;
    }

    __syncthreads();
  }
}

template<class Tensor, class T>
struct filter3d_operation
{
  typedef typename Tensor::dim_t dim_t;
  typedef typename Tensor::value_t value_t;
  static const unsigned dimCount = Tensor::dimCount;

  typedef Tensor tensor_t;

public:
  filter3d_operation(const Tensor& input, const Tensor& kernel, unsigned batches)
   : input(input), kernel(kernel), _size(input.size()), _batches(batches)
  {
    assert(max(input.size(), kernel.size()) == input.size());

//    for (unsigned i = dimension; i < dimCount; ++i)
//      assert(kernel.size()[i] == input.size()[i]);
  }

  void apply(tensor_t& output) const {
    dim_t blockDim(1);
    blockDim[0] = 16;
    blockDim[1] = 4;
    blockDim[2] = 4;

    assert(kernel.size()[0] % 2 == 1);
    assert(kernel.size()[1] % 2 == 1);
    assert(kernel.size()[2] % 2 == 1);

    const int haloSizeX = kernel.size()[0] / 2;
    const int haloSizeY = kernel.size()[1] / 2;
    const int haloSizeZ = kernel.size()[2] / 2;
    const int paddedHaloSizeX = (haloSizeX + granularity - 1) / granularity * granularity;
    const int readingWidth = 2 * paddedHaloSizeX + blockDim[0];
    const int readingHeight = 2 * haloSizeY + blockDim[1];
    const int readingDepth = 2 * haloSizeZ + blockDim[2];

    const int sharedMemorySize = (readingWidth * readingHeight * readingDepth + kernel.size()[0] * kernel.size()[1] * kernel.size()[2]) * sizeof(value_t);
//    tbblas_print(haloSizeX);
//    tbblas_print(paddedHaloSizeX);
//    tbblas_print(sharedMemorySize);

    filter3d_kernel<<<(_size + blockDim - 1) / blockDim, blockDim, sharedMemorySize>>>(
            T(),
            input.data().data().get(), input.size(),
            kernel.data().data().get(), kernel.size(),
            output.data().data().get(), output.size(),
            _batches
    );
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

private:
  const Tensor& input;
  const Tensor& kernel;
  dim_t _size;
  unsigned _batches;
};

template<class T1, class T2>
struct is_operation<filter3d_operation<T1, T2> > {
  static const bool value = true;
};

template<class Tensor, class T>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if_c<Tensor::dimCount == 3,
      filter3d_operation<Tensor, T>
    >::type
  >::type
>::type
filter3d(const Tensor& input, const Tensor& kernel, T) {
  return filter3d_operation<Tensor, T>(input, kernel, 1);
}

template<class Tensor, class T>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if_c<Tensor::dimCount == 4,
      filter3d_operation<Tensor, T>
    >::type
  >::type
>::type
filter3d(const Tensor& input, const Tensor& kernel, T) {
  assert(input.size()[3] == kernel.size()[3]);
  return filter3d_operation<Tensor, T>(input, kernel, input.size()[3]);
}

}

#endif /* TBBLAS_FILTER2_HPP_ */
