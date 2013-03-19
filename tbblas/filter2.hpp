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

template<class T>
__global__ void filter3d_kernel(const T* input, dim3 size, const T* kernel, dim3 kernelSize, T* output, unsigned batches) {
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x >= size.x || y >= size.y || z >= size.z)
    return;

  const int count = size.x * size.y * size.z;

  for (int offset = 0; offset < batches * count; offset += count) {
    T sum = 0;
    for (int kz = 0, i = 0; kz < kernelSize.z; ++kz) {
      for (int ky = 0; ky < kernelSize.y; ++ky) {
        for (int kx = 0; kx < kernelSize.x; ++kx, ++i) {
          // calculate input image index
          const int ix = x + kx - kernelSize.x / 2;
          const int iy = y + ky - kernelSize.y / 2;
          const int iz = z + kz - kernelSize.z / 2;

          if (ix >= 0 && iy >= 0 && iz >= 0 && ix < size.x && iy < size.y && iz < size.z)
            sum += input[offset + ix + size.x * (iy + size.y * iz)] * kernel[offset + i];
        }
      }
    }

    output[offset + x + size.x * (y + size.y * z)] = sum;
  }
}

template<class Tensor>
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
    blockDim[0] = 32;
    blockDim[1] = 32;
    blockDim[2] = 1;
    filter3d_kernel<<<(_size + blockDim - 1) / blockDim, blockDim>>>(
        input.data().data().get(), input.size(),
        kernel.data().data().get(), kernel.size(),
        output.data().data().get(), _batches);
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

template<class T1>
struct is_operation<filter3d_operation<T1> > {
  static const bool value = true;
};

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if_c<Tensor::dimCount == 3,
      filter3d_operation<Tensor>
    >::type
  >::type
>::type
filter3d(const Tensor& input, const Tensor& kernel) {
  return filter3d_operation<Tensor>(input, kernel, 1);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<Tensor::cuda_enabled == true,
    typename boost::enable_if_c<Tensor::dimCount == 4,
      filter3d_operation<Tensor>
    >::type
  >::type
>::type
filter3d(const Tensor& input, const Tensor& kernel) {
  assert(input.size()[3] == kernel.size()[3]);
  return filter3d_operation<Tensor>(input, kernel, input.size()[3]);
}

}

#endif /* TBBLAS_FILTER2_HPP_ */
