/*
 * mult_sum.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_MULT_SUM_HPP_
#define TBBLAS_MULT_SUM_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/tensor.hpp>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T>
__global__ void tbblas_conj_mult_sum(complex<T>* input, complex<T>* filters, complex<T>* output, size_t layerVoxelCount, size_t channelCount, size_t filterCount) {
  // threadidx.x + blockIdx.x * blockDim.x goes over pixels of the image
  // TODO: blockIdx.y goes over 16 filters
  // TODO: each thread loops over all channels and calculates a batch of 16 filters

  const int inPixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layerVoxelCount * channelCount;
  const int iFilter = blockIdx.y;

  if (inPixel >= layerVoxelCount)
    return;

  complex<T> sum = 0;
  for (int iChannel = 0; iChannel < channelCount; ++iChannel) {
    const complex<T>& f = filters[inPixel + iChannel * layerVoxelCount + iFilter * count];
    sum += complex<T>(f.real, -f.img) * input[inPixel + iChannel * layerVoxelCount];
  }

  output[inPixel + iFilter * layerVoxelCount] = sum;
}

template<class Tensor>
struct conj_mult_sum_operation {
  typedef Tensor tensor_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;

  conj_mult_sum_operation(Tensor& tensor, Tensor& filters)
   : _tensor(tensor), _filters(filters), _size(tensor.size()), _fullsize(tensor.fullsize())
  {
    channelCount = tensor.size()[3];
    filterCount = filters.size()[3] / channelCount;
    _size[3] = filterCount;
    _fullsize[3] = filterCount;
  }

  void apply(tensor_t& output) const {

    // TODO: Process batches of filters

    const int threadCount = 1024;
    const int channelCount = _tensor.size()[3];
    const int layerVoxelCount = _tensor.count() / channelCount;
    const int filterCount = _filters.size()[3] / channelCount;

    dim3 blockSize(threadCount);
    dim3 gridSize((layerVoxelCount + threadCount - 1) / threadCount, filterCount);
    tbblas_conj_mult_sum<<<gridSize, blockSize>>>(_tensor.data().data().get(), _filters.data().data().get(), output.data().data().get(),
        layerVoxelCount, channelCount, filterCount);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Tensor& _tensor;
  Tensor& _filters;
  int channelCount, filterCount;
  dim_t _size, _fullsize;
};

template<class T>
struct is_operation<conj_mult_sum_operation<T> > {
  static const bool value = true;
};

template<class T>
conj_mult_sum_operation<tensor<tbblas::complex<T>, 4, true> >
conj_mult_sum(tensor<tbblas::complex<T>, 4, true>& image, tensor<tbblas::complex<T>, 4, true>& filters) {
  typedef tensor<tbblas::complex<T>, 4, true> Tensor;
  assert(filters.size()[3] % image.size()[3] == 0);
  return conj_mult_sum_operation<Tensor>(image, filters);
}

}

#endif /* TBBLAS_MULT_SUM_HPP_ */
