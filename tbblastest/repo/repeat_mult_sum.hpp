/*
 * repeat_mult_sum.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_REPEAT_MULT_SUM_HPP_
#define TBBLAS_REPEAT_MULT_SUM_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/tensor.hpp>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T>
__global__ void tbblas_repeat_mult_sum(complex<T>* hiddens, complex<T>* filters, complex<T>* visibles, size_t layerVoxelCount, size_t channelCount, size_t filterCount) {
  // threadidx.x + blockIdx.x * blockDim.x goes over pixels of the image
  // blockIdx.y goes over channels
  // each thread loops over all filters

  const int inPixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layerVoxelCount * channelCount;
  const int iChannel = blockIdx.y;

  if (inPixel >= layerVoxelCount)
    return;

  complex<T> sum = 0;
  for (int iFilter = 0; iFilter < filterCount; ++iFilter) {
    const complex<T>& f = filters[inPixel + iChannel * layerVoxelCount + iFilter * count];
    sum += f * hiddens[inPixel + iFilter * layerVoxelCount];
  }

  visibles[inPixel + iChannel * layerVoxelCount] = sum;
}

template<class T>
__global__ void tbblas_repeat_mult_sum_inc(complex<T>* hiddens, complex<T>* filters, complex<T>* visibles, size_t layerVoxelCount, size_t channelCount, size_t filterCount) {
  // threadidx.x + blockIdx.x * blockDim.x goes over pixels of the image
  // blockIdx.y goes over channels
  // each thread loops over all filters

  const int inPixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layerVoxelCount * channelCount;
  const int iChannel = blockIdx.y;

  if (inPixel >= layerVoxelCount)
    return;

  complex<T> sum = 0;
  for (int iFilter = 0; iFilter < filterCount; ++iFilter) {
    const complex<T>& f = filters[inPixel + iChannel * layerVoxelCount + iFilter * count];
    sum += f * hiddens[inPixel + iFilter * layerVoxelCount];
  }

  visibles[inPixel + iChannel * layerVoxelCount] += sum;
}

template<class Tensor>
struct repeat_mult_sum_operation {
  typedef Tensor tensor_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;

  repeat_mult_sum_operation(Tensor& hiddens, Tensor& filters)
   : _hiddens(hiddens), _filters(filters), _size(hiddens.size()), _fullsize(hiddens.fullsize())
  {
    _size[3] = filters.size()[3] / hiddens.size()[3];
    _fullsize[3] = filters.size()[3] / hiddens.size()[3];
  }

  void apply(tensor_t& output) const {

    const int threadCount = 1024;
    const int filterCount = _hiddens.size()[3];
    const int channelCount = _filters.size()[3] / filterCount;
    const int layerVoxelCount = _hiddens.count() / filterCount;

    dim3 blockSize(threadCount);
    dim3 gridSize((layerVoxelCount + threadCount - 1) / threadCount, channelCount);
    tbblas_repeat_mult_sum<<<gridSize, blockSize>>>(_hiddens.data().data().get(), _filters.data().data().get(), output.data().data().get(),
        layerVoxelCount, channelCount, filterCount);
  }

  void apply_inc(tensor_t& output) const {

    const int threadCount = 1024;
    const int filterCount = _hiddens.size()[3];
    const int channelCount = _filters.size()[3] / filterCount;
    const int layerVoxelCount = _hiddens.count() / filterCount;

    dim3 blockSize(threadCount);
    dim3 gridSize((layerVoxelCount + threadCount - 1) / threadCount, channelCount);
    tbblas_repeat_mult_sum_inc<<<gridSize, blockSize>>>(_hiddens.data().data().get(), _filters.data().data().get(), output.data().data().get(),
        layerVoxelCount, channelCount, filterCount);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Tensor& _hiddens;
  Tensor& _filters;
  dim_t _size, _fullsize;
};

template<class T>
struct is_operation<repeat_mult_sum_operation<T> > {
  static const bool value = true;
};

template<class T>
struct is_inc_operation<repeat_mult_sum_operation<T> > {
  static const bool value = true;
};

template<class T>
repeat_mult_sum_operation<tensor<tbblas::complex<T>, 4, true> >
repeat_mult_sum(tensor<tbblas::complex<T>, 4, true>& hiddens, tensor<tbblas::complex<T>, 4, true>& filters) {
  typedef tensor<tbblas::complex<T>, 4, true> Tensor;
  assert(filters.size()[3] % hiddens.size()[3] == 0);
  return repeat_mult_sum_operation<Tensor>(hiddens, filters);
}

}

#endif /* TBBLAS_REPEAT_MULT_SUM_HPP_ */
