/*
 * repeat_mult.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_REPEAT_MULT_HPP_
#define TBBLAS_REPEAT_MULT_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/tensor.hpp>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T>
__global__ void tbblas_conj_repeat_mult(complex<T>* visibles, complex<T>* hiddens, complex<T>* filters,
    size_t layerVoxelCount, size_t channelCount, size_t filterCount, T eps)
{
  // threadidx.x + blockIdx.x * blockDim.x goes over pixels of the image
  // TODO: blockIdx.y goes over 16 channels
  // TODO: each thread loops over all filters and calculates a batch of 16 channels

  const int inPixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layerVoxelCount * channelCount;
  const int inChannel = blockIdx.y;

  if (inPixel >= layerVoxelCount)
    return;

  complex<T> v = visibles[inPixel + inChannel * layerVoxelCount];

  for (int iFilter = 0; iFilter < filterCount; ++iFilter) {
    complex<T> h = hiddens[inPixel + iFilter * layerVoxelCount];
    filters[inPixel + inChannel * layerVoxelCount + iFilter * count] = complex<T>(eps * h.real, eps * -h.img) * v;
  }
}

template<class T>
__global__ void tbblas_conj_repeat_mult_inc(complex<T>* visibles, complex<T>* hiddens, complex<T>* filters,
    size_t layerVoxelCount, size_t channelCount, size_t filterCount, T eps)
{
  // threadidx.x + blockIdx.x * blockDim.x goes over pixels of the image
  // TODO: blockIdx.y goes over 16 channels
  // TODO: each thread loops over all filters and calculates a batch of 16 channels

  const int inPixel = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layerVoxelCount * channelCount;
  const int inChannel = blockIdx.y;

  if (inPixel >= layerVoxelCount)
    return;

  complex<T> v = visibles[inPixel + inChannel * layerVoxelCount];

  for (int iFilter = 0; iFilter < filterCount; ++iFilter) {
    complex<T> h = hiddens[inPixel + iFilter * layerVoxelCount];
    filters[inPixel + inChannel * layerVoxelCount + iFilter * count] += complex<T>(eps * h.real, eps * -h.img) * v;
  }
}

template<class Tensor>
struct conj_repeat_mult_operation {
  typedef Tensor tensor_t;
  typedef typename Tensor::value_t value_t;
  typedef typename value_t::value_t real_value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;

  conj_repeat_mult_operation(Tensor& visibles, Tensor& hiddens, real_value_t eps)
   : _visibles(visibles), _hiddens(hiddens), _eps(eps), _size(visibles.size()), _fullsize(visibles.fullsize())
  {
    _size[3] = visibles.size()[3] * hiddens.size()[3];
    _fullsize[3] = visibles.size()[3] * hiddens.size()[3];
  }

  void apply(tensor_t& filters) const {
    const int threadCount = 1024;
    const int channelCount = _visibles.size()[3];
    const int layerVoxelCount = _visibles.count() / channelCount;
    const int filterCount = _hiddens.size()[3];

    dim3 blockSize(threadCount);
    dim3 gridSize((layerVoxelCount + threadCount - 1) / threadCount, channelCount);
    tbblas_conj_repeat_mult<<<gridSize, blockSize>>>(_visibles.data().data().get(), _hiddens.data().data().get(), filters.data().data().get(),
        layerVoxelCount, channelCount, filterCount, _eps);
  }

  void apply_inc(tensor_t& filters) const {
    const int threadCount = 1024;
    const int channelCount = _visibles.size()[3];
    const int layerVoxelCount = _visibles.count() / channelCount;
    const int filterCount = _hiddens.size()[3];

    dim3 blockSize(threadCount);
    dim3 gridSize((layerVoxelCount + threadCount - 1) / threadCount, channelCount);
    tbblas_conj_repeat_mult_inc<<<gridSize, blockSize>>>(_visibles.data().data().get(), _hiddens.data().data().get(), filters.data().data().get(),
        layerVoxelCount, channelCount, filterCount, _eps);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Tensor& _visibles;
  Tensor& _hiddens;
  real_value_t _eps;
  dim_t _size, _fullsize;
};

template<class T>
struct is_operation<conj_repeat_mult_operation<T> > {
  static const bool value = true;
};

template<class T>
struct is_inc_operation<conj_repeat_mult_operation<T> > {
  static const bool value = true;
};

template<class T>
conj_repeat_mult_operation<tensor<tbblas::complex<T>, 4, true> >
conj_repeat_mult(tensor<tbblas::complex<T>, 4, true>& visibles, tensor<tbblas::complex<T>, 4, true>& hiddens, T eps) {
  typedef tensor<tbblas::complex<T>, 4, true> Tensor;
  assert(visibles.count() / visibles.size()[3] == hiddens.count() / hiddens.size()[3]);
  return conj_repeat_mult_operation<Tensor>(visibles, hiddens, eps);
}

}

#endif /* TBBLAS_REPEAT_MULT_HPP_ */
