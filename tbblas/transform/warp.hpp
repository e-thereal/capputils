/*
 * warp.hpp
 *
 *  Created on: 2014-11-07
 *      Author: tombr
 */

#ifndef TBBLAS_TRANSFORM_WARP_HPP_
#define TBBLAS_TRANSFORM_WARP_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>

namespace tbblas {

namespace transform {

// TODO: introduce transform_plan to avoid multiple allocation and delocation of the cuda array
//       since we don't know if the participating tensor has changed, we still need to copy
//       the tensor to the array. Need to measure how much this impacts the performance

template<class T>
void warp(tensor<T, 3, true>& input, const typename tensor<T, 3, true>::dim_t& voxel_size, tensor<T, 4, true>& deformation, tensor<T, 3, true>& output);

template<>
void warp(tensor<float, 3, true>& input, const tensor<float, 3, true>::dim_t& voxel_size, tensor<float, 4, true>& deformation, tensor<float, 3, true>& output);

template<class Tensor, class Deformation>
struct warp_operation {
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;

  typedef Tensor tensor_t;
  typedef Deformation deformation_t;

  warp_operation(Tensor& tensor, Deformation& deformation, const dim_t& voxel_size)
   : _tensor(tensor), _deformation(deformation), _size(tensor.size()), _fullsize(tensor.fullsize()), _voxel_size(voxel_size)
  { }

  void apply(tensor_t& output) const {
    warp(_tensor, _voxel_size, _deformation, output);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Tensor& _tensor;
  Deformation& _deformation;
  dim_t _size, _fullsize, _voxel_size;
};

}

template<class T, class D>
struct is_operation<tbblas::transform::warp_operation<T, D> > {
  static const bool value = true;
};

namespace transform {

template<class Tensor, class Deformation>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if<is_tensor<Deformation>,
      typename boost::enable_if_c<Tensor::dimCount == 3 && Tensor::cuda_enabled && Deformation::dimCount == 4 && Deformation::cuda_enabled,
        tbblas::transform::warp_operation<Tensor, Deformation>
      >::type
    >::type
>::type
warp(Tensor& tensor, Deformation& deformation, const typename Tensor::dim_t& voxel_size) {
  for (unsigned i = 0; i < Tensor::dimCount; ++i)
    assert(deformation.size()[i] == tensor.size()[i]);
  assert(deformation.size()[Tensor::dimCount] == Tensor::dimCount);

  return tbblas::transform::warp_operation<Tensor, Deformation>(tensor, deformation, voxel_size);
}

}

}

#endif /* TBBLAS_TRANSFORM_WARP_HPP_ */
