/*
 * transform.hpp
 *
 *  Created on: 2014-10-05
 *      Author: tombr
 */

#ifndef TBBLAS_IMGPROC_TRANSFORM_HPP_
#define TBBLAS_IMGPROC_TRANSFORM_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>

#include <tbblas/imgproc/fmatrix4.hpp>

namespace tbblas {

namespace imgproc {

// TODO: introduce transform_plan to avoid multiple allocation and deallocation of the CUDA array
//       since we don't know if the participating tensor has changed, we still need to copy
//       the tensor to the array. Need to measure how much this impacts the performance

template<class T>
void transform(tensor<T, 3, true>& input, const fmatrix4& matrix, tensor<T, 3, true>& output);

template<class T>
void transform(tensor<T, 4, true>& input, const fmatrix4& matrix, tensor<T, 4, true>& output);

template<>
void transform(tensor<float, 3, true>& input, const fmatrix4& matrix, tensor<float, 3, true>& output);

template<>
void transform(tensor<float, 4, true>& input, const fmatrix4& matrix, tensor<float, 4, true>& output);

template<class Tensor>
struct transform_operation {
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;

  typedef Tensor tensor_t;

  transform_operation(Tensor& tensor, const fmatrix4& mat, dim_t outSize)
   : _tensor(tensor), _mat(mat), _size(outSize), _fullsize(outSize)
  {
  }

  void apply(tensor_t& output) const {
    transform(_tensor, _mat, output);
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Tensor& _tensor;
  dim_t _size, _fullsize;
  fmatrix4 _mat;
};

}

template<class T>
struct is_operation<tbblas::imgproc::transform_operation<T> > {
  static const bool value = true;
};

namespace imgproc {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount == 3 && Tensor::cuda_enabled == true,
      transform_operation<Tensor>
    >::type
>::type
transform(Tensor& tensor, const fmatrix4& mat, typename Tensor::dim_t outSize = seq<3>(-1)) {
  if (outSize == seq<3>(-1))
    outSize = tensor.size();
  return transform_operation<Tensor>(tensor, mat, outSize);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount == 4 && Tensor::cuda_enabled == true,
      transform_operation<Tensor>
    >::type
>::type
transform(Tensor& tensor, const fmatrix4& mat, typename Tensor::dim_t outSize = seq<4>(-1)) {
  if (outSize == seq<4>(-1))
    outSize = tensor.size();

  tbblas_assert(tensor.size()[3] == outSize[3]);
  return transform_operation<Tensor>(tensor, mat, outSize);
}

//void transform(tensor, matrix, skip, wrap);

}

}

#endif /* TBBLAS_TRANSFORM_TRANSFORM_HPP_ */
