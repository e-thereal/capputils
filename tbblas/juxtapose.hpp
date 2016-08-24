/*
 * juxtapose.hpp
 *
 *  Created on: Aug 24, 2016
 *      Author: Tom Brosch
 */

#ifndef TBBLAS_JUXTAPOSE_HPP_
#define TBBLAS_JUXTAPOSE_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/type_traits.hpp>
#include <tbblas/assert.hpp>

namespace tbblas {

template<class Tensor>
struct juxtapose_operation {
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;

  static const unsigned dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;

  typedef Tensor tensor_t;

  juxtapose_operation(const Tensor& input, int inDim, int outDim1, int outDim2)
   : _input(input), _inDim(inDim)
  {
    tbblas_assert(inDim < dimCount);
    tbblas_assert(outDim1 < dimCount);
    tbblas_assert(outDim2 < dimCount);

    _slice_size = input.size();
    _slice_size[inDim] = 1;

    int count = input.size()[inDim];

    int columnCount = (int)::ceil(::sqrt((float)count));
	int rowCount = (int)::ceil((float)count / (float)columnCount);

    _matrix_size = seq<dimCount>(1);
    _matrix_size[outDim1] = columnCount;
    _matrix_size[outDim2] = rowCount;

    _size = _slice_size * _matrix_size;
  }

  void apply(tensor_t& output) const {
    output = zeros<value_t>(_size);

    int i = 0;
    dim_t topleft = seq<dimCount>(0);
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), _matrix_size); iter && i < _input.size()[_inDim]; ++iter, ++i)
    {
        topleft[_inDim] = i;
        dim_t tl = *iter * _slice_size;
        output[tl, _slice_size] = _input[topleft, _slice_size];
    }
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

private:
  const Tensor& _input;
  dim_t _size, _slice_size, _matrix_size;
  int _inDim;
};

template<class T>
struct is_operation<juxtapose_operation<T> > {
  static const bool value = true;
};

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount >= 2,
      juxtapose_operation<Tensor>
    >::type
>::type
juxtapose(const Tensor& input, int inDim, int outDim1, int outDim2)
{
  return juxtapose_operation<Tensor>(input, inDim, outDim1, outDim2);
}

}

#endif /* TBBLAS_JUXTAPOSE_HPP_ */
