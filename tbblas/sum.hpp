/*
 * sum2.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SUM_HPP_
#define TBBLAS_SUM_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/complex.hpp>

#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

/* Sum of all elements */

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  typename Expression::value_t
>::type
sum(const Expression& expr) {
  return thrust::reduce(expr.begin(), expr.end(), typename Expression::value_t());
}

/* Sum along one dimension */

template<class Proxy>
struct sum_operation {
  typedef typename Proxy::value_t value_t;
  typedef typename Proxy::dim_t dim_t;

  static const unsigned dimCount = Proxy::dimCount;

  typedef tensor<value_t, dimCount, Proxy::cuda_enabled> tensor_t;

  template <typename T1>
  struct linear_index_to_column_index : public thrust::unary_function<T1, T1> {
    T1 rows; // number of rows

    __host__ __device__
    linear_index_to_column_index(T1 rows) : rows(rows) {}

    __host__ __device__
    T1 operator()(T1 i)
    {
        return i / rows;
    }
  };

  sum_operation(const Proxy& proxy, unsigned dimIdx)
   : _proxy(proxy), _size(proxy.size()), _fullsize(proxy.fullsize()), dimIdx(dimIdx)
  {
    typename Proxy::dim_t order;
    order[0] = dimIdx;
    for (int i = 1; i < Proxy::dimCount; ++i) {
      order[i] = i - (i <= dimIdx);
    }
    _proxy.reorder(order);
    _size[dimIdx] = 1;
    _fullsize[dimIdx] = 1;
  }

  void apply(tensor_t& output) const {
    thrust::reduce_by_key(
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])) + _proxy.count(),
        _proxy.begin(),
        thrust::make_discard_iterator(),
        output.begin()
    );
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  Proxy _proxy;
  dim_t _size, _fullsize;
  unsigned dimIdx;
};

template<class T>
struct is_operation<sum_operation<T> > {
  static const bool value = true;
};

template<class Proxy>
typename boost::enable_if<is_proxy<Proxy>,
    typename boost::enable_if_c<Proxy::dimCount >= 1,
      sum_operation<Proxy>
    >::type
>::type
sum(const Proxy& proxy, unsigned dimIdx) {
  assert(dimIdx < Proxy::dimCount);
  return sum_operation<Proxy>(proxy, dimIdx);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount >= 1,
      typename boost::disable_if_c<Tensor::dimCount == 3,
        sum_operation<proxy<Tensor> >
      >::type
    >::type
>::type
sum(Tensor& tensor, unsigned dimIdx) {
  assert(dimIdx < Tensor::dimCount);
  return sum(proxy<Tensor>(tensor), dimIdx);
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount >= 1,
      typename boost::disable_if_c<Tensor::cuda_enabled == true,
        sum_operation<proxy<Tensor> >
      >::type
    >::type
>::type
sum(Tensor& tensor, unsigned dimIdx) {
  assert(dimIdx < Tensor::dimCount);
  return sum(proxy<Tensor>(tensor), dimIdx);
}

/*** special implementations ***/

template<class T>
void sumLast(const proxy<tensor<T, 3, true> >& input, tensor<T, 3, true>& output);

#ifdef __CUDACC__
template<class T>
__global__ void tbblas_sumLast(T* output, const T* input, int width, int height, int depth) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  int count = width * height;
  if (x >= width || y >= height)
    return;

  const int offset = y * width + x;
  input += offset;

  T result = 0;

  for (int z = 0; z < depth; ++z, input += count) {
    result = result + *input;
  }

  output[offset] = result;
}

/// Memory must be allocated
template<class T>
void sumLast(const proxy<tensor<T, 3, true> >& input, tensor<T, 3, true>& output) {
  dim3 blockSize(32, 32);
  dim3 gridSize((input.size()[0] + 31) / 32, (input.size()[1] + 31) / 32);
  tbblas_sumLast<<<gridSize, blockSize>>>(output.data().data().get(), input.data().data().get(), input.size()[0], input.size()[1], input.size()[2]);
}

template<class T>
void sumLast(const proxy<tensor<complex<T>, 3, true> >& input, tensor<complex<T>, 3, true>& output) {
  dim3 blockSize(32, 32);
  dim3 gridSize((2 * input.size()[0] + 31) / 32, (input.size()[1] + 31) / 32);
  tbblas_sumLast<<<gridSize, blockSize>>>((T*)output.data().data().get(), (T*)input.data().data().get(), 2 * input.size()[0], input.size()[1], input.size()[2]);
}
#endif

template<class T>
struct sum_operation<tensor<T, 3, true> > {
  typedef T value_t;
  typedef typename tensor<T, 3, true>::dim_t dim_t;

  static const unsigned dimCount = 3;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef proxy<tensor_t> proxy_t;

  template <typename T1>
  struct linear_index_to_column_index : public thrust::unary_function<T1, T1> {
    T1 rows; // number of rows

    __host__ __device__
    linear_index_to_column_index(T1 rows) : rows(rows) {}

    __host__ __device__
    T1 operator()(T1 i)
    {
        return i / rows;
    }
  };

  sum_operation(tensor_t& tensor, unsigned dimIdx)
   : _proxy(tensor), _size(tensor.size()), _fullsize(tensor.fullsize()), _dimIdx(dimIdx)
  {
    if (dimIdx != dimCount - 1) {
      dim_t order;
      order[0] = dimIdx;
      for (int i = 1; i < dimCount; ++i) {
        order[i] = i - (i <= dimIdx);
      }
      _proxy.reorder(order);
    }
    _size[dimIdx] = 1;
    _fullsize[dimIdx] = 1;
  }

  void apply(tensor_t& output) const {
    if (_dimIdx == dimCount - 1) {
      sumLast(_proxy, output);
    } else {
      thrust::reduce_by_key(
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])) + _proxy.count(),
          _proxy.begin(),
          thrust::make_discard_iterator(),
          output.begin()
      );
    }
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  proxy_t _proxy;
  dim_t _size, _fullsize;
  unsigned _dimIdx;
};

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount == 3,
      typename boost::enable_if_c<Tensor::cuda_enabled == true,
        sum_operation<Tensor>
      >::type
    >::type
>::type
sum(Tensor& tensor, unsigned dimIdx) {
  assert(dimIdx < Tensor::dimCount);
  return sum_operation<Tensor>(tensor, dimIdx);
}

}

#endif /* TBBLAS_SUM_HPP_ */
