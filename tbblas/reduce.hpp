/*
 * reduce.hpp
 *
 *  Created on: Sep 2, 2016
 *      Author: tombr
 */

#ifndef TBBLAS_REDUCE_HPP_
#define TBBLAS_REDUCE_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/proxy.hpp>
#include <tbblas/complex.hpp>
#include <tbblas/context.hpp>
#include <tbblas/assert.hpp>

#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <tbblas/detail/reduce.hpp>
#include <tbblas/detail/copy.hpp>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

/* reduce over all elements */

template<class Expression, class Operation>
typename boost::enable_if<is_expression<Expression>,
  typename Expression::value_t
>::type
reduce(const Expression& expr, const Operation& op) {
  typename Expression::value_t init = *expr.begin();
  return tbblas::detail::reduce(
      typename tbblas::detail::select_system<Expression::cuda_enabled>::system(),
      expr.begin() + 1, expr.end(), init, op);
}

/* Reduce along one dimension */

template<class Proxy, class Operation>
struct reduce_operation {
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

  reduce_operation(const Proxy& proxy, const Operation& op, unsigned dimIdx)
   : _proxy(proxy), _op(op), _size(proxy.size()), _fullsize(proxy.fullsize()), dimIdx(dimIdx)
  {
    typename Proxy::dim_t order;
    order[0] = dimIdx;
    for (unsigned i = 1; i < Proxy::dimCount; ++i) {
      order[i] = (int)i - (i <= dimIdx);
    }
    _proxy.reorder(order);
    _size[dimIdx] = 1;
    _fullsize[dimIdx] = 1;
  }

  void apply(tensor_t& output) const {
    // special case for the trivial sum operation of only one value (size[dimIdx] == 1)
    // Since we reordered the proxy, we test if the first dimension is 1
    if (_proxy.size()[0] == 1) {
      tbblas::detail::copy(typename tbblas::detail::select_system<Proxy::cuda_enabled>::system(),
          _proxy.begin(), _proxy.end(), output.begin());
    } else {
      tbblas::detail::reduce_by_key(
          typename tbblas::detail::select_system<Proxy::cuda_enabled>::system(),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])) + _proxy.count(),
          _proxy.begin(),
          thrust::make_discard_iterator(),
          output.begin(),
          thrust::equal_to<int>(),
          _op
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
  Proxy _proxy;
  Operation _op;
  dim_t _size, _fullsize;
  unsigned dimIdx;
};

template<class T, class Operation>
struct is_operation<reduce_operation<T, Operation> > {
  static const bool value = true;
};

template<class Proxy, class Operation>
typename boost::enable_if<is_proxy<Proxy>,
    typename boost::enable_if_c<Proxy::dimCount >= 1,
      reduce_operation<Proxy, Operation>
    >::type
>::type
reduce(const Proxy& proxy, unsigned dimIdx, const Operation& op) {
  tbblas_assert(dimIdx < Proxy::dimCount);
  return reduce_operation<Proxy, Operation>(proxy, op, dimIdx);
}

template<class Tensor, class Operation>
typename boost::enable_if<is_tensor<Tensor>,
    typename boost::enable_if_c<Tensor::dimCount >= 1,
      typename boost::disable_if_c<(Tensor::dimCount == 3 || Tensor::dimCount == 4) && Tensor::cuda_enabled == true,
        reduce_operation<proxy<Tensor>, Operation>
      >::type
    >::type
>::type
reduce(Tensor& tensor, unsigned dimIdx, const Operation& op) {
  tbblas_assert(dimIdx < Tensor::dimCount);
  return reduce(proxy<Tensor>(tensor), dimIdx, op);
}

/*** special implementations ***/

template<class T, class Operation>
void reduceLast(const proxy<tensor<T, 3, true> >& input, const Operation& op, tensor<T, 3, true>& output);

template<class T, class Operation>
void reduceLast(const proxy<tensor<T, 4, true> >& input, const Operation& op, tensor<T, 4, true>& output);

#ifdef __CUDACC__
template<class T, class Operation>
__global__ void tbblas_reduceLast(T* output, const T* input, int width, int height, int depth, const Operation& op) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  int count = width * height;
  if (x >= width || y >= height)
    return;

  const int offset = y * width + x;
  input += offset;

  T result = *input;
  input += count;

  for (int z = 1; z < depth; ++z, input += count) {
    result = op(result, *input);
  }

  output[offset] = result;
}

/// Memory must be allocated
template<class T, class Operation>
void reduceLast(const proxy<tensor<T, 3, true> >& input, const Operation& op, tensor<T, 3, true>& output) {
  dim3 blockSize(32, 32);
  dim3 gridSize((input.size()[0] + 31) / 32, (input.size()[1] + 31) / 32);
  tbblas_reduceLast<<<gridSize, blockSize, 0, context::get().stream>>>(output.data().data().get(), input.data().data().get(), input.size()[0], input.size()[1], input.size()[2], op);
}

template<class T, class Operation>
void reduceLast(const proxy<tensor<complex<T>, 3, true> >& input, const Operation& op, tensor<complex<T>, 3, true>& output) {
  dim3 blockSize(32, 32);
  dim3 gridSize((2 * input.size()[0] + 31) / 32, (input.size()[1] + 31) / 32);
  tbblas_reduceLast<<<gridSize, blockSize, 0, context::get().stream>>>((T*)output.data().data().get(), (T*)input.data().data().get(), 2 * input.size()[0], input.size()[1], input.size()[2], op);
}

template<class T, class Operation>
__global__ void tbblas_reduceLast(T* output, const T* input, int size1, int size2, int size3, int size4, const Operation& op) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  int count = size1 * size2 * size3;
  if (x >= size1 || y >= size2)
    return;

  for (int z = 0; z < size3; ++z) {
    const int offset = (z * size2 + y) * size1 + x;
    const T* pinput = input + offset;

    T result = *pinput;
    pinput += count;

    for (int k = 1; k < size4; ++k, pinput += count)
      result = op(result, *pinput);

    output[offset] = result;
  }
}

/// Memory must be allocated
template<class T, class Operation>
void reduceLast(const proxy<tensor<T, 4, true> >& input, const Operation& op, tensor<T, 4, true>& output) {
  dim3 blockSize(32, 32);
  dim3 gridSize((input.size()[0] + 31) / 32, (input.size()[1] + 31) / 32);
  tbblas_reduceLast<<<gridSize, blockSize, 0, context::get().stream>>>(output.data().data().get(), input.data().data().get(), input.size()[0], input.size()[1], input.size()[2], input.size()[3], op);
}

template<class T, class Operation>
void reduceLast(const proxy<tensor<complex<T>, 4, true> >& input, const Operation& op, tensor<complex<T>, 4, true>& output) {
  dim3 blockSize(32, 32);
  dim3 gridSize((2 * input.size()[0] + 31) / 32, (input.size()[1] + 31) / 32);
  tbblas_reduceLast<<<gridSize, blockSize, 0, context::get().stream>>>((T*)output.data().data().get(), (T*)input.data().data().get(), 2 * input.size()[0], input.size()[1], input.size()[2], input.size()[3], op);
}
#endif

template<class T, class Operation>
struct reduce_operation<tensor<T, 3, true>, Operation> {
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

  reduce_operation(tensor_t& tensor, const Operation& op, unsigned dimIdx)
   : _proxy(tensor), _op(op), _size(tensor.size()), _fullsize(tensor.fullsize()), _dimIdx(dimIdx)
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
      reduceLast(_proxy, _op, output);
    } else {
      tbblas::detail::reduce_by_key(
          typename tbblas::detail::select_system<true>::system(),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])) + _proxy.count(),
          _proxy.begin(),
          thrust::make_discard_iterator(),
          output.begin(),
          thrust::equal_to<int>(),
          _op
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
  Operation _op;
  dim_t _size, _fullsize;
  unsigned _dimIdx;
};

template<class T, class Operation>
struct reduce_operation<tensor<T, 4, true>, Operation> {
  typedef T value_t;
  typedef typename tensor<T, 4, true>::dim_t dim_t;

  static const unsigned dimCount = 4;
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

  reduce_operation(tensor_t& tensor, const Operation& op, const unsigned dimIdx)
   : _proxy(tensor), _op(op), _size(tensor.size()), _fullsize(tensor.fullsize()), _dimIdx(dimIdx)
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
      reduceLast(_proxy, _op, output);
    } else {
      tbblas::detail::reduce_by_key(
          typename tbblas::detail::select_system<true>::system(),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(_proxy.size()[0])) + _proxy.count(),
          _proxy.begin(),
          thrust::make_discard_iterator(),
          output.begin(),
          _op
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
  Operation _op;
  dim_t _size, _fullsize;
  unsigned _dimIdx;
};

template<class Tensor, class Operation>
typename boost::enable_if<is_tensor<Tensor>,
  typename boost::enable_if_c<(Tensor::dimCount == 3 || Tensor::dimCount == 4) && Tensor::cuda_enabled == true,
    reduce_operation<Tensor, Operation>
  >::type
>::type
reduce(Tensor& tensor, unsigned dimIdx, const Operation& op) {
  tbblas_assert(dimIdx < Tensor::dimCount);
  return reduce_operation<Tensor, Operation>(tensor, op, dimIdx);
}

}

#endif /* TBBLAS_REDUCE_HPP_ */
