/*
 * sum.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SUM_HPP_
#define TBBLAS_SUM_HPP_

#include <tbblas/tensor_base.hpp>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <tbblas/device/sum.hpp>
#include <cassert>

namespace tbblas {

template<class T, unsigned dim, bool device>
T sum(const tensor_base<T, dim, device>& t) {
  return thrust::reduce(t.begin(), t.end());
}

template<class Tensor>
struct tensor_sum {
  Tensor tensor;
  size_t iDim;

  tensor_sum(const Tensor& tensor, const size_t& iDim)
   : tensor(tensor), iDim(iDim) { }
};

template<class T, unsigned dim, bool device>
tensor_sum<tensor_base<T, dim, device> > sum(
    const tensor_base<T, dim, device>& t, const size_t& iDim)
{
  return tensor_sum<tbblas::tensor_base<T, dim, device> >(t, iDim);
}

template<class T, unsigned dim, bool device>
void sum(const tensor_base<T, dim + 1, device>& in, size_t iDim,
    tensor_base<T, dim, device>& out);

template<class T, bool device>
void sum(const tensor_base<T, 1, device>& in, size_t iDim,
    tensor_base<T, 0, device>& out)
{
  out.data()[0] = sum(in);
}

template<class T>
struct sum_kernel {
  int pitch, depth;
  T scalar;

  sum_kernel(int pitch, int depth, const T& scalar)
   : pitch(pitch), depth(depth), scalar(scalar) { }

  __device__ __host__ T operator()(const T& x) const {
    T result = 0;
    const T* i = &x;
    const T* last = depth * pitch + i;
    for (; i < last; i += pitch)
      result += *i;
    return result * scalar;
  }
};

template <typename T1>
struct linear_index_to_column_index : public thrust::unary_function<T1,T1>
{
    T1 rows; // number of columns

    __host__ __device__
    linear_index_to_column_index(T1 rows) : rows(rows) {}

    __host__ __device__
    T1 operator()(T1 i)
    {
        return i / rows;
    }
};

template<class T, bool device>
void sum(const tensor_base<T, 2, device>& in, size_t iDim,
    tensor_base<T, 1, device>& out)
{
  using namespace thrust::placeholders;

  assert(!in.flipped());
  assert(!out.flipped());

  if (iDim == 0) {
    assert(in.scalar() == out.scalar());

    thrust::reduce_by_key(
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(in.size()[0])),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_column_index<int>(in.size()[0])) + in.count(),
        in.data().begin(),
        thrust::make_discard_iterator(),
        out.data().begin()
    );
  } else {
    const int pitch = in.size()[0];
    const int depth = in.size()[1];

    thrust::transform(in.data().begin(), in.data().begin() + pitch, out.begin(),
        sum_kernel<T>(pitch, depth, in.scalar() / out.scalar()));
  }
}

template<class T, unsigned dim, bool device>
void apply_operation(tensor_base<T, dim, device>& tensor,
    const tensor_sum<tensor_base<T, dim + 1, device> >& op)
{
  assert(op.iDim < dim + 1);
  sum(op.tensor, op.iDim, tensor);
}

}

#endif /* SUM_HPP_ */
