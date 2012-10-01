/*
 * proxy.hpp
 *
 *  Created on: Sep 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_PROXY_HPP_
#define TBBLAS_PROXY_HPP_

#include <tbblas/sequence.hpp>

#include <tbblas/tensor.hpp>
#include <boost/static_assert.hpp>

#include <cassert>

#ifndef __CUDACC__
#define __global__
#define __host__
#define __device__
#endif

namespace tbblas {

template<class T1, class T2, unsigned dim>
void copy(T1 src, sequence<unsigned, dim> srcStart, sequence<unsigned, dim> srcPitch, sequence<bool, dim> srcFlipped,
    T2 dst, sequence<unsigned, dim> dstStart, sequence<unsigned, dim> dstPitch, sequence<bool, dim> dstFlipped,
    sequence<unsigned, dim> size)
{
  //BOOST_STATIC_ASSERT(0);
  assert(0);
}

template<class Tensor>
struct proxy {

  template<class T, unsigned dim, bool device>
  friend class tensor;

  typedef proxy<Tensor> proxy_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;
  typedef typename Tensor::data_t data_t;

  static const int dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;

  typedef int difference_type;

  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    typedef bool flipped_t[dimCount];

    dim_t _size;
    dim_t _pitch;
    size_t _first;
    flipped_t _flipped;

    index_functor(const dim_t& start, const dim_t& size, const dim_t& pitch, const flipped_t& flipped) {
      for (int i = 0; i < dimCount; ++i) {
        _size[i] = size[i];
        _pitch[i] = pitch[i];
        _flipped[i] = flipped[i];
      }

      _first = start[0];
      _pitch[0] = 1;
      for (int k = 1; k < dimCount; ++k) {
        _pitch[k] = _pitch[k-1] * pitch[k-1];
        _first += _pitch[k] * start[k];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type i) const
    {
      difference_type index;
      index = (_flipped[0] ? _size[0] - (i % _size[0]) - 1 : i % _size[0]) * _pitch[0];
      for (int k = 1; k < dimCount; ++k) {
        index += (_flipped[k] ? _size[k] - ((i /=  _size[k-1]) % _size[k]) - 1 : (i /= _size[k-1]) % _size[k]) * _pitch[k];
      }
      return index + _first;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Tensor::iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator iterator;
  typedef iterator const_iterator;

  proxy(Tensor& tensor)
   : data(tensor.shared_data()), size(tensor.size()), pitch(tensor.size()), flipped(false)
  { }

  proxy(Tensor& tensor, const sequence<unsigned, dimCount>& start,
      const sequence<unsigned, dimCount>& size)
   : data(tensor.shared_data()), start(start), size(size), pitch(tensor.size()), flipped(false)
  {
    for (unsigned i = 0; i < dimCount; ++i) {
      assert(start[i] + size[i] <= pitch[i]);
    }
  }

  inline iterator begin() const {
    index_functor functor(start.get(), size.get(), pitch.get(), flipped.get());
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(data->begin(), transform);
    return permu;
//    return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(_size, _pitch)));
  }

  inline iterator end() const {
    return begin() + count();
  }

  inline size_t count() const {
    size_t count = 1;
    for (int i = 0; i < dimCount; ++i)
      count *= size[i];
    return count;
  }

  template<class Expression>
  inline const typename boost::enable_if<is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount,
      proxy_t
    >::type
  >::type&
  operator=(const Expression& expr) const {
    const dim_t& size = expr.size();
    for (unsigned i = 0; i < dimCount; ++i) {
      assert(size[i] == this->size[i]);
    }
    thrust::copy(expr.begin(), expr.end(), begin());
    return *this;
  }

#ifdef OLD_METHOD
  const proxy<Tensor>&
  operator=(const proxy<Tensor>& p) {
    for (unsigned i = 0; i < dimCount; ++i) {
      assert(size[i] == p.size[i]);
    }
    copy(&(*p.data)[0], p.start, p.pitch, p.flipped, &(*data)[0], start, pitch, flipped, size);
    return p;
  }

  const Tensor&
  operator=(const Tensor& t) {
    for (unsigned i = 0; i < dimCount; ++i) {
      assert(size[i] == t.size()[i]);
    }
    copy(&t.data()[0], sequence<unsigned, dimCount>(), size, sequence<bool, dimCount>(false),
        &(*data)[0], start, pitch, sequence<bool, dimCount>(false), size);
    return t;
  }
#endif

public:
  boost::shared_ptr<data_t> data;
  sequence<unsigned, dimCount> start, size, pitch;
  sequence<bool, dimCount> flipped;
};

template<class T>
struct is_proxy<proxy<T> > {
  static const bool value = true;
};

template<class T>
struct is_expression<proxy<T> > {
  static const bool value = true;
};

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > subrange(tensor<T, dim, device>& t,
    const sequence<unsigned, dim>& start, const sequence<unsigned, dim>& size)
{
  return proxy<tensor<T, dim, device> >(t, start, size);
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > subrange(tensor<T, dim, device>& t,
    const sequence<int, dim>& start, const sequence<int, dim>& size)
{
  return proxy<tensor<T, dim, device> >(t, start, size);
}

/*** CUDA IMPLEMENTATIONS ***/

template<class T>
void copy(thrust::device_ptr<const T> src, sequence<unsigned, 1> srcStart, sequence<unsigned, 1> srcPitch, sequence<bool, 1> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 1> dstStart, sequence<unsigned, 1> dstPitch, sequence<bool, 1> dstFlipped,
    sequence<unsigned, 1> size)
{
  // TODO: implement flipped using reverse iterator
  assert(srcFlipped[0] == false);
  assert(dstFlipped[0] == false);
  thrust::copy(src + srcStart[0], src + srcStart[0] + size[0], dst + dstStart[0]);
}

template<class T>
void copy(thrust::device_ptr<T> src, sequence<unsigned, 1> srcStart, sequence<unsigned, 1> srcPitch, sequence<bool, 1> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 1> dstStart, sequence<unsigned, 1> dstPitch, sequence<bool, 1> dstFlipped,
    sequence<unsigned, 1> size)
{
  // TODO: implement flipped using reverse iterator
  assert(srcFlipped[0] == false);
  assert(dstFlipped[0] == false);
  thrust::copy(src + srcStart[0], src + srcStart[0] + size[0], dst + dstStart[0]);
}

template<class T>
__global__ void copy2D_kernel(const T* src, sequence<unsigned, 2> srcStart, sequence<unsigned, 2> srcPitch, sequence<bool, 2> srcFlipped,
    T* dst, sequence<unsigned, 2> dstStart, sequence<unsigned, 2> dstPitch, sequence<bool, 2> dstFlipped,
    sequence<unsigned, 2> size)
{
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;
  const int ty = threadIdx.y + blockIdx.y * blockDim.y;

  if (tx >= size[0] || ty >= size[1])
    return;

  const int sx = srcFlipped[0] ? size[0] - tx + srcStart[0] - 1 : tx + srcStart[0];
  const int sy = srcFlipped[1] ? size[1] - ty + srcStart[1] - 1 : ty + srcStart[1];

  const int dx = dstFlipped[0] ? size[0] - tx + dstStart[0] - 1 : tx + dstStart[0];
  const int dy = dstFlipped[1] ? size[1] - ty + dstStart[1] - 1 : ty + dstStart[1];

  dst[dy * dstPitch[0] + dx] = src[sy * srcPitch[0] + sx];
}

template<class T>
void copy(thrust::device_ptr<const T> src, sequence<unsigned, 2> srcStart, sequence<unsigned, 2> srcPitch, sequence<bool, 2> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 2> dstStart, sequence<unsigned, 2> dstPitch, sequence<bool, 2> dstFlipped,
    sequence<unsigned, 2> size)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((size[0] + blockDim.x - 1)/ blockDim.x, (size[1] + blockDim.y - 1) / blockDim.y, 1);

  copy2D_kernel<<<gridDim, blockDim>>>(src.get(), srcStart, srcPitch, srcFlipped,
      dst.get(), dstStart, dstPitch, dstFlipped, size);
}

template<class T>
void copy(thrust::device_ptr<T> src, sequence<unsigned, 2> srcStart, sequence<unsigned, 2> srcPitch, sequence<bool, 2> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 2> dstStart, sequence<unsigned, 2> dstPitch, sequence<bool, 2> dstFlipped,
    sequence<unsigned, 2> size)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((size[0] + blockDim.x - 1)/ blockDim.x, (size[1] + blockDim.y - 1) / blockDim.y, 1);

  copy2D_kernel<<<gridDim, blockDim>>>(src.get(), srcStart, srcPitch, srcFlipped,
      dst.get(), dstStart, dstPitch, dstFlipped, size);
}

template<class T>
__global__ void copy3D_kernel(const T* src, sequence<unsigned, 3> srcStart, sequence<unsigned, 3> srcPitch, sequence<bool, 3> srcFlipped,
    T* dst, sequence<unsigned, 3> dstStart, sequence<unsigned, 3> dstPitch, sequence<bool, 3> dstFlipped,
    sequence<unsigned, 3> size)
{
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;
  const int ty = threadIdx.y + blockIdx.y * blockDim.y;
  const int tz = blockIdx.z;

  if (tx >= size[0] || ty >= size[1] || tz >= size[2])
    return;

  const int sx = srcFlipped[0] ? size[0] - tx + srcStart[0] - 1 : tx + srcStart[0];
  const int sy = srcFlipped[1] ? size[1] - ty + srcStart[1] - 1 : ty + srcStart[1];
  const int sz = srcFlipped[2] ? size[2] - tz + srcStart[2] - 1 : tz + srcStart[2];

  const int dx = dstFlipped[0] ? size[0] - tx + dstStart[0] - 1 : tx + dstStart[0];
  const int dy = dstFlipped[1] ? size[1] - ty + dstStart[1] - 1 : ty + dstStart[1];
  const int dz = dstFlipped[2] ? size[2] - tz + dstStart[2] - 1 : tz + dstStart[2];

  dst[(dz * dstPitch[1] + dy) * dstPitch[0] + dx]
      = src[(sz * srcPitch[1] + sy) * srcPitch[0] + sx];
}

template<class T>
void copy(thrust::device_ptr<const T> src, sequence<unsigned, 3> srcStart, sequence<unsigned, 3> srcPitch, sequence<bool, 3> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 3> dstStart, sequence<unsigned, 3> dstPitch, sequence<bool, 3> dstFlipped,
    sequence<unsigned, 3> size)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((size[0] + blockDim.x - 1)/ blockDim.x, (size[1] + blockDim.y - 1) / blockDim.y, size[2]);

  copy3D_kernel<<<gridDim, blockDim>>>(src.get(), srcStart, srcPitch, srcFlipped,
      dst.get(), dstStart, dstPitch, dstFlipped, size);
}

template<class T>
void copy(thrust::device_ptr<T> src, sequence<unsigned, 3> srcStart, sequence<unsigned, 3> srcPitch, sequence<bool, 3> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 3> dstStart, sequence<unsigned, 3> dstPitch, sequence<bool, 3> dstFlipped,
    sequence<unsigned, 3> size)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((size[0] + blockDim.x - 1)/ blockDim.x, (size[1] + blockDim.y - 1) / blockDim.y, size[2]);

  copy3D_kernel<<<gridDim, blockDim>>>(src.get(), srcStart, srcPitch, srcFlipped,
      dst.get(), dstStart, dstPitch, dstFlipped, size);
}

template<class T>
__global__ void copy4D_kernel(const T* src, sequence<unsigned, 4> srcStart, sequence<unsigned, 4> srcPitch, sequence<bool, 4> srcFlipped,
    T* dst, sequence<unsigned, 4> dstStart, sequence<unsigned, 4> dstPitch, sequence<bool, 4> dstFlipped,
    sequence<unsigned, 4> size)
{
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;
  const int ty = threadIdx.y + blockIdx.y * blockDim.y;
  const int tz = blockIdx.z;

  if (tx >= size[0] || ty >= size[1] || tz >= size[2])
    return;

  const int sx = srcFlipped[0] ? size[0] - tx + srcStart[0] - 1 : tx + srcStart[0];
  const int sy = srcFlipped[1] ? size[1] - ty + srcStart[1] - 1 : ty + srcStart[1];
  const int sz = srcFlipped[2] ? size[2] - tz + srcStart[2] - 1 : tz + srcStart[2];

  const int dx = dstFlipped[0] ? size[0] - tx + dstStart[0] - 1 : tx + dstStart[0];
  const int dy = dstFlipped[1] ? size[1] - ty + dstStart[1] - 1 : ty + dstStart[1];
  const int dz = dstFlipped[2] ? size[2] - tz + dstStart[2] - 1 : tz + dstStart[2];

  for (unsigned w = 0; w < size[3]; ++w) {
    const int sw = srcFlipped[3] ? size[3] - w + srcStart[3] - 1 : w + srcStart[3];
    const int dw = dstFlipped[3] ? size[3] - w + dstStart[3] - 1 : w + dstStart[3];

    dst[((dw * dstPitch[2] + dz) * dstPitch[1] + dy) * dstPitch[0] + dx]
        = src[((sw * srcPitch[2] + sz) * srcPitch[1] + sy) * srcPitch[0] + sx];
  }
}

template<class T>
void copy(thrust::device_ptr<const T> src, sequence<unsigned, 4> srcStart, sequence<unsigned, 4> srcPitch, sequence<bool, 4> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 4> dstStart, sequence<unsigned, 4> dstPitch, sequence<bool, 4> dstFlipped,
    sequence<unsigned, 4> size)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((size[0] + blockDim.x - 1)/ blockDim.x, (size[1] + blockDim.y - 1) / blockDim.y, size[2]);

  copy4D_kernel<<<gridDim, blockDim>>>(src.get(), srcStart, srcPitch, srcFlipped,
      dst.get(), dstStart, dstPitch, dstFlipped, size);
}

template<class T>
void copy(thrust::device_ptr<T> src, sequence<unsigned, 4> srcStart, sequence<unsigned, 4> srcPitch, sequence<bool, 4> srcFlipped,
    thrust::device_ptr<T> dst, sequence<unsigned, 4> dstStart, sequence<unsigned, 4> dstPitch, sequence<bool, 4> dstFlipped,
    sequence<unsigned, 4> size)
{
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((size[0] + blockDim.x - 1)/ blockDim.x, (size[1] + blockDim.y - 1) / blockDim.y, size[2]);

  copy4D_kernel<<<gridDim, blockDim>>>(src.get(), srcStart, srcPitch, srcFlipped,
      dst.get(), dstStart, dstPitch, dstFlipped, size);
}

}

#endif /* TBBLAS_PROXY_HPP_ */