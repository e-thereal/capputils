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
#include <tbblas/fill.hpp>

#include <boost/static_assert.hpp>

#include <tbblas/detail/copy.hpp>
#include <tbblas/detail/system.hpp>

#include <cassert>

namespace tbblas {

template<class Tensor>
struct proxy {

  template<class T, unsigned dim, bool device>
  friend class tensor;

  typedef proxy<Tensor> proxy_t;
  typedef typename Tensor::value_t value_t;
  typedef typename Tensor::dim_t dim_t;
  typedef typename Tensor::data_t data_t;

  static const unsigned dimCount = Tensor::dimCount;
  static const bool cuda_enabled = Tensor::cuda_enabled;

  typedef int difference_type;

  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    typedef sequence<bool,dimCount> flipped_t;

    dim_t _size;
    dim_t _pitch;
    size_t _first;
    flipped_t _flipped;
    dim_t _stride;

    index_functor(const dim_t& start, const dim_t& size, const dim_t& pitch, const flipped_t& flipped, const dim_t& order, const dim_t& stride)  {
      for (unsigned i = 0; i < dimCount; ++i) {
        _size[i] = size[i];
        _flipped[i] = flipped[i];
        _stride[i] = stride[i];
      }

      dim_t originalPitch;
      originalPitch[0] = 1;
      for (unsigned k = 1; k < dimCount; ++k)
        originalPitch[k] = originalPitch[k-1] * pitch[k-1];

      _first = 0;
      for (unsigned k = 0; k < dimCount; ++k) {
        _pitch[k] = originalPitch[order[k]];
        _first += _pitch[k] * start[k];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type i) const
    {
      difference_type index;
      index = (_flipped[0] ? _size[0] - (i % _size[0]) - 1 : i % _size[0]) * _stride[0] * _pitch[0];
      for (unsigned k = 1; k < dimCount; ++k) {
        index += (_flipped[k] ? _size[k] - ((i /=  _size[k-1]) % _size[k]) - 1 : (i /= _size[k-1]) % _size[k]) * _stride[k] * _pitch[k];
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
   : _data(tensor.shared_data()), _size(tensor.size()), _fullsize(tensor.fullsize()), _pitch(tensor.size()), _stride(tbblas::seq<dimCount>(1)), _flipped(tbblas::seq<dimCount>(false))
  {
    for (unsigned i = 0; i < dimCount; ++i)
      _order[i] = i;
  }

  proxy(Tensor& tensor, const dim_t& size)
   : _data(tensor.shared_data()), _size(size), _fullsize(size), _pitch(size), _stride(tbblas::seq<dimCount>(1)), _flipped(false)
  {
    for (unsigned i = 0; i < dimCount; ++i)
      _order[i] = i;
  }

  proxy(Tensor& tensor, const dim_t& start, const dim_t& size)
   : _data(tensor.shared_data()), _start(start), _size(size), _fullsize(size), _pitch(tensor.size()), _stride(tbblas::seq<dimCount>(1)), _flipped(tbblas::seq<dimCount>(false))
  {
    for (unsigned i = 0; i < dimCount; ++i) {
      assert(_start[i] + _size[i] <= _pitch[i]);
      _order[i] = i;
    }
  }

  proxy(Tensor& tensor, const dim_t& start, const dim_t& stride, const dim_t& size)
   : _data(tensor.shared_data()), _start(start), _size((size + stride - 1) / stride), _fullsize(size / stride), _pitch(tensor.size()), _stride(stride), _flipped(tbblas::seq<dimCount>(false))
  {
    for (unsigned i = 0; i < dimCount; ++i) {
      assert(_start[i] + _size[i] * _stride[i] - _stride[i] < _pitch[i]);
      _order[i] = i;
    }
  }

  inline iterator begin() const {
    index_functor functor(_start, _size, _pitch, _flipped, _order, _stride);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(_data->begin(), transform);
    return permu;
//    return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(_size, _pitch)));
  }

  inline iterator end() const {
    return begin() + count();
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= _size[i];
    return count;
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

  inline void set_flipped(const sequence<bool, dimCount>& flipped) {
    _flipped = flipped;
  }

  inline sequence<bool, dimCount> flipped() const {
    return _flipped;
  }

  inline void reorder(const dim_t& order) {
    dim_t oldOrder = _order;
    dim_t oldSize = _size;
    dim_t oldFullsize = _fullsize;
    dim_t oldStart = _start;
    dim_t oldStride = _stride;

    for (unsigned i = 0; i < dimCount; ++i) {
      _order[i] = oldOrder[order[i]];
      _size[i] = oldSize[order[i]];
      _fullsize[i] = oldFullsize[order[i]];
      _start[i] = oldStart[order[i]];
      _stride[i] = oldStride[order[i]];
    }
  }

  inline dim_t start() const {
    return _start;
  }

  inline dim_t pitch() const {
    return _pitch;
  }

  inline dim_t order() const {
    return _order;
  }

  inline const data_t& data() const {
    return *_data;
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
      assert(size[i] == _size[i]);
    }
    tbblas::detail::copy(typename tbblas::detail::select_system<cuda_enabled && Expression::cuda_enabled>::system(), expr.begin(), expr.end(), begin());
    return *this;
  }

  // todo: create a proxy that maps row,cols,depth,... to col,row,depth,...
  proxy_filler<proxy_t> operator=(const value_t& value) const {
    proxy_t p = *this;
    if (dimCount > 1) {
      dim_t order;
      for (unsigned i = 0; i < dimCount; ++i) {
        if (i < 2)
          order[i] = !i;
        else
          order[i] = i;
      }
      p.reorder(order);
    }
    return proxy_filler<proxy_t>(p), value;
  }

private:
  boost::shared_ptr<data_t> _data;
  dim_t _start, _size, _fullsize, _pitch, _order, _stride;
  sequence<bool, dimCount> _flipped;
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
const proxy<tensor<T, dim, device> > subrange(tensor<T, dim, device>& t,
    const sequence<unsigned, dim>& start, const sequence<unsigned, dim>& size)
{
  return proxy<tensor<T, dim, device> >(t, start, size);
}

template<class T, unsigned dim, bool device>
const proxy<tensor<T, dim, device> > subrange(tensor<T, dim, device>& t,
    const sequence<int, dim>& start, const sequence<int, dim>& size)
{
  return proxy<tensor<T, dim, device> >(t, start, size);
}

template<class T, unsigned dim, bool device>
const proxy<tensor<T, dim, device> > slice(tensor<T, dim, device>& t,
    const sequence<int, dim>& start, const sequence<int, dim>& stride, const sequence<int, dim>& size)
{
  return proxy<tensor<T, dim, device> >(t, start, stride, size);
}

}

#endif /* TBBLAS_PROXY_HPP_ */
