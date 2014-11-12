/*
 * slice.hpp
 *
 *  Created on: Nov 11, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_SLICE_HPP_
#define TBBLAS_SLICE_HPP_

#include <tbblas/tensor.hpp>

namespace tbblas {

template<class Expression>
struct slice_expression {

  typedef slice_expression<Expression> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const unsigned dimCount = Expression::dimCount;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t inSize, outSize, pitch, block;

    index_functor(dim_t inSize, dim_t outSize, dim_t block) : inSize(inSize), outSize(outSize), block(block) {
      pitch[0] = 1;
      for (unsigned i = 1; i < dimCount; ++i) {
        pitch[i] = pitch[i - 1] * inSize[i - 1];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type idx) const {
      // get x, y, z, ... components of the index
      // calculate % outSize[i]
      // calculate new index

      difference_type index;
      index = (idx % outSize[0]) * block[0] * pitch[0];
      for (unsigned k = 1; k < dimCount; ++k)
        index += ((idx /= outSize[k-1]) % outSize[k]) * block[k] * pitch[k];

      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  slice_expression(const Expression& expr, const dim_t& stride)
   : _expr(expr), _stride(stride)
  {
    for (unsigned i = 0; i < dimCount; ++i) {
      _outSize[i] = expr.size()[i] / stride[i];
    }
  }

  inline const_iterator begin() const {
    index_functor functor(_expr.size(), _outSize, _stride);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(_expr.begin(), transform);
    return permu;
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return _outSize;
  }

  inline dim_t fullsize() const {
    return _outSize;
  }

  inline size_t count() const {
    return size().prod();
  }

private:
  const Expression& _expr;
  dim_t _stride, _outSize;
};

template<class T>
struct is_expression<slice_expression<T> > {
  static const bool value = true;
};

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    slice_expression<Expression>
>::type
slice(const Expression& expr, const sequence<int, Expression::dimCount>& stride)
{
  return slice_expression<Expression>(expr, stride);
}

/*** THE REVERSE CASE ***/

template<class Expression>
struct slice_r_expression {

  typedef slice_r_expression<Expression> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const unsigned dimCount = Expression::dimCount;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t inSize, outSize, pitch, block;

    index_functor(dim_t inSize, dim_t outSize, dim_t block) : inSize(inSize), outSize(outSize), block(block) {
      pitch[0] = 1;
      for (unsigned i = 1; i < dimCount; ++i) {
        pitch[i] = pitch[i - 1] * inSize[i - 1];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type idx) const {
      // get x, y, z, ... components of the index
      // calculate % inSize[i]
      // calculate new index

      difference_type index;

      index = ((idx % outSize[0]) / block[0]) * pitch[0];
      for (unsigned k = 1; k < dimCount; ++k) {
        index += (((idx /= outSize[k-1]) % outSize[k]) / block[k]) * pitch[k];
      }

      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  slice_r_expression(const Expression& expr, const dim_t& stride)
   : _expr(expr), _stride(stride)
  {
    for (unsigned i = 0; i < dimCount; ++i) {
      _outSize[i] = expr.size()[i] * stride[i];
    }
  }

  inline const_iterator begin() const {
    index_functor functor(_expr.size(), _outSize, _stride);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(_expr.begin(), transform);
    return permu;
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return _outSize;
  }

  inline dim_t fullsize() const {
    return _outSize;
  }

  inline size_t count() const {
    return _outSize.prod();
  }

private:
  const Expression& _expr;
  dim_t _stride, _outSize;
};

template<class T>
struct is_expression<slice_r_expression<T> > {
  static const bool value = true;
};

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    slice_r_expression<Expression>
>::type
slice_r(const Expression& expr, const sequence<int, Expression::dimCount>& stride)
{
  return slice_r_expression<Expression>(expr, stride);
}

}

#endif /* TBBLAS_SLICE_HPP_ */
