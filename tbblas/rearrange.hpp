/*
 * rearrange.hpp
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_REARRANGE_HPP_
#define TBBLAS_REARRANGE_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/assert.hpp>

namespace tbblas {

template<class Expression>
struct rearrange_expression {

  typedef rearrange_expression<Expression> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const unsigned dimCount = Expression::dimCount;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t inSize, outSize, pitch, block;
    unsigned count, layerCount;

    index_functor(dim_t inSize, dim_t outSize, dim_t block) : inSize(inSize), outSize(outSize), block(block) {
      pitch[0] = 1;
      count = block[0];
      for (unsigned i = 1; i < dimCount; ++i) {
        pitch[i] = pitch[i - 1] * inSize[i - 1];
        count *= block[i];
      }
      layerCount = 1;
      for (unsigned i = 0; i < dimCount - 1; ++i)
        layerCount *= outSize[i];
    }

    __host__ __device__
    difference_type operator()(difference_type idx) const {
      // get x, y, z, ... components of the index
      // calculate % inSize[i]
      // calculate new index

      int cIdx = (idx / layerCount) % count;

      difference_type index;
      index = ((idx % outSize[0]) * block[0] + (cIdx % block[0])) * pitch[0];
      for (unsigned k = 1; k < dimCount - 1; ++k)
        index += (((idx /= outSize[k-1]) % outSize[k]) * block[k] + ((cIdx /= block[k - 1]) % block[k])) * pitch[k];
      index += (((idx /= outSize[dimCount - 2]) % outSize[dimCount - 1]) / count) * pitch[dimCount - 1];

      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  rearrange_expression(const Expression& expr, const dim_t& block)
   : expr(expr), block(block)
  {
    int count = 1;
    for (unsigned i = 0; i < dimCount - 1; ++i) {
      outSize[i] = expr.size()[i] / block[i];
      count *= block[i];
    }
    outSize[dimCount - 1] = expr.size()[dimCount - 1] * count;
  }

  inline const_iterator begin() const {
    index_functor functor(expr.size(), outSize, block);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(expr.begin(), transform);
    return permu;
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return outSize;
  }

  inline dim_t fullsize() const {
    return outSize;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= outSize[i];
    return count;
  }

private:
  const Expression& expr;
  dim_t block, outSize;
};

template<class T>
struct is_expression<rearrange_expression<T> > {
  static const bool value = true;
};

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    rearrange_expression<Expression>
>::type
rearrange(const Expression& expr, const sequence<int, Expression::dimCount>& block)
{
  tbblas_assert(block[Expression::dimCount - 1] == 1);
  return rearrange_expression<Expression>(expr, block);
}

/*** THE REVERSE CASE ***/

template<class Expression>
struct rearrange_r_expression {

  typedef rearrange_r_expression<Expression> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const unsigned dimCount = Expression::dimCount;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t inSize, outSize, pitch, block, blockPitch;
    unsigned count;

    index_functor(dim_t inSize, dim_t outSize, dim_t block) : inSize(inSize), outSize(outSize), block(block) {
      pitch[0] = 1;
      blockPitch[0] = 1;
      count = block[0];
      for (unsigned i = 1; i < dimCount; ++i) {
        pitch[i] = pitch[i - 1] * inSize[i - 1];
        blockPitch[i] = blockPitch[i - 1] * block[i - 1];
        count *= block[i];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type idx) const {
      // get x, y, z, ... components of the index
      // calculate % inSize[i]
      // calculate new index

      difference_type index;
      int cIdx = 0;

      index = ((idx % outSize[0]) / block[0]) * pitch[0];
      cIdx += (idx % block[0]) * blockPitch[0];
      for (unsigned k = 1; k < dimCount - 1; ++k) {
        index += (((idx /= outSize[k-1]) % outSize[k]) / block[k]) * pitch[k];
        cIdx += (idx % block[k]) * blockPitch[k];
      }

      cIdx += (idx /= outSize[dimCount - 2]) * blockPitch[dimCount - 1];
      index += cIdx * pitch[dimCount - 1];

      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  rearrange_r_expression(const Expression& expr, const dim_t& block)
   : expr(expr), block(block)
  {
    int count = 1;
    for (unsigned i = 0; i < dimCount - 1; ++i) {
      outSize[i] = expr.size()[i] * block[i];
      count *= block[i];
    }
    outSize[dimCount - 1] = expr.size()[dimCount - 1] / count;
  }

  inline const_iterator begin() const {
    index_functor functor(expr.size(), outSize, block);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(expr.begin(), transform);
    return permu;
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return outSize;
  }

  inline dim_t fullsize() const {
    return outSize;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= outSize[i];
    return count;
  }

private:
  const Expression& expr;
  dim_t block, outSize;
};

template<class T>
struct is_expression<rearrange_r_expression<T> > {
  static const bool value = true;
};

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    rearrange_r_expression<Expression>
>::type
rearrange_r(const Expression& expr, const sequence<int, Expression::dimCount>& block)
{
  tbblas_assert(block[Expression::dimCount - 1] == 1);
  return rearrange_r_expression<Expression>(expr, block);
}

}


#endif /* TBBLAS_REARRANGE_HPP_ */
