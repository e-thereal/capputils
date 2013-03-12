/*
 * repeat.hpp
 *
 *  Created on: Nov 28, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_REPEAT_HPP_
#define TBBLAS_REPEAT_HPP_

#include <tbblas/tensor.hpp>

namespace tbblas {

template<class Expression>
struct repeat_expression {

  typedef repeat_expression<Expression> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const unsigned dimCount = Expression::dimCount;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t inSize, outSize, pitch;

    index_functor(dim_t inSize, dim_t outSize) : inSize(inSize), outSize(outSize) {
      pitch[0] = 1;
      for (unsigned i = 1; i < dimCount; ++i)
        pitch[i] = pitch[i - 1] * inSize[i - 1];
    }

    __host__ __device__
    difference_type operator()(difference_type idx) const {
      // get x, y, z, ... components of the index
      // calculate % inSize[i]
      // calculate new index

      difference_type index;
      index = (idx % inSize[0]) * pitch[0];
      for (unsigned k = 1; k < dimCount; ++k) {
        index += ((idx /= outSize[k-1]) % inSize[k]) * pitch[k];

      }
      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  repeat_expression(const Expression& expr, const dim_t& reps)
   : expr(expr), outSize(expr.size() * reps), outFullsize(expr.fullsize() * reps) { }

  inline const_iterator begin() const {
    index_functor functor(expr.size(), outSize);
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
    return outFullsize;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= outSize[i];
    return count;
  }

private:
  const Expression& expr;
  dim_t outSize, outFullsize;
};

template<class T>
struct is_expression<repeat_expression<T> > {
  static const bool value = true;
};

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    repeat_expression<Expression>
>::type
repeat(const Expression& expr, const sequence<int, Expression::dimCount>& reps)
{
  return repeat_expression<Expression>(expr, reps);
}

}

#endif /* TBBLAS_REPEAT_HPP_ */
