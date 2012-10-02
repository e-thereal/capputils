/*
 * shift.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SHIFT_HPP_
#define TBBLAS_SHIFT_HPP_

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

template<class Expression, bool inverse>
struct fftshift_expression {

  typedef fftshift_expression<Expression, inverse> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const int dimCount = Expression::dimCount;

  typedef int difference_type;

  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t _size;
    dim_t _pitch;

    index_functor(const dim_t& size) {
      for (int i = 0; i < dimCount; ++i) {
        _size[i] = size[i];
        if (i == 0)
          _pitch[0] = 1;
        else
          _pitch[i] = _pitch[i-1] * size[i-1];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type i) const
    {
      difference_type index;
      index = ((i + (_size[0]+(!inverse)) / 2) % _size[0]) * _pitch[0];
      for (int k = 1; k < dimCount; ++k) {
        index += (((i /= _size[k-1]) + (_size[k]+!inverse) / 2) % _size[k]) * _pitch[k];
      }
      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  fftshift_expression(const Expression& expr) : expr(expr) { }

  inline const_iterator begin() const {
    index_functor functor(size());
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(expr.begin(), transform);
    return permu;
//    return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(_size, _pitch)));
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline const dim_t& size() const {
    return expr.size();
  }

  inline size_t count() const {
    return expr.count();
  }

private:
  const Expression& expr;
};

template<class T, bool inverse>
struct is_expression<fftshift_expression<T, inverse> > {
  static const bool value = true;
};

template<class Expression>
fftshift_expression<Expression, false> fftshift(const Expression& expr)
{
  return fftshift_expression<Expression, false>(expr);
}

template<class Expression>
fftshift_expression<Expression, true> ifftshift(const Expression& expr)
{
  return fftshift_expression<Expression, true>(expr);
}

}

#endif /* TBBLAS_SHIFT_HPP_ */
