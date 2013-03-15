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

#include <boost/utility/enable_if.hpp>

#include <cassert>

namespace tbblas {

template<class Expression, bool inverse>
struct fftshift_expression {

  typedef fftshift_expression<Expression, inverse> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename Expression::dim_t dim_t;

  static const unsigned dimCount = Expression::dimCount;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef int difference_type;

  struct index_functor : public thrust::unary_function<difference_type,difference_type> {

    dim_t _size;
    dim_t _pitch;
    const unsigned dimension;

    index_functor(const dim_t& size, unsigned dimension) : dimension(dimension) {
      for (int i = 0; i < (int)dimCount; ++i) {
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
      for (int k = 1; k < (int)dimCount; ++k) {
        if (k < (int)dimension)
          index += (((i /= _size[k-1]) + (_size[k]+!inverse) / 2) % _size[k]) * _pitch[k];
        else
          index += (i /= _size[k-1]) * _pitch[k];
      }
      return index;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef thrust::permutation_iterator<typename Expression::const_iterator,TransformIterator> PermutationIterator;
  typedef PermutationIterator const_iterator;

  fftshift_expression(const Expression& expr, unsigned dimension) : expr(expr), dimension(dimension) { }

  inline const_iterator begin() const {
    index_functor functor(size(), dimension);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(expr.begin(), transform);
    return permu;
//    return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(_size, _pitch)));
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return expr.size();
  }

  inline dim_t fullsize() const {
    return expr.fullsize();
  }

  inline size_t count() const {
    return expr.count();
  }

private:
  const Expression& expr;
  const unsigned dimension;
};

template<class T, bool inverse>
struct is_expression<fftshift_expression<T, inverse> > {
  static const bool value = true;
};

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  fftshift_expression<Expression, false>
>::type
fftshift(const Expression& expr, unsigned dimension = Expression::dimCount)
{
  return fftshift_expression<Expression, false>(expr, dimension);
}

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
  fftshift_expression<Expression, true>
>::type
ifftshift(const Expression& expr, unsigned dimension = Expression::dimCount)
{
  return fftshift_expression<Expression, true>(expr, dimension);
}

}

#endif /* TBBLAS_SHIFT_HPP_ */
