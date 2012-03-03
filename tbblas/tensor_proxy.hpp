/*
 * device_tensor_proxy.hpp
 *
 *  Created on: Mar 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_TENSOR_PROXY_HPP_
#define TBBLAS_TENSOR_PROXY_HPP_

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

namespace tbblas {

template<class Iterator, unsigned dim>
class tensor_proxy {
public:
  //typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef int difference_type;
  typedef size_t dim_t[dim];

protected:
  dim_t _size;
  dim_t _pitch;

  Iterator first;

public:
  struct subrange_functor : public thrust::unary_function<difference_type,difference_type> {
    dim_t _size;
    dim_t _pitch;

    subrange_functor(const dim_t& size, const dim_t& pitch) {
      for (int i = 0; i < dim; ++i) {
        _size[i] = size[i];
        _pitch[i] = pitch[i];
      }
    }

    __host__ __device__
    difference_type operator()(difference_type i) const
    {
      difference_type index;
      index = (i % _size[0]) * _pitch[0];
      for (int k = 1; k < dim; ++k) {
        index += ((i /= _size[k-1]) % _size[k]) * _pitch[k];
      }
      return index;
    }
  };

  typedef typename thrust::counting_iterator<difference_type>                     CountingIterator;
  typedef typename thrust::transform_iterator<subrange_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>       PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  tensor_proxy(Iterator first, const dim_t& size, const dim_t& pitch)
      : first(first)
  {
    for (int i = 0; i < dim; ++i) {
      _size[i] = size[i];
      _pitch[i] = pitch[i];
    }
  }

  iterator begin(void) const {
    subrange_functor functor(_size, _pitch);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    PermutationIterator permu(first, transform);
    return permu;
//    return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(_size, _pitch)));
  }

  iterator end(void) const {
    difference_type count = 1;
    for (int i = 0; i < dim; ++i)
      count *= _size[i];
    return begin() + count;
  }
};

}


#endif /* TBBLAS_TENSOR_PROXY_HPP_ */
