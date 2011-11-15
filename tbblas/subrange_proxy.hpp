#pragma once
#ifndef _TBBLAS_SUBRANGEPROXY_HPP_
#define _TBBLAS_SUBRANGEPROXY_HPP_

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>

namespace tbblas {

template <typename Iterator>
class subrange_proxy {
public:
  //typedef typename thrust::iterator_difference<Iterator>::type difference_type;
  typedef int difference_type;

protected:
  difference_type width, height, pitch;
  Iterator first;

public:
  struct subrange_functor : public thrust::unary_function<difference_type,difference_type> {
    difference_type width, pitch;

    subrange_functor(difference_type width, difference_type pitch)
        : width(width), pitch(pitch) {}

    __host__ __device__
    difference_type operator()(const difference_type& i) const
    { 
        return i + (i / width) * (pitch - width);
    }
  };

  typedef typename thrust::counting_iterator<difference_type>                     CountingIterator;
  typedef typename thrust::transform_iterator<subrange_functor, CountingIterator> TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator,TransformIterator>       PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  subrange_proxy(Iterator first, difference_type width, difference_type height, difference_type pitch)
      : first(first), width(width), height(height), pitch(pitch) {}
   
  iterator begin(void) const
  {
      return PermutationIterator(first, TransformIterator(CountingIterator(0), subrange_functor(width, pitch)));
  }

  iterator end(void) const
  {
      return begin() + width * height;
  }
};

}

#endif /*  _TBBLAS_SUBRANGEPROXY_HPP_ */