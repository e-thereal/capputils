/*
 * forward_reverse_iterator_base.hpp
 *
 *  Created on: Feb 27, 2012
 *      Author: tombr
 */

#pragma once

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>

namespace tbblas
{

template <typename> class forward_reverse_iterator;

namespace detail
{

template<typename BidirectionalIterator>
  struct forward_reverse_iterator_base
{
  typedef thrust::experimental::iterator_adaptor<
    tbblas::forward_reverse_iterator<BidirectionalIterator>,
    BidirectionalIterator,
    typename thrust::iterator_pointer<BidirectionalIterator>::type,
    typename thrust::iterator_value<BidirectionalIterator>::type,
    typename thrust::iterator_space<BidirectionalIterator>::type,
    typename thrust::iterator_traversal<BidirectionalIterator>::type,
    typename thrust::iterator_reference<BidirectionalIterator>::type
  > type;
}; // end forward_reverse_iterator_base

} // end detail

} // end tbblas
