/*
 *  Copyright 2008-2011 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "forward_reverse_iterator.hpp"

#include <thrust/detail/backend/dereference.h>
#include <thrust/iterator/iterator_traits.h>

namespace tbblas
{

namespace detail
{

/// returns a new decremented iterator
template<typename Iterator>
__host__ __device__
  Iterator prior(Iterator x)
{
  return --x;
} // end prior()

} // end detail

template<typename BidirectionalIterator>
forward_reverse_iterator<BidirectionalIterator>::forward_reverse_iterator(
    BidirectionalIterator x, bool reverse)
 :super_t(x), _reverse(reverse)
{ }

template<typename BidirectionalIterator>
template<typename OtherBidirectionalIterator>
forward_reverse_iterator<BidirectionalIterator>::forward_reverse_iterator(
    forward_reverse_iterator<OtherBidirectionalIterator> const &r
// XXX msvc screws this up
#ifndef _MSC_VER
                     , typename thrust::detail::enable_if<
                         thrust::detail::is_convertible<
                           OtherBidirectionalIterator,
                           BidirectionalIterator
                         >::value
                       >::type *
#endif // _MSC_VER
    )
 : super_t(r.base()), _reverse(r.reverse())
{ }

template<typename BidirectionalIterator>
typename forward_reverse_iterator<BidirectionalIterator>::super_t::reference
forward_reverse_iterator<BidirectionalIterator>::dereference(void) const
{
  if (_reverse)
    return *tbblas::detail::prior(this->base());
  else
    return *this->base();
}

template<typename BidirectionalIterator>
void forward_reverse_iterator<BidirectionalIterator>::increment(void)
{
  if (_reverse)
    --this->base_reference();
  else
    ++this->base_reference();
}

template<typename BidirectionalIterator>
void forward_reverse_iterator<BidirectionalIterator>::decrement(void)
{
  if (_reverse)
    ++this->base_reference();
  else
    --this->base_reference();
}

template<typename BidirectionalIterator>
void forward_reverse_iterator<BidirectionalIterator>::advance(typename super_t::difference_type n)
{
  if (_reverse)
    this->base_reference() += -n;
  else
    this->base_reference() += n;
}

template<typename BidirectionalIterator>
template<typename OtherBidirectionalIterator>
typename forward_reverse_iterator<BidirectionalIterator>::super_t::difference_type
forward_reverse_iterator<BidirectionalIterator>::distance_to(
    forward_reverse_iterator<OtherBidirectionalIterator> const &y) const
{
  if (_reverse)
    return this->base_reference() - y.base();
  else
    return y.base() - this->base_reference();
}

template<typename BidirectionalIterator>
__host__ __device__
forward_reverse_iterator<BidirectionalIterator> make_forward_reverse_iterator(
    BidirectionalIterator x, bool reverse)
{
  return forward_reverse_iterator<BidirectionalIterator>(x, reverse);
}

} // end tbblas

namespace thrust
{

namespace detail
{

namespace backend
{

template<typename DeviceBidirectionalIterator>
struct dereference_result<tbblas::forward_reverse_iterator<DeviceBidirectionalIterator> >
    : dereference_result<DeviceBidirectionalIterator>
{
};

template<typename BidirectionalIterator>
inline __host__ __device__
typename dereference_result<tbblas::forward_reverse_iterator<BidirectionalIterator> >::type
dereference(const tbblas::forward_reverse_iterator<BidirectionalIterator> &iter)
{
  // TODO: only if reverse
  if (iter.reverse())
    return dereference(tbblas::detail::prior(iter.base()));
  else
    return dereference(iter.base());
}

template<typename BidirectionalIterator, typename IndexType>
inline __host__ __device__
typename dereference_result<tbblas::forward_reverse_iterator<BidirectionalIterator> >::type
dereference(const tbblas::forward_reverse_iterator<BidirectionalIterator> &iter, IndexType n)
{
  tbblas::forward_reverse_iterator<BidirectionalIterator> temp = iter;
  temp += n;
  return dereference(temp);
}

} // end device

} // end detail

} // thrust
