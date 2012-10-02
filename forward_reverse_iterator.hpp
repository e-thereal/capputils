/*
 * forward_reverse_iterator.hpp
 *
 *  Created on: Feb 27, 2012
 *      Author: tombr
 */
#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

#include "forward_reverse_iterator_base.hpp"

namespace tbblas {

template<typename BidirectionalIterator>
class forward_reverse_iterator
    : public detail::forward_reverse_iterator_base<BidirectionalIterator>::type
{
private:
  typedef typename tbblas::detail::forward_reverse_iterator_base<BidirectionalIterator>::type super_t;

  friend class thrust::experimental::iterator_core_access;

  bool _reverse;

public:
  /*! Default constructor does nothing.
   */
  __host__ __device__
  forward_reverse_iterator(void) : _reverse(false) {}

  /*! \p Constructor accepts a \c BidirectionalIterator pointing to a range
   *  for this \p reverse_iterator to reverse.
   *
   *  \param x A \c BidirectionalIterator pointing to a range to reverse.
   */
  __host__ __device__
  explicit forward_reverse_iterator(BidirectionalIterator x, bool reverse = false);

  /*! \p Copy constructor allows construction from a related compatible
   *  \p reverse_iterator.
   *
   *  \param r A \p reverse_iterator to copy from.
   */
  template<typename OtherBidirectionalIterator>
  __host__ __device__
  forward_reverse_iterator(forward_reverse_iterator<OtherBidirectionalIterator> const &r
// XXX msvc screws this up
// XXX remove these guards when we have static_assert
#ifndef _MSC_VER
                   , typename thrust::detail::enable_if<
                       thrust::detail::is_convertible<
                         OtherBidirectionalIterator,
                         BidirectionalIterator
                       >::value
                     >::type * = 0
#endif // _MSC_VER
                   );

  __host__ __device__
  bool reverse() const { return _reverse; }

/*! \cond
 */
private:
  typename super_t::reference dereference(void) const;

  __host__ __device__
  void increment(void);

  __host__ __device__
  void decrement(void);

  __host__ __device__
  void advance(typename super_t::difference_type n);

  template<typename OtherBidirectionalIterator>
  __host__ __device__
  typename super_t::difference_type
  distance_to(forward_reverse_iterator<OtherBidirectionalIterator> const &y) const;
  /*! \endcond
   */
}; // end reverse_iterator


/*! \p make_reverse_iterator creates a \p reverse_iterator
 *  from a \c BidirectionalIterator pointing to a range of elements to reverse.
 *
 *  \param x A \c BidirectionalIterator pointing to a range to reverse.
 *  \return A new \p reverse_iterator which reverses the range \p x.
 */
template<typename BidirectionalIterator>
__host__ __device__
forward_reverse_iterator<BidirectionalIterator> make_forward_reverse_iterator(
    BidirectionalIterator x, bool reverse = false);

} // end tbblas

#include "forward_reverse_iterator_template.cpp"
