/*
 * seq.hpp
 *
 *  Created on: Sep 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SEQUENCE_HPP_
#define TBBLAS_SEQUENCE_HPP_

#include <utility>
#include <iostream>
#include <boost/utility/enable_if.hpp>

#include <cmath>

namespace tbblas {

template<class T, unsigned size>
struct dim_type_trait {
};

template<class T, unsigned size>
struct sequence {
public:
  typedef T value_t;
  typedef T seq_t[size];
  typedef sequence<T, size> sequence_t;

  static const unsigned dimCount = size;

  __host__ __device__
  sequence() {
    for (unsigned i = 0; i < size; ++i)
      _seq[i] = 0;
  }

  __host__ __device__
  sequence(const seq_t& seq) {
    for (unsigned i = 0; i < size; ++i)
      _seq[i] = seq[i];
  }

  template<class T2>
  __host__ __device__
  sequence(const sequence<T2, size>& seq) {
    for (unsigned i = 0; i < size; ++i)
      _seq[i] = seq[i];
  }

  __host__ __device__
  seq_t& get() {
    return _seq;
  }

  __host__ __device__
  const seq_t& get() const {
    return _seq;
  }

  __host__ __device__
  T& operator[](int i) {
    return _seq[i];
  }

  __host__ __device__
  const T& operator[](int i) const {
    return _seq[i];
  }

  __host__ __device__
  const seq_t& operator()() const {
    return _seq;
  }

  // TODO: this function is obsolete and prod() should be used instead.
  __host__ __device__
  T count() const {
    T result = 1;
    for (unsigned i = 0; i < size; ++i)
      result *= _seq[i];
    return result;
  }

  /*** Scalar arithmetic operators ***/

  __host__ __device__
  sequence_t operator*(const value_t& x) const {
    sequence_t result;
    for (unsigned i = 0; i < size; ++i)
      result[i] = _seq[i] * x;
    return result;
  }

  /*** Binary arithmetic operators ***/

  __host__ __device__
  sequence_t operator+(const sequence_t& seq) const {
    sequence_t result;
    for (unsigned i = 0; i < size; ++i)
      result[i] = _seq[i] + seq[i];
    return result;
  }

  __host__ __device__
  sequence_t operator-(const sequence_t& seq) const {
    sequence_t result;
    for (unsigned i = 0; i < size; ++i)
      result[i] = _seq[i] - seq[i];
    return result;
  }

  __host__ __device__
  sequence_t operator*(const sequence_t& seq) const {
    sequence_t result;
    for (unsigned i = 0; i < size; ++i)
      result[i] = _seq[i] * seq[i];
    return result;
  }

  __host__ __device__
  sequence_t operator/(const sequence_t& seq) const {
    sequence_t result;
    for (unsigned i = 0; i < size; ++i)
      result[i] = _seq[i] / seq[i];
    return result;
  }

  __host__ __device__
  sequence_t operator%(const sequence_t& seq) const {
    sequence_t result;
    for (unsigned i = 0; i < size; ++i)
      result[i] = _seq[i] % seq[i];
    return result;
  }

  /*** Logic operators ***/

//  sequence_t operator||(const sequence_t& seq) const {
//    sequence_t result;
//    for (unsigned i = 0; i < size; ++i)
//      result[i] = _seq[i] || seq[i];
//    return result;
//  }
//
//  sequence_t operator&&(const sequence_t& seq) const {
//    sequence_t result;
//    for (unsigned i = 0; i < size; ++i)
//      result[i] = _seq[i] && seq[i];
//    return result;
//  }

  /*** Comparison operators ***/

  __host__ __device__
  bool operator<(const sequence_t& seq) const {
    for (unsigned i = 0; i < size; ++i)
      if (_seq[i] < seq[i])
        return true;
      else if (_seq[i] > seq[i])
        return false;
    return false;
  }

  __host__ __device__
  bool operator>(const sequence_t& seq) const {
    for (unsigned i = 0; i < size; ++i)
      if (_seq[i] > seq[i])
        return true;
      else if (_seq[i] < seq[i])
        return false;
    return false;
  }

  __host__ __device__
  bool operator==(const sequence_t& seq) const {
    for (unsigned i = 0; i < size; ++i)
      if (_seq[i] != seq[i])
        return false;
    return true;
  }

  __host__ __device__
  bool operator!=(const sequence_t& seq) const {
    for (unsigned i = 0; i < size; ++i)
      if (_seq[i] != seq[i])
        return true;
    return false;
  }

  operator typename boost::enable_if_c<size <= 4, dim3>::type() const {
    return dim_type_trait<T, size>::convert(*this);
  }

  __host__ __device__
  value_t sum() const {
    value_t sum = 0;
    for (unsigned i = 0; i < size; ++i)
      sum += _seq[i];
    return sum;
  }

  __host__ __device__
  value_t prod() const {
    value_t product = 1;
    for (unsigned i = 0; i < size; ++i)
      product *= _seq[i];
    return product;
  }

//  template<class T2>
//  operator sequence<T2, size>() const {
//    typename sequence<T2, size>::seq_t seq;
//    for (unsigned i = 0; i < size; ++i)
//      seq[i] = _seq[i];
//    return sequence<T2, size>(seq);
//  }

private:
  seq_t _seq;
};

template<class T>
struct is_sequence {
  static const bool value = false;
};

template<class T, unsigned size>
struct is_sequence<tbblas::sequence<T, size> > {
  static const bool value = false;
};

template<class T>
struct dim_type_trait<T, 1> {
  static dim3 convert(const sequence<T, 1>& s) {
    return dim3(s[0]);
  }
};

template<class T>
struct dim_type_trait<T, 2> {
  static dim3 convert(const sequence<T, 2>& s) {
    return dim3(s[0], s[1]);
  }
};

template<class T>
struct dim_type_trait<T, 3> {
  static dim3 convert(const sequence<T, 3>& s) {
    return dim3(s[0], s[1], s[2]);
  }
};

template<class T>
struct dim_type_trait<T, 4> {
  static dim3 convert(const sequence<T, 4>& s) {
    return dim3(s[0], s[1], s[2]);
  }
};

template<unsigned dims, class T>
__host__ __device__
sequence<T, dims> seq(const T& x) {
  sequence<T, dims> s;
  for (unsigned i = 0; i < dims; ++i)
    s[i] = x;
  return s;
}

template<class T>
__host__ __device__
sequence<T, 1u> seq(T x1) {
  T seq[] = {x1};
  return sequence<T, 1u>(seq);
}

template<class T>
__host__ __device__
sequence<T, 2u> seq(T x1, T x2) {
  T seq[] = {x1, x2};
  return sequence<T, 2u>(seq);
}

template<class T>
__host__ __device__
sequence<T, 3u> seq(T x1, T x2, T x3) {
  T seq[] = {x1, x2, x3};
  return sequence<T, 3u>(seq);
}

template<class T>
__host__ __device__
sequence<T, 4u> seq(T x1, T x2, T x3, T x4) {
  T seq[] = {x1, x2, x3, x4};
  return sequence<T, 4u>(seq);
}

/*** Useful functions ***/

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> abs(const tbblas::sequence<T, size>& seq)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = ::abs(seq[i]);
  return result;
}

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> max(const tbblas::sequence<T, size>& seq1,
    const tbblas::sequence<T, size>& seq2)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = (seq1[i] > seq2[i] ? seq1[i] : seq2[i]);
  return result;
}

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> min(const tbblas::sequence<T, size>& seq1,
    const tbblas::sequence<T, size>& seq2)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = (seq1[i] < seq2[i] ? seq1[i] : seq2[i]);
  return result;
}

/*** Pair creation ***/

template<class T, unsigned size>
std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> > operator,(const tbblas::sequence<T, size>& seq1,
    const tbblas::sequence<T, size>& seq2)
{
  return std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> >(seq1, seq2);
}

template<class T, unsigned size>
std::pair<std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> >, tbblas::sequence<T, size> > operator,(const std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> >& pair,
    const tbblas::sequence<T, size>& seq)
{
  return std::pair<std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> >, tbblas::sequence<T, size> >(pair, seq);
}

/*** Input/output ***/

template<class T, unsigned size>
std::ostream& operator<<(std::ostream& out, const tbblas::sequence<T, size>& seq) {
  out << seq[0];
  for (unsigned i = 1; i < size; ++i)
    out << " " << seq[i];
  return out;
}

template<class T, unsigned size>
std::istream& operator>>(std::istream& in, tbblas::sequence<T, size>& seq) {
  in >> seq[0];
  for (unsigned i = 1; i < size; ++i) {
//    in.ignore(2);
    in >> seq[i];
  }
  return in;
}

/*** Scalar operations ***/

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> operator+(const tbblas::sequence<T, size>& seq,
    const T& value)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = seq[i] + value;
  return result;
}

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> operator-(const tbblas::sequence<T, size>& seq,
    const T& value)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = seq[i] - value;
  return result;
}

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> operator*(const T& value, const tbblas::sequence<T, size>& seq)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = seq[i] * value;
  return result;
}

template<class T, unsigned size>
__host__ __device__
tbblas::sequence<T, size> operator/(const tbblas::sequence<T, size>& seq,
    const T& value)
{
  tbblas::sequence<T, size> result;
  for (unsigned i = 0; i < size; ++i)
    result[i] = seq[i] / value;
  return result;
}

}

#endif /* TBBLAS_SEQUENCE_HPP_ */
