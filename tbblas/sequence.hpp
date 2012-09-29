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

namespace tbblas {

template<class T, unsigned size>
struct sequence {
  typedef T seq_t[size];
  typedef sequence<T, size> sequence_t;

  sequence(const T& value = 0) {
    for (unsigned i = 0; i < size; ++i)
      _seq[i] = value;
  }

  sequence(const seq_t& seq) {
    for (unsigned i = 0; i < size; ++i)
      _seq[i] = seq[i];
  }

  template<class T2>
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

  bool operator==(const sequence_t& seq) const {
    for (unsigned i = 0; i < size; ++i)
      if (_seq[i] != seq[i])
        return false;
    return true;
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

struct POD_sequence {
  unsigned data[20];
};

template<class T>
sequence<T, 1u> seq(T x1) {
  T seq[] = {x1};
  return sequence<T, 1u>(seq);
}

template<class T>
sequence<T, 2u> seq(T x1, T x2) {
  T seq[] = {x1, x2};
  return sequence<T, 2u>(seq);
}

template<class T>
sequence<T, 3u> seq(T x1, T x2, T x3) {
  T seq[] = {x1, x2, x3};
  return sequence<T, 3u>(seq);
}

template<class T>
sequence<T, 4u> seq(T x1, T x2, T x3, T x4) {
  T seq[] = {x1, x2, x3, x4};
  return sequence<T, 4u>(seq);
}

}

template<class T, unsigned size>
std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> > operator,(const tbblas::sequence<T, size>& seq1,
    const tbblas::sequence<T, size>& seq2)
{
  return std::pair<tbblas::sequence<T, size>, tbblas::sequence<T, size> >(seq1, seq2);
}

template<class T, unsigned size>
std::ostream& operator<<(std::ostream& out, const tbblas::sequence<T, size>& seq) {
  out << seq[0];
  for (unsigned i = 1; i < size; ++i)
    out << ", " << seq[i];
  return out;
}

#endif /* TBBLAS_SEQUENCE_HPP_ */
