/*
 * sequence_iterator.hpp
 *
 *  Created on: Dec 4, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_SEQUENCE_ITERATOR_HPP_
#define TBBLAS_SEQUENCE_ITERATOR_HPP_

#include <tbblas/sequence.hpp>

namespace tbblas {

template<class T>
class sequence_iterator {

  typedef T sequence_t;
  typedef typename T::value_t value_t;
  static const unsigned dimCount = T::dimCount;

public:

  __host__ __device__
  sequence_iterator(const sequence_t& start, const sequence_t& size) : _current(start), _start(start), _size(size) { }

  __host__ __device__
  size_t count() {
    return _size.prod();
  }

  __host__ __device__
  size_t current() {
    size_t curr = _current[dimCount - 1] - _start[dimCount - 1];

    for (int i = dimCount - 2; i >= 0; --i) {
      curr = curr * _size[i] + _current[i] - _start[i];
    }

    return curr;
  }

  __host__ __device__
  sequence_iterator<T> operator++() {
    ++_current[0];
    for (size_t i = 0; i < dimCount - 1; ++i) {
      if (_current[i] >= _start[i] + _size[i]) {
        _current[i] = _start[i];
        ++_current[i + 1];
      } else {
        break;
      }
    }

    return *this;
  }

  __host__ __device__
  sequence_iterator<T> operator++(int) {
    sequence_iterator<T> old = *this;

    ++_current[0];
    for (size_t i = 0; i < dimCount - 1; ++i) {
      if (_current[i] >= _start[i] + _size[i]) {
        _current[i] = _start[i];
        ++_current[i + 1];
      } else {
        break;
      }
    }

    return old;
  }

  __host__ __device__
  operator bool() const {
    return valid();
  }

  __host__ __device__
  bool valid() const {
    for (size_t i = 0; i < dimCount; ++i) {
      if (_current[i] >= _start[i] + _size[i])
        return false;
    }
    return true;
  }

  __host__ __device__
  sequence_t& operator*() {
    return _current;
  }

private:
  sequence_t _current, _start, _size;
};

}

#endif /* TBBLAS_SEQUENCE_ITERATOR_HPP_ */
