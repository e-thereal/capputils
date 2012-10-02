/*
 * complex.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_COMPLEX_HPP_
#define TBBLAS_COMPLEX_HPP_

#include <tbblas/type_traits.hpp>
#include <ostream>

namespace tbblas {

template<class T>
struct complex {
  typedef typename complex_type<T>::type complex_t;
  typedef T value_t;

  union {
    struct { T real, img; };
    complex_t value;
  };

  __host__ __device__
  complex() : real(0), img(0) { }

  __host__ __device__
  complex(const complex_t& value) : value(value) { }

  __host__ __device__
  complex(float real, float img) : real(real), img(img) { }

  __host__ __device__
  operator complex_t() {
    return value;
  }

  __host__ __device__
  complex<T> operator*(const complex<T>& x) const {
    return complex<T>(tbblas::complex_type<T>::mult(x.value, value));
  }
};

template<class T>
struct is_complex<complex<T> > {
  static const bool value = true;
};

}

template<class T>
std::ostream& operator<<(std::ostream& out, const tbblas::complex<T>& c) {
  out << "(" << c.real << ", " << c.img << ")";
  return out;
}

#include <tbblas/real.hpp>
#include <tbblas/img.hpp>

#endif /* TBBLAS_COMPLEX_HPP_ */
