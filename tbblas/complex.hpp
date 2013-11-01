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
struct __builtin_align__(16) complex {
  typedef T value_t;
  typedef typename complex_type<value_t>::type complex_t;

  union {
    struct { value_t real, img; };
    complex_t value;
  };

  __host__ __device__
  complex() : real(0), img(0) { }

  __host__ __device__
  complex(const complex_t& value) : value(value) { }

  __host__ __device__
  complex(value_t real) : real(real), img(0) { }

  __host__ __device__
  complex(value_t real, value_t img) : real(real), img(img) { }

  __host__ __device__
  operator complex_t() {
    return value;
  }

  __host__ __device__
  complex<value_t> operator*(const complex<value_t>& x) const {
    return complex<value_t>(tbblas::complex_type<value_t>::mult(x.value, value));
  }

  __host__ __device__
  complex<value_t> operator+(const complex<value_t>& x) const {
    complex<value_t> ret;
    ret.real = real + x.real;
    ret.img = img + x.img;
    return ret;
  }

  __host__ __device__
  complex<value_t> operator-(const complex<value_t>& x) const {
    complex<value_t> ret;
    ret.real = real - x.real;
    ret.img = img - x.img;
    return ret;
  }

  __host__ __device__
  complex<value_t> operator+=(const complex<value_t>& x) {
    real += x.real;
    img += x.img;
    return *this;
  }
};

template<>
struct
#if !defined(_WIN32) || defined(_WIN64)
  __builtin_align__(8)
#endif
  complex<float> {
  typedef float value_t;
  typedef complex_type<value_t>::type complex_t;

  union {
    struct { value_t real, img; };
    complex_t value;
  };

  __host__ __device__
  complex() : real(0), img(0) { }

  __host__ __device__
  complex(const complex_t& value) : value(value) { }

  __host__ __device__
  complex(value_t real) : real(real), img(0) { }

  __host__ __device__
  complex(value_t real, value_t img) : real(real), img(img) { }

  __host__ __device__
  operator complex_t() {
    return value;
  }

  __host__ __device__
  complex<value_t> operator*(const complex<value_t>& x) const {
    return complex<value_t>(tbblas::complex_type<value_t>::mult(x.value, value));
  }

  __host__ __device__
  complex<value_t> operator+(const complex<value_t>& x) const {
    complex<value_t> ret;
    ret.real = real + x.real;
    ret.img = img + x.img;
    return ret;
  }

  __host__ __device__
  complex<value_t> operator-(const complex<value_t>& x) const {
    complex<value_t> ret;
    ret.real = real - x.real;
    ret.img = img - x.img;
    return ret;
  }

  __host__ __device__
  complex<value_t> operator+=(const complex<value_t>& x) {
    real += x.real;
    img += x.img;
    return *this;
  }
};

template<class T>
struct is_complex<complex<T> > {
  static const bool value = true;
};

template<class T>
__host__ __device__
T abs(const complex<T>& x) {
  return ::sqrt(x.real * x.real + x.img * x.img);
}

}

template<class T>
std::ostream& operator<<(std::ostream& out, const tbblas::complex<T>& c) {
  out << "(" << c.real << ", " << c.img << ")";
  return out;
}

#include <tbblas/real.hpp>
#include <tbblas/img.hpp>

#endif /* TBBLAS_COMPLEX_HPP_ */
