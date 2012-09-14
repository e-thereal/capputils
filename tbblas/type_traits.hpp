/*
 * type_traits.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TYPE_TRAITS_HPP_
#define TYPE_TRAITS_HPP_

#include <cufft.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace tbblas {

template<class T>
struct complex_type {
};

template<>
struct complex_type<float> {
  typedef cufftComplex complex_t;

  __host__ __device__
  static inline complex_t mult(const complex_t& c1, const complex_t& c2) {
    return cuCmulf(c1, c2);
  }
};

template<>
struct complex_type<double> {
  typedef cufftDoubleComplex complex_t;

  __host__ __device__
  static inline complex_t mult(const complex_t& c1, const complex_t& c2) {
    return cuCmul(c1, c2);
  }
};

template<class T, bool device = false>
struct vector_type {
  typedef thrust::host_vector<T> vector_t;
};

template<class T>
struct vector_type<T, true> {
  typedef thrust::device_vector<T> vector_t;
};

template<class T>
struct is_expression {
  static const bool value = false;
};

template<class T>
struct is_operation {
  static const bool value = false;
};

template<class T>
struct is_tensor {
  static const bool value = false;
};

}


#endif /* TYPE_TRAITS_HPP_ */
