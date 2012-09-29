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
  typedef cufftComplex type;

  __host__ __device__
  static inline type mult(const type& c1, const type& c2) {
    return cuCmulf(c1, c2);
  }
};

template<>
struct complex_type<double> {
  typedef cufftDoubleComplex type;

  __host__ __device__
  static inline type mult(const type& c1, const type& c2) {
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

/**
 * An expression must define the following interface:
 *
 * typedef ... dim_t;
 * typedef ... value_t;
 * typedef ... const_iterator;
 * static const unsigned dimCount;
 *
 * inline const_iterator begin() const;
 * inline const_iterator end() const;
 * inline const dim_t& size() const;
 * inline size_t count() const;
 */
template<class T>
struct is_expression {
  static const bool value = false;
};

/**
 * A proxy must define the following interface:
 *
 * typedef typename ... value_t;
 * typedef typename ... dim_t;
 * typedef typename ... data_t;
 *
 * static const int dimCount;
 * static const bool cuda_enabled;
 */
template<class T>
struct is_proxy {
  static const bool value = false;
};

/**
 * An operation must define the following interface:
 *
 * typedef ... tensor_t; // type of the tensor to which the operation will be applied
 *
 * void apply(tensor_t& t) const;
 * inline const dim_t& size() const;
 *
 * \remark
 * - is_tensor and is_operation are mutually exclusive
 */
template<class T>
struct is_operation {
  static const bool value = false;
};

/**
 * A tensor must be of type tbblas::tensor<class T, unsigned dim, bool device>
 *
 * \remark
 * - is_tensor and is_operation are mutually exclusive
 */
template<class T>
struct is_tensor {
  static const bool value = false;
};

template<class T>
struct is_complex {
  static const bool value = false;
};

}

#endif /* TYPE_TRAITS_HPP_ */
