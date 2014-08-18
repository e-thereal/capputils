/*
 * identity.hpp
 *
 *  Created on: Aug 16, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_IDENTITY_HPP_
#define TBBLAS_IDENTITY_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T>
struct identity_expression {
  typedef typename tensor<T, 2>::dim_t dim_t;
  typedef T value_t;
  static const unsigned dimCount = 2;
  static const bool cuda_enabled = true;    // can be executed on the device

  typedef int difference_type;

  // Maps an expanded index to a memory index
  struct index_functor : public thrust::unary_function<difference_type,value_t> {

   int size;

   index_functor(int size) : size(size) { }

   __host__ __device__
   value_t operator()(difference_type idx) const {
     return !(idx % (size + 1));
   }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef TransformIterator const_iterator;

  identity_expression(int size) : _size(seq<2>(size)) { }

  inline const_iterator begin() const {
    index_functor functor(_size[0]);
    CountingIterator counting(0);
    TransformIterator transform(counting, functor);
    return transform;
  }

  inline const_iterator end() const {
    return begin() + count();
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= _size[i];
    return count;
  }

private:
  dim_t _size;
};

template<class T>
struct is_expression<identity_expression<T> > {
  static const bool value = true;
};

template<class T>
identity_expression<T> identity(int size) {
  return identity_expression<T>(size);
}

}

#endif /* TBBLAS_IDENTITY_HPP_ */
