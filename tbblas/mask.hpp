/*
 * mask.hpp
 *
 *  Created on: Nov 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_MASK_HPP_
#define TBBLAS_MASK_HPP_

#include <tbblas/tensor.hpp>

namespace tbblas {

template<class T, unsigned dim>
struct mask_expression {

  typedef mask_expression<T, dim> expression_t;
  typedef typename tensor<T, dim>::dim_t dim_t;
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = tensor<T, dim>::cuda_enabled;

  typedef int difference_type;

  struct index_functor : public thrust::unary_function<difference_type,value_t> {

    dim_t size, maskSize;

    index_functor(dim_t size, dim_t maskSize)
     : size(size), maskSize(maskSize) { }

    __host__ __device__
    value_t operator()(difference_type idx) const {
      value_t result = 1;
      result = result * (((idx + maskSize[0] / 2) % size[0]) < maskSize[0]);
      for (unsigned k = 1; k < dimCount; ++k)
        result = result * ((((idx /= size[k-1]) + maskSize[k] / 2) % size[k]) < maskSize[k]);
      return result;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef TransformIterator const_iterator;

  mask_expression(const dim_t& size, const dim_t& fullsize, const dim_t& maskSize)
   : _size(size), _fullsize(fullsize), _maskSize(maskSize) { }

  inline const_iterator begin() const {
    index_functor functor(_size, _maskSize);
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
    return _fullsize;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i)
      count *= _size[i];
    return count;
  }

private:
  dim_t _size, _fullsize, _maskSize;
};

template<class T, unsigned dim>
struct is_expression<mask_expression<T, dim> > {
  static const bool value = true;
};

template<class T, unsigned dim>
mask_expression<T, dim> mask(const sequence<int, dim>& size, const sequence<int, dim>& maskSize) {
  return mask_expression<T, dim>(size, size, maskSize);
}

template<class T, unsigned dim>
mask_expression<T, dim> mask(const sequence<int, dim>& size,
    const sequence<int, dim>& fullsize, const sequence<int, dim>& maskSize)
{
  return mask_expression<T, dim>(size, fullsize, maskSize);
}

}

#endif /* TBBLAS_MASK_HPP_ */
