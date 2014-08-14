/*
 * gaussian.hpp
 *
 *  Created on: Nov 28, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_GAUSSIAN_HPP_
#define TBBLAS_GAUSSIAN_HPP_

#include <tbblas/tensor.hpp>

#include <cmath>

namespace tbblas {

template<class T, unsigned dim>
struct gaussian_expression {

  typedef gaussian_expression<T, dim> expression_t;
  typedef typename tensor<T, dim>::dim_t dim_t;
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = true;

  typedef int difference_type;

  struct index_functor : public thrust::unary_function<difference_type,value_t> {

    dim_t size;
    double sigma;
    const value_t factor;

    index_functor(dim_t size, double sigma)
     : size(size), sigma(sigma), factor(1.0 / sigma / ::sqrt(2.0 * M_PI)) { }

    __host__ __device__
    value_t operator()(difference_type idx) const {
      value_t result = 1;
      const value_t x = ((idx % size[0]) + size[0] / 2) % size[0] - size[0] / 2;
      result *= factor * ::exp(-x * x / (2.0 * sigma * sigma));
      for (unsigned k = 1; k < dimCount; ++k) {
        const value_t y = (((idx /= size[k-1]) % size[k]) + size[k] / 2) % size[k] - size[k] / 2;
        result *= factor * ::exp(-y * y / (2.0 * sigma * sigma));
      }
      return result;
    }
  };

  typedef thrust::counting_iterator<difference_type>                  CountingIterator;
  typedef thrust::transform_iterator<index_functor, CountingIterator> TransformIterator;
  typedef TransformIterator const_iterator;

  gaussian_expression(const dim_t& size, const dim_t& fullsize, double sigma)
   : _size(size), _fullsize(fullsize), _sigma(sigma) { }

  inline const_iterator begin() const {
    index_functor functor(_size, _sigma);
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
  dim_t _size, _fullsize;
  double _sigma;
};

template<class T, unsigned dim>
struct is_expression<gaussian_expression<T, dim> > {
  static const bool value = true;
};

template<class T, unsigned dim>
gaussian_expression<T, dim> gaussian(const sequence<int, dim>& size, double sigma) {
  return gaussian_expression<T, dim>(size, size, sigma);
}

template<class T, unsigned dim>
gaussian_expression<T, dim> gaussian(const sequence<int, dim>& size, const sequence<int, dim>& fullsize, double sigma) {
  return gaussian_expression<T, dim>(size, fullsize, sigma);
}

}

#endif /* GAUSSIAN_HPP_ */
