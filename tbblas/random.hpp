/*
 * random.hpp
 *
 *  Created on: Sep 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_RANDOM_HPP_
#define TBBLAS_RANDOM_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <boost/utility/enable_if.hpp>

#include <curand_kernel.h>
#include <boost/static_assert.hpp>

namespace tbblas {

template<class T, unsigned dim, bool device, class Distribution>
struct random_tensor : boost::enable_if_c<device == true>
{
  typedef typename tensor<T, dim, device>::dim_t dim_t;
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = device;

  typedef curandState generator_t;
  typedef typename vector_type<generator_t, device>::vector_t generators_t;

  typedef thrust::tuple<unsigned, generator_t> init_tuple_t;

  struct init_generator {

    init_generator(unsigned seed) : seed(seed) { }

    template<class Tuple>
    __device__
    void operator()(Tuple t) {
      generator_t gen = thrust::get<1>(t);
      //curand_init (seed, thrust::get<0>(t), 0, &gen);
      curand_init (thrust::get<0>(t) + seed, thrust::get<0>(t) & 0xFF, 0, &gen);
      thrust::get<1>(t) = gen;
    }

  private:
    unsigned seed;
  };

  struct get_random : thrust::unary_function<generator_t, value_t> {

    __device__
    value_t operator()(const generator_t& gen) const {
      return Distribution::rand(const_cast<generator_t*>(&gen));
    }
  };

  typedef thrust::transform_iterator<get_random, typename generators_t::const_iterator> const_iterator;

  random_tensor(size_t x1 = 1, size_t x2 = 1, size_t x3 = 1, size_t x4 = 1, size_t x5 = 1)
  {
    BOOST_STATIC_ASSERT(dimCount < 5);

    size_t size[] = {x1, x2, x3, x4, x5};

    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    generators = boost::shared_ptr<generators_t>(new generators_t(count));
    reset(size[dimCount]);
  }

  random_tensor(const dim_t& size, unsigned seed = 0)
  {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    generators = boost::shared_ptr<generators_t>(new generators_t(count));
    reset(seed);
  }

  void reset(unsigned seed = 0) {
    const size_t maxCount = 0xFFFF;
    size_t count = this->count(), first = 0, last = 0;

    while (count) {
      first = last;
      last = first + std::min(count, maxCount);
      count -= std::min(count, maxCount);
      thrust::for_each(
          thrust::make_zip_iterator(thrust::make_tuple(
              thrust::counting_iterator<unsigned>(first),
              generators->begin())),
          thrust::make_zip_iterator(thrust::make_tuple(
              thrust::counting_iterator<unsigned>(last),
              generators->end())),
          init_generator(seed)
      );
    }
  }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(generators->begin(), get_random());
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(generators->end(), get_random());
  }

  inline dim_t size() const {
    return _size;
  }

  inline size_t count() const {
    return generators->size();
  }

private:
  dim_t _size;
  boost::shared_ptr<generators_t> generators;
};

template<class T, unsigned dim, bool device, class Distribution>
struct is_expression<random_tensor<T, dim, device, Distribution> > {
  static const bool value = true;
};

/*** Distributions ***/

template<class T>
struct uniform { };

template<>
struct uniform<float> {
  __device__
  static inline float rand(curandState* state) {
    return curand_uniform(state);
  }
};

template<>
struct uniform<double> {
  __device__
  static inline double rand(curandState* state) {
    return curand_uniform_double(state);
  }
};

template<class T>
struct normal { };

template<>
struct normal<float> {
  __device__
  static inline float rand(curandState* state) {
    return curand_normal(state);
  }
};

template<>
struct normal<double> {
  __device__
  static inline double rand(curandState* state) {
    return curand_normal_double(state);
  }
};

}

#endif /* TBBLAS_RANDOM_HPP_ */
