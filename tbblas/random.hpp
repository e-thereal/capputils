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
//#include <thrust/transform.h>

#include <tbblas/detail/for_each.hpp>
#include <tbblas/detail/system.hpp>

#include <boost/utility/enable_if.hpp>

#include <curand_kernel.h>
#include <boost/static_assert.hpp>

#ifndef __CUDACC__
#include <boost/math/special_functions/erf.hpp>
double erfcinv(double x) {
  return boost::math::erfc_inv(x);
}
#endif

#include <thrust/random.h>

namespace tbblas {

template<class T, unsigned dim, bool device, class Distribution>
struct random_tensor
{
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = device;
  typedef typename tensor<value_t, dimCount, cuda_enabled>::dim_t dim_t;

  typedef curandState generator_t;
  typedef typename vector_type<generator_t, cuda_enabled>::vector_t generators_t;

  typedef thrust::tuple<unsigned, generator_t> init_tuple_t;

  struct init_generator {

    init_generator(unsigned seed) : seed(seed) { }

    template<class Tuple>
    __device__
    void operator()(Tuple t) {
      generator_t gen = thrust::get<1>(t);
//      curand_init (seed, thrust::get<0>(t), 0, &gen);
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

  random_tensor(const dim_t& size, unsigned seed = 0) {
    resize(size, seed);
  }

  void resize(const dim_t& size, unsigned seed = 0) {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    generators = boost::shared_ptr<generators_t>(new generators_t(count));
    reset(seed);
  }

  void reset(unsigned seed = 0) {
    const int maxCount = 0xFFFF;
    int count = this->count(), first = 0, last = 0;

    while (count > 0) {
      first = last;
      last = first + std::min(count, maxCount);
      count -= std::min(count, maxCount);
      tbblas::detail::for_each(
          typename tbblas::detail::select_system<cuda_enabled>::system(),
          thrust::make_zip_iterator(thrust::make_tuple(
              thrust::counting_iterator<unsigned>(first),
              generators->begin() + first)),
          thrust::make_zip_iterator(thrust::make_tuple(
              thrust::counting_iterator<unsigned>(last),
              generators->begin() + last)),
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

  inline dim_t fullsize() const {
    return _size;
  }

  inline size_t count() const {
    return generators->size();
  }

private:
  dim_t _size;
  boost::shared_ptr<generators_t> generators;
};

template<class T, unsigned dim, class Distribution>
struct random_tensor<T, dim, false, Distribution>
{
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = false;
  typedef typename tensor<value_t, dimCount, cuda_enabled>::dim_t dim_t;

  typedef thrust::random::default_random_engine generator_t;

  struct get_random : thrust::unary_function<const generator_t*, value_t> {

//    __host__
    value_t operator()(const generator_t* gen) const {
      return Distribution::rand(const_cast<generator_t*>(gen));
    }
  };

  typedef thrust::transform_iterator<get_random, thrust::constant_iterator<const generator_t*> > const_iterator;

  random_tensor(size_t x1 = 1, size_t x2 = 1, size_t x3 = 1, size_t x4 = 1, size_t x5 = 1) {
    BOOST_STATIC_ASSERT(dimCount < 5);

    size_t size[] = {x1, x2, x3, x4, x5};

    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
    }
    reset(size[dimCount]);
  }

  random_tensor(const dim_t& size, unsigned seed = 0) : _size(size) {
    reset(seed);
  }

  void reset(unsigned seed = 0) {
    srand(seed);
  }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        thrust::make_constant_iterator(&generator), get_random());
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
    for (unsigned i = 0; i < dimCount; ++i) {
      count *= _size[i];
    }
    return count;
  }

private:
  dim_t _size;
  generator_t generator;
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
  typedef thrust::random::default_random_engine generator_t;

  __device__
  static inline float rand(curandState* state) {
    return curand_uniform(state);
  }

//  __host__
  static inline float rand(generator_t* gen) {
    thrust::random::uniform_real_distribution<float> dist;
    return dist(*gen);
  }
};

template<>
struct uniform<double> {
  typedef thrust::random::default_random_engine generator_t;

  __device__
  static inline double rand(curandState* state) {
    return curand_uniform_double(state);
  }

//  __host__
  static inline double rand(generator_t* gen) {
    thrust::random::uniform_real_distribution<double> dist;
    return dist(*gen);
  }
};

template<class T>
struct normal { };

template<>
struct normal<float> {

  typedef thrust::random::default_random_engine generator_t;

  __device__
  static inline float rand(curandState* state) {
    return curand_normal(state);
  }

//  __host__
  static inline float rand(generator_t* gen) {
    thrust::random::normal_distribution<float> dist;
    return dist(*gen);
  }
};

template<>
struct normal<double> {
  typedef thrust::random::default_random_engine generator_t;

  __device__
  static inline double rand(curandState* state) {
    return curand_normal_double(state);
  }

//  __host__
  static inline double rand(generator_t* gen) {
    thrust::random::normal_distribution<double> dist;
    return dist(*gen);
  }
};

}

#endif /* TBBLAS_RANDOM_HPP_ */
