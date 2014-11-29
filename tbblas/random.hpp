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
#include <tbblas/context.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

#include <tbblas/detail/for_each.hpp>
#include <tbblas/detail/system.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/static_assert.hpp>

#include <curand_kernel.h>
#include <curand.h>

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

/*** NEW RANDOM TENSOR CODE STARTS HERE ***/

template<class T, unsigned dim, bool device>
struct random_values {
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = device;

  typedef T value_t;
  typedef typename tensor<value_t, dimCount, cuda_enabled>::dim_t dim_t;
  typedef typename vector_type<value_t, cuda_enabled>::vector_t data_t;

  typedef typename data_t::const_iterator const_iterator;

  random_values() {
    resize(seq<dimCount>(0));
  }

  random_values(dim_t size) {
    resize(size);
  }

  void resize(const dim_t& size) {
    size_t count = 1;
    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = size[i];
      count *= size[i];
    }
    _values.resize((count + 1) & ~1);   // round up to the nearest even number
  }

  inline const_iterator begin() const {
    return _values.begin();
  }

  inline const_iterator end() const {
    return _values.begin() + count();
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

  inline size_t count() const {
    size_t count = 1;
    for (unsigned i = 0; i < dim; ++i) {
      count *= _size[i];
    }
    return count;
  }

  inline size_t total_count() const {
    return _values.size();
  }

  inline value_t* raw_pointer() {
    return thrust::raw_pointer_cast(_values.data());
  }

private:
  dim_t _size;
  data_t _values;
};

template<class T, unsigned dim, bool device>
struct is_expression<random_values<T, dim, device> > {
  static const bool value = true;
};

template<class T, unsigned dim, bool device, class Distribution>
struct random_tensor2
{
  typedef T value_t;
  static const unsigned dimCount = dim;
  static const bool cuda_enabled = device;
  typedef typename tensor<value_t, dimCount, cuda_enabled>::dim_t dim_t;
  typedef random_values<T, dim, device> data_t;

  random_tensor2(size_t x1 = 1, size_t x2 = 1, size_t x3 = 1, size_t x4 = 1, size_t x5 = 1)
  {
    if (cuda_enabled)
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    else
      curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    BOOST_STATIC_ASSERT(dimCount < 5);

    size_t input[] = {x1, x2, x3, x4, x5};

    for (unsigned i = 0; i < dimCount; ++i) {
      _size[i] = input[i];
    }

    resize(_size);
    reset(input[dimCount]);
  }

  random_tensor2(const dim_t& size, unsigned seed = 0) {
    if (cuda_enabled)
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    else
      curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    resize(size);
    reset(seed);
  }

  ~random_tensor2() {
    curandDestroyGenerator(gen);
  }

  void resize(const dim_t& size) {
    _values.resize(size);
  }

  void resize(const dim_t& size, unsigned seed) {
    _values.resize(size);
    reset(seed);
  }

  void reset(unsigned seed = 0) {
    curandSetPseudoRandomGeneratorSeed(gen, seed);
  }

  const data_t& operator()(bool redraw = true) {
    if (redraw) {
      if (cuda_enabled)
        curandSetStream(gen, context::get().stream);
      Distribution::generate(gen, _values.raw_pointer(), _values.total_count());
    }
    return _values;
  }

  inline dim_t size() const {
    return _size;
  }

private:
  data_t _values;
  dim_t _size;
  curandGenerator_t gen;
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

  static inline void generate(curandGenerator_t generator, float *outputPtr, size_t num) {
    curandGenerateUniform(generator, outputPtr, num);
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

  static inline void generate(curandGenerator_t generator, double *outputPtr, size_t num) {
    curandGenerateUniformDouble(generator, outputPtr, num);
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

  static inline void generate(curandGenerator_t generator, float *outputPtr, size_t num) {
    curandGenerateNormal(generator, outputPtr, num, 0.0f, 1.0f);
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

  static inline void generate(curandGenerator_t generator, double *outputPtr, size_t num) {
    curandGenerateNormalDouble(generator, outputPtr, num, 0.0, 1.0);
  }
};

}

#endif /* TBBLAS_RANDOM_HPP_ */
