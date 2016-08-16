/*
 * real.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_REAL_HPP_
#define TBBLAS_REAL_HPP_

#include <tbblas/type_traits.hpp>
#include <tbblas/assert.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <boost/utility/enable_if.hpp>

#include <tbblas/detail/for_each.hpp>

namespace tbblas {

template<class T>
struct const_real_expression {
  typedef const_real_expression<T> expression_t;
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t complex_t;
  typedef typename complex_t::value_t value_t;
  static const unsigned dimCount = T::dimCount;
  static const bool cuda_enabled = T::cuda_enabled;

  struct get_real : public thrust::unary_function<complex_t, value_t> {
    __host__ __device__
    inline value_t operator()(const complex_t& value) const {
      return value.real;
    }
  };

  typedef thrust::transform_iterator<get_real, typename T::const_iterator> const_iterator;

  const_real_expression(const T& expr) : expr(expr) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), get_real());
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), get_real());
  }

  inline dim_t size() const {
    return expr.size();
  }

  inline dim_t fullsize() const {
    return expr.fullsize();
  }

  inline size_t count() const {
    return expr.count();
  }

private:
  const T& expr;
};

template<class T>
struct real_expression {
  typedef real_expression<T> expression_t;
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t complex_t;
  typedef typename complex_t::value_t value_t;
  static const unsigned dimCount = T::dimCount;
  static const bool cuda_enabled = T::cuda_enabled;

  struct get_real : public thrust::unary_function<complex_t, value_t> {
    __host__ __device__
    inline value_t operator()(const complex_t& value) const {
      return value.real;
    }
  };

  typedef thrust::tuple<complex_t, value_t> tuple_t;

  struct set_real {
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t) const {
      thrust::get<0>(t).real = thrust::get<1>(t);
    }
  };

  typedef thrust::transform_iterator<get_real, typename T::const_iterator> const_iterator;

  real_expression(T& expr) : expr(expr) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), get_real());
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), get_real());
  }

  inline dim_t size() const {
    return expr.size();
  }

  inline dim_t fullsize() const {
    return expr.fullsize();
  }

  inline size_t count() const {
    return expr.count();
  }

  template<class Expression>
  inline const typename boost::enable_if<is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount,
      expression_t
    >::type
  >::type&
  operator=(const Expression& expr) const {
    const dim_t& size = expr.size();
    for (unsigned i = 0; i < dimCount; ++i) {
      tbblas_assert(size[i] == this->size()[i]);
    }
    tbblas::detail::for_each(
        typename tbblas::detail::select_system<cuda_enabled && Expression::cuda_enabled>(),
        thrust::make_zip_iterator(thrust::make_tuple(this->expr.begin(), expr.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(this->expr.end(), expr.end())),
        set_real());
    return *this;
  }

private:
  T& expr;
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  typename boost::enable_if<is_complex<typename Expression::value_t>,
    const_real_expression<Expression>
  >::type
>::type
real(const Expression& expr) {
  return const_real_expression<Expression>(expr);
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  typename boost::enable_if<is_complex<typename Expression::value_t>,
    real_expression<Expression>
  >::type
>::type
real(Expression& expr) {
  return real_expression<Expression>(expr);
}

template<class T>
struct is_expression<const_real_expression<T> > {
  static const bool value = true;
};

template<class T>
struct is_expression<real_expression<T> > {
  static const bool value = true;
};

}

#endif /* TBBLAS_REAL_HPP_ */
