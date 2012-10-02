/*
 * img.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_IMG_HPP_
#define TBBLAS_IMG_HPP_

#include <tbblas/type_traits.hpp>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <boost/utility/enable_if.hpp>

namespace tbblas {

template<class T>
struct const_img_expression {
  typedef const_img_expression<T> expression_t;
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t complex_t;
  typedef typename complex_t::value_t value_t;
  static const unsigned dimCount = T::dimCount;

  struct get_img : public thrust::unary_function<complex_t, value_t> {
    __host__ __device__
    inline value_t operator()(const complex_t& value) const {
      return value.img;
    }
  };

  typedef thrust::transform_iterator<get_img, typename T::const_iterator> const_iterator;

  const_img_expression(const T& expr) : expr(expr) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), get_img());
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), get_img());
  }

  inline const dim_t& size() const {
    return expr.size();
  }

  inline size_t count() const {
    return expr.count();
  }

private:
  const T& expr;
};

template<class T>
struct img_expression {
  typedef img_expression<T> expression_t;
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t complex_t;
  typedef typename complex_t::value_t value_t;
  static const unsigned dimCount = T::dimCount;

  struct get_img : public thrust::unary_function<complex_t, value_t> {
    __host__ __device__
    inline value_t operator()(const complex_t& value) const {
      return value.img;
    }
  };

  typedef thrust::tuple<complex_t, value_t> tuple_t;

  struct set_img {
    template<class Tuple>
    __host__ __device__
    void operator()(Tuple t) const {
      thrust::get<0>(t).img = thrust::get<1>(t);
    }
  };

  typedef thrust::transform_iterator<get_img, typename T::const_iterator> const_iterator;

  img_expression(T& expr) : expr(expr) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), get_img());
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), get_img());
  }

  inline const dim_t& size() const {
    return expr.size();
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
      assert(size[i] == this->size()[i]);
    }
    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(this->expr.begin(), expr.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(this->expr.end(), expr.end())),
        set_img());
    return *this;
  }

private:
  T& expr;
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  typename boost::enable_if<is_complex<typename Expression::value_t>,
    const_img_expression<Expression>
  >::type
>::type
img(const Expression& expr) {
  return const_img_expression<Expression>(expr);
}

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>,
  typename boost::enable_if<is_complex<typename Expression::value_t>,
    img_expression<Expression>
  >::type
>::type
img(Expression& expr) {
  return img_expression<Expression>(expr);
}

template<class T>
struct is_expression<const_img_expression<T> > {
  static const bool value = true;
};

template<class T>
struct is_expression<img_expression<T> > {
  static const bool value = true;
};

}

#endif /* TBBLAS_IMG_HPP_ */
