/*
 * divides.hpp
 *
 *  Created on: Sep 9, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DIVIDES_HPP_
#define TBBLAS_DIVIDES_HPP_

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace tbblas {

/*** Scalar divides operations ***/

template<class T>
struct scalar_divides_expression {
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t value_t;
  static const unsigned dimCount = T::dimCount;
  static const bool cuda_enabled = T::cuda_enabled;

  struct apply_divides : public thrust::unary_function<value_t, value_t> {
    __host__ __device__
    inline value_t operator()(const value_t& value) const {
      return value / scalar;
    }

    apply_divides(const value_t& scalar) : scalar(scalar) { }

  private:
    value_t scalar;
  };

  typedef thrust::transform_iterator<apply_divides, typename T::const_iterator> const_iterator;

  scalar_divides_expression(const T& expr, const value_t& scalar) : expr(expr), scalar(scalar) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), apply_divides(scalar));
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), apply_divides(scalar));
  }

  inline dim_t size() const {
    return expr.size();
  }

  inline size_t count() const {
    return expr.count();
  }

private:
  const T& expr;
  value_t scalar;
};

template<class Expression>
inline typename boost::enable_if<is_expression<Expression>, scalar_divides_expression<Expression> >::type
operator/(const Expression& expr, const typename Expression::value_t& value) {
  return scalar_divides_expression<Expression>(expr, value);
}

template<class T>
struct is_expression<scalar_divides_expression<T> > {
  static const bool value = true;
};

/*** Element-wise multiplies operations ***/

//template<class T1, class T2>
//struct divides_expression {
//  typedef typename T1::dim_t dim_t;
//  typedef typename T1::value_t value_t;
//  static const unsigned dimCount = T1::dimCount;
//
//  typedef thrust::tuple<value_t, value_t> tuple_t;
//
//  struct apply_divides : public thrust::unary_function<tuple_t, value_t> {
//    __host__ __device__
//    inline value_t operator()(const tuple_t& t) const {
//      return thrust::get<0>(t) / thrust::get<1>(t);
//    }
//  };
//
//  typedef thrust::zip_iterator<thrust::tuple<typename T1::const_iterator, typename T2::const_iterator> > zip_iterator_t;
//  typedef thrust::transform_iterator<apply_divides, zip_iterator_t> const_iterator;
//
//  divides_expression(const T1& expr1, const T2& expr2) : expr1(expr1), expr2(expr2) { }
//
//  inline const_iterator begin() const {
//    return thrust::make_transform_iterator(
//        thrust::make_zip_iterator(thrust::make_tuple(expr1.begin(), expr2.begin())),
//        apply_divides());
//  }
//
//  inline const_iterator end() const {
//    return thrust::make_transform_iterator(
//        thrust::make_zip_iterator(thrust::make_tuple(expr1.end(), expr2.end())),
//        apply_divides());
//  }
//
//  inline const dim_t& size() const {
//    return expr1.size();
//  }
//
//  inline size_t count() const {
//    return expr1.count();
//  }
//
//private:
//  const T1& expr1;
//  const T2& expr2;
//};
//
//template<class Expression1, class Expression2>
//inline typename boost::enable_if<
//  is_expression<Expression1>,
//  typename boost::enable_if<
//    is_expression<Expression2>,
//    typename boost::enable_if<
//      boost::is_same<typename Expression1::value_t, typename Expression2::value_t>,
//      typename boost::enable_if_c<
//        Expression1::dimCount == Expression2::dimCount,
//        divides_expression<Expression1, Expression2>
//      >::type
//    >::type
//  >::type
//>::type
//operator/(const Expression1& expr1, const Expression2& expr2) {
//  return divides_expression<Expression1, Expression2>(expr1, expr2);
//}
//
//template<class T1, class T2>
//struct is_expression<divides_expression<T1, T2> > {
//  static const bool value = true;
//};

}

#endif /* TBBLAS_DIVIDES_HPP_ */
