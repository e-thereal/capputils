/*
 * binary_expression.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_BINARY_EXPRESSION_HPP_
#define TBBLAS_BINARY_EXPRESSION_HPP_

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace tbblas {

template<class T1, class T2, class Operation>
struct binary_expression {
  typedef typename T1::dim_t dim_t;
  typedef typename T1::value_t value_t;
  static const unsigned dimCount = T1::dimCount;

  typedef thrust::tuple<value_t, value_t> tuple_t;

  struct apply_operation : public thrust::unary_function<tuple_t, value_t> {

    apply_operation(const Operation& op) : op(op) { }

    __host__ __device__
    inline value_t operator()(const tuple_t& t) const {
      return op(thrust::get<0>(t), thrust::get<1>(t));
    }

  private:
    const Operation& op;
  };

  typedef thrust::zip_iterator<thrust::tuple<typename T1::const_iterator, typename T2::const_iterator> > zip_iterator_t;
  typedef thrust::transform_iterator<apply_operation, zip_iterator_t> const_iterator;

  binary_expression(const T1& expr1, const T2& expr2, const Operation& op)
   : expr1(expr1), expr2(expr2), op(op) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(expr1.begin(), expr2.begin())),
        apply_operation(op));
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(expr1.end(), expr2.end())),
        apply_operation(op));
  }

  inline const dim_t& size() const {
    return expr1.size();
  }

  inline size_t count() const {
    return expr1.count();
  }

private:
  const T1& expr1;
  const T2& expr2;
  const Operation& op;
};

template<class T1, class T2, class Operation>
struct is_expression<binary_expression<T1, T2, Operation> > {
  static const bool value = true;
};

}

#endif /* TBBLAS_BINARY_EXPRESSION_HPP_ */
