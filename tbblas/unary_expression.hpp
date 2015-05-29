/*
 * scalar_expression.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_UNARY_EXPRESSION_HPP_
#define TBBLAS_UNARY_EXPRESSION_HPP_

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

#include <thrust/functional.h>

namespace tbblas {

template<class T, class Operation>
struct unary_expression {
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t input_t;
  typedef typename Operation::value_t value_t;
  static const unsigned dimCount = T::dimCount;
  static const bool cuda_enabled = T::cuda_enabled;

  struct apply_operation : public thrust::unary_function<input_t, value_t> {
    __host__ __device__
    inline value_t operator()(const input_t& value) const {
      return op(value);
    }

    apply_operation(const Operation& op) : op(op) { }

  private:
    Operation op;
  };

  typedef thrust::transform_iterator<apply_operation, typename T::const_iterator> const_iterator;

  unary_expression(const T& expr, const Operation& op) : expr(expr), op(op) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), apply_operation(op));
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), apply_operation(op));
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
  Operation op;
};

template<class T, class Operation>
struct is_expression<unary_expression<T, Operation> > {
  static const bool value = true;
};

template<class T, class Operation>
struct scalar_first_operation {
  typedef T value_t;

  scalar_first_operation(const T& scalar, const Operation& op) : scalar(scalar), op(op) { }

  __host__ __device__
  T operator()(const T& x) const {
    return op(scalar, x);
  }

private:
  T scalar;
  Operation op;
};

template<class T, class Operation>
struct scalar_last_operation {
  typedef T value_t;

  scalar_last_operation(const T& scalar, const Operation& op) : scalar(scalar), op(op) { }

  __host__ __device__
  T operator()(const T& x) const {
    return op(x, scalar);
  }

private:
  T scalar;
  Operation op;
};

}

#endif /* SCALAR_EXPRESSION_HPP_ */
