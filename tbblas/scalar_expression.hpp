/*
 * scalar_expression.hpp
 *
 *  Created on: Oct 1, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SCALAR_EXPRESSION_HPP_
#define TBBLAS_SCALAR_EXPRESSION_HPP_

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <tbblas/tensor.hpp>
#include <tbblas/type_traits.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>

#include <thrust/functional.h>

namespace tbblas {

template<class T, class Operation>
struct scalar_expression {
  typedef typename T::dim_t dim_t;
  typedef typename T::value_t value_t;
  static const unsigned dimCount = T::dimCount;

  struct apply_operation : public thrust::unary_function<value_t, value_t> {
    __host__ __device__
    inline value_t operator()(const value_t& value) const {
      return op(value);
    }

    apply_operation(const Operation& op) : op(op) { }

  private:
    const Operation& op;
  };

  typedef thrust::transform_iterator<apply_operation, typename T::const_iterator> const_iterator;

  scalar_expression(const T& expr, const Operation& op) : expr(expr), op(op) { }

  inline const_iterator begin() const {
    return thrust::make_transform_iterator(expr.begin(), apply_operation(op));
  }

  inline const_iterator end() const {
    return thrust::make_transform_iterator(expr.end(), apply_operation(op));
  }

  inline const dim_t& size() const {
    return expr.size();
  }

  inline size_t count() const {
    return expr.count();
  }

private:
  const T& expr;
  const Operation& op;
};

template<class T, class Operation>
struct is_expression<scalar_expression<T, Operation> > {
  static const bool value = true;
};

}

#endif /* SCALAR_EXPRESSION_HPP_ */
