/*
 * reshape.hpp
 *
 *  Created on: Nov 26, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_RESHAPE_HPP_
#define TBBLAS_RESHAPE_HPP_

#include <tbblas/proxy.hpp>

namespace tbblas {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
  proxy<Tensor>
>::type
reshape(Tensor& tensor, const typename Tensor::dim_t& size) {
  return proxy<Tensor>(tensor, size);
}

template<class Expression, unsigned dims>
struct reshape_expression {

  typedef reshape_expression<Expression, dims> expression_t;
  typedef typename Expression::value_t value_t;
  typedef typename tensor<value_t, dims>::dim_t dim_t;

  static const unsigned dimCount = dims;
  static const bool cuda_enabled = Expression::cuda_enabled;

  typedef typename Expression::const_iterator const_iterator;

  reshape_expression(const Expression& expr, const dim_t& size)
   : _expr(expr), _size(size) { }

  inline const_iterator begin() const {
    return _expr.begin();
  }

  inline const_iterator end() const {
    return _expr.end();
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

  inline size_t count() const {
    return _expr.count();
  }

private:
  const Expression& _expr;
  dim_t _size;
};

template<class T, unsigned dims>
struct is_expression<reshape_expression<T, dims> > {
  static const bool value = true;
};

template<class Expression, unsigned dims>
typename boost::enable_if<is_expression<Expression>,
    reshape_expression<Expression, dims>
>::type
reshape(const Expression& expr, const sequence<int, dims>& size)
{
  int count = 1;
  for (int i = 0; i < dims; ++i)
    count *= size[i];
  assert(expr.count() == count);
  return reshape_expression<Expression, dims>(expr, size);
}

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    reshape_expression<Expression, 1>
>::type
reshape(const Expression& expr, int width)
{
  assert(expr.count() == width);
  return reshape(expr, seq(width));
}

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    reshape_expression<Expression, 2>
>::type
reshape(const Expression& expr, int width, int height)
{
  assert(expr.count() == width * height);
  return reshape(expr, seq(width, height));
}

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    reshape_expression<Expression, 3>
>::type
reshape(const Expression& expr, int width, int height, int depth)
{
  assert(expr.count() == width * height * depth);
  return reshape(expr, seq(width, height, depth));
}

template<class Expression>
typename boost::enable_if<is_expression<Expression>,
    reshape_expression<Expression, 4>
>::type
reshape(const Expression& expr, int width, int height, int depth, int channels)
{
  assert(expr.count() == width * height * depth * channels);
  return reshape(expr, seq(width, height, depth, channels));
}

}

#endif /* TBBLAS_RESHAPE_HPP_ */
