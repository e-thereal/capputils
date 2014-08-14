/*
 * conv.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_CONV_HPP_
#define TBBLAS_CONV_HPP_

#include <tbblas/tensor.hpp>

#include <tbblas/fft.hpp>
#include <tbblas/zeros.hpp>

namespace tbblas {

template<class Expression1, class Expression2>
struct conv_operation
{
  typedef typename Expression1::dim_t dim_t;
  typedef typename Expression1::value_t value_t;
  typedef complex<value_t> complex_t;
  static const unsigned dimCount = Expression1::dimCount;

  typedef tensor<value_t, dimCount, Expression1::cuda_enabled> tensor_t;
  typedef tensor<complex_t, dimCount, Expression1::cuda_enabled> ctensor_t;
  typedef fft_plan<dimCount> plan_t;

private:
  unsigned int upper_power_of_two(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
  }

public:
  conv_operation(const Expression1& expr1, const Expression2& expr2) : expr1(expr1), expr2(expr2) {
    _size = abs(expr1.size() - expr2.size()) + 1;
    _fullsize = abs(expr1.fullsize() - expr2.fullsize()) + 1;
    _maxSize = max(expr1.size(), expr2.size());

    for (unsigned i = 0; i < dimCount; ++i)
      _paddedSize[i] = upper_power_of_two(_maxSize[i]);
  }

  void apply(tensor_t& output) const {
    tensor_t padded1 = zeros<value_t>(_paddedSize), padded2 = zeros<value_t>(_paddedSize);
    padded1[seq<dimCount>(0), expr1.size()] = expr1;
    padded2[seq<dimCount>(0), expr2.size()] = expr2;

    plan_t plan;
    ctensor_t ctens1 = fft(padded1, plan);
    ctensor_t ctens2 = fft(padded2, plan);
    ctens1 = ctens1 * ctens2;
    padded1 = ifft(ctens1);
    output = padded1[_maxSize - _size, output.size()];
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _fullsize;
  }

private:
  const Expression1& expr1;
  const Expression2& expr2;
  dim_t _size, _fullsize, _paddedSize, _maxSize;
};

template<class T1, class T2>
struct is_operation<conv_operation<T1, T2> > {
  static const bool value = true;
};

template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if_c<Expression1::cuda_enabled == true,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        typename boost::enable_if_c<Expression1::dimCount <= 3,
          conv_operation<Expression1, Expression2>
        >::type
      >::type
    >::type
  >::type
>::type
conv(const Expression1& expr1, const Expression2& expr2) {
  return conv_operation<Expression1, Expression2>(expr1, expr2);
}

}

#endif /* TBBLAS_CONV_HPP_ */
