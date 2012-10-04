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
  static const int dimCount = Expression1::dimCount;

  typedef tensor<value_t, dimCount, Expression1::cuda_enabled> tensor_t;
  typedef tensor<complex_t, dimCount, Expression1::cuda_enabled> ctensor_t;
  typedef fft_plan<dimCount> plan_t;

  conv_operation(const Expression1& expr1, const Expression2& expr2) : expr1(expr1), expr2(expr2) {
    _size = abs(expr1.size() - expr2.size()) + 1;
    _paddedSize = max(expr1.size(), expr2.size());

//    std::cout << "size1: " << expr1.size() << std::endl;
//    std::cout << "size2: " << expr2.size() << std::endl;
//    std::cout << "result: " << _size << std::endl;
//    std::cout << "padded: " << _paddedSize << std::endl;
  }

  void apply(tensor_t& output) const {
//    std::cout << "output: " << output.size() << std::endl;

    tensor_t padded1 = zeros<value_t>(_paddedSize), padded2 = zeros<value_t>(_paddedSize);
    padded1[sequence<int,dimCount>(0), expr1.size()] = expr1;
    padded2[sequence<int,dimCount>(0), expr2.size()] = expr2;

    plan_t plan;
    ctensor_t ctens1 = fft(padded1, plan);
    ctensor_t ctens2 = fft(padded2, plan);
    ctens1 = ctens1 * ctens2;
    padded1 = ifft(ctens1);
    output = padded1[_paddedSize - _size, output.size()];
  }

  inline const dim_t& size() const {
    return _size;
  }

private:
  const Expression1& expr1;
  const Expression2& expr2;
  dim_t _size, _paddedSize;
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
