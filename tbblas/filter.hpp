/*
 * filter.hpp
 *
 *  Created on: Mar 12, 2013
 *      Author: tombr
 */

#ifndef TBBLAS_FILTER_HPP_
#define TBBLAS_FILTER_HPP_

#include <tbblas/tensor.hpp>

#include <tbblas/fft.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/assert.hpp>

namespace tbblas {

template<class Expression1, class Expression2>
struct filter_operation
{
  typedef typename Expression1::dim_t dim_t;
  typedef typename Expression1::value_t value_t;
  typedef complex<value_t> complex_t;
  static const unsigned dimCount = Expression1::dimCount;

  typedef tensor<value_t, dimCount, Expression1::cuda_enabled> tensor_t;
  typedef tensor<complex_t, dimCount, Expression1::cuda_enabled> ctensor_t;
  typedef fft_plan<dimCount> plan_t;

private:
//  unsigned int upper_power_of_two(unsigned int v) {
//    v--;
//    v |= v >> 1;
//    v |= v >> 2;
//    v |= v >> 4;
//    v |= v >> 8;
//    v |= v >> 16;
//    v++;
//    return v;
//  }

public:
  filter_operation(const Expression1& input, const Expression2& kernel, const dim_t& size, unsigned dimension)
   : input(input), kernel(kernel), _size(size), dimension(dimension)
  {
    tbblas_assert(max(input.size(), kernel.size()) == input.size());

    for (unsigned i = dimension; i < dimCount; ++i)
      tbblas_assert(kernel.size()[i] == input.size()[i]);
  }

  void apply(tensor_t& output) const {
    dim_t topleft = input.size() / 2 - kernel.size() / 2;
    dim_t outputTopleft = input.size() / 2 - output.size() / 2;

    tensor_t padded = zeros<value_t>(input.size());

    padded[topleft, kernel.size()] = kernel;
    tensor_t shiftedKernel = ifftshift(padded, dimension);
    padded = input;

    plan_t plan;
    ctensor_t ctens1 = fft(padded, dimension, plan);
    ctensor_t ctens2 = fft(shiftedKernel, dimension, plan);
    ctens1 = ctens1 * ctens2;
    padded = ifft(ctens1, dimension);
    output = padded[outputTopleft, output.size()];
  }

  inline dim_t size() const {
    return _size;
  }

  inline dim_t fullsize() const {
    return _size;
  }

private:
  const Expression1& input;
  const Expression2& kernel;
  dim_t _size;
  unsigned dimension;
};

template<class T1, class T2>
struct is_operation<filter_operation<T1, T2> > {
  static const bool value = true;
};

/**
 * The true dimension of the kernel must be \c dimension meaning that
 * kernel.size()[i] == 1, for i >= dimension
 */
template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if_c<Expression1::cuda_enabled == true,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        typename boost::enable_if_c<Expression1::dimCount <= 3,
          filter_operation<Expression1, Expression2>
        >::type
      >::type
    >::type
  >::type
>::type
filter(const Expression1& input, const Expression2& kernel) {
  return filter_operation<Expression1, Expression2>(input, kernel, input.size(), Expression1::dimCount);
}

template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if_c<Expression1::cuda_enabled == true,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        filter_operation<Expression1, Expression2>
      >::type
    >::type
  >::type
>::type
filter(const Expression1& input, const Expression2& kernel, unsigned dimension) {
  tbblas_assert(dimension <= 3);
  return filter_operation<Expression1, Expression2>(input, kernel, input.size(), dimension);
}

template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if_c<Expression1::cuda_enabled == true,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        typename boost::enable_if_c<Expression1::dimCount <= 3,
          filter_operation<Expression1, Expression2>
        >::type
      >::type
    >::type
  >::type
>::type
filter(const Expression1& input, const Expression2& kernel, const typename Expression1::dim_t& size) {
  return filter_operation<Expression1, Expression2>(input, kernel, size, Expression1::dimCount);
}

template<class Expression1, class Expression2>
typename boost::enable_if<is_expression<Expression1>,
  typename boost::enable_if<is_expression<Expression2>,
    typename boost::enable_if_c<Expression1::cuda_enabled == true,
      typename boost::enable_if_c<Expression1::dimCount == Expression2::dimCount,
        filter_operation<Expression1, Expression2>
      >::type
    >::type
  >::type
>::type
filter(const Expression1& input, const Expression2& kernel, const typename Expression1::dim_t& size, unsigned dimension) {
  tbblas_assert(dimension <= 3);
  return filter_operation<Expression1, Expression2>(input, kernel, size, dimension);
}

}

#endif /* TBBLAS_FILTER_HPP_ */
