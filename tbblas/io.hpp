/*
 * print.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_IO_HPP_
#define TBBLAS_IO_HPP_

#include <iostream>
#include <iomanip>
#include <vector>

#include <tbblas/tensor.hpp>

#include <thrust/copy.h>

namespace tbblas {

template<class T, unsigned dim, bool device>
void print(const tbblas::tensor<T, dim, device>& tensor, int precision = 5,
    std::ostream& out = std::cout) {
  out << "Cannot print tensors of dimension > 2." << std::endl;
}

template<class Expression>
typename boost::enable_if<is_expression<Expression> >::type
print(const Expression& expr, int precision = 5,
    std::ostream& out = std::cout)
{
  tbblas::tensor<typename Expression::value_t,
    Expression::dimCount, Expression::cuda_enabled> tensor = expr;
  print(tensor, precision, out);
}

template<class T, bool device>
void print(const tbblas::tensor<T, 0u, device>& tensor, int precision = 5,
    std::ostream& out = std::cout)
{
  out << "[1 x 1]" << std::endl;
  out << std::setprecision(precision) << std::setw(precision + 5) << tensor.data()[0] << std::endl;
}

template<class T, bool device>
void print(const tbblas::tensor<T, 1u, device>& tensor, int precision = 5,
    std::ostream& out = std::cout)
{
  const size_t count = tensor.size()[0];

  std::vector<T> data(count);
  thrust::copy(tensor.begin(), tensor.end(), data.begin());
  out << "[" << count << "]" << std::endl;

  for (size_t i = 0; i < count; ++i) {
    out << std::setprecision(precision) << std::setw(precision + 5) << data[i] << " ";
  }
  out << std::endl;
}

template<class T, bool device>
void print(const tbblas::tensor<T, 2u, device>& tensor, int precision = 5,
    std::ostream& out = std::cout)
{
  const size_t rowCount = tensor.size()[0], columnCount = tensor.size()[1];
  const size_t count = rowCount * columnCount;

  std::vector<T> data(count);
  thrust::copy(tensor.begin(), tensor.end(), data.begin());
  out << "[" << rowCount << " x " << columnCount << "]" << std::endl;

  for (size_t i = 0; i < rowCount; ++i) {
    for (size_t j = 0; j < columnCount; ++j) {
      out << std::setprecision(precision) << std::setw(precision + 5) << data[i + j * rowCount] << " ";
    }
    out << std::endl;
  }
}

// generic solution
template <class T>
int numDigits(T number) {
  if (number == 0)
    return 1;
  int digits = 0;
  while (number) {
      number /= 10;
      digits++;
  }
  return digits;
}

template<class T, bool device>
void print(const tbblas::tensor<T, 3u, device>& tensor, int precision = 5,
    std::ostream& out = std::cout)
{
  const size_t rowCount = tensor.size()[0], columnCount = tensor.size()[1], depth = tensor.size()[2];
  const size_t count = rowCount * columnCount * depth, sliceCount = rowCount * columnCount;

  std::vector<T> data(count);
  thrust::copy(tensor.begin(), tensor.end(), data.begin());
  out << "[" << rowCount << " x " << columnCount << " x " << depth << "]" << std::endl;

  int indent = numDigits(depth) + 4;

  for (size_t k = 0; k < depth; ++k) {
    out << "[" << k << "]";
    for (size_t i = 0; i < rowCount; ++i) {
      if (i == 0) {
        for (int iIndent = 0; iIndent < indent - numDigits(k) - 2; ++iIndent)
          out << " ";
      } else {
        for (int iIndent = 0; iIndent < indent; ++iIndent)
          out << " ";
      }
      for (size_t j = 0; j < columnCount; ++j) {
        out << std::setprecision(precision) << std::setw(precision + 5) << data[i + j * rowCount + k * sliceCount] << " ";
      }
      out << std::endl;
    }
    if (k < depth - 1)
      out << std::endl;
  }
}

}

template<class Expression>
typename boost::enable_if<tbblas::is_expression<Expression>,
  std::ostream
>::type&
operator<<(std::ostream& os, const Expression& expr) {
  print(expr, 5, os);
  return os;
}

//template<class T, unsigned dim, bool device>
//std::ostream& operator<<(std::ostream& os, const tbblas::tensor<T, dim, device>& tensor) {
//  print(tensor, 5, os);
//  return os;
//}
//
//template<class T, unsigned dim, bool device>
//std::ostream& operator<<(std::ostream& os, const tbblas::proxy<tbblas::tensor<T, dim, device> >& proxy) {
//  print(proxy, 5, os);
//  return os;
//}

#define tbblas_print(args...) std::cout << #args" = " << (args) << std::endl

#endif /* TBBLAS_IO_HPP_ */
