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
void print(const tbblas::tensor<T, dim, device>& tensor, int precision = 6,
    std::ostream& out = std::cout) {
  out << "Cannot print tensors of dimension > 2." << std::endl;
}

template<class T, bool device>
void print(const tbblas::tensor<T, 0u, device>& tensor, int precision = 6,
    std::ostream& out = std::cout)
{
  out << "[1 x 1]" << std::endl;
  out << std::setprecision(precision) << std::setw(precision + 3) << tensor.data()[0] << std::endl;
}

template<class T, bool device>
void print(const tbblas::tensor<T, 1u, device>& tensor, int precision = 6,
    std::ostream& out = std::cout)
{
  const size_t count = tensor.size()[0];

  std::vector<T> data(count);
  thrust::copy(tensor.begin(), tensor.end(), data.begin());
  out << "[" << count << " x 1]" << std::endl;

  for (size_t i = 0; i < count; ++i) {
    out << std::setprecision(precision) << std::setw(precision + 3) << data[i] << std::endl;
  }
}

template<class T, bool device>
void print(const tbblas::tensor<T, 2u, device>& tensor, int precision = 6,
    std::ostream& out = std::cout)
{
  const size_t rowCount = tensor.size()[0], columnCount = tensor.size()[1];
  const size_t count = rowCount * columnCount;

  std::vector<T> data(count);
  thrust::copy(tensor.begin(), tensor.end(), data.begin());
  out << "[" << rowCount << " x " << columnCount << "]" << std::endl;

  for (size_t i = 0; i < rowCount; ++i) {
    for (size_t j = 0; j < columnCount; ++j) {
      out << std::setprecision(precision) << std::setw(precision + 3) << data[i + j * rowCount];
    }
    out << std::endl;
  }
}

}

template<class T, unsigned dim, bool device>
std::ostream& operator<<(std::ostream& os, const tbblas::tensor<T, dim, device>& tensor) {
  print(tensor, 6, os);
  return os;
}

#endif /* TBBLAS_IO_HPP_ */
