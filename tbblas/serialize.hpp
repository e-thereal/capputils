/*
 * serialize.hpp
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_SERIALIZE_HPP_
#define TBBLAS_SERIALIZE_HPP_

#include <tbblas/tensor.hpp>
#include <iostream>
#include <fstream>

#include <tbblas/detail/copy.hpp>
#include <tbblas/detail/system.hpp>

namespace tbblas {

template<class T, unsigned dim, bool device>
void serialize(const tbblas::tensor<T, dim, device>& tensor, std::ostream& out) {
  typename tbblas::tensor<T, dim, device>::dim_t size = tensor.size();
  out.write((char*)&size, sizeof(size));

  typename tbblas::tensor<T, dim, device>::dim_t fullsize = tensor.fullsize();
  out.write((char*)&fullsize, sizeof(fullsize));

  if (device) {
    std::vector<T> buffer(tensor.count());
    thrust::copy(tensor.begin(), tensor.end(), buffer.begin());
    out.write((char*)&buffer[0], sizeof(T) * tensor.count());
  } else {
    out.write((char*)thrust::raw_pointer_cast(tensor.data().data()), sizeof(T) * tensor.count());
  }
}

template<class T, unsigned dim, bool device>
void serialize(const tbblas::tensor<T, dim, device>& tensor, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(tensor, out);
}

template<class T, unsigned dim, bool device>
void deserialize(std::istream& in, tbblas::tensor<T, dim, device>& tensor) {
  typename tbblas::tensor<T, dim, device>::dim_t size, fullsize;
  in.read((char*)&size, sizeof(size));
  in.read((char*)&fullsize, sizeof(fullsize));
  tensor.resize(size, fullsize);

  if (device) {
    std::vector<T> buffer(tensor.count());
    in.read((char*)&buffer[0], sizeof(T) * tensor.count());
    thrust::copy(buffer.begin(), buffer.end(), tensor.begin());
  } else {
    in.read((char*)thrust::raw_pointer_cast(tensor.data().data()), sizeof(T) * tensor.count());
  }
}

template<class T, unsigned dim, bool device>
void deserialize(const std::string& filename, tbblas::tensor<T, dim, device>& tensor) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, tensor);
}

}

#endif /* TBBLAS_SERIALIZE_HPP_ */
