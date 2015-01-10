/*
 * serialize_reverse_cnn_layer.hpp
 *
 *  Created on: Aug 19, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_REVERSE_CNN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_REVERSE_CNN_LAYER_HPP_

#include <tbblas/deeplearn/reverse_cnn_layer_model.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/ones.hpp>
#include <iostream>
#include <fstream>


namespace tbblas {

namespace deeplearn {

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::reverse_cnn_layer_model<T, dim>& model, std::ostream& out) {
  unsigned count = 0;

  uint16_t version = model.version();

  out.write((char*)&version, sizeof(version));

  count = model.filters().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.filters()[i], out);

  serialize(model.bias(), out);

  if (model.version() >= 2) {
    serialize(model.mask(), out);
  }

  typename tbblas::deeplearn::reverse_cnn_layer_model<T, dim>::dim_t size = model.kernel_size();
  out.write((char*)&size, sizeof(size));

  size = model.stride_size();
  out.write((char*)&size, sizeof(size));

  size = model.pooling_size();
  out.write((char*)&size, sizeof(size));

  T mean = model.mean();
  out.write((char*)&mean, sizeof(mean));

  T stddev = model.stddev();
  out.write((char*)&stddev, sizeof(stddev));

  bool shared = model.shared_bias();
  out.write((char*)&shared, sizeof(shared));

  capputils::attributes::serialize_trait<activation_function>::writeToFile(model.activation_function(), out);
  capputils::attributes::serialize_trait<convolution_type>::writeToFile(model.convolution_type(), out);
  capputils::attributes::serialize_trait<pooling_method>::writeToFile(model.pooling_method(), out);
}

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::reverse_cnn_layer_model<T, dim>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T, unsigned dim>
void deserialize(std::istream& in, tbblas::deeplearn::reverse_cnn_layer_model<T, dim>& model) {

  typedef typename reverse_cnn_layer_model<T, dim>::host_tensor_t host_tensor_t;
  typedef typename reverse_cnn_layer_model<T, dim>::v_host_tensor_t v_host_tensor_t;

  unsigned count = 0;
  uint16_t version;
  host_tensor_t tensor;

  in.read((char*)&version, sizeof(version));
  model.set_version(version);

  if (version < 1) {
    throw std::runtime_error("This version of tbblas can not read modules prior to version 1.");
  }

  in.read((char*)&count, sizeof(count));
  v_host_tensor_t filters(count);
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, tensor);
    filters[i] = boost::make_shared<host_tensor_t>(tensor);
  }
  model.set_filters(filters);

  deserialize(in, tensor);
  model.set_bias(tensor);

  if (model.version() >= 2) {
    deserialize(in, tensor);
  } else {
    typename tbblas::deeplearn::reverse_cnn_layer_model<T, dim>::dim_t mask_size = model.visibles_size();
    mask_size[dim - 1] = 1;
    tensor = ones<T>(mask_size);
  }
  model.set_mask(tensor);

  typename tbblas::deeplearn::reverse_cnn_layer_model<T, dim>::dim_t size;
  in.read((char*)&size, sizeof(size));
  model.set_kernel_size(size);

  in.read((char*)&size, sizeof(size));
  model.set_stride_size(size);

  in.read((char*)&size, sizeof(size));
  model.set_pooling_size(size);

  T mean = 0;
  in.read((char*)&mean, sizeof(mean));
  model.set_mean(mean);

  T stddev = 1;
  in.read((char*)&stddev, sizeof(stddev));
  model.set_stddev(stddev);

  bool shared = false;
  in.read((char*)&shared, sizeof(shared));
  model.set_shared_bias(shared);

  activation_function function;
  capputils::attributes::serialize_trait<activation_function>::readFromFile(function, in);
  model.set_activation_function(function);

  convolution_type conv_type;
  capputils::attributes::serialize_trait<convolution_type>::readFromFile(conv_type, in);
  model.set_convolution_type(conv_type);

  pooling_method pooling;
  capputils::attributes::serialize_trait<pooling_method>::readFromFile(pooling, in);
  model.set_pooling_method(pooling);
}

template<class T, unsigned dim>
void deserialize(const std::string& filename, tbblas::deeplearn::reverse_cnn_layer_model<T, dim>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_REVERSE_CNN_LAYER_HPP_ */
