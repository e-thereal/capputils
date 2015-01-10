/*
 * conv_rbm.hpp
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_CONV_RBM_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_CONV_RBM_HPP_

#include <tbblas/deeplearn/conv_rbm_model.hpp>
#include <tbblas/serialize.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

/*** CONV RBM ***/

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::conv_rbm_model<T, dim>& model, std::ostream& out) {

  unsigned magic = 0xDEE9DEE9;
  unsigned version = model.version();
  unsigned count = 0;

  out.write((char*)&magic, sizeof(magic));
  out.write((char*)&version, sizeof(version));

  count = model.filters().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.filters()[i], out);

  serialize(model.visible_bias(), out);

  count = model.hidden_bias().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.hidden_bias()[i], out);

  typename tbblas::deeplearn::conv_rbm_model<T, dim>::dim_t size = model.kernel_size();
  out.write((char*)&size, sizeof(size));

  if (version >= 1) {
    size = model.stride_size();
    out.write((char*)&size, sizeof(size));
  }

  if (version >= 2) {
    size = model.pooling_size();
    out.write((char*)&size, sizeof(size));

    capputils::attributes::serialize_trait<pooling_method>::writeToFile(model.pooling_method(), out);
  }

  T mean = model.mean();
  out.write((char*)&mean, sizeof(mean));

  T stddev = model.stddev();
  out.write((char*)&stddev, sizeof(stddev));

  capputils::attributes::serialize_trait<unit_type>::writeToFile(model.visibles_type(), out);
  capputils::attributes::serialize_trait<unit_type>::writeToFile(model.hiddens_type(), out);
  serialize(model.mask(), out);
  capputils::attributes::serialize_trait<convolution_type>::writeToFile(model.convolution_type(), out);

  bool shared = model.shared_bias();
  out.write((char*)&shared, sizeof(shared));
}

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::conv_rbm_model<T, dim>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T, unsigned dim>
void deserialize(std::istream& in, tbblas::deeplearn::conv_rbm_model<T, dim>& model) {

  typedef typename conv_rbm_model<T, dim>::host_tensor_t host_tensor_t;
  typedef typename conv_rbm_model<T, dim>::v_host_tensor_t v_host_tensor_t;

  unsigned count = 0;
  unsigned magic = 0;
  unsigned version = 0;
  host_tensor_t tensor;

  // If at least version 1 (magic code was introduced), read version and count
  in.read((char*)&magic, sizeof(magic));
  if (magic == 0xDEE9DEE9) {
    in.read((char*)&version, sizeof(version));
    in.read((char*)&count, sizeof(count));
  } else {  // else, there is no version number, so the first number is the filter count
    count = magic;
    version = 0;
  }

  if (version < 3)
    throw std::runtime_error("This version of tbblas does not support conv_rbm versions older than 3.");

  model.set_version(version);

  v_host_tensor_t filters(count);
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, tensor);
    filters[i] = boost::make_shared<host_tensor_t>(tensor);
  }
  model.set_filters(filters);

  deserialize(in, tensor);
  model.set_visible_bias(tensor);

  in.read((char*)&count, sizeof(count));
  v_host_tensor_t hiddenBiases(count);
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, tensor);
    hiddenBiases[i] = boost::make_shared<host_tensor_t>(tensor);
  }
  model.set_hidden_bias(hiddenBiases);

  typename tbblas::deeplearn::conv_rbm_model<T, dim>::dim_t size;
  in.read((char*)&size, sizeof(size));
  model.set_kernel_size(size);

  if (version >= 1) {
    in.read((char*)&size, sizeof(size));
    model.set_stride_size(size);
  } else {
    model.set_stride_size(seq<dim>(1));
  }

  if (version >= 2) {
    in.read((char*)&size, sizeof(size));
    model.set_pooling_size(size);

    pooling_method method;
    capputils::attributes::serialize_trait<pooling_method>::readFromFile(method, in);
    model.set_pooling_method(method);
  } else {
    model.set_pooling_size(seq<dim>(1));
  }

  T mean = 0;
  in.read((char*)&mean, sizeof(mean));
  model.set_mean(mean);

  T stddev = 1;
  in.read((char*)&stddev, sizeof(stddev));
  model.set_stddev(stddev);

  unit_type type;
  capputils::attributes::serialize_trait<unit_type>::readFromFile(type, in);
  model.set_visibles_type(type);

  capputils::attributes::serialize_trait<unit_type>::readFromFile(type, in);
  model.set_hiddens_type(type);

  deserialize(in, tensor);
  model.set_mask(tensor);

  convolution_type conv_type;
  capputils::attributes::serialize_trait<convolution_type>::readFromFile(conv_type, in);
  model.set_convolution_type(conv_type);

  bool shared = false;
  in.read((char*)&shared, sizeof(shared));
  model.set_shared_bias(shared);
}

template<class T, unsigned dim>
void deserialize(const std::string& filename, tbblas::deeplearn::conv_rbm_model<T, dim>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_CONV_RBM_HPP_ */
