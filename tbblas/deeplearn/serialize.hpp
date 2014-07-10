/*
 * serialize.hpp
 *
 *  Created on: Jul 1, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_HPP_

#include <tbblas/deeplearn/conv_rbm_model.hpp>
#include <tbblas/deeplearn/rbm_model.hpp>
#include <tbblas/serialize.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

/*** CONV RBM ***/

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::conv_rbm_model<T, dim>& model, std::ostream& out) {
  unsigned count = 0;

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

  T mean = model.mean();
  out.write((char*)&mean, sizeof(mean));

  T stddev = model.stddev();
  out.write((char*)&stddev, sizeof(stddev));

  capputils::attributes::serialize_trait<unit_type>::writeToFile(model.visibles_type(), out);
  capputils::attributes::serialize_trait<unit_type>::writeToFile(model.hiddens_type(), out);
  serialize(model.mask(), out);
  capputils::attributes::serialize_trait<convolution_type>::writeToFile(model.convolution_type(), out);
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
  host_tensor_t tensor;

  in.read((char*)&count, sizeof(count));
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
}

template<class T, unsigned dim>
void deserialize(const std::string& filename, tbblas::deeplearn::conv_rbm_model<T, dim>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

/*** RBM ***/

template<class T>
void serialize(const tbblas::deeplearn::rbm_model<T>& model, std::ostream& out) {
  serialize(model.visible_bias(), out);
  serialize(model.hidden_bias(), out);
  serialize(model.weights(), out);
  serialize(model.mean(), out);
  serialize(model.stddev(), out);
  capputils::attributes::serialize_trait<unit_type>::writeToFile(model.visibles_type(), out);
  capputils::attributes::serialize_trait<unit_type>::writeToFile(model.hiddens_type(), out);
  serialize(model.mask(), out);
}

template<class T>
void serialize(const tbblas::deeplearn::rbm_model<T>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T>
void deserialize(std::istream& in, tbblas::deeplearn::rbm_model<T>& model) {

  typedef typename rbm_model<T>::host_matrix_t matrix_t;

  matrix_t matrix;

  deserialize(in, matrix);
  model.set_visible_bias(matrix);

  deserialize(in, matrix);
  model.set_hidden_bias(matrix);

  deserialize(in, matrix);
  model.set_weights(matrix);

  deserialize(in, matrix);
  model.set_mean(matrix);

  deserialize(in, matrix);
  model.set_stddev(matrix);

  unit_type type;
  capputils::attributes::serialize_trait<unit_type>::readFromFile(type, in);
  model.set_visibles_type(type);

  capputils::attributes::serialize_trait<unit_type>::readFromFile(type, in);
  model.set_hiddens_type(type);

  deserialize(in, matrix);
  model.set_mask(matrix);
}

template<class T>
void deserialize(const std::string& filename, tbblas::deeplearn::rbm_model<T>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_HPP_ */
