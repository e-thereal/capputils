/*
 * serialize_nn_layer.hpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_NN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_NN_LAYER_HPP_

#include <tbblas/deeplearn/nn_layer_model.hpp>
#include <tbblas/serialize.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T>
void serialize(const tbblas::deeplearn::nn_layer_model<T>& model, std::ostream& out) {
  serialize(model.bias(), out);
  serialize(model.weights(), out);
  serialize(model.mean(), out);
  serialize(model.stddev(), out);
  capputils::attributes::serialize_trait<activation_function>::writeToFile(model.activation_function(), out);
}

template<class T>
void serialize(const tbblas::deeplearn::nn_layer_model<T>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T>
void deserialize(std::istream& in, tbblas::deeplearn::nn_layer_model<T>& model) {

  typedef typename nn_layer_model<T>::host_matrix_t matrix_t;

  matrix_t matrix;

  deserialize(in, matrix);
  model.set_bias(matrix);

  deserialize(in, matrix);
  model.set_weights(matrix);

  deserialize(in, matrix);
  model.set_mean(matrix);

  deserialize(in, matrix);
  model.set_stddev(matrix);

  activation_function function;
  capputils::attributes::serialize_trait<activation_function>::readFromFile(function, in);
  model.set_activation_function(function);
}

template<class T>
void deserialize(const std::string& filename, tbblas::deeplearn::nn_layer_model<T>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_NN_LAYER_HPP_ */
