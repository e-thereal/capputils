/*
 * serialize_rbm.hpp
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_RBM_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_RBM_HPP_

#include <tbblas/deeplearn/rbm_model.hpp>
#include <tbblas/serialize.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

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

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_RBM_HPP_ */
