/*
 * serialize_dbn.hpp
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_

#include <tbblas/deeplearn/nn_model.hpp>
#include <tbblas/deeplearn/serialize_nn_layer.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T>
void serialize(const tbblas::deeplearn::nn_model<T>& model, std::ostream& out) {
  unsigned count = 0;

  count = model.layers().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.layers()[i], out);
}

template<class T>
void serialize(const tbblas::deeplearn::nn_model<T>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T>
void deserialize(std::istream& in, tbblas::deeplearn::nn_model<T>& model) {

  typedef nn_layer_model<T> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

  unsigned count = 0;
  nn_layer_t layer;

  in.read((char*)&count, sizeof(count));
  v_nn_layer_t layers(count);
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, layer);
    layers[i] = boost::make_shared<nn_layer_t>(layer);
  }
  model.set_layers(layers);
}

template<class T>
void deserialize(const std::string& filename, tbblas::deeplearn::nn_model<T>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_ */
