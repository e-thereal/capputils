/*
 * serialize_cnn.hpp
 *
 *  Created on: Aug 19, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_CNN_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_CNN_HPP_

#include <tbblas/deeplearn/cnn_model.hpp>
#include <tbblas/deeplearn/serialize_cnn_layer.hpp>
#include <tbblas/deeplearn/serialize_nn_layer.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::cnn_model<T, dims>& model, std::ostream& out) {
  unsigned count = 0;

  count = model.cnn_layers().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.cnn_layers()[i], out);

  count = model.nn_layers().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.nn_layers()[i], out);
}

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::cnn_model<T, dims>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T, unsigned dims>
void deserialize(std::istream& in, tbblas::deeplearn::cnn_model<T, dims>& model) {

  typedef cnn_layer_model<T, dims> cnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  unsigned count = 0;
  in.read((char*)&count, sizeof(count));

  cnn_layer_t cnn_layer;
  model.cnn_layers().clear();
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, cnn_layer);
    model.append_cnn_layer(cnn_layer);
  }

  in.read((char*)&count, sizeof(count));

  nn_layer_t nn_layer;
  model.nn_layers().clear();
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, nn_layer);
    model.append_nn_layer(nn_layer);
  }
}

template<class T, unsigned dims>
void deserialize(const std::string& filename, tbblas::deeplearn::cnn_model<T, dims>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_ */
