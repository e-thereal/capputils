/*
 * serialize_encoder.hpp
 *
 *  Created on: Dec 05, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_ENCODER_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_ENCODER_HPP_

#include <tbblas/deeplearn/encoder_model.hpp>
#include <tbblas/deeplearn/serialize_cnn_layer.hpp>
#include <tbblas/deeplearn/serialize_reverse_cnn_layer.hpp>
#include <tbblas/deeplearn/serialize_nn_layer.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::encoder_model<T, dims>& model, std::ostream& out) {
  unsigned count = 0;

  count = model.cnn_encoders().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.cnn_encoders()[i], out);

  count = model.cnn_decoders().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.cnn_decoders()[i], out);

  count = model.nn_encoders().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.nn_encoders()[i], out);

  count = model.nn_decoders().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.nn_decoders()[i], out);
}

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::encoder_model<T, dims>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T, unsigned dims>
void deserialize(std::istream& in, tbblas::deeplearn::encoder_model<T, dims>& model) {

  typedef cnn_layer_model<T, dims> cnn_layer_t;
  typedef reverse_cnn_layer_model<T, dims> reverse_cnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  unsigned count = 0;

  model.cnn_encoders().clear();
  in.read((char*)&count, sizeof(count));

  cnn_layer_t cnn_layer;
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, cnn_layer);
    model.append_cnn_encoder(cnn_layer);
  }

  model.cnn_decoders().clear();
  in.read((char*)&count, sizeof(count));

  reverse_cnn_layer_t rcnn_layer;
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, rcnn_layer);
    model.append_cnn_decoder(rcnn_layer);
  }

  model.nn_encoders().clear();
  in.read((char*)&count, sizeof(count));

  nn_layer_t nn_layer;
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, nn_layer);
    model.append_nn_encoder(nn_layer);
  }

  model.nn_decoders().clear();
  in.read((char*)&count, sizeof(count));

  for (size_t i = 0; i < count; ++i) {
    deserialize(in, nn_layer);
    model.append_nn_decoder(nn_layer);
  }
}

template<class T, unsigned dims>
void deserialize(const std::string& filename, tbblas::deeplearn::encoder_model<T, dims>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_ENCODER_HPP_ */
