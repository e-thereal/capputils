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
#include <tbblas/deeplearn/serialize_dnn_layer.hpp>
#include <tbblas/deeplearn/serialize_nn_layer.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::encoder_model<T, dims>& model, std::ostream& out) {
  unsigned count = 0;

  count = model.cnn_encoders().size() + model.cnn_shortcuts().size();

  if (model.cnn_shortcuts().size()) {
    count |= 0x8000;    // CNN shortcut code
  }

  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < model.cnn_encoders().size(); ++i)
    serialize(*model.cnn_encoders()[i], out);
  for (size_t i = 0; i < model.cnn_shortcuts().size(); ++i)
    serialize(*model.cnn_shortcuts()[i], out);

  count = model.dnn_decoders().size() + model.dnn_shortcuts().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < model.dnn_decoders().size(); ++i)
    serialize(*model.dnn_decoders()[i], out);
  for (size_t i = 0; i < model.dnn_shortcuts().size(); ++i)
    serialize(*model.dnn_shortcuts()[i], out);

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
  typedef dnn_layer_model<T, dims> dnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  unsigned count = 0;

  model.cnn_encoders().clear();
  model.cnn_shortcuts().clear();
  in.read((char*)&count, sizeof(count));

  cnn_layer_t cnn_layer;
  if (count & 0x8000) {
    // has CNN shortcut
    count &= 0x7FFF;

    assert(count % 2 == 1);
    for (size_t i = 0; i < count / 2 + 1; ++i) {
      deserialize(in, cnn_layer);
      model.append_cnn_encoder(cnn_layer);
    }
    for (size_t i = 0; i < count / 2; ++i) {
      deserialize(in, cnn_layer);
      model.append_cnn_shortcut(cnn_layer);
    }
  } else {
    // no short cuts
    for (size_t i = 0; i < count; ++i) {
      deserialize(in, cnn_layer);
      model.append_cnn_encoder(cnn_layer);
    }
  }

  model.dnn_decoders().clear();
  model.dnn_shortcuts().clear();
  in.read((char*)&count, sizeof(count));

  assert(count == model.cnn_encoders().size() || count == 2 * model.cnn_encoders().size() - 1);

  // if has CNN shortcuts, set dnn_layer.set_visible_pooling(true);

  dnn_layer_t dnn_layer;
  for (size_t i = 0; i < model.cnn_encoders().size(); ++i) {
    deserialize(in, dnn_layer);
    model.append_dnn_decoder(dnn_layer);
  }

  if (count > model.dnn_decoders().size()) {
    for (size_t i = 0; i < model.cnn_encoders().size() - 1; ++i) {
      deserialize(in, dnn_layer);
      model.append_dnn_shortcut(dnn_layer);
    }
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
