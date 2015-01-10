/*
 * encoder_model.hpp
 *
 *  Created on: Jan 02, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_ENCODER_MODEL_HPP_
#define TBBLAS_DEEPLEARN_ENCODER_MODEL_HPP_

#include <tbblas/deeplearn/cnn_layer_model.hpp>
#include <tbblas/deeplearn/reverse_cnn_layer_model.hpp>
#include <tbblas/deeplearn/nn_layer_model.hpp>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class encoder_model {
public:
  typedef T value_t;
  static const unsigned dimCount = dims;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef cnn_layer_model<T, dims> cnn_layer_t;
  typedef reverse_cnn_layer_model<T, dims> reverse_cnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;
  typedef std::vector<boost::shared_ptr<reverse_cnn_layer_t> > v_reverse_cnn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

protected:
  v_cnn_layer_t _cnn_encoders;
  v_reverse_cnn_layer_t _cnn_decoders;
  v_nn_layer_t _nn_encoders, _nn_decoders;

public:
  encoder_model() { }

  encoder_model(const encoder_model<T, dims>& model) {
    set_cnn_encoders(model.cnn_encoders());
    set_cnn_decoders(model.cnn_decoders());
    set_nn_encoders(model.nn_encoders());
    set_nn_decoders(model.nn_decoders());
  }

  virtual ~encoder_model() { }

public:
  template<class U>
  void set_cnn_encoders(const std::vector<boost::shared_ptr<cnn_layer_model<U, dims> > >& layers) {
    _cnn_encoders.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _cnn_encoders[i] = boost::make_shared<cnn_layer_t>(*layers[i]);
  }

  v_cnn_layer_t& cnn_encoders() {
    return _cnn_encoders;
  }

  const v_cnn_layer_t& cnn_encoders() const {
    return _cnn_encoders;
  }

  template<class U>
  void append_cnn_encoder(const cnn_layer_model<U, dims>& layer) {
    _cnn_encoders.push_back(boost::make_shared<cnn_layer_t>(layer));
  }

  template<class U>
  void set_cnn_decoders(const std::vector<boost::shared_ptr<reverse_cnn_layer_model<U, dims> > >& layers) {
    _cnn_decoders.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _cnn_decoders[i] = boost::make_shared<reverse_cnn_layer_t>(*layers[i]);
  }

  v_reverse_cnn_layer_t& cnn_decoders() {
    return _cnn_decoders;
  }

  const v_reverse_cnn_layer_t& cnn_decoders() const {
    return _cnn_decoders;
  }

  template<class U>
  void append_cnn_decoder(const reverse_cnn_layer_model<U, dims>& layer) {
    _cnn_decoders.push_back(boost::make_shared<reverse_cnn_layer_t>(layer));
  }

  template<class U>
  void set_nn_encoders(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _nn_encoders.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _nn_encoders[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& nn_encoders() {
    return _nn_encoders;
  }

  const v_nn_layer_t& nn_encoders() const {
    return _nn_encoders;
  }

  template<class U>
  void append_nn_encoder(const nn_layer_model<U>& layer) {
    _nn_encoders.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  template<class U>
  void set_nn_decoders(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _nn_decoders.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _nn_decoders[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& nn_decoders() {
    return _nn_decoders;
  }

  const v_nn_layer_t& nn_decoders() const {
    return _nn_decoders;
  }

  template<class U>
  void append_nn_decoder(const nn_layer_model<U>& layer) {
    _nn_decoders.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  dim_t inputs_size() const {
    assert(_cnn_encoders.size());
    return _cnn_encoders[0]->visibles_size();
  }

  size_t inputs_count() const {
    return inputs_size().prod();
  }

  dim_t outputs_size() const {
    assert(_cnn_decoders.size());
    return _cnn_decoders[_cnn_decoders.size() - 1]->visibles_size();
  }

  size_t outputs_count() const {
    return outputs_size().prod();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_MODEL_HPP_ */
