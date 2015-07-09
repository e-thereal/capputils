/*
 * encoder_model.hpp
 *
 *  Created on: Jan 02, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_ENCODER_MODEL_HPP_
#define TBBLAS_DEEPLEARN_ENCODER_MODEL_HPP_

#include <tbblas/deeplearn/cnn_layer_model.hpp>
#include <tbblas/deeplearn/dnn_layer_model.hpp>
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
  typedef dnn_layer_model<T, dims> dnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;
  typedef std::vector<boost::shared_ptr<dnn_layer_t> > v_dnn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

protected:
  v_cnn_layer_t _cnn_encoders, _cnn_shortcuts;
  v_dnn_layer_t _dnn_decoders, _dnn_shortcuts;
  v_nn_layer_t _nn_encoders, _nn_decoders;

public:
  encoder_model() { }

  encoder_model(const encoder_model<T, dims>& model) {
    set_cnn_encoders(model.cnn_encoders());
    set_dnn_decoders(model.dnn_decoders());
    set_cnn_shortcuts(model.cnn_shortcuts());
    set_dnn_shortcuts(model.dnn_shortcuts());
    set_nn_encoders(model.nn_encoders());
    set_nn_decoders(model.nn_decoders());
  }

  virtual ~encoder_model() { }

public:

  /*** CNN encoders ***/

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

  /*** CNN decoders***/

  template<class U>
  void set_dnn_decoders(const std::vector<boost::shared_ptr<dnn_layer_model<U, dims> > >& layers) {
    _dnn_decoders.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _dnn_decoders[i] = boost::make_shared<dnn_layer_t>(*layers[i]);
  }

  v_dnn_layer_t& dnn_decoders() {
    return _dnn_decoders;
  }

  const v_dnn_layer_t& dnn_decoders() const {
    return _dnn_decoders;
  }

  template<class U>
  void append_dnn_decoder(const dnn_layer_model<U, dims>& layer) {
    _dnn_decoders.push_back(boost::make_shared<dnn_layer_t>(layer));
  }

  template<class U>
  void set_nn_encoders(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _nn_encoders.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _nn_encoders[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  /*** CNN shortcuts ***/

  template<class U>
  void set_cnn_shortcuts(const std::vector<boost::shared_ptr<cnn_layer_model<U, dims> > >& layers) {
    _cnn_shortcuts.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _cnn_shortcuts[i] = boost::make_shared<cnn_layer_t>(*layers[i]);
  }

  v_cnn_layer_t& cnn_shortcuts() {
    return _cnn_shortcuts;
  }

  const v_cnn_layer_t& cnn_shortcuts() const {
    return _cnn_shortcuts;
  }

  template<class U>
  void append_cnn_shortcut(const cnn_layer_model<U, dims>& layer) {
    _cnn_shortcuts.push_back(boost::make_shared<cnn_layer_t>(layer));
  }

  /*** DNN shortcuts ***/

  template<class U>
  void set_dnn_shortcuts(const std::vector<boost::shared_ptr<dnn_layer_model<U, dims> > >& layers) {
    _dnn_shortcuts.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _dnn_shortcuts[i] = boost::make_shared<dnn_layer_t>(*layers[i]);
  }

  v_dnn_layer_t& dnn_shortcuts() {
    return _dnn_shortcuts;
  }

  const v_dnn_layer_t& dnn_shortcuts() const {
    return _dnn_shortcuts;
  }

  template<class U>
  void append_dnn_shortcut(const dnn_layer_model<U, dims>& layer) {
    _dnn_shortcuts.push_back(boost::make_shared<dnn_layer_t>(layer));
  }

  /*** NN layers ***/

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

  /*** General properties ***/

  size_t parameter_count() const {
    // Get number of parameters
    size_t count = 0;
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      count += _cnn_encoders[i]->parameter_count();

    for (size_t i = 0; i < _cnn_shortcuts.size(); ++i)
      count += _cnn_shortcuts[i]->parameter_count(true);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      count += _dnn_decoders[i]->parameter_count();

    for (size_t i = 0; i < _dnn_shortcuts.size(); ++i)
      count += _dnn_shortcuts[i]->parameter_count(true);

    return count;
  }

  void set_shared_bias(bool shared) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->set_shared_bias(shared);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->set_shared_bias(shared);
  }

  dim_t inputs_size() const {
    assert(_cnn_encoders.size());
    return _cnn_encoders[0]->visibles_size();
  }

  size_t inputs_count() const {
    return inputs_size().prod();
  }

  dim_t outputs_size() const {
    assert(_dnn_decoders.size());
    return _dnn_decoders[_dnn_decoders.size() - 1]->inputs_size();
  }

  size_t outputs_count() const {
    return outputs_size().prod();
  }

  size_t layer_count() const {
    return _cnn_encoders.size() + _nn_encoders.size() + _nn_decoders.size() + _dnn_decoders.size();
  }

  bool has_cnn_shortcuts() const {
    return _cnn_shortcuts.size();
  }

  bool has_dnn_shortcuts() const {
    return _dnn_shortcuts.size();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_MODEL_HPP_ */
