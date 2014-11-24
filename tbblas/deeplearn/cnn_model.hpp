/*
 * cnn_model.hpp
 *
 *  Created on: Aug 19, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CNN_MODEL_HPP_
#define TBBLAS_DEEPLEARN_CNN_MODEL_HPP_

#include <tbblas/deeplearn/cnn_layer_model.hpp>
#include <tbblas/deeplearn/nn_layer_model.hpp>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class cnn_model {
public:
  typedef T value_t;
  static const unsigned dimCount = dims;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef cnn_layer_model<T, dims> cnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

protected:
  v_cnn_layer_t _cnn_layers;
  v_nn_layer_t _nn_layers;

public:
  cnn_model() { }

  cnn_model(const cnn_model<T, dims>& model) {
    set_cnn_layers(model.cnn_layers());
    set_nn_layers(model.nn_layers());
  }

  virtual ~cnn_model() { }

public:
  template<class U>
  void set_cnn_layers(const std::vector<boost::shared_ptr<cnn_layer_model<U, dims> > >& layers) {
    _cnn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _cnn_layers[i] = boost::make_shared<cnn_layer_t>(*layers[i]);
  }

  v_cnn_layer_t& cnn_layers() {
    return _cnn_layers;
  }

  const v_cnn_layer_t& cnn_layers() const {
    return _cnn_layers;
  }

  template<class U>
  void append_cnn_layer(const cnn_layer_model<U, dims>& layer) {
    _cnn_layers.push_back(boost::make_shared<cnn_layer_t>(layer));
  }

  template<class U>
  void set_nn_layers(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _nn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _nn_layers[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& nn_layers() {
    return _nn_layers;
  }

  const v_nn_layer_t& nn_layers() const {
    return _nn_layers;
  }

  template<class U>
  void append_nn_layer(const nn_layer_model<U>& layer) {
    _nn_layers.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  dim_t input_size() const {
    assert(_cnn_layers.size());
    return _cnn_layers[0]->input_size();
  }

  size_t visibles_count() const {
    assert(_cnn_layers.size());
    return _cnn_layers[0]->visibles_count();
  }

  size_t hiddens_count() const {
    assert(_nn_layers.size());
    return _nn_layers[_nn_layers.size() - 1]->hiddens_count();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_MODEL_HPP_ */
