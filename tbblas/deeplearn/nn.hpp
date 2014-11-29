/*
 * nn.hpp
 *
 *  Created on: 2014-08-15
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_NN_HPP_
#define TBBLAS_DEEPLEARN_NN_HPP_

#include <tbblas/deeplearn/nn_model.hpp>
#include <tbblas/deeplearn/nn_layer.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T>
class nn {
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  typedef nn_layer<value_t> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;
  typedef nn_model<value_t> model_t;

protected:
  model_t& _model;
  v_nn_layer_t _layers;

public:
  nn(model_t& model) : _model(model) {
    if (model.layers().size() == 0)
      throw std::runtime_error("At least one layer required to build a neural network.");

    _layers.resize(model.layers().size());
    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.layers()[i]));
    }
  }

private:
  nn(const nn<T>&);

public:
  void normalize_visibles() {
    _layers[0]->normalize_visibles();
  }

  void infer_hiddens() {
    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i]->infer_hiddens();
      if (i + 1 < _layers.size())
        _layers[i + 1]->visibles() = _layers[i]->hiddens();
    }
  }

//  void init_gradient_updates(value_t epsilon, value_t momentum, value_t weightcost) {
//    for (size_t i = 0; i < _layers.size(); ++i)
//      _layers[i]->init_gradient_updates(epsilon, momentum, weightcost);
//  }

  // requires the hidden units to be inferred
  void update_gradient(matrix_t& target) {
    _layers[_layers.size() - 1]->calculate_deltas(target);
    _layers[_layers.size() - 1]->update_gradient();

    // Perform back propagation
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i + 1]->backprop_visible_deltas();
      _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
      _layers[i]->update_gradient();
    }
  }

  void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->momentum_step(epsilon, momentum, weightcost);
  }

  // requires the hidden units to be inferred
  void momentum_update(matrix_t& target, value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    _layers[_layers.size() - 1]->calculate_deltas(target);

    // Perform back propagation
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i + 1]->backprop_visible_deltas();
      _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
    }

    // Update gradients
    _layers[_layers.size() - 1]->momentum_update(epsilon, momentum, weightcost);
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i]->momentum_update(epsilon, momentum, weightcost);
    }
  }

  void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->adadelta_step(epsilon, momentum, weightcost);
  }

  // requires the hidden units to be inferred
  void adadelta_update(matrix_t& target, value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    _layers[_layers.size() - 1]->calculate_deltas(target);

    // Perform back propagation
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i + 1]->backprop_visible_deltas();
      _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
    }

    // Update gradients
    _layers[_layers.size() - 1]->adadelta_update(epsilon, momentum, weightcost);
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i]->adadelta_update(epsilon, momentum, weightcost);
    }
  }

  matrix_t& visibles() {
    return _layers[0]->visibles();
  }

  matrix_t& hiddens() {
    return _layers[_layers.size() - 1]->hiddens();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_NN_HPP_ */
