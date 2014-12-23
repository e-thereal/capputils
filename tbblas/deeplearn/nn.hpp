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
  tbblas::deeplearn::objective_function _objective;

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
  void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective = objective;
    _layers[_layers.size() - 1]->set_objective_function(objective);
  }

  tbblas::deeplearn::objective_function objective_function() const {
    assert(_layers[_layers.size() - 1]->objective_function() == _objective);
    return _objective;
  }

  void set_sensitivity_ratio(const value_t& ratio) {
    _layers[_layers.size() - 1]->set_sensitivity_ratio(ratio);
  }

  value_t sensitivity_ratio() const {
    return _layers[_layers.size() - 1]->sensitivity_ratio();
  }

  void set_dropout_rate(int iLayer, const value_t& rate) {
    if (iLayer > 0 && iLayer < _layers.size())
      _layers[iLayer]->set_dropout_rate(rate);
  }

  void normalize_visibles() {
    _layers[0]->normalize_visibles();
  }

  void infer_hiddens(bool dropout = false) {
    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i]->infer_hiddens(dropout);
      if (i + 1 < _layers.size())
        _layers[i + 1]->visibles() = _layers[i]->hiddens();
    }
  }

  // Doesn't requires the hidden units to be inferred
  void update_gradient(matrix_t& target) {

    // Infer hiddens
    infer_hiddens(true);

    switch(_objective) {
    case tbblas::deeplearn::objective_function::SSD:
    case tbblas::deeplearn::objective_function::SenSpe:

      // Calculate deltas of the target layer and update gradient
      _layers[_layers.size() - 1]->calculate_deltas(target);
      _layers[_layers.size() - 1]->update_gradient();

      // Perform back propagation and update gradients
      for (int i = _layers.size() - 2; i >= 0; --i) {
        _layers[i + 1]->backprop_visible_deltas();
        _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
        _layers[i]->update_gradient();
      }
      break;

    case tbblas::deeplearn::objective_function::DSC:
      {
        // DSC = u/v
        // u = 2 * sum(y_j * d_j)
        // v = sum(d_j) + sum(y_j)

        // Pre-calculate u and v
        value_t u = 2 * sum(hiddens() * target);
        value_t v = sum(hiddens()) + sum(target);

        // Calculate u deltas of the target layer and update u gradient
        _layers[_layers.size() - 1]->calculate_u_deltas(target);
        _layers[_layers.size() - 1]->update_u_gradient();

        // Perform back propagation and update u gradients
        for (int i = _layers.size() - 2; i >= 0; --i) {
          _layers[i + 1]->backprop_visible_deltas();
          _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
          _layers[i]->update_u_gradient();
        }

        // Calculate v deltas of the target layer and update v gradient
        _layers[_layers.size() - 1]->calculate_v_deltas(target);
        _layers[_layers.size() - 1]->update_v_gradient(u, v);

        // Perform back propagation and update v gradients
        for (int i = _layers.size() - 2; i >= 0; --i) {
          _layers[i + 1]->backprop_visible_deltas();
          _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
          _layers[i]->update_v_gradient(u, v);
        }
      }
      break;

    case tbblas::deeplearn::objective_function::DSC2:
      {
        // DSC = u/v
        // u = 2 * sum(y_j * d_j)
        // v = sum(d_j) + sum(y_j)

        // Pre-calculate u and v
        value_t u = 2 * sum((value_t(1) + value_t(-1) * (hiddens() - target) * (hiddens() - target)) * target);
        value_t v = sum(hiddens() * hiddens()) + sum(target);

        // Calculate u deltas of the target layer and update u gradient
        _layers[_layers.size() - 1]->calculate_u_deltas(target);
        _layers[_layers.size() - 1]->update_u_gradient();

        // Perform back propagation and update u gradients
        for (int i = _layers.size() - 2; i >= 0; --i) {
          _layers[i + 1]->backprop_visible_deltas();
          _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
          _layers[i]->update_u_gradient();
        }

        // Calculate v deltas of the target layer and update v gradient
        _layers[_layers.size() - 1]->calculate_v_deltas(target);
        _layers[_layers.size() - 1]->update_v_gradient(u, v);

        // Perform back propagation and update v gradients
        for (int i = _layers.size() - 2; i >= 0; --i) {
          _layers[i + 1]->backprop_visible_deltas();
          _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
          _layers[i]->update_v_gradient(u, v);
        }
      }
      break;

    default:
      throw std::runtime_error("Unsupported objective function in nn::update_gradient(target)");
    }
  }

  // Apply gradient to the current parameter set using the momentum method
  void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->momentum_step(epsilon, momentum, weightcost);
  }

  void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->adadelta_step(epsilon, momentum, weightcost);
  }

  // Does not requires the hidden units to be inferred
  void momentum_update(matrix_t& target, value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    update_gradient(target);
    momentum_step(epsilon, momentum, weightcost);
  }

  // requires the hidden units to be inferred
  void adadelta_update(matrix_t& target, value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    update_gradient(target);
    adadelta_step(epsilon, momentum, weightcost);
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
