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
#include <tbblas/deeplearn/nn_base.hpp>

#include <tbblas/deeplearn/opt/type_traits.hpp>
#include <tbblas/deeplearn/opt/void_trainer.hpp>

#include <boost/make_shared.hpp>
#include <boost/ref.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {
// Forward declaration for hessian_free
template<class Network>
class hessian_free2;
}

template<class T, class Trainer = opt::void_trainer<T>, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class nn : public virtual nn_base<T>, public Trainer {
  friend class opt::hessian_free2<nn<T, Trainer, Enable> >;

  typedef T value_t;
  typedef Trainer trainer_t;

  typedef tbblas::tensor<value_t, 1, true> vector_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  typedef nn_layer<value_t, trainer_t> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;
  typedef nn_model<value_t> model_t;

public:
  matrix_t _Rs1, _Rs2, _Ra1, _a1, _a2;

protected:
  model_t& _model;
  v_nn_layer_t _layers;
  tbblas::deeplearn::objective_function _objective;
  vector_t _parameters, _gradient, _Rgradient;
  matrix_t _inputs, _outputs, _Routputs;
  bool _parameters_expired, _gradient_expired, _Rgradient_expired, _is_encoder;

public:
  nn(model_t& model) : _model(model), _objective(objective_function::SSD), _parameters_expired(true), _gradient_expired(true), _Rgradient_expired(true), _is_encoder(false) {
    if (model.layers().size() == 0)
      throw std::runtime_error("At least one layer required to build a neural network.");

    _layers.resize(model.layers().size());
    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.layers()[i]), this);
    }

    _Rgradient = _gradient = _parameters = zeros<value_t>(_model.parameter_count());
  }

private:
  nn(const nn&);

public:
  void set_is_encoder(bool is_encoder) {
    _is_encoder = is_encoder;
    assert(_layers.size() % 2 == 0);
  }

  virtual void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective = objective;
    _layers[_layers.size() - 1]->set_objective_function(objective);
  }

  virtual tbblas::deeplearn::objective_function objective_function() const {
    assert(_layers[_layers.size() - 1]->objective_function() == _objective);
    return _objective;
  }

  virtual void set_sensitivity_ratio(const value_t& ratio) {
    _layers[_layers.size() - 1]->set_sensitivity_ratio(ratio);
  }

  virtual value_t sensitivity_ratio() const {
    return _layers[_layers.size() - 1]->sensitivity_ratio();
  }

  virtual void set_dropout_rate(int iLayer, const value_t& rate) {
    if (iLayer > 0 && iLayer < _layers.size())
      _layers[iLayer]->set_dropout_rate(rate);
  }

  vector_t& parameters() {
    if (_parameters_expired) {
      size_t offset = 0;
      for (size_t i = 0; i < _layers.size(); ++i)
        offset = _layers[i]->get_parameters(_parameters, offset, _is_encoder && i >= _layers.size() / 2);
      assert(offset == _parameters.count());
      assert(_gradient.size() == _parameters.size());
    }
    _parameters_expired = false;

    return _parameters;
  }

  void set_parameters(vector_t& parameters) {
    size_t offset = 0;
    for (size_t i = 0; i < _layers.size(); ++i) {
      offset = _layers[i]->set_parameters(parameters, offset, _is_encoder && i >= _layers.size() / 2);
    }
    _parameters = parameters;
    _parameters_expired = false;
  }

  value_t loss(matrix_t& target) {
    infer_hiddens();

    switch (objective_function()) {
    case objective_function::SSD:
      return 0.5 * dot(_outputs - target, _outputs - target);

    case objective_function::SenSpe:
      {
        const value_t sen = dot(_outputs - target, (_outputs - target) * (target > 0.5)) / sum(target > 0.5);
        const value_t spe = dot(_outputs - target, (_outputs - target) * (target < 0.5)) / sum(target < 0.5);
        return sensitivity_ratio() * sen + (1.0 - sensitivity_ratio()) * spe;
      }
    }
    assert(0);
    return -1;
  }

  virtual void infer_hiddens(bool dropout = false) {
    _layers[0]->visibles() = _inputs;
    _layers[0]->normalize_visibles();

    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i]->infer_hiddens(dropout);
      if (i + 1 < _layers.size())
        _layers[i + 1]->visibles() = _layers[i]->hiddens();
    }
    _outputs = _layers[_layers.size() - 1]->hiddens();
  }

  void infer_Rhiddens(bool dropout = false) {
    _layers[0]->Rvisibles() = zeros<value_t>(_inputs.size());
    _layers[0]->visibles() = _inputs;
    _layers[0]->normalize_visibles();

    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i]->infer_Rhiddens(dropout);
      if (i + 1 < _layers.size()) {
        _layers[i + 1]->visibles() = _layers[i]->hiddens();
        _layers[i + 1]->Rvisibles() = _layers[i]->Rhiddens();
      }
    }
    _outputs = _layers[_layers.size() - 1]->hiddens();
    _Routputs = _layers[_layers.size() - 1]->Rhiddens();
  }

  void reset_gradient() {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->reset_gradient();
    _gradient_expired = true;
  }

  void reset_Rgradient() {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->reset_Rgradient();
    _Rgradient_expired = true;
  }

  // Doesn't requires the hidden units to be inferred
  virtual void update_gradient(matrix_t& target) {

    // Infer hiddens
    infer_hiddens(true);

    // Calculate deltas of the target layer and update gradient
    _layers[_layers.size() - 1]->calculate_deltas(target);
    _layers[_layers.size() - 1]->update_gradient();

    // Perform back propagation and update gradients
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i + 1]->backprop_visible_deltas();
      _layers[i]->backprop_hidden_deltas(_layers[i + 1]->visible_deltas());
      _layers[i]->update_gradient();
    }

    _gradient_expired = true;
  }

  // Doesn't requires the hidden units to be inferred
  void update_Gv(vector_t& vector, matrix_t& target) {

    size_t offset = 0;
    for (size_t i = 0; i < _layers.size(); ++i) {
      offset = _layers[i]->set_vector(vector, offset, _is_encoder && i >= _layers.size() / 2);
    }

    // Infer hiddens
    infer_Rhiddens();

    // Calculate deltas of the target layer and update gradient
    _layers[_layers.size() - 1]->calculate_Rdeltas(target);
    _Rs2 = _layers[_layers.size() - 1]->_Rs;
    _a2 = _layers[_layers.size() - 1]->visibles();
    _layers[_layers.size() - 1]->update_Rgradient();

    // Perform back propagation and update gradients
    for (int i = _layers.size() - 2; i >= 0; --i) {
      _layers[i + 1]->backprop_visible_Rdeltas();
      _Ra1 = _layers[i + 1]->visible_Rdeltas();
      _layers[i]->backprop_hidden_Rdeltas(_layers[i + 1]->visible_Rdeltas());
      _Rs1 = _layers[i]->_Rs;
      _a1 = _layers[i]->visibles();
      _layers[i]->update_Rgradient();
    }

    _Rgradient_expired = true;
  }

  vector_t& gradient(value_t weightcost) {
    if (_gradient_expired) {
      size_t offset = 0;
      for (size_t i = 0; i < _layers.size(); ++i) {
        offset = _layers[i]->get_gradient(_gradient, offset, weightcost, _is_encoder && i >= _layers.size() / 2);
      }
      assert(offset == _parameters.count());
      assert(_gradient.size() == _parameters.size());
    }
    _gradient_expired = false;
    return _gradient;
  }

  vector_t& Rgradient(value_t weightcost) {
    if (_Rgradient_expired) {
      size_t offset = 0;
      for (size_t i = 0; i < _layers.size(); ++i) {
        offset = _layers[i]->get_Rgradient(_Rgradient, offset, weightcost, _is_encoder && i >= _layers.size() / 2);
      }
      assert(offset == _parameters.count());
      assert(_Rgradient.size() == _parameters.size());
    }
    _Rgradient_expired = false;
    return _Rgradient;
  }

  virtual void update_model(value_t weightcost) {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->update_model(weightcost);

    _gradient_expired = true;
    _parameters_expired = true;
  }

  // Does not requires the hidden units to be inferred
  virtual void update(matrix_t& target, value_t weightcost = 0) {
    update_gradient(target);
    update_model(weightcost);
  }

  virtual matrix_t& visibles() {
    return _inputs;
  }

  virtual matrix_t& hiddens() {
    return _outputs;
  }

  matrix_t& Rhiddens() {
    return _Routputs;
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_NN_HPP_ */
