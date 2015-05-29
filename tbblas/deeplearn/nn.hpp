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

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, class Trainer = opt::void_trainer<T>, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class nn : public virtual nn_base<T>, public Trainer {
  typedef T value_t;
  typedef Trainer trainer_t;

  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  typedef nn_layer<value_t, trainer_t> nn_layer_t;
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
      _layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.layers()[i]), this);
    }
  }

private:
  nn(const nn&);

public:
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

  virtual void normalize_visibles() {
    _layers[0]->normalize_visibles();
  }

  virtual void infer_hiddens(bool dropout = false) {
    for (size_t i = 0; i < _layers.size(); ++i) {
      _layers[i]->infer_hiddens(dropout);
      if (i + 1 < _layers.size())
        _layers[i + 1]->visibles() = _layers[i]->hiddens();
    }
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
  }

  virtual void update_model(value_t weightcost) {
    for (size_t i = 0; i < _layers.size(); ++i)
      _layers[i]->update_model(weightcost);
  }

  // Does not requires the hidden units to be inferred
  virtual void update(matrix_t& target, value_t weightcost = 0) {
    update_gradient(target);
    update_model(weightcost);
  }

  virtual matrix_t& visibles() {
    return _layers[0]->visibles();
  }

  virtual matrix_t& hiddens() {
    return _layers[_layers.size() - 1]->hiddens();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_NN_HPP_ */
