/*
 * encoder.hpp
 *
 *  Created on: 2015-01-06
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_ENCODER_HPP_
#define TBBLAS_DEEPLEARN_ENCODER_HPP_

#include <tbblas/deeplearn/encoder_model.hpp>
#include <tbblas/deeplearn/encoder_base.hpp>
#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>
#include <tbblas/deeplearn/dnn_layer.hpp>
#include <tbblas/deeplearn/cdnn_layer.hpp>
#include <tbblas/deeplearn/ddnn_layer.hpp>

#include <tbblas/deeplearn/opt/type_traits.hpp>
#include <tbblas/deeplearn/opt/void_trainer.hpp>

#include <tbblas/context.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/swap.hpp>

#include <stdexcept>

#include <boost/ref.hpp>

namespace tbblas {

namespace deeplearn {

namespace opt {
// Forward declaration for hessian_free
template<class T, unsigned dims>
class hessian_free;
}

// If you end up here, the specified Trainer most likely does not fulfill the opt::is_trainer<> condition

template<class T, unsigned dims, class Trainer = opt::void_trainer<T>, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class encoder : public virtual encoder_base<T, dims>, public Trainer {

  friend class opt::hessian_free<T, dims>;

  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef Trainer trainer_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, 1, true> vector_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef nn_layer<value_t, trainer_t> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

  typedef cnn_layer<value_t, dimCount, trainer_t> cnn_layer_t;
  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;

  typedef dnn_layer<value_t, dimCount, trainer_t> dnn_layer_t;
  typedef std::vector<boost::shared_ptr<dnn_layer_t> > v_dnn_layer_t;

  typedef cdnn_layer<value_t, dimCount, trainer_t> cdnn_layer_t;
  typedef ddnn_layer<value_t, dimCount, trainer_t> ddnn_layer_t;

  typedef encoder_model<value_t, dimCount> model_t;

//public:
//  tensor_t _Rs1, _Rs2, _Ra1, _a1, _a2;

protected:
  model_t& _model;
  v_cnn_layer_t _cnn_encoders;
  v_dnn_layer_t _dnn_decoders;
  v_nn_layer_t _nn_encoders, _nn_decoders;
  tbblas::deeplearn::objective_function _objective;
  tensor_t _inputs, _outputs;
  vector_t _gradient, _parameters, _weight_mask;
  bool _gradient_expired, _parameters_expired, _weight_mask_expired;

public:
  encoder(model_t& model, const dim_t& patch_count = seq<dimCount>(1)) : _model(model), _gradient_expired(true), _parameters_expired(true), _weight_mask_expired(true) {
    if (model.cnn_encoders().size() == 0 || model.dnn_decoders().size() == 0)
      throw std::runtime_error("At least one convolutional encoder and decoder is required to build a convolutional encoder neural network.");

    if (model.cnn_encoders().size() != model.dnn_decoders().size())
      throw std::runtime_error("An encoder network needs to have the same number of encoders and decoders.");

    if (model.dnn_shortcuts().size() && model.dnn_decoders().size() != model.dnn_shortcuts().size() + 1) {
      throw std::runtime_error("An encoder network must have either no shortcut connections or one shortcut less than decoders.");
    }

    _cnn_encoders.resize(model.cnn_encoders().size());
    for (size_t i = 0; i < _cnn_encoders.size(); ++i) {
      _cnn_encoders[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.cnn_encoders()[i]), this, boost::ref(patch_count));
    }

    _dnn_decoders.resize(model.dnn_decoders().size());
    for (size_t i = 0; i < _dnn_decoders.size(); ++i) {
      if (i < model.cnn_shortcuts().size()) {
        std::cout << "Adding cdnn shortcut connection" << std::endl;
        _dnn_decoders[i] = boost::make_shared<cdnn_layer_t>(boost::ref(*model.cnn_shortcuts()[_cnn_encoders.size() - i - 2]), boost::ref(*model.dnn_decoders()[i]), boost::ref(*_cnn_encoders[_cnn_encoders.size() - i - 2]), this, boost::ref(patch_count));
      } else if (i > 0 && model.dnn_shortcuts().size()) {
        std::cout << "Adding ddnn shortcut connection" << std::endl;
        _dnn_decoders[i] = boost::make_shared<ddnn_layer_t>(boost::ref(*model.dnn_shortcuts()[i - 1]), boost::ref(*model.dnn_decoders()[i]), boost::ref(*_cnn_encoders[_cnn_encoders.size() - i - 1]), this, boost::ref(patch_count));
      } else {
        std::cout << "Adding regular decoder" << std::endl;
        _dnn_decoders[i] = boost::make_shared<dnn_layer_t>(boost::ref(*model.dnn_decoders()[i]), this, boost::ref(patch_count));
        _dnn_decoders[i]->tie_switches(*_cnn_encoders[_cnn_encoders.size() - i - 1]);
      }
    }

    _nn_encoders.resize(model.nn_encoders().size());
    for (size_t i = 0; i < _nn_encoders.size(); ++i) {
      _nn_encoders[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_encoders()[i]), this);
    }

    _nn_decoders.resize(model.nn_decoders().size());
    for (size_t i = 0; i < _nn_decoders.size(); ++i) {
      _nn_decoders[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_decoders()[i]), this);
    }

    if (_nn_encoders.size() != _nn_decoders.size())
      throw std::runtime_error("The model needs to have the same number of dense encoder and decoder layers.");
  }

private:
  encoder(const encoder<T, dims, Trainer, Enable>&);

public:
//  virtual ~encoder() {
//    _dnn_decoders.clear();
//    _cnn_encoders.clear();
//  }

public:
  virtual void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective = objective;
    _dnn_decoders[_dnn_decoders.size() - 1]->set_objective_function(objective);
  }

  virtual tbblas::deeplearn::objective_function objective_function() const {
    assert(_dnn_decoders[_dnn_decoders.size() - 1]->objective_function() == _objective);
    return _objective;
  }

  virtual void set_sensitivity_ratio(const value_t& ratio) {
    _dnn_decoders[_dnn_decoders.size() - 1]->set_sensitivity_ratio(ratio);
  }

  virtual value_t sensitivity_ratio() const {
    return _dnn_decoders[_dnn_decoders.size() - 1]->sensitivity_ratio();
  }

  void set_dropout_rate(value_t rate) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->set_dropout_rate(rate);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->set_dropout_rate(rate);
  }

  void reinitialize_dropout() {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->reinitialize_dropout();

    for (size_t i = 0; i < _dnn_decoders.size() - 1; ++i)
      _dnn_decoders[i]->reinitialize_dropout();
  }

  virtual void write_model_to_host() {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->write_model_to_host();

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->write_model_to_host();

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->write_model_to_host();

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->write_model_to_host();
  }

  // Infer hidden units recursively
  virtual void infer_outputs() {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();

    infer_layer(0, _cnn_encoders.size() + _dnn_decoders.size() + _nn_encoders.size() + _nn_decoders.size() - 1);

    _dnn_decoders[_dnn_decoders.size() - 1]->diversify_visibles();
    _outputs = _dnn_decoders[_dnn_decoders.size() - 1]->visibles();
  }

  virtual void infer_layer(const size_t maxLayer) {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    infer_layer(0, maxLayer);
  }

  value_t loss(tensor_t& target) {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();

    infer_layer(0, _cnn_encoders.size() + _dnn_decoders.size() + _nn_encoders.size() + _nn_decoders.size() - 1);

    switch (objective_function()) {
    case objective_function::SSD:
      return 0.5 * dot(_outputs - target, _outputs - target) / (value_t)target.count();

    case objective_function::SenSpe:
      {
        const value_t sen = dot(_outputs - target, (_outputs - target) * (target > 0.5)) / sum(target > 0.5);
        const value_t spe = dot(_outputs - target, (_outputs - target) * (target < 0.5)) / sum(target < 0.5);
        return sensitivity_ratio() * sen + (1.0 - sensitivity_ratio()) * spe;
      }
    }

    return -1;
  }

  // Does not require the hidden units to be inferred
  virtual value_t update_gradient(tensor_t& target) {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    update_gradient(0, target);

    value_t error;

    switch (objective_function()) {
    case objective_function::SSD:
      error = 0.5 * dot(_outputs - target, _outputs - target) / (value_t)target.count();
      break;

    case objective_function::SenSpe:
      {
        const value_t sen = dot(_outputs - target, (_outputs - target) * (target > 0.5)) / sum(target > 0.5);
        const value_t spe = dot(_outputs - target, (_outputs - target) * (target < 0.5)) / sum(target < 0.5);
        error = sensitivity_ratio() * sen + (1.0 - sensitivity_ratio()) * spe;
      }
      break;
    }

    _gradient_expired = true;
    return error;
  }

  // The result will be saved to the gradient
  virtual void update_gv(vector_t& v, tensor_t& target) {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    update_gv_part1(0, v);                                                          // infer activations and first part of RWa part of Ra
    update_gv_part2(0, target);                                                     // Infer RDa

    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    update_gv_part3(0);                                                             // Restore activations and backprop RDa

    _gradient_expired = true;
  }

  vector_t& gradient(value_t weightcost = 0) {

    if (_gradient_expired) {

      // Resize gradient to hold everything
      _gradient.resize(seq(_model.parameter_count()));

      size_t offset = 0;
      for (size_t i = 0; i < _cnn_encoders.size(); ++i) {
        offset = _cnn_encoders[i]->collect_gradient(_gradient, offset, weightcost);
      }

      for (size_t i = 0; i < _dnn_decoders.size(); ++i) {
        offset = _dnn_decoders[i]->collect_gradient(_gradient, offset, weightcost);
      }

      assert(_nn_encoders.size() == 0 && _nn_decoders.size() == 0);

//    for (size_t i = 0; i < _nn_encoders.size(); ++i)
//      _nn_encoders[i]->update_model(weightcost);
//
//    for (size_t i = 0; i < _nn_decoders.size(); ++i)
//      _nn_decoders[i]->update_model(weightcost);
    }

    _gradient_expired = false;

    return _gradient;
  }

  void reset_gradient() {
    assert(_nn_encoders.size() == 0 && _nn_decoders.size() == 0);

    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->reset_gradient();

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->reset_gradient();

    _gradient_expired = true;
  }

  void update_model(vector_t& delta) {
    size_t offset = 0;
    for (size_t i = 0; i < _cnn_encoders.size(); ++i) {
      offset = _cnn_encoders[i]->update_model(delta, offset);
    }

    for (size_t i = 0; i < _dnn_decoders.size(); ++i) {
      offset = _dnn_decoders[i]->update_model(delta, offset);
    }

    assert(_nn_encoders.size() == 0 && _nn_decoders.size() == 0);

    _parameters_expired = true;
  }

  vector_t& parameters() {
    if (_parameters_expired) {

      // Resize gradient to hold everything
      _parameters.resize(seq(_model.parameter_count()));

      size_t offset = 0;
      for (size_t i = 0; i < _cnn_encoders.size(); ++i)
        offset = _cnn_encoders[i]->get_parameters(_parameters, offset);

      for (size_t i = 0; i < _dnn_decoders.size(); ++i)
        offset = _dnn_decoders[i]->get_parameters(_parameters, offset);

      assert(_nn_encoders.size() == 0 && _nn_decoders.size() == 0);
    }

    _parameters_expired = false;

    return _parameters;
  }

  vector_t& weight_mask() {
    if (_weight_mask_expired) {
      // Resize gradient to hold everything
      _weight_mask.resize(seq(_model.parameter_count()));

      size_t offset = 0;
      for (size_t i = 0; i < _cnn_encoders.size(); ++i)
        offset = _cnn_encoders[i]->get_weight_mask(_weight_mask, offset);

      for (size_t i = 0; i < _dnn_decoders.size(); ++i)
        offset = _dnn_decoders[i]->get_weight_mask(_weight_mask, offset);
    }
    _weight_mask_expired = false;
    return _weight_mask;
  }

  void set_parameters(vector_t& parameters) {
    size_t offset = 0;
    for (size_t i = 0; i < _cnn_encoders.size(); ++i) {
      offset = _cnn_encoders[i]->set_parameters(parameters, offset);
    }

    for (size_t i = 0; i < _dnn_decoders.size(); ++i) {
      offset = _dnn_decoders[i]->set_parameters(parameters, offset);
    }

    assert(_nn_encoders.size() == 0 && _nn_decoders.size() == 0);
    _parameters = parameters;
    _parameters_expired = false;
  }

  virtual void infer_deltas(const size_t layer, tensor_t& target) {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    update_gradient(0, target, true);

    if (layer < _cnn_encoders.size()) {
      _outputs = _dnn_decoders[layer]->hiddens();
    } else if (layer - _cnn_encoders.size() < _nn_encoders.size()) {

    } else if (layer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

    } else if (layer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {
      _outputs = _dnn_decoders[layer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size()]->visibles();
    } else {

      // should never happen
      assert(0);
    }
  }

  virtual void update_model(value_t weightcost) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->update_model(weightcost);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->update_model(weightcost);

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->update_model(weightcost);

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->update_model(weightcost);

    _parameters_expired = true;
  }

  virtual void set_batch_length(int layer, int length) {
    if (layer < _cnn_encoders.size())
      _cnn_encoders[layer]->set_batch_length(length);
    else if (layer - _cnn_encoders.size() < _dnn_decoders.size())
      _dnn_decoders[layer - _cnn_encoders.size()]->set_batch_length(length);
  }

  virtual tensor_t& inputs() {
    return _inputs;
  }

  virtual tensor_t& outputs() {
    return _outputs;
  }

protected:
#ifdef ENCODER_INFER_OUTPUTS
  void infer_outputs(int iLayer) {

    if (iLayer < _cnn_encoders.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_encoders[i]->infer_hiddens();

      // If more convolutional encoders, ...
      if (i + 1 < _cnn_encoders.size()) {

        // Transition to next layer and repeat
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();
        infer_outputs(iLayer + 1);
      } else {

        // If the model has dense layers, ...
        if (_nn_encoders.size()) {

          // Transition from convolutional model to dense model
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1, _cnn_encoders[i]->hiddens().count());
          infer_outputs(iLayer + 1);
        } else {

          // Transition to decoding layers
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();
          infer_outputs(iLayer + 1);
        }
      }
    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_encoders.size();

      _nn_encoders[i]->infer_hiddens();

      // If more dense encoders
      if (i + 1 < _nn_encoders.size()) {

        // Transition to next layer and repeat
        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();
        infer_outputs(iLayer + 1);
      } else {

        // Transition to dense decoding layer
        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();
        infer_outputs(iLayer + 1);
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();

      _nn_decoders[i]->infer_hiddens();

      // If more dense decoders
      if (i + 1 < _nn_decoders.size()) {

        // Transition to next layer and repeat
        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();
        infer_outputs(iLayer + 1);
      } else {

        // Transition to convolutional decoding layer
        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(), _dnn_decoders[0]->hiddens().size());
        infer_outputs(iLayer + 1);
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

      // Infer convolutional layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _dnn_decoders[i]->infer_visibles();

      // If more convolutional decoders, ...
      if (i + 1 < _dnn_decoders.size()) {

        // Transition to next layer and repeat
        _dnn_decoders[i + 1]->hiddens() = _dnn_decoders[i]->visibles();
        infer_outputs(iLayer + 1);
      }
    } else {

      // should never happen
      assert(0);
    }
  }
#endif

  void infer_layer(int iLayer, int maxLayer) {

    if (iLayer < _cnn_encoders.size()) {

      // Infer convolutional layer
      const int i = iLayer;
      _cnn_encoders[i]->infer_hiddens();

      // Early stopping
      if (iLayer >= maxLayer) {
        _outputs = _cnn_encoders[i]->hiddens();
        return;
      }

      // Transitions
      if (i + 1 < _cnn_encoders.size()) {                                           // If more convolutional encoders, ...
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();             // Transition to next layer and repeat
      } else {                                                                      // else
        if (_nn_encoders.size()) {                                                  // If the model has dense layers, ...
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1,     // Transition from convolutional model to dense model
              _cnn_encoders[i]->hiddens().count());
        } else {                                                                    // else
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();                // Transition to decoding layers
        }
      }
      infer_layer(iLayer + 1, maxLayer);
    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_encoders.size();
      _nn_encoders[i]->infer_hiddens();

      if (i + 1 < _nn_encoders.size()) {                                            // If more dense encoders
        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();               // Transition to next layer and repeat
      } else {
        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();                   // Transition to dense decoding layer
      }
      infer_layer(iLayer + 1, maxLayer);

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();
      _nn_decoders[i]->infer_hiddens();

      if (i + 1 < _nn_decoders.size()) {                                            // If more dense decoders
        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();               // Transition to next layer and repeat
      } else {
        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(),           // Transition to convolutional decoding layer
            _dnn_decoders[0]->hiddens().size());
      }
      infer_layer(iLayer + 1, maxLayer);

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

      // Infer convolutional layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();
      _dnn_decoders[i]->infer_visibles();

      // Early stopping
      if (iLayer >= maxLayer) {
        _outputs = _dnn_decoders[i]->visibles();
        return;
      }

      if (i + 1 < _dnn_decoders.size()) {                                           // If more convolutional decoders, ...
        _dnn_decoders[i + 1]->hiddens() = _dnn_decoders[i]->visibles();             // Transition to next layer and repeat
        infer_layer(iLayer + 1, maxLayer);
      } else {
        _outputs = _dnn_decoders[i]->visibles();
      }
    } else {

      // should never happen
      assert(0);
    }
  }

  void update_gradient(int iLayer, tensor_t& target, bool infer_only_deltas = false) {

    if (iLayer < _cnn_encoders.size()) {

        /*** INFER CONVOLUTIONAL ENCODING LAYER ***/

      const int i = iLayer;
      _cnn_encoders[i]->infer_hiddens();

      if (i + 1 < _cnn_encoders.size()) {                                           // If more convolutional encoders exists, ...
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();             // Transition to next layer and repeat
        update_gradient(iLayer + 1, target, infer_only_deltas);
        _cnn_encoders[i + 1]->backprop_visibles();                                  // Back-propagate errors

        // Handle shortcuts
        if (_model.has_dnn_shortcuts()) {
          boost::dynamic_pointer_cast<ddnn_layer_t>(_dnn_decoders[_dnn_decoders.size() - i - 1])->backprop_hidden_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles(), true);
        } else if ((int)_dnn_decoders.size() - i - 3 >= 0 &&_model.has_cnn_shortcuts()) {
          boost::dynamic_pointer_cast<cdnn_layer_t>(_dnn_decoders[_dnn_decoders.size() - i - 3])->backprop_hidden_deltas(); // this accumulates it with _cnn_encoders[i + 1] visibles
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles());
        } else {
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles());
        }

        if (!infer_only_deltas)
          _cnn_encoders[i]->update_gradient();
      } else {

        // If the model has dense encoding layers, ...
        if (_nn_encoders.size()) {

          // Transition from convolutional model to dense model
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1, _cnn_encoders[i]->hiddens().count());
          update_gradient(iLayer + 1, target, infer_only_deltas);

          // Transition back
          _nn_encoders[0]->backprop_visible_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(reshape(_nn_encoders[0]->visible_deltas(), _model.cnn_encoders()[i]->hiddens_size()));
          if (!infer_only_deltas)
            _cnn_encoders[i]->update_gradient();
        } else {

          // Transition to decoding layers
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();
          update_gradient(iLayer + 1, target, infer_only_deltas);

          // Transition back
          _dnn_decoders[0]->backprop_hiddens();
          _cnn_encoders[i]->backprop_hidden_deltas(_dnn_decoders[0]->hiddens());
          if (!infer_only_deltas)
            _cnn_encoders[i]->update_gradient();
        }
      }
    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

        /*** INFER DENSE ENCODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size();
      _nn_encoders[i]->infer_hiddens();

      // If there are more encoding layers, ...
      if (i + 1 < _nn_encoders.size()) {

        // Transition to next layer and repeat
        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();
        update_gradient(iLayer + 1, target, infer_only_deltas);

        _nn_encoders[i + 1]->backprop_visible_deltas();
        _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
        if (!infer_only_deltas)
          _nn_encoders[i]->update_gradient();
      } else {

        // Transition to decoding layer and repeat
        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();
        update_gradient(iLayer + 1, target, infer_only_deltas);

        _nn_decoders[0]->backprop_visible_deltas();
        _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
        if (!infer_only_deltas)
          _nn_encoders[i]->update_gradient();
      }
    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

        /*** INFER DENSE DECODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();
      _nn_decoders[i]->infer_hiddens();

      // If there are more decoding layers, ...
      if (i + 1 < _nn_decoders.size()) {

        // Transition to next layer and repeat
        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();
        update_gradient(iLayer + 1, target, infer_only_deltas);

        _nn_decoders[i + 1]->backprop_visible_deltas();
        _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
        if (!infer_only_deltas)
          _nn_decoders[i]->update_gradient();
      } else {

        // Transition to convolutional decoding layer and repeat
        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(), _dnn_decoders[0]->hiddens().size());
        update_gradient(iLayer + 1, target, infer_only_deltas);

        _dnn_decoders[0]->backprop_hiddens();
        _nn_decoders[i]->backprop_hidden_deltas(reshape(_dnn_decoders[0]->hiddens(), _nn_decoders[i]->hiddens().size()));
        if (!infer_only_deltas)
          _nn_decoders[i]->update_gradient();
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

        /*** INFER CONVOLUTIONAL DECODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _dnn_decoders[i]->infer_visibles();

      // If more convolutional decoders, ...
      if (i + 1 < _dnn_decoders.size()) {

        // Transition to next layer and repeat
        _dnn_decoders[i + 1]->hiddens() = _dnn_decoders[i]->visibles();
        update_gradient(iLayer + 1, target, infer_only_deltas);

        // Back-propagate errors
        _dnn_decoders[i + 1]->backprop_hiddens();
        _dnn_decoders[i]->backprop_visible_deltas(_dnn_decoders[i + 1]->hiddens());
        if (!infer_only_deltas)
          _dnn_decoders[i]->update_gradient();
      } else {
        _dnn_decoders[_dnn_decoders.size() - 1]->diversify_visibles();
        _outputs = _dnn_decoders[_dnn_decoders.size() - 1]->visibles();

        // Start back-propagation
        _dnn_decoders[i]->calculate_deltas(target);
        if (!infer_only_deltas)
          _dnn_decoders[i]->update_gradient();
      }
    } else {
      assert(0); // should never happen
    }
  }

  // Calculates Rs = RWa + Rb
  // In the forward pass, all the a are calculated. The backward pass calculates the WRa + Rb
  // At the end, the hidden units contain one part of the Ra calculation and the visible units still contain
  // the forward activation
  // The activation of the last layer is written to _output
  void update_gv_part1(int iLayer, vector_t& v) {
    if (iLayer < _cnn_encoders.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_encoders[i]->infer_hiddens(cnn_layer_t::DEFAULT | cnn_layer_t::BACKUP_ACTIVATION);

      if (i + 1 < _cnn_encoders.size()) {                                           // If more convolutional encoders exists, ...
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();             // Transition to next layer and repeat
      } else {                                                                      // else
        if (_nn_encoders.size()) {                                                  // If the model has dense encoding layers, ...
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1,     // Transition from convolutional model to dense model
              _cnn_encoders[i]->hiddens().count());
        } else {                                                                    // else
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();                // Transition to decoding layers
        }
      }

      update_gv_part1(iLayer + 1, v);                                               // Proceed to next layer
      _cnn_encoders[i]->infer_hiddens(cnn_layer_t::APPLY_BIAS);                     // Infer RWa

    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

      // Infer dense layer
//      const int i = iLayer - _cnn_encoders.size();
//      _nn_encoders[i]->infer_hiddens();
//
//      if (i + 1 < _nn_encoders.size()) {                                            // If there are more encoding layers, ...
//        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();               // Transition to next layer
//      } else {
//        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();                   // Transition to decoding layer and repeat
//      }
//      update_gv_part1(iLayer + 1, v);                                               // Proceed to next layer
//      _nn_encoders[i]->infer_hiddens(cnn_layer_t::APPLY_BIAS);                      // Infer RWa

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

      // Infer dense layer
//      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();
//      _nn_decoders[i]->infer_hiddens();
//
//      if (i + 1 < _nn_decoders.size()) {                                            // If there are more decoding layers, ...
//        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();               // Transition to next layer and repeat
//      } else {
//        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(),           // Transition to convolutional decoding layer and repeat
//            _dnn_decoders[0]->hiddens().size());
//      }
//      update_gv_part1(iLayer + 1, v);                                               // Proceed to next layer
//      _nn_decoders[i]->infer_hiddens(cnn_layer_t::APPLY_BIAS);                      // Infer RWa

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

      // Infer convolutional layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _dnn_decoders[i]->infer_visibles();

      if (i + 1 < _dnn_decoders.size()) {                                           // If more convolutional decoders, ...
        _dnn_decoders[i + 1]->hiddens() = _dnn_decoders[i]->visibles();             // Transition to next layer and repeat
        update_gv_part1(iLayer + 1, v);
        _dnn_decoders[i]->infer_visibles(cnn_layer_t::APPLY_BIAS);
      } else {
        _outputs = _dnn_decoders[i]->visibles();

        // Change model to calculate RWa + Rb in the backward pass
        swap(v, parameters());
        set_parameters(_parameters);

        _dnn_decoders[i]->infer_visibles(cnn_layer_t::APPLY_BIAS);
      }
    } else {
      assert(0); // should never happen
    }

    if (iLayer == 0) {
      // Swap parameters back
      swap(v, parameters());
      set_parameters(_parameters);
    }
  }

  // compute_gv_part2 then calculates Rs = Rs + WRa = RWa + WRa + Rb
  // Ra = Rs * f'(s), where f'(s) is backproped from the previous layer
  // swap hiddens and visibles at the transition of two layers -> Ra in the visibles and a in the hiddens
  // Also restore full hidden activation
  // The last layer calculates RDal
  void update_gv_part2(int iLayer, tensor_t& target) {

    if (iLayer < _cnn_encoders.size()) {

        /*** INFER CONVOLUTIONAL ENCODING LAYER ***/

      const int i = iLayer;

      if (iLayer == 0)
        _cnn_encoders[i]->visibles() = zeros<value_t>(_cnn_encoders[i]->visibles().size());
      _cnn_encoders[i]->infer_hiddens(cnn_layer_t::ACCUMULATE | cnn_layer_t::APPLY_DERIVATIVE);

      if (i + 1 < _cnn_encoders.size()) {                                           // If more convolutional encoders exists, ...
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();
      } else {
        if (_nn_encoders.size()) {                                                  // If the model has dense encoding layers, ...
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1,     // Transition from convolutional model to dense model
              _cnn_encoders[i]->hiddens().count());
        } else {
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();
        }
      }
      update_gv_part2(iLayer + 1, target);
      _cnn_encoders[i]->restore_hiddens();
    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

        /*** INFER DENSE ENCODING LAYER ***/

//      const int i = iLayer - _cnn_encoders.size();
//      _nn_encoders[i]->infer_hiddens();
//
//      // If there are more encoding layers, ...
//      if (i + 1 < _nn_encoders.size()) {
//
//        // Transition to next layer and repeat
//        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();
//        update_gradient(iLayer + 1, target, infer_only_deltas);
//
//        _nn_encoders[i + 1]->backprop_visible_deltas();
//        _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
//        if (!infer_only_deltas)
//          _nn_encoders[i]->update_gradient();
//      } else {
//
//        // Transition to decoding layer and repeat
//        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();
//        update_gradient(iLayer + 1, target, infer_only_deltas);
//
//        _nn_decoders[0]->backprop_visible_deltas();
//        _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
//        if (!infer_only_deltas)
//          _nn_encoders[i]->update_gradient();
//      }
    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

        /*** INFER DENSE DECODING LAYER ***/

//      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();
//      _nn_decoders[i]->infer_hiddens();
//
//      // If there are more decoding layers, ...
//      if (i + 1 < _nn_decoders.size()) {
//
//        // Transition to next layer and repeat
//        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();
//        update_gradient(iLayer + 1, target, infer_only_deltas);
//
//        _nn_decoders[i + 1]->backprop_visible_deltas();
//        _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
//        if (!infer_only_deltas)
//          _nn_decoders[i]->update_gradient();
//      } else {
//
//        // Transition to convolutional decoding layer and repeat
//        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(), _dnn_decoders[0]->hiddens().size());
//        update_gradient(iLayer + 1, target, infer_only_deltas);
//
//        _dnn_decoders[0]->backprop_hiddens();
//        _nn_decoders[i]->backprop_hidden_deltas(reshape(_dnn_decoders[0]->hiddens(), _nn_decoders[i]->hiddens().size()));
//        if (!infer_only_deltas)
//          _nn_decoders[i]->update_gradient();
//      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

        /*** INFER CONVOLUTIONAL DECODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _dnn_decoders[i]->infer_visibles(dnn_layer_t::ACCUMULATE);

      if (i + 1 < _dnn_decoders.size()) {                                           // If more convolutional decoders, ...
        _dnn_decoders[i]->infer_visible_Ra(_dnn_decoders[i + 1]->hiddens());        // Infer visible Ra
        swap(_dnn_decoders[i + 1]->hiddens(), _dnn_decoders[i]->visibles());        // Swap visible Ra, hidden activation -> visible activation, hidden Ra
        update_gv_part2(iLayer + 1, target);
      } else {
        _dnn_decoders[i]->infer_visible_Ra(_outputs);                               // Infer visible Ra from output activation

        // calculate RDa * f'(s)
        _dnn_decoders[i]->calculate_RDs(target, _outputs);
      }
    } else {
      assert(0); // should never happen
    }
  }

  // Forward propagate the activations from the hiddens to the visibles (hiddens already have the activations)
  // The backwards like regular backprop except that the last layer hiddens contain RDa instead of Da
  void update_gv_part3(int iLayer) {

    if (iLayer < _cnn_encoders.size()) {

        /*** INFER CONVOLUTIONAL ENCODING LAYER ***/

      const int i = iLayer;

      if (i + 1 < _cnn_encoders.size()) {                                           // If more convolutional encoders exists, ...
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();             // Transfer activation to the next layer
        update_gv_part3(iLayer + 1);
        _cnn_encoders[i + 1]->backprop_visibles();                                  // Back-propagate RDa

        // Handle shortcuts
        if (_model.has_dnn_shortcuts()) {
          boost::dynamic_pointer_cast<ddnn_layer_t>(_dnn_decoders[_dnn_decoders.size() - i - 1])->backprop_hidden_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles(), true);
        } else if ((int)_dnn_decoders.size() - i - 3 >= 0 &&_model.has_cnn_shortcuts()) {
          boost::dynamic_pointer_cast<cdnn_layer_t>(_dnn_decoders[_dnn_decoders.size() - i - 3])->backprop_hidden_deltas(); // this accumulates it with _cnn_encoders[i + 1] visibles
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles());
        } else {
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles());
        }
        _cnn_encoders[i]->update_gradient();
      } else {
        if (_nn_encoders.size()) {                                                  // If the model has dense encoding layers, ...
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1,     // Transition from convolutional model to dense model
              _cnn_encoders[i]->hiddens().count());
          update_gv_part3(iLayer + 1);

          // Transition back
          _nn_encoders[0]->backprop_visible_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(reshape(_nn_encoders[0]->visible_deltas(), _model.cnn_encoders()[i]->hiddens_size()));
          _cnn_encoders[i]->update_gradient();
        } else {
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();                // Transition to the next layer
          update_gv_part3(iLayer + 1);

          // Transition back
          _dnn_decoders[0]->backprop_hiddens();
          _cnn_encoders[i]->backprop_hidden_deltas(_dnn_decoders[0]->hiddens());
          _cnn_encoders[i]->update_gradient();
        }
      }
    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

        /*** INFER DENSE ENCODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size();

      if (i + 1 < _nn_encoders.size()) {                                            // If there are more encoding layers, ...
        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();               // Transition to next layer and repeat
        update_gv_part3(iLayer + 1);

        _nn_encoders[i + 1]->backprop_visible_deltas();
        _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
        _nn_encoders[i]->update_gradient();
      } else {
        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();                   // Transition to decoding layer and repeat
        update_gv_part3(iLayer + 1);

        _nn_decoders[0]->backprop_visible_deltas();
        _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
        _nn_encoders[i]->update_gradient();
      }
    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

        /*** INFER DENSE DECODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();

      if (i + 1 < _nn_decoders.size()) {                                            // If there are more decoding layers, ...
        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();               // Transition to next layer and repeat
        update_gv_part3(iLayer + 1);

        _nn_decoders[i + 1]->backprop_visible_deltas();
        _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
        _nn_decoders[i]->update_gradient();
      } else {
        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(),           // Transition to convolutional decoding layer and repeat
            _dnn_decoders[0]->hiddens().size());
        update_gv_part3(iLayer + 1);

        _dnn_decoders[0]->backprop_hiddens();
        _nn_decoders[i]->backprop_hidden_deltas(reshape(_dnn_decoders[0]->hiddens(), _nn_decoders[i]->hiddens().size()));
        _nn_decoders[i]->update_gradient();
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

        /*** INFER CONVOLUTIONAL DECODING LAYER ***/

      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      if (i + 1 < _dnn_decoders.size()) {                                           // If more convolutional decoders, ...
        _dnn_decoders[i + 1]->hiddens() = _dnn_decoders[i]->visibles();             // Transition to the next layer
        update_gv_part3(iLayer + 1);

        // Back-propagate errors
        _dnn_decoders[i + 1]->backprop_hiddens();
        _dnn_decoders[i]->backprop_visible_deltas(_dnn_decoders[i + 1]->hiddens());
        _dnn_decoders[i]->update_gradient();
      } else {
        // Start back-propagation
        _dnn_decoders[i]->update_gradient();
      }
    } else {
      assert(0); // should never happen
    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_ENCODER_HPP_ */
