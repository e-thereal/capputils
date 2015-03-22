/*
 * encoder.hpp
 *
 *  Created on: 2015-01-06
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_ENCODER_HPP_
#define TBBLAS_DEEPLEARN_ENCODER_HPP_

#include <tbblas/deeplearn/encoder_model.hpp>
#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>
#include <tbblas/deeplearn/dnn_layer.hpp>
#include <tbblas/deeplearn/cdnn_layer.hpp>

#include <tbblas/context.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/reshape.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class encoder {
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef nn_layer<value_t> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

  typedef cnn_layer<value_t, dimCount> cnn_layer_t;
  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;

  typedef dnn_layer<value_t, dimCount> dnn_layer_t;
  typedef std::vector<boost::shared_ptr<dnn_layer_t> > v_dnn_layer_t;

  typedef cdnn_layer<value_t, dimCount> cdnn_layer_t;
  typedef std::vector<boost::shared_ptr<cdnn_layer_t> > v_cdnn_layer_t;

  typedef encoder_model<value_t, dimCount> model_t;

protected:
  model_t& _model;
  v_cnn_layer_t _cnn_encoders;
  v_dnn_layer_t _dnn_decoders;
  v_nn_layer_t _nn_encoders, _nn_decoders;
  tbblas::deeplearn::objective_function _objective;
  value_t u, v;
  tensor_t _inputs, _outputs;

public:
  encoder(model_t& model, const dim_t& patch_count = seq<dimCount>(1)) : _model(model) {
    if (model.cnn_encoders().size() == 0 || model.dnn_decoders().size() == 0)
      throw std::runtime_error("At least one convolutional encoder and decoder is required to build a convolutional encoder neural network.");

    if (model.cnn_encoders().size() != model.dnn_decoders().size())
      throw std::runtime_error("An encoder network needs to have the same number of encoders and decoders.");

    if (model.dnn_shortcuts().size() && model.dnn_decoders().size() != model.dnn_shortcuts().size() + 1) {
      throw std::runtime_error("An encoder network must have either no shortcut connections or one shortcut less than decoders.");
    }

    _cnn_encoders.resize(model.cnn_encoders().size());
    for (size_t i = 0; i < _cnn_encoders.size(); ++i) {
      _cnn_encoders[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.cnn_encoders()[i]), boost::ref(patch_count));
    }

    _dnn_decoders.resize(model.dnn_decoders().size());
    for (size_t i = 0; i < _dnn_decoders.size(); ++i) {
      if (i > 0 && model.dnn_shortcuts().size()) {
        std::cout << "Adding shortcut connection" << std::endl;
        _dnn_decoders[i] = boost::make_shared<cdnn_layer_t>(boost::ref(*model.dnn_shortcuts()[i - 1]), boost::ref(*model.dnn_decoders()[i]), boost::ref(*_cnn_encoders[_cnn_encoders.size() - i - 1]), boost::ref(patch_count));
      } else {
        std::cout << "Adding regular decoder" << std::endl;
        _dnn_decoders[i] = boost::make_shared<dnn_layer_t>(boost::ref(*model.dnn_decoders()[i]), boost::ref(patch_count));
        _dnn_decoders[i]->tie_switches(*_cnn_encoders[_cnn_encoders.size() - i - 1]);
      }
    }

    _nn_encoders.resize(model.nn_encoders().size());
    for (size_t i = 0; i < _nn_encoders.size(); ++i) {
      _nn_encoders[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_encoders()[i]));
    }

    _nn_decoders.resize(model.nn_decoders().size());
    for (size_t i = 0; i < _nn_decoders.size(); ++i) {
      _nn_decoders[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_decoders()[i]));
    }

    if (_nn_encoders.size() != _nn_decoders.size())
      throw std::runtime_error("The model needs to have the same number of dense encoder and decoder layers.");
  }

private:
  encoder(const encoder<T, dims>&);

public:
//  virtual ~encoder() {
//    _dnn_decoders.clear();
//    _cnn_encoders.clear();
//  }

public:
  void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective = objective;
    _dnn_decoders[_dnn_decoders.size() - 1]->set_objective_function(objective);
  }

  tbblas::deeplearn::objective_function objective_function() const {
    assert(_dnn_decoders[_dnn_decoders.size() - 1]->objective_function() == _objective);
    return _objective;
  }

  void set_sensitivity_ratio(const value_t& ratio) {
    _dnn_decoders[_dnn_decoders.size() - 1]->set_sensitivity_ratio(ratio);
  }

  value_t sensitivity_ratio() const {
    return _dnn_decoders[_dnn_decoders.size() - 1]->sensitivity_ratio();
  }

  void write_model_to_host() {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->write_model_to_host();

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->write_model_to_host();

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->write_model_to_host();

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->write_model_to_host();
  }

//  void normalize_inputs() {
//    _cnn_encoders[0]->normalize_visibles();
//  }
//
//  void diversify_outputs() {
//    _dnn_decoders[_dnn_decoders.size() - 1]->diversify_visibles();
//  }

  // Infer hidden units recursively
  void infer_outputs() {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    infer_outputs(0);
    _dnn_decoders[_dnn_decoders.size() - 1]->diversify_visibles();
    _outputs = _dnn_decoders[_dnn_decoders.size() - 1]->visibles();
  }

  // Does not require the hidden units to be inferred
  void update_gradient(tensor_t& target) {
    _cnn_encoders[0]->visibles() = _inputs;
    _cnn_encoders[0]->normalize_visibles();
    update_gradient(0, target);
  }

  void momentum_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->momentum_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->momentum_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->momentum_step(epsilon2, momentum, weightcost);

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->momentum_step(epsilon2, momentum, weightcost);
  }

  void adadelta_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->adadelta_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->adadelta_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->adadelta_step(epsilon2, momentum, weightcost);

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->adadelta_step(epsilon2, momentum, weightcost);
  }

  void adam_step(value_t alpha = 0.0002, value_t beta1 = 0.1, value_t beta2 = 0.001, value_t epsilon = 1e-8, value_t betaDecay = 1e-8, value_t weightcost = 0.0002) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->adam_step(alpha, beta1, beta2, epsilon, betaDecay, weightcost);

    for (size_t i = 0; i < _dnn_decoders.size(); ++i)
      _dnn_decoders[i]->adam_step(alpha, beta1, beta2, epsilon, betaDecay, weightcost);

    // TODO: implement Adam for dense layers
//    for (size_t i = 0; i < _nn_encoders.size(); ++i)
//      _nn_encoders[i]->adadelta_step(epsilon2, momentum, weightcost);
//
//    for (size_t i = 0; i < _nn_decoders.size(); ++i)
//      _nn_decoders[i]->adadelta_step(epsilon2, momentum, weightcost);
  }

  void set_batch_length(int layer, int length) {
    if (layer < _cnn_encoders.size())
      _cnn_encoders[layer]->set_batch_length(length);
    else if (layer - _cnn_encoders.size() < _dnn_decoders.size())
      _dnn_decoders[layer - _cnn_encoders.size()]->set_batch_length(length);
  }

  tensor_t& inputs() {
    return _inputs;
  }

  tensor_t& outputs() {
    return _outputs;
  }

protected:
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

  void update_gradient(int iLayer, tensor_t& target) {

    if (iLayer < _cnn_encoders.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_encoders[i]->infer_hiddens();

      // If more convolutional encoders exists, ...
      if (i + 1 < _cnn_encoders.size()) {

        // Transition to next layer and repeat
        _cnn_encoders[i + 1]->visibles() = _cnn_encoders[i]->hiddens();
        update_gradient(iLayer + 1, target);

        // Back-propagate errors
        _cnn_encoders[i + 1]->backprop_visibles();

        if (_model.has_shortcuts()) {
          boost::dynamic_pointer_cast<cdnn_layer_t>(_dnn_decoders[_dnn_decoders.size() - i - 1])->backprop_hidden_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles(), true);
        } else {
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visibles());
        }

        _cnn_encoders[i]->update_gradient();
      } else {

        // If the model has dense encoding layers, ...
        if (_nn_encoders.size()) {

          // Transition from convolutional model to dense model
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1, _cnn_encoders[i]->hiddens().count());
          update_gradient(iLayer + 1, target);

          // Transition back
          _nn_encoders[0]->backprop_visible_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(reshape(_nn_encoders[0]->visible_deltas(), _model.cnn_encoders()[i]->hiddens_size()));
          _cnn_encoders[i]->update_gradient();
        } else {

          // Transition to decoding layers
          _dnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();
          update_gradient(iLayer + 1, target);

          // Transition back
          _dnn_decoders[0]->backprop_hiddens();
          _cnn_encoders[i]->backprop_hidden_deltas(_dnn_decoders[0]->hiddens());
          _cnn_encoders[i]->update_gradient();
        }
      }
    } else if (iLayer - _cnn_encoders.size() < _nn_encoders.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_encoders.size();
      _nn_encoders[i]->infer_hiddens();

      // If there are more encoding layers, ...
      if (i + 1 < _nn_encoders.size()) {

        // Transition to next layer and repeat
        _nn_encoders[i + 1]->visibles() = _nn_encoders[i]->hiddens();
        update_gradient(iLayer + 1, target);

        _nn_encoders[i + 1]->backprop_visible_deltas();
        _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
        _nn_encoders[i]->update_gradient();
      } else {

        // Transition to decoding layer and repeat
        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();
        update_gradient(iLayer + 1, target);

        _nn_decoders[0]->backprop_visible_deltas();
        _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
        _nn_encoders[i]->update_gradient();
      }
    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() < _nn_decoders.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size();
      _nn_decoders[i]->infer_hiddens();

      // If there are more decoding layers, ...
      if (i + 1 < _nn_decoders.size()) {

        // Transition to next layer and repeat
        _nn_decoders[i + 1]->visibles() = _nn_decoders[i]->hiddens();
        update_gradient(iLayer + 1, target);

        _nn_decoders[i + 1]->backprop_visible_deltas();
        _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
        _nn_decoders[i]->update_gradient();
      } else {

        // Transition to convolutional decoding layer and repeat
        _dnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(), _dnn_decoders[0]->hiddens().size());
        update_gradient(iLayer + 1, target);

        _dnn_decoders[0]->backprop_hiddens();
        _nn_decoders[i]->backprop_hidden_deltas(reshape(_dnn_decoders[0]->hiddens(), _nn_decoders[i]->hiddens().size()));
        _nn_decoders[i]->update_gradient();
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _dnn_decoders.size()) {

      // Infer convolutional layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _dnn_decoders[i]->infer_visibles();

      // If more convolutional decoders, ...
      if (i + 1 < _dnn_decoders.size()) {

        // Transition to next layer and repeat
        _dnn_decoders[i + 1]->hiddens() = _dnn_decoders[i]->visibles();
        update_gradient(iLayer + 1, target);

        // Back-propagate errors
        _dnn_decoders[i + 1]->backprop_hiddens();
        _dnn_decoders[i]->backprop_visible_deltas(_dnn_decoders[i + 1]->hiddens());
        _dnn_decoders[i]->update_gradient();
      } else {
        _dnn_decoders[_dnn_decoders.size() - 1]->diversify_visibles();
        _outputs = _dnn_decoders[_dnn_decoders.size() - 1]->visibles();

        // Start back-propagation
        _dnn_decoders[i]->calculate_deltas(target);
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
