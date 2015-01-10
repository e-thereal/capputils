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
#include <tbblas/deeplearn/reverse_cnn_layer.hpp>

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

  typedef reverse_cnn_layer<value_t, dimCount> reverse_cnn_layer_t;
  typedef std::vector<boost::shared_ptr<reverse_cnn_layer_t> > v_reverse_cnn_layer_t;

  typedef encoder_model<value_t, dimCount> model_t;

protected:
  model_t& _model;
  v_cnn_layer_t _cnn_encoders;
  v_reverse_cnn_layer_t _cnn_decoders;
  v_nn_layer_t _nn_encoders, _nn_decoders;
  tbblas::deeplearn::objective_function _objective;
  value_t u, v;

public:
  encoder(model_t& model) : _model(model) {
    if (model.cnn_encoders().size() == 0 || model.cnn_decoders().size() == 0)
      throw std::runtime_error("At least one convolutional encoder and decoder is required to build a convolutional encoder neural network.");

    _cnn_encoders.resize(model.cnn_encoders().size());
    for (size_t i = 0; i < _cnn_encoders.size(); ++i) {
      _cnn_encoders[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.cnn_encoders()[i]));
    }

    _cnn_decoders.resize(model.cnn_decoders().size());
    for (size_t i = 0; i < _cnn_decoders.size(); ++i) {
      _cnn_decoders[i] = boost::make_shared<reverse_cnn_layer_t>(boost::ref(*model.cnn_decoders()[i]));
    }

    _nn_encoders.resize(model.nn_encoders().size());
    for (size_t i = 0; i < _nn_encoders.size(); ++i) {
      _nn_encoders[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_encoders()[i]));
    }

    _nn_decoders.resize(model.nn_decoders().size());
    for (size_t i = 0; i < _nn_decoders.size(); ++i) {
      _nn_decoders[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_decoders()[i]));
    }

    if (_cnn_encoders.size() != _cnn_decoders.size())
      throw std::runtime_error("The model needs to have the same number of convolutional encoder and decoder layers.");

    if (_nn_encoders.size() != _nn_decoders.size())
      throw std::runtime_error("The model needs to have the same number of dense encoder and decoder layers.");
  }

private:
  encoder(const encoder<T, dims>&);

public:
  void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective = objective;
    _cnn_decoders[_cnn_decoders.size() - 1]->set_objective_function(objective);
  }

  tbblas::deeplearn::objective_function objective_function() const {
    assert(_cnn_decoders[_cnn_decoders.size() - 1]->objective_function() == _objective);
    return _objective;
  }

  void set_sensitivity_ratio(const value_t& ratio) {
    _cnn_decoders[_cnn_decoders.size() - 1]->set_sensitivity_ratio(ratio);
  }

  value_t sensitivity_ratio() const {
    return _cnn_decoders[_cnn_decoders.size() - 1]->sensitivity_ratio();
  }

  void normalize_inputs() {
    _cnn_encoders[0]->normalize_visibles();
  }

  void diversify_outputs() {
    _cnn_decoders[_cnn_decoders.size() - 1]->diversify_visibles();
  }

  // Infer hidden units recursively
  void infer_outputs() {
    infer_outputs(0);
  }

  // Does not require the hidden units to be inferred
  void update_gradient(tensor_t& target) {
    update_gradient(0, target);
  }

  void momentum_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->momentum_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _cnn_decoders.size(); ++i)
      _cnn_decoders[i]->momentum_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->momentum_step(epsilon2, momentum, weightcost);

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->momentum_step(epsilon2, momentum, weightcost);
  }

  void adadelta_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _cnn_encoders.size(); ++i)
      _cnn_encoders[i]->adadelta_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _cnn_decoders.size(); ++i)
      _cnn_decoders[i]->adadelta_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _nn_encoders.size(); ++i)
      _nn_encoders[i]->adadelta_step(epsilon2, momentum, weightcost);

    for (size_t i = 0; i < _nn_decoders.size(); ++i)
      _nn_decoders[i]->adadelta_step(epsilon2, momentum, weightcost);
  }

  void set_batch_length(int layer, int length) {
    if (layer < _cnn_encoders.size())
      _cnn_encoders[layer]->set_batch_length(length);
    else if (layer - _cnn_encoders.size() < _cnn_decoders.size())
      _cnn_decoders[layer - _cnn_encoders.size()]->set_batch_length(length);
  }

  const proxy<tensor_t> inputs() {
    return _cnn_encoders[0]->visibles();
  }

  const proxy<tensor_t> outputs() {
    return _cnn_decoders[_cnn_decoders.size() - 1]->visibles();
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
          _cnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();
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
        _cnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(), _cnn_decoders[0]->hiddens().size());
        infer_outputs(iLayer + 1);
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _cnn_decoders.size()) {

      // Infer convolutional layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _cnn_decoders[i]->infer_visibles();

      // If more convolutional decoders, ...
      if (i + 1 < _cnn_decoders.size()) {

        // Transition to next layer and repeat
        _cnn_decoders[i + 1]->hiddens() = _cnn_decoders[i]->visibles();
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
        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _cnn_encoders[i + 1]->backprop_visible_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visible_deltas());
          _cnn_encoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visible_deltas());
          _cnn_encoders[i]->update_u_gradient(u, v);
          if (iLayer)
            _cnn_encoders[i]->backprop_visible_deltas();

          _cnn_encoders[i + 1]->backprop_visible_deltas();
          _cnn_encoders[i]->backprop_hidden_deltas(_cnn_encoders[i + 1]->visible_deltas());
          _cnn_encoders[i]->update_v_gradient(u, v);
          break;
        }
      } else {

        // If the model has dense encoding layers, ...
        if (_nn_encoders.size()) {

          // Transition from convolutional model to dense model
          _nn_encoders[0]->visibles() = reshape(_cnn_encoders[i]->hiddens(), 1, _cnn_encoders[i]->hiddens().count());
          update_gradient(iLayer + 1, target);

          // Transition back
          switch (_objective) {
          case tbblas::deeplearn::objective_function::SSD:
          case tbblas::deeplearn::objective_function::SenSpe:
            _nn_encoders[0]->backprop_visible_deltas();
            _cnn_encoders[i]->backprop_hidden_deltas(reshape(_nn_encoders[0]->visible_deltas(), _model.cnn_encoders()[i]->hiddens_size()));
            _cnn_encoders[i]->update_gradient();
            break;

          case tbblas::deeplearn::objective_function::DSC:
            _cnn_encoders[i]->backprop_hidden_deltas(reshape(_nn_encoders[0]->visible_deltas(), _model.cnn_encoders()[i]->hiddens_size()));
            _cnn_encoders[i]->update_u_gradient(u, v);
            if (iLayer)
              _cnn_encoders[i]->backprop_visible_deltas();

            _nn_encoders[0]->backprop_visible_deltas();
            _cnn_encoders[i]->backprop_hidden_deltas(reshape(_nn_encoders[0]->visible_deltas(), _model.cnn_encoders()[i]->hiddens_size()));
            _cnn_encoders[i]->update_v_gradient(u, v);
            break;
          }
        } else {

          // Transition to decoding layers
          _cnn_decoders[0]->hiddens() = _cnn_encoders[i]->hiddens();
          update_gradient(iLayer + 1, target);

          // Transition back
          switch (_objective) {
          case tbblas::deeplearn::objective_function::SSD:
          case tbblas::deeplearn::objective_function::SenSpe:
            _cnn_decoders[0]->backprop_hidden_deltas();
            _cnn_encoders[i]->backprop_hidden_deltas(_cnn_decoders[0]->hidden_deltas());
            _cnn_encoders[i]->update_gradient();
            break;

          case tbblas::deeplearn::objective_function::DSC:
            _cnn_encoders[i]->backprop_hidden_deltas(_cnn_decoders[0]->hidden_deltas());
            _cnn_encoders[i]->update_u_gradient(u, v);
            if (iLayer)
              _cnn_encoders[i]->backprop_visible_deltas();

            _cnn_decoders[0]->backprop_hidden_deltas();
            _cnn_encoders[i]->backprop_hidden_deltas(_cnn_decoders[0]->hidden_deltas());
            _cnn_encoders[i]->update_v_gradient(u, v);
            break;
          }
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

        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _nn_encoders[i + 1]->backprop_visible_deltas();
          _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
          _nn_encoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
          _nn_encoders[i]->update_u_gradient();
          _nn_encoders[i]->backprop_visible_deltas();

          _nn_encoders[i + 1]->backprop_visible_deltas();
          _nn_encoders[i]->backprop_hidden_deltas(_nn_encoders[i + 1]->visible_deltas());
          _nn_encoders[i]->update_v_gradient(u, v);
          break;
        }
      } else {

        // Transition to decoding layer and repeat
        _nn_decoders[0]->visibles() = _nn_encoders[i]->hiddens();
        update_gradient(iLayer + 1, target);

        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _nn_decoders[0]->backprop_visible_deltas();
          _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
          _nn_encoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
          _nn_encoders[i]->update_u_gradient();
          _nn_encoders[i]->backprop_visible_deltas();

          _nn_decoders[0]->backprop_visible_deltas();
          _nn_encoders[i]->backprop_hidden_deltas(_nn_decoders[0]->visible_deltas());
          _nn_encoders[i]->update_v_gradient(u, v);
          break;
        }
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

        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _nn_decoders[i + 1]->backprop_visible_deltas();
          _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
          _nn_decoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
          _nn_decoders[i]->update_u_gradient();
          _nn_decoders[i]->backprop_visible_deltas();

          _nn_decoders[i + 1]->backprop_visible_deltas();
          _nn_decoders[i]->backprop_hidden_deltas(_nn_decoders[i + 1]->visible_deltas());
          _nn_decoders[i]->update_v_gradient(u, v);
          break;
        }
      } else {

        // Transition to convolutional decoding layer and repeat
        _cnn_decoders[0]->hiddens() = reshape(_nn_decoders[i]->hiddens(), _cnn_decoders[0]->hiddens().size());
        update_gradient(iLayer + 1, target);

        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _cnn_decoders[0]->backprop_hidden_deltas();
          _nn_decoders[i]->backprop_hidden_deltas(reshape(_cnn_decoders[0]->hidden_deltas(), _nn_decoders[i]->hiddens().size()));
          _nn_decoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          _nn_decoders[i]->backprop_hidden_deltas(reshape(_cnn_decoders[0]->hidden_deltas(), _nn_decoders[i]->hiddens().size()));
          _nn_decoders[i]->update_u_gradient();
          _nn_decoders[0]->backprop_visible_deltas();

          _cnn_decoders[0]->backprop_hidden_deltas();
          _nn_decoders[i]->backprop_hidden_deltas(reshape(_cnn_decoders[0]->hidden_deltas(), _nn_decoders[i]->hiddens().size()));
          _nn_decoders[i]->update_v_gradient(u, v);
          break;
        }
      }

    } else if (iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size() < _cnn_decoders.size()) {

      // Infer convolutional layer
      const int i = iLayer - _cnn_encoders.size() - _nn_encoders.size() - _nn_decoders.size();

      _cnn_decoders[i]->infer_visibles();

      // If more convolutional decoders, ...
      if (i + 1 < _cnn_decoders.size()) {

        // Transition to next layer and repeat
        _cnn_decoders[i + 1]->hiddens() = _cnn_decoders[i]->visibles();
        update_gradient(iLayer + 1, target);

        // Back-propagate errors
        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _cnn_decoders[i + 1]->backprop_hidden_deltas();
          _cnn_decoders[i]->backprop_visible_deltas(_cnn_decoders[i + 1]->hidden_deltas());
          _cnn_decoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          _cnn_decoders[i]->backprop_visible_deltas(_cnn_decoders[i + 1]->hidden_deltas());
          _cnn_decoders[i]->update_u_gradient(u, v);
          _cnn_decoders[i]->backprop_hidden_deltas();

          _cnn_decoders[i + 1]->backprop_hidden_deltas();
          _cnn_decoders[i]->backprop_visible_deltas(_cnn_decoders[i + 1]->hidden_deltas());
          _cnn_decoders[i]->update_v_gradient(u, v);
          break;
        }
      } else {

        // Start back-propagation
        switch (_objective) {
        case tbblas::deeplearn::objective_function::SSD:
        case tbblas::deeplearn::objective_function::SenSpe:
          _cnn_decoders[i]->calculate_deltas(target);
          _cnn_decoders[i]->update_gradient();
          break;

        case tbblas::deeplearn::objective_function::DSC:
          u = 2 * sum(outputs() * target);
          v = sum(outputs()) + sum(target);

          // Update gradient and calculate hidden u deltas and visible v deltas
          _cnn_decoders[i]->calculate_u_deltas(target);
          _cnn_decoders[i]->update_u_gradient(u, v);
          _cnn_decoders[i]->backprop_hidden_deltas();

          _cnn_decoders[i]->calculate_v_deltas(target);
          _cnn_decoders[i]->update_v_gradient(u, v);
          break;

        default:
          throw std::runtime_error("Unsupported objective function in encoder::update_gradient(target)");
        }
      }
    } else {
      assert(0); // should never happen
    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_ENCODER_HPP_ */
