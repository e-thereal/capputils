/*
 * cnn.hpp
 *
 *  Created on: 2014-08-19
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CNN_HPP_
#define TBBLAS_DEEPLEARN_CNN_HPP_

#include <tbblas/deeplearn/cnn_model.hpp>
#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>
#include <tbblas/deeplearn/cnn_base.hpp>

#include <tbblas/deeplearn/opt/type_traits.hpp>
#include <tbblas/deeplearn/opt/void_trainer.hpp>

#include <tbblas/context.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/reshape.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims, class Trainer = opt::void_trainer<T>, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class cnn : public virtual cnn_base<T, dims>, public Trainer {
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;
  typedef Trainer trainer_t;

  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef nn_layer<value_t, trainer_t> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

  typedef cnn_layer<value_t, dimCount, trainer_t> cnn_layer_t;
  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;

  typedef cnn_model<value_t, dimCount> model_t;

protected:
  model_t& _model;
  v_cnn_layer_t _cnn_layers;
  v_nn_layer_t _nn_layers;

public:
  cnn(model_t& model) : _model(model) {
    if (model.cnn_layers().size() == 0 || model.nn_layers().size() == 0)
      throw std::runtime_error("At least one convolutional and one dense layer required to build a convolutional neural network.");

    _cnn_layers.resize(model.cnn_layers().size());
    for (size_t i = 0; i < _cnn_layers.size(); ++i) {
      _cnn_layers[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.cnn_layers()[i]), this);
    }

    _nn_layers.resize(model.nn_layers().size());
    for (size_t i = 0; i < _nn_layers.size(); ++i) {
      _nn_layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_layers()[i]), this);
    }
  }

private:
  cnn(const cnn<T, dims>&);

public:
  virtual void normalize_visibles() {
    _cnn_layers[0]->normalize_visibles();
  }

  // Infer hidden units recursively
  virtual void infer_hiddens() {
    infer_hiddens(0);
  }

  // Does not require the hidden units to be inferred
  virtual void update_gradient(matrix_t& target) {
    update_gradient(0, target);
  }

  virtual void update_model(value_t weightcost) {
    for (size_t i = 0; i < _cnn_layers.size(); ++i)
      _cnn_layers[i]->update_model(weightcost);

    for (size_t i = 0; i < _nn_layers.size(); ++i)
      _nn_layers[i]->update_model(weightcost);
  }

  virtual void set_batch_length(int layer, int length) {
    if (layer < _cnn_layers.size())
      _cnn_layers[layer]->set_batch_length(length);
  }

  virtual const proxy<tensor_t> visibles() {
    return _cnn_layers[0]->visibles();
  }

  virtual matrix_t& hiddens() {
    return _nn_layers[_nn_layers.size() - 1]->hiddens();
  }

protected:
  void infer_hiddens(int iLayer) {
    if (iLayer < _cnn_layers.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_layers[i]->infer_hiddens();
      if (iLayer + 1 < _cnn_layers.size()) {
        _cnn_layers[i + 1]->visibles() = _cnn_layers[i]->hiddens();
        infer_hiddens(iLayer + 1);
      } else {
        // Transition from convolutional model to dense model
        _nn_layers[0]->visibles() = reshape(_cnn_layers[i]->hiddens(), 1, _cnn_layers[i]->hiddens().count());
        infer_hiddens(iLayer + 1);
      }
    } else if (iLayer - _cnn_layers.size() < _nn_layers.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_layers.size();

      _nn_layers[i]->infer_hiddens();
      if (i + 1 < _nn_layers.size()) {
        _nn_layers[i + 1]->visibles() = _nn_layers[i]->hiddens();
        infer_hiddens(iLayer + 1);
      }

    } else {
      // should never happen
      assert(0);
    }
  }

  void update_gradient(int iLayer, matrix_t& target) {
    if (iLayer < _cnn_layers.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_layers[i]->infer_hiddens();
      if (iLayer + 1 < _cnn_layers.size()) {
        _cnn_layers[i + 1]->visibles() = _cnn_layers[i]->hiddens();
        update_gradient(iLayer + 1, target);

        _cnn_layers[i + 1]->backprop_visibles();
        _cnn_layers[i]->backprop_hidden_deltas(_cnn_layers[i + 1]->visibles());
        _cnn_layers[i]->update_gradient();
      } else {
        // Transition from convolutional model to dense model
        _nn_layers[0]->visibles() = reshape(_cnn_layers[i]->hiddens(), 1, _cnn_layers[i]->hiddens().count());
        update_gradient(iLayer + 1, target);

        // Transition back
        _nn_layers[0]->backprop_visible_deltas();
        _cnn_layers[i]->backprop_hidden_deltas(reshape(_nn_layers[0]->visible_deltas(), _model.cnn_layers()[i]->hiddens_size()));
        _cnn_layers[i]->update_gradient();
      }
    } else if (iLayer - _cnn_layers.size() < _nn_layers.size()) {
      // Infer dense layer
      const int i = iLayer - _cnn_layers.size();

      _nn_layers[i]->infer_hiddens();
      if (i + 1 < _nn_layers.size()) {
        _nn_layers[i + 1]->visibles() = _nn_layers[i]->hiddens();
        update_gradient(iLayer + 1, target);

        _nn_layers[i + 1]->backprop_visible_deltas();
        _nn_layers[i]->backprop_hidden_deltas(_nn_layers[i + 1]->visible_deltas());
        _nn_layers[i]->update_gradient();
      } else {
        _nn_layers[i]->calculate_deltas(target);
        _nn_layers[i]->update_gradient();
      }
    } else {
      assert(0); // should never happen
    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_HPP_ */
