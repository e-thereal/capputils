/*
 * joint_cnn.hpp
 *
 *  Created on: 2014-11-23
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_JOINT_CNN_HPP_
#define TBBLAS_DEEPLEARN_JOINT_CNN_HPP_

#include <tbblas/deeplearn/joint_cnn_model.hpp>
#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>

#include <tbblas/context.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/reshape.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class joint_cnn {
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

  typedef joint_cnn_model<value_t, dimCount> model_t;

protected:
  model_t& _model;
  v_cnn_layer_t _left_cnn_layers, _right_cnn_layers;
  v_nn_layer_t _left_nn_layers, _right_nn_layers, _joint_nn_layers;

public:
  joint_cnn(model_t& model) : _model(model) {
    if (model.left_cnn_layers().size() == 0 || model.left_nn_layers().size() == 0 ||
        model.right_cnn_layers().size() == 0 || model.right_nn_layers().size() == 0 ||
        model.joint_nn_layers().size() == 0)
    {
      throw std::runtime_error("At least one convolutional and one dense layer required to build a convolutional neural network.");
    }

    _left_cnn_layers.resize(model.left_cnn_layers().size());
    for (size_t i = 0; i < _left_cnn_layers.size(); ++i) {
      _left_cnn_layers[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.left_cnn_layers()[i]));
    }

    _right_cnn_layers.resize(model.right_cnn_layers().size());
    for (size_t i = 0; i < _right_cnn_layers.size(); ++i) {
      _right_cnn_layers[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.right_cnn_layers()[i]));
    }

    _left_nn_layers.resize(model.left_nn_layers().size());
    for (size_t i = 0; i < _left_nn_layers.size(); ++i) {
      _left_nn_layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.left_nn_layers()[i]));
    }

    _right_nn_layers.resize(model.right_nn_layers().size());
    for (size_t i = 0; i < _right_nn_layers.size(); ++i) {
      _right_nn_layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.right_nn_layers()[i]));
    }

    _joint_nn_layers.resize(model.joint_nn_layers().size());
    for (size_t i = 0; i < _joint_nn_layers.size(); ++i) {
      _joint_nn_layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.joint_nn_layers()[i]));
    }
  }

private:
  joint_cnn(const joint_cnn<T, dims>&);

public:
  void normalize_visibles() {
    _left_cnn_layers[0]->normalize_visibles();
    _right_cnn_layers[0]->normalize_visibles();
  }

  void infer_hiddens() {

    /* Infer hiddens of left pathway */

    for (size_t i = 0; i < _left_cnn_layers.size(); ++i) {
      _left_cnn_layers[i]->infer_hiddens();
      if (i + 1 < _left_cnn_layers.size())
        _left_cnn_layers[i + 1]->visibles() = rearrange(_left_cnn_layers[i]->hiddens(), _model.left_cnn_layers()[i + 1]->stride_size());
    }

    // Transition from convolutional model to dense model
    _left_nn_layers[0]->visibles().resize(seq(1, (int)_left_cnn_layers[_left_cnn_layers.size() - 1]->hiddens().count()));

    thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream), _left_cnn_layers[_left_cnn_layers.size() - 1]->hiddens().begin(),
        _left_cnn_layers[_left_cnn_layers.size() - 1]->hiddens().end(), _left_nn_layers[0]->visibles().begin());

    for (size_t i = 0; i < _left_nn_layers.size(); ++i) {
      _left_nn_layers[i]->infer_hiddens();
      if (i + 1 < _left_nn_layers.size()) {
        _left_nn_layers[i + 1]->visibles() = _left_nn_layers[i]->hiddens();
      }
    }

    /* Infer hiddens of right pathway */

    for (size_t i = 0; i < _right_cnn_layers.size(); ++i) {
      _right_cnn_layers[i]->infer_hiddens();
      if (i + 1 < _right_cnn_layers.size())
        _right_cnn_layers[i + 1]->visibles() = rearrange(_right_cnn_layers[i]->hiddens(), _model.right_cnn_layers()[i + 1]->stride_size());
    }

    // Transition from convolutional model to dense model
    _right_nn_layers[0]->visibles().resize(seq(1, (int)_right_cnn_layers[_right_cnn_layers.size() - 1]->hiddens().count()));

    thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream), _right_cnn_layers[_right_cnn_layers.size() - 1]->hiddens().begin(),
        _right_cnn_layers[_right_cnn_layers.size() - 1]->hiddens().end(), _right_nn_layers[0]->visibles().begin());

    for (size_t i = 0; i < _right_nn_layers.size(); ++i) {
      _right_nn_layers[i]->infer_hiddens();
      if (i + 1 < _right_nn_layers.size()) {
        _right_nn_layers[i + 1]->visibles() = _right_nn_layers[i]->hiddens();
      }
    }

    /* Infer joint hiddens */

    _joint_nn_layers[0]->visibles().resize(seq(1,
        (int)_left_nn_layers[_left_nn_layers.size() - 1]->hiddens().count() +
        (int)_right_nn_layers[_right_nn_layers.size() - 1]->hiddens().count()));

    _joint_nn_layers[0]->visibles()[seq(0,0), _left_nn_layers[_left_nn_layers.size() - 1]->hiddens().size()] = _left_nn_layers[_left_nn_layers.size() - 1]->hiddens();
    _joint_nn_layers[0]->visibles()[seq(0, (int)_left_nn_layers[_left_nn_layers.size() - 1]->hiddens().count()), _right_nn_layers[_right_nn_layers.size() - 1]->hiddens().size()] = _right_nn_layers[_right_nn_layers.size() - 1]->hiddens();


    for (size_t i = 0; i < _joint_nn_layers.size(); ++i) {
      _joint_nn_layers[i]->infer_hiddens();
      if (i + 1 < _joint_nn_layers.size()) {
        _joint_nn_layers[i + 1]->visibles() = _joint_nn_layers[i]->hiddens();
      }
    }
  }

//  void init_gradient_updates(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
//    for (size_t i = 0; i < _left_cnn_layers.size(); ++i)
//      _left_cnn_layers[i]->init_gradient_updates(epsilon1, momentum, weightcost);
//    for (size_t i = 0; i < _right_cnn_layers.size(); ++i)
//      _right_cnn_layers[i]->init_gradient_updates(epsilon1, momentum, weightcost);
//
//    for (size_t i = 0; i < _left_nn_layers.size(); ++i)
//      _left_nn_layers[i]->init_gradient_updates(epsilon2, momentum, weightcost);
//    for (size_t i = 0; i < _right_nn_layers.size(); ++i)
//      _right_nn_layers[i]->init_gradient_updates(epsilon2, momentum, weightcost);
//    for (size_t i = 0; i < _joint_nn_layers.size(); ++i)
//      _joint_nn_layers[i]->init_gradient_updates(epsilon2, momentum, weightcost);
//  }

  void update_gradient(matrix_t& target) {

    infer_hiddens();

    /* Back prop joint layers */

    _joint_nn_layers[_joint_nn_layers.size() - 1]->calculate_deltas(target);
    _joint_nn_layers[_joint_nn_layers.size() - 1]->update_gradient();

    // Perform back propagation
    for (int i = _joint_nn_layers.size() - 2; i >= 0; --i) {
      _joint_nn_layers[i + 1]->backprop_visible_deltas();
      _joint_nn_layers[i]->backprop_hidden_deltas(_joint_nn_layers[i + 1]->visible_deltas());
      _joint_nn_layers[i]->update_gradient();
    }

    _joint_nn_layers[0]->backprop_visible_deltas();

    /* Back prop left pathway */

    _left_nn_layers[_left_nn_layers.size() - 1]->backprop_hidden_deltas(
        _joint_nn_layers[0]->visible_deltas()[seq(0,0), _left_nn_layers[_left_nn_layers.size() - 1]->hiddens().size()]);
    _left_nn_layers[_left_nn_layers.size() - 1]->update_gradient();

    // Perform back propagation
    for (int i = _left_nn_layers.size() - 2; i >= 0; --i) {
      _left_nn_layers[i + 1]->backprop_visible_deltas();
      _left_nn_layers[i]->backprop_hidden_deltas(_left_nn_layers[i + 1]->visible_deltas());
      _left_nn_layers[i]->update_gradient();
    }

    {
      const size_t clast = _left_cnn_layers.size() - 1;
      _left_nn_layers[0]->backprop_visible_deltas();
      _left_cnn_layers[clast]->backprop_hidden_deltas(reshape(
          _left_nn_layers[0]->visible_deltas(),
          _model.left_cnn_layers()[clast]->hiddens_size()));
      _left_cnn_layers[clast]->update_gradient();

      for (int i = _left_cnn_layers.size() - 2; i >= 0; --i) {
        _left_cnn_layers[i + 1]->backprop_visible_deltas();
        _left_cnn_layers[i]->backprop_hidden_deltas(rearrange_r(
            _left_cnn_layers[i + 1]->visible_deltas(),
            _model.left_cnn_layers()[i + 1]->stride_size()));
        _left_cnn_layers[i]->update_gradient();
      }
    }

    /* Back prop right pathway */

    _right_nn_layers[_right_nn_layers.size() - 1]->backprop_hidden_deltas(
        _joint_nn_layers[0]->visible_deltas()[seq(0, (int)_left_nn_layers[_left_nn_layers.size() - 1]->hiddens().count()), _right_nn_layers[_right_nn_layers.size() - 1]->hiddens().size()]);
    _right_nn_layers[_right_nn_layers.size() - 1]->update_gradient();

    // Perform back propagation
    for (int i = _right_nn_layers.size() - 2; i >= 0; --i) {
      _right_nn_layers[i + 1]->backprop_visible_deltas();
      _right_nn_layers[i]->backprop_hidden_deltas(_right_nn_layers[i + 1]->visible_deltas());
      _right_nn_layers[i]->update_gradient();
    }

    {
      const size_t clast = _right_cnn_layers.size() - 1;
      _right_nn_layers[0]->backprop_visible_deltas();
      _right_cnn_layers[clast]->backprop_hidden_deltas(reshape(
          _right_nn_layers[0]->visible_deltas(),
          _model.right_cnn_layers()[clast]->hiddens_size()));
      _right_cnn_layers[clast]->update_gradient();

      for (int i = _right_cnn_layers.size() - 2; i >= 0; --i) {
        _right_cnn_layers[i + 1]->backprop_visible_deltas();
        _right_cnn_layers[i]->backprop_hidden_deltas(rearrange_r(
            _right_cnn_layers[i + 1]->visible_deltas(),
            _model.right_cnn_layers()[i + 1]->stride_size()));
        _right_cnn_layers[i]->update_gradient();
      }
    }
  }

  void momentum_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _left_cnn_layers.size(); ++i)
      _left_cnn_layers[i]->momentum_step(epsilon1, momentum, weightcost);
    for (size_t i = 0; i < _right_cnn_layers.size(); ++i)
      _right_cnn_layers[i]->momentum_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _left_nn_layers.size(); ++i)
      _left_nn_layers[i]->momentum_step(epsilon2, momentum, weightcost);
    for (size_t i = 0; i < _right_nn_layers.size(); ++i)
      _right_nn_layers[i]->momentum_step(epsilon2, momentum, weightcost);
    for (size_t i = 0; i < _joint_nn_layers.size(); ++i)
      _joint_nn_layers[i]->momentum_step(epsilon2, momentum, weightcost);
  }

  void adadelta_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _left_cnn_layers.size(); ++i)
      _left_cnn_layers[i]->adadelta_step(epsilon1, momentum, weightcost);
    for (size_t i = 0; i < _right_cnn_layers.size(); ++i)
      _right_cnn_layers[i]->adadelta_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _left_nn_layers.size(); ++i)
      _left_nn_layers[i]->adadelta_step(epsilon2, momentum, weightcost);
    for (size_t i = 0; i < _right_nn_layers.size(); ++i)
      _right_nn_layers[i]->adadelta_step(epsilon2, momentum, weightcost);
    for (size_t i = 0; i < _joint_nn_layers.size(); ++i)
      _joint_nn_layers[i]->adadelta_step(epsilon2, momentum, weightcost);
  }

  void set_left_batch_length(int layer, int length) {
    if (layer < _left_cnn_layers.size())
      _left_cnn_layers[layer]->set_batch_length(length);
  }

  void set_right_batch_length(int layer, int length) {
    if (layer < _right_cnn_layers.size())
      _right_cnn_layers[layer]->set_batch_length(length);
  }

  void set_left_input(tensor_t& input) {
    assert(_model.left_cnn_layers()[0]->input_size() == input.size());
    _left_cnn_layers[0]->visibles() = rearrange(input, _model.left_cnn_layers()[0]->stride_size());
  }

  void set_right_input(tensor_t& input) {
    assert(_model.right_cnn_layers()[0]->input_size() == input.size());
    _right_cnn_layers[0]->visibles() = rearrange(input, _model.right_cnn_layers()[0]->stride_size());
  }

  matrix_t& hiddens() {
    return _joint_nn_layers[_joint_nn_layers.size() - 1]->hiddens();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_JOINT_CNN_HPP_ */
