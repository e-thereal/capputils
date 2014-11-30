/*
 * nn_layer.hpp
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_NN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_NN_LAYER_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/sequence.hpp>
#include <tbblas/util.hpp>
#include <tbblas/linalg.hpp>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/nn_layer_model.hpp>

#include <iostream>
#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T>
class nn_layer {
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef typename matrix_t::dim_t dim_t;

  typedef nn_layer_model<value_t> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  matrix_t W, b, dW, db, mean, stddev;
  matrix_t dW2, db2, deltaW2, deltab2, deltaW, deltab;

  // visible and hidden units in GPU memory
  matrix_t V, H, dV, dH;

  matrix_t prods, hidact, hidnorm;

  bool _memory_allocated, _host_updated;
  value_t _current_batch_size;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  nn_layer(model_t& model) : model(model),
    _memory_allocated(false), _host_updated(true), _current_batch_size(0)
  { }

private:
  nn_layer(const nn_layer&);

public:
  virtual ~nn_layer() {
    if (!_host_updated)
      write_model_to_host();
  }

  /// Transforms
  void allocate_gpu_memory() {
    using namespace tbblas;

    if (_memory_allocated)
      return;

    _memory_allocated = true;

    W = model.weights();
    b = model.bias();

    mean = model.mean();
    stddev = model.stddev();
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    model.set_weights(W);
    model.set_bias(b);
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    V = (V - repeat(mean, V.size() / mean.size())) / repeat(stddev, V.size() / stddev.size());
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    H = prod(V, W);
    H = H + repeat(b, H.size() / b.size());

    switch (model.activation_function()) {
      case activation_function::Sigmoid: H = sigm(H);    break;
      case activation_function::ReLU:    H = max(0, H);  break;
      case activation_function::Softmax:
        H = exp(H);
        hidnorm = sum(H, 1);
        H = H / (tolerance + repeat(hidnorm, H.size() / hidnorm.size()));
        break;
      case activation_function::Linear:  break;
    }
  }

  /// Requires hidden activation and hidden total activation
  void calculate_deltas(matrix_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    // delta = (hidden - target) * f'(X)
    switch (model.activation_function()) {
    case activation_function::Sigmoid:
      dH = (H - repeat(target, H.size() / target.size())) * H * (1 + -H);
      break;

    case activation_function::ReLU:
      dH = (H - repeat(target, H.size() / target.size())) * (H > 0);
      break;

    case activation_function::Softmax:
    case activation_function::Linear:
      dH = H - repeat(target, H.size() / target.size());
      break;
    }
  }

  void backprop_visible_deltas() {
    if (!dH.count())
      throw std::runtime_error("You need to calculate the deltas before bproping them.");

    // will be called by the previous layer
    dV = prod(dH, trans(W));
  }

  void backprop_visibles() {
    V = prod(H, trans(W));
  }

  /// Takes visible deltas of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == 2, int>::type >::type
  backprop_hidden_deltas(const Expression& deltas) {
    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      dH = deltas * H * (1 + -H);
      break;

    case activation_function::ReLU:
      dH = deltas * (H > 0);
      break;

    case activation_function::Linear:
      dH = deltas;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
    return 0;
  }

  void update_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (!dH.count())
      throw std::runtime_error("Hidden deltas not calculated.");

    if (!dW.count())
      dW = zeros<value_t>(W.size());
    if (!db.count())
      db = zeros<value_t>(b.size());

    // (x_n)(mu_n)'
    prods = tbblas::prod(trans(V), dH);

    // Calculate the total activation of the hidden and visible units
    hidact = sum(dH, 0);

    dW += prods / V.size()[0];
    db += hidact / V.size()[0];

    ++_current_batch_size;
  }

  void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    // Lazy initialisation
    if (!deltaW.count())
      deltaW = zeros<value_t>(W.size());
    if (!deltab.count())
      deltab = zeros<value_t>(b.size());

    if (_current_batch_size) {
      deltaW = momentum * deltaW + dW / _current_batch_size + weightcost * W;
      deltab = momentum * deltab + db / _current_batch_size;

      W = W - epsilon * deltaW;
      b = b - epsilon * deltab;

      dW = zeros<value_t>(dW.size());
      db = zeros<value_t>(db.size());
    }

    _current_batch_size = 0;

    _host_updated = false;
  }

  void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {
    if (!dW2.count())
      dW2 = zeros<value_t>(W.size());
    if (!db2.count())
      db2 = zeros<value_t>(b.size());
    if (!deltaW2.count())
      deltaW2 = zeros<value_t>(W.size());
    if (!deltab2.count())
      deltab2 = zeros<value_t>(b.size());

    if (_current_batch_size) {
      dW = dW / _current_batch_size + weightcost * W;
      db = db / _current_batch_size;

      dW2 = momentum * dW2 + (1.0 - momentum) * dW * dW;
      db2 = momentum * db2 + (1.0 - momentum) * db * db;

      // note that deltaW = - sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      W = W - sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      b = b - sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db;

      deltaW2 = momentum * deltaW2 + (1.0 - momentum) * sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW * sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      deltab2 = momentum * deltab2 + (1.0 - momentum) * sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db * sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db;

      dW = zeros<value_t>(dW.size());
      db = zeros<value_t>(db.size());
    }

    _current_batch_size = 0;

    _host_updated = false;
  }

  /// Requires hidden deltas and visibles
  void momentum_update(value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    update_gradient();
    momentum_step(epsilon, momentum, weightcost);
  }

  void adadelta_update(value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    update_gradient();
    adadelta_step(epsilon, momentum, weightcost);
  }

  // Access to model data
  matrix_t& visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return V;
  }

  matrix_t& hiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return H;
  }

  // Don't change the return value.
  matrix_t& visible_deltas() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return dV;
  }
};

template<class T>
const T nn_layer<T>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_RBM_HPP_ */
