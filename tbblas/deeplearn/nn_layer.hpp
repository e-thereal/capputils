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

#include <tbblas/deeplearn/opt/type_traits.hpp>
#include <tbblas/deeplearn/opt/void_trainer.hpp>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/nn_layer_model.hpp>
#include <tbblas/deeplearn/objective_function.hpp>

#include <iostream>
#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, class Trainer = opt::void_trainer<T>, class Enable = typename boost::enable_if<opt::is_trainer<Trainer > >::type>
class nn_layer : public Trainer {
  typedef T value_t;

  typedef tbblas::tensor<value_t, 1, true> vector_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef typename matrix_t::dim_t dim_t;

  typedef tbblas::random_tensor2<value_t, 2, true, tbblas::uniform<value_t> > uniform_t;

  typedef nn_layer_model<value_t> model_t;

  static const value_t tolerance = 1e-8;

public:
  matrix_t _Rs;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  matrix_t W, b, dW, db, mean, stddev;
//  matrix_t dW2, db2, deltaW2, deltab2, deltaW, deltab, duW, dub;

  // visible and hidden units in GPU memory
  matrix_t V, H, dV, dH;
  uniform_t h_rand;

  // For the Gv update
  matrix_t RW, Rb, RdW, Rdb, RV, RH, RdV, RdH;

  matrix_t prods, hidact, hidnorm;

  bool _memory_allocated, _host_updated;
  value_t _current_batch_size;

  tbblas::deeplearn::objective_function _objective_function;
  value_t _sensitivity_ratio, _dropout_rate;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  nn_layer(model_t& model, const Trainer* parameters = NULL) : Trainer(parameters), model(model),
    _memory_allocated(false), _host_updated(true), _current_batch_size(0), _sensitivity_ratio(0.5), _dropout_rate(0)
  { }

private:
  nn_layer(const nn_layer&);

public:
  virtual ~nn_layer() {
    if (!_host_updated)
      write_model_to_host();
  }

  void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective_function = objective;
  }

  tbblas::deeplearn::objective_function objective_function() const {
    return _objective_function;
  }

  void set_sensitivity_ratio(const value_t& ratio) {
    _sensitivity_ratio = ratio;
  }

  value_t sensitivity_ratio() const {
    return _sensitivity_ratio;
  }

  void set_dropout_rate(const value_t& rate) {
    if (rate < 0 || rate >= 1)
      throw std::runtime_error("Drop out rate must be in [0,1).");

    _dropout_rate = rate;
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

    RW = zeros<value_t>(W.size());
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

  int set_parameters(vector_t& parameters, int offset, bool transpose = false) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (transpose)
      trans(W) = reshape(parameters[seq(offset), seq((int)W.count())], trans(W).size());
    else
      W = reshape(parameters[seq(offset), seq((int)W.count())], W.size());
    offset += W.count();

    b = reshape(parameters[seq(offset), seq((int)b.count())], b.size());
    offset += b.count();
    return offset;
  }

  int set_vector(vector_t& parameters, int offset, bool transpose = false) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (transpose)
      trans(RW) = reshape(parameters[seq(offset), seq((int)W.count())], trans(W).size());
    else
      RW = reshape(parameters[seq(offset), seq((int)W.count())], W.size());
    offset += W.count();

    Rb = reshape(parameters[seq(offset), seq((int)b.count())], b.size());
    offset += b.count();
    return offset;
  }

  int get_parameters(vector_t& parameters, int offset, bool transpose = false) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (transpose)
      parameters[seq(offset), seq((int)W.count())] = reshape(trans(W), seq((int)W.count()));
    else
      parameters[seq(offset), seq((int)W.count())] = reshape(W, seq((int)W.count()));
    offset += W.count();

    parameters[seq(offset), seq((int)b.count())] = reshape(b, seq((int)b.count()));
    offset += b.count();
    return offset;
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    V = (V - repeat(mean, V.size() / mean.size())) / repeat(stddev, V.size() / stddev.size());
  }

  void infer_hiddens(bool dropout = false) {
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

    if (dropout && _dropout_rate > 0) {
      if (h_rand.size() != H.size())
        h_rand.resize(H.size());
      H = H * (h_rand() > _dropout_rate) / (1. - _dropout_rate);
    }
  }

  void infer_Rhiddens(bool dropout = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    // Regular forward pass
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

    // R-op forward pass
    RH = prod(V, RW);
    RH += prod(RV, W);
    RH = RH + repeat(Rb, RH.size() / Rb.size());

    switch (model.activation_function()) {
      case activation_function::Sigmoid: RH = RH * H * (1.0 - H);    break;
      case activation_function::ReLU:    RH = RH * (H > 0);  break;
      case activation_function::Linear:  break;
    }
  }

  /// Requires hidden activation and hidden total activation
  void calculate_deltas(matrix_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    switch (_objective_function) {
    case tbblas::deeplearn::objective_function::SSD:

      // delta = (hidden - target) * f'(X)
      switch (model.activation_function()) {
      case activation_function::Sigmoid:
        dH = (H - target) * H * (1 + -H);
        break;

      case activation_function::ReLU:
        dH = (H - target) * (H > 0);
        break;

      case activation_function::Softmax:
      case activation_function::Linear:
        dH = H - target;
        break;
      }
      break;

    case tbblas::deeplearn::objective_function::SenSpe:
      {
        if (model.activation_function() != activation_function::Sigmoid)
          throw std::runtime_error("Activation function for objective function 'Sensitivity + Specificity' must be 'Sigmoid'.");

        // delta = (-alpha* target - beta * target + beta) * f'(X)

        const value_t positive_ratio = sum(target) / (value_t)target.count();
        const value_t alpha = _sensitivity_ratio / (positive_ratio + value_t(1e-8));
        const value_t beta = (value_t(1) - _sensitivity_ratio) / (value_t(1) - positive_ratio + value_t(1e-8));

        dH = (alpha * target + beta * (1 + -target)) * (H - target) * H * (1 + -H);
      }
      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_deltas(target).");
    }
  }

  /// Requires hidden activation and hidden total activation
  void calculate_Rdeltas(matrix_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    switch (_objective_function) {
    case tbblas::deeplearn::objective_function::SSD:

      // delta = (hidden - target) * f'(X)
      switch (model.activation_function()) {
      case activation_function::Sigmoid:
        RdH = RH * H * (1.0 - H);
        break;

      case activation_function::ReLU:
        RdH = RH * (H > 0);
        break;

      case activation_function::Linear:
        RdH = RH;
        break;
      }
      break;

//    case tbblas::deeplearn::objective_function::SenSpe:
//      {
//        if (model.activation_function() != activation_function::Sigmoid)
//          throw std::runtime_error("Activation function for objective function 'Sensitivity + Specificity' must be 'Sigmoid'.");
//
//        // delta = (-alpha* target - beta * target + beta) * f'(X)
//
//        const value_t positive_ratio = sum(target) / (value_t)target.count();
//        const value_t alpha = _sensitivity_ratio / (positive_ratio + value_t(1e-8));
//        const value_t beta = (value_t(1) - _sensitivity_ratio) / (value_t(1) - positive_ratio + value_t(1e-8));
//
//        dH = (alpha * target + beta * (1 + -target)) * (H - target) * H * (1 + -H);
//      }
//      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_deltas(target).");
    }
    _Rs = RdH;
  }

  void backprop_visible_deltas() {
    if (!dH.count())
      throw std::runtime_error("You need to calculate the deltas before bproping them.");

    // will be called by the previous layer
    dV = prod(dH, trans(W));
  }

  void backprop_visible_Rdeltas() {
    if (!RdH.count())
      throw std::runtime_error("You need to calculate the deltas before bproping them.");

    // will be called by the previous layer
    RdV = prod(RdH, trans(W));
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
      dH = deltas * H * (value_t(1) - H);
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

  /// Takes visible Rdeltas of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == 2, int>::type >::type
  backprop_hidden_Rdeltas(const Expression& Rdeltas) {
    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      RdH = Rdeltas * H * (value_t(1) - H);
      break;

    case activation_function::ReLU:
      RdH = Rdeltas * (H > 0);
      break;

    case activation_function::Linear:
      RdH = Rdeltas;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
    _Rs = RdH;
    return 0;
  }

  void reset_gradient() {
    dW = zeros<value_t>(W.size());
    db = zeros<value_t>(b.size());
    _current_batch_size = 0;
  }

  void reset_Rgradient() {
    RdW = zeros<value_t>(W.size());
    Rdb = zeros<value_t>(b.size());
    _current_batch_size = 0;
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

  void update_Rgradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (!RdH.count())
      throw std::runtime_error("Hidden deltas not calculated.");

    if (!RdW.count())
      RdW = zeros<value_t>(W.size());
    if (!Rdb.count())
      Rdb = zeros<value_t>(b.size());

    // (x_n)(mu_n)'
    prods = tbblas::prod(trans(V), RdH);

    // Calculate the total activation of the hidden and visible units
    hidact = sum(RdH, 0);

    RdW += prods / V.size()[0];
    Rdb += hidact / V.size()[0];

    ++_current_batch_size;
  }

//  void update_u_gradient() {
//    if (!_memory_allocated)
//      allocate_gpu_memory();
//
//    if (!dH.count())
//      throw std::runtime_error("Hidden deltas not calculated.");
//
//    // (x_n)(mu_n)'
//    prods = tbblas::prod(trans(V), dH);
//
//    // Calculate the total activation of the hidden and visible units
//    hidact = sum(dH, 0);
//
//    duW = prods;
//    dub = hidact;
//  }
//
//  void update_v_gradient(value_t u, value_t v) {
//    if (!_memory_allocated)
//      allocate_gpu_memory();
//
//    if (!dH.count())
//      throw std::runtime_error("Hidden deltas not calculated.");
//
//    if (!dW.count())
//      dW = zeros<value_t>(W.size());
//    if (!db.count())
//      db = zeros<value_t>(b.size());
//
//    // (x_n)(mu_n)'
//    prods = tbblas::prod(trans(V), dH);
//
//    // Calculate the total activation of the hidden and visible units
//    hidact = sum(dH, 0);
//
//    // TODO: decide the sign depending on the objective function
//    // TODO: need to update the objective function for every layer in the nn class
//
//    dW += value_t(-1) * (duW * v - u * prods) / (v * v);
//    db += value_t(-1) * (dub * v - u * hidact) / (v * v);
//
//    ++_current_batch_size;
//  }

  int get_gradient(vector_t& parameters, int offset, value_t weightcost, bool transpose = false) {
    assert(dW.size() == W.size());
    assert(db.size() == b.size());

    if (transpose)
      parameters[seq(offset), seq((int)W.count())] = reshape(trans(dW) / _current_batch_size + weightcost * trans(W), seq((int)W.count()));
    else
      parameters[seq(offset), seq((int)W.count())] = reshape(dW / _current_batch_size + weightcost * W, seq((int)W.count()));
    offset += W.count();

    parameters[seq(offset), seq((int)b.count())] = reshape(db / _current_batch_size, seq((int)b.count()));
    offset += b.count();
    return offset;
  }

  int get_Rgradient(vector_t& parameters, int offset, value_t weightcost, bool transpose = false) {
    assert(RdW.size() == W.size());
    assert(Rdb.size() == b.size());

    // TODO: don't know what to do with the weight cost
    if (transpose)
      parameters[seq(offset), seq((int)W.count())] = reshape(trans(RdW) / _current_batch_size, seq((int)W.count()));
    else
      parameters[seq(offset), seq((int)W.count())] = reshape(RdW / _current_batch_size, seq((int)W.count()));
    offset += W.count();

    parameters[seq(offset), seq((int)b.count())] = reshape(Rdb / _current_batch_size, seq((int)b.count()));
    offset += b.count();
    return offset;
  }

  void update_model(value_t weightcost) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (_current_batch_size) {
      this->update_delta(dW / _current_batch_size + weightcost * W, 0);
      this->update_delta(db / _current_batch_size, 1);

      W = W + reshape(this->delta(0), W.size());
      b = b + reshape(this->delta(1), b.size());

      dW = zeros<value_t>(dW.size());
      db = zeros<value_t>(db.size());
    }

    _current_batch_size = 0;
    _host_updated = false;
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

  // Access to model data
  matrix_t& Rvisibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return RV;
  }

  matrix_t& Rhiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return RH;
  }

  // Don't change the return value.
  matrix_t& visible_Rdeltas() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return RdV;
  }
};

template<class T, class Trainer, class Enable>
const T nn_layer<T, Trainer, Enable>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_RBM_HPP_ */
