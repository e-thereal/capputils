/*
 * rbm.hpp
 *
 *  Created on: Jul 3, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_RBM_HPP_
#define TBBLAS_DEEPLEARN_RBM_HPP_

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
#include <tbblas/deeplearn/rbm_model.hpp>

#include <iostream>
#include <stdexcept>

namespace tbblas {

namespace deeplearn {

/// This class creates multiple threads
/**
 * Some changes to the previous design:
 * - No thread local variables. Thread local variables are replaced by vectors of
 *   shared pointers. Better control over the creation and destruction of variables.
 * - Every thread has a local reference to the memory. Makes code cleaner.
 */

template<class T>
class rbm {
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef typename matrix_t::dim_t dim_t;

  typedef tbblas::random_tensor<value_t, 2, true, tbblas::uniform<value_t> > uniform_t;
  typedef tbblas::random_tensor<value_t, 2, true, tbblas::normal<value_t> > normal_t;

  typedef rbm_model<value_t> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  matrix_t W, b, c, dW, db, dc, m, mean, stddev;

  // visible and hidden units in GPU memory
  matrix_t V, H, h_drop;
  uniform_t v_rand, h_rand;
  normal_t v_noise, h_noise;

  // Helper variables
  matrix_t prods, visact, hidact, diffprobs;

  bool _memory_allocated, _double_weights, _host_updated;

  value_t _sparsity_target, _sparsity_weight, _dropout_rate;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  rbm(model_t& model) : model(model),
    _memory_allocated(false), _double_weights(false), _host_updated(true),
    _sparsity_target(0.1), _sparsity_weight(0), _dropout_rate(0)
  { }

private:
  rbm(const rbm&);

public:
  virtual ~rbm() {
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
    b = model.visible_bias();
    c = model.hidden_bias();
    m = model.mask();

    mean = model.mean();
    stddev = model.stddev();

    if (m.size() != mean.size())
      m = ones<value_t>(mean.size());

    W = W * repeat(trans(m), W.size() / trans(m).size());

    dW = zeros<value_t>(W.size());
    db = zeros<value_t>(b.size());
    dc = zeros<value_t>(c.size());
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    model.set_weights(W);
    model.set_visible_bias(b);
    model.set_hidden_bias(c);
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    V = ((V - repeat(mean, V.size() / mean.size())) / repeat(stddev, V.size() / stddev.size())) * tbblas::repeat(m, V.size() / m.size());
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    V = ((V * repeat(stddev, V.size() / stddev.size())) + repeat(mean, V.size() / mean.size())) * tbblas::repeat(m, V.size() / m.size());
  }

  void infer_visibles(bool onlyFilters = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    V = prod(H, trans(W));

    if (!onlyFilters) {
      V = V + repeat(b, V.size() / b.size());

      switch (model.visibles_type()) {
        case unit_type::Gaussian:  break;
        case unit_type::Bernoulli: V = sigm(V);    break;
        case unit_type::ReLU:      V = max(0, V);  break;
        case unit_type::MyReLU:    V = nrelu_mean(V); break;
        case unit_type::ReLU1:     V = min(1.0, max(0.0, V));  break;
        case unit_type::ReLU2:     V = min(2.0, max(0.0, V));  break;
        case unit_type::ReLU4:     V = min(4.0, max(0.0, V));  break;
        case unit_type::ReLU8:     V = min(8.0, max(0.0, V));  break;
      }

      V = V * repeat(m, V.size() / m.size());
    }
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    H = prod(V, W);
    H = H + repeat(c, H.size() / c.size());

    switch (model.hiddens_type()) {
      case unit_type::Bernoulli: H = sigm(H);    break;
      case unit_type::ReLU:      H = max(0, H);  break;
      case unit_type::MyReLU:    H = nrelu_mean(H); break;
      case unit_type::ReLU1:     H = min(1.0, max(0.0, H));  break;
      case unit_type::ReLU2:     H = min(2.0, max(0.0, H));  break;
      case unit_type::ReLU4:     H = min(4.0, max(0.0, H));  break;
      case unit_type::ReLU8:     H = min(8.0, max(0.0, H));  break;
    }

    if (_dropout_rate > 0)
      H = H * h_drop / (1. - _dropout_rate);
  }

  void sample_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();


    V = prod(H, trans(W));
    V = V + repeat(b, V.size() / b.size());

    // Initialize random matrices if necessary
    switch (model.visibles_type()) {
      case unit_type::Bernoulli:
        if (v_rand.size() != V.size())
          v_rand.resize(V.size());
        break;

      case unit_type::ReLU:
      case unit_type::MyReLU:
      case unit_type::ReLU1:
      case unit_type::ReLU2:
      case unit_type::ReLU4:
      case unit_type::ReLU8:
        if (v_noise.size() != V.size())
          v_noise.resize(V.size());
        break;
    }

    switch (model.visibles_type()) {
      case unit_type::Gaussian:  break;
      case unit_type::Bernoulli: V = sigm(V) > v_rand;    break;
      case unit_type::MyReLU:
      case unit_type::ReLU:      V = max(0.0, V + sqrt(sigm(V)) * v_noise); break;
      case unit_type::ReLU1:     V = min(1.0, max(0.0, V + (V > 0) * (V < 1.0) * v_noise)); break;
      case unit_type::ReLU2:     V = min(2.0, max(0.0, V + (V > 0) * (V < 2.0) * v_noise)); break;
      case unit_type::ReLU4:     V = min(4.0, max(0.0, V + (V > 0) * (V < 4.0) * v_noise)); break;
      case unit_type::ReLU8:     V = min(8.0, max(0.0, V + (V > 0) * (V < 8.0) * v_noise)); break;
    }

    V = V * repeat(m, V.size() / m.size());
  }

  void sample_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    H = prod(V, W);
    H = H + repeat(c, H.size() / c.size());

    // Initialize random matrices if necessary
    switch (model.hiddens_type()) {
      case unit_type::Bernoulli:
        if (h_rand.size() != H.size())
          h_rand.resize(H.size());
        break;

      case unit_type::ReLU:
      case unit_type::MyReLU:
      case unit_type::ReLU1:
      case unit_type::ReLU2:
      case unit_type::ReLU4:
      case unit_type::ReLU8:
        if (h_noise.size() != H.size())
          h_noise.resize(H.size());
        break;
    }

    switch (model.hiddens_type()) {
      case unit_type::Bernoulli: H = sigm(H) > h_rand; break;
      case unit_type::MyReLU:
      case unit_type::ReLU:      H = max(0.0, H + sqrt(sigm(H)) * h_noise); break;
      case unit_type::ReLU1:     H = min(1.0, max(0.0, H + (H > 0) * (H < 1.0) * h_noise)); break;
      case unit_type::ReLU2:     H = min(2.0, max(0.0, H + (H > 0) * (H < 2.0) * h_noise)); break;
      case unit_type::ReLU4:     H = min(4.0, max(0.0, H + (H > 0) * (H < 4.0) * h_noise)); break;
      case unit_type::ReLU8:     H = min(8.0, max(0.0, H + (H > 0) * (H < 8.0) * h_noise)); break;
    }

    if (_dropout_rate > 0)
      H = H * h_drop / (1. - _dropout_rate);
  }

  void init_dropout(value_t rate) {
    using namespace tbblas;

    if (rate < 0 || rate >= 1)
      throw std::runtime_error("Drop out rate must be in [0,1).");

    _dropout_rate = rate;

    if (!_memory_allocated)
      allocate_gpu_memory();

    dim_t h_size = seq(V.size()[0], W.size()[1]);
    if (h_rand.size() != h_size)
      h_rand.resize(h_size);

    h_drop = h_rand > _dropout_rate;
  }

  void init_gradient_updates(value_t momentum = 0, value_t weightcost = 0) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    dW = momentum * dW - weightcost * W;
    db = momentum * db;
    dc = momentum * dc;
  }

  void update_positive_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    // (x_n)(mu_n)'
    prods = tbblas::prod(trans(V), H);

    // Calculate the total activation of the hidden and visible units
    hidact = sum(H, 0);
    visact = sum(V, 0);

    dW = dW + epsilonw * prods / V.size()[0];
    db = db + epsilonvb * visact / V.size()[0];
    dc = dc + epsilonhb * hidact / V.size()[0];

    if (_sparsity_weight > 0) {
      diffprobs = H - _sparsity_target;

      prods = prod(trans(V), diffprobs);
      hidact = sum(diffprobs, 0);

      dW = dW + epsilonw * _sparsity_weight * prods / V.size()[0];
      dc = dc + epsilonhb * _sparsity_weight* hidact / V.size()[0];
    }
  }

  void update_negative_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    // (x_n)(mu_n)'
    prods = tbblas::prod(trans(V), H);

    // Calculate the total activation of the hidden and visible units
    hidact = sum(H, 0);
    visact = sum(V, 0);

    dW = dW - epsilonw * prods / V.size()[0];
    db = db - epsilonvb * visact / V.size()[0];
    dc = dc - epsilonhb * hidact / V.size()[0];
  }

  // CAUTION: ONLY USE THIS FUNCTION IF YOU KNOW WHAT YOU ARE DOING
  // Should probably check, if the model is the same and needs a don't
  // write to model mechanism
  void accumulate_gradients(rbm<value_t>& rbm) {
    rbm.dW = dW = dW + rbm.dW;
    rbm.db = db = db + rbm.db;
    rbm.dc = dc = dc + rbm.dc;
  }

  void apply_gradient() {
    _host_updated = false;

    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    W = W + dW;
    b = b + db;
    c = c + dc;
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

  void set_sparsity_target(value_t target) {
    _sparsity_target = target;
  }

  void set_sparsity_weight(value_t weight) {
    _sparsity_weight = weight;
  }
};

template<class T>
const T rbm<T>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_RBM_HPP_ */
