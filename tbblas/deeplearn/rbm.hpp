/*
 * rbm.hpp
 *
 *  Created on: Jul 3, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_RBM_HPP_
#define TBBLAS_DEEPLEARN_RBM_HPP_

// TODO: lazy initialization of variables used for training
// TODO: momentum_step and adadelta_step replace init_gradient and apply_gradient
// TODO: counter in update_gradient

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

template<class T>
class rbm {
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef typename matrix_t::dim_t dim_t;

  typedef tbblas::random_tensor2<value_t, 2, true, tbblas::uniform<value_t> > uniform_t;
  typedef tbblas::random_tensor2<value_t, 2, true, tbblas::normal<value_t> > normal_t;

  typedef rbm_model<value_t> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  matrix_t W, b, c, dW, db, dc, m, mean, stddev;
  matrix_t dW2, db2, dc2, deltaW2, deltab2, deltac2, deltaW, deltab, deltac;

  // visible and hidden units in GPU memory
  matrix_t V, H, h_drop;
  uniform_t v_rand, h_rand;
  normal_t v_noise, h_noise;

  // Helper variables
  matrix_t prods, visact, hidact, diffprobs;

  bool _memory_allocated, _double_weights, _host_updated;

  value_t _sparsity_target, _sparsity_weight, _dropout_rate, _positive_update_count, _negative_update_count;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  rbm(model_t& model) : model(model),
    _memory_allocated(false), _double_weights(false), _host_updated(true),
    _sparsity_target(0.1), _sparsity_weight(0), _dropout_rate(0),
    _positive_update_count(0), _negative_update_count(0)
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

    if (m.size() != b.size())
      m = ones<value_t>(b.size());

    W = W * repeat(trans(m), W.size() / trans(m).size());

//    dW = zeros<value_t>(W.size());
//    db = zeros<value_t>(b.size());
//    dc = zeros<value_t>(c.size());
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

    tbblas::synchronize();
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
      case unit_type::Bernoulli: V = sigm(V) > v_rand();    break;
      case unit_type::MyReLU:
      case unit_type::ReLU:      V = max(0.0, V + sqrt(sigm(V)) * v_noise()); break;
      case unit_type::ReLU1:     V = min(1.0, max(0.0, V + (V > 0) * (V < 1.0) * v_noise())); break;
      case unit_type::ReLU2:     V = min(2.0, max(0.0, V + (V > 0) * (V < 2.0) * v_noise())); break;
      case unit_type::ReLU4:     V = min(4.0, max(0.0, V + (V > 0) * (V < 4.0) * v_noise())); break;
      case unit_type::ReLU8:     V = min(8.0, max(0.0, V + (V > 0) * (V < 8.0) * v_noise())); break;
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
      case unit_type::Bernoulli: H = sigm(H) > h_rand(); break;
      case unit_type::MyReLU:
      case unit_type::ReLU:      H = max(0.0, H + sqrt(sigm(H)) * h_noise()); break;
      case unit_type::ReLU1:     H = min(1.0, max(0.0, H + (H > 0) * (H < 1.0) * h_noise())); break;
      case unit_type::ReLU2:     H = min(2.0, max(0.0, H + (H > 0) * (H < 2.0) * h_noise())); break;
      case unit_type::ReLU4:     H = min(4.0, max(0.0, H + (H > 0) * (H < 4.0) * h_noise())); break;
      case unit_type::ReLU8:     H = min(8.0, max(0.0, H + (H > 0) * (H < 8.0) * h_noise())); break;
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

    h_drop = h_rand() > _dropout_rate;
  }

//  void init_gradient_updates(value_t epsilonw = 1, value_t momentum = 0, value_t weightcost = 0) {
//    if (!_memory_allocated)
//      allocate_gpu_memory();
//
//    dW = momentum * dW - epsilonw * weightcost * W;
//    db = momentum * db;
//    dc = momentum * dc;
//  }

  void update_positive_gradient() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (!H.count())
      throw std::runtime_error("Hidden units not calculated.");

    if (!dW.count())
      dW = zeros<value_t>(W.size());
    if (!db.count())
      db = zeros<value_t>(b.size());
    if (!dc.count())
      dc = zeros<value_t>(c.size());

    // (x_n)(mu_n)'
    prods = tbblas::prod(trans(V), H);

    // Calculate the total activation of the hidden and visible units
    hidact = sum(H, 0);
    visact = sum(V, 0);

    dW += prods / V.size()[0];
    db += visact / V.size()[0];
    dc += hidact / V.size()[0];

    if (_sparsity_weight > 0) {
      diffprobs = -H + _sparsity_target;

      prods = prod(trans(V), diffprobs);
      hidact = sum(diffprobs, 0);

      dW += _sparsity_weight * prods / V.size()[0];
      dc += _sparsity_weight* hidact / V.size()[0];
    }

    ++_positive_update_count;
  }

  void update_negative_gradient() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    // (x_n)(mu_n)'
    prods = tbblas::prod(trans(V), H);

    // Calculate the total activation of the hidden and visible units
    hidact = sum(H, 0);
    visact = sum(V, 0);

    dW = dW - prods / V.size()[0];
    db = db - visact / V.size()[0];
    dc = dc - hidact / V.size()[0];

    ++_negative_update_count;
  }

  // CAUTION: ONLY USE THIS FUNCTION IF YOU KNOW WHAT YOU ARE DOING
  // Should probably check, if the model is the same and needs a don't
  // write to model mechanism
  void accumulate_gradients(rbm<value_t>& rbm) {
    rbm.dW = dW = dW + rbm.dW;
    rbm.db = db = db + rbm.db;
    rbm.dc = dc = dc + rbm.dc;
  }

  void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    // Lazy initialization
    if (!deltaW.count())
      deltaW = zeros<value_t>(W.size());
    if (!deltab.count())
      deltab = zeros<value_t>(b.size());
    if (!deltac.count())
      deltac = zeros<value_t>(c.size());

    if (_positive_update_count != _negative_update_count)
      throw std::runtime_error("Number of positive gradient updates must be equal to the number of negative updates.");

    if (_positive_update_count) {
      deltaW = momentum * deltaW + dW / _positive_update_count - weightcost * W;
      deltab = momentum * deltab + db / _positive_update_count;
      deltac = momentum * deltac + dc / _positive_update_count;

      W = W + epsilon * deltaW;
      b = b + epsilon * deltab;
      c = c + epsilon * deltac;

      dW = zeros<value_t>(dW.size());
      db = zeros<value_t>(db.size());
      dc = zeros<value_t>(dc.size());
    }

    _positive_update_count = _negative_update_count = 0;

    _host_updated = false;
  }

  void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {
    if (!dW2.count())
      dW2 = zeros<value_t>(W.size());
    if (!db2.count())
      db2 = zeros<value_t>(b.size());
    if (!dc2.count())
      dc2 = zeros<value_t>(c.size());
    if (!deltaW2.count())
      deltaW2 = zeros<value_t>(W.size());
    if (!deltab2.count())
      deltab2 = zeros<value_t>(b.size());
    if (!deltac2.count())
      deltac2 = zeros<value_t>(c.size());

    if (_positive_update_count != _negative_update_count)
      throw std::runtime_error("Number of positive gradient updates must be equal to the number of negative updates.");

    if (_positive_update_count) {
      dW = dW / _positive_update_count - weightcost * W;
      db = db / _positive_update_count;
      dc = dc / _positive_update_count;

      dW2 = momentum * dW2 + (1.0 - momentum) * dW * dW;
      db2 = momentum * db2 + (1.0 - momentum) * db * db;

      // note that deltaW = sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      W = W + sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      b = b + sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db;
      b = b + sqrt(deltac2 + epsilon) / sqrt(dc2 + epsilon) * dc;

      deltaW2 = momentum * deltaW2 + (1.0 - momentum) * sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW * sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      deltab2 = momentum * deltab2 + (1.0 - momentum) * sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db * sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db;
      deltac2 = momentum * deltac2 + (1.0 - momentum) * sqrt(deltac2 + epsilon) / sqrt(dc2 + epsilon) * dc * sqrt(deltac2 + epsilon) / sqrt(dc2 + epsilon) * dc;

      dW = zeros<value_t>(dW.size());
      db = zeros<value_t>(db.size());
      dc = zeros<value_t>(dc.size());
    }

    _positive_update_count = _negative_update_count = 0;

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
