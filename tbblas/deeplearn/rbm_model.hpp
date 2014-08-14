/*
 * rbm_model.hpp
 *
 *  Created on: Jul 3, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_RBM_MODEL_HPP_
#define TBBLAS_DEEPLEARN_RBM_MODEL_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/deeplearn/unit_type.hpp>

namespace tbblas {

namespace deeplearn {

/// This class contains the parameters of an RBM
/**
 * Use the rbm class for inference, sampling and training of RBMs
 */

template<class T>
class rbm_model {
public:
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> device_matrix_t;
  typedef typename host_matrix_t::dim_t dim_t;

protected:
  // Model in CPU memory
  host_matrix_t _visibleBiases, _hiddenBiases, _weights, _mask, _mean, _stddev;
  unit_type _visibleUnitType, _hiddenUnitType;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  rbm_model() { }

  rbm_model(const rbm_model<T>& model)
   : _visibleBiases(model.visible_bias()), _hiddenBiases(model.hidden_bias()),
     _weights(model.weights()), _mask(model.mask()),
     _mean(model.mean()), _stddev(model.stddev()),
     _visibleUnitType(model.visibles_type()), _hiddenUnitType(model.hiddens_type())
  { }

  template<class U>
  rbm_model(const rbm_model<U>& model)
   : _visibleBiases(model.visible_bias()), _hiddenBiases(model.hidden_bias()),
     _weights(model.weights()), _mask(model.mask()),
     _mean(model.mean()), _stddev(model.stddev()),
     _visibleUnitType(model.visibles_type()), _hiddenUnitType(model.hiddens_type())
  { }

  virtual ~rbm_model() { }

public:
  void set_weights(host_matrix_t& weights) {
    _weights = weights;
  }

  void set_weights(device_matrix_t& weights) {
    _weights = weights;
  }

  const host_matrix_t& weights() const {
    return _weights;
  }

  void set_visible_bias(host_matrix_t& bias) {
    _visibleBiases = bias;
  }

  void set_visible_bias(device_matrix_t& bias) {
    _visibleBiases = bias;
  }

  const host_matrix_t& visible_bias() const {
    return _visibleBiases;
  }

  void set_hidden_bias(host_matrix_t& bias) {
    _hiddenBiases = bias;
  }

  void set_hidden_bias(device_matrix_t& bias) {
    _hiddenBiases = bias;
  }

  const host_matrix_t& hidden_bias() const {
    return _hiddenBiases;
  }

  void set_mask(host_matrix_t& mask) {
    _mask = mask;
  }

  void set_mask(device_matrix_t& mask) {
    _mask = mask;
  }

  const host_matrix_t& mask() const {
    return _mask;
  }

  void set_visibles_type(const unit_type& type) {
    _visibleUnitType = type;
  }

  const unit_type& visibles_type() const {
    return _visibleUnitType;
  }

  void set_hiddens_type(const unit_type& type) {
    _hiddenUnitType = type;
  }

  const unit_type& hiddens_type() const {
    return _hiddenUnitType;
  }

  size_t visibles_count() const {
    return _visibleBiases.count();
  }

  size_t hiddens_count() const {
    return _hiddenBiases.count();
  }

  void set_mean(host_matrix_t& mean) {
    _mean = mean;
  }

  void set_mean(device_matrix_t& mean) {
    _mean = mean;
  }

  const host_matrix_t& mean() const {
    return _mean;
  }

  void set_stddev(host_matrix_t& stddev) {
    _stddev = stddev;
  }

  void set_stddev(device_matrix_t& stddev) {
    _stddev = stddev;
  }

  const host_matrix_t& stddev() const {
    return _stddev;
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_RBM_MODEL_HPP_ */
