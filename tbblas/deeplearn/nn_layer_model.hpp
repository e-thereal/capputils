/*
 * nn_layer_model.hpp
 *
 *  Created on: Aug 12, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_NN_LAYER_MODEL_HPP_
#define TBBLAS_DEEPLEARN_NN_LAYER_MODEL_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/deeplearn/activation_function.hpp>

namespace tbblas {

namespace deeplearn {

/// This class contains the parameters of a single layer of a neural network
/**
 * Use the nn_layer class for inference, sampling and training of a single neural network layer
 */

template<class T>
class nn_layer_model {
public:
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> device_matrix_t;
  typedef typename host_matrix_t::dim_t dim_t;

protected:
  // Model in CPU memory
  host_matrix_t _biases, _weights, _mean, _stddev;
  tbblas::deeplearn::activation_function _activation_function;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  nn_layer_model() { }

  nn_layer_model(const nn_layer_model<T>& model)
   : _biases(model.bias()), _weights(model.weights()),
     _mean(model.mean()), _stddev(model.stddev()), _activation_function(model.activation_function())
  { }

  template<class U>
  nn_layer_model(const nn_layer_model<U>& model)
    : _biases(model.bias()), _weights(model.weights()),
      _mean(model.mean()), _stddev(model.stddev()), _activation_function(model.activation_function())
  { }

  virtual ~nn_layer_model() { }

public:
  template<class U>
  void set_weights(const tensor<U, 2>& weights) {
    _weights = weights;
  }

  template<class U>
  void set_weights(const tensor<U, 2, true>& weights) {
    _weights = weights;
  }

  const host_matrix_t& weights() const {
    return _weights;
  }

  template<class U>
  void set_bias(const tensor<U, 2>& bias) {
    _biases = bias;
  }

  template<class U>
  void set_bias(const tensor<U, 2, true>& bias) {
    _biases = bias;
  }

  const host_matrix_t& bias() const {
    return _biases;
  }

  void set_activation_function(const tbblas::deeplearn::activation_function& function) {
    _activation_function = function;
  }

  const tbblas::deeplearn::activation_function& activation_function() const {
    return _activation_function;
  }

  size_t visibles_count() const {
    return _weights.size()[0];
  }

  size_t hiddens_count() const {
    return _weights.size()[1];
  }

  template<class U>
  void set_mean(const tensor<U, 2>& mean) {
    _mean = mean;
  }

  template<class U>
  void set_mean(const tensor<U, 2, true>& mean) {
    _mean = mean;
  }

  const host_matrix_t& mean() const {
    return _mean;
  }

  template<class U>
  void set_stddev(const tensor<U, 2>& stddev) {
    _stddev = stddev;
  }

  template<class U>
  void set_stddev(const tensor<U, 2, true>& stddev) {
    _stddev = stddev;
  }

  const host_matrix_t& stddev() const {
    return _stddev;
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_RBM_MODEL_HPP_ */
