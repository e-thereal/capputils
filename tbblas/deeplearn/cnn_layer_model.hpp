/*
 * cnn_layer_model.hpp
 *
 *  Created on: Aug 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CNN_LAYER_MODEL_HPP_
#define TBBLAS_DEEPLEARN_CNN_LAYER_MODEL_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/util.hpp>

#include <tbblas/ones.hpp>
#include <tbblas/sum.hpp>

#include <tbblas/deeplearn/convolution_type.hpp>
#include <tbblas/deeplearn/activation_function.hpp>
#include <tbblas/deeplearn/pooling_method.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class cnn_layer_model {
public:
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  /**
   * Versions:
   * 0: initial version, already comes with a version number (no magic code needed)
   */
  static const uint16_t CURRENT_VERSION = 0;

protected:
  uint16_t _version;

  // Model in CPU memory
  v_host_tensor_t _filters, _biases;
  dim_t _kernel_size, _stride_size, _pooling_size;

  tbblas::deeplearn::activation_function _activation_function;
  tbblas::deeplearn::convolution_type _convolution_type;
  tbblas::deeplearn::pooling_method _pooling_method;

  value_t _mean, _stddev;
  bool _shared_biases;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  cnn_layer_model()
   : _version(CURRENT_VERSION),
     _stride_size(seq<dimCount>(1)), _pooling_size(seq<dimCount>(1)), _mean(0), _stddev(1), _shared_biases(false) { }

  cnn_layer_model(const cnn_layer_model<T,dims>& model)
   : _version(model._version),
     _kernel_size(model.kernel_size()), _stride_size(model.stride_size()), _pooling_size(model.pooling_size()),
     _activation_function(model.activation_function()),
     _convolution_type(model.convolution_type()), _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias())
  {
    set_filters(model.filters());
    set_bias(model.bias());
  }

  template<class U>
  cnn_layer_model(const cnn_layer_model<U,dims>& model)
   : _kernel_size(model.kernel_size()), _stride_size(model.stride_size()), _pooling_size(model.pooling_size()),
     _activation_function(model.activation_function()),
     _convolution_type(model.convolution_type()), _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias())
  {
    _filters.resize(model.filters().size());
    for (size_t i = 0; i < model.filters().size(); ++i)
      _filters[i] = boost::make_shared<host_tensor_t>(*model.filters()[i]);

    _biases.resize(model.bias().size());
    for (size_t i = 0; i < model.bias().size(); ++i)
      _biases[i] = boost::make_shared<host_tensor_t>(*model.bias()[i]);
  }

  virtual ~cnn_layer_model() { }

public:
  void set_version(uint16_t version) {
    _version = version;
  }

  uint16_t version() const {
    return _version;
  }

  void set_filters(const v_host_tensor_t& filters) {
    if (filters.size() == 0)
      throw std::runtime_error("A CNN layer must contain at least one filter.");
    _filters.resize(filters.size());
    for (size_t i = 0; i < filters.size(); ++i)
      _filters[i] = boost::make_shared<host_tensor_t>(*filters[i]);
  }

  const v_host_tensor_t& filters() const {
    return _filters;
  }

  size_t filter_count() const {
    return _filters.size();
  }

  void set_bias(const v_host_tensor_t& bias) {
    _biases.resize(bias.size());
    for (size_t i = 0; i < bias.size(); ++i)
      _biases[i] = boost::make_shared<host_tensor_t>(*bias[i]);
  }

  const v_host_tensor_t& bias() const {
    return _biases;
  }

  void set_kernel_size(const dim_t& size) {
    _kernel_size = size;
  }

  const dim_t& kernel_size() const {
    return _kernel_size;
  }

  void set_stride_size(const dim_t& size) {
    _stride_size = size;
  }

  const dim_t& stride_size() const {
    return _stride_size;
  }

  void set_activation_function(const tbblas::deeplearn::activation_function& function) {
    _activation_function = function;
  }

  const tbblas::deeplearn::activation_function& activation_function() const {
    return _activation_function;
  }

  dim_t input_size() const {
    dim_t size = visibles_size() * _stride_size;

    size_t count = 1;
    for (size_t i = 0; i < dimCount; ++i)
      count *= _stride_size[i];

    size[dimCount - 1] = size[dimCount - 1] / count;
    return size;
  }

  dim_t visibles_size() const {
    if (_biases.size() == 0)
      throw std::runtime_error("A CNN layer must contain at least one filter.");

    dim_t size = _biases[0]->size();
    size[dimCount - 1] = _filters[0]->size()[dimCount - 1];

    return size;
  }

  size_t visibles_count() const {
    if (_biases.size() == 0)
      throw std::runtime_error("A CNN layer must contain at least one filter.");
    return _biases[0]->count() * _filters[0]->size()[dimCount - 1];
  }

  dim_t hiddens_size() const {
    dim_t hidden_size = visibles_size();
    if (convolution_type() == convolution_type::Valid){
      hidden_size = hidden_size - kernel_size() + 1;
    }
    hidden_size[dimCount - 1] = filters().size();

    return hidden_size;
  }

  size_t hiddens_count() const {
    return hiddens_size().prod();
  }

  dim_t pooled_size() const {
    return hiddens_size() / _pooling_size;
  }

  size_t pooled_count() const {
    return pooled_size().prod();
  }

  dim_t output_size() const {
    if (_pooling_method == tbblas::deeplearn::pooling_method::NoPooling)
      return hiddens_size();
    else
      return pooled_size();
  }

  size_t output_count() const {
    return output_size().prod();
  }

  void set_convolution_type(const tbblas::deeplearn::convolution_type& type) {
    _convolution_type = type;
  }

  const tbblas::deeplearn::convolution_type& convolution_type() const {
    return _convolution_type;
  }

  void set_pooling_size(const dim_t& size) {
    _pooling_size = size;
  }

  const dim_t& pooling_size() const {
    return _pooling_size;
  }

  void set_pooling_method(const tbblas::deeplearn::pooling_method& method) {
    _pooling_method = method;
  }

  const tbblas::deeplearn::pooling_method& pooling_method() const {
    return _pooling_method;
  }

  void set_mean(value_t mean) {
    _mean = mean;
  }

  value_t mean() const {
    return _mean;
  }

  void set_stddev(value_t stddev) {
    _stddev = stddev;
  }

  value_t stddev() const {
    return _stddev;
  }

  void set_shared_bias(bool shared) {
    if (!_shared_biases && shared) {
      for (size_t i = 0; i < _biases.size(); ++i) {
        *_biases[i] = ones<value_t>(_biases[i]->size()) * sum(*_biases[i]) / _biases[i]->count();
      }
    }
    _shared_biases = shared;
  }

  bool shared_bias() const {
    return _shared_biases;
  }

  void change_size(dim_t size) {
    if (!_shared_biases)
      throw std::runtime_error("Changing the size of a CNN layer is only supported for shared bias terms.");

    dim_t layerSize = size;
    layerSize[dimCount - 1] = 1;

    for (size_t i = 0; i < _filters.size(); ++i) {
      *_biases[i] = ones<value_t>(layerSize) * (*_biases[i])[seq<dimCount>(0)];
    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_LAYER_MODEL_HPP_ */
