/*
 * dnn_layer_model.hpp
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DNN_LAYER_MODEL_HPP_
#define TBBLAS_DEEPLEARN_DNN_LAYER_MODEL_HPP_

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
class dnn_layer_model {
public:
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  /**
   * Versions:
   * 0: initial version, already comes with a version number (no magic code needed)
   * 1: Visible bias and filters are stored in it's original form. cnn_layer module takes care of rearranging things
   *    Kernel size is original kernel size before rearranging due to striding. Has a dedicated field for
   *    visibles size.
   * 2: Mask added.
   */
  static const uint16_t CURRENT_VERSION = 2;

protected:
  uint16_t _version;

  // Model in CPU memory
  v_host_tensor_t _filters;
  host_tensor_t _bias, _mask;
  dim_t _kernel_size, _stride_size, _pooling_size;

  tbblas::deeplearn::activation_function _activation_function;
  tbblas::deeplearn::convolution_type _convolution_type;
  tbblas::deeplearn::pooling_method _pooling_method;

  value_t _mean, _stddev;
  bool _shared_biases;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  dnn_layer_model()
   : _version(CURRENT_VERSION),
     _stride_size(seq<dimCount>(1)), _pooling_size(seq<dimCount>(1)),
     _mean(0), _stddev(1), _shared_biases(false) { }

  dnn_layer_model(const dnn_layer_model<T,dims>& model)
   : _version(model._version),
     _kernel_size(model.kernel_size()), _stride_size(model.stride_size()),
     _pooling_size(model.pooling_size()),
     _activation_function(model.activation_function()),
     _convolution_type(model.convolution_type()),
     _pooling_method(model.pooling_method()),
     _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias())
  {
    set_filters(model.filters());
    set_bias(model.bias());
    set_mask(model.mask());
  }

  template<class U>
  dnn_layer_model(const dnn_layer_model<U,dims>& model)
   : _kernel_size(model.kernel_size()), _stride_size(model.stride_size()),
     _pooling_size(model.pooling_size()),
     _activation_function(model.activation_function()),
     _convolution_type(model.convolution_type()),
     _pooling_method(model.pooling_method()),
     _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias())
  {
    _filters.resize(model.filters().size());
    for (size_t i = 0; i < model.filters().size(); ++i)
      _filters[i] = boost::make_shared<host_tensor_t>(*model.filters()[i]);

    _bias = model.bias();
    _mask = model.mask();
  }

  virtual ~dnn_layer_model() { }

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

  void set_bias(const host_tensor_t& bias) {
    _bias = bias;
  }

  const host_tensor_t& bias() const {
    return _bias;
  }

  void set_mask(const host_tensor_t& mask) {
    _mask = mask;
  }

  const host_tensor_t& mask() const {
    return _mask;
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

  dim_t visibles_size() const {
    return _bias.size();
  }

  size_t visibles_count() const {
    return _bias.count();
  }

  dim_t hiddens_size() const {
    // Initialize with rearranged size (ceil(visible size / stride size))
    dim_t hidden_size = (visibles_size() + _stride_size - 1) / _stride_size;
    if (convolution_type() == convolution_type::Valid) {
      dim_t kernel_size = (_kernel_size + _stride_size - 1) / _stride_size;
      hidden_size = hidden_size - kernel_size + 1;
    }
    hidden_size[dimCount - 1] = filter_count();

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

  dim_t outputs_size() const {
    if (has_pooling_layer())
      return pooled_size();
    else
      return hiddens_size();
  }

  size_t outputs_count() const {
    return outputs_size().prod();
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

  bool has_pooling_layer() const {
    return _pooling_method != tbblas::deeplearn::pooling_method::NoPooling &&
        _pooling_size != seq<dimCount>(1);
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
      _bias = ones<value_t>(_bias.size()) * sum(_bias) / _bias.count();
    }
    _shared_biases = shared;
  }

  bool shared_bias() const {
    return _shared_biases;
  }

  bool is_valid() const {
    return ((_kernel_size % _stride_size) == (visibles_size() % _stride_size)) &&
        ((hiddens_size() % _pooling_size) == seq<dimCount>(0));
  }

  void change_size(dim_t size) {
    throw std::runtime_error("Not implemented.");
//    if (!_shared_biases)
//      throw std::runtime_error("Changing the size of a CNN layer is only supported for shared bias terms.");
//
//    dim_t layerSize = size;
//    layerSize[dimCount - 1] = 1;
//
//    for (size_t i = 0; i < _filters.size(); ++i) {
//      *_biases[i] = ones<value_t>(layerSize) * (*_biases[i])[seq<dimCount>(0)];
//    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_DNN_LAYER_MODEL_HPP_ */
