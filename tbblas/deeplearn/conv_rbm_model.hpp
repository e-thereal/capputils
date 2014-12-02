/*
 * conv_rbm_model.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONV_RBM_MODEL_HPP_
#define TBBLAS_DEEPLEARN_CONV_RBM_MODEL_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/util.hpp>

#include <tbblas/ones.hpp>
#include <tbblas/sum.hpp>

#include <tbblas/deeplearn/convolution_type.hpp>
#include <tbblas/deeplearn/unit_type.hpp>
#include <tbblas/deeplearn/pooling_method.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

/// This class contains the parameters of a convRBM
/**
 * Use the conv_rbm class for inference, sampling and training of convRBMs
 */

template<class T, unsigned dims>
class conv_rbm_model {
public:
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  /**
   * Versions:
   * 0: initial version, doesn't even have magic code or version number
   * 1: first real version. Comes with magic code, version number and stride size
   * 2: Added pooling information
   */
  static const unsigned CURRENT_VERSION = 2;

protected:
  // Model in CPU memory
  host_tensor_t _visible_biases, _mask;
  v_host_tensor_t _filters, _hidden_biases;
  dim_t _kernel_size, _stride_size, _pooling_size;

  unit_type _visibles_type, _hiddens_type;
  tbblas::deeplearn::convolution_type _convolution_type;
  tbblas::deeplearn::pooling_method _pooling_method;

  value_t _mean, _stddev;
  bool _shared_biases;
  unsigned _version;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm_model() : _stride_size(seq<dimCount>(1)), _pooling_size(seq<dimCount>(1)), _mean(0), _stddev(1), _shared_biases(false), _version(CURRENT_VERSION)
  {
  }

  conv_rbm_model(const conv_rbm_model<T,dims>& model)
   : _visible_biases(model.visible_bias()), _mask(model.mask()), _kernel_size(model.kernel_size()), _stride_size(model.stride_size()), _pooling_size(model.pooling_size()),
     _visibles_type(model.visibles_type()), _hiddens_type(model.hiddens_type()), _convolution_type(model.convolution_type()), _pooling_method(model.pooling_method()),
     _mean(model.mean()), _stddev(model.stddev()), _shared_biases(model.shared_bias()), _version(model.version())
  {
    set_filters(model.filters());
    set_hidden_bias(model.hidden_bias());
  }

  template<class U>
  conv_rbm_model(const conv_rbm_model<U,dims>& model)
   : _visible_biases(model.visible_bias()), _mask(model.mask()), _kernel_size(model.kernel_size()), _stride_size(model.stride_size()), _pooling_size(model.pooling_size()),
     _visibles_type(model.visibles_type()), _hiddens_type(model.hiddens_type()), _convolution_type(model.convolution_type()), _pooling_method(model.pooling_method()),
     _mean(model.mean()), _stddev(model.stddev()), _shared_biases(model.shared_bias()), _version(model.version())
  {
    _filters.resize(model.filters().size());
    for (size_t i = 0; i < model.filters().size(); ++i)
      _filters[i] = boost::make_shared<host_tensor_t>(*model.filters()[i]);

    _hidden_biases.resize(model.hidden_bias().size());
    for (size_t i = 0; i < model.hidden_bias().size(); ++i)
      _hidden_biases[i] = boost::make_shared<host_tensor_t>(*model.hidden_bias()[i]);
  }

  virtual ~conv_rbm_model() { }

public:
  void set_filters(const v_host_tensor_t& filters) {
    _filters.resize(filters.size());
    for (size_t i = 0; i < filters.size(); ++i)
      _filters[i] = boost::make_shared<host_tensor_t>(*filters[i]);
  }

  v_host_tensor_t& filters() {
    return _filters;
  }

  const v_host_tensor_t& filters() const {
    return _filters;
  }

  size_t filter_count() const {
    return _filters.size();
  }

  void set_visible_bias(host_tensor_t& bias) {
    _visible_biases = bias;
  }

  const host_tensor_t& visible_bias() const {
    return _visible_biases;
  }

  void set_hidden_bias(const v_host_tensor_t& bias) {
    _hidden_biases.resize(bias.size());
    for (size_t i = 0; i < bias.size(); ++i)
      _hidden_biases[i] = boost::make_shared<host_tensor_t>(*bias[i]);
  }

  v_host_tensor_t& hidden_bias() {
    return _hidden_biases;
  }

  const v_host_tensor_t& hidden_bias() const {
    return _hidden_biases;
  }

  void set_mask(host_tensor_t& mask) {
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

  void set_visibles_type(const unit_type& type) {
    _visibles_type = type;
  }

  const unit_type& visibles_type() const {
    return _visibles_type;
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
    return visible_bias().size();
  }

  size_t visibles_count() const {
    return visible_bias().count();
  }

  void set_hiddens_type(const unit_type& type) {
    _hiddens_type = type;
  }

  const unit_type& hiddens_type() const {
    return _hiddens_type;
  }

  dim_t hiddens_size() const {
    dim_t hidden_size = visible_bias().size();
    if (convolution_type() == convolution_type::Valid){
      hidden_size = hidden_size - kernel_size() + 1;
    }
    hidden_size[dimCount - 1] = filters().size();

    return hidden_size;
  }

  size_t hiddens_count() const {
    dim_t hidden_size = hiddens_size();
    size_t count = 1;
    for (size_t i = 0; i < dims; ++i)
      count *= hidden_size[i];
    return count;
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
      _visible_biases = ones<value_t>(_visible_biases.size()) * sum(_visible_biases) / _visible_biases.count();
      for (size_t i = 0; i < _hidden_biases.size(); ++i) {
        *_hidden_biases[i] = ones<value_t>(_hidden_biases[i]->size()) * sum(*_hidden_biases[i]) / _hidden_biases[i]->count();
      }
    }
    _shared_biases = shared;
  }

  bool shared_bias() const {
    return _shared_biases;
  }

  void set_version(unsigned version) {
    _version = version;
  }

  unsigned version() const {
    return _version;
  }

  void change_stride(dim_t stride) {
    if (!_shared_biases)
      throw std::runtime_error("Changing the stride is only supported for shared bias models.");

    if (_stride_size == seq<dimCount>(1)) {

      // Add stride to the model

      host_tensor_t v = rearrange(_visible_biases, stride);
      _visible_biases = v;

      dim_t layerSize = v.size();
      layerSize[dimCount - 1] = 1;

      _mask = ones<value_t>(layerSize);

      host_tensor_t f;
      for (size_t i = 0; i < _filters.size(); ++i) {
        f = rearrange(*_filters[i], stride);
        *_filters[i] = f;
        *_hidden_biases[i] = ones<value_t>(layerSize) * (*_hidden_biases[i])[seq<dimCount>(0)];
      }

      _kernel_size = _filters[0]->size();
    } else if (stride == seq<dimCount>(1)) {

      // Unstride the model

      host_tensor_t v = rearrange_r(_visible_biases, _stride_size);
      _visible_biases = v;

      dim_t layerSize = v.size();
      layerSize[dimCount - 1] = 1;

      _mask = ones<value_t>(layerSize);

      host_tensor_t f;
      for (size_t i = 0; i < _filters.size(); ++i) {
        f = rearrange_r(*_filters[i], _stride_size);
        *_filters[i] = f;
        *_hidden_biases[i] = ones<value_t>(layerSize) * (*_hidden_biases[i])[seq<dimCount>(0)];
      }
      _kernel_size = _filters[0]->size();
    } else {
      throw std::runtime_error("Either the old or the new stride must be 1.");
    }

    set_stride_size(stride);
  }

  void change_size(dim_t size) {
    dim_t layerSize = size;
    layerSize[dimCount - 1] = 1;

    _mask = ones<value_t>(layerSize);
    _visible_biases = ones<value_t>(size) * _visible_biases[seq<dimCount>(0)];

    for (size_t i = 0; i < _filters.size(); ++i) {
      *_hidden_biases[i] = ones<value_t>(layerSize) * (*_hidden_biases[i])[seq<dimCount>(0)];
    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_MODEL_HPP_ */
