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

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

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
   */
  static const unsigned CURRENT_VERSION = 1;

protected:
  // Model in CPU memory
  host_tensor_t _visibleBiases, _mask;
  v_host_tensor_t _filters, _hiddenBiases;
  dim_t _filterKernelSize, _stride_size;

  unit_type _visibleUnitType, _hiddenUnitType;
  tbblas::deeplearn::convolution_type _convolutionType;

  value_t _mean, _stddev;
  bool _shared_biases;
  unsigned _version;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm_model() : _stride_size(seq<dimCount>(1)), _mean(0), _stddev(1), _shared_biases(false), _version(CURRENT_VERSION)
  {
  }

  conv_rbm_model(const conv_rbm_model<T,dims>& model)
   : _visibleBiases(model.visible_bias()), _mask(model.mask()), _filterKernelSize(model.kernel_size()), _stride_size(model.stride_size()),
     _visibleUnitType(model.visibles_type()), _hiddenUnitType(model.hiddens_type()), _convolutionType(model.convolution_type()), _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias()), _version(model.version())
  {
    set_filters(model.filters());
    set_hidden_bias(model.hidden_bias());
  }

  template<class U>
  conv_rbm_model(const conv_rbm_model<U,dims>& model)
   : _visibleBiases(model.visible_bias()), _mask(model.mask()), _filterKernelSize(model.kernel_size()), _stride_size(model.stride_size()),
     _visibleUnitType(model.visibles_type()), _hiddenUnitType(model.hiddens_type()), _convolutionType(model.convolution_type()), _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias()), _version(model.version())
  {
    _filters.resize(model.filters().size());
    for (size_t i = 0; i < model.filters().size(); ++i)
      _filters[i] = boost::make_shared<host_tensor_t>(*model.filters()[i]);

    _hiddenBiases.resize(model.hidden_bias().size());
    for (size_t i = 0; i < model.hidden_bias().size(); ++i)
      _hiddenBiases[i] = boost::make_shared<host_tensor_t>(*model.hidden_bias()[i]);
  }

  virtual ~conv_rbm_model() { }

public:
  void set_filters(const v_host_tensor_t& filters) {
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

  void set_visible_bias(host_tensor_t& bias) {
    _visibleBiases = bias;
  }

  const host_tensor_t& visible_bias() const {
    return _visibleBiases;
  }

  void set_hidden_bias(const v_host_tensor_t& bias) {
    _hiddenBiases.resize(bias.size());
    for (size_t i = 0; i < bias.size(); ++i)
      _hiddenBiases[i] = boost::make_shared<host_tensor_t>(*bias[i]);
  }

  const v_host_tensor_t& hidden_bias() const {
    return _hiddenBiases;
  }

  void set_mask(host_tensor_t& mask) {
    _mask = mask;
  }

  const host_tensor_t& mask() const {
    return _mask;
  }

  void set_kernel_size(const dim_t& size) {
    _filterKernelSize = size;
  }

  const dim_t& kernel_size() const {
    return _filterKernelSize;
  }

  void set_stride_size(const dim_t& size) {
    _stride_size = size;
  }

  const dim_t& stride_size() const {
    return _stride_size;
  }

  void set_visibles_type(const unit_type& type) {
    _visibleUnitType = type;
  }

  const unit_type& visibles_type() const {
    return _visibleUnitType;
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
    _hiddenUnitType = type;
  }

  const unit_type& hiddens_type() const {
    return _hiddenUnitType;
  }

  dim_t hiddens_size() const {
    dim_t hidden_topleft = seq<dimCount>(0);
    if (convolution_type() == convolution_type::Valid){
      hidden_topleft = kernel_size() / 2;
      hidden_topleft[dimCount - 1] = 0;
    }

    dim_t hidden_size = visible_bias().size() - 2 * hidden_topleft;
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

  void set_convolution_type(const tbblas::deeplearn::convolution_type& type) {
    _convolutionType = type;
  }

  const tbblas::deeplearn::convolution_type& convolution_type() const {
    return _convolutionType;
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
      _visibleBiases = ones<value_t>(_visibleBiases.size()) * sum(_visibleBiases) / _visibleBiases.count();
      for (size_t i = 0; i < _hiddenBiases.size(); ++i) {
        *_hiddenBiases[i] = ones<value_t>(_hiddenBiases[i]->size()) * sum(*_hiddenBiases[i]) / _hiddenBiases[i]->count();
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
};

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_MODEL_HPP_ */
