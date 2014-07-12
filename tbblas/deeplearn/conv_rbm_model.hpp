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

protected:
  // Model in CPU memory
  host_tensor_t _visibleBiases, _mask;
  v_host_tensor_t _filters, _hiddenBiases;
  dim_t _filterKernelSize;

  unit_type _visibleUnitType, _hiddenUnitType;
  tbblas::deeplearn::convolution_type _convolutionType;

  value_t _mean, _stddev;
  bool _shared_biases;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm_model() : _mean(0), _stddev(1), _shared_biases(false)
  {
  }

  conv_rbm_model(const conv_rbm_model<T,dims>& model)
   : _visibleBiases(model.visible_bias()), _mask(model.mask()), _filterKernelSize(model.kernel_size()),
     _visibleUnitType(model.visibles_type()), _hiddenUnitType(model.hiddens_type()), _convolutionType(model.convolution_type()), _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias())
  {
    set_filters(model.filters());
    set_hidden_bias(model.hidden_bias());
  }

  template<class U>
  conv_rbm_model(const conv_rbm_model<U,dims>& model)
   : _visibleBiases(model.visible_bias()), _mask(model.mask()), _filterKernelSize(model.kernel_size()),
     _visibleUnitType(model.visibles_type()), _hiddenUnitType(model.hiddens_type()), _convolutionType(model.convolution_type()), _mean(model.mean()), _stddev(model.stddev()),
     _shared_biases(model.shared_bias())
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

  void set_visible_bias(const host_tensor_t& bias) {
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

  void set_mask(const host_tensor_t& mask) {
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
};

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_MODEL_HPP_ */
