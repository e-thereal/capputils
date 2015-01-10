/*
 * dbn.hpp
 *
 *  Created on: Jul 10, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DBN_HPP_
#define TBBLAS_DEEPLEARN_DBN_HPP_

// TODO: rename to conv_dbn and create separated dbn class
// TODO: momentum_step and adadelta_step replace init_gradient and apply_gradient

#include <tbblas/rearrange.hpp>

#include <tbblas/deeplearn/dbn_model.hpp>
#include <tbblas/deeplearn/rbm.hpp>

#include <thrust/execution_policy.h>

#include <boost/shared_ptr.hpp>
#include <boost/ref.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T>
class dbn {
  typedef T value_t;

  typedef tbblas::tensor<value_t, 2> host_matrix_t;

  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  typedef dbn_model<value_t> model_t;
  typedef rbm<value_t> rbm_t;

protected:
  model_t& _model;

  std::vector<boost::shared_ptr<rbm_t> > _rbms;

public:
  dbn(model_t& model) : _model(model) {
    _rbms.resize(model.rbms().size());
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i] = boost::make_shared<rbm_t>(boost::ref(*model.rbms()[i]));
  }

private:
  dbn(const dbn<T>&);

public:
  virtual ~dbn() { }

  void allocate_gpu_memory() {
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->allocate_gpu_memory();
  }

  void normalize_visibles() {
    if (_rbms.size() < 1)
      throw std::runtime_error("A DBN requires at least one RBM layer.");

  _rbms[0]->normalize_visibles();
  }

  void diversify_visibles() {
    if (_rbms.size() < 1)
      throw std::runtime_error("A DBN requires at least one RBM layer.");

    _rbms[0]->diversify_visibles();
  }

  //
  void infer_visibles(int topLayer = -1, bool onlyFilters = false) {
    if (topLayer < 0 || topLayer >= _rbms.size())
      topLayer = _rbms.size() - 1;

    // top-down inference
    for (int i = topLayer; i >= 0; --i) {
      _rbms[i]->infer_visibles(onlyFilters);
      if (i > 0) {
        _rbms[i - 1]->hiddens() = _rbms[i]->visibles();
      }
    }
  }

  // -1 indicates the top-most layer
  void infer_hiddens(int maxLayer = -1) {
    if (maxLayer < 0 || maxLayer >= _rbms.size())
      maxLayer = _rbms.size();

    for (size_t i = 0; i <= maxLayer; ++i) {
      _rbms[i]->infer_hiddens();
      if (i + 1 < _rbms.size()) {
        _rbms[i + 1]->visibles() = _rbms[i]->hiddens();
      }
    }
  }

  void sample_visibles(int topLayer = -1) {
    if (topLayer < 0 || topLayer >= _rbms.size())
      topLayer = _rbms.size() - 1;

    // top-down inference
    for (int i = topLayer; i >= 0; --i) {
      _rbms[i]->sample_visibles();
      if (i > 0) {
        _rbms[i - 1]->hiddens() = _rbms[i]->visibles();
      }
    }
  }

  // -1 indicates the top-most layer
  void sample_hiddens(int maxLayer = -1) {
    if (maxLayer < 0 || maxLayer >= _rbms.size())
      maxLayer = _rbms.size();

    for (size_t i = 0; i <= maxLayer; ++i) {
      _rbms[i]->sample_hiddens();
      if (i + 1 < _rbms.size()) {
        _rbms[i + 1]->visibles() = _rbms[i]->hiddens();
      }
    }
  }

  void init_gradient_updates(value_t momentum = 0, value_t weightcost = 0) {
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->init_gradient_updates(momentum, weightcost);
  }

  void update_positive_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->update_positive_gradient(epsilonw, epsilonvb, epsilonhb);
  }

  void update_negative_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->update_negative_gradient(epsilonw, epsilonvb, epsilonhb);
  }

  void apply_gradient() {
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->apply_gradient();
  }

  void set_batch_length(int layer, int length) {
    if (layer < _crbms.size())
      _crbms[layer]->set_batch_length(length);
  }

  matrix_t& visibles(int layer = 0) {
    if (layer < 0 || layer >= _rbms.size())
      throw std::runtime_error("The given layer is not part of the DBN!");

    return _rbms[layer]->visibles();
  }

  // Default layer is the last hidden layer
  matrix_t& hiddens(int layer = -1) {
    if (layer == -1)
      layer = _rbms.size() - 1;
    if (layer < 0 || layer >= _rbms.size())
      throw std::runtime_error("The given layer is not part of the DBN!");

    return _rbms[layer]->hiddens();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_DBN_HPP_ */
