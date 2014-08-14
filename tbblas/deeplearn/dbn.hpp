/*
 * dbn.hpp
 *
 *  Created on: Jul 10, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DBN_HPP_
#define TBBLAS_DEEPLEARN_DBN_HPP_

#include <tbblas/rearrange.hpp>

#include <tbblas/deeplearn/dbn_model.hpp>
#include <tbblas/deeplearn/conv_rbm.hpp>
#include <tbblas/deeplearn/rbm.hpp>

#include <thrust/execution_policy.h>

#include <boost/shared_ptr.hpp>
#include <boost/ref.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class dbn {
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef dbn_model<value_t, dimCount> model_t;
  typedef conv_rbm<value_t, dimCount> crbm_t;
  typedef rbm<value_t> rbm_t;

protected:
  model_t& _model;

  std::vector<boost::shared_ptr<crbm_t> > _crbms;
  std::vector<boost::shared_ptr<rbm_t> > _rbms;

public:
  dbn(model_t& model, size_t gpu_count = 1) : _model(model) {
    _crbms.resize(model.crbms().size());
    for (size_t i = 0; i < _crbms.size(); ++i) {
      _crbms[i] = boost::make_shared<crbm_t>(boost::ref(*model.crbms()[i]));
    }

    _rbms.resize(model.rbms().size());
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i] = boost::make_shared<rbm_t>(boost::ref(*model.rbms()[i]));
  }

private:
  dbn(const dbn<T, dims>&);

public:
  virtual ~dbn() { }

  void allocate_gpu_memory() {
    for (size_t i = 0; i < _crbms.size(); ++i)
      _crbms[i]->allocate_gpu_memory();
    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->allocate_gpu_memory();
  }

  void normalize_visibles() {
    if (_crbms.size()) {
      _crbms[0]->normalize_visibles();
    } else if (_rbms.size()) {
      _rbms[0]->normalize_visibles();
    }
  }

  void diversify_visibles() {
    if (_crbms.size()) {
      _crbms[0]->diversify_visibles();
    } else if (_rbms.size()) {
      _rbms[0]->diversify_visibles();
    }
  }

  void infer_visibles(int topLayer = -1) {
    if (topLayer == -1)
      topLayer = _crbms.size() + _rbms.size();
    else
      topLayer = std::min(topLayer, (int)(_crbms.size() + _rbms.size()));

    // top-down inference
    for (int i = topLayer - _crbms.size() - 1; i >= 0; --i) {
      _rbms[i]->infer_visibles();
      if (i > 0) {
        _rbms[i - 1]->hiddens() = _rbms[i]->visibles();
      }
    }

    // Transition from convolutional model to dense model
    if (_crbms.size() && _rbms.size() && topLayer > _crbms.size()) {
      _crbms[_crbms.size() - 1]->allocate_hiddens();
      thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream),
          _rbms[0]->visibles().begin(), _rbms[0]->visibles().end(), _crbms[_crbms.size() - 1]->hiddens().begin());
    }

    for (int i = std::min(topLayer, (int)_crbms.size()) - 1; i >= 0; --i) {
      _crbms[i]->infer_visibles();
      if (i > 0) {
        _crbms[i - 1]->allocate_hiddens();

        dim_t block = _crbms[i - 1]->hiddens().size() / _crbms[i]->visibles().size();
        block[dimCount - 1] = 1;
        _crbms[i - 1]->hiddens() = rearrange_r(_crbms[i]->visibles(), block);
      }
    }
  }

  // -1 indicates the top-most layer
  void infer_hiddens(int maxLayer = -1) {

    int currentLayer = 0;

    if (maxLayer == -1)
      maxLayer = _crbms.size() + _rbms.size();
    else
      maxLayer = std::min(maxLayer, (int)(_crbms.size() + _rbms.size()));

    // bottom-up inference
    for (size_t i = 0; i < _crbms.size() && currentLayer < maxLayer; ++i, ++currentLayer) {
      _crbms[i]->infer_hiddens();
      if (i + 1 < _crbms.size()) {
        dim_t block = _crbms[i]->hiddens().size() / _model.crbms()[i + 1]->visible_bias().size();
        block[dimCount - 1] = 1;

        _crbms[i + 1]->visibles() = rearrange(_crbms[i]->hiddens(), block);
      }
    }

    // Transition from convolutional model to dense model
    if (_crbms.size() && _rbms.size() && maxLayer > _crbms.size()) {
      _rbms[0]->visibles().resize(seq(1, (int)_crbms[_crbms.size() - 1]->hiddens().count()));

      thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream), _crbms[_crbms.size() - 1]->hiddens().begin(),
          _crbms[_crbms.size() - 1]->hiddens().end(), _rbms[0]->visibles().begin());
    }

    for (size_t i = 0; i < _rbms.size() && currentLayer < maxLayer; ++i, ++currentLayer) {
      _rbms[i]->infer_hiddens();
      if (i + 1 < _rbms.size()) {
        _rbms[i + 1]->visibles() = _rbms[i]->hiddens();
      }
    }
  }

  void sample_visibles(int topLayer = -1) {
    if (topLayer == -1)
      topLayer = _crbms.size() + _rbms.size();
    else
      topLayer = std::min(topLayer, (int)(_crbms.size() + _rbms.size()));

    // top-down inference
    for (int i = topLayer - _crbms.size() - 1; i >= 0; --i) {
      _rbms[i]->sample_visibles();
      if (i > 0) {
        _rbms[i - 1]->hiddens() = _rbms[i]->visibles();
      }
    }

    // Transition from convolutional model to dense model
    if (_crbms.size() && _rbms.size() && topLayer > _crbms.size()) {
      _crbms[_crbms.size() - 1]->allocate_hiddens();
      thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream),
          _rbms[0]->visibles().begin(), _rbms[0]->visibles().end(), _crbms[_crbms.size() - 1]->hiddens().begin());
    }

    for (int i = std::min(topLayer, (int)_crbms.size()) - 1; i >= 0; --i) {
      _crbms[i]->sample_visibles();
      if (i > 0) {
        _crbms[i - 1]->allocate_hiddens();

        dim_t block = _crbms[i - 1]->hiddens().size() / _crbms[i]->visibles().size();
        block[dimCount - 1] = 1;
        _crbms[i - 1]->hiddens() = rearrange_r(_crbms[i]->visibles(), block);
      }
    }
  }

  // -1 indicates the top-most layer
  void sample_hiddens(int maxLayer = -1) {

    int currentLayer = 0;

    if (maxLayer == -1)
      maxLayer = _crbms.size() + _rbms.size();
    else
      maxLayer = std::min(maxLayer, (int)(_crbms.size() + _rbms.size()));

    // bottom-up inference
    for (size_t i = 0; i < _crbms.size() && currentLayer < maxLayer; ++i, ++currentLayer) {
      _crbms[i]->sample_hiddens();
      if (i + 1 < _crbms.size()) {
        dim_t block = _crbms[i]->hiddens().size() / _model.crbms()[i + 1]->visible_bias().size();
        block[dimCount - 1] = 1;

        _crbms[i + 1]->visibles() = rearrange(_crbms[i]->hiddens(), block);
      }
    }

    // Transition from convolutional model to dense model
    if (_crbms.size() && _rbms.size() && maxLayer > _crbms.size()) {
      _rbms[0]->visibles().resize(seq(1, (int)_crbms[_crbms.size() - 1]->hiddens().count()));

      thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream), _crbms[_crbms.size() - 1]->hiddens().begin(),
          _crbms[_crbms.size() - 1]->hiddens().end(), _rbms[0]->visibles().begin());
    }

    for (size_t i = 0; i < _rbms.size() && currentLayer < maxLayer; ++i, ++currentLayer) {
      _rbms[i]->sample_hiddens();
      if (i + 1 < _rbms.size()) {
        _rbms[i + 1]->visibles() = _rbms[i]->hiddens();
      }
    }
  }

  void init_gradient_updates(value_t momentum = 0, value_t weightcost = 0) {
    for (size_t i = 0; i < _crbms.size(); ++i)
      _crbms[i]->init_gradient_updates(momentum, weightcost);

    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->init_gradient_updates(momentum, weightcost);
  }

  void update_positive_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    for (size_t i = 0; i < _crbms.size(); ++i)
      _crbms[i]->update_positive_gradient(epsilonw, epsilonvb, epsilonhb);

    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->update_positive_gradient(epsilonw, epsilonvb, epsilonhb);
  }

  void update_negative_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    for (size_t i = 0; i < _crbms.size(); ++i)
      _crbms[i]->update_negative_gradient(epsilonw, epsilonvb, epsilonhb);

    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->update_negative_gradient(epsilonw, epsilonvb, epsilonhb);
  }

  // CAUTION: ONLY USE THIS FUNCTION IF YOU KNOW WHAT YOU ARE DOING
  // Should probably check, if the model is the same and needs a don't
  // write to model mechanism
  void accumulate_gradients(dbn<value_t, dimCount>& dbn) {
    for (size_t i = 0; i < _crbms.size(); ++i)
      _crbms[i]->accumulate_gradients(*dbn._crbms[i]);

    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->accumulate_gradients(*dbn._rbms[i]);
  }

  void apply_gradient() {
    for (size_t i = 0; i < _crbms.size(); ++i)
      _crbms[i]->apply_gradient();

    for (size_t i = 0; i < _rbms.size(); ++i)
      _rbms[i]->apply_gradient();
  }

  void set_batch_length(int layer, int length) {
    if (layer < _crbms.size())
      _crbms[layer]->set_batch_length(length);
  }

  // Access to model data
  tensor_t& cvisibles(int layer = 0) {
    if (layer >= 0 && layer < _crbms.size())
      return _crbms[layer]->visibles();

    throw std::runtime_error("The given layer is not part of the DBN!");
  }

  tensor_t& chiddens(int layer) {
    if (layer >= 0 && layer < _crbms.size())
      return _crbms[layer]->visibles();

    throw std::runtime_error("The given layer is not part of the DBN!");
  }

  matrix_t& visibles(int layer) {
    if (layer >= 0 && layer < _rbms.size())
      return _rbms[layer]->visibles();

    throw std::runtime_error("The given layer is not part of the DBN!");
  }

  // Default layer is the last hidden layer
  matrix_t& hiddens(int layer = -1) {
    if (layer == -1)
      layer = _rbms.size() - 1;
    if (layer >= 0 && layer < _rbms.size())
      return _rbms[layer]->visibles();

    throw std::runtime_error("The given layer is not part of the DBN!");
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_DBN_HPP_ */
