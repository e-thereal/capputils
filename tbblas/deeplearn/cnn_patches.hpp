/*
 * cnn_patches.hpp
 *
 *  Created on: 2014-12-01
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CNN_PATCHES_HPP_
#define TBBLAS_DEEPLEARN_CNN_PATCHES_HPP_

#include <tbblas/deeplearn/cnn_model.hpp>
#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>

#include <tbblas/context.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/reshape.hpp>

#include <tbblas/sequence_iterator.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class cnn_patches {
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;
  typedef std::vector<dim_t> v_dim_t;

  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef nn_layer<value_t> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

  typedef cnn_layer<value_t, dimCount> cnn_layer_t;
  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;

  typedef cnn_model<value_t, dimCount> model_t;

protected:
  model_t& _model;
  v_cnn_layer_t _cnn_layers;
  v_nn_layer_t _nn_layers;
  v_dim_t _patch_sizes, _patch_counts;
  v_tensor_t _target_tensors, _deltas, _hiddens;
  matrix_t _target_matrix;

public:
  cnn_patches(model_t& model, const dim_t& patch_count) : _model(model) {
    if (model.cnn_layers().size() == 0 || model.nn_layers().size() == 0)
      throw std::runtime_error("At least one convolutional and one dense layer required to build a convolutional neural network.");

    // Change the model size and safe old size
    // Check that the patch count is valid for all layers

    _cnn_layers.resize(model.cnn_layers().size());
    _patch_sizes.resize(_cnn_layers.size());
    _patch_counts.resize(_cnn_layers.size() + 1);
    _target_tensors.resize(_cnn_layers.size());
    _deltas.resize(_cnn_layers.size());
    _hiddens.resize(_cnn_layers.size());

    for (size_t i = 0; i < _target_tensors.size(); ++i) {
      _target_tensors[i] = boost::make_shared<tensor_t>();
      _deltas[i] = boost::make_shared<tensor_t>();
      _hiddens[i] = boost::make_shared<tensor_t>();
    }

    _patch_counts[0] = patch_count;
    _patch_counts[0][dimCount - 1] = 1;

    for (size_t i = 0; i < _cnn_layers.size(); ++i) {
      if (model.cnn_layers()[i]->stride_size().prod() != 1)
        throw std::runtime_error("Strides are not supported for CNN patches.");

      for (size_t j = 0; j < dimCount; ++j) {
        if (((_patch_counts[i][j] % model.cnn_layers()[i]->pooling_size()[j]) != 0) &&
            _patch_counts[i][j] > 1)
        {
          throw std::runtime_error("Invalid number of patches selected.");
        }
      }

      _patch_sizes[i] = model.cnn_layers()[i]->input_size();
      model.cnn_layers()[i]->change_size(_patch_sizes[i] + _patch_counts[i] - 1);

      _cnn_layers[i] = boost::make_shared<cnn_layer_t>(boost::ref(*model.cnn_layers()[i]));
      _patch_counts[i + 1] = max(seq<dimCount>(1), _patch_counts[i] / model.cnn_layers()[i]->pooling_size());
    }

    _nn_layers.resize(model.nn_layers().size());
    for (size_t i = 0; i < _nn_layers.size(); ++i) {
      _nn_layers[i] = boost::make_shared<nn_layer_t>(boost::ref(*model.nn_layers()[i]));
    }
  }

  virtual ~cnn_patches() {
    for (size_t i = 0; i < _cnn_layers.size(); ++i) {
      _model.cnn_layers()[i]->change_size(_patch_sizes[i]);
    }
  }

private:
  cnn_patches(const cnn_patches<T, dims>&);

public:
  void normalize_visibles() {
    _cnn_layers[0]->normalize_visibles();
  }

  void infer_hiddens() {
    infer_hiddens(0);
  }

  void update_gradient(tensor_t& target) {
    dim_t targetSize = _patch_counts[0];
    targetSize[dimCount - 1] = target.size()[dimCount - 1];

    if (targetSize != target.size()) {
      throw std::runtime_error("Target size doesn't match the expected patch count.");
    }

    *_target_tensors[0] = target;
    update_gradient(0);
  }

  void momentum_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _cnn_layers.size(); ++i)
      _cnn_layers[i]->momentum_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _nn_layers.size(); ++i)
      _nn_layers[i]->momentum_step(epsilon2, momentum, weightcost);
  }

  void adadelta_step(value_t epsilon1, value_t epsilon2, value_t momentum, value_t weightcost) {
    for (size_t i = 0; i < _cnn_layers.size(); ++i)
      _cnn_layers[i]->adadelta_step(epsilon1, momentum, weightcost);

    for (size_t i = 0; i < _nn_layers.size(); ++i)
      _nn_layers[i]->adadelta_step(epsilon2, momentum, weightcost);
  }

  void set_batch_length(int layer, int length) {
    if (layer < _cnn_layers.size())
      _cnn_layers[layer]->set_batch_length(length);
  }

  void set_input(tensor_t& input) {
    assert(_model.cnn_layers()[0]->input_size() == input.size());
    _cnn_layers[0]->visibles() = rearrange(input, _model.cnn_layers()[0]->stride_size());
  }

  tensor_t& hiddens() {
    return *_hiddens[0];
  }

protected:
  // TODO: infer_hiddens function takes tensor as input and writes the hidden values to that tensor
  // Writing occurs recursively in the back-recursion.

  void infer_hiddens(int iLayer) {
    if (iLayer < _cnn_layers.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_layers[i]->infer_hiddens();
      if (iLayer + 1 < _cnn_layers.size()) {
        // iterate over pooled slices and run inference again
        dim_t blockCount = _model.cnn_layers()[i]->pooling_size();
        for (size_t iDim = 0; iDim < dimCount; ++iDim) {
          if (_patch_counts[i][iDim] == 1)
            blockCount[iDim] = 1;
        }

        for (sequence_iterator<dim_t> iBlock(seq<dimCount>(0), blockCount); iBlock; ++iBlock) {
          _cnn_layers[i + 1]->visibles() =
              _cnn_layers[i]->hiddens()[*iBlock, _model.cnn_layers()[i]->pooling_size(), _cnn_layers[i]->hiddens().size() - *iBlock];

          infer_hiddens(iLayer + 1);
        }
      } else {
        dim_t blockCount = _patch_counts[i];
        dim_t hiddenSize = _model.cnn_layers()[i]->hiddens_size() - _patch_sizes[i] + 1;
        dim_t blockSize = hiddenSize / _model.cnn_layers()[i]->pooling_size();
        _nn_layers[0]->visibles().resize(seq(blockCount.prod(), blockSize.prod()));

        sequence_iterator<dim_t> iBlock(seq<dimCount>(0), blockCount);
        for (int i = 0; iBlock; ++iBlock, ++i) {

          // Transition from convolutional model to dense model
          row(_nn_layers[0]->visibles(), i) = reshape(_cnn_layers[i]->hiddens()[*iBlock, _model.cnn_layers()[i]->pooling_size(), hiddenSize], 1, blockSize.prod());
        }
        infer_hiddens(iLayer + 1);
      }
    } else if (iLayer - _cnn_layers.size() < _nn_layers.size()) {

      // Infer dense layer
      const int i = iLayer - _cnn_layers.size();

      _nn_layers[i]->infer_hiddens();
      if (i + 1 < _nn_layers.size()) {
        _nn_layers[i + 1]->visibles() = _nn_layers[i]->hiddens();
        infer_hiddens(iLayer + 1);
      }

    } else {
      // should never happen
      assert(0);
    }
  }

  void update_gradient(int iLayer) {

    if (iLayer < _cnn_layers.size()) {

      // Infer convolutional layer
      const int i = iLayer;

      _cnn_layers[i]->infer_hiddens();
      if (iLayer + 1 < _cnn_layers.size()) {

        // iterate over pooled slices, run inference and collect deltas
        dim_t blockCount = _model.cnn_layers()[i]->pooling_size();
        for (size_t iDim = 0; iDim < dimCount; ++iDim) {
          if (_patch_counts[i][iDim] == 1)
            blockCount[iDim] = 1;
        }

        _deltas[i]->resize(_cnn_layers[i]->hiddens().size());
        _hiddens[i]->resize(_target_tensors[i]->size());
        for (sequence_iterator<dim_t> iBlock(seq<dimCount>(0), blockCount); iBlock; ++iBlock) {

          // Wrap target and visibles
          *_target_tensors[i + 1] = (*_target_tensors[i])[*iBlock, _model.cnn_layers()[i]->pooling_size(), _target_tensors[i]->size() - *iBlock];
          _cnn_layers[i + 1]->visibles() =
              _cnn_layers[i]->hiddens()[*iBlock, _model.cnn_layers()[i]->pooling_size(), _cnn_layers[i]->hiddens().size() - blockCount + 1];

          update_gradient(iLayer + 1);
          _cnn_layers[i + 1]->backprop_visible_deltas();
          (*_deltas[i])[*iBlock, _model.cnn_layers()[i]->pooling_size(), _cnn_layers[i]->hiddens().size() - blockCount + 1] = _cnn_layers[i + 1]->visible_deltas();

          (*_hiddens[i])[*iBlock, _model.cnn_layers()[i]->pooling_size(), _target_tensors[i]->size() - *iBlock] = *_hiddens[i + 1];
        }
        _cnn_layers[i]->backprop_hidden_deltas(*_deltas[i]);
        _cnn_layers[i]->update_gradient();
      } else {

        dim_t blockCount = _patch_counts[i];
        dim_t hiddenSize = _model.cnn_layers()[i]->hiddens_size() - _patch_counts[i] + 1;
        dim_t blockSize = hiddenSize / _model.cnn_layers()[i]->pooling_size();

        _nn_layers[0]->visibles().resize(seq(blockCount.prod(), blockSize.prod()));
        _target_matrix = reshape(*_target_tensors[i], blockCount.prod(), _target_tensors[i]->size()[dimCount - 1]);

        // Transition from convolutional model to dense model
        sequence_iterator<dim_t> iBlock(seq<dimCount>(0), blockCount);
        for (int iRow = 0; iBlock; ++iBlock, ++iRow) {
          row(_nn_layers[0]->visibles(), iRow) = reshape(_cnn_layers[i]->hiddens()[*iBlock, _model.cnn_layers()[i]->pooling_size(), hiddenSize], 1, blockSize.prod());
        }

        update_gradient(iLayer + 1);
        *_hiddens[i] = reshape(_nn_layers[_nn_layers.size() - 1]->hiddens(), _target_tensors[i]->size());

        // Transition back
        _nn_layers[0]->backprop_visible_deltas();
        _deltas[i]->resize(_cnn_layers[i]->hiddens().size());
        iBlock = sequence_iterator<dim_t>(seq<dimCount>(0), blockCount);
        for (int iRow = 0; iBlock; ++iBlock, ++iRow) {
          (*_deltas[i])[*iBlock, _model.cnn_layers()[i]->pooling_size(), hiddenSize] = reshape(row(_nn_layers[0]->visible_deltas(), iRow), blockSize);
        }

        _cnn_layers[i]->backprop_hidden_deltas(*_deltas[i]);
        _cnn_layers[i]->update_gradient();
      }
    } else if (iLayer - _cnn_layers.size() < _nn_layers.size()) {
      // Infer dense layer
      const int i = iLayer - _cnn_layers.size();

      _nn_layers[i]->infer_hiddens();
      if (i + 1 < _nn_layers.size()) {
        _nn_layers[i + 1]->visibles() = _nn_layers[i]->hiddens();
        update_gradient(iLayer + 1);

        _nn_layers[i + 1]->backprop_visible_deltas();
        _nn_layers[i]->backprop_hidden_deltas(_nn_layers[i + 1]->visible_deltas());
        _nn_layers[i]->update_gradient();
      } else {
        _nn_layers[i]->calculate_deltas(_target_matrix);
        _nn_layers[i]->update_gradient();
      }
    } else {
      assert(0); // should never happen
    }
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_PATCHES_HPP_ */
