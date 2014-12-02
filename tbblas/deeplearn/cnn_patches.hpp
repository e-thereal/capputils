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
  v_tensor_t _target_tensors;
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

    for (size_t i = 0; i < _target_tensors.size(); ++i)
      _target_tensors[i] = boost::make_shared<tensor_t>();

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

      tbblas_print(i);
      tbblas_print(_patch_counts[i]);

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
//    infer_hiddens(0);
    // blockCount = product of pooling_sizes = image counts in the last layer
    // Iterate over all images of the last layer and calculate them
    // For each image, iterator over all patches and write them to the hidden unit matrix

//    for (dim_t iBlock = seq<dimCount>(0); iBlock < blockCount; inc(iBlock, blockCount)) {
//      // calculate the block index for each layer. If the block index has changed from the previous calculation, e.g. the remainder is 1,
//      // then recalculate the layer
//    }

    // Dense hidden units will be processed in one batch. Allocate memory for them

//    _nn_layers[0]->visibles().resize(seq(
//        (int)_super_patch_counts[dimCount].prod() * _patch_counts[dimCount],
//        (int)_cnn_layers[_cnn_layers.size() - 1]->filter_count()));
//
//    for (dim_t iSuperPatch = seq<dimCount(0); iSuperPatch < superPatchCount; inc(iSuperPatch, superPatchCount)) {
//      for (size_t i = 0; i < _cnn_layers.size(); ++i) {
//
//        _cnn_layers[i]->infer_hiddens();
//
//        // TODO: get
//        tensor_t slices = _cnn_layers[i]->hiddens()[seq(0,0,0,0), model.pooling_size(), _cnn_layers[i]->hiddens().size() - seq(0,0,0,0)];
//
//        if (i + 1 < _cnn_layers.size()) {
//          _cnn_layers[i + 1]->visibles() = slice;
//        } else if (i + 1 == _cnn_layers.size()) {
//
//
//          // Transition from convolutional model to dense model
////          thrust::copy(thrust::cuda::par.on(tbblas::context::get().stream), _cnn_layers[_cnn_layers.size() - 1]->hiddens().begin(),
////              _cnn_layers[_cnn_layers.size() - 1]->hiddens().end(), _nn_layers[0]->visibles().begin());
//
//          dim_t patch_count = slice.size(), column_size = seq<dimCount>(1);
//          patch_count[dimCount - 1] = 1;
//          column_size[dimCount - 1] = slice.size()[dimCount - 1];
//
//          for (dim_t iPatch = seq<dimCount>(0); iPatch < patch_count; inc(iPatch, patch_count)) {
//            // TODO: calculate the 1D patch index
//            const int iRow = 0;
//
//            // Reshape one column of the 4D super patch to a row of the 2D visible units matrix
//            row(_nn_layers[0]->visibles(), iRow) = reshape(slice[iPatch, column_size], 1, _nn_layers[0]->visibles().size()[1]);
//          }
//        }
//      }
//    }
//
//
//    for (size_t i = 0; i < _nn_layers.size(); ++i) {
//      _nn_layers[i]->infer_hiddens();
//      if (i + 1 < _nn_layers.size()) {
//        _nn_layers[i + 1]->visibles() = _nn_layers[i]->hiddens();
//      }
//    }
  }

//  // requires the hidden units to be inferred
//  void update_gradient(matrix_t& target) {
//    _nn_layers[_nn_layers.size() - 1]->calculate_deltas(target);
//    _nn_layers[_nn_layers.size() - 1]->update_gradient();
//
//    // Perform back propagation
//    for (int i = _nn_layers.size() - 2; i >= 0; --i) {
//      _nn_layers[i + 1]->backprop_visible_deltas();
//      _nn_layers[i]->backprop_hidden_deltas(_nn_layers[i + 1]->visible_deltas());
//      _nn_layers[i]->update_gradient();
//    }
//
//    const size_t clast = _cnn_layers.size() - 1;
//    _nn_layers[0]->backprop_visible_deltas();
//    _cnn_layers[clast]->backprop_hidden_deltas(reshape(
//        _nn_layers[0]->visible_deltas(),
//        _model.cnn_layers()[clast]->hiddens_size()));
//    _cnn_layers[clast]->update_gradient();
//
//    for (int i = _cnn_layers.size() - 2; i >= 0; --i) {
//      _cnn_layers[i + 1]->backprop_visible_deltas();
//      _cnn_layers[i]->backprop_hidden_deltas(rearrange_r(
//          _cnn_layers[i + 1]->visible_deltas(),
//          _model.cnn_layers()[i + 1]->stride_size()));
//      _cnn_layers[i]->update_gradient();
//    }
//  }

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

  matrix_t& hiddens() {
    return _nn_layers[_nn_layers.size() - 1]->hiddens();
  }

public:
  // TODO: infer_hiddens function takes tensor as input and writes the hidden values to that tensor
  // Writing occurs recursively and the back-recursion.
  void infer_hiddens2() {
    infer_hiddens(0);
  }

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

        for (dim_t iBlock = seq<dimCount>(0); iBlock < blockCount; inc(iBlock, blockCount)) {
          _cnn_layers[i + 1]->visibles() =
              _cnn_layers[i]->hiddens()[iBlock, _model.cnn_layers()[i]->pooling_size(), _cnn_layers[i]->hiddens().size() - iBlock];

          infer_hiddens(iLayer + 1);
        }
      } else {
        dim_t iBlock = seq<dimCount>(0);
        dim_t blockCount = _patch_counts[i];
        dim_t hiddenSize = _model.cnn_layers()[i]->hiddens_size() - _patch_sizes[i] + 1;
        dim_t blockSize = hiddenSize / _model.cnn_layers()[i]->pooling_size();
        _nn_layers[0]->visibles().resize(seq(blockCount.prod(), blockSize.prod()));
        for (int i = 0; iBlock < blockCount; inc(iBlock, blockCount), ++i) {

          // Transition from convolutional model to dense model
          row(_nn_layers[0]->visibles(), i) = reshape(_cnn_layers[i]->hiddens()[iBlock, _model.cnn_layers()[i]->pooling_size(), hiddenSize], 1, blockSize.prod());
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

  void update_gradient2(tensor_t& target) {
    dim_t targetSize = _patch_counts[0];
    targetSize[dimCount - 1] = target.size()[dimCount - 1];

    if (targetSize != target.size()) {
      throw std::runtime_error("Target size doesn't match the expected patch count.");
    }

    *_target_tensors[0] = target;
    update_gradient(0, target);
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

        tensor_t deltas(_cnn_layers[i]->hiddens());
        for (dim_t iBlock = seq<dimCount>(0); iBlock < blockCount; inc(iBlock, blockCount)) {

          // Wrap target and visibles
          *_target_tensors[i + 1] = (*_target_tensors[i])[iBlock, _model.cnn_layers()[i]->poolig_size(), _target_tensors[i]->size() - iBlock];
          _cnn_layers[i + 1]->visibles() =
              _cnn_layers[i]->hiddens()[iBlock, _model.cnn_layers()[i]->pooling_size(), _cnn_layers[i]->hiddens().size() - iBlock];

          update_gradient(iLayer + 1);
          _cnn_layers[i + 1]->backprop_visible_deltas();
          deltas[iBlock, _model.cnn_layers()[i]->pooling_size(), _cnn_layers[i]->hiddens().size() - iBlock] = _cnn_layers[i + 1]->visible_deltas();
        }
        _cnn_layers[i]->backprop_hidden_deltas(deltas);
        _cnn_layers[i]->update_gradient();
      } else {

        dim_t iBlock = seq<dimCount>(0);
        dim_t blockCount = _patch_counts[i];
        dim_t hiddenSize = _model.cnn_layers()[i]->hiddens_size() - _patch_sizes[i] + 1;
        dim_t blockSize = hiddenSize / _model.cnn_layers()[i]->pooling_size();

        // TODO: completed reshaping the target into a matrix. Need to revise from here after

        _nn_layers[0]->visibles().resize(seq(blockCount.prod(), blockSize.prod()));
        _target_matrix = trans(reshape(target, target.size()[dimCount - 1], blockCount.prod()));
        for (int i = 0; iBlock < blockCount; inc(iBlock, blockCount), ++i) {

          // Transition from convolutional model to dense model
          row(_nn_layers[0]->visibles(), i) = reshape(_cnn_layers[i]->hiddens()[iBlock, _model.cnn_layers()[i]->pooling_size(), hiddenSize], 1, blockSize.prod());
        }

        update_gradient(iLayer + 1);


        // Transition from convolutional model to dense model
//        _nn_layers[0]->visibles() = reshape(_cnn_layers[i]->hiddens(), 1, _cnn_layers[i]->hiddens().count());
//        update_gradient(iLayer + 1, target);

        // Transition back
        _nn_layers[0]->backprop_visible_deltas();

        // TODO: properly collect deltas
        _cnn_layers[i]->backprop_hidden_deltas(reshape(_nn_layers[0]->visible_deltas(), _model.cnn_layers()[i]->hiddens_size()));
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
