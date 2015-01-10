/*
 * conv_rbm.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONV_RBM_HPP_
#define TBBLAS_DEEPLEARN_CONV_RBM_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>
#include <tbblas/random.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/sequence.hpp>
#include <tbblas/util.hpp>
#include <tbblas/mask.hpp>
#include <tbblas/change_stream.hpp>
#include <tbblas/context.hpp>

#include <boost/thread/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/mult_sum.hpp>
#include <tbblas/deeplearn/repeat_mult.hpp>
#include <tbblas/deeplearn/repeat_mult_sum.hpp>
#include <tbblas/deeplearn/convolution_type.hpp>
#include <tbblas/deeplearn/unit_type.hpp>
#include <tbblas/deeplearn/dropout_method.hpp>
#include <tbblas/deeplearn/sparsity_method.hpp>

#include <tbblas/deeplearn/conv_rbm_model.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

/// This class creates multiple threads
/**
 * Some changes to the previous design:
 * - No thread local variables. Thread local variables are replaced by vectors of
 *   shared pointers. Better control over the creation and destruction of variables.
 * - Every thread has a local reference to the memory. Makes code cleaner.
 */

template<class T, unsigned dims>
class conv_rbm {
public:
  const static unsigned dimCount = dims;
  typedef T value_t;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;
  typedef tbblas::complex<value_t> complex_t;
  typedef tbblas::fft_plan<dimCount> plan_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef tbblas::tensor<complex_t, dimCount, true> ctensor_t;
  typedef std::vector<boost::shared_ptr<ctensor_t> > v_ctensor_t;

  typedef tbblas::random_tensor2<value_t, dimCount, true, tbblas::uniform<value_t> > uniform_t;
  typedef tbblas::random_tensor2<value_t, dimCount, true, tbblas::normal<value_t> > normal_t;

  typedef conv_rbm_model<value_t, dimCount> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  tensor_t _b, _binc;
  tensor_t deltab, db2, deltab2;
  v_ctensor_t cF, cFinc;
  v_tensor_t _c, _cinc;
  v_tensor_t deltaF, deltac, dF2, dc2, deltaF2, deltac2;
  v_tensor_t drops;

  // visible and hidden units in GPU memory

  // Sizes
  dim_t padded_visible_size, visible_size, hidden_size,
        visible_layer_size, hidden_layer_size,
        visible_layer_batch_size, hidden_layer_batch_size,
        visible_batch_size, kernel_size, padded_kernel_size, hidden_topleft,
        patch_count, patch_size, patch_batch_size, patch_layer_size, patch_layer_batch_size;
//        vbMaskSize, hbMaskSize, spMaskSize;

  // one element per thread
  tensor_t v, h, shifted_f, padded_f, f, padded_k, v_mask, h_mask;  // TODO: do I need shifted_f, padded_f, and f or can I replace some of them with v?
  ctensor_t cv, ch, chdiff;
  plan_t plan_v, iplan_v, plan_h, iplan_h;
  uniform_t v_rand, h_rand;
  normal_t v_noise, h_noise;

  tensor_t _visibles, _hiddens, _pooled;  // interface tensors

  int _filter_batch_length, _voxel_count;
  bool _memory_allocated, _double_weights, _host_updated;

  value_t _dropout_rate, _sparsity_target, _sparsity_weight, _positive_batch_size, _negative_batch_size;
  sparsity_method _sparsity_method;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm(model_t& model, dim_t patch_count = seq<dimCount>(1)) : model(model), patch_count(patch_count),
    _filter_batch_length(1),
    _memory_allocated(false), _double_weights(false), _host_updated(true),
    _dropout_rate(0), _sparsity_target(0.1), _sparsity_weight(0),
    _positive_batch_size(0), _negative_batch_size(0),
    _sparsity_method(sparsity_method::OnlySharedBias)
  {
  }

private:
  conv_rbm(const conv_rbm&);

public:
  // Automatically frees GPU memory
  virtual ~conv_rbm() {
    if (_memory_allocated)
      free_gpu_memory();
  }

  // These functions can run in parallel. They also create threads

  /// Transforms
  void allocate_gpu_memory() {
    using namespace tbblas;

    if (_memory_allocated)
      return;

    _memory_allocated = true;

    // Prepare sizes
    visible_size = (model.visibles_size() + model.stride_size() - 1) / model.stride_size();
    kernel_size = (model.kernel_size() + model.stride_size() - 1) / model.stride_size();
    padded_visible_size = visible_size * model.stride_size();
    padded_kernel_size = kernel_size * model.stride_size();
    kernel_size[dimCount - 1] = visible_size[dimCount - 1] = model.visibles_size()[dimCount - 1] * model.stride_size().prod();
    hidden_size = model.hiddens_size();

    patch_size = (visible_size + patch_count - 1) / patch_count + kernel_size - 1;
    patch_layer_size = patch_layer_batch_size = patch_batch_size = patch_size;
    tbblas_print(visible_size);
    tbblas_print(patch_count);
    tbblas_print(patch_size);

    visible_batch_size = visible_layer_batch_size = visible_layer_size = visible_size;
    hidden_layer_size = hidden_layer_batch_size = hidden_size;
    patch_layer_size[dimCount - 1] = hidden_layer_size[dimCount - 1] = visible_layer_size[dimCount - 1] = 1;
    patch_batch_size[dimCount - 1] = visible_batch_size[dimCount - 1] = visible_size[dimCount - 1] * _filter_batch_length;
    patch_layer_batch_size[dimCount - 1] = visible_layer_batch_size[dimCount - 1] = _filter_batch_length;
    hidden_layer_batch_size = hidden_layer_size * seq(1,1,1,_filter_batch_length);

    if (model.convolution_type() == convolution_type::Valid){
      hidden_topleft = kernel_size / 2;
      hidden_topleft[dimCount - 1] = 0;
    } else {
      hidden_topleft = seq<dimCount>(0);
    }

    _hiddens = zeros<value_t>(hidden_size);

#ifndef TBBLAS_CONV_RBM_NO_SELFTEST
    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor2<value_t, dimCount, true, normal<value_t> > v_noise(visible_size);

      tensor_t A = v_noise(), B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
    {
      random_tensor2<value_t, dimCount, true, normal<value_t> > v_noise(visible_layer_batch_size);

      tensor_t A = v_noise(), B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
#endif

    if (model.filters().size() % _filter_batch_length)
      throw std::runtime_error("The number of filters must be a multiple of the filter batch length.");

    cF.resize(model.filters().size() / _filter_batch_length);
    _c.resize(model.hidden_bias().size() / _filter_batch_length);

    drops.resize(cF.size());

    {
      dim_t padded_mask_size = padded_visible_size;
      padded_mask_size[dimCount - 1] = 1;

      v_mask = zeros<value_t>(padded_mask_size);
      v_mask[seq<dimCount>(0), model.mask().size()] = model.mask();

      _voxel_count = sum(v_mask) * padded_visible_size[dimCount - 1];

      tensor_t temp = rearrange(v_mask, model.stride_size());
      tensor_t mask = sum(temp, dimCount - 1);
      mask = mask > 0;

      // pad h mask according to convolution shrinkage
      if (model.convolution_type() == convolution_type::Valid) {
        h_mask = zeros<value_t>(visible_layer_size);
        h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
        h_mask = h_mask * mask;
      } else {
        h_mask = mask;
      }
    }

    _visibles = zeros<value_t>(padded_visible_size);
    _visibles[seq<dimCount>(0), model.visibles_size()] = model.visible_bias();
    _b = rearrange(_visibles, model.stride_size());
//    cb = fft(v, dimCount - 1, plan_v);

    _visibles = zeros<value_t>(padded_visible_size);

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
      plan_t plan_f;

      padded_k = zeros<value_t>(padded_kernel_size);

      for (size_t k = 0; k < cF.size(); ++k) {
        // Create padded tensor for an entire filter batch
        pad = zeros<value_t>(patch_batch_size);

        // Insert kernels into padded tensor
        for (size_t j = 0; j < _filter_batch_length; ++j) {
          // Copy kernel to GPU
          padded_k[seq<dimCount>(0), model.kernel_size()] = *model.filters()[k * _filter_batch_length + j];
          kern = rearrange(padded_k, model.stride_size());

          assert(kern.size() == kernel_size);

          dim_t topleft = patch_size / 2 - kernel_size / 2;
          topleft[dimCount - 1] = j * patch_size[dimCount - 1];
          pad[topleft, kernel_size] = kern;
        }
        f = ifftshift(pad, dimCount - 1);
        cf = fft(f, dimCount - 1, plan_f);
        cF[k] = boost::make_shared<ctensor_t>(cf);

        h = zeros<value_t>(visible_layer_batch_size);
        for (int j = 0; j < _filter_batch_length; ++j) {
          h[seq(0,0,0,j), visible_layer_size] = *model.hidden_bias()[k * _filter_batch_length + j];
        }
//        ch = fft(h, dimCount - 1, plan_h);
//        cc[k] = boost::make_shared<ctensor_t>(ch);
        _c[k] = boost::make_shared<tensor_t>(h);

        drops[k] = boost::make_shared<tensor_t>();
      }
    }

    if (model.hiddens_type() == unit_type::Bernoulli)
      h_rand.resize(visible_layer_batch_size);

    if (model.hiddens_type() == unit_type::MyReLU ||
        model.hiddens_type() == unit_type::ReLU ||
        model.hiddens_type() == unit_type::ReLU1 ||
        model.hiddens_type() == unit_type::ReLU2 ||
        model.hiddens_type() == unit_type::ReLU4)
    {
      h_noise.resize(visible_layer_batch_size);
    }

    if (model.visibles_type() == unit_type::MyReLU ||
        model.visibles_type() == unit_type::ReLU ||
        model.visibles_type() == unit_type::ReLU1 ||
        model.visibles_type() == unit_type::ReLU2 ||
        model.visibles_type() == unit_type::ReLU4)
    {
      v_noise.resize(visible_size);
    }

    if (model.visibles_type() == unit_type::Bernoulli) {
      v_rand.resize(visible_size);
    }

//    vbMaskSize = cb.size();
//    if (model.shared_bias()) {
//      for (size_t i = 0; i < dimCount - 1; ++i)
//        vbMaskSize[i] = 1;
//    }

//    hbMaskSize = cc[0]->size();
//    if (model.shared_bias()) {
//      for (size_t i = 0; i < dimCount - 1; ++i)
//        hbMaskSize[i] = 1;
//    }

//    spMaskSize = cc[0]->size();
//    for (size_t i = 0; i < dimCount - 1; ++i)
//      spMaskSize[i] = 1;
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    dim_t fullsize = cv.fullsize();
    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract one filter from the filter batch
        dim_t topleft = patch_size / 2 - kernel_size / 2;
        cv = (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);
        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());
        *model.filters()[k * _filter_batch_length + j] = padded_k[seq<dimCount>(0), model.kernel_size()];
      }
//      h = ifft(*cc[k], dimCount - 1, iplan_h);
      *_c[k] = *_c[k] * (abs(*_c[k]) > 1e-16);

      for (int j = 0; j < _filter_batch_length; ++j) {
        *model.hidden_bias()[k * _filter_batch_length + j] = (*_c[k])[seq(0,0,0,j), visible_layer_size];
      }
    }

//    f = ifft(cb, dimCount - 1, iplan_v);
    _b = _b * (abs(_b) > 1e-16);
    tensor_t padded = rearrange_r(_b, model.stride_size());
    host_tensor_t b = padded[seq<dimCount>(0), model.visibles_size()];
    tbblas::synchronize();
    model.set_visible_bias(b);
  }

  void free_gpu_memory() {
    if (!_host_updated)
       write_model_to_host();

    _b = tensor_t();
    _binc = tensor_t();

    for (size_t k = 0; k < cF.size(); ++k) {
      cF[k] = boost::shared_ptr<ctensor_t>();
      _c[k] = boost::shared_ptr<tensor_t>();
      drops[k] = boost::shared_ptr<tensor_t>();
    }

    for (size_t k = 0; k < cFinc.size(); ++k) {
      cFinc[k] = boost::shared_ptr<ctensor_t>();
      _cinc[k] = boost::shared_ptr<tensor_t>();
    }

    _memory_allocated = false;
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_visibles.size() == padded_visible_size);

    _visibles = ((_visibles - model.mean()) / model.stddev()) * tbblas::repeat(v_mask, _visibles.size() / v_mask.size());
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_visibles.size() == padded_visible_size);

    _visibles = ((_visibles * model.stddev()) + model.mean()) * tbblas::repeat(v_mask, _visibles.size() / v_mask.size());
  }

  void pool_hiddens() {
    switch (model.pooling_method()) {
    case pooling_method::NoPooling:
      _pooled = _hiddens;
      break;

    case pooling_method::StridePooling:
      _pooled = _hiddens[seq<dimCount>(0), model.pooling_size(), _hiddens.size()];
      break;

    default:
      throw std::runtime_error("Unsupported pooling method.");
    }
  }

  void unpool_hiddens() {
    switch (model.pooling_method()) {
    case pooling_method::NoPooling:
      _hiddens = _pooled;
      break;

    case pooling_method::StridePooling:
      allocate_hiddens();
      _hiddens[seq<dimCount>(0), model.pooling_size(), _hiddens.size()] = _pooled;
      break;

    default:
      throw std::runtime_error("Unsupported pooling method.");
    }
  }

  void infer_visibles_from_outputs(bool onlyFilters = false) {
    if (model.has_pooling_layer())
      unpool_hiddens();

    infer_visibles(onlyFilters);
  }

  void infer_visibles(bool onlyFilters = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    // TODO: divide convolution into subregions

    // for each subregion
    {
      cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = 0; k < cF.size(); ++k) {
        h = zeros<value_t>(patch_layer_batch_size);

        // TODO: use overlap size here
        h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
        if (!onlyFilters) // TODO: probably don't need to mask here
          h = h * repeat(h_mask, h.size() / h_mask.size());
        ch = fft(h, dimCount - 1, plan_h);
        cv += repeat_mult_sum(ch, *cF[k]);
      }
      v = ifft(cv, dimCount - 1, iplan_v);

      // fill a patch
    }

    if (!onlyFilters)
      v += _b;

    if (!onlyFilters) {
      switch(model.visibles_type()) {
        case unit_type::Bernoulli: v = sigm(v); break;
        case unit_type::Gaussian:  break;
        case unit_type::ReLU:      v = max(0.0, v);  break;
        case unit_type::MyReLU:    v = nrelu_mean(v); break;
        case unit_type::ReLU1:     v = min(1.0, max(0.0, v));  break;
        case unit_type::ReLU2:     v = min(2.0, max(0.0, v));  break;
        case unit_type::ReLU4:     v = min(4.0, max(0.0, v));  break;
        case unit_type::ReLU8:     v = min(8.0, max(0.0, v));  break;
      }
    }

    _visibles = rearrange_r(v, model.stride_size());
    _visibles = _visibles * repeat(v_mask, _visibles.size() / v_mask.size());
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_visibles.size() == padded_visible_size);

    v = rearrange(_visibles, model.stride_size());

    // TODO: iterate over subregions
    // for each subregion
    {
      cv = tbblas::fft(v, dimCount - 1, plan_v);  // TODO: fft of a sub region

      for (size_t k = 0; k < cF.size(); ++k) {
        ch = conj_mult_sum(cv, *cF[k]);

        h = ifft(ch, dimCount - 1, iplan_h);
        h += *_c[k];  // TODO: apply sub-region of bias terms

        switch (model.hiddens_type()) {
          case unit_type::Bernoulli: h = sigm(h); break;
          case unit_type::ReLU:      h = max(0.0, h);  break;
          case unit_type::MyReLU:    h = nrelu_mean(h); break;
          case unit_type::ReLU1:     h = min(1.0, max(0.0, h));  break;
          case unit_type::ReLU2:     h = min(2.0, max(0.0, h));  break;
          case unit_type::ReLU4:     h = min(4.0, max(0.0, h));  break;
          case unit_type::ReLU8:     h = min(8.0, max(0.0, h));  break;
        }
        if (_dropout_rate > 0)
          h = h * *drops[k] / (1. - _dropout_rate) * repeat(h_mask, h.size() / h_mask.size());
        else
          h = h * repeat(h_mask, h.size() / h_mask.size());

        // TODO: fill a subregion of the hiddens
        _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
      }
    }
  }

  void infer_outputs() {
    infer_hiddens();

    if (model.has_pooling_layer())
      pool_hiddens();
  }

  void sample_visibles_from_outputs() {
    if (model.has_pooling_layer())
      unpool_hiddens();

    sample_visibles();
  }

  void sample_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    v = ifft(cv, dimCount - 1, iplan_v);
    v += _b;

    switch(model.visibles_type()) {
      case unit_type::Gaussian:  break;
      case unit_type::Bernoulli: v = sigm(v) > v_rand(); break;
      case unit_type::MyReLU:
      case unit_type::ReLU:      v = max(0.0, v + sqrt(sigm(v)) * v_noise()); break;
      case unit_type::ReLU1:     v = min(1.0, max(0.0, v + (v > 0) * (v < 1.0) * v_noise())); break;
      case unit_type::ReLU2:     v = min(2.0, max(0.0, v + (v > 0) * (v < 2.0) * v_noise())); break;
      case unit_type::ReLU4:     v = min(4.0, max(0.0, v + (v > 0) * (v < 4.0) * v_noise())); break;
      case unit_type::ReLU8:     v = min(8.0, max(0.0, v + (v > 0) * (v < 8.0) * v_noise())); break;
    }
    _visibles = rearrange_r(v, model.stride_size());
    _visibles = _visibles * repeat(v_mask, _visibles.size() / v_mask.size());
  }

  void sample_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_visibles.size() == padded_visible_size);

    v = rearrange(_visibles, model.stride_size());
    cv = tbblas::fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cF.size(); ++k) {
      ch = conj_mult_sum(cv, *cF[k]);

      h = ifft(ch, dimCount - 1, iplan_h);
      h += *_c[k];

      switch (model.hiddens_type()) {
        case unit_type::Bernoulli: h = sigm(h) > h_rand(); break;
        case unit_type::MyReLU:
        case unit_type::ReLU:      h = max(0.0, h + sqrt(sigm(h)) * h_noise()); break;
        case unit_type::ReLU1:     h = min(1.0, max(0.0, h + (h > 0) * (h < 1.0) * h_noise())); break;
        case unit_type::ReLU2:     h = min(2.0, max(0.0, h + (h > 0) * (h < 2.0) * h_noise())); break;
        case unit_type::ReLU4:     h = min(4.0, max(0.0, h + (h > 0) * (h < 4.0) * h_noise())); break;
        case unit_type::ReLU8:     h = min(8.0, max(0.0, h + (h > 0) * (h < 8.0) * h_noise())); break;
      }
      // TODO: sub-mask and sub-drop required or do this stuff after the entire H was calculated
      if (_dropout_rate > 0)
        h = h * *drops[k] / (1. - _dropout_rate) * repeat(h_mask, h.size() / h_mask.size());
      else
        h = h * repeat(h_mask, h.size() / h_mask.size());
      _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
  }

  void sample_outputs() {
    sample_hiddens();

    if (model.has_pooling_layer())
      pool_hiddens();
  }

  void init_dropout(value_t rate, dropout_method method = dropout_method::DropUnit) {
    using namespace tbblas;

    if (rate < 0 || rate >= 1)
      throw std::runtime_error("Drop out rate must be in [0,1).");

    if (method == dropout_method::NoDrop) {
      _dropout_rate = 0;
      return;
    }

    if (!_memory_allocated)
      allocate_gpu_memory();

    _dropout_rate = rate;

    if (h_rand.size() != visible_layer_batch_size)
      h_rand.resize(visible_layer_batch_size);

    if (method == dropout_method::DropColumn) {
      *drops[0] = repeat((*drops[0])[seq(0,0,0,0), visible_layer_size], visible_layer_batch_size / visible_layer_size);

      for (size_t k = 1; k < drops.size(); ++k)
        *drops[k] = *drops[0];
    } else {
      for (size_t k = 0; k < drops.size(); ++k)
        *drops[k] = h_rand() > _dropout_rate;
    }
  }

  void update_positive_gradient() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (_binc.size() != _b.size())
      _binc = zeros<value_t>(_b.size());

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    if (!_cinc.size()) {
      _cinc.resize(_c.size());
      for (size_t i = 0; i < _cinc.size(); ++i)
        _cinc[i] = boost::make_shared<tensor_t>(zeros<value_t>(_c[i]->size()));
    }

    // TODO: how much time does this take?

    // TODO: do this on a sub-region and add up the results from all subregions

//    _binc += value_t(1) / visible_size[dimCount - 1] * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;
    if (model.shared_bias()) {
      // TODO: calculate shared bias per channel
      _binc = _binc + sum(v) / v.count();
    } else {
      _binc += v;
    }

    v = rearrange(_visibles, model.stride_size());
    cv = tbblas::fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cFinc.size(); ++k) {

      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(1) / _voxel_count);

      if (model.shared_bias()) {
        // TODO: calculate shared bias per channel
        *_cinc[k] = *_cinc[k] + sum(h) / h.count();
      } else {
        *_cinc[k] += h;
      }
      switch(_sparsity_method) {
      case sparsity_method::NoSparsity:
        break;

//      case sparsity_method::WeightsAndBias:
//        chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
//        *cFinc[k] = *cFinc[k] + value_t(1) / _voxel_count * _sparsity_weight * repeat(conj(chdiff), cFinc[k]->size() / ch.size()) * repeat(cv, cFinc[k]->size() / cv.size());
//        *ccinc[k] = *ccinc[k] + value_t(1) / visible_size[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
//        break;

      case sparsity_method::OnlyBias:
        *_cinc[k] += _sparsity_weight * (_sparsity_target + -h);
        break;

      case sparsity_method::OnlySharedBias:
        *_cinc[k] = *_cinc[k] + _sparsity_weight * sum(_sparsity_target + -h) / h.count();
        break;

      default:
        throw std::runtime_error("Unsupported sparsity method.");
      }
    }

    ++_positive_batch_size;
  }

  void update_negative_gradient() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_binc.size() == _b.size());
    assert(cFinc.size());
    assert(_cinc.size());

    if (model.shared_bias()) {
      _binc = _binc - sum(v) / v.count();
    } else {
      _binc = _binc - v;
    }

    v = rearrange(_visibles, model.stride_size());
    cv = tbblas::fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(-1) / _voxel_count);

      if (model.shared_bias()) {
        *_cinc[k] = *_cinc[k] - sum(h) / h.count();
      } else {
        *_cinc[k] = *_cinc[k] - h;
      }
    }

    ++_negative_batch_size;
  }

  void momentum_step(value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    _host_updated = false;

    using namespace tbblas;

    if (_positive_batch_size != _negative_batch_size)
      throw std::runtime_error("Number of positive and negative updates does not match.");

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (deltab.size() != visible_size)
      deltab = zeros<value_t>(visible_size);

    if (!deltaF.size()) {
      deltaF.resize(model.filter_count());
      for (size_t k = 0; k < deltaF.size(); ++k)
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
    }

    if (!deltac.size()) {
      deltac.resize(_cinc.size());
      for (size_t k = 0; k < deltac.size(); ++k)
        deltac[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
    }

    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - kernel_size / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _positive_batch_size) - weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);
        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        // update deltaF
        *deltaF[iFilter] = momentum * *deltaF[iFilter] + padded_k[seq<dimCount>(0), model.kernel_size()];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = *deltaF[iFilter];
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cv = fft(shifted_f, dimCount - 1, plan_v);
        (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] = cv;
      }

//      h = ifft(*ccinc[k], dimCount - 1, iplan_h);
      *deltac[k] = momentum * *deltac[k] + *_cinc[k] / _positive_batch_size;
//      *cc[k] = fft(*deltac[k], dimCount - 1, plan_h);

      // Apply delta to current filters
      *cF[k] = *cF[k] + epsilon * *cFinc[k];
      *_c[k] = *_c[k] + epsilon * *deltac[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *_cinc[k] = zeros<value_t>(_cinc[k]->size());
    }

    // TODO: do masking
//    padded_f = ifft(cbinc, dimCount - 1, iplan_v);
    deltab = (momentum * deltab + _binc / _positive_batch_size);// * repeat(v_mask, deltab.size() / v_mask.size());
//    cbinc = fft(deltab, dimCount - 1, plan_v);

    _b += epsilon * deltab;
    _binc = zeros<value_t>(_binc.size());

    _positive_batch_size = _negative_batch_size = 0;
  }

  void adadelta_step(value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    _host_updated = false;

    using namespace tbblas;

    value_t gradient_reduction = 1;

    if (_positive_batch_size != _negative_batch_size)
      throw std::runtime_error("Number of positive and negative updates does not match.");

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (db2.size() != visible_size || deltab.size() != visible_size || deltab2.size() != visible_size) {
      db2 = deltab = deltab2 = zeros<value_t>(visible_size);
    }

    if (!dF2.size() || !deltaF.size() || !deltaF2.size()) {
      dF2.resize(model.filter_count());
      deltaF.resize(model.filter_count());
      deltaF2.resize(model.filter_count());
      for (size_t k = 0; k < dF2.size(); ++k) {
        dF2[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
        deltaF2[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
      }
    }

    if (!dc2.size()|| !deltac.size() || !deltac2.size()) {
      dc2.resize(_cinc.size());
      deltac.resize(_cinc.size());
      deltac2.resize(_cinc.size());
      for (size_t k = 0; k < dc2.size(); ++k) {
        dc2[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
        deltac[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
        deltac2[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
      }
    }


    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - kernel_size / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _positive_batch_size) - weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);
        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        // update deltaF
        *dF2[iFilter] = momentum * *dF2[iFilter] + (1.0 - momentum) * padded_k[seq<dimCount>(0), model.kernel_size()] * padded_k[seq<dimCount>(0), model.kernel_size()];
        *deltaF[iFilter] = sqrt(*deltaF2[iFilter] + epsilon) / sqrt(*dF2[iFilter] + epsilon) * padded_k[seq<dimCount>(0), model.kernel_size()];
        *deltaF2[iFilter] = momentum * *deltaF2[iFilter] + (1.0 - momentum) * *deltaF[iFilter] * *deltaF[iFilter];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = *deltaF[iFilter];
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cv = fft(shifted_f, dimCount - 1, plan_v);
        (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] = cv;
      }

//      h = ifft(*ccinc[k], dimCount - 1, iplan_h);
      *_cinc[k] = *_cinc[k] / _positive_batch_size;

      *dc2[k] = momentum * *dc2[k] + (1.0 - momentum) * *_cinc[k] * *_cinc[k];
      *deltac[k] = sqrt(*deltac2[k] + epsilon) / sqrt(*dc2[k] + epsilon) * *_cinc[k];
      *deltac2[k] = momentum * *deltac2[k] + (1.0 - momentum) * *deltac[k] * *deltac[k];
//      *cc[k] = fft(*deltac[k], dimCount - 1, plan_h);

      // Apply delta to current filters
      *cF[k] = *cF[k] + gradient_reduction * *cFinc[k];
      *_c[k] = *_c[k] + gradient_reduction * *deltac[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *_cinc[k] = zeros<value_t>(_cinc[k]->size());
    }

//    padded_f = ifft(cbinc, dimCount - 1, iplan_v);
    _binc = _binc / _positive_batch_size; // * repeat(v_mask, padded_f.size() / v_mask.size());


    db2 = momentum * db2 + (1.0 - momentum) * _binc * _binc;
    deltab = sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * _binc;
    deltab2 = momentum * deltab2 + (1.0 - momentum) * deltab * deltab;
//    cbinc = fft(deltab, dimCount - 1, plan_v);

    _b += gradient_reduction * deltab;
    _binc = zeros<value_t>(_binc.size());

    _positive_batch_size = _negative_batch_size = 0;
  }

  // Access to model data
  const proxy<tensor_t> visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return _visibles[seq<dimCount>(0), model.visibles_size()];
  }

  void allocate_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
  }

  tensor_t& hiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return _hiddens;
  }

  void allocate_hiddens() {
    if (_hiddens.size() != model.hiddens_size())
      _hiddens.resize(model.hiddens_size());
  }

  tensor_t& pooled_units() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return _pooled;
  }

  void allocate_pooled_units() {
    if (_pooled.size() != model.pooled_size())
      _pooled.resize(model.pooled_size());
  }

  tensor_t& outputs() {
    if (model.has_pooling_layer())
      return pooled_units();
    else
      return hiddens();
  }

  void allocate_outputs() {
    if (model.has_pooling_layer())
      allocate_pooled_units();
    else
      allocate_hiddens();
  }

  void set_batch_length(int length) {
    if (length < 1 || model.filter_count() % length != 0)
      throw std::runtime_error("Filter count must be a multiple of filter batch length!");

    const bool reallocate = (length != _filter_batch_length);
    _filter_batch_length = length;

    if (reallocate) {
      if (_memory_allocated) {
        free_gpu_memory();
        allocate_gpu_memory();
      }
    }
  }

  int batch_length() const {
    return _filter_batch_length;
  }

  void set_sparsity_target(value_t target) {
    _sparsity_target = target;
  }

  void set_sparsity_weight(value_t weight) {
    _sparsity_weight = weight;
  }

  void set_sparsity_method(const sparsity_method& method) {
    _sparsity_method = method;
  }


  void change_mask(host_tensor_t& mask) {
    using namespace tbblas;

    throw std::runtime_error("change_mask not implemented");

//    if (!_memory_allocated)
//      allocate_gpu_memory();
//
//    v_mask = zeros<value_t>(layer_size);
//    v_mask[seq<dimCount>(0), mask.size()] = mask;
//
//    // pad h mask according to convolution shrinkage
//    if (model.convolution_type() == convolution_type::Valid){
//      h_mask = zeros<value_t>(layer_size);
//      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
//      h_mask = h_mask * v_mask;
//    } else {
//      h_mask = v_mask;
//    }
  }
};

template<class T, unsigned dims>
const T conv_rbm<T, dims>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_HPP_ */
