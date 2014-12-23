/*
 * conv_rbm.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONV_RBM_HPP_
#define TBBLAS_DEEPLEARN_CONV_RBM_HPP_

// TODO: lazy initialisation of variables used for training
// TODO: momentum_step and adadelta_step replace init_gradient and apply_gradient
// TODO: counter in update_gradient

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
  ctensor_t cb, cbinc;
  tensor_t deltab, db2, deltab2;
  v_ctensor_t cF, cc, cFinc, ccinc;
  v_tensor_t deltaF, deltac, dF2, dc2, deltaF2, deltac2;
  v_tensor_t drops;

  // visible and hidden units in GPU memory

  // Sizes
  dim_t visible_size, hidden_size, size,
        visible_layer_size, hidden_layer_size, layer_size,
        hidden_layer_batch_size, layer_batch_size,
        filter_batch_size, hidden_topleft,
        vbMaskSize, hbMaskSize, spMaskSize;

  // one element per thread
  tensor_t v, h, shifted_f, padded_f, f, v_mask, h_mask;
  ctensor_t cv, ch, chdiff;
  plan_t plan_v, iplan_v, plan_h, iplan_h;
  uniform_t v_rand, h_rand;
  normal_t v_noise, h_noise;

  tensor_t _visibles, _hiddens, _pooled;  // interface tensors

  int _filter_batch_length, _hidden_count;
  bool _memory_allocated, _double_weights, _host_updated;

  value_t _dropout_rate, _sparsity_target, _sparsity_weight, _positive_batch_size, _negative_batch_size;
  sparsity_method _sparsity_method;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm(model_t& model) : model(model),
    _filter_batch_length(1),
    _memory_allocated(false), _double_weights(false), _host_updated(true),
    _dropout_rate(0), _sparsity_target(0.1), _sparsity_weight(0),
    _positive_batch_size(0), _negative_batch_size(0),
    _sparsity_method(sparsity_method::OnlySharedBias)
  {
    _hidden_count = sum(model.mask()) * model.hiddens_size()[dimCount - 1];
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
    size = visible_size = model.visibles_size();
    hidden_size = model.hiddens_size();

    visible_layer_size = visible_size;
    layer_size = filter_batch_size = layer_batch_size = size;
    hidden_layer_size = hidden_layer_batch_size = hidden_size;
    hidden_layer_size[dimCount - 1] = visible_layer_size[dimCount - 1] = layer_size[dimCount - 1] = 1;
    filter_batch_size[dimCount - 1] = size[dimCount - 1] * _filter_batch_length;
    layer_batch_size[dimCount - 1] = _filter_batch_length;
    hidden_layer_batch_size = hidden_layer_size * seq(1,1,1,_filter_batch_length);

    if (model.convolution_type() == convolution_type::Valid){
      hidden_topleft = model.kernel_size() / 2;
      hidden_topleft[dimCount - 1] = 0;
    } else {
      hidden_topleft = seq<dimCount>(0);
    }

    _hiddens = zeros<value_t>(hidden_size);

#ifndef TBBLAS_CONV_RBM_NO_SELFTEST
    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor2<value_t, dimCount, true, normal<value_t> > v_noise(size);

      tensor_t A = v_noise(), B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
    {
      random_tensor2<value_t, dimCount, true, normal<value_t> > v_noise(layer_batch_size);

      tensor_t A = v_noise(), B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
#endif

    cF.resize(model.filters().size() / _filter_batch_length);
    cc.resize(model.hidden_bias().size() / _filter_batch_length);

    drops.resize(cF.size());

    v_mask = zeros<value_t>(layer_size);
    v_mask[seq<dimCount>(0), model.mask().size()] = model.mask();

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
      h_mask = h_mask * v_mask;
    } else {
      h_mask = v_mask;
    }

    tensor_t b = model.visible_bias();
    cb = fft(b, dimCount - 1, plan_v);

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
      plan_t plan_f;
      for (size_t k = 0; k < cF.size(); ++k) {
        pad = zeros<value_t>(filter_batch_size);
        for (size_t j = 0; j < _filter_batch_length; ++j) {
          kern = *model.filters()[k * _filter_batch_length + j];
          dim_t topleft = size / 2 - kern.size() / 2;
          topleft[dimCount - 1] = j * size[dimCount - 1];
          pad[topleft, kern.size()] = kern;
        }
        f = ifftshift(pad, dimCount - 1);
        cf = fft(f, dimCount - 1, plan_f);
        cF[k] = boost::make_shared<ctensor_t>(cf);

        h = zeros<value_t>(layer_batch_size);
        for (int j = 0; j < _filter_batch_length; ++j) {
          h[seq(0,0,0,j), visible_layer_size] = *model.hidden_bias()[k * _filter_batch_length + j];
        }
        ch = fft(h, dimCount - 1, plan_h);
        cc[k] = boost::make_shared<ctensor_t>(ch);

        drops[k] = boost::make_shared<tensor_t>();
      }
    }

    if (model.hiddens_type() == unit_type::Bernoulli)
      h_rand.resize(layer_batch_size);

    if (model.hiddens_type() == unit_type::MyReLU ||
        model.hiddens_type() == unit_type::ReLU ||
        model.hiddens_type() == unit_type::ReLU1 ||
        model.hiddens_type() == unit_type::ReLU2 ||
        model.hiddens_type() == unit_type::ReLU4)
    {
      h_noise.resize(layer_batch_size);
    }

    if (model.visibles_type() == unit_type::MyReLU ||
        model.visibles_type() == unit_type::ReLU ||
        model.visibles_type() == unit_type::ReLU1 ||
        model.visibles_type() == unit_type::ReLU2 ||
        model.visibles_type() == unit_type::ReLU4)
    {
      v_noise.resize(size);
    }

    if (model.visibles_type() == unit_type::Bernoulli) {
      v_rand.resize(size);
    }

    vbMaskSize = cb.size();
    if (model.shared_bias()) {
      vbMaskSize[0] = 1;
      vbMaskSize[1] = 1;
      vbMaskSize[2] = 1;
    }

    hbMaskSize = cc[0]->size();
    if (model.shared_bias()) {
      hbMaskSize[0] = 1;
      hbMaskSize[1] = 1;
      hbMaskSize[2] = 1;
    }

    spMaskSize = cc[0]->size();
    spMaskSize[0] = 1;
    spMaskSize[1] = 1;
    spMaskSize[2] = 1;
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    dim_t fullsize = cv.fullsize();
    tensor_t p;

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {
        dim_t topleft = size / 2 - model.kernel_size() / 2;
        cv = (*cF[k])[seq(0,0,0,j*cv.size()[3]), cv.size()];
        cv.set_fullsize(fullsize);
        f = ifft(cv, dimCount - 1, iplan_v);
        p = fftshift(f, dimCount - 1);

        *model.filters()[k * _filter_batch_length + j] = p[topleft, model.kernel_size()];
      }

      h = ifft(*cc[k], dimCount - 1, iplan_h);
      h = h * (abs(h) > 1e-16);

      for (int j = 0; j < _filter_batch_length; ++j) {
        *model.hidden_bias()[k * _filter_batch_length + j] = h[seq(0,0,0,j), layer_size];
      }
    }

    f = ifft(cb, dimCount - 1, iplan_v);
    f = f * (abs(f) > 1e-16);
    host_tensor_t b = f;
    model.set_visible_bias(b);
  }

  void free_gpu_memory() {
    if (!_host_updated)
       write_model_to_host();

    cb = ctensor_t();
    cbinc = ctensor_t();

    for (size_t k = 0; k < cF.size(); ++k) {
      cF[k] = cc[k] = boost::shared_ptr<ctensor_t>();
      drops[k] = boost::shared_ptr<tensor_t>();
    }

    for (size_t k = 0; k < cFinc.size(); ++k) {
      cFinc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
    }

    _memory_allocated = false;
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);

    v = ((v - model.mean()) / model.stddev()) * tbblas::repeat(v_mask, size / layer_size);
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);

    v = ((v * model.stddev()) + model.mean()) * tbblas::repeat(v_mask, size / layer_size);
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

  void infer_visibles(bool onlyFilters = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      if (!onlyFilters)
        h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }

    if (!onlyFilters)
      cv += cb;

    v = ifft(cv, dimCount - 1, iplan_v);

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
      v = v * repeat(v_mask, size / layer_size);
    }
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);

    cv = tbblas::fft(v, dimCount - 1, plan_v);
    for (size_t k = 0; k < cF.size(); ++k) {
      ch = conj_mult_sum(cv, *cF[k]);

      ch = ch + *cc[k];
      h = ifft(ch, dimCount - 1, iplan_h);

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
      _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
  }

  void sample_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    cv += cb;
    v = ifft(cv, dimCount - 1, iplan_v);

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
    v = v * repeat(v_mask, size / layer_size);
  }

  void sample_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);

    cv = tbblas::fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cF.size(); ++k) {
      ch = conj_mult_sum(cv, *cF[k]);

      ch = ch + *cc[k];
      h = ifft(ch, dimCount - 1, iplan_h);

      switch (model.hiddens_type()) {
        case unit_type::Bernoulli: h = sigm(h) > h_rand(); break;
        case unit_type::MyReLU:
        case unit_type::ReLU:      h = max(0.0, h + sqrt(sigm(h)) * h_noise()); break;
        case unit_type::ReLU1:     h = min(1.0, max(0.0, h + (h > 0) * (h < 1.0) * h_noise())); break;
        case unit_type::ReLU2:     h = min(2.0, max(0.0, h + (h > 0) * (h < 2.0) * h_noise())); break;
        case unit_type::ReLU4:     h = min(4.0, max(0.0, h + (h > 0) * (h < 4.0) * h_noise())); break;
        case unit_type::ReLU8:     h = min(8.0, max(0.0, h + (h > 0) * (h < 8.0) * h_noise())); break;
      }
      if (_dropout_rate > 0)
        h = h * *drops[k] / (1. - _dropout_rate) * repeat(h_mask, h.size() / h_mask.size());
      else
        h = h * repeat(h_mask, h.size() / h_mask.size());
      _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
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

    if (h_rand.size() != layer_batch_size)
      h_rand.resize(layer_batch_size);

    if (method == dropout_method::DropColumn) {
      *drops[0] = repeat((*drops[0])[seq(0,0,0,0),layer_size], layer_batch_size / layer_size);

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

    if (cbinc.size() != cb.size())
      cbinc = zeros<complex_t>(cb.size(), cb.fullsize());

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    if (!ccinc.size()) {
      ccinc.resize(cc.size());
      for (size_t i = 0; i < ccinc.size(); ++i)
        ccinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cc[i]->size(), cc[i]->fullsize()));
    }

    // TODO: how much time does this take?
    cv = tbblas::fft(v, dimCount - 1, plan_v);

    cbinc = cbinc + value_t(1) / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = 0; k < cFinc.size(); ++k) {

      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(1) / _hidden_count);
      *ccinc[k] = *ccinc[k] + value_t(1) / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
      switch(_sparsity_method) {
      case sparsity_method::WeightsAndBias:
        chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
        *cFinc[k] = *cFinc[k] + value_t(1) / _hidden_count * _sparsity_weight * repeat(conj(chdiff), cFinc[k]->size() / ch.size()) * repeat(cv, cFinc[k]->size() / cv.size());
        *ccinc[k] = *ccinc[k] + value_t(1) / model.visibles_size()[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
        break;

      case sparsity_method::OnlyBias:
        chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
        *ccinc[k] = *ccinc[k] + value_t(1) / model.visibles_size()[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
        break;

      case sparsity_method::OnlySharedBias:
        *ccinc[k] = *ccinc[k] + value_t(1) / model.visibles_size()[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) * (_sparsity_target * h.count() + -ch);
        break;
      }
    }

    ++_positive_batch_size;
  }

  void update_negative_gradient() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (cbinc.size() != cb.size())
      cbinc = zeros<complex_t>(cb.size(), cb.fullsize());

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    if (!ccinc.size()) {
      ccinc.resize(cc.size());
      for (size_t i = 0; i < ccinc.size(); ++i)
        ccinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cc[i]->size(), cc[i]->fullsize()));
    }

    cv = tbblas::fft(v, dimCount - 1, plan_v);

    cbinc = cbinc - value_t(1) / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(-1) / _hidden_count);
      *ccinc[k] = *ccinc[k] - value_t(1) / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
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
      deltac.resize(ccinc.size());
      for (size_t k = 0; k < deltac.size(); ++k)
        deltac[k] = boost::make_shared<tensor_t>(zeros<value_t>(layer_batch_size));
    }

    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - model.kernel_size() / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _positive_batch_size) - weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);
        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);

        // update deltaF
        *deltaF[iFilter] = momentum * *deltaF[iFilter] + padded_f[topleft, model.kernel_size()];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, model.kernel_size()] = *deltaF[iFilter];
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cv = fft(shifted_f, dimCount - 1, plan_v);
        (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] = cv;
      }

      h = ifft(*ccinc[k], dimCount - 1, iplan_h);
      *deltac[k] = momentum * *deltac[k] + h / _positive_batch_size;
      *cc[k] = fft(*deltac[k], dimCount - 1, plan_h);

      // Apply delta to current filters
      *cF[k] = *cF[k] + epsilon * *cFinc[k];
      *cc[k] = *cc[k] + epsilon * *ccinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *ccinc[k] = zeros<complex_t>(ccinc[k]->size(), ccinc[k]->fullsize());
    }

    padded_f = ifft(cbinc, dimCount - 1, iplan_v);
    deltab = momentum * deltab + padded_f / _positive_batch_size;
    cbinc = fft(deltab, dimCount - 1, plan_v);

    cb = cb + epsilon * cbinc;
    cbinc = zeros<complex_t>(cbinc.size(), cbinc.fullsize());

    _positive_batch_size = _negative_batch_size = 0;
  }

  void adadelta_step(value_t epsilon, value_t momentum = 0, value_t weightcost = 0) {
    _host_updated = false;

    using namespace tbblas;

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
      dc2.resize(ccinc.size());
      deltac.resize(ccinc.size());
      deltac2.resize(ccinc.size());
      for (size_t k = 0; k < dc2.size(); ++k) {
        dc2[k] = boost::make_shared<tensor_t>(zeros<value_t>(layer_batch_size));
        deltac[k] = boost::make_shared<tensor_t>(zeros<value_t>(layer_batch_size));
        deltac2[k] = boost::make_shared<tensor_t>(zeros<value_t>(layer_batch_size));
      }
    }

    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - model.kernel_size() / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _positive_batch_size) - weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);
        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);

        // update deltaF
        *dF2[iFilter] = momentum * *dF2[iFilter] + (1.0 - momentum) * padded_f[topleft, model.kernel_size()] * padded_f[topleft, model.kernel_size()];
        *deltaF[iFilter] = sqrt(*deltaF2[iFilter] + epsilon) / sqrt(*dF2[iFilter] + epsilon) * padded_f[topleft, model.kernel_size()];
        *deltaF2[iFilter] = momentum * *deltaF2[iFilter] + (1.0 - momentum) * *deltaF[iFilter] * *deltaF[iFilter];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, model.kernel_size()] = *deltaF[iFilter];
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cv = fft(shifted_f, dimCount - 1, plan_v);
        (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] = cv;
      }

      h = ifft(*ccinc[k], dimCount - 1, iplan_h);
      h = h / _positive_batch_size;

      *dc2[k] = momentum * *dc2[k] + (1.0 - momentum) * h * h;
      *deltac[k] = sqrt(*deltac2[k] + epsilon) / sqrt(*dc2[k] + epsilon) * h;
      *deltac2[k] = momentum * *deltac2[k] + (1.0 - momentum) * *deltac[k] * *deltac[k];
      *cc[k] = fft(*deltac[k], dimCount - 1, plan_h);

      // Apply delta to current filters
      *cF[k] = *cF[k] + 0.1 * *cFinc[k];
      *cc[k] = *cc[k] + 0.1 * *ccinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *ccinc[k] = zeros<complex_t>(ccinc[k]->size(), ccinc[k]->fullsize());
    }

    padded_f = ifft(cbinc, dimCount - 1, iplan_v);
    padded_f = padded_f / _positive_batch_size;

    db2 = momentum * db2 + (1.0 - momentum) * padded_f * padded_f;
    deltab = sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * padded_f;
    deltab2 = momentum * deltab2 + (1.0 - momentum) * deltab * deltab;
    cbinc = fft(deltab, dimCount - 1, plan_v);

    cb = cb + 0.1 * cbinc;
    cbinc = zeros<complex_t>(cbinc.size(), cbinc.fullsize());

    _positive_batch_size = _negative_batch_size = 0;
  }

  void set_input(tensor_t& input) {
    assert(model.input_size() == input.size());
    visibles() = rearrange(input, model.stride_size());
  }

  void get_input(tensor_t& input) {
    input = rearrange_r(visibles(), model.stride_size());
  }

  // Access to model data
  tensor_t& visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return v;
  }

  void allocate_visibles() {
    if (v.size() != model.visibles_size())
      v.resize(model.visibles_size());
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

    if (!_memory_allocated)
      allocate_gpu_memory();

    v_mask = zeros<value_t>(layer_size);
    v_mask[seq<dimCount>(0), mask.size()] = mask;

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
      h_mask = h_mask * v_mask;
    } else {
      h_mask = v_mask;
    }
  }
};

template<class T, unsigned dims>
const T conv_rbm<T, dims>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_HPP_ */
