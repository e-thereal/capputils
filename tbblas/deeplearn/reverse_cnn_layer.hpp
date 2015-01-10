/*
 * reverse_cnn_layer.hpp
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_REVERSE_CNN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_REVERSE_CNN_LAYER_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/sequence.hpp>
#include <tbblas/random.hpp>
#include <tbblas/mask.hpp>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/mult_sum.hpp>
#include <tbblas/deeplearn/repeat_mult.hpp>
#include <tbblas/deeplearn/repeat_mult_sum.hpp>

#include <tbblas/deeplearn/reverse_cnn_layer_model.hpp>
#include <tbblas/deeplearn/objective_function.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class reverse_cnn_layer {
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

  typedef reverse_cnn_layer_model<value_t, dimCount> model_t;

  typedef tbblas::random_tensor2<value_t, 4, true, tbblas::uniform<value_t> > uniform_t;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  ctensor_t cb, cbinc;
  tensor_t deltab, db2, deltab2;
  v_ctensor_t cF, cFinc;
  v_tensor_t deltaF, dF2, deltaF2;

  // Sizes
  dim_t padded_visible_size, visible_size, hidden_size,
        visible_layer_size, hidden_layer_size,
        hidden_layer_batch_size, visible_layer_batch_size,
        visible_batch_size, kernel_size, padded_kernel_size,
        hidden_topleft, vbMaskSize;

  tensor_t v, h, shifted_f, padded_f, padded_k, v_mask, h_mask;
  tensor_t padded_v, H, padded_dv, dH;
  ctensor_t cv, ch, chdiff;
  plan_t plan_v, iplan_v, plan_h, iplan_h;
  uniform_t h_rand, v_rand;

  int _filter_batch_length, _voxel_count;
  bool _memory_allocated, _host_updated;
  value_t _current_batch_size, _dropout_rate, _sensitivity_ratio;

  tbblas::deeplearn::objective_function _objective_function;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  reverse_cnn_layer(model_t& model) : model(model), _filter_batch_length(1),
      _memory_allocated(false), _host_updated(true), _current_batch_size(0), _dropout_rate(0), _sensitivity_ratio(0.5)
  {
    _voxel_count = model.visibles_count();
  }

private:
  reverse_cnn_layer(const reverse_cnn_layer&);

public:
  virtual ~reverse_cnn_layer() {
    if (!_host_updated)
      write_model_to_host();
  }

  void set_objective_function(const tbblas::deeplearn::objective_function& objective) {
    _objective_function = objective;
  }

  tbblas::deeplearn::objective_function objective_function() const {
    return _objective_function;
  }

  void set_sensitivity_ratio(const value_t& ratio) {
    _sensitivity_ratio = ratio;
  }

  value_t sensitivity_ratio() const {
    return _sensitivity_ratio;
  }

  void set_dropout_rate(const value_t& rate) {
    if (rate < 0 || rate >= 1)
      throw std::runtime_error("Drop out rate must be in [0,1).");

    _dropout_rate = rate;
  }

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

    visible_batch_size = visible_layer_batch_size = visible_layer_size = visible_size;
    hidden_layer_size = hidden_layer_batch_size = hidden_size;
    hidden_layer_size[dimCount - 1] = visible_layer_size[dimCount - 1] = 1;
    visible_batch_size[dimCount - 1] = visible_size[dimCount - 1] * _filter_batch_length;
    visible_layer_batch_size[dimCount - 1] = _filter_batch_length;
    hidden_layer_batch_size = hidden_layer_size * seq(1,1,1,_filter_batch_length);

    if (model.convolution_type() == convolution_type::Valid){
      hidden_topleft = kernel_size / 2;
      hidden_topleft[dimCount - 1] = 0;
    } else {
      hidden_topleft = seq<dimCount>(0);
    }

    padded_v = zeros<value_t>(padded_visible_size);
    padded_dv = zeros<value_t>(padded_visible_size);
    H = zeros<value_t>(hidden_size);

#ifndef TBBLAS_CNN_NO_SELFTEST
    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(visible_size);

      tensor_t A = v_noise, B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(visible_layer_batch_size);

      tensor_t A = v_noise, B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
#endif

    cF.resize(model.filters().size() / _filter_batch_length);

    {
      dim_t padded_mask_size = padded_visible_size;
      padded_mask_size[dimCount - 1] = 1;

      v_mask = zeros<value_t>(padded_mask_size);
      v_mask[seq<dimCount>(0), model.mask().size()] = model.mask();

      _voxel_count = sum(v_mask) * padded_visible_size[dimCount - 1];

//
//      dim_t padded_mask_size = padded_visible_size;
//      dim_t mask_size = model.visibles_size();
//      mask_size[dimCount - 1] = padded_mask_size[dimCount - 1] = 1;
//
//      v_mask = zeros<value_t>(padded_mask_size);
//      v_mask[seq<dimCount>(0), mask_size] = ones<value_t>(mask_size);
    }

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(visible_layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
    } else {
      h_mask = ones<value_t>(visible_layer_size);
    }

    padded_v = zeros<value_t>(padded_visible_size);
    padded_v[seq<dimCount>(0), model.visibles_size()] = model.bias();
    v = rearrange(padded_v, model.stride_size());
    cb = fft(v, dimCount - 1, plan_v);

    padded_v = zeros<value_t>(padded_visible_size);

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
      plan_t plan_f;

      padded_k = zeros<value_t>(padded_kernel_size);

      for (size_t k = 0; k < cF.size(); ++k) {
        // Created padded tensor for an entire filter batch
        pad = zeros<value_t>(visible_batch_size);

        // Insert kernels into padded tensor
        for (size_t j = 0; j < _filter_batch_length; ++j) {
          // Copy kernel to GPU
          padded_k[seq<dimCount>(0), model.kernel_size()] = *model.filters()[k * _filter_batch_length + j];
          kern = rearrange(padded_k, model.stride_size());

          assert(kern.size() == kernel_size);

          dim_t topleft = visible_size / 2 - kernel_size / 2;
          topleft[dimCount - 1] = j * visible_size[dimCount - 1];
          pad[topleft, kernel_size] = kern;
        }

        // Shift the kernel to be centered around 0 and not the image center
        f = ifftshift(pad, dimCount - 1);
        cf = fft(f, dimCount - 1, plan_f);
        cF[k] = boost::make_shared<ctensor_t>(cf);

//        h = zeros<value_t>(visible_layer_batch_size);
//        for (int j = 0; j < _filter_batch_length; ++j) {
//          h[seq(0,0,0,j), visible_layer_size] = *model.bias()[k * _filter_batch_length + j];
//        }
//        ch = fft(h, dimCount - 1, plan_h);
//        cb[k] = boost::make_shared<ctensor_t>(ch);
      }
    }

    vbMaskSize = cb.size();
    if (model.shared_bias()) {
      for (size_t i = 0; i < dimCount - 1; ++i)
        vbMaskSize[i] = 1;
    }
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
        dim_t topleft = visible_size / 2 - kernel_size / 2;
        cv = (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);

        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());
        *model.filters()[k * _filter_batch_length + j] = padded_k[seq<dimCount>(0), model.kernel_size()];
      }
    }

    v = ifft(cb, dimCount - 1, iplan_v);
    v = v * (abs(v) > 1e-16);
    tensor_t padded = rearrange_r(v, model.stride_size());
    host_tensor_t b = padded[seq<dimCount>(0), model.visibles_size()];
    tbblas::synchronize();
    model.set_bias(b);
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);

    padded_v = ((padded_v * model.stddev()) + model.mean()) * repeat(v_mask, padded_v.size() / v_mask.size());
  }

  void infer_visibles(bool dropout = false) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(H.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));
    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    cv += cb;

    v = ifft(cv, dimCount - 1, iplan_v);
    switch (model.activation_function()) {
      case activation_function::Sigmoid: v = sigm(v); break;
      case activation_function::ReLU:    v = max(0.0, v);  break;
      case activation_function::Linear:  break;
      default:
        throw std::runtime_error("Unsupported activation function.");
    }

    if (dropout && _dropout_rate > 0) {
      if (v_rand.size() != v.size())
        v_rand.resize(v.size());

      v = v * (v_rand() > _dropout_rate) / (1. - _dropout_rate);
    }

    padded_v = rearrange_r(v, model.stride_size());
    padded_v = padded_v * repeat(v_mask, padded_v.size() / v_mask.size());
  }


  /// Requires visible activation and hidden total activation
  void calculate_deltas(tensor_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function


    assert(target.size() == model.visibles_size());

    switch (_objective_function) {
    case tbblas::deeplearn::objective_function::SSD:

      // delta = (visible - target) * f'(X)
      switch (model.activation_function()) {
      case activation_function::Sigmoid:
        padded_dv[seq<dimCount>(0), model.visibles_size()] =
            (visibles() - target) * visibles() * (1 + -visibles());
        break;

      case activation_function::ReLU:
        padded_dv[seq<dimCount>(0), model.visibles_size()] =
            (visibles() - target) * (visibles() > 0);
        break;

      case activation_function::Softmax:
      case activation_function::Linear:
        padded_dv[seq<dimCount>(0), model.visibles_size()] =
            visibles() - target;
        break;
      }
      break;

    case tbblas::deeplearn::objective_function::SenSpe:
      {
        if (model.activation_function() != activation_function::Sigmoid)
          throw std::runtime_error("Activation function for objective function 'Sensitivity + Specificity' must be 'Sigmoid'.");

        // delta = (-alpha* target - beta * target + beta) * f'(X)

        const value_t positive_ratio = sum(target) / (value_t)target.count();
        const value_t alpha = _sensitivity_ratio / (positive_ratio + value_t(1e-8));
        const value_t beta = (value_t(1) - _sensitivity_ratio) / (value_t(1) - positive_ratio + value_t(1e-8));

        padded_dv[seq<dimCount>(0), model.visibles_size()] =
            (alpha * target + beta * (1 + -target)) * (visibles() - target) * visibles() * (1 + -visibles());
      }
      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_deltas(target).");
    }
  }

  /// Requires visible activation
  void calculate_u_deltas(tensor_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    switch (_objective_function) {
    case tbblas::deeplearn::objective_function::DSC:
      if (model.activation_function() != activation_function::Sigmoid)
        throw std::runtime_error("Activation function for objective function 'DSC' must be 'Sigmoid'.");

      padded_dv[seq<dimCount>(0), model.visibles_size()] =
          2 * target * visibles() * (1 + -visibles());
      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_u_deltas(target).");
    }
  }

  /// Requires visible activation and hidden total activation
  void calculate_v_deltas(tensor_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    switch (_objective_function) {
    case tbblas::deeplearn::objective_function::DSC:
      if (model.activation_function() != activation_function::Sigmoid)
        throw std::runtime_error("Activation function for objective function 'DSC' must be 'Sigmoid'.");

      padded_dv[seq<dimCount>(0), model.visibles_size()] =
          visibles() * (1 + -visibles());
      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_v_deltas(target).");
    }
  }

  void backprop_hidden_deltas() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_dv.size() == padded_visible_size);

    v = rearrange(padded_dv, model.stride_size());
    cv = fft(v, dimCount - 1, plan_v);

    dH = zeros<value_t>(hidden_size);
    for (size_t k = 0; k < cF.size(); ++k) {
      ch = conj_mult_sum(cv, *cF[k]);
      h = ifft(ch, dimCount - 1, iplan_h);
      dH[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
  }

  void backprop_hiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);

    v = rearrange(padded_v, model.stride_size());
    cv = fft(v, dimCount - 1, plan_v);

    dH = zeros<value_t>(hidden_size);
    for (size_t k = 0; k < cF.size(); ++k) {
      ch = conj_mult_sum(cv, *cF[k]);
      h = ifft(ch, dimCount - 1, iplan_h);
      H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
  }

  /// Takes visible deltas of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type >::type
  backprop_visible_deltas(const Expression& deltas) {

    assert(deltas.size() == model.visibles_size());

    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      padded_dv[seq<dimCount>(0), model.visibles_size()] =
          deltas * visibles() * (1 + -visibles());
      break;

    case activation_function::ReLU:
      padded_dv[seq<dimCount>(0), model.visibles_size()] =
          deltas * (visibles() > 0);
      break;

    case activation_function::Linear:
      padded_dv[seq<dimCount>(0), model.visibles_size()] =
          deltas;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
    return 0;
  }

  /// Requires visible deltas and hiddens
  void update_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (cbinc.size() != cb.size())
      cbinc = zeros<complex_t>(cb.size(), cb.fullsize());

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    v = rearrange(padded_dv, model.stride_size());
    cv = tbblas::fft(v, dimCount - 1, plan_v);

    cbinc = cbinc + value_t(1) / visible_size[dimCount - 1] * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(1) / _voxel_count);
    }

    ++_current_batch_size;
  }

  void update_u_gradient(value_t u, value_t v) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    if (cbinc.size() != cb.size())
      cbinc = zeros<complex_t>(cb.size(), cb.fullsize());

    this->v = rearrange(padded_dv, model.stride_size());
    cv = tbblas::fft(this->v, dimCount - 1, plan_v);

    // u part
    // dW += value_t(-1) * prod1 / v;

    cbinc += value_t(-1) / v * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(-1) / v);
    }
  }

  void update_v_gradient(value_t u, value_t v) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(cFinc.size());

    this->v = rearrange(padded_dv, model.stride_size());
    cv = tbblas::fft(this->v, dimCount - 1, plan_v);

    // v part
    // dW += u * prods1 / (v * v);

    cbinc = cbinc + u / (v * v) * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, u / (v * v));
    }

    ++_current_batch_size;
  }

  void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");

    if (deltab.size() != visible_size)
      deltab = zeros<value_t>(visible_size);

    if (!deltaF.size()) {
      deltaF.resize(model.filter_count());
      for (size_t k = 0; k < deltaF.size(); ++k)
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
    }

    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - kernel_size / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
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

      // Apply delta to current filters
      *cF[k] = *cF[k] - epsilon * *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    // TODO: masking of bias terms
    padded_f = ifft(cbinc, dimCount - 1, iplan_v);
    deltab = (momentum * deltab + padded_f / _current_batch_size);// * repeat(v_mask, deltab.size() / v_mask.size());
    cbinc = fft(deltab, dimCount - 1, plan_v);

    cb = cb - epsilon * cbinc;
    cbinc = zeros<complex_t>(cbinc.size(), cbinc.fullsize());

    _current_batch_size = 0;
    _host_updated = false;
  }

  void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");

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

    /*
      dW = dW / _current_batch_size + weightcost * W;
      db = db / _current_batch_size;

      dW2 = momentum * dW2 + (1.0 - momentum) * dW * dW;
      db2 = momentum * db2 + (1.0 - momentum) * db * db;

      // note that deltaW = - sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      W = W - sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      b = b - sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db;

      deltaW2 = momentum * deltaW2 + (1.0 - momentum) * sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW * sqrt(deltaW2 + epsilon) / sqrt(dW2 + epsilon) * dW;
      deltab2 = momentum * deltab2 + (1.0 - momentum) * sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db * sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * db;

      dW = zeros<value_t>(dW.size());
      db = zeros<value_t>(db.size());
     */

    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - kernel_size / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
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

      // Apply delta to current filters
      *cF[k] = *cF[k] - *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    // TODO: masking of bias terms
    padded_f = ifft(cbinc, dimCount - 1, iplan_v);
    padded_f = padded_f / _current_batch_size;

    db2 = momentum * db2 + (1.0 - momentum) * padded_f * padded_f;
    deltab = sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * padded_f;
    deltab2 = momentum * deltab2 + (1.0 - momentum) * deltab * deltab;
    cbinc = fft(deltab, dimCount - 1, plan_v);

    cb = cb - cbinc;
    cbinc = zeros<complex_t>(cbinc.size(), cbinc.fullsize());

    _current_batch_size = 0;
    _host_updated = false;
  }

  // Access to model data
  const proxy<tensor_t> visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return padded_v[seq<dimCount>(0), model.visibles_size()];
  }

  tensor_t& hiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return H;
  }

  tensor_t& hidden_deltas() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return dH;
  }

  void set_batch_length(int length) {
    if (length < 1 || model.filter_count() % length != 0)
      throw std::runtime_error("Filter count must be a multiple of filter batch length!");

    const bool reallocate = (length != _filter_batch_length);
    _filter_batch_length = length;

    if (reallocate) {
      if (_memory_allocated) {
        allocate_gpu_memory();
      }
    }
  }

  int batch_length() const {
    return _filter_batch_length;
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_LAYER_HPP_ */
