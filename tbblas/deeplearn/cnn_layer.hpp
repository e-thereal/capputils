/*
 * cnn_layer.hpp
 *
 *  Created on: Aug 19, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CNN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_CNN_LAYER_HPP_

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

#include <tbblas/deeplearn/cnn_layer_model.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class cnn_layer {
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

  typedef cnn_layer_model<value_t, dimCount> model_t;

  static const value_t tolerance = 1e-8;

  typedef tbblas::random_tensor2<value_t, 4, true, tbblas::uniform<value_t> > uniform_t;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  v_ctensor_t cF, cb, cFinc, cbinc;
  v_tensor_t deltaF, deltab, dF2, db2, deltaF2, deltab2;

  // Sizes
  dim_t visible_size, hidden_size,
        visible_layer_size, hidden_layer_size,
        hidden_layer_batch_size, visible_layer_batch_size,
        visible_batch_size, hidden_topleft, hbMaskSize;

  tensor_t v, h, shifted_f, padded_f, h_mask, H, dv, dH;
  ctensor_t cv, ch, chdiff;
  plan_t plan_v, iplan_v, plan_h, iplan_h;
  uniform_t h_rand;

  int _filter_batch_length, _voxel_count;
  bool _memory_allocated, _host_updated;
  value_t _current_batch_size, _dropout_rate;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  cnn_layer(model_t& model) : model(model), _filter_batch_length(1),
      _memory_allocated(false), _host_updated(true), _current_batch_size(0), _dropout_rate(0)
  {
    _voxel_count = model.visibles_count();
  }

private:
  cnn_layer(const cnn_layer&);

public:
  virtual ~cnn_layer() {
    if (!_host_updated)
      write_model_to_host();
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
    visible_size = model.visibles_size();
    hidden_size = model.hiddens_size();

    visible_batch_size = visible_layer_batch_size = visible_layer_size = visible_size;
    hidden_layer_size = hidden_layer_batch_size = hidden_size;
    hidden_layer_size[dimCount - 1] = visible_layer_size[dimCount - 1] = 1;
    visible_batch_size[dimCount - 1] = visible_size[dimCount - 1] * _filter_batch_length;
    visible_layer_batch_size[dimCount - 1] = _filter_batch_length;
    hidden_layer_batch_size = hidden_layer_size * seq(1,1,1,_filter_batch_length);

    if (model.convolution_type() == convolution_type::Valid){
      hidden_topleft = model.kernel_size() / 2;
      hidden_topleft[dimCount - 1] = 0;
    } else {
      hidden_topleft = seq<4>(0);
    }

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
    cb.resize(model.bias().size() / _filter_batch_length);

//    cFinc.resize(cF.size());
//    cbinc.resize(cb.size());

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(visible_layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
    } else {
      h_mask = ones<value_t>(visible_layer_size);
    }

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
      plan_t plan_f;
      for (size_t k = 0; k < cF.size(); ++k) {
        // Created padded tensor for an entire filter batch
        pad = zeros<value_t>(visible_batch_size);

        // Insert kernels into padded tensor
        for (size_t j = 0; j < _filter_batch_length; ++j) {
          // Copy kernel to GPU
          kern = *model.filters()[k * _filter_batch_length + j];

          // Place kernel in the right spot (image center plus appropriate channel)
          dim_t topleft = visible_size / 2 - kern.size() / 2;
          topleft[dimCount - 1] = j * visible_size[dimCount - 1];
          pad[topleft, kern.size()] = kern;
        }

        // Shift the kernel to be centered around 0 and not the image center
        f = ifftshift(pad, dimCount - 1);
        cf = fft(f, dimCount - 1, plan_f);
        cF[k] = boost::make_shared<ctensor_t>(cf);
//        cf = zeros<complex_t>(cf.size(), cf.fullsize());
//        cFinc[k] = boost::make_shared<ctensor_t>(cf);

        h = zeros<value_t>(visible_layer_batch_size);
        for (int j = 0; j < _filter_batch_length; ++j) {
          h[seq(0,0,0,j), visible_layer_size] = *model.bias()[k * _filter_batch_length + j];
        }
        ch = fft(h, dimCount - 1, plan_h);
        cb[k] = boost::make_shared<ctensor_t>(ch);
//        ch = zeros<complex_t>(ch.size(), ch.fullsize());
//        cbinc[k] = boost::make_shared<ctensor_t>(ch);
      }
    }

    hbMaskSize = cb[0]->size();
    if (model.shared_bias()) {
      hbMaskSize[0] = 1;
      hbMaskSize[1] = 1;
      hbMaskSize[2] = 1;
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
        dim_t topleft = visible_size / 2 - model.kernel_size() / 2;
        cv = (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
        cv.set_fullsize(fullsize);
        shifted_f = ifft(cv, dimCount - 1, iplan_v);
        padded_f = fftshift(shifted_f, dimCount - 1);

        *model.filters()[k * _filter_batch_length + j] = padded_f[topleft, model.kernel_size()];
      }

      h = ifft(*cb[k], dimCount - 1, iplan_h);
      h = h * (abs(h) > 1e-16);

      for (int j = 0; j < _filter_batch_length; ++j) {
        *model.bias()[k * _filter_batch_length + j] = h[seq(0,0,0,j),visible_layer_size];
      }
    }
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);
    v = (v - model.mean()) / model.stddev();
  }

  void infer_hiddens(bool dropout = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);

//    if (dropout && _dropout_rate > 0) {
//      dim_t h_size = seq(V.size()[0], W.size()[1]);
//      if (h_rand.size() != h_size)
//        h_rand.resize(h_size);
//
//      h_drop = h_rand() > _dropout_rate;
//    }

    cv = fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cF.size(); ++k) {
      ch = conj_mult_sum(cv, *cF[k]);

      ch = ch + *cb[k];
      h = ifft(ch, dimCount - 1, iplan_h);

      switch (model.activation_function()) {
        case activation_function::Sigmoid: h = sigm(h); break;
        case activation_function::ReLU:    h = max(0.0, h);  break;
        case activation_function::Linear:  break;
        default:
          throw std::runtime_error("Unsupported activation function.");
      }

      if (dropout && _dropout_rate > 0) {
        if (h_rand.size() != h.size())
          h_rand.resize(h.size());

        h = h * (h_rand() > _dropout_rate) / (1. - _dropout_rate);
      }

      H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
  }

  /// Requires hidden activation and hidden total activation
  void calculate_deltas(tensor_t& target) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    // delta = (hidden - target) * f'(X)
    switch (model.activation_function()) {
    case activation_function::Sigmoid:
      dH = (H - target) * H * (1 + -H);
      break;

    case activation_function::ReLU:
      dH = (H - target) * (H > 0);
      break;

    case activation_function::Softmax:
    case activation_function::Linear:
      dH = H - target;
      break;
    }
  }

  void backprop_visible_deltas() {
    // will be called by the previous layer

    assert(dH.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));
    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = dH[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    dv = ifft(cv, dimCount - 1, iplan_v);
  }

  void backprop_visibles() {
    assert(H.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));
    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    v = ifft(cv, dimCount - 1, iplan_v);
  }

  /// Takes visible deltas of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type >::type
  backprop_hidden_deltas(const Expression& deltas) {
    assert(deltas.size() == hidden_size);

    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      dH = deltas * H * (1 + -H);
      break;

    case activation_function::ReLU:
      dH = deltas * (H > 0);
      break;

    case activation_function::Linear:
      dH = deltas;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
    return 0;
  }

//  void init_gradient_updates(value_t epsilon, value_t momentum, value_t weightcost) {
//    for (size_t k = 0; k < cF.size(); ++k) {
//      *cFinc[k] = momentum * *cFinc[k] + epsilon * weightcost * *cF[k];
//      *cbinc[k] = momentum * *cbinc[k];
//    }
//  }

  /// Requires hidden deltas and visibles
  void update_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    if (!cbinc.size()) {
      cbinc.resize(cb.size());
      for (size_t i = 0; i < cbinc.size(); ++i)
        cbinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cb[i]->size(), cb[i]->fullsize()));
    }

    cv = tbblas::fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(visible_layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = dH[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(1) / _voxel_count);
      *cbinc[k] = *cbinc[k] + value_t(1) / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
    }

    ++_current_batch_size;
  }

  void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {
    dim_t fullsize = cv.fullsize();

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size() || !cbinc.size())
      throw std::runtime_error("No gradient calculated.");

    if (!deltaF.size()) {
      deltaF.resize(model.filter_count());
      for (size_t k = 0; k < deltaF.size(); ++k)
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
    }

    if (!deltab.size()) {
      deltab.resize(cbinc.size());
      for (size_t k = 0; k < deltab.size(); ++k)
        deltab[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
    }

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - model.kernel_size() / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
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

      h = ifft(*cbinc[k], dimCount - 1, iplan_h);
      *deltab[k] = momentum * *deltab[k] + h / _current_batch_size;
      *cb[k] = fft(*deltab[k], dimCount - 1, plan_h);

      // Apply delta to current filters
      *cF[k] = *cF[k] - epsilon * *cFinc[k];
      *cb[k] = *cb[k] - epsilon * *cbinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *cbinc[k] = zeros<complex_t>(cbinc[k]->size(), cbinc[k]->fullsize());
    }

    _current_batch_size = 0;
    _host_updated = false;
  }

  void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {
    dim_t fullsize = cv.fullsize();

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size() || !cbinc.size())
      throw std::runtime_error("No gradient calculated.");

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

    if (!db2.size()|| !deltab.size() || !deltab2.size()) {
      db2.resize(cbinc.size());
      deltab.resize(cbinc.size());
      deltab2.resize(cbinc.size());
      for (size_t k = 0; k < db2.size(); ++k) {
        db2[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
        deltab[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
        deltab2[k] = boost::make_shared<tensor_t>(zeros<value_t>(visible_layer_batch_size));
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

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = visible_size / 2 - model.kernel_size() / 2;

        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cv.size()[dimCount - 1]), cv.size()];
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

      h = ifft(*cbinc[k], dimCount - 1, iplan_h);
      h = h / _current_batch_size;

      *db2[k] = momentum * *db2[k] + (1.0 - momentum) * h * h;
      *deltab[k] = sqrt(*deltab2[k] + epsilon) / sqrt(*db2[k] + epsilon) * h;
      *deltab2[k] = momentum * *deltab2[k] + (1.0 - momentum) * *deltab[k] * *deltab[k];

      *cb[k] = fft(*deltab[k], dimCount - 1, plan_h);

      // Apply delta to current filters
      *cF[k] = *cF[k] - *cFinc[k];
      *cb[k] = *cb[k] - *cbinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *cbinc[k] = zeros<complex_t>(cbinc[k]->size(), cbinc[k]->fullsize());
    }

    _current_batch_size = 0;
    _host_updated = false;
  }

  // Access to model data
  tensor_t& visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return v;
  }

  tensor_t& hiddens() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return H;
  }

  const tensor_t& visible_deltas() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return dv;
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

template<class T, unsigned dims>
const T cnn_layer<T, dims>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_LAYER_HPP_ */
