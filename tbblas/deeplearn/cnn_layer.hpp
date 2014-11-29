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

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  v_ctensor_t cF, cb, cFinc, cbinc;

  // Sizes
  dim_t visible_size, hidden_size, size,
        visible_layer_size, hidden_layer_size, layer_size,
        hidden_layer_batch_size, layer_batch_size,
        filter_batch_size, hidden_topleft, hbMaskSize;

  tensor_t v, h, f, h_mask, H, dv, dH;
  ctensor_t cv, ch, chdiff;
  plan_t plan_v, iplan_v, plan_h, iplan_h;

  int _filter_batch_length, _voxel_count;
  bool _memory_allocated, _host_updated;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  cnn_layer(model_t& model) : model(model), _filter_batch_length(1),
      _memory_allocated(false), _host_updated(true)
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
      hidden_topleft = seq<4>(0);
    }

    H = zeros<value_t>(hidden_size);

#ifndef TBBLAS_CNN_NO_SELFTEST
    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(size);

      tensor_t A = v_noise, B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(layer_batch_size);

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

    cFinc.resize(cF.size());
    cbinc.resize(cb.size());

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
    } else {
      h_mask = ones<value_t>(layer_size);
    }

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
        cf = zeros<complex_t>(cf.size(), cf.fullsize());
        cFinc[k] = boost::make_shared<ctensor_t>(cf);

        h = zeros<value_t>(layer_batch_size);
        for (int j = 0; j < _filter_batch_length; ++j) {
          h[seq(0,0,0,j), visible_layer_size] = *model.bias()[k * _filter_batch_length + j];
        }
        ch = fft(h, dimCount - 1, plan_h);
        cb[k] = boost::make_shared<ctensor_t>(ch);
        ch = zeros<complex_t>(ch.size(), ch.fullsize());
        cbinc[k] = boost::make_shared<ctensor_t>(ch);
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

      h = ifft(*cb[k], dimCount - 1, iplan_h);
      h = h * (abs(h) > 1e-16);

      for (int j = 0; j < _filter_batch_length; ++j) {
        *model.bias()[k * _filter_batch_length + j] = h[seq(0,0,0,j),layer_size];
      }
    }
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);
    v = (v - model.mean()) / model.stddev();
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v.size() == visible_size);

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
      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = dH[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    dv = ifft(cv, dimCount - 1, iplan_v);

//    dV = prod(dH, trans(W));
  }

  void backprop_visibles() {
    assert(H.size() == hidden_size);

    cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));
    for (size_t k = 0; k < cF.size(); ++k) {
      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = H[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    v = ifft(cv, dimCount - 1, iplan_v);

//    V = prod(H, trans(W));
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

    cv = tbblas::fft(v, dimCount - 1, plan_v);

    for (size_t k = 0; k < cF.size(); ++k) {

      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = dH[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, value_t(1) / _voxel_count);
      *cbinc[k] = *cbinc[k] + value_t(1) / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
    }
  }

  void perform_momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {
    dim_t fullsize = cv.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {
        cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[3]), cv.size()];
        cv.set_fullsize(fullsize);
        f = ifft(cv, dimCount - 1, iplan_v);
        f = f * tbblas::mask<value_t>(f.size(), model.kernel_size());
        cv = fft(f, dimCount - 1, plan_v);
        (*cFinc[k])[seq(0,0,0,j*cv.size()[3]), cv.size()] = cv;
      }

      *cF[k] = *cF[k] - epsilon * *cFinc[k];
      *cb[k] = *cb[k] - epsilon * *cbinc[k];

      *cFinc[k] = momentum * *cFinc[k] + weightcost * *cF[k];
      *cbinc[k] = momentum * *cbinc[k];
    }

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
