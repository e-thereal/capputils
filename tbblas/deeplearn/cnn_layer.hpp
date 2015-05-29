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
#include <tbblas/sequence_iterator.hpp>
#include <tbblas/io.hpp>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/mult_sum.hpp>
#include <tbblas/deeplearn/repeat_mult.hpp>
#include <tbblas/deeplearn/repeat_mult_sum.hpp>
#include <tbblas/deeplearn/max_pooling.hpp>
#include <tbblas/deeplearn/avg_pooling.hpp>

#include <tbblas/deeplearn/cnn_layer_model.hpp>

#include <tbblas/deeplearn/opt/type_traits.hpp>
#include <tbblas/deeplearn/opt/void_trainer.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>
#include <cstdio>

#define CNN_LAYER_KEEP_HIDDENS

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims, class Trainer, class Enable>
class dnn_layer;

template<class T, unsigned dims, class Trainer, class Enable>
class cdnn_layer;

template<class T, unsigned dims, class Trainer = opt::void_trainer<T>, class Enable = typename boost::enable_if<opt::is_trainer<Trainer > >::type>
class cnn_layer : public Trainer
{

  friend class dnn_layer<T, dims, Trainer, Enable>;
  friend class cdnn_layer<T, dims, Trainer, Enable>;

  const static unsigned dimCount = dims;
  typedef uint8_t switches_t;
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

  typedef tensor<switches_t, dimCount, true> stensor_t;

  typedef cnn_layer_model<value_t, dimCount> model_t;

  static const value_t tolerance = 1e-8;

  typedef tbblas::random_tensor2<value_t, 4, true, tbblas::uniform<value_t> > uniform_t;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  v_ctensor_t cF, cFinc;
  v_tensor_t _b, _binc;

  // Sizes
  dim_t padded_visible_size, visible_size, hidden_size,
        visible_layer_size, hidden_layer_size,
        hidden_layer_batch_size, visible_layer_batch_size,
        visible_batch_size, kernel_size, padded_kernel_size,
        hidden_topleft,
        patch_count, patch_size, patch_batch_size, patch_layer_size, patch_layer_batch_size,
        hidden_patch_layer_batch_size, step_size, step_count;

  tensor_t vp, vr, hr, hpr, pr, padded_f, padded_k, v_mask, h_mask;
  ctensor_t cvr, chr, chrdiff;
  plan_t plan_vr, iplan_vr, plan_hr, iplan_hr;
  uniform_t hr_rand;

  tensor_t padded_v, H;
#ifdef CNN_LAYER_KEEP_HIDDENS
  tensor_t _pooled, dH;
#endif

  boost::shared_ptr<stensor_t> _p_switches;
  stensor_t &_switches, sr;

  int _filter_batch_length, _voxel_count;
  bool _memory_allocated, _host_updated;
  value_t _current_batch_size, _dropout_rate, _current_iteration;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  cnn_layer(model_t& model, dim_t patch_count = seq<dimCount>(1)) : Trainer(NULL), model(model),
      _p_switches(new stensor_t()), _switches(*_p_switches),
      patch_count(patch_count), _filter_batch_length(1),
      _memory_allocated(false), _host_updated(true), _current_batch_size(0), _dropout_rate(0), _current_iteration(0)
  {
    _voxel_count = model.visibles_count();
  }

  cnn_layer(model_t& model, const Trainer* parameters, dim_t patch_count = seq<dimCount>(1)) : Trainer(parameters), model(model),
      _p_switches(new stensor_t()), _switches(*_p_switches),
      patch_count(patch_count), _filter_batch_length(1),
      _memory_allocated(false), _host_updated(true), _current_batch_size(0), _dropout_rate(0), _current_iteration(0)
  {
    _voxel_count = model.visibles_count();
  }

private:
  cnn_layer(const cnn_layer&, const Trainer*);

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
    visible_size = (model.visibles_size() + model.stride_size() - 1) / model.stride_size();
    kernel_size = (model.kernel_size() + model.stride_size() - 1) / model.stride_size();
    padded_visible_size = visible_size * model.stride_size();
    padded_kernel_size = kernel_size * model.stride_size();
    kernel_size[dimCount - 1] = visible_size[dimCount - 1] = model.visibles_size()[dimCount - 1] * model.stride_size().prod();
    hidden_size = model.hiddens_size();

    // TODO: Choose patch_size such that step_size is a multiple of pooling size
    step_size = min((hidden_size + patch_count - 1) / patch_count, hidden_size);
    step_size[dimCount - 1] = 1;

    if (model.has_pooling_layer()) {
      step_size = min((step_size + model.pooling_size() - 1) / model.pooling_size() * model.pooling_size(), hidden_size);
    }

    patch_size = step_size + kernel_size - 1;
//    patch_size = min((visible_size + patch_count - 1) / patch_count + kernel_size - 1, visible_size);
    patch_layer_size = patch_layer_batch_size = patch_batch_size = patch_size;

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

    hidden_patch_layer_batch_size = patch_size - kernel_size + 1;
    hidden_patch_layer_batch_size[dimCount - 1] = _filter_batch_length;

    step_count = (hidden_layer_size + step_size - 1) / step_size;

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
    _b.resize(model.bias().size() / _filter_batch_length);

    {
      dim_t padded_mask_size = padded_visible_size;
      dim_t mask_size = model.visibles_size();
      mask_size[dimCount - 1] = padded_mask_size[dimCount - 1] = 1;

      v_mask = zeros<value_t>(padded_mask_size);
      v_mask[seq<dimCount>(0), mask_size] = ones<value_t>(mask_size);
    }

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(visible_layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
    } else {
      h_mask = ones<value_t>(visible_layer_size);
    }

    padded_v = zeros<value_t>(padded_visible_size);

#ifdef CNN_LAYER_KEEP_HIDDENS
    H = zeros<value_t>(hidden_size);
#else
    H = zeros<value_t>(hidden_size / model.pooling_size());
#endif

    if (model.has_pooling_layer()) {
      _switches = zeros<switches_t>(H.size());

      hpr = zeros<value_t>(hidden_patch_layer_batch_size);
      pr = zeros<value_t>(hidden_patch_layer_batch_size / model.pooling_size());
      sr = zeros<value_t>(hidden_patch_layer_batch_size / model.pooling_size());
    }

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
      plan_t plan_f;

      padded_k = zeros<value_t>(padded_kernel_size);

      for (size_t k = 0; k < cF.size(); ++k) {
        // Created padded tensor for an entire filter batch
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

        // Shift the kernel to be centered around 0 and not the image center
        f = ifftshift(pad, dimCount - 1);
        cf = fft(f, dimCount - 1, plan_f);
        cF[k] = boost::make_shared<ctensor_t>(cf);

        if (model.shared_bias()) {
          h = zeros<value_t>(visible_layer_batch_size / visible_layer_size);
          for (int j = 0; j < _filter_batch_length; ++j) {
            h[seq(0,0,0,j), seq<dimCount>(1)] = ones<value_t>(seq<dimCount>(1)) * (*model.bias()[k * _filter_batch_length + j])[hidden_topleft];
          }
        } else {
          h = zeros<value_t>(visible_layer_batch_size);
          for (int j = 0; j < _filter_batch_length; ++j) {
            h[seq(0,0,0,j), visible_layer_size] = *model.bias()[k * _filter_batch_length + j];
          }
        }
        _b[k] = boost::make_shared<tensor_t>(h);
      }
    }

    // Allocate cvr vr, chr hr
//    vr = zeros<value_t>(patch_size);
//    cvr = fft(vr, dimCount - 1, plan_vr);
//    vr = ifft(cvr, dimCount - 1, iplan_vr);
//
//    hr = zeros<value_t>(patch_layer_batch_size);
//    chr = fft(hr, dimCount - 1, plan_hr);
//    hr = ifft(chr, dimCount - 1, iplan_hr);
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract one filter from the filter batch
        dim_t topleft = patch_size / 2 - kernel_size / 2;
        cvr = (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);

        vr = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(vr, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());
        *model.filters()[k * _filter_batch_length + j] = padded_k[seq<dimCount>(0), model.kernel_size()];
      }
      *_b[k] = *_b[k] * (abs(*_b[k]) > 1e-16);

      if (model.shared_bias()) {
        for (int j = 0; j < _filter_batch_length; ++j) {
          *model.bias()[k * _filter_batch_length + j] = ones<value_t>(visible_layer_size) * (*_b[k])[seq(0,0,0,j)];
        }
      } else {
        for (int j = 0; j < _filter_batch_length; ++j) {
          *model.bias()[k * _filter_batch_length + j] = (*_b[k])[seq(0,0,0,j), visible_layer_size];
        }
      }
    }
    tbblas::synchronize();
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);
    padded_v = (padded_v - model.mean()) / model.stddev() * repeat(v_mask, padded_v.size() / v_mask.size());
  }

  void infer_hiddens(bool dropout = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);

    vp = rearrange(padded_v, model.stride_size());

    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t overlap_layer_size = min(patch_layer_size, visible_layer_size - topleft);
      dim_t overlap_layer_batch_size = min(patch_layer_batch_size, visible_layer_batch_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);

      if (model.has_pooling_layer())
        assert(hidden_overlap_layer_batch_size % model.pooling_size() == seq<dimCount>(0));

      vr = zeros<value_t>(patch_size);
      vr[seq<dimCount>(0), overlap_size] = vp[topleft, overlap_size];
      cvr = fft(vr, dimCount - 1, plan_vr);

      for (size_t k = 0; k < cF.size(); ++k) {
        chr = conj_mult_sum(cvr, *cF[k]);

        hr = ifft(chr, dimCount - 1, iplan_hr);
        if (model.shared_bias()) {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] + repeat(*_b[k], overlap_layer_batch_size / _b[k]->size());  // apply sub-region of bias terms
        } else {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] + (*_b[k])[topleft, overlap_layer_batch_size];  // apply sub-region of bias terms
        }

        switch (model.activation_function()) {
          case activation_function::Sigmoid: hr = sigm(hr); break;
          case activation_function::ReLU:    hr = max(0.0, hr);  break;
          case activation_function::Linear:  break;
          default:
            throw std::runtime_error("Unsupported activation function.");
        }

        if (dropout && _dropout_rate > 0) {
          if (hr_rand.size() != hr.size())
            hr_rand.resize(hr.size());

          hr = hr * (hr_rand() > _dropout_rate) / (1. - _dropout_rate);
        }

#ifndef CNN_LAYER_KEEP_HIDDENS
        if (model.has_pooling_layer()) {
          hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size] = hr[hidden_topleft, hidden_overlap_layer_batch_size];

          switch (model.pooling_method()) {
          case pooling_method::StridePooling:
            pr = hpr[seq<dimCount>(0), model.pooling_size(), hpr.size()];
            break;

          case pooling_method::MaxPooling:
            sr = get_max_pooling_switches(hpr, model.pooling_size());
            pr = max_pooling(hpr, sr, model.pooling_size());
            _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()] =
                sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()];
            break;

          case pooling_method::AvgPooling:
            pr = avg_pooling(hpr, model.pooling_size());
            break;

          default:
            throw std::runtime_error("Unsupported pooling method.");
          }

          H[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()] =
              pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()];
        } else {
          H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size] =
              hr[hidden_topleft, hidden_overlap_layer_batch_size];
        }
#else
        H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size] =
            hr[hidden_topleft, hidden_overlap_layer_batch_size];
#endif
      }
    }
#ifdef CNN_LAYER_KEEP_HIDDENS
    if (model.has_pooling_layer()) {

      switch (model.pooling_method()) {
      case pooling_method::StridePooling:
        _pooled = H[seq<dimCount>(0), model.pooling_size(), H.size()];
        break;

      case pooling_method::MaxPooling:
        _switches = get_max_pooling_switches(H, model.pooling_size());
        _pooled = max_pooling(H, _switches, model.pooling_size());
        break;

      case pooling_method::AvgPooling:
        _pooled = avg_pooling(H, model.pooling_size());
        break;

      default:
        throw std::runtime_error("Unsupported pooling method.");
      }
    }
#endif
  }

//  /// Requires hidden activation and hidden total activation
//  void calculate_deltas(tensor_t& target) {
//    // this is used for the output layer
//    // Needs target values as input
//    // Need to know the objective function
//
//    // delta = (hidden - target) * f'(X)
//    switch (model.activation_function()) {
//    case activation_function::Sigmoid:
//      H = (H - target) * H * (1 + -H);
//      break;
//
//    case activation_function::ReLU:
//      H = (H - target) * (H > 0);
//      break;
//
//    case activation_function::Softmax:
//    case activation_function::Linear:
//      H = H - target;
//      break;
//
//    default:
//      throw std::runtime_error("Undefined objective function for cnn_layer::calculate_deltas(target).");
//    }
//  }

  void backprop_visibles() {
#ifdef CNN_LAYER_KEEP_HIDDENS
    assert(H.size() == hidden_size);

    if (model.has_pooling_layer()) {
      switch (model.pooling_method()) {
      case pooling_method::StridePooling:
        H = zeros<value_t>(H.size());
        H[seq<dimCount>(0), model.pooling_size(), H.size()] = _pooled;
        break;

      case pooling_method::MaxPooling:
        H = unpooling(_pooled, _switches, model.pooling_size());
        break;

      case pooling_method::AvgPooling:
        H = unpooling(_pooled, model.pooling_size());
        break;

      default:
        throw std::runtime_error("Unsupported pooling method.");
      }
    }
#else
    assert(H.size() == hidden_size / model.pooling_size());
#endif

    vp = zeros<value_t>(visible_size);

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);// for each subregion

      cvr = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = 0; k < cF.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);

#ifdef CNN_LAYER_KEEP_HIDDENS
      hr[hidden_topleft, hidden_overlap_layer_batch_size] = H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
#else
        if (model.has_pooling_layer()) {
          pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              H[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];

          switch (model.pooling_method()) {
          case pooling_method::StridePooling:
            hpr = zeros<value_t>(hpr.size());
            hpr[seq<dimCount>(0), model.pooling_size(), hpr.size()] = pr;
            break;

          case pooling_method::MaxPooling:
            sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
                _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
            hpr = unpooling(pr, sr, model.pooling_size());
            break;

          case pooling_method::AvgPooling:
            hpr = unpooling(pr, model.pooling_size());
            break;

          default:
            throw std::runtime_error("Unsupported pooling method.");
          }

          hr[hidden_topleft, hidden_overlap_layer_batch_size] = hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size];
        } else {
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        }
#endif

        chr = fft(hr, dimCount - 1, plan_hr);
        cvr += repeat_mult_sum(chr, *cF[k]);
      }
      vr = ifft(cvr, dimCount - 1, iplan_vr);
      vp[topleft, overlap_size] = vp[topleft, overlap_size] + vr[seq<dimCount>(0), overlap_size];
    }
    padded_v = rearrange_r(vp, model.stride_size());
    padded_v = padded_v * repeat(v_mask, padded_v.size() / v_mask.size());
  }

  /// Takes visible deltas of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type >::type
  backprop_hidden_deltas(const Expression& deltas, bool accumulate = false) {
    assert(deltas.size() == hiddens().size());

#ifdef CNN_LAYER_KEEP_HIDDENS
    if (!accumulate)
      dH = zeros<value_t>(H.size());


    if (model.has_pooling_layer()) {
      switch (model.pooling_method()) {
      case pooling_method::StridePooling:
        dH[seq<dimCount>(0), model.pooling_size(), H.size()] = dH[seq<dimCount>(0), model.pooling_size(), H.size()] + deltas;
        break;

      case pooling_method::MaxPooling:
        _pooled = deltas;
        dH = dH + unpooling(_pooled, _switches, model.pooling_size());
        break;

      case pooling_method::AvgPooling:
        _pooled = deltas;
        dH = dH + unpooling(_pooled, model.pooling_size());
        break;

      default:
        throw std::runtime_error("Unsupported pooling method.");
      }
    } else {
      dH += deltas;
    }

    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      H = dH * H * (1 + -H);
      break;

    case activation_function::ReLU:
      H = dH * (H > 0);
      break;

    case activation_function::Linear:
      H = dH;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
#else
    assert(accumulate == false);
    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      hiddens() = deltas * hiddens() * (1 + -hiddens());
      break;

    case activation_function::ReLU:
      hiddens() = deltas * (hiddens() > 0);
      break;

    case activation_function::Linear:
      hiddens() = deltas;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
#endif

    return 0;
  }

  /// Requires hidden deltas and visibles
  void update_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    if (!_binc.size()) {
      _binc.resize(_b.size());
      for (size_t i = 0; i < _binc.size(); ++i)
        _binc[i] = boost::make_shared<tensor_t>(zeros<value_t>(_b[i]->size()));
    }

    vp = rearrange(padded_v, model.stride_size());

    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t overlap_layer_size = min(patch_layer_size, visible_layer_size - topleft);
      dim_t overlap_layer_batch_size = min(patch_layer_batch_size, visible_layer_batch_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);

      vr = zeros<value_t>(patch_size);
      vr[seq<dimCount>(0), overlap_size] = vp[topleft, overlap_size];
      cvr = tbblas::fft(vr, dimCount - 1, plan_vr);

      for (size_t k = 0; k < cF.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);

#ifdef CNN_LAYER_KEEP_HIDDENS
      hr[hidden_topleft, hidden_overlap_layer_batch_size] = H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
#else
        if (model.has_pooling_layer()) {
          pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              H[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];

          switch (model.pooling_method()) {
          case pooling_method::StridePooling:
            hpr = zeros<value_t>(hpr.size());
            hpr[seq<dimCount>(0), model.pooling_size(), hpr.size()] = pr;
            break;

          case pooling_method::MaxPooling:
            sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
                _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
            hpr = unpooling(pr, sr, model.pooling_size());
            break;

          case pooling_method::AvgPooling:
            hpr = unpooling(pr, model.pooling_size());
            break;

          default:
            throw std::runtime_error("Unsupported pooling method.");
          }

          hr[hidden_topleft, hidden_overlap_layer_batch_size] = hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size];
        } else {
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        }
#endif

        chr = fft(hr, dimCount - 1, plan_hr);
        *cFinc[k] += conj_repeat_mult(cvr, chr, value_t(1) / _voxel_count);

        if (model.shared_bias()) {
          for (int j = 0; j < _filter_batch_length; ++j) {
            (*_binc[k])[seq(0,0,0,j), seq<dimCount>(1)] =
                (*_binc[k])[seq(0,0,0,j), seq<dimCount>(1)] + sum(hr[seq<dimCount>(0), overlap_layer_batch_size]) / visible_layer_size.prod();
          }
        } else {
          (*_binc[k])[topleft, overlap_layer_batch_size] =
              (*_binc[k])[topleft, overlap_layer_batch_size] + hr[seq<dimCount>(0), overlap_layer_batch_size];
        }
      }
    }

    ++_current_batch_size;
  }

  // trainer will have an update method to pass on the step parameters
  // The update method will call the step method from the model
  // The step method from the model will call the get step method from the trainer
  // Not a good idea. Parameters will be passed using functions.

  void update_model(value_t weightcost) {
    if (!cFinc.size() || !_binc.size())
      throw std::runtime_error("No gradient calculated.");

    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        cvr = (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);
        vr = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(vr, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        // Update delta given the current gradient
        this->update_delta(padded_k[seq<dimCount>(0), model.kernel_size()], iFilter);

        // Put delta into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = reshape(this->delta(iFilter), model.kernel_size());
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        vr = ifftshift(padded_f, dimCount - 1);

        cvr = fft(vr, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }

      this->update_delta(*_binc[k] / _current_batch_size, model.filter_count() + k);

      // Apply delta to current filters
      *cF[k] = *cF[k] - *cFinc[k];
      *_b[k] = *_b[k] - reshape(this->delta(model.filter_count() + k), _b[k]->size());

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *_binc[k] = zeros<value_t>(_binc[k]->size());
    }

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

    if (model.has_pooling_layer())
      return _pooled;
    else
      return H;
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

template<class T, unsigned dims, class Trainer, class Enable>
const T cnn_layer<T, dims, Trainer, Enable>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CNN_LAYER_HPP_ */
