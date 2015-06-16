/*
 * dnn_layer.hpp
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DNN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_DNN_LAYER_HPP_

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
#include <tbblas/io.hpp>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/mult_sum.hpp>
#include <tbblas/deeplearn/repeat_mult.hpp>
#include <tbblas/deeplearn/repeat_mult_sum.hpp>
#include <tbblas/deeplearn/max_pooling.hpp>
#include <tbblas/deeplearn/avg_pooling.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>

#include <tbblas/deeplearn/dnn_layer_model.hpp>
#include <tbblas/deeplearn/objective_function.hpp>

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims, class Trainer, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class dnn_layer : public Trainer {
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

  typedef dnn_layer_model<value_t, dimCount> model_t;

  typedef tbblas::random_tensor2<value_t, 4, true, tbblas::uniform<value_t> > uniform_t;

public:
  enum flags_t {
    APPLY_NONLINEARITY = 1,
    APPLY_BIAS = 2,
    ACCUMULATE = 4,
    DROPOUT = 8,
    DEFAULT = APPLY_NONLINEARITY | APPLY_BIAS
  };

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  tensor_t _b, _binc;
  v_ctensor_t cF, cFinc;

  // Sizes
  dim_t padded_visible_size, visible_size, hidden_size,
        visible_layer_size, hidden_layer_size,
        hidden_layer_batch_size, visible_layer_batch_size,
        visible_batch_size, kernel_size, padded_kernel_size, hidden_topleft,
        patch_count, patch_size, patch_batch_size, patch_layer_size, patch_layer_batch_size,
        hidden_patch_layer_batch_size, step_size, step_count;

  tensor_t vp, vr, hr, hpr, pr, shifted_f, padded_f, padded_k, v_mask, h_mask;
  ctensor_t cvr, chr, chrdiff;
  plan_t plan_vr, iplan_vr, plan_hr, iplan_hr;
  uniform_t v_rand;

  tensor_t padded_v, H;

  boost::shared_ptr<stensor_t> _p_switches;
  stensor_t sr;

  int _filter_batch_length, _layer_voxel_count;
  bool _memory_allocated, _host_updated;
  value_t _current_batch_size, _dropout_rate, _sensitivity_ratio;

  tbblas::deeplearn::objective_function _objective_function;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  dnn_layer(model_t& model, const Trainer* parameters, dim_t patch_count = seq<dimCount>(1)) : Trainer(parameters),
    model(model), patch_count(patch_count),
    _p_switches(new stensor_t()),
    _filter_batch_length(1), _memory_allocated(false), _host_updated(true), _current_batch_size(0), _dropout_rate(0), _sensitivity_ratio(0.5)
  {
    _layer_voxel_count = model.visibles_count() / model.visibles_size()[dimCount - 1];
  }

private:
  dnn_layer(const dnn_layer&);

public:
  virtual ~dnn_layer() {
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

  void tie_switches(const cnn_layer<value_t, dimCount, Trainer, Enable>& layer) {
    _p_switches = layer._p_switches;
  }

  void set_dropout_rate(const value_t& rate) {
    if (rate < 0 || rate >= 1)
      throw std::runtime_error("Drop out rate must be in [0,1).");

    _dropout_rate = rate;
  }

  /// Transforms
  virtual void allocate_gpu_memory() {
    using namespace tbblas;

    stensor_t& _switches = *_p_switches;

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

    padded_v = zeros<value_t>(padded_visible_size);
    H = zeros<value_t>(hidden_size / model.pooling_size());

    if (model.has_pooling_layer()) {
      if (_switches.size() != H.size())
        _switches = zeros<switches_t>(H.size());
      hpr = zeros<value_t>(hidden_patch_layer_batch_size);
      pr = zeros<value_t>(hidden_patch_layer_batch_size / model.pooling_size());
      sr = zeros<value_t>(hidden_patch_layer_batch_size / model.pooling_size());
    }

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

//      _layer_voxel_count = sum(v_mask) * padded_visible_size[dimCount - 1];
    }

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(visible_layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
    } else {
      h_mask = ones<value_t>(visible_layer_size);
    }

    if (model.shared_bias()) {
      _b = sum(model.bias()) / model.visibles_count() * ones<value_t>(seq<dimCount>(1));
    } else {
      padded_v = zeros<value_t>(padded_visible_size);
      padded_v[seq<dimCount>(0), model.visibles_size()] = model.bias();
      _b = rearrange(padded_v, model.stride_size());
    }

    padded_v = zeros<value_t>(padded_visible_size);

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
      }
    }

    vr = zeros<value_t>(patch_size);
    cvr = fft(vr, dimCount - 1, plan_vr);
    vr = ifft(cvr, dimCount - 1, iplan_vr);

    hr = zeros<value_t>(patch_layer_batch_size);
    chr = fft(hr, dimCount - 1, plan_hr);
    hr = ifft(chr, dimCount - 1, iplan_hr);
  }

  virtual void write_model_to_host() {
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

        shifted_f = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());
        *model.filters()[k * _filter_batch_length + j] = padded_k[seq<dimCount>(0), model.kernel_size()];
      }
    }

    _b = _b * (abs(_b) > 1e-16);
    if (model.shared_bias()) {
      host_tensor_t b = ones<value_t>(model.visibles_size()) * _b[seq<dimCount>(0)];
      tbblas::synchronize();
      model.set_bias(b);
    } else {
      tensor_t padded = rearrange_r(_b, model.stride_size());
      host_tensor_t b = padded[seq<dimCount>(0), model.visibles_size()];
      tbblas::synchronize();
      model.set_bias(b);
    }
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);
    padded_v = (padded_v - model.mean()) / model.stddev() * repeat(v_mask, padded_v.size() / v_mask.size());
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);

    padded_v = ((padded_v * model.stddev()) + model.mean()) * repeat(v_mask, padded_v.size() / v_mask.size());
  }

  virtual void infer_visibles(int flags = DEFAULT) {
    stensor_t& _switches = *_p_switches;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(H.size() == hidden_size / model.pooling_size());

//    bool dropout = false, accumulateActivation = false;

    if (flags & ACCUMULATE) {
      if (flags & APPLY_BIAS) {
        if (model.shared_bias())
          vp += repeat(_b, visible_size / _b.size());
        else
          vp += _b;
      }
    } else {
      if (flags & APPLY_BIAS) {
        if (model.shared_bias())
          vp = repeat(_b, visible_size / _b.size());
        else
          vp = _b;
      } else {
        vp = zeros<value_t>(visible_size);
      }
    }

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);// for each subregion

      cvr = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));
      for (size_t k = 0; k < cF.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);

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

        chr = fft(hr, dimCount - 1, plan_hr);
        cvr += repeat_mult_sum(chr, *cF[k]);
      }
      vr = ifft(cvr, dimCount - 1, iplan_vr);
      vp[topleft, overlap_size] = vp[topleft, overlap_size] + vr[seq<dimCount>(0), overlap_size];
    }

    if (flags & APPLY_NONLINEARITY) {
      switch (model.activation_function()) {
        case activation_function::Sigmoid: vp = sigm(vp); break;
        case activation_function::ReLU:    vp = max(0.0, vp);  break;
        case activation_function::Linear:  break;
        default:
          throw std::runtime_error("Unsupported activation function.");
      }
    }

    if ((flags & DROPOUT) && _dropout_rate > 0) {
      if (v_rand.size() != vp.size())
        v_rand.resize(vp.size());

      vp = vp * (v_rand() > _dropout_rate) / (1. - _dropout_rate);
    }

    padded_v = rearrange_r(vp, model.stride_size());
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
        padded_v[seq<dimCount>(0), model.visibles_size()] =
            (visibles() - target) * visibles() * (1 + -visibles()) / (value_t)target.count();
        break;

      case activation_function::ReLU:
        padded_v[seq<dimCount>(0), model.visibles_size()] =
            (visibles() - target) * (visibles() > 0) / (value_t)target.count();
        break;

//      case activation_function::Softmax:
      case activation_function::Linear:
        padded_v[seq<dimCount>(0), model.visibles_size()] =
            (visibles() - target) / (value_t)target.count();

        // Apply normalization
//        padded_v = padded_v / model.stddev() * repeat(v_mask, padded_v.size() / v_mask.size());
        break;
      }
      break;

    case tbblas::deeplearn::objective_function::SenSpe:
      {
        if (model.activation_function() != activation_function::Sigmoid)
          throw std::runtime_error("Activation function for objective function 'Sensitivity + Specificity' must be 'Sigmoid'.");

        // delta = (-alpha* target - beta * target + beta) * f'(X)

        const value_t positive_ratio = sum(target) / (value_t)target.count();
        const value_t alpha = value_t(2) * _sensitivity_ratio / (positive_ratio + value_t(1e-8));
        const value_t beta =  value_t(2) * (value_t(1) - _sensitivity_ratio) / (value_t(1) - positive_ratio + value_t(1e-8));

        padded_v[seq<dimCount>(0), model.visibles_size()] =
            (alpha * target + beta * (value_t(1) - target)) * (visibles() - target) * visibles() * (value_t(1) - visibles()) / (value_t)target.count();
      }
      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_deltas(target).");
    }
  }

  /// Requires visible Ra
  void calculate_RDs(tensor_t& target, tensor_t& activation) {
    // this is used for the output layer
    // Needs target values as input
    // Need to know the objective function

    assert(target.size() == model.visibles_size());

    switch (_objective_function) {
    case tbblas::deeplearn::objective_function::SSD:

      // delta = (visible - target) * f'(X)
      // RDa = I*Ra * f'(X)
      switch (model.activation_function()) {
      case activation_function::Sigmoid:
        padded_v[seq<dimCount>(0), model.visibles_size()] = visibles() * activation * (1 - activation) / (value_t)target.count();
        break;

      case activation_function::ReLU:
        padded_v[seq<dimCount>(0), model.visibles_size()] = visibles() * (activation > 0) / (value_t)target.count();
        break;

//      case activation_function::Softmax:
      case activation_function::Linear:
        padded_v[seq<dimCount>(0), model.visibles_size()] = visibles() / (value_t)target.count();

        // Apply normalization
//        padded_v = padded_v / model.stddev() * repeat(v_mask, padded_v.size() / v_mask.size());
        break;
      }
      break;

    case tbblas::deeplearn::objective_function::SenSpe:
      {
        if (model.activation_function() != activation_function::Sigmoid)
          throw std::runtime_error("Activation function for objective function 'Sensitivity + Specificity' must be 'Sigmoid'.");

        // delta = (-alpha* target - beta * target + beta) * f'(X)

        const value_t positive_ratio = sum(target) / (value_t)target.count();
        const value_t alpha = value_t(2) * _sensitivity_ratio / (positive_ratio + value_t(1e-8));
        const value_t beta =  value_t(2) * (value_t(1) - _sensitivity_ratio) / (value_t(1) - positive_ratio + value_t(1e-8));

        padded_v[seq<dimCount>(0), model.visibles_size()] =
            (alpha * target + beta * (value_t(1) - target)) * visibles() * activation * (value_t(1) - activation) / (value_t)target.count();
      }
      break;

    default:
      throw std::runtime_error("Undefined objective function for calculate_deltas(target).");
    }
  }

  void backprop_hiddens() {
    stensor_t& _switches = *_p_switches;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(padded_v.size() == padded_visible_size);

    vp = rearrange(padded_v, model.stride_size());
    H = zeros<value_t>(hidden_size / model.pooling_size());

    // Iterate over sub-regions
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
        chr = conj_mult_sum(cvr, *cF[k]);
        hr = ifft(chr, dimCount - 1, iplan_hr);

        if (model.has_pooling_layer()) {
          hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size] = hr[hidden_topleft, hidden_overlap_layer_batch_size];

          switch (model.pooling_method()) {
          case pooling_method::StridePooling:
            pr = hpr[seq<dimCount>(0), model.pooling_size(), hpr.size()];
            break;

          case pooling_method::MaxPooling:
            sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
                _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
            pr = max_pooling(hpr, sr, model.pooling_size());
            break;

          case pooling_method::AvgPooling:
            pr = avg_pooling(hpr, model.pooling_size()) * (value_t)model.pooling_size().prod();
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
      }
    }
  }

  /// Takes hidden activation of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type >::type
  infer_visible_Ra(const Expression& hiddens) {

   assert(hiddens.size() == model.visibles_size());

   switch(model.activation_function()) {
   case activation_function::Sigmoid:
     padded_v[seq<dimCount>(0), model.visibles_size()] =
         visibles() * hiddens * (1.0 - hiddens);
     break;

   case activation_function::ReLU:
     padded_v[seq<dimCount>(0), model.visibles_size()] =
         visibles() * (hiddens > 0);
     break;

   case activation_function::Linear:
     break;

   default:
     throw std::runtime_error("Unsupported activation function.");
   }
   return 0;
  }

  /// Takes hidden deltas of successive layer as input
  template<class Expression>
  typename boost::enable_if<tbblas::is_expression<Expression>,
    typename boost::enable_if_c<Expression::dimCount == dimCount, int>::type >::type
  backprop_visible_deltas(const Expression& deltas) {

    assert(deltas.size() == model.visibles_size());

    switch(model.activation_function()) {
    case activation_function::Sigmoid:
      padded_v[seq<dimCount>(0), model.visibles_size()] =
          deltas * visibles() * (1 + -visibles());
      break;

    case activation_function::ReLU:
      padded_v[seq<dimCount>(0), model.visibles_size()] =
          deltas * (visibles() > 0);
      break;

    case activation_function::Linear:
      padded_v[seq<dimCount>(0), model.visibles_size()] =
          deltas;
      break;

    default:
      throw std::runtime_error("Unsupported activation function.");
    }
    return 0;
  }

  /// Requires visible deltas and hiddens
  virtual void update_gradient() {
    stensor_t& _switches = *_p_switches;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (_binc.size() != _b.size())
      _binc = zeros<value_t>(_b.size());

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
    }

    vp = rearrange(padded_v, model.stride_size());
    // changed the weighting from 1 / vp.count() to 1
    if (model.shared_bias()) {
      _binc = _binc + sum(vp);
    } else {
      _binc += vp;
    }

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

        chr = fft(hr, dimCount - 1, plan_hr);
        // TODO: changed the weighting to 1 from 1 / _layer_voxel_count
        *cFinc[k] += conj_repeat_mult(cvr, chr, value_t(1));
      }
    }

    ++_current_batch_size;
  }

  virtual size_t gradient_length() const {
    return model.kernel_size().prod() * cF.size() * _filter_batch_length + _b.count();
  }

  // Writes the gradient of the current layer to the vector gradient starting at offset. Returns the index of the last component + 1 (the new offset)
  virtual int collect_gradient(tensor<value_t, 1, true>& gradient, int offset, value_t weightcost) {
    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");

    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        cvr = (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);
        shifted_f = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        gradient[seq(offset), seq(model.kernel_size().prod())] = reshape(padded_k[seq<dimCount>(0), model.kernel_size()], seq(model.kernel_size().prod()));
        offset += model.kernel_size().prod();
      }
    }

    // TODO: masking of bias terms
    gradient[seq(offset), seq(_binc.size().prod())] = reshape(_binc / _current_batch_size, seq(_binc.size().prod()));
    offset += _binc.count();

    return offset;
  }

  virtual void reset_gradient() {
    if (!cFinc.size())
      return;

    for (size_t k = 0; k < cF.size(); ++k) {
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    _binc = zeros<value_t>(_binc.size());

    _current_batch_size = 0;
  }

  // Writes the gradient of the current layer to the vector gradient starting at offset. Returns the index of the last component + 1 (the new offset)
  virtual int get_parameters(tensor<value_t, 1, true>& parameters, int offset) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        cvr = (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);
        shifted_f = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        parameters[seq(offset), seq(model.kernel_size().prod())] = reshape(padded_k[seq<dimCount>(0), model.kernel_size()], seq(model.kernel_size().prod()));
        offset += model.kernel_size().prod();
      }
    }

    // TODO: masking of bias terms
    parameters[seq(offset), seq(_b.size().prod())] = reshape(_b, seq(_b.size().prod()));
    offset += _b.count();

    return offset;
  }

  virtual int get_weight_mask(tensor<value_t, 1, true>& weight_mask, int offset) {
    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {
        weight_mask[seq(offset), seq(model.kernel_size().prod())] = ones<value_t>(seq(model.kernel_size().prod()));
        offset += model.kernel_size().prod();
      }
    }

    // TODO: masking of bias terms
    weight_mask[seq(offset), seq(_b.size().prod())] = zeros<value_t>(seq(_b.size().prod()));
    offset += _b.count();

    return offset;
  }

  virtual int set_parameters(tensor<value_t, 1, true>& parameters, int offset) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!_memory_allocated)
      allocate_gpu_memory();


    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_kernel_size);
        padded_k[seq<dimCount>(0), model.kernel_size()] = reshape(parameters[seq(offset), seq(model.kernel_size().prod())], model.kernel_size());
        offset += model.kernel_size().prod();

        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cvr = fft(shifted_f, dimCount - 1, plan_vr);
        (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }
    }

    // TODO: masking of bias terms
    _b = reshape(parameters[seq(offset), seq(_b.size().prod())], _b.size());
    offset += _b.size().prod();

    return offset;
  }

  virtual int update_model(tensor<value_t, 1, true>& delta, int offset) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");


    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_kernel_size);
        padded_k[seq<dimCount>(0), model.kernel_size()] = reshape(delta[seq(offset), seq(model.kernel_size().prod())], model.kernel_size());
        offset += model.kernel_size().prod();

        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cvr = fft(shifted_f, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }

      // Apply delta to current filters
      *cF[k] = *cF[k] + *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    // TODO: masking of bias terms
    _b = _b + reshape(delta[seq(offset), seq(_b.size().prod())], _b.size());
    offset += _b.size().prod();

    _binc = zeros<value_t>(_binc.size());

    _current_batch_size = 0;
    _host_updated = false;

    return offset;
  }

  virtual void update_model(value_t weightcost) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");


    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        cvr = (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] * (value_t(1) / _current_batch_size) + weightcost * (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);
        shifted_f = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(shifted_f, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        // update deltaF
        this->update_delta(padded_k[seq<dimCount>(0), model.kernel_size()], iFilter);

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = reshape(this->delta(iFilter), model.kernel_size());
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cvr = fft(shifted_f, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }

      // Apply delta to current filters
      *cF[k] = *cF[k] + *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    // TODO: masking of bias terms
    this->update_delta(_binc / _current_batch_size, model.filter_count());
    _b = _b + reshape(this->delta(model.filter_count()), _b.size());

    _binc = zeros<value_t>(_binc.size());

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

#endif /* TBBLAS_DEEPLEARN_DNN_LAYER_HPP_ */
