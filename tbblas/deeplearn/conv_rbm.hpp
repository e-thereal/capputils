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
#include <tbblas/sequence_iterator.hpp>

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
#include <tbblas/deeplearn/max_pooling.hpp>

#include <tbblas/deeplearn/conv_rbm_model.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

#include <cstdio>

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

  typedef tbblas::random_tensor2<value_t, dimCount, true, tbblas::uniform<value_t> > uniform_t;
  typedef tbblas::random_tensor2<value_t, dimCount, true, tbblas::normal<value_t> > normal_t;

  typedef conv_rbm_model<value_t, dimCount> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory (for shared biases, the biases are of size 1x1x1x... and need to be repeated to fit the image size
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
        patch_count, patch_size, patch_batch_size, patch_layer_size, patch_layer_batch_size,
        hidden_patch_layer_batch_size, step_size, step_count;
//        vbMaskSize, hbMaskSize, spMaskSize;

  // one element per thread
  tensor_t v, vr, hr, hpr, pr, padded_f, padded_k, v_mask, h_mask;
  ctensor_t cvr, chr, chdiff;
  plan_t plan_vr, iplan_vr, plan_hr, iplan_hr;
  uniform_t v_rand, h_rand, hr_rand;
  normal_t v_noise, hr_noise;

  tensor_t _visibles, _hiddens;  // interface tensors
  tensor<switches_t, dimCount, true> _switches, sr;

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

    std::cout << "Start allocating convRBM memory." << std::endl;

    // Prepare sizes
    visible_size = (model.visibles_size() + model.stride_size() - 1) / model.stride_size();
    kernel_size = (model.kernel_size() + model.stride_size() - 1) / model.stride_size();
    padded_visible_size = visible_size * model.stride_size();
    padded_kernel_size = kernel_size * model.stride_size();
    kernel_size[dimCount - 1] = visible_size[dimCount - 1] = model.visibles_size()[dimCount - 1] * model.stride_size().prod();
    hidden_size = model.hiddens_size();

    // TODO: Choose patch_size such that step_size is a multiple of pooling size
//    step_size = patch_size - kernel_size + 1;
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

    if (model.shared_bias()) {
      _b = sum(model.visible_bias()) / model.visibles_count() * ones<value_t>(seq<dimCount>(1));
    } else {
      _visibles = zeros<value_t>(padded_visible_size);
      _visibles[seq<dimCount>(0), model.visibles_size()] = model.visible_bias();
      _b = rearrange(_visibles, model.stride_size());
    }

    _visibles = zeros<value_t>(padded_visible_size);
    _hiddens = zeros<value_t>(hidden_size / model.pooling_size());
    _switches = zeros<switches_t>(_hiddens.size());
    hpr = zeros<value_t>(hidden_patch_layer_batch_size);
    pr = zeros<value_t>(hidden_patch_layer_batch_size / model.pooling_size());
    sr = zeros<value_t>(hidden_patch_layer_batch_size / model.pooling_size());

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

        if (model.shared_bias()) {
          h = zeros<value_t>(visible_layer_batch_size / visible_layer_size);
          for (int j = 0; j < _filter_batch_length; ++j) {
            h[seq(0,0,0,j), seq<dimCount>(1)] = ones<value_t>(seq<dimCount>(1)) * (*model.hidden_bias()[k * _filter_batch_length + j])[hidden_topleft];
          }
        } else {
          h = zeros<value_t>(visible_layer_batch_size);
          for (int j = 0; j < _filter_batch_length; ++j) {
            h[seq(0,0,0,j), visible_layer_size] = *model.hidden_bias()[k * _filter_batch_length + j];
          }
        }
        _c[k] = boost::make_shared<tensor_t>(h);
        drops[k] = boost::make_shared<tensor_t>();
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

    if (model.hiddens_type() == unit_type::Bernoulli)
      hr_rand.resize(patch_layer_batch_size);

    if (model.hiddens_type() == unit_type::MyReLU ||
        model.hiddens_type() == unit_type::ReLU ||
        model.hiddens_type() == unit_type::ReLU1 ||
        model.hiddens_type() == unit_type::ReLU2 ||
        model.hiddens_type() == unit_type::ReLU4)
    {
      hr_noise.resize(patch_layer_batch_size);
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
        tbblas::synchronize();
      }
      *_c[k] = *_c[k] * (abs(*_c[k]) > 1e-16);

      if (model.shared_bias()) {
        for (int j = 0; j < _filter_batch_length; ++j) {
          *model.hidden_bias()[k * _filter_batch_length + j] = ones<value_t>(visible_layer_size) * (*_c[k])[seq(0,0,0,j)];
        }
      } else {
        for (int j = 0; j < _filter_batch_length; ++j) {
          *model.hidden_bias()[k * _filter_batch_length + j] = (*_c[k])[seq(0,0,0,j), visible_layer_size];
        }
      }
      tbblas::synchronize();
    }

    _b = _b * (abs(_b) > 1e-16);
    if (model.shared_bias()) {
      host_tensor_t b = ones<value_t>(model.visibles_size()) * _b[seq<dimCount>(0)];
      tbblas::synchronize();
      model.set_visible_bias(b);
    } else {
      tensor_t padded = rearrange_r(_b, model.stride_size());
      host_tensor_t b = padded[seq<dimCount>(0), model.visibles_size()];
      tbblas::synchronize();
      model.set_visible_bias(b);
    }
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

//  void pool_hiddens() {
//    switch (model.pooling_method()) {
//    case pooling_method::NoPooling:
//      _pooled = _hiddens;
//      break;
//
//    case pooling_method::StridePooling:
//      _pooled = _hiddens[seq<dimCount>(0), model.pooling_size(), _hiddens.size()];
//      break;
//
//    case pooling_method::MaxPooling:
//      _switches = get_max_pooling_switches(_hiddens, model.pooling_size());
//      _pooled = max_pooling(_hiddens, _switches, model.pooling_size());
//      break;
//
//    default:
//      throw std::runtime_error("Unsupported pooling method.");
//    }
//  }
//
//  void unpool_hiddens() {
//    switch (model.pooling_method()) {
//    case pooling_method::NoPooling:
//      _hiddens = _pooled;
//      break;
//
//    case pooling_method::StridePooling:
//      allocate_hiddens();
//      _hiddens = zeros<value_t>(_hiddens.size());
//      _hiddens[seq<dimCount>(0), model.pooling_size(), _hiddens.size()] = _pooled;
//      break;
//
//    case pooling_method::MaxPooling:
//      _hiddens = unpooling(_pooled, _switches, model.pooling_size());
//      break;
//
//    default:
//      throw std::runtime_error("Unsupported pooling method.");
//    }
//  }

//  void infer_visibles_from_outputs(bool onlyFilters = false) {
//    if (model.has_pooling_layer())
//      unpool_hiddens();
//
//    infer_visibles(onlyFilters);
//  }

  void infer_visibles(bool onlyFilters = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size / model.pooling_size());

    if (onlyFilters)
      v = zeros<value_t>(visible_size);
    else if (model.shared_bias())
      v = repeat(_b, visible_size / _b.size());
    else
      v = _b;

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);// for each subregion

      if (model.has_pooling_layer())
        assert(hidden_overlap_layer_batch_size % model.pooled_size() == seq<dimCount>(0));

      cvr = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = 0; k < cF.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);

        if (model.has_pooling_layer()) {
          pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _hiddens[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
          sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];

          hpr = unpooling(pr, sr, model.pooling_size());
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size];
        } else {
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = _hiddens[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        }

        chr = fft(hr, dimCount - 1, plan_hr);
        cvr += repeat_mult_sum(chr, *cF[k]);
      }
      vr = ifft(cvr, dimCount - 1, iplan_vr);
      v[topleft, overlap_size] = v[topleft, overlap_size] + vr[seq<dimCount>(0), overlap_size];
    }

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

    if (!onlyFilters)
      _visibles = _visibles * repeat(v_mask, _visibles.size() / v_mask.size());
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_visibles.size() == padded_visible_size);

    v = rearrange(_visibles, model.stride_size());

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t overlap_layer_size = min(patch_layer_size, visible_layer_size - topleft);
      dim_t overlap_layer_batch_size = min(patch_layer_batch_size, visible_layer_batch_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);

      // TODO: add assertion
      if (model.has_pooling_layer())
        assert(hidden_overlap_layer_batch_size % model.pooled_size() == seq<dimCount>(0));

      vr = zeros<value_t>(patch_size);
      vr[seq<dimCount>(0), overlap_size] = v[topleft, overlap_size];
      cvr = tbblas::fft(vr, dimCount - 1, plan_vr);

      for (size_t k = 0; k < cF.size(); ++k) {
        chr = conj_mult_sum(cvr, *cF[k]);

        hr = ifft(chr, dimCount - 1, iplan_hr);
        if (model.shared_bias()) {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] + repeat(*_c[k], overlap_layer_batch_size / _c[k]->size());  // apply sub-region of bias terms
        } else {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] + (*_c[k])[topleft, overlap_layer_batch_size];  // apply sub-region of bias terms
        }

        switch (model.hiddens_type()) {
          case unit_type::Bernoulli: hr = sigm(hr); break;
          case unit_type::ReLU:      hr = max(0.0, hr);  break;
          case unit_type::MyReLU:    hr = nrelu_mean(hr); break;
          case unit_type::ReLU1:     hr = min(1.0, max(0.0, hr));  break;
          case unit_type::ReLU2:     hr = min(2.0, max(0.0, hr));  break;
          case unit_type::ReLU4:     hr = min(4.0, max(0.0, hr));  break;
          case unit_type::ReLU8:     hr = min(8.0, max(0.0, hr));  break;
        }

        if (_dropout_rate > 0) {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] *
              (*drops[k])[seq<dimCount>(0), overlap_layer_batch_size] / (1. - _dropout_rate) *
              repeat(h_mask[topleft, overlap_layer_size], overlap_layer_batch_size / overlap_layer_size);
        } else {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] *
              repeat(h_mask[topleft, overlap_layer_size], overlap_layer_batch_size / overlap_layer_size);
        }

        // TODO: do the pooling
        if (model.has_pooling_layer()) {
          hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size] = hr[hidden_topleft, hidden_overlap_layer_batch_size];
          sr = get_max_pooling_switches(hpr, model.pooling_size());
          pr = max_pooling(hpr, sr, model.pooling_size());

          _hiddens[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()] =
              pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()];
          _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()] =
              sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()];
        } else {
          _hiddens[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size] =
              hr[hidden_topleft, hidden_overlap_layer_batch_size];
        }
      }
    }
  }

  void sample_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size / model.pooling_size());

    if (model.shared_bias()) {
      v = repeat(_b, visible_size / _b.size());
    } else {
      v = _b;
    }

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);// for each subregion

      if (model.has_pooling_layer())
        assert(hidden_overlap_layer_batch_size % model.pooled_size() == seq<dimCount>(0));

      cvr = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = 0; k < cF.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);

        if (model.has_pooling_layer()) {
          pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _hiddens[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
          sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];

          hpr = unpooling(pr, sr, model.pooling_size());
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size];
        } else {
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = _hiddens[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        }

        chr = fft(hr, dimCount - 1, plan_hr);
        cvr += repeat_mult_sum(chr, *cF[k]);
      }
      vr = ifft(cvr, dimCount - 1, iplan_vr);
      v[topleft, overlap_size] = v[topleft, overlap_size] + vr[seq<dimCount>(0), overlap_size];
    }

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

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t overlap_layer_size = min(patch_layer_size, visible_layer_size - topleft);
      dim_t overlap_layer_batch_size = min(patch_layer_batch_size, visible_layer_batch_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);

      if (model.has_pooling_layer())
        assert(hidden_overlap_layer_batch_size % model.pooled_size() == seq<dimCount>(0));

      vr = zeros<value_t>(patch_size);
      vr[seq<dimCount>(0), overlap_size] = v[topleft, overlap_size];
      cvr = tbblas::fft(vr, dimCount - 1, plan_vr);

      for (size_t k = 0; k < cF.size(); ++k) {
        chr = conj_mult_sum(cvr, *cF[k]);

        hr = ifft(chr, dimCount - 1, iplan_hr);
        if (model.shared_bias()) {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] + repeat(*_c[k], overlap_layer_batch_size / _c[k]->size());  // apply sub-region of bias terms
        } else {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] + (*_c[k])[topleft, overlap_layer_batch_size];  // apply sub-region of bias terms
        }

        switch (model.hiddens_type()) {
          case unit_type::Bernoulli: hr = sigm(hr) > hr_rand(); break;
          case unit_type::MyReLU:
          case unit_type::ReLU:      hr = max(0.0, hr + sqrt(sigm(hr)) * hr_noise()); break;
          case unit_type::ReLU1:     hr = min(1.0, max(0.0, hr + (hr > 0) * (hr < 1.0) * hr_noise())); break;
          case unit_type::ReLU2:     hr = min(2.0, max(0.0, hr + (hr > 0) * (hr < 2.0) * hr_noise())); break;
          case unit_type::ReLU4:     hr = min(4.0, max(0.0, hr + (hr > 0) * (hr < 4.0) * hr_noise())); break;
          case unit_type::ReLU8:     hr = min(8.0, max(0.0, hr + (hr > 0) * (hr < 8.0) * hr_noise())); break;
        }

        if (_dropout_rate > 0) {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] *
              (*drops[k])[seq<dimCount>(0), overlap_layer_batch_size] / (1. - _dropout_rate) *
              repeat(h_mask[topleft, overlap_layer_size], overlap_layer_batch_size / overlap_layer_size);
        } else {
          hr[seq<dimCount>(0), overlap_layer_batch_size] =
              hr[seq<dimCount>(0), overlap_layer_batch_size] *
              repeat(h_mask[topleft, overlap_layer_size], overlap_layer_batch_size / overlap_layer_size);
        }

        if (model.has_pooling_layer()) {
          hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size] = hr[hidden_topleft, hidden_overlap_layer_batch_size];
          sr = get_max_pooling_switches(hpr, model.pooling_size());
          pr = max_pooling(hpr, sr, model.pooling_size());

          _hiddens[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()] =
              pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()];
          _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()] =
              sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()];
        } else {
          _hiddens[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size] =
              hr[hidden_topleft, hidden_overlap_layer_batch_size];
        }
      }
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

    if (h_rand.size() != visible_layer_batch_size)
      h_rand.resize(visible_layer_batch_size);

    if (method == dropout_method::DropColumn) {
      *drops[0] = h_rand() > _dropout_rate;
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


    v = rearrange(_visibles, model.stride_size());

    if (model.shared_bias()) {
      _binc = _binc + sum(v) / v.count();
    } else {
      _binc += v;
    }

    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t overlap_layer_size = min(patch_layer_size, visible_layer_size - topleft);
      dim_t overlap_layer_batch_size = min(patch_layer_batch_size, visible_layer_batch_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);

      vr = zeros<value_t>(patch_size);
      vr[seq<dimCount>(0), overlap_size] = v[topleft, overlap_size];
      cvr = tbblas::fft(vr, dimCount - 1, plan_vr);

      for (size_t k = 0; k < cFinc.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);

        if (model.has_pooling_layer()) {
          pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _hiddens[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
          sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];

          hpr = unpooling(pr, sr, model.pooling_size());
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size];
        } else {
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = _hiddens[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        }

        chr = fft(hr, dimCount - 1, plan_hr);

        *cFinc[k] += conj_repeat_mult(cvr, chr, value_t(1) / _voxel_count);

        if (model.shared_bias()) {
          for (int j = 0; j < _filter_batch_length; ++j) {
            (*_cinc[k])[seq(0,0,0,j), seq<dimCount>(1)] =
                (*_cinc[k])[seq(0,0,0,j), seq<dimCount>(1)] + sum(hr[seq<dimCount>(0), overlap_layer_batch_size]) / visible_layer_size.prod();
          }
        } else {
          (*_cinc[k])[topleft, overlap_layer_batch_size] =
              (*_cinc[k])[topleft, overlap_layer_batch_size] + hr[seq<dimCount>(0), overlap_layer_batch_size];
        }

        switch(_sparsity_method) {
        case sparsity_method::NoSparsity:
          break;

  //      case sparsity_method::WeightsAndBias:
  //        chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
  //        *cFinc[k] = *cFinc[k] + value_t(1) / _voxel_count * _sparsity_weight * repeat(conj(chdiff), cFinc[k]->size() / ch.size()) * repeat(cv, cFinc[k]->size() / cv.size());
  //        *ccinc[k] = *ccinc[k] + value_t(1) / visible_size[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
  //        break;

          // TODO: convert to use new shared bias model
//        case sparsity_method::OnlyBias:
//          (*_cinc[k])[topleft, overlap_layer_batch_size] =
//              (*_cinc[k])[topleft, overlap_layer_batch_size] + _sparsity_weight * (_sparsity_target + -hr[seq<dimCount>(0), overlap_layer_batch_size]);
//          break;

          // TODO: conver to use new shared bias model
//        case sparsity_method::OnlySharedBias:
//          (*_cinc[k])[topleft, overlap_layer_batch_size] =
//              (*_cinc[k])[topleft, overlap_layer_batch_size] +
//              _sparsity_weight * sum(_sparsity_target + -hr[seq<dimCount>(0), overlap_layer_batch_size]) / overlap_layer_batch_size.prod();
//          break;

        default:
          throw std::runtime_error("Unsupported sparsity method.");
        }
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

    v = rearrange(_visibles, model.stride_size());

    if (model.shared_bias()) {
      _binc = _binc - sum(v) / v.count();
    } else {
      _binc = _binc - v;
    }

    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t overlap_layer_size = min(patch_layer_size, visible_layer_size - topleft);
      dim_t overlap_layer_batch_size = min(patch_layer_batch_size, visible_layer_batch_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);

      vr = zeros<value_t>(patch_size);
      vr[seq<dimCount>(0), overlap_size] = v[topleft, overlap_size];
      cvr = tbblas::fft(vr, dimCount - 1, plan_vr);

      for (size_t k = 0; k < cF.size(); ++k) {

        hr = zeros<value_t>(patch_layer_batch_size);

        if (model.has_pooling_layer()) {
          pr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _hiddens[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];
          sr[seq<dimCount>(0), hidden_overlap_layer_batch_size / model.pooling_size()] =
              _switches[topleft / model.pooling_size() + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size / model.pooling_size()];

          hpr = unpooling(pr, sr, model.pooling_size());
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = hpr[seq<dimCount>(0), hidden_overlap_layer_batch_size];
        } else {
          hr[hidden_topleft, hidden_overlap_layer_batch_size] = _hiddens[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        }

        chr = fft(hr, dimCount - 1, plan_hr);

        *cFinc[k] += conj_repeat_mult(cvr, chr, value_t(-1) / _voxel_count);

        if (model.shared_bias()) {
          for (int j = 0; j < _filter_batch_length; ++j) {
            (*_cinc[k])[seq(0,0,0,j), seq<dimCount>(1)] =
                (*_cinc[k])[seq(0,0,0,j), seq<dimCount>(1)] - sum(hr[seq<dimCount>(0), overlap_layer_batch_size]) / visible_layer_size.prod();
          }
        } else {
          (*_cinc[k])[topleft, overlap_layer_batch_size] =
              (*_cinc[k])[topleft, overlap_layer_batch_size] - hr[seq<dimCount>(0), overlap_layer_batch_size];
        }
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

    if (deltab.size() != _b.size())
      deltab = zeros<value_t>(_b.size());

    if (!deltaF.size()) {
      deltaF.resize(model.filter_count());
      for (size_t k = 0; k < deltaF.size(); ++k)
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
    }

    if (!deltac.size()) {
      deltac.resize(_cinc.size());
      for (size_t k = 0; k < deltac.size(); ++k)
        deltac[k] = boost::make_shared<tensor_t>(zeros<value_t>(_cinc[k]->size()));
    }

    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        cvr = (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] * (value_t(1) / _positive_batch_size) - weightcost * (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);
        vr = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(vr, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        // update deltaF
        *deltaF[iFilter] = momentum * *deltaF[iFilter] + padded_k[seq<dimCount>(0), model.kernel_size()];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = *deltaF[iFilter];
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        vr = ifftshift(padded_f, dimCount - 1);

        cvr = fft(vr, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;

//        if (model.shared_bias())
//          (*_cinc[k])[seq(0,0,0,j), visible_layer_size] = ones<value_t>(visible_layer_size) * sum((*_cinc[k])[seq(0,0,0,j), visible_layer_size]) / visible_layer_size.prod();
      }

      *deltac[k] = momentum * *deltac[k] + *_cinc[k] / _positive_batch_size;

      // Apply delta to current filters
      *cF[k] = *cF[k] + epsilon * *cFinc[k];
      *_c[k] = *_c[k] + epsilon * *deltac[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *_cinc[k] = zeros<value_t>(_cinc[k]->size());
    }

    // TODO: do masking (unless I'm using shared bias)
    deltab = (momentum * deltab + _binc / _positive_batch_size);// * repeat(v_mask, deltab.size() / v_mask.size());

    _b += epsilon * deltab;
    _binc = zeros<value_t>(_binc.size());

    _positive_batch_size = _negative_batch_size = 0;
  }

  /// If 'average_filter' is true, and average learning rate per filter is calculated
  void adadelta_step(value_t epsilon, value_t momentum = 0, value_t weightcost = 0, bool average_filter = false) {
    _host_updated = false;

    using namespace tbblas;

    value_t gradient_reduction = 1;

    if (_positive_batch_size != _negative_batch_size)
      throw std::runtime_error("Number of positive and negative updates does not match.");

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (db2.size() != _b.size() || deltab.size() != _b.size() || deltab2.size() != _b.size()) {
      db2 = deltab = deltab2 = zeros<value_t>(_b.size());
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
        dc2[k] = boost::make_shared<tensor_t>(zeros<value_t>(_cinc[k]->size()));
        deltac[k] = boost::make_shared<tensor_t>(zeros<value_t>(_cinc[k]->size()));
        deltac2[k] = boost::make_shared<tensor_t>(zeros<value_t>(_cinc[k]->size()));
      }
    }

    dim_t fullsize = cvr.fullsize();

    for (size_t k = 0; k < cF.size(); ++k) {
      // Mask filters
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        const int iFilter = k * _filter_batch_length + j;
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        cvr = (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] * (value_t(1) / _positive_batch_size) - weightcost * (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()];
        cvr.set_fullsize(fullsize);
        vr = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(vr, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        // update deltaF
        *dF2[iFilter] = momentum * *dF2[iFilter] + (1.0 - momentum) * padded_k[seq<dimCount>(0), model.kernel_size()] * padded_k[seq<dimCount>(0), model.kernel_size()];
        if (average_filter)
          *dF2[iFilter] = ones<value_t>(dF2[iFilter]->size()) * sum(*dF2[iFilter]) / dF2[iFilter]->count();

        *deltaF[iFilter] = sqrt(*deltaF2[iFilter] + epsilon) / sqrt(*dF2[iFilter] + epsilon) * padded_k[seq<dimCount>(0), model.kernel_size()];

        *deltaF2[iFilter] = momentum * *deltaF2[iFilter] + (1.0 - momentum) * *deltaF[iFilter] * *deltaF[iFilter];
        if (average_filter)
          *deltaF2[iFilter] = ones<value_t>(deltaF2[iFilter]->size()) * sum(*deltaF2[iFilter]) / deltaF2[iFilter]->count();

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = *deltaF[iFilter];
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        vr = ifftshift(padded_f, dimCount - 1);

        cvr = fft(vr, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;

//        if (model.shared_bias())
//          (*_cinc[k])[seq(0,0,0,j), visible_layer_size] = ones<value_t>(visible_layer_size) * sum((*_cinc[k])[seq(0,0,0,j), visible_layer_size]) / visible_layer_size.prod();
      }

      *_cinc[k] = *_cinc[k] / _positive_batch_size;

      *dc2[k] = momentum * *dc2[k] + (1.0 - momentum) * *_cinc[k] * *_cinc[k];
      *deltac[k] = sqrt(*deltac2[k] + epsilon) / sqrt(*dc2[k] + epsilon) * *_cinc[k];
      *deltac2[k] = momentum * *deltac2[k] + (1.0 - momentum) * *deltac[k] * *deltac[k];

      // Apply delta to current filters
      *cF[k] = *cF[k] + gradient_reduction * *cFinc[k];
      *_c[k] = *_c[k] + gradient_reduction * *deltac[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
      *_cinc[k] = zeros<value_t>(_cinc[k]->size());
    }

    _binc = _binc / _positive_batch_size; // * repeat(v_mask, padded_f.size() / v_mask.size());


    db2 = momentum * db2 + (1.0 - momentum) * _binc * _binc;
    deltab = sqrt(deltab2 + epsilon) / sqrt(db2 + epsilon) * _binc;
    deltab2 = momentum * deltab2 + (1.0 - momentum) * deltab * deltab;

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
    if (!_memory_allocated)
      allocate_gpu_memory();
  }

//  tensor_t& pooled_units() {
//    if (!_memory_allocated)
//      allocate_gpu_memory();
//    return _pooled;
//  }
//
//  void allocate_pooled_units() {
//    if (_pooled.size() != model.pooled_size())
//      _pooled.resize(model.pooled_size());
//  }
//
//  tensor_t& outputs() {
//    if (model.has_pooling_layer())
//      return pooled_units();
//    else
//      return hiddens();
//  }
//
//  void allocate_outputs() {
//    if (model.has_pooling_layer())
//      allocate_pooled_units();
//    else
//      allocate_hiddens();
//  }

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
