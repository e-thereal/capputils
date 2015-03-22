/*
 * cdnn_layer.hpp
 *
 *  Created on: Mar 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CDNN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_CDNN_LAYER_HPP_

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
#include <tbblas/deeplearn/cnn_layer.hpp>
#include <tbblas/deeplearn/dnn_layer.hpp>
#include <tbblas/deeplearn/objective_function.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class cdnn_layer : public dnn_layer<T, dims> {

  typedef dnn_layer<T, dims> base_t;

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

  typedef cnn_layer<value_t, dimCount> cnn_layer_t;
  typedef dnn_layer<value_t, dimCount> dnn_layer_t;

  typedef tbblas::random_tensor2<value_t, 4, true, tbblas::uniform<value_t> > uniform_t;

protected:
  model_t& model;
  cnn_layer_t& cnn_layer;

  // weights and bias terms in GPU memory
  v_ctensor_t cF, cFinc;
  v_tensor_t deltaF, dF2, deltaF2;
  tensor_t deltaF_hat, deltaF2_hat;

  using base_t::padded_visible_size;
  using base_t::visible_size;
  using base_t::hidden_size;
  using base_t::visible_layer_size;
  using base_t::hidden_layer_size;
  using base_t::hidden_layer_batch_size;
  using base_t::visible_layer_batch_size;
  using base_t::visible_batch_size;
  using base_t::kernel_size;
  using base_t::padded_kernel_size;
  using base_t::hidden_topleft;
  using base_t::patch_count;
  using base_t::patch_size;
  using base_t::patch_batch_size;
  using base_t::patch_layer_size;
  using base_t::patch_layer_batch_size;
  using base_t::hidden_patch_layer_batch_size;
  using base_t::step_size;
  using base_t::step_count;

  using base_t::vp;
  using base_t::vr;
  using base_t::hr;
  using base_t::hpr;
  using base_t::pr;
  using base_t::shifted_f;
  using base_t::padded_f;
  using base_t::padded_k;
  using base_t::v_mask;
  using base_t::h_mask;

  using base_t::cvr;
  using base_t::chr;
  using base_t::chrdiff;

  using base_t::plan_vr;
  using base_t::iplan_vr;
  using base_t::plan_hr;
  using base_t::iplan_hr;

  using base_t::v_rand;

  using base_t::_p_switches;
  using base_t::sr;

  using base_t::_filter_batch_length;
  using base_t::_voxel_count;

  using base_t::_memory_allocated;
  using base_t::_host_updated;

  using base_t::_current_batch_size;
  using base_t::_dropout_rate;
  using base_t::_sensitivity_ratio;
  using base_t::_current_iteration;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  cdnn_layer(model_t& shortcut_model, model_t& decoder_model, cnn_layer_t& cnn_layer, dim_t patch_count = seq<dimCount>(1))
    : base_t(decoder_model, patch_count),
      model(shortcut_model), cnn_layer(cnn_layer)
  {
    base_t::_p_switches = cnn_layer._p_switches;
    // TODO: check some preconditions: same stride size, same layer size, same batch size, same pooling size and method
  }

private:
  cdnn_layer(const cdnn_layer&);

public:
  virtual ~cdnn_layer() {
    if (!_host_updated)
      write_model_to_host();
  }

  /// Transforms
  virtual void allocate_gpu_memory() {
    using namespace tbblas;

    if (_memory_allocated)
      return;

    base_t::allocate_gpu_memory();

    cF.resize(model.filters().size() / _filter_batch_length);

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
  }

  virtual void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    base_t::write_model_to_host();

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
  }

  virtual void infer_visibles(bool dropout = false, bool accumulateActivation = false) {

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(cnn_layer.H.size() == hidden_size);

    vp = zeros<value_t>(visible_size);

    // Iterate over sub-regions
    for (sequence_iterator<dim_t> iter(seq<dimCount>(0), step_count); iter; ++iter) {
      dim_t topleft = *iter * step_size;

      dim_t overlap_size = min(patch_size, visible_size - topleft);
      dim_t hidden_overlap_layer_batch_size = min(hidden_layer_batch_size - topleft, hidden_patch_layer_batch_size);// for each subregion
      cvr = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = 0; k < cF.size(); ++k) {
        hr = zeros<value_t>(patch_layer_batch_size);
        hr[hidden_topleft, hidden_overlap_layer_batch_size] = cnn_layer.H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];
        chr = fft(hr, dimCount - 1, plan_hr);
        cvr += repeat_mult_sum(chr, *cF[k]);
      }
      vr = ifft(cvr, dimCount - 1, iplan_vr);
      vp[topleft, overlap_size] = vp[topleft, overlap_size] + vr[seq<dimCount>(0), overlap_size];
    }

    base_t::infer_visibles(dropout, true);
  }

  void backprop_hidden_deltas() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(this->padded_v.size() == padded_visible_size);

    vp = rearrange(this->padded_v, model.stride_size());
    cnn_layer.dH = zeros<value_t>(hidden_size);

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
        cnn_layer.dH[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size] =
            hr[hidden_topleft, hidden_overlap_layer_batch_size];
      }
    }
  }

  /// Requires visible deltas and hiddens
  virtual void update_gradient() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    base_t::update_gradient();

    if (!cFinc.size()) {
      cFinc.resize(cF.size());
      for (size_t i = 0; i < cFinc.size(); ++i)
        cFinc[i] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[i]->size(), cF[i]->fullsize()));
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
        hr[hidden_topleft, hidden_overlap_layer_batch_size] = cnn_layer.H[topleft + seq(0,0,0,(int)k * _filter_batch_length), hidden_overlap_layer_batch_size];

        chr = fft(hr, dimCount - 1, plan_hr);
        *cFinc[k] += conj_repeat_mult(cvr, chr, value_t(1) / _voxel_count);
      }
    }
  }

  virtual void momentum_step(value_t epsilon, value_t momentum, value_t weightcost) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");

    if (!deltaF.size()) {
      deltaF.resize(model.filter_count());
      for (size_t k = 0; k < deltaF.size(); ++k)
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
    }

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
        *deltaF[iFilter] = momentum * *deltaF[iFilter] + padded_k[seq<dimCount>(0), model.kernel_size()];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = *deltaF[iFilter];
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cvr = fft(shifted_f, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }

      // Apply delta to current filters
      *cF[k] = *cF[k] - epsilon * *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    base_t::momentum_step(epsilon, momentum, weightcost);
  }

  virtual void adadelta_step(value_t epsilon, value_t momentum, value_t weightcost) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
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
        *dF2[iFilter] = momentum * *dF2[iFilter] + (1.0 - momentum) * padded_k[seq<dimCount>(0), model.kernel_size()] * padded_k[seq<dimCount>(0), model.kernel_size()];
        *deltaF[iFilter] = sqrt(*deltaF2[iFilter] + epsilon) / sqrt(*dF2[iFilter] + epsilon) * padded_k[seq<dimCount>(0), model.kernel_size()];
        *deltaF2[iFilter] = momentum * *deltaF2[iFilter] + (1.0 - momentum) * *deltaF[iFilter] * *deltaF[iFilter];

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = *deltaF[iFilter];
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cvr = fft(shifted_f, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }

      // Apply delta to current filters
      *cF[k] = *cF[k] - *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    base_t::adadelta_step(epsilon, momentum, weightcost);
  }

  virtual void adam_step(value_t alpha, value_t beta1, value_t beta2, value_t epsilon, value_t betaDecay, value_t weightcost) {
    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");

    if (!deltaF.size() || !deltaF2.size()) {
      deltaF.resize(model.filter_count());
      deltaF2.resize(model.filter_count());
      for (size_t k = 0; k < deltaF.size(); ++k) {
        deltaF[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
        deltaF2[k] = boost::make_shared<tensor_t>(zeros<value_t>(model.kernel_size()));
      }
    }

    dim_t fullsize = cvr.fullsize();
    const value_t beta1t = 1.0 - (1.0 - beta1) * ::pow(betaDecay, _current_iteration);
    ++_current_iteration;

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
        *deltaF[iFilter] = beta1t * padded_k[seq<dimCount>(0), model.kernel_size()] + (1. - beta1t) * *deltaF[iFilter];
        *deltaF2[iFilter] = beta2 * padded_k[seq<dimCount>(0), model.kernel_size()] * padded_k[seq<dimCount>(0), model.kernel_size()]
                               + (1. - beta2) * *deltaF2[iFilter];

        deltaF_hat = *deltaF[iFilter] / (value_t(1) - ::pow(value_t(1) - beta1, _current_iteration));
        deltaF2_hat = *deltaF2[iFilter] / (value_t(1) - ::pow(value_t(1) - beta2, _current_iteration));

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = deltaF_hat / (sqrt(deltaF2_hat) + epsilon);
        padded_f = zeros<value_t>(padded_f.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        shifted_f = ifftshift(padded_f, dimCount - 1);

        cvr = fft(shifted_f, dimCount - 1, plan_vr);
        (*cFinc[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }

      // Apply delta to current filters
      *cF[k] = *cF[k] - alpha * *cFinc[k];

      // Reset filter gradient
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    --_current_iteration;
    base_t::adam_step(alpha, beta1, beta2, epsilon, betaDecay, weightcost);
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CDNN_LAYER_HPP_ */
