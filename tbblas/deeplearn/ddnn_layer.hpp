/*
 * ddnn_layer.hpp
 *
 *  Created on: Mar 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DDNN_LAYER_HPP_
#define TBBLAS_DEEPLEARN_DDNN_LAYER_HPP_

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

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims, class Trainer, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class ddnn_layer : public dnn_layer<T, dims, Trainer> {

  typedef dnn_layer<T, dims, Trainer> base_t;

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

  typedef cnn_layer<value_t, dimCount, Trainer> cnn_layer_t;
  typedef dnn_layer<value_t, dimCount, Trainer> dnn_layer_t;

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
  model_t& model;
  cnn_layer_t& cnn_layer;

  // weights in GPU memory
  v_ctensor_t cF, cFinc;

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

  using base_t::_memory_allocated;
  using base_t::_host_updated;

  using base_t::_current_batch_size;
  using base_t::_dropout_rate;
  using base_t::_sensitivity_ratio;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  ddnn_layer(model_t& shortcut_model, model_t& decoder_model, cnn_layer_t& cnn_layer, const Trainer* parameters, dim_t patch_count = seq<dimCount>(1))
    : base_t(decoder_model, parameters, patch_count),
      model(shortcut_model), cnn_layer(cnn_layer)
  {
    base_t::_p_switches = cnn_layer._p_switches;
    // TODO: check some preconditions: same stride size, same layer size, same batch size, same pooling size and method
  }

private:
  ddnn_layer(const ddnn_layer&);

public:
  virtual ~ddnn_layer() {
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

    tbblas::synchronize();
  }

  virtual void infer_visibles(int flags = DEFAULT) {

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(cnn_layer.H.size() == hidden_size);

    if (!(flags & ACCUMULATE))
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

    base_t::infer_visibles(flags | ACCUMULATE);
  }

  virtual void backprop_hidden_deltas() {
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
        *cFinc[k] += conj_repeat_mult(cvr, chr, value_t(1));
      }
    }
  }

  virtual size_t gradient_length() const {
    return model.kernel_size().prod() * cF.size() * _filter_batch_length + base_t::gradient_length();
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

    return base_t::collect_gradient(gradient, offset, weightcost);
  }

  virtual void reset_gradient() {
    if (!cFinc.size())
      return;

    for (size_t k = 0; k < cF.size(); ++k) {
      *cFinc[k] = zeros<complex_t>(cFinc[k]->size(), cFinc[k]->fullsize());
    }

    base_t::reset_gradient();
  }

  virtual int set_parameters(tensor<value_t, 1, true>& parameters, int offset) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    dim_t fullsize = cvr.fullsize();

    // Get filters first
    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {

        // Extract filter
        dim_t topleft = patch_size / 2 - kernel_size / 2;

        // Put weights into cF
        padded_k = zeros<value_t>(padded_kernel_size);
        padded_k[seq<dimCount>(0), model.kernel_size()] = reshape(parameters[seq(offset), seq(model.kernel_size().prod())], model.kernel_size());
        offset += model.kernel_size().prod();

        padded_f = zeros<value_t>(vr.size());
        padded_f[topleft, kernel_size] = rearrange(padded_k, model.stride_size());
        vr = ifftshift(padded_f, dimCount - 1);

        cvr = fft(vr, dimCount - 1, plan_vr);
        (*cF[k])[seq(0,0,0,j*cvr.size()[dimCount - 1]), cvr.size()] = cvr;
      }
    }

    return base_t::set_parameters(parameters, offset);
  }

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
        vr = ifft(cvr, dimCount - 1, iplan_vr);
        padded_f = fftshift(vr, dimCount - 1);
        padded_k = rearrange_r(padded_f[topleft, kernel_size], model.stride_size());

        assert(padded_k.size() == padded_kernel_size);

        parameters[seq(offset), seq(model.kernel_size().prod())] = reshape(padded_k[seq<dimCount>(0), model.kernel_size()], seq(model.kernel_size().prod()));
        offset += model.kernel_size().prod();
      }
    }

    return base_t::get_parameters(parameters, offset);
  }

  virtual int get_weight_mask(tensor<value_t, 1, true>& weight_mask, int offset) {
    for (size_t k = 0; k < cF.size(); ++k) {
      for (int j = 0; j < _filter_batch_length; ++j) {
        weight_mask[seq(offset), seq(model.kernel_size().prod())] = ones<value_t>(seq(model.kernel_size().prod()));
        offset += model.kernel_size().prod();
      }
    }

    return base_t::get_weight_mask(weight_mask, offset);
  }

  virtual int update_model(tensor<value_t, 1, true>& delta, int offset) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cF.size())
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

    return base_t::update_model(delta, offset);
  }

  virtual void update_model(value_t weightcost) {

    // This function temporarily puts the deltas into the
    // gradient in frequency domain just to apply the deltas
    // to the model in the frequency domain. Afterwards, the
    // gradients are reset to zero

    if (!cFinc.size())
      throw std::runtime_error("No gradient calculated.");

    dim_t fullsize = cvr.fullsize();

    const int param_offset = model.filter_count() + 1;

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
        this->update_delta(padded_k[seq<dimCount>(0), model.kernel_size()], iFilter + param_offset);

        // Put deltaF into cFinc to apply the delta to the current filter
        padded_k = zeros<value_t>(padded_k.size());
        padded_k[seq<dimCount>(0), model.kernel_size()] = reshape(this->delta(iFilter + param_offset), model.kernel_size());
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

    base_t::update_model(weightcost);
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_DDNN_LAYER_HPP_ */
