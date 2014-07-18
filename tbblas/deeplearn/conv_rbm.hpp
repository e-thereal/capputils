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

#include <omp.h>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/mult_sum.hpp>
#include <tbblas/deeplearn/repeat_mult.hpp>
#include <tbblas/deeplearn/repeat_mult_sum.hpp>
#include <tbblas/deeplearn/convolution_type.hpp>
#include <tbblas/deeplearn/unit_type.hpp>
#include <tbblas/deeplearn/sparsity_method.hpp>
#include <tbblas/deeplearn/dropout_method.hpp>

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
  typedef tbblas::tensor<complex_t, dimCount, false> host_ctensor_t;

  typedef tbblas::random_tensor<value_t, dimCount, true, tbblas::uniform<value_t> > uniform_t;
  typedef tbblas::random_tensor<value_t, dimCount, true, tbblas::normal<value_t> > normal_t;

  typedef conv_rbm_model<value_t, dimCount> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  ctensor_t cb, cbinc;
  std::vector<boost::shared_ptr<ctensor_t> > cF, cc, cFinc, ccinc;
  v_tensor_t drops;

  // visible and hidden units in GPU memory

  // Sizes
  dim_t visible_size, hidden_size, size,
        visible_layer_size, hidden_layer_size, layer_size,
        hidden_layer_batch_size, layer_batch_size,
        filter_batch_size, hidden_topleft,
        vbMaskSize, hbMaskSize, spMaskSize;

  // one element per thread
  std::vector<boost::shared_ptr<tensor_t> > v_v, v_h, v_f, v_v_mask, v_h_mask;
  std::vector<boost::shared_ptr<ctensor_t> > v_cv, v_ch, v_chdiff;
  std::vector<boost::shared_ptr<plan_t> > v_plan_v, v_iplan_v, v_plan_h, v_iplan_h;
  std::vector<boost::shared_ptr<uniform_t> > v_v_rand, v_h_rand;
  std::vector<boost::shared_ptr<normal_t> > v_v_noise, v_h_noise;

  tensor_t _hiddens;

  int _gpu_count, _device_count, _filter_batch_length;
  bool _memory_allocated, _double_weights, _host_updated;

  value_t _sparsity_target, _sparsity_weight, _dropout_rate;
  sparsity_method _sparsity_method;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm(model_t& model, size_t gpu_count = 1) : model(model),
    _gpu_count(gpu_count), _device_count(0), _filter_batch_length(1),
    _memory_allocated(false), _double_weights(false), _host_updated(true),
    _sparsity_target(0.1), _sparsity_weight(0), _dropout_rate(0), _sparsity_method(sparsity_method::OnlySharedBias)
  {
    assert(_gpu_count > 0);

    v_v.resize(gpu_count);
    v_f.resize(gpu_count);
    v_h.resize(gpu_count);
    v_v_mask.resize(gpu_count);
    v_h_mask.resize(gpu_count);
    v_cv.resize(gpu_count);
    v_ch.resize(gpu_count);
    v_chdiff.resize(gpu_count);

    v_plan_v.resize(gpu_count);
    v_iplan_v.resize(gpu_count);
    v_plan_h.resize(gpu_count);
    v_iplan_h.resize(gpu_count);

    v_v_rand.resize(gpu_count);
    v_h_rand.resize(gpu_count);
    v_v_noise.resize(gpu_count);
    v_h_noise.resize(gpu_count);
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

    if (_gpu_count > 1)
      enable_peer_access(_gpu_count);

    setup_threads();

    // Prepare sizes
    visible_size = model.visibles_size();
    size = visible_size;

    visible_layer_size = visible_size;
    layer_size = filter_batch_size = layer_batch_size = size;
    visible_layer_size[dimCount - 1] = layer_size[dimCount - 1] = 1;
    filter_batch_size[dimCount - 1] = size[dimCount - 1] * _filter_batch_length;
    layer_batch_size[dimCount - 1] = _filter_batch_length;

    if (model.convolution_type() == convolution_type::Valid){
      hidden_topleft = model.kernel_size() / 2;
      hidden_topleft[dimCount - 1] = 0;
    } else {
      hidden_topleft = seq<4>(0);
    }

    hidden_layer_size = visible_layer_size - 2 * hidden_topleft;
    hidden_layer_batch_size = hidden_layer_size * seq(1,1,1,_filter_batch_length);
    hidden_size = visible_size - 2 * hidden_topleft;
    hidden_size[dimCount - 1] = model.filters().size();

    _hiddens = zeros<value_t>(hidden_size);

    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(size);

      tensor_t A = v_noise, B = A;
      ctensor_t cA = fft(A, 3), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, 3);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
    {
      random_tensor<value_t, dimCount, true, normal<value_t> > v_noise(layer_batch_size);

      tensor_t A = v_noise, B = A;
      ctensor_t cA = fft(A, 3), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, 3);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }

    cF.resize(model.filters().size() / _filter_batch_length);
    cc.resize(model.hidden_bias().size() / _filter_batch_length);

    cFinc.resize(cF.size());
    ccinc.resize(cc.size());
    drops.resize(cF.size());

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      v_v[tid] = boost::make_shared<tensor_t>();
      v_f[tid] = boost::make_shared<tensor_t>();
      v_h[tid] = boost::make_shared<tensor_t>();
      v_cv[tid] = boost::make_shared<ctensor_t>();
      v_ch[tid] = boost::make_shared<ctensor_t>();
      v_chdiff[tid] = boost::make_shared<ctensor_t>();

      plan_t& plan_v = *(v_plan_v[tid] = boost::make_shared<plan_t>());
      plan_t& iplan_v = *(v_iplan_v[tid] = boost::make_shared<plan_t>());
      plan_t& plan_h = *(v_plan_h[tid] = boost::make_shared<plan_t>());
      plan_t& iplan_h = *(v_iplan_h[tid] = boost::make_shared<plan_t>());

      tensor_t& v_mask = *(v_v_mask[tid] = boost::make_shared<tensor_t>());
      tensor_t& h_mask = *(v_h_mask[tid] = boost::make_shared<tensor_t>());
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

      #pragma omp master
      {
        tensor_t b = model.visible_bias();
        cb = fft(b, dimCount - 1, plan_v);
        cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
      }

      // Copy filters to the device and pre-calculate the FFT
      {
        tensor_t f, h, kern, pad;
        ctensor_t cf, ch;
        plan_t plan_f;
        for (size_t k = tid; k < cF.size(); k += _gpu_count) {
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
            h[seq(0,0,0,j), visible_layer_size] = *model.hidden_bias()[k * _filter_batch_length + j];
          }
          ch = fft(h, dimCount - 1, plan_h);
          cc[k] = boost::make_shared<ctensor_t>(ch);
          ch = zeros<complex_t>(ch.size(), ch.fullsize());
          ccinc[k] = boost::make_shared<ctensor_t>(ch);

          drops[k] = boost::make_shared<tensor_t>();
        }
      }

      uniform_t& h_rand = *(v_h_rand[tid] = boost::make_shared<uniform_t>());
      uniform_t& v_rand = *(v_v_rand[tid] = boost::make_shared<uniform_t>());
      normal_t& h_noise = *(v_h_noise[tid] = boost::make_shared<normal_t>());
      normal_t& v_noise = *(v_v_noise[tid] = boost::make_shared<normal_t>());

      if (model.hiddens_type() == unit_type::Bernoulli)
        h_rand.resize(layer_batch_size, tid);

      if (model.hiddens_type() == unit_type::MyReLU ||
          model.hiddens_type() == unit_type::ReLU ||
          model.hiddens_type() == unit_type::ReLU1 ||
          model.hiddens_type() == unit_type::ReLU2 ||
          model.hiddens_type() == unit_type::ReLU4)
      {
        h_noise.resize(layer_batch_size, tid);
      }

      if (model.visibles_type() == unit_type::MyReLU ||
          model.visibles_type() == unit_type::ReLU ||
          model.visibles_type() == unit_type::ReLU1 ||
          model.visibles_type() == unit_type::ReLU2 ||
          model.visibles_type() == unit_type::ReLU4)
      {
        v_noise.resize(size, tid);
      }

      if (model.visibles_type() == unit_type::Bernoulli) {
        v_rand.resize(size, tid);
      }

      #pragma omp master
      {
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
    } /* parallel */
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      cudaStreamSynchronize(0);
      #pragma omp barrier

      tensor_t& f = *v_f[tid];
      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];
      plan_t& plan_v = *v_plan_v[tid];
      plan_t& iplan_v = *v_iplan_v[tid];
      plan_t& iplan_h = *v_iplan_h[tid];

      dim_t fullsize = cv.fullsize();
      tensor_t p;

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
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
          *model.hidden_bias()[k * _filter_batch_length + j] = h[seq(0,0,0,j),layer_size];
        }
      }

      #pragma omp master
      {
        f = ifft(cb, dimCount - 1, iplan_v);
        f = f * (abs(f) > 1e-16);
        host_tensor_t b = f;
        model.set_visible_bias(b);
      }
    }
  }

  void free_gpu_memory() {
    if (!_host_updated)
      write_model_to_host();

    _memory_allocated = false;

    setup_threads();

    cb = ctensor_t();
    cbinc = ctensor_t();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      cudaStreamSynchronize(0);
      #pragma omp barrier

      v_v[tid] = boost::shared_ptr<tensor_t>();
      v_f[tid] = boost::shared_ptr<tensor_t>();
      v_h[tid] = boost::shared_ptr<tensor_t>();
      v_v_mask[tid] = boost::shared_ptr<tensor_t>();
      v_h_mask[tid] = boost::shared_ptr<tensor_t>();
      v_cv[tid] = boost::shared_ptr<ctensor_t>();
      v_ch[tid] = boost::shared_ptr<ctensor_t>();
      v_chdiff[tid] = boost::shared_ptr<ctensor_t>();

      v_plan_v[tid] = boost::shared_ptr<plan_t>();
      v_iplan_v[tid] = boost::shared_ptr<plan_t>();
      v_plan_h[tid] = boost::shared_ptr<plan_t>();
      v_iplan_h[tid] = boost::shared_ptr<plan_t>();

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        cF[k] = cc[k] = cFinc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
        drops[k] = boost::shared_ptr<tensor_t>();
      }

      v_v_rand[tid] = boost::shared_ptr<uniform_t>();
      v_h_rand[tid] = boost::shared_ptr<uniform_t>();
      v_v_noise[tid] = boost::shared_ptr<normal_t>();
      v_h_noise[tid] = boost::shared_ptr<normal_t>();

      cudaStreamSynchronize(0);
    }

    if (_memory_allocated && _gpu_count > 1)
      disable_peer_access(_gpu_count);
  }

  void normalize_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v_v[0]->size() == visible_size);

    tensor_t& v = *v_v[0];
    tensor_t& v_mask = *v_v_mask[0];
    v = ((v - model.mean()) / model.stddev()) * tbblas::repeat(v_mask, size / layer_size);
  }

  void diversify_visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v_v[0]->size() == visible_size);

    tensor_t& v = *v_v[0];
    tensor_t& v_mask = *v_v_mask[0];
    v = ((v * model.stddev()) + model.mean()) * tbblas::repeat(v_mask, size / layer_size);
  }

  void infer_visibles(bool onlyFilters = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    setup_threads();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& v = *v_v[tid];
      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];

      plan_t& iplan_v = *v_iplan_v[tid];
      plan_t& plan_h = *v_plan_h[tid];
      tensor_t& h_mask = *v_h_mask[tid];
      tensor_t& v_mask = *v_v_mask[tid];

      cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        h = zeros<value_t>(layer_batch_size);
        h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
        h = h * repeat(h_mask, h.size() / h_mask.size());
        ch = fft(h, dimCount - 1, plan_h);
        cv += repeat_mult_sum(ch, *cF[k]);
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier

      #pragma omp critical
      {
        if (tid != 0)
          *v_cv[0] = *v_cv[0] + cv;
        else if (!onlyFilters)
          cv += cb;
        cudaStreamSynchronize(0);
      }
      #pragma omp barrier

      #pragma omp master
      {
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
        cudaStreamSynchronize(0);
      }
      #pragma omp barrier
    }
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v_v[0]->size() == visible_size);
    setup_threads();

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];
      tensor_t& h_mask = *v_h_mask[tid];

      plan_t& iplan_h = *v_iplan_h[tid];

      cudaStreamSynchronize(0);
      #pragma omp barrier

      if (tid != 0)
        cv = *v_cv[0];

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
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
      cudaStreamSynchronize(0);
      #pragma omp barrier
    }
  }

  void sample_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    setup_threads();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& v = *v_v[tid];
      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];

      plan_t& iplan_v = *v_iplan_v[tid];
      plan_t& plan_h = *v_plan_h[tid];
      tensor_t& h_mask = *v_h_mask[tid];
      tensor_t& v_mask = *v_v_mask[tid];

      uniform_t& v_rand = *v_v_rand[tid];
      normal_t& v_noise = *v_v_noise[tid];

      cv = zeros<complex_t>(cF[0]->size() / seq(1,1,1,_filter_batch_length), cF[0]->fullsize() / seq(1,1,1,_filter_batch_length));

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        h = zeros<value_t>(layer_batch_size);
        h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
        h = h * repeat(h_mask, h.size() / h_mask.size());
        ch = fft(h, dimCount - 1, plan_h);
        cv += repeat_mult_sum(ch, *cF[k]);
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier

      #pragma omp critical
      {
        if (tid != 0)
          *v_cv[0] = *v_cv[0] + cv;
        else
          cv += cb;
        cudaStreamSynchronize(0);
      }
      #pragma omp barrier

      #pragma omp master
      {
        v = ifft(cv, dimCount - 1, iplan_v);

        switch(model.visibles_type()) {
          case unit_type::Gaussian:  break;
          case unit_type::Bernoulli: v = sigm(v) > v_rand; break;
          case unit_type::MyReLU:
          case unit_type::ReLU:      v = max(0.0, v + sqrt(sigm(v)) * v_noise); break;
          case unit_type::ReLU1:     v = min(1.0, max(0.0, v + (v > 0) * (v < 1.0) * v_noise)); break;
          case unit_type::ReLU2:     v = min(2.0, max(0.0, v + (v > 0) * (v < 2.0) * v_noise)); break;
          case unit_type::ReLU4:     v = min(4.0, max(0.0, v + (v > 0) * (v < 4.0) * v_noise)); break;
          case unit_type::ReLU8:     v = min(8.0, max(0.0, v + (v > 0) * (v < 8.0) * v_noise)); break;
        }
        v = v * repeat(v_mask, size / layer_size);
        cudaStreamSynchronize(0);
      }
      #pragma omp barrier
    }
  }

  void sample_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v_v[0]->size() == visible_size);

    setup_threads();

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];
      tensor_t& h_mask = *v_h_mask[tid];

      plan_t& iplan_h = *v_iplan_h[tid];

      uniform_t& h_rand = *v_h_rand[tid];
      normal_t& h_noise = *v_h_noise[tid];

      cudaStreamSynchronize(0);
      #pragma omp barrier

      if (tid != 0)
        cv = *v_cv[0];

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        ch = conj_mult_sum(cv, *cF[k]);

        ch = ch + *cc[k];
        h = ifft(ch, dimCount - 1, iplan_h);

        switch (model.hiddens_type()) {
          case unit_type::Bernoulli: h = sigm(h) > h_rand; break;
          case unit_type::MyReLU:
          case unit_type::ReLU:      h = max(0.0, h + sqrt(sigm(h)) * h_noise); break;
          case unit_type::ReLU1:     h = min(1.0, max(0.0, h + (h > 0) * (h < 1.0) * h_noise)); break;
          case unit_type::ReLU2:     h = min(2.0, max(0.0, h + (h > 0) * (h < 2.0) * h_noise)); break;
          case unit_type::ReLU4:     h = min(4.0, max(0.0, h + (h > 0) * (h < 4.0) * h_noise)); break;
          case unit_type::ReLU8:     h = min(8.0, max(0.0, h + (h > 0) * (h < 8.0) * h_noise)); break;
        }
        if (_dropout_rate > 0)
          h = h * *drops[k] / (1. - _dropout_rate) * repeat(h_mask, h.size() / h_mask.size());
        else
          h = h * repeat(h_mask, h.size() / h_mask.size());
        _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
      }
      cudaStreamSynchronize(0);
      #pragma omp barrier
    }
  }

  void init_dropout(value_t rate, dropout_method method = dropout_method::DropUnit) {
    if (rate < 0 || rate >= 1)
      throw std::runtime_error("Drop out rate must be in [0,1).");

    if (method == dropout_method::NoDrop) {
      _dropout_rate = 0;
      return;
    }

    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    _dropout_rate = rate;

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      uniform_t& h_rand = *v_h_rand[tid];

      if (h_rand.size() != layer_batch_size)
        h_rand.resize(layer_batch_size, tid);

      if (method == dropout_method::DropColumn) {
        #pragma omp master
        {
          *drops[0] = h_rand > _dropout_rate;
          *drops[0] = repeat((*drops[0])[seq(0,0,0,0),layer_size], layer_batch_size / layer_size);
        }
        cudaStreamSynchronize(0);
        #pragma omp barrier

        if (tid != 0)
          *drops[tid] = *drops[0];

        for (size_t k = tid + _gpu_count; k < drops.size(); k += _gpu_count)
          *drops[k] = *drops[tid];
      } else {
        for (size_t k = tid; k < drops.size(); k += _gpu_count)
          *drops[k] = h_rand > _dropout_rate;
      }
    }
  }

  void init_gradient_updates(value_t momentum = 0, value_t weightcost = 0) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    cbinc = momentum * cbinc;

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        *cFinc[k] = momentum * *cFinc[k] - weightcost * *cF[k];
        *ccinc[k] = momentum * *ccinc[k];
      }
    }
  }

  void update_positive_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    // TODO: how much time does this take?
    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];
      ctensor_t& chdiff = *v_chdiff[tid];
      plan_t& plan_h = *v_plan_h[tid];

      // TODO: how much time does this take?
      if (tid != 0)
        cv = *v_cv[0];

      #pragma omp master
      cbinc = cbinc + epsilonvb * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {

        h = zeros<value_t>(layer_batch_size);
        h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
        ch = fft(h, dimCount - 1, plan_h);

        *cFinc[k] += conj_repeat_mult(cv, ch, epsilonw);
        *ccinc[k] = *ccinc[k] + epsilonhb * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
        switch(_sparsity_method) {
        case sparsity_method::WeightsAndBias:
          chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
          *cFinc[k] = *cFinc[k] + epsilonw * _sparsity_weight * repeat(conj(chdiff), cFinc[k]->size() / ch.size()) * repeat(cv, cFinc[k]->size() / cv.size());
          *ccinc[k] = *ccinc[k] + epsilonhb * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
          break;

        case sparsity_method::OnlyBias:
          chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
          *ccinc[k] = *ccinc[k] + epsilonhb * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
          break;

        case sparsity_method::OnlySharedBias:
          *ccinc[k] = *ccinc[k] + epsilonhb * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) * (_sparsity_target * h.count() + -ch);
          break;
        }
      }
    }
  }

  void update_negative_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];
      plan_t& plan_h = *v_plan_h[tid];

      // TODO: how much time does this take?
      if (tid != 0)
        cv = *v_cv[0];

      #pragma omp master
      cbinc = cbinc - epsilonvb * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {

        h = zeros<value_t>(layer_batch_size);
        h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
        ch = fft(h, dimCount - 1, plan_h);

        *cFinc[k] += conj_repeat_mult(cv, ch, -epsilonw);
        *ccinc[k] = *ccinc[k] - epsilonhb * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
      }
    }
  }

  void apply_gradient() {
    _host_updated = false;

    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    setup_threads();

    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      cudaSetDevice(tid % _device_count);

      tensor_t& f = *v_f[tid];
      tensor_t& h = *v_h[tid];
      ctensor_t& cv = *v_cv[tid];
      ctensor_t& ch = *v_ch[tid];
      plan_t& plan_v = *v_plan_v[tid];
      plan_t& iplan_v = *v_iplan_v[tid];

      dim_t fullsize = cv.fullsize();

      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
        // Mask filters
        for (int j = 0; j < _filter_batch_length; ++j) {
          cv = (*cFinc[k])[seq(0,0,0,j*cv.size()[3]), cv.size()];
          cv.set_fullsize(fullsize);
          f = ifft(cv, dimCount - 1, iplan_v);
          f = f * tbblas::mask<value_t>(f.size(), model.kernel_size());
          cv = fft(f, dimCount - 1, plan_v);
          (*cFinc[k])[seq(0,0,0,j*cv.size()[3]), cv.size()] = cv;
        }

        *cF[k] = *cF[k] + *cFinc[k];
        *cc[k] = *cc[k] + *ccinc[k];
      }

      #pragma omp master
      cb = cb + cbinc;
    }
  }

  // Access to model data
  tensor_t& visibles() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return *v_v[0];
  }

  void allocate_visibles() {
    if (v_v[0]->size() != model.visibles_size())
      v_v[0]->resize(model.visibles_size());
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

  void set_batch_length(int length) {
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

private:
  void setup_threads() {
    cudaGetDeviceCount(&_device_count);

    assert (_device_count >= _gpu_count);
    assert(omp_get_num_threads() == 1);

    cudaSetDevice(0);
    omp_set_dynamic(0);
    omp_set_num_threads(_gpu_count);
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_HPP_ */
