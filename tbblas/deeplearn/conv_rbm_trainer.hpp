/*
 * conv_rbm_trainer.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONV_RBM_TRAINER_HPP_
#define TBBLAS_DEEPLEARN_CONV_RBM_TRAINER_HPP_

#include <tbblas/deeplearn/conv_rbm.hpp>
#include <tbblas/deeplearn/sparsity_method.hpp>

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
class conv_rbm_trainer : public conv_rbm<T, dims> {
  typedef conv_rbm<T, dims> base_t;

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

  using base_t::_gpu_count;
  using base_t::_device_count;
  using base_t::_filter_batch_length;
  using base_t::_hidden_count;
  using base_t::_hiddens;
  using base_t::_memory_allocated;
  using base_t::cb;
  using base_t::cc;
  using base_t::cF;
  using base_t::hbMaskSize;
  using base_t::vbMaskSize;
  using base_t::hidden_layer_batch_size;
  using base_t::hidden_topleft;
  using base_t::layer_batch_size;
  using base_t::layer_size;
  using base_t::model;
  using base_t::size;
  using base_t::v_ch;
  using base_t::v_cv;
  using base_t::v_f;
  using base_t::v_h;
  using base_t::v_v;
  using base_t::v_iplan_v;
  using base_t::v_iplan_h;
  using base_t::v_plan_v;
  using base_t::v_plan_h;
  using base_t::v_stream;

protected:
  // gradients in GPU memory
  ctensor_t cbinc;
  v_ctensor_t cFinc, ccinc;

  // visible and hidden units in GPU memory

  // Sizes
  dim_t spMaskSize;

  // one element per thread
  std::vector<boost::shared_ptr<ctensor_t> > v_chdiff;

  bool _host_updated;

  value_t _sparsity_target, _sparsity_weight;
  sparsity_method _sparsity_method;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm_trainer(model_t& model, size_t gpu_count = 1) : base_t(model, gpu_count),
    _host_updated(true), _sparsity_target(0.1), _sparsity_weight(0),
    _sparsity_method(sparsity_method::OnlySharedBias)
  {
    v_chdiff.resize(gpu_count);
  }

private:
  conv_rbm_trainer(const conv_rbm_trainer<T, dims>&);

public:
  // Automatically frees GPU memory
  virtual ~conv_rbm_trainer() {
    if (_memory_allocated)
      free_gpu_memory();
  }

  // These functions can run in parallel. They also create threads

  /// Transforms
  void allocate_gpu_memory() {
    using namespace tbblas;

    if (_memory_allocated)
      return;

    base_t::allocate_gpu_memory();

    // Prepare sizes
    cFinc.resize(cF.size());
    ccinc.resize(cc.size());

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::allocate_gpu_memory_parallel, this, i);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      allocate_gpu_memory_parallel(0);
    }
  }

  void write_model_to_host() {
    using namespace tbblas;

    if (_host_updated)
      return;

    _host_updated = true;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::write_model_to_host_parallel, this, i);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      write_model_to_host_parallel(0);
    }
  }

  void free_gpu_memory() {
    if (!_host_updated)
      write_model_to_host();

    cbinc = ctensor_t();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::free_gpu_memory_parallel, this, i);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      free_gpu_memory_parallel(0);
    }

    base_t::free_gpu_memory();
  }

  void init_gradient_updates(value_t momentum = 0, value_t weightcost = 0) {
    if (!_memory_allocated)
      allocate_gpu_memory();

    cbinc = momentum * cbinc;

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::init_gradient_updates_parallel, this, i, momentum, weightcost);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      init_gradient_updates_parallel(0, momentum, weightcost);
    }
  }

  void update_positive_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    // TODO: how much time does this take?
    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);
    if (_gpu_count > 1)
      tbblas::synchronize();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::update_positive_gradient_parallel, this, i, epsilonw, epsilonvb, epsilonhb);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      update_positive_gradient_parallel(0, epsilonw, epsilonvb, epsilonhb);
    }
  }

  void update_negative_gradient(value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);
    if (_gpu_count > 1)
      tbblas::synchronize();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::update_negative_gradient_parallel, this, i, epsilonw, epsilonvb, epsilonhb);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      update_negative_gradient_parallel(0, epsilonw, epsilonvb, epsilonhb);
    }
  }

  // CAUTION: ONLY USE THIS FUNCTION IF YOU KNOW WHAT YOU ARE DOING
  // Should probably check, if the model is the same and needs a don't
  // write to model mechanism
  void accumulate_gradients(conv_rbm_trainer<value_t, dimCount>& trainer) {
    assert(_gpu_count == 1);

    for (size_t k = 0; k < cFinc.size(); ++k) {
      *trainer.cFinc[k] = *cFinc[k] = *cFinc[k] + *trainer.cFinc[k];
      *trainer.ccinc[k] = *ccinc[k] = *ccinc[k] + *trainer.ccinc[k];
    }

    trainer.cbinc = cbinc = cbinc + trainer.cbinc;
  }

  void apply_gradient() {
    _host_updated = false;

    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm_trainer<T,dims>::apply_gradient_parallel, this, i);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      apply_gradient_parallel(0);
    }
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
  void allocate_gpu_memory_parallel(int tid) {
    if (_gpu_count > 1) {
      cudaSetDevice(tid % _device_count);
      cudaStreamCreate(&v_stream[tid]);
    }

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    v_chdiff[tid] = boost::make_shared<ctensor_t>();

//    plan_t& plan_v = *(v_plan_v[tid] = boost::make_shared<plan_t>());
//    plan_t& iplan_v = *(v_iplan_v[tid] = boost::make_shared<plan_t>());
//    plan_t& plan_h = *(v_plan_h[tid] = boost::make_shared<plan_t>());
//    plan_t& iplan_h = *(v_iplan_h[tid] = boost::make_shared<plan_t>());
//
//    tensor_t& v_mask = *(v_v_mask[tid] = boost::make_shared<tensor_t>());
//    tensor_t& h_mask = *(v_h_mask[tid] = boost::make_shared<tensor_t>());
//    v_mask = zeros<value_t>(layer_size);
//    v_mask[seq<dimCount>(0), model.mask().size()] = model.mask();
//
//    // pad h mask according to convolution shrinkage
//    if (model.convolution_type() == convolution_type::Valid){
//      h_mask = zeros<value_t>(layer_size);
//      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
//      h_mask = h_mask * v_mask;
//    } else {
//      h_mask = v_mask;
//    }

    if (tid == 0) {
//      tensor_t b = model.visible_bias();
//      cb = fft(b, dimCount - 1, plan_v);
      cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
    }

    // Copy filters to the device and pre-calculate the FFT
    {
//      tensor_t f, h, kern, pad;
      ctensor_t cf, ch;
//      plan_t plan_f;
      for (size_t k = tid; k < cF.size(); k += _gpu_count) {
//        pad = zeros<value_t>(filter_batch_size);
//        for (size_t j = 0; j < _filter_batch_length; ++j) {
//          kern = *model.filters()[k * _filter_batch_length + j];
//          dim_t topleft = size / 2 - kern.size() / 2;
//          topleft[dimCount - 1] = j * size[dimCount - 1];
//          pad[topleft, kern.size()] = kern;
//        }
//        f = ifftshift(pad, dimCount - 1);
//        cf = fft(f, dimCount - 1, plan_f);
//        cF[k] = boost::make_shared<ctensor_t>(cf);
        cf = zeros<complex_t>(cF[k]->size(), cF[k]->fullsize());
        cFinc[k] = boost::make_shared<ctensor_t>(cf);

//        h = zeros<value_t>(layer_batch_size);
//        for (int j = 0; j < _filter_batch_length; ++j) {
//          h[seq(0,0,0,j), visible_layer_size] = *model.hidden_bias()[k * _filter_batch_length + j];
//        }
//        ch = fft(h, dimCount - 1, plan_h);
//        cc[k] = boost::make_shared<ctensor_t>(ch);
        ch = zeros<complex_t>(cc[k]->size(), cc[k]->fullsize());
        ccinc[k] = boost::make_shared<ctensor_t>(ch);

//        drops[k] = boost::make_shared<tensor_t>();
      }
    }

//    uniform_t& h_rand = *(v_h_rand[tid] = boost::make_shared<uniform_t>());
//    uniform_t& v_rand = *(v_v_rand[tid] = boost::make_shared<uniform_t>());
//    normal_t& h_noise = *(v_h_noise[tid] = boost::make_shared<normal_t>());
//    normal_t& v_noise = *(v_v_noise[tid] = boost::make_shared<normal_t>());
//
//    if (model.hiddens_type() == unit_type::Bernoulli)
//      h_rand.resize(layer_batch_size, tid);
//
//    if (model.hiddens_type() == unit_type::MyReLU ||
//        model.hiddens_type() == unit_type::ReLU ||
//        model.hiddens_type() == unit_type::ReLU1 ||
//        model.hiddens_type() == unit_type::ReLU2 ||
//        model.hiddens_type() == unit_type::ReLU4)
//    {
//      h_noise.resize(layer_batch_size, tid);
//    }
//
//    if (model.visibles_type() == unit_type::MyReLU ||
//        model.visibles_type() == unit_type::ReLU ||
//        model.visibles_type() == unit_type::ReLU1 ||
//        model.visibles_type() == unit_type::ReLU2 ||
//        model.visibles_type() == unit_type::ReLU4)
//    {
//      v_noise.resize(size, tid);
//    }
//
//    if (model.visibles_type() == unit_type::Bernoulli) {
//      v_rand.resize(size, tid);
//    }

    if (tid == 0) {
//      vbMaskSize = cb.size();
//      if (model.shared_bias()) {
//        vbMaskSize[0] = 1;
//        vbMaskSize[1] = 1;
//        vbMaskSize[2] = 1;
//      }
//
//      hbMaskSize = cc[0]->size();
//      if (model.shared_bias()) {
//        hbMaskSize[0] = 1;
//        hbMaskSize[1] = 1;
//        hbMaskSize[2] = 1;
//      }

      spMaskSize = cc[0]->size();
      spMaskSize[0] = 1;
      spMaskSize[1] = 1;
      spMaskSize[2] = 1;
    }
    if (_gpu_count > 1)
      tbblas::synchronize();
  }

  void write_model_to_host_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    tensor_t& f = *v_f[tid];
    tensor_t& h = *v_h[tid];
    ctensor_t& cv = *v_cv[tid];
//    ctensor_t& ch = *v_ch[tid];
//    plan_t& plan_v = *v_plan_v[tid];
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
        *model.hidden_bias()[k * _filter_batch_length + j] = h[seq(0,0,0,j), layer_size];
      }
    }

    if (tid == 0) {
      f = ifft(cb, dimCount - 1, iplan_v);
      f = f * (abs(f) > 1e-16);
      host_tensor_t b = f;
      model.set_visible_bias(b);
    }

    tbblas::synchronize();
  }

  void free_gpu_memory_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    v_chdiff[tid] = boost::shared_ptr<ctensor_t>();

    for (size_t k = tid; k < cFinc.size(); k += _gpu_count) {
      cFinc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
    }

    if (_gpu_count > 1) {
      tbblas::synchronize();
      cudaStreamDestroy(v_stream[tid]);
    }
  }

  void init_gradient_updates_parallel(size_t tid, value_t momentum, value_t weightcost) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    for (size_t k = tid; k < cFinc.size(); k += _gpu_count) {
      *cFinc[k] = momentum * *cFinc[k] - weightcost * *cF[k];
      *ccinc[k] = momentum * *ccinc[k];
    }
  }

  void update_positive_gradient_parallel(size_t tid, value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    tensor_t& h = *v_h[tid];
    ctensor_t& cv = *v_cv[tid];
    ctensor_t& ch = *v_ch[tid];
    ctensor_t& chdiff = *v_chdiff[tid];
    plan_t& plan_h = *v_plan_h[tid];

    // TODO: how much time does this take?
    if (tid != 0)
      cv = *v_cv[0];

    if (tid == 0)
      cbinc = cbinc + epsilonvb / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = tid; k < cFinc.size(); k += _gpu_count) {

      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, epsilonw / _hidden_count);
      *ccinc[k] = *ccinc[k] + epsilonhb / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
      switch(_sparsity_method) {
      case sparsity_method::WeightsAndBias:
        chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
        *cFinc[k] = *cFinc[k] + epsilonw / _hidden_count * _sparsity_weight * repeat(conj(chdiff), cFinc[k]->size() / ch.size()) * repeat(cv, cFinc[k]->size() / cv.size());
        *ccinc[k] = *ccinc[k] + epsilonhb / model.visibles_size()[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
        break;

      case sparsity_method::OnlyBias:
        chdiff = _sparsity_target * h.count() * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) - ch;
        *ccinc[k] = *ccinc[k] + epsilonhb / model.visibles_size()[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * chdiff;
        break;

      case sparsity_method::OnlySharedBias:
        *ccinc[k] = *ccinc[k] + epsilonhb / model.visibles_size()[dimCount - 1] * _sparsity_weight * mask<complex_t>(ch.size(), ch.fullsize(), spMaskSize) * (_sparsity_target * h.count() + -ch);
        break;
      }
    }
  }

  void update_negative_gradient_parallel(size_t tid, value_t epsilonw, value_t epsilonvb, value_t epsilonhb) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    tensor_t& h = *v_h[tid];
    ctensor_t& cv = *v_cv[tid];
    ctensor_t& ch = *v_ch[tid];
    plan_t& plan_h = *v_plan_h[tid];

    // TODO: how much time does this take?
    if (tid != 0)
      cv = *v_cv[0];

    if (tid == 0)
      cbinc = cbinc - epsilonvb / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(cv.size(), cv.fullsize(), vbMaskSize) * cv;

    for (size_t k = tid; k < cF.size(); k += _gpu_count) {

      h = zeros<value_t>(layer_batch_size);
      h[hidden_topleft, hidden_layer_batch_size] = _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size];
      ch = fft(h, dimCount - 1, plan_h);

      *cFinc[k] += conj_repeat_mult(cv, ch, -epsilonw / _hidden_count);
      *ccinc[k] = *ccinc[k] - epsilonhb / model.visibles_size()[dimCount - 1] * tbblas::mask<complex_t>(ch.size(), ch.fullsize(), hbMaskSize) * ch;
    }
  }

  void apply_gradient_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

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

    if (tid == 0)
      cb = cb + cbinc;
  }
};

template<class T, unsigned dims>
const T conv_rbm_trainer<T, dims>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_TRAINER_HPP_ */
