/*
 * conv_rbm.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONV_RBM_HPP_
#define TBBLAS_DEEPLEARN_CONV_RBM_HPP_

// TODO: lazy initialisation of variables used for training
// TODO: momentum_step and adadelta_step replace init_gradient and apply_gradient
// TODO: counter in update_gradient

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
#include <tbblas/change_stream.hpp>
#include <tbblas/context.hpp>

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
  typedef std::vector<boost::shared_ptr<ctensor_t> > v_ctensor_t;

  typedef tbblas::random_tensor2<value_t, dimCount, true, tbblas::uniform<value_t> > uniform_t;
  typedef tbblas::random_tensor2<value_t, dimCount, true, tbblas::normal<value_t> > normal_t;

  typedef conv_rbm_model<value_t, dimCount> model_t;

  static const value_t tolerance = 1e-8;

protected:
  // Model in CPU memory
  model_t& model;

  // weights and bias terms in GPU memory
  ctensor_t cb;
  v_ctensor_t cF, cc;
  v_tensor_t drops;

  // visible and hidden units in GPU memory

  // Sizes
  dim_t visible_size, hidden_size, size,
        visible_layer_size, hidden_layer_size, layer_size,
        hidden_layer_batch_size, layer_batch_size,
        filter_batch_size, hidden_topleft,
        vbMaskSize, hbMaskSize;

  // one element per thread
  std::vector<boost::shared_ptr<tensor_t> > v_v, v_h, v_f, v_v_mask, v_h_mask;
  std::vector<boost::shared_ptr<ctensor_t> > v_cv, v_ch;
  std::vector<boost::shared_ptr<plan_t> > v_plan_v, v_iplan_v, v_plan_h, v_iplan_h;
  std::vector<boost::shared_ptr<uniform_t> > v_v_rand, v_h_rand;
  std::vector<boost::shared_ptr<normal_t> > v_v_noise, v_h_noise;
  std::vector<cudaStream_t> v_stream;

  tensor_t _hiddens, _pooled;

  int _gpu_count, _device_count, _filter_batch_length, _hidden_count;
  bool _memory_allocated, _double_weights;

  value_t _dropout_rate;

  boost::barrier _barrier;
  boost::mutex _mtx;

public:
  /// Creates a new conv_rbm layer (called from non-parallel code)
  conv_rbm(model_t& model, size_t gpu_count = 1) : model(model),
    _gpu_count(gpu_count), _device_count(0), _filter_batch_length(1),
    _memory_allocated(false), _double_weights(false),
    _dropout_rate(0), _barrier(gpu_count)
  {
    cudaGetDeviceCount(&_device_count);

    assert(_gpu_count > 0);
    assert (_device_count >= _gpu_count);

    v_v.resize(gpu_count);
    v_f.resize(gpu_count);
    v_h.resize(gpu_count);
    v_v_mask.resize(gpu_count);
    v_h_mask.resize(gpu_count);
    v_cv.resize(gpu_count);
    v_ch.resize(gpu_count);

    v_plan_v.resize(gpu_count);
    v_iplan_v.resize(gpu_count);
    v_plan_h.resize(gpu_count);
    v_iplan_h.resize(gpu_count);

    v_v_rand.resize(gpu_count);
    v_h_rand.resize(gpu_count);
    v_v_noise.resize(gpu_count);
    v_h_noise.resize(gpu_count);

    v_stream.resize(gpu_count);

    _hidden_count = sum(model.mask()) * model.hiddens_size()[dimCount - 1];
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

    // Prepare sizes
    size =visible_size = model.visibles_size();
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
      hidden_topleft = seq<dimCount>(0);
    }

    _hiddens = zeros<value_t>(hidden_size);

#ifndef TBBLAS_CONV_RBM_NO_SELFTEST
    // Test if the FFT bug is gonna bug us ;)
    {
      random_tensor2<value_t, dimCount, true, normal<value_t> > v_noise(size);

      tensor_t A = v_noise(), B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
    {
      random_tensor2<value_t, dimCount, true, normal<value_t> > v_noise(layer_batch_size);

      tensor_t A = v_noise(), B = A;
      ctensor_t cA = fft(A, dimCount - 1), cB = cA;

      if (dot(A - B, A - B) != 0)
        throw std::runtime_error("Bug detected in cuFFT (forward transform)!");

      A = ifft(cA, dimCount - 1);
      if (abs(dot(cA - cB, cA - cB)) != 0)
        throw std::runtime_error("Bug detected in cuFFT (backward transform)!");
    }
#endif

    cF.resize(model.filters().size() / _filter_batch_length);
    cc.resize(model.hidden_bias().size() / _filter_batch_length);

    drops.resize(cF.size());

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::allocate_gpu_memory_parallel, this, i);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      allocate_gpu_memory_parallel(0);
    }
  }

  void free_gpu_memory() {
    cb = ctensor_t();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::free_gpu_memory_parallel, this, i);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      free_gpu_memory_parallel(0);
    }

    if (_memory_allocated && _gpu_count > 1)
      disable_peer_access(_gpu_count);

    _memory_allocated = false;
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

  void pool_hiddens() {
    switch (model.pooling_method()) {
    case pooling_method::NoPooling:
      _pooled = _hiddens;
      break;

    case pooling_method::StridePooling:
      _pooled = _hiddens[seq<dimCount>(0), model.pooling_size(), _hiddens.size()];
      break;

    default:
      throw std::runtime_error("Unsupported pooling method.");
    }
  }

  void unpool_hiddens() {
    switch (model.pooling_method()) {
    case pooling_method::NoPooling:
      _hiddens = _pooled;
      break;

    case pooling_method::StridePooling:
      allocate_hiddens();
      _hiddens[seq<dimCount>(0), model.pooling_size(), _hiddens.size()] = _pooled;
      break;

    default:
      throw std::runtime_error("Unsupported pooling method.");
    }
  }

  void infer_visibles(bool onlyFilters = false) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::infer_visibles_parallel, this, i, onlyFilters);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      infer_visibles_parallel(0, onlyFilters);
    }
  }

  void infer_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v_v[0]->size() == visible_size);

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);
    if (_gpu_count > 1)
      tbblas::synchronize();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::infer_hiddens_parallel, this, i);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      infer_hiddens_parallel(0);
    }
  }

  void sample_visibles() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(_hiddens.size() == hidden_size);

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::sample_visibles_parallel, this, i);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      sample_visibles_parallel(0);
    }
  }

  void sample_hiddens() {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    assert(v_v[0]->size() == visible_size);

    *v_cv[0] = tbblas::fft(*v_v[0], dimCount - 1, *v_plan_v[0]);

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::sample_hiddens_parallel, this, i);

      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      sample_hiddens_parallel(0);
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

    _dropout_rate = rate;

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::init_dropout_parallel, this, i, method);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      init_dropout_parallel(0, method);
    }
  }

  void set_input(tensor_t& input) {
    assert(model.input_size() == input.size());
    visibles() = rearrange(input, model.stride_size());
  }

  void get_input(tensor_t& input) {
    input = rearrange_r(visibles(), model.stride_size());
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

  tensor_t& pooled_units() {
    if (!_memory_allocated)
      allocate_gpu_memory();
    return _pooled;
  }

  void allocate_pooled_units() {
    if (_pooled.size() != model.pooled_size())
      _pooled.resize(model.pooled_size());
  }

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

  void change_mask(host_tensor_t& mask) {
    using namespace tbblas;

    if (!_memory_allocated)
      allocate_gpu_memory();

    if (_gpu_count > 1) {
      std::vector<boost::shared_ptr<boost::thread> > threads(_gpu_count);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i] = boost::make_shared<boost::thread>(&conv_rbm<T,dims>::change_mask_parallel, this, i, mask);
      for (size_t i = 0; i < threads.size(); ++i)
        threads[i]->join();
    } else {
      change_mask_parallel(0, mask);
    }
  }

private:
  void allocate_gpu_memory_parallel(int tid) {
    if (_gpu_count > 1) {
      cudaSetDevice(tid % _device_count);
      cudaStreamCreate(&v_stream[tid]);
    }

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    v_v[tid] = boost::make_shared<tensor_t>();
    v_f[tid] = boost::make_shared<tensor_t>();
    v_h[tid] = boost::make_shared<tensor_t>();
    v_cv[tid] = boost::make_shared<ctensor_t>();
    v_ch[tid] = boost::make_shared<ctensor_t>();

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

    if (tid == 0) {
      tensor_t b = model.visible_bias();
      cb = fft(b, dimCount - 1, plan_v);
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

        h = zeros<value_t>(layer_batch_size);
        for (int j = 0; j < _filter_batch_length; ++j) {
          h[seq(0,0,0,j), visible_layer_size] = *model.hidden_bias()[k * _filter_batch_length + j];
        }
        ch = fft(h, dimCount - 1, plan_h);
        cc[k] = boost::make_shared<ctensor_t>(ch);

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

    if (tid == 0) {
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
    }
    if (_gpu_count > 1)
      tbblas::synchronize();
  }

  void free_gpu_memory_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    v_v[tid] = boost::shared_ptr<tensor_t>();
    v_f[tid] = boost::shared_ptr<tensor_t>();
    v_h[tid] = boost::shared_ptr<tensor_t>();
    v_v_mask[tid] = boost::shared_ptr<tensor_t>();
    v_h_mask[tid] = boost::shared_ptr<tensor_t>();
    v_cv[tid] = boost::shared_ptr<ctensor_t>();
    v_ch[tid] = boost::shared_ptr<ctensor_t>();

    v_plan_v[tid] = boost::shared_ptr<plan_t>();
    v_iplan_v[tid] = boost::shared_ptr<plan_t>();
    v_plan_h[tid] = boost::shared_ptr<plan_t>();
    v_iplan_h[tid] = boost::shared_ptr<plan_t>();

    for (size_t k = tid; k < cF.size(); k += _gpu_count) {
      cF[k] = cc[k] = boost::shared_ptr<ctensor_t>();
      drops[k] = boost::shared_ptr<tensor_t>();
    }

    v_v_rand[tid] = boost::shared_ptr<uniform_t>();
    v_h_rand[tid] = boost::shared_ptr<uniform_t>();
    v_v_noise[tid] = boost::shared_ptr<normal_t>();
    v_h_noise[tid] = boost::shared_ptr<normal_t>();

    if (_gpu_count > 1) {
      tbblas::synchronize();
      cudaStreamDestroy(v_stream[tid]);
    }
  }

  void infer_visibles_parallel(size_t tid, bool onlyFilters) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

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
      if (!onlyFilters)
        h = h * repeat(h_mask, h.size() / h_mask.size());
      ch = fft(h, dimCount - 1, plan_h);
      cv += repeat_mult_sum(ch, *cF[k]);
    }
    if (_gpu_count > 1) {
      tbblas::synchronize();
      _barrier.wait();
    }

    {
      boost::lock_guard<boost::mutex> guard(_mtx);
      if (tid != 0)
        *v_cv[0] = *v_cv[0] + cv;
      else if (!onlyFilters)
        cv += cb;
      if (_gpu_count > 1)
        tbblas::synchronize();
    }
    if (_gpu_count > 1)
      _barrier.wait();

    if (tid == 0) {
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
      if (_gpu_count > 1)
        tbblas::synchronize();
    }
  }

  void infer_hiddens_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    tensor_t& h = *v_h[tid];
    ctensor_t& cv = *v_cv[tid];
    ctensor_t& ch = *v_ch[tid];
    tensor_t& h_mask = *v_h_mask[tid];

    plan_t& iplan_h = *v_iplan_h[tid];

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
    if (_gpu_count > 1)
      tbblas::synchronize();
  }

  void sample_visibles_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

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
    if (_gpu_count > 1) {
      tbblas::synchronize();
      _barrier.wait();
    }

    {
      boost::lock_guard<boost::mutex> guard(_mtx);
      if (tid != 0)
        *v_cv[0] = *v_cv[0] + cv;
      else
        cv += cb;
      if (_gpu_count > 1)
        tbblas::synchronize();
    }
    if (_gpu_count > 1)
      _barrier.wait();

    if (tid == 0) {
      v = ifft(cv, dimCount - 1, iplan_v);

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
      v = v * repeat(v_mask, size / layer_size);
      if (_gpu_count > 1)
        tbblas::synchronize();
    }
  }

  void sample_hiddens_parallel(size_t tid) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    tensor_t& h = *v_h[tid];
    ctensor_t& cv = *v_cv[tid];
    ctensor_t& ch = *v_ch[tid];
    tensor_t& h_mask = *v_h_mask[tid];

    plan_t& iplan_h = *v_iplan_h[tid];

    uniform_t& h_rand = *v_h_rand[tid];
    normal_t& h_noise = *v_h_noise[tid];

    if (tid != 0)
      cv = *v_cv[0];

    for (size_t k = tid; k < cF.size(); k += _gpu_count) {
      ch = conj_mult_sum(cv, *cF[k]);

      ch = ch + *cc[k];
      h = ifft(ch, dimCount - 1, iplan_h);

      switch (model.hiddens_type()) {
        case unit_type::Bernoulli: h = sigm(h) > h_rand(); break;
        case unit_type::MyReLU:
        case unit_type::ReLU:      h = max(0.0, h + sqrt(sigm(h)) * h_noise()); break;
        case unit_type::ReLU1:     h = min(1.0, max(0.0, h + (h > 0) * (h < 1.0) * h_noise())); break;
        case unit_type::ReLU2:     h = min(2.0, max(0.0, h + (h > 0) * (h < 2.0) * h_noise())); break;
        case unit_type::ReLU4:     h = min(4.0, max(0.0, h + (h > 0) * (h < 4.0) * h_noise())); break;
        case unit_type::ReLU8:     h = min(8.0, max(0.0, h + (h > 0) * (h < 8.0) * h_noise())); break;
      }
      if (_dropout_rate > 0)
        h = h * *drops[k] / (1. - _dropout_rate) * repeat(h_mask, h.size() / h_mask.size());
      else
        h = h * repeat(h_mask, h.size() / h_mask.size());
      _hiddens[seq(0,0,0,(int)k * _filter_batch_length), hidden_layer_batch_size] = h[hidden_topleft, hidden_layer_batch_size];
    }
    if (_gpu_count > 1)
      tbblas::synchronize();
  }

  void init_dropout_parallel(size_t tid, dropout_method method) {
    if (_gpu_count > 1)
      cudaSetDevice(tid % _device_count);

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    uniform_t& h_rand = *v_h_rand[tid];

    if (h_rand.size() != layer_batch_size)
      h_rand.resize(layer_batch_size, tid);

    if (method == dropout_method::DropColumn) {
      if (tid == 0) {
        *drops[0] = h_rand() > _dropout_rate;
        *drops[0] = repeat((*drops[0])[seq(0,0,0,0),layer_size], layer_batch_size / layer_size);
      }
      if (_gpu_count > 1) {
        tbblas::synchronize();
        _barrier.wait();
      }

      if (tid != 0)
        *drops[tid] = *drops[0];

      for (size_t k = tid + _gpu_count; k < drops.size(); k += _gpu_count)
        *drops[k] = *drops[tid];
    } else {
      for (size_t k = tid; k < drops.size(); k += _gpu_count)
        *drops[k] = h_rand() > _dropout_rate;
    }
  }

  void change_mask_parallel(int tid, host_tensor_t& mask) {
    if (_gpu_count > 1) {
      cudaSetDevice(tid % _device_count);
      cudaStreamCreate(&v_stream[tid]);
    }

    change_stream context(_gpu_count == 1 ? context::get().stream : v_stream[tid]);

    tensor_t& v_mask = *v_v_mask[tid];
    tensor_t& h_mask = *v_h_mask[tid];
    v_mask = zeros<value_t>(layer_size);
    v_mask[seq<dimCount>(0), mask.size()] = mask;

    // pad h mask according to convolution shrinkage
    if (model.convolution_type() == convolution_type::Valid){
      h_mask = zeros<value_t>(layer_size);
      h_mask[hidden_topleft, hidden_layer_size] = ones<value_t>(hidden_layer_size);
      h_mask = h_mask * v_mask;
    } else {
      h_mask = v_mask;
    }

    if (_gpu_count > 1)
      tbblas::synchronize();
  }
};

template<class T, unsigned dims>
const T conv_rbm<T, dims>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_HPP_ */
