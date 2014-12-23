/*
 * conv_rbm_trainer.hpp
 *
 *  Created on: Mar 26, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CONV_RBM_TRAINER_HPP_
#define TBBLAS_DEEPLEARN_CONV_RBM_TRAINER_HPP_

// TODO: no separated trainer module. Fuse it into the conv_rbm class with lazy initialization of variables

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

    cbinc = zeros<complex_t>(cb.size(), cb.fullsize());
    for (size_t k = tid; k < cF.size(); k += _gpu_count) {
      cFinc[k] = boost::make_shared<ctensor_t>(zeros<complex_t>(cF[k]->size(), cF[k]->fullsize()));
      ccinc[k] = boost::make_shared<ctensor_t>(zeros<complex_t>(cc[k]->size(), cc[k]->fullsize()));
    }

    spMaskSize = cc[0]->size();
    spMaskSize[0] = 1;
    spMaskSize[1] = 1;
    spMaskSize[2] = 1;
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

      h = ifft(*cc[k], dimCount - 1, iplan_h);
      h = h * (abs(h) > 1e-16);

      for (int j = 0; j < _filter_batch_length; ++j) {
        *model.hidden_bias()[k * _filter_batch_length + j] = h[seq(0,0,0,j), layer_size];
      }
    }

    f = ifft(cb, dimCount - 1, iplan_v);
    f = f * (abs(f) > 1e-16);
    host_tensor_t b = f;
    model.set_visible_bias(b);
  }

  void free_gpu_memory() {
    if (!_host_updated)
      write_model_to_host();

    cbinc = ctensor_t();

    for (size_t k = 0; k < cFinc.size(); ++k) {
      cFinc[k] = ccinc[k] = boost::shared_ptr<ctensor_t>();
    }

    base_t::free_gpu_memory();
  }


};

template<class T, unsigned dims>
const T conv_rbm_trainer<T, dims>::tolerance;

}

}

#endif /* TBBLAS_DEEPLEARN_CONV_RBM_TRAINER_HPP_ */
