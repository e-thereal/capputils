/*
 * dbn_model.hpp
 *
 *  Created on: Jul 10, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DBN_MODEL_HPP_
#define TBBLAS_DEEPLEARN_DBN_MODEL_HPP_

#include <tbblas/deeplearn/conv_rbm_model.hpp>
#include <tbblas/deeplearn/rbm_model.hpp>

#include <tbblas/io.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class dbn_model {
public:
  typedef T value_t;
  static const unsigned dimCount = dims;
  typedef typename tensor<value_t, dimCount>::dim_t dim_t;

  typedef conv_rbm_model<T, dims> crbm_t;
  typedef rbm_model<T> rbm_t;

  typedef std::vector<boost::shared_ptr<crbm_t> > v_crbm_t;
  typedef std::vector<boost::shared_ptr<rbm_t> > v_rbm_t;

protected:
  v_crbm_t _crbms;
  v_rbm_t _rbms;

public:
  dbn_model() { }

  dbn_model(const dbn_model<T, dims>& model) {
    set_crbms(model.crbms());
    set_rbms(model.rbms());
  }

  virtual ~dbn_model() { }

public:
  template<class U>
  void set_crbms(const std::vector<boost::shared_ptr<conv_rbm_model<U, dims> > >& crbms) {
    _crbms.resize(crbms.size());
    for (size_t i = 0; i < crbms.size(); ++i) {
      _crbms[i] = boost::make_shared<crbm_t>(*crbms[i]);
      stride_size(i);
    }
  }

  v_crbm_t& crbms() {
    return _crbms;
  }

  const v_crbm_t& crbms() const {
    return _crbms;
  }

  dim_t stride_size(int layer) {
    if (layer < 0 || layer >= (int)_crbms.size())
      throw std::runtime_error("Invalid layer specified.");
    if (layer == 0)
      return _crbms[0]->stride_size();

    dim_t block = _crbms[layer - 1]->hiddens_size() / _crbms[layer]->visibles_size();
    block[dimCount - 1] = 1;
    if (block != _crbms[layer]->stride_size()) {
      std::cout << "[Warning] Stride size does not match calculated stride size." << std::endl;
      tbblas_print(block);
      tbblas_print(_crbms[layer]->stride_size());
//      throw std::runtime_error("Stride size does not match calculated stride size.");
    }
    return block;
  }

  template<class U>
  void append_crbm(const conv_rbm_model<U, dims>& crbm) {
    _crbms.push_back(boost::make_shared<crbm_t>(crbm));
    stride_size(_crbms.size() - 1);
  }

  template<class U>
  void set_rbms(const std::vector<boost::shared_ptr<rbm_model<U> > >& rbms) {
    _rbms.resize(rbms.size());
    for (size_t i = 0; i < rbms.size(); ++i)
      _rbms[i] = boost::make_shared<rbm_t>(*rbms[i]);
  }

  v_rbm_t& rbms() {
    return _rbms;
  }

  const v_rbm_t& rbms() const {
    return _rbms;
  }

  template<class U>
  void append_rbm(const rbm_model<U>& rbm) {
    _rbms.push_back(boost::make_shared<rbm_t>(rbm));
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_DBN_MODEL_HPP_ */
