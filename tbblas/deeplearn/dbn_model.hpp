/*
 * dbn_model.hpp
 *
 *  Created on: Jul 10, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_DBN_MODEL_HPP_
#define TBBLAS_DEEPLEARN_DBN_MODEL_HPP_

#include <tbblas/deeplearn/rbm_model.hpp>

#include <tbblas/io.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

template<class T>
class dbn_model {
public:
  typedef T value_t;

  typedef rbm_model<T> rbm_t;
  typedef std::vector<boost::shared_ptr<rbm_t> > v_rbm_t;

protected:
  v_rbm_t _rbms;

public:
  dbn_model() { }

  dbn_model(const dbn_model<T>& model) {
    set_rbms(model.rbms());
  }

  virtual ~dbn_model() { }

public:
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
