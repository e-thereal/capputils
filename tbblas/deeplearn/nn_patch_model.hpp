/*
 * nn_patch_model.hpp
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_NN_PATCH_MODEL_HPP_
#define TBBLAS_DEEPLEARN_NN_PATCH_MODEL_HPP_

#include <tbblas/deeplearn/nn_model.hpp>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dim>
class nn_patch_model {
public:
  typedef T value_t;

  static const int dimCount = dim;

  typedef nn_model<value_t> model_t;
  typedef typename tensor<value_t, dimCount>::dim_t dim_t;

protected:
  model_t _model;
  dim_t _patch_size;
  value_t _threshold;

public:
  nn_patch_model () { }

  virtual ~nn_patch_model() { }

public:
  void set_model(const model_t& model) {
    _model = model;
  }

  const model_t& model() const {
    return _model;
  }

  model_t& model() {
    return _model;
  }

  void set_patch_size(const dim_t& patch_size) {
    _patch_size = patch_size;
  }

  const dim_t& patch_size() const {
    return _patch_size;
  }

  dim_t& patch_size() {
    return _patch_size;
  }

  void set_threshold(const value_t& theshold) {
    _threshold = theshold;
  }

  value_t threshold() const {
    return _threshold;
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_NN_PATCH_MODEL_HPP_ */
