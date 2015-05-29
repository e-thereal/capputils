/*
 * cnn_base.hpp
 *
 *  Created on: Apr 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_CNN_BASE_HPP_
#define TBBLAS_DEEPLEARN_CNN_BASE_HPP_

#include <tbblas/deeplearn/objective_function.hpp>
#include <tbblas/tensor.hpp>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class cnn_base {

  typedef T value_t;
  typedef tbblas::tensor<value_t, dims, true> tensor_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

public:
  virtual ~cnn_base() { }

  virtual void normalize_visibles() = 0;
  virtual void infer_hiddens() = 0;
  virtual void update_gradient(matrix_t& target) = 0;
  virtual void update_model(value_t weightcost) = 0;
  virtual void set_batch_length(int layer, int length) = 0;
  virtual const proxy<tensor_t> visibles() = 0;
  virtual matrix_t& hiddens() = 0;
};

}

}


#endif /* TBBLAS_DEEPLEARN_CNN_BASE_HPP_ */
