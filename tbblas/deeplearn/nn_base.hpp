/*
 * nn_base.hpp
 *
 *  Created on: Apr 22, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_NN_BASE_HPP_
#define TBBLAS_DEEPLEARN_NN_BASE_HPP_

#include <tbblas/tensor.hpp>
#include <tbblas/deeplearn/objective_function.hpp>

namespace tbblas {

namespace deeplearn {

template<class T>
class nn_base {
public:
  typedef T value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

public:
  virtual ~nn_base() { }

  virtual void set_objective_function(const tbblas::deeplearn::objective_function& objective) = 0;
  virtual tbblas::deeplearn::objective_function objective_function() const = 0;
  virtual void set_sensitivity_ratio(const value_t& ratio) = 0;
  virtual value_t sensitivity_ratio() const = 0;
  virtual void set_dropout_rate(int iLayer, const value_t& rate) = 0;
  virtual void normalize_visibles() = 0;
  virtual void infer_hiddens(bool dropout = false) = 0;
  virtual void update_gradient(matrix_t& target) = 0;
  virtual void update_model(value_t weightcost) = 0;
  virtual void update(matrix_t& target, value_t weightcost = 0) = 0;
  virtual matrix_t& visibles() = 0;
  virtual matrix_t& hiddens() = 0;
};

}

}

#endif /* TBBLAS_DEEPLEARN_NN_BASE_HPP_ */
