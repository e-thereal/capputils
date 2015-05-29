/*
 * encoder_base.hpp
 *
 *  Created on: Apr 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_ENCODER_BASE_HPP_
#define TBBLAS_DEEPLEARN_ENCODER_BASE_HPP_

#include <tbblas/deeplearn/objective_function.hpp>
#include <tbblas/tensor.hpp>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class encoder_base {

  typedef T value_t;
  typedef tbblas::tensor<value_t, dims, true> tensor_t;

public:
  virtual ~encoder_base() { }

  virtual void set_objective_function(const tbblas::deeplearn::objective_function& objective) = 0;
  virtual tbblas::deeplearn::objective_function objective_function() const = 0;
  virtual void set_sensitivity_ratio(const value_t& ratio) = 0;
  virtual value_t sensitivity_ratio() const = 0;
  virtual void write_model_to_host() = 0;
  virtual void infer_outputs() = 0;
  virtual void infer_layer(const size_t maxLayer) = 0;
  virtual void update_gradient(tensor_t& target) = 0;
  virtual void update_model(value_t weightcost) = 0;
  virtual void set_batch_length(int layer, int length) = 0;
  virtual tensor_t& inputs() = 0;
  virtual tensor_t& outputs() = 0;
};

}

}

#endif /* TBBLAS_DEEPLEARN_ENCODER_BASE_HPP_ */
