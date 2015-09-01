/*
 * first_order.hpp
 *
 *  Created on: May 29, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_FIRST_ORDER_HPP_
#define TBBLAS_DEEPLEARN_OPT_FIRST_ORDER_HPP_

#include <tbblas/deeplearn/encoder.hpp>
#include <tbblas/deeplearn/opt/type_traits.hpp>
#include <tbblas/deeplearn/opt/trainer_base.hpp>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T, unsigned dims, class Trainer, class Enable = typename boost::enable_if<opt::is_trainer<Trainer> >::type>
class first_order : public trainer_base<T, dims>, public Trainer
{
  typedef trainer_base<T, dims> base_t;

  typedef typename base_t::value_t value_t;
  typedef typename base_t::vector_t vector_t;
  typedef typename base_t::tensor_t tensor_t;
  typedef typename base_t::host_tensor_t host_tensor_t;

  typedef typename base_t::v_host_tensor_t v_host_tensor_t;

  typedef typename base_t::encoder_t encoder_t;

protected:
  using base_t::_encoder;
  using base_t::_weightcost;

public:
  first_order(encoder_t& encoder) : base_t(encoder) { }

  virtual void update(v_host_tensor_t& samples, v_host_tensor_t& labels) {
    tensor_t v, target;
//    vector_t parameters;

    // Calculate gradient
    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      v = *samples[iSample];
      target = *labels[iSample];
      _encoder.inputs() = v;
      _encoder.update_gradient(target);
    }

    // Update model
    this->update_delta(_encoder.gradient(_weightcost), 0);

    _encoder.update_model(this->delta(0));
//    parameters = _encoder.parameters();
//    parameters += this->delta(0);
//    _encoder.set_parameters(parameters);
//    _encoder.reset_gradient();
  }
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_FIRST_ORDER_HPP_ */
