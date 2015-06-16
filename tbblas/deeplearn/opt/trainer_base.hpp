/*
 * trainer_base.hpp
 *
 *  Created on: Jun 15, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_TRAINER_BASE_HPP_
#define TBBLAS_DEEPLEARN_OPT_TRAINER_BASE_HPP_

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T, unsigned dims>
class trainer_base {
public:
  typedef T value_t;
  typedef tbblas::tensor<T, 1, true> vector_t;
  typedef tbblas::tensor<T, dims, true> tensor_t;
  typedef tbblas::tensor<T, dims> host_tensor_t;

  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  typedef encoder<T, dims> encoder_t;

protected:
  value_t _weightcost;
  encoder_t& _encoder;

public:
  trainer_base(encoder_t& encoder) : _encoder(encoder), _weightcost(0.0002) { }

public:
  value_t weightcost() const {
    return _weightcost;
  }

  void set_weightcost(value_t weightcost) {
    _weightcost = weightcost;
  }

  virtual void update(v_host_tensor_t& samples, v_host_tensor_t& labels) = 0;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_TRAINER_BASE_HPP_ */
