/*
 * adasecant.hpp
 *
 *  Created on: Apr 15, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_ADASECANT_HPP_
#define TBBLAS_DEEPLEARN_OPT_ADASECANT_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class adasecant {

  typedef T value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const adasecant* _parent;
  value_t _epsilon, _decay_rate;
  v_vector_t _delta;

public:
  // Used during the construction of the network
  adasecant() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  adasecant(const adasecant* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size()) {
      _delta.resize(index + 1);
    }

    if (!_delta[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
    }

    // Block-wise normalize the gradient
    vector_t norm_grad = reshape(gradient, _delta[index]->size()) / (sqrt(dot(reshape(gradient, _delta[index]->size()), reshape(gradient, _delta[index]->size()))) + get_epsilon());

    // Calculate running averages
    mg = get_decay_rate() * mg + (1.0 - get_decay_rate()) * norm_grad;

    gamma_nume_sqr = gamma_nume_sqr * (1 - 1 / taus_x_t) + (norm_grad - old_grad) * (old_grad - mg) * (norm_grad - old_grad) * (old_grad - mg) / taus_x_t;
    gamma_deno_sqr = gamma_dneo_sqr * (1 - 1 / taus_x_t) + (mg - norm_grad) * (old_grad - mg) * (mg - norm_grad) * (old_grad - mg) / taus_x_t;
    _gamma = sqrt(_gamma_nume) / (sqrt(_gamma_deno_sqr) + get_epsilon());

    // Update delta with momentum and epsilon parameter
    *_delta[index] =  _gamma * mg;
  }

  vector_t& delta(int index) {
    // return delta for the current index
    if (index >= _delta.size())
      throw std::runtime_error("Requested delta has not yet been calculated.");

    return *_delta[index];
  }

  void set_epsilon(value_t epsilon) {
    _epsilon = epsilon;
  }

  value_t epsilon() const {
    return _epsilon;
  }

  value_t get_epsilon() const {
    if (_parent)
      return _parent->epsilon();
    else
      return _epsilon;
  }

  void set_decay_rate(value_t rate) {
    _decay_rate = rate;
  }

  value_t decay_rate() const {
    return _decay_rate;
  }

  value_t get_decay_rate() const {
    if (_parent)
      return _parent->decay_rate();
    else
      return _decay_rate;
  }
};

template<class T>
struct is_trainer<classic_momentum<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_CLASSIC_MOMENTUM_HPP_ */
