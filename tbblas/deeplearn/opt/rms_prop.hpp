/*
 * rms_prop.hpp
 *
 *  Created on: Apr 15, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_RMSPROP_HPP_
#define TBBLAS_DEEPLEARN_OPT_RMSPROP_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class rms_prop {

  typedef T value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const rms_prop* _parent;
  value_t _learning_rate, _decay_rate;
  v_vector_t _delta, _cache;

public:
  // Used during the construction of the network
  rms_prop() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  rms_prop(const rms_prop* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size() || index > _cache.size()) {
      _delta.resize(index + 1);
      _cache.resize(index + 1);
    }

    if (!_delta[index] || !_cache[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _cache[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
    }

    // Update delta with momentum and epsilon parameter
    *_cache[index] = get_decay_rate() * *_cache[index] + (1.0 - get_decay_rate()) * reshape(gradient, _delta[index]->size()) * reshape(gradient, _delta[index]->size());
    *_delta[index] = -get_learning_rate() * reshape(gradient, _delta[index]->size()) / sqrt(*_cache[index] + 1e-8);
  }

  vector_t& delta(int index) {
    // return delta for the current index
    if (index >= _delta.size())
      throw std::runtime_error("Requested delta has not yet been calculated.");

    return *_delta[index];
  }

  void set_learning_rate(value_t rate) {
    _learning_rate = rate;
  }

  value_t learning_rate() const {
    return _learning_rate;
  }

  value_t get_learning_rate() const {
    if (_parent)
      return _parent->learning_rate();
    else
      return _learning_rate;
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
struct is_trainer<rms_prop<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_CLASSIC_MOMENTUM_HPP_ */
