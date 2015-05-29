/*
 * classic_momentum.hpp
 *
 *  Created on: Apr 15, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_CLASSIC_MOMENTUM_HPP_
#define TBBLAS_DEEPLEARN_OPT_CLASSIC_MOMENTUM_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class classic_momentum {

  typedef T value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const classic_momentum* _parent;
  value_t _learning_rate, _momentum;
  v_vector_t _delta;

public:
  // Used during the construction of the network
  classic_momentum() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  classic_momentum(const classic_momentum* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size()) {
      _delta.resize(index + 1);
    }

    if (!_delta[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
    }

    // Update delta with momentum and epsilon parameter
    *_delta[index] =  get_momentum() * *_delta[index] + get_learning_rate() * reshape(gradient, _delta[index]->size());
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

  void set_momentum(value_t momentum) {
    _momentum = momentum;
  }

  value_t momentum() const {
    return _momentum;
  }

  value_t get_momentum() const {
    if (_parent)
      return _parent->momentum();
    else
      return _momentum;
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
