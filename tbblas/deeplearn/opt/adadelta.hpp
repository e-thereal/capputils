/*
 * adadelta.hpp
 *
 *  Created on: Apr 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_ADADELTA_HPP_
#define TBBLAS_DEEPLEARN_OPT_ADADELTA_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/math.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class adadelta {

  typedef T value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const adadelta* _parent;
  value_t _decay_rate, _epsilon;
  v_vector_t _d2, _delta, _delta2;

public:
  // Used during the construction of the network
  adadelta() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  adadelta(const adadelta* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size() || index >= _d2.size() || index >= _delta2.size()) {
      _delta.resize(index + 1);
      _delta2.resize(index + 1);
      _d2.resize(index + 1);
    }

    if (!_delta[index] || !_delta2[index] || !_d2[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _delta2[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _d2[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
    }

    // Perform adadelta updates
    *_d2[index] = get_decay_rate() * *_d2[index] + (1.0 - get_decay_rate()) * reshape(gradient, _d2[index]->size()) * reshape(gradient, _d2[index]->size());
    *_delta[index] = sqrt(*_delta2[index] + get_epsilon()) / sqrt(*_d2[index] + get_epsilon()) * reshape(gradient, _d2[index]->size());
    *_delta2[index] = get_decay_rate() * *_delta2[index] + (1.0 - get_decay_rate()) * *_delta[index] * *_delta[index];
  }

  vector_t& delta(int index) {
    // return delta for the current index
    if (index >= _delta.size())
      throw std::runtime_error("Requested delta has not yet been calculated.");

    return *_delta[index];
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
};

template<class T>
struct is_trainer<adadelta<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_ADADELTA_HPP_ */
