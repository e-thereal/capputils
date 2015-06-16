/*
 * adagrad.hpp
 *
 *  Created on: May 7, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_ADAGRAD_HPP_
#define TBBLAS_DEEPLEARN_OPT_ADAGRAD_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/math.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class adagrad {

  typedef T value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const adagrad* _parent;
  value_t _learning_rate, _epsilon;
  v_vector_t _delta, _cache;

public:
  // Used during the construction of the network
  adagrad() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  adagrad(const adagrad* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size() || index >= _cache.size()) {
      _delta.resize(index + 1);
      _cache.resize(index + 1);
    }

    if (!_delta[index] || !_cache[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _cache[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
    }

    // Perform adadelta updates
    *_cache[index] = *_cache[index] + reshape(gradient, _delta[index]->size()) * reshape(gradient, _delta[index]->size());
    *_delta[index] = -get_learning_rate() * reshape(gradient, _delta[index]->size()) / sqrt(*_cache[index] + get_epsilon());
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
struct is_trainer<adagrad<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_ADAGRAD_HPP_ */
