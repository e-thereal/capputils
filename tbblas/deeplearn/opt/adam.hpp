/*
 * adam.hpp
 *
 *  Created on: Apr 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_ADAM_HPP_
#define TBBLAS_DEEPLEARN_OPT_ADAM_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/math.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class adam {

  typedef T value_t;
  typedef std::vector<value_t> v_value_t;

  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const adam* _parent;
  value_t _alpha, _beta1, _beta2, _epsilon;
  v_vector_t _delta, _first_moment, _second_moment;
  v_value_t _current_iteration;

public:
  // Used during the construction of the network
  adam() : _parent(0), _alpha(0.0002), _beta1(0.1), _beta2(0.001), _epsilon(1e-8) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  adam(const adam* parent) : _parent(parent), _alpha(0.0002), _beta1(0.1), _beta2(0.001), _epsilon(1e-8) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size() || index >= _first_moment.size() || index >= _second_moment.size() || index >= _current_iteration.size()) {
      _delta.resize(index + 1);
      _first_moment.resize(index + 1);
      _second_moment.resize(index + 1);
      _current_iteration.resize(index + 1);
    }

    if (!_delta[index] || !_first_moment[index] || !_second_moment[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _first_moment[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _second_moment[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _current_iteration[index] = 0;
    }

    ++_current_iteration[index];

    // Perform adadelta updates
    *_first_moment[index] = get_beta1() * reshape(gradient, _delta[index]->size()) + (1.0 - get_beta1()) * *_first_moment[index];
    *_second_moment[index] = get_beta2() * reshape(gradient, _delta[index]->size()) * reshape(gradient, _delta[index]->size()) + (1.0 - get_beta2()) * *_second_moment[index];

    *_delta[index] = get_alpha() * *_first_moment[index] / (value_t(1) - ::pow(value_t(1) - get_beta1(), _current_iteration[index])) /
        (sqrt(*_second_moment[index] / (value_t(1) - ::pow(value_t(1) - get_beta2(), _current_iteration[index]))) + get_epsilon());
  }

  vector_t& delta(int index) {
    // return delta for the current index
    if (index >= _delta.size())
      throw std::runtime_error("Requested delta has not yet been calculated.");

    return *_delta[index];
  }

  void set_alpha(value_t alpha) {
    _alpha = alpha;
  }

  value_t alpha() const {
    return _alpha;
  }

  value_t get_alpha() const {
    if (_parent)
      return _parent->alpha();
    else
      return _alpha;
  }

  void set_beta1(value_t beta) {
    _beta1 = beta;
  }

  value_t beta1() const {
    return _beta1;
  }

  value_t get_beta1() const {
    if (_parent)
      return _parent->beta1();
    else
      return _beta1;
  }

  void set_beta2(value_t beta) {
    _beta2 = beta;
  }

  value_t beta2() const {
    return _beta2;
  }

  value_t get_beta2() const {
    if (_parent)
      return _parent->beta2();
    else
      return _beta2;
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
struct is_trainer<adam<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_ADADELTA_HPP_ */
