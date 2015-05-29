/*
 * vsgd_fd_v2.hpp
 *
 *  Created on: Apr 16, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_VSGD_FD_V2_HPP_
#define TBBLAS_DEEPLEARN_OPT_VSGD_FD_V2_HPP_

/*
 * This optimizer implements the "Adaptive learning rates and parallelization for
 * stochastic, sparse, non-smooth gradients" update. But instead of using finite differences along the last gradient to estimate
 * the diagonal Hessian, a finite difference approximation is done along the last update, similar to ADASECANT.
 */

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/math.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class vsgd_fd_v2 {

  typedef T value_t;
  typedef std::vector<value_t> v_value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const vsgd_fd_v2* _parent;
  value_t _epsilon, _c;
  v_vector_t _delta, _old_grad, _g, _g2, _h, _h2, _tau;
  v_value_t _current_iteration;

public:
  // Used during the construction of the network
  vsgd_fd_v2() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  vsgd_fd_v2(const vsgd_fd_v2* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _delta.size() || index >= _old_grad.size() || index >= _g.size() || index >= _g2.size() ||
        index >= _h.size() || index >= _h2.size() || index >= _tau.size() || index >= _current_iteration.size())
    {
      _delta.resize(index + 1);
      _old_grad.resize(index + 1);
      _g.resize(index + 1);
      _g2.resize(index + 1);
      _h.resize(index + 1);
      _h2.resize(index + 1);
      _tau.resize(index + 1);
      _current_iteration.resize(index + 1);
    }

    if (!_delta[index] || !_old_grad[index] || !_g[index] || !_g2[index] || !_h[index] || !_h2[index] || !_tau[index]) {
      _delta[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _old_grad[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _g[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _g2[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _h[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _h2[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
      _tau[index] = boost::make_shared<vector_t>(ones<value_t>(gradient.count()));
      _current_iteration[index] = 0;
    }

    ++_current_iteration[index];

    // Detect outlier
//    if (abs(reshape(gradient, _delta[index]->size()) - *_g[index]) > 2 * sqrt(*_g2[index] - *_g[index] * *_g[index]) ||
//        abs(abs((reshape(gradient, _delta[index]->size()) - *_old_grad[index]) / (*_delta[index] + get_epsilon())) - *_h[index]) > 2 * sqrt(*_h2[index] - *_h[index] * *_h[index]))
//    {
//      *_tau[index] = *_tau[index] + 1.0;
//    }

    *_tau[index] = *_tau[index] + abs(reshape(gradient, _delta[index]->size()) - *_g[index]) > 2 * sqrt(*_g2[index] - *_g[index] * *_g[index]);

    // Update moving averages
    *_g[index]  = (1.0 - 1.0 / *_tau[index]) * *_g[index]  + 1.0 / *_tau[index] * reshape(gradient, _delta[index]->size());
    *_g2[index] = (1.0 - 1.0 / *_tau[index]) * *_g2[index] + 1.0 / *_tau[index] * reshape(gradient, _delta[index]->size()) * reshape(gradient, _delta[index]->size());

    if (_current_iteration[index] > 1) {
      // Calculate h and do h updates
      // diag(H) = abs((reshape(gradient, _delta[index]->size()) - *_old_grad[index]) / (*_delta[index] + get_epsilon()));

      *_h[index]  = (1.0 - 1.0 / *_tau[index]) * *_h[index]  + 1.0 / *_tau[index] * abs((reshape(gradient, _delta[index]->size()) - *_old_grad[index]) / (*_delta[index] + get_epsilon()));
      *_h2[index]  = (1.0 - 1.0 / *_tau[index]) * *_h[index]  + 1.0 / *_tau[index] * ((reshape(gradient, _delta[index]->size()) - *_old_grad[index]) / (*_delta[index] + get_epsilon())) * ((reshape(gradient, _delta[index]->size()) - *_old_grad[index]) / (*_delta[index] + get_epsilon()));

      // Initialization phase -> multiply with C where C = D/10
      if (_current_iteration[index] == 2) {
        *_g2[index] = *_g2[index] * get_c();
        *_h[index] = *_h[index] * get_c();
        *_h2[index] = *_h2[index] * get_c();
      }

      *_delta[index] = *_h[index] * *_g[index] * *_g[index] / (*_h2[index] * *_g2[index] + get_epsilon()) * reshape(gradient, _delta[index]->size());
    } else {
      *_delta[index] = get_epsilon() * *_g[index];
    }

    *_tau[index] = (1.0 - *_g[index] * *_g[index] / (*_g2[index] + get_epsilon())) * *_tau[index] + 1;

    *_old_grad[index] = reshape(gradient, _delta[index]->size());
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

  void set_c(value_t c) {
    _c = c;
  }

  value_t c() const {
    return _c;
  }

  value_t get_c() const {
    if (_parent)
      return _parent->c();
    else
      return _c;
  }
};

template<class T>
struct is_trainer<vsgd_fd_v2<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_VSGD_FD_V2_HPP_ */
