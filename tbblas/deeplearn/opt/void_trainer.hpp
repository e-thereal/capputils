/*
 * void_trainer.hpp
 *
 *  Created on: Apr 15, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_VOID_TRAINER_HPP_
#define TBBLAS_DEEPLEARN_OPT_VOID_TRAINER_HPP_

#include <tbblas/deeplearn/opt/type_traits.hpp>

#include <tbblas/tensor.hpp>
#include <tbblas/reshape.hpp>

#include <boost/make_shared.hpp>

#include <stdexcept>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T>
class void_trainer {

  typedef T value_t;
  typedef tensor<value_t, 1, true> vector_t;
  typedef std::vector<boost::shared_ptr<vector_t> > v_vector_t;

private:
  const void_trainer* _parent;
  value_t _learning_rate, _momentum, _weight_cost;
  v_vector_t _deltas;

public:
  // Used during the construction of the network
  void_trainer() : _parent(0) { }

  // Used during the construction of a single layer, with the this pointer of the network passed as a parameter
  void_trainer(const void_trainer* parent) : _parent(parent) { }

  template<class Expression>
  void update_delta(const Expression& gradient, int index) {

    if (index >= _deltas.size()) {
      _deltas.resize(index + 1);
    }

    if (!_deltas[index]) {
      _deltas[index] = boost::make_shared<vector_t>(zeros<value_t>(gradient.count()));
    }
  }

  vector_t& delta(int index) {
    // return delta for the current index
    if (index >= _deltas.size())
      throw std::runtime_error("Requested delta has not yet been calculated.");

    return *_deltas[index];
  }
};

template<class T>
struct is_trainer<void_trainer<T> > {
  static const bool value = true;
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_VOID_TRAINER_HPP_ */
