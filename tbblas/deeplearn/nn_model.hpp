/*
 * nn_model.hpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_NN_MODEL_HPP_
#define TBBLAS_DEEPLEARN_NN_MODEL_HPP_

#include <tbblas/deeplearn/nn_layer_model.hpp>

namespace tbblas {

namespace deeplearn {

template<class T>
class nn_model {
public:
  typedef T value_t;

  typedef nn_layer_model<T> nn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

protected:
  v_nn_layer_t _layers;

public:
  nn_model() { }

  nn_model(const nn_model<T>& model) {
    set_layers(model.layers());
  }

  virtual ~nn_model() { }

public:
  template<class U>
  void set_layers(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _layers[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& layers() {
    return _layers;
  }

  const v_nn_layer_t& layers() const {
    return _layers;
  }

  template<class U>
  void append_layer(const nn_layer_model<U>& layer) {
    _layers.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  size_t visibles_count() const {
    assert(_layers.size());
    return _layers[0]->visibles_count();
  }

  size_t hiddens_count() const {
    assert(_layers.size());
    return _layers[_layers.size() - 1]->hiddens_count();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_NN_MODEL_HPP_ */
