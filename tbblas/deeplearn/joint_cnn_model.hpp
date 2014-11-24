/*
 * joint_cnn_model.hpp
 *
 *  Created on: Nov 23, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_JOINT_CNN_MODEL_HPP_
#define TBBLAS_DEEPLEARN_JOINT_CNN_MODEL_HPP_

#include <tbblas/deeplearn/cnn_layer_model.hpp>
#include <tbblas/deeplearn/nn_layer_model.hpp>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
class joint_cnn_model {
public:
  typedef T value_t;
  static const unsigned dimCount = dims;
  typedef typename tbblas::tensor<value_t, dimCount>::dim_t dim_t;

  typedef cnn_layer_model<T, dims> cnn_layer_t;
  typedef nn_layer_model<T> nn_layer_t;

  typedef std::vector<boost::shared_ptr<cnn_layer_t> > v_cnn_layer_t;
  typedef std::vector<boost::shared_ptr<nn_layer_t> > v_nn_layer_t;

protected:
  v_cnn_layer_t _left_cnn_layers, _right_cnn_layers;
  v_nn_layer_t _left_nn_layers, _right_nn_layers, _joint_nn_layers;

public:
  joint_cnn_model() { }

  joint_cnn_model(const joint_cnn_model<T, dims>& model) {
    set_left_cnn_layers(model.left_cnn_layers());
    set_right_cnn_layers(model.right_cnn_layers());
    set_left_nn_layers(model.left_nn_layers());
    set_right_nn_layers(model.right_nn_layers());
    set_joint_nn_layers(model.joint_nn_layers());
  }

  virtual ~joint_cnn_model() { }

public:

  /* Convolutional layers */

  template<class U>
  void set_left_cnn_layers(const std::vector<boost::shared_ptr<cnn_layer_model<U, dims> > >& layers) {
    _left_cnn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _left_cnn_layers[i] = boost::make_shared<cnn_layer_t>(*layers[i]);
  }

  v_cnn_layer_t& left_cnn_layers() {
    return _left_cnn_layers;
  }

  const v_cnn_layer_t& left_cnn_layers() const {
    return _left_cnn_layers;
  }

  template<class U>
  void append_left_cnn_layer(const cnn_layer_model<U, dims>& layer) {
    _left_cnn_layers.push_back(boost::make_shared<cnn_layer_t>(layer));
  }

  template<class U>
  void set_right_cnn_layers(const std::vector<boost::shared_ptr<cnn_layer_model<U, dims> > >& layers) {
    _right_cnn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _right_cnn_layers[i] = boost::make_shared<cnn_layer_t>(*layers[i]);
  }

  v_cnn_layer_t& right_cnn_layers() {
    return _right_cnn_layers;
  }

  const v_cnn_layer_t& right_cnn_layers() const {
    return _right_cnn_layers;
  }

  template<class U>
  void append_right_cnn_layer(const cnn_layer_model<U, dims>& layer) {
    _right_cnn_layers.push_back(boost::make_shared<cnn_layer_t>(layer));
  }

  /* Dense layers */

  template<class U>
  void set_left_nn_layers(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _left_nn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _left_nn_layers[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& left_nn_layers() {
    return _left_nn_layers;
  }

  const v_nn_layer_t& left_nn_layers() const {
    return _left_nn_layers;
  }

  template<class U>
  void append_left_nn_layer(const nn_layer_model<U>& layer) {
    _left_nn_layers.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  template<class U>
  void set_right_nn_layers(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _right_nn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _right_nn_layers[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& right_nn_layers() {
    return _right_nn_layers;
  }

  const v_nn_layer_t& right_nn_layers() const {
    return _right_nn_layers;
  }

  template<class U>
  void append_right_nn_layer(const nn_layer_model<U>& layer) {
    _right_nn_layers.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  template<class U>
  void set_joint_nn_layers(const std::vector<boost::shared_ptr<nn_layer_model<U> > >& layers) {
    _joint_nn_layers.resize(layers.size());
    for (size_t i = 0; i < layers.size(); ++i)
      _joint_nn_layers[i] = boost::make_shared<nn_layer_t>(*layers[i]);
  }

  v_nn_layer_t& joint_nn_layers() {
    return _joint_nn_layers;
  }

  const v_nn_layer_t& joint_nn_layers() const {
    return _joint_nn_layers;
  }

  template<class U>
  void append_joint_nn_layer(const nn_layer_model<U>& layer) {
    _joint_nn_layers.push_back(boost::make_shared<nn_layer_t>(layer));
  }

  /* Model information */

  dim_t left_input_size() const {
    assert(_left_cnn_layers.size());
    return _left_cnn_layers[0]->input_size();
  }

  size_t left_visibles_count() const {
    assert(_left_cnn_layers.size());
    return _left_cnn_layers[0]->visibles_count();
  }

  dim_t right_input_size() const {
    assert(_right_cnn_layers.size());
    return _right_cnn_layers[0]->input_size();
  }

  size_t right_visibles_count() const {
    assert(_right_cnn_layers.size());
    return _right_cnn_layers[0]->visibles_count();
  }

  size_t hiddens_count() const {
    assert(_joint_nn_layers.size());
    return _joint_nn_layers[_joint_nn_layers.size() - 1]->hiddens_count();
  }
};

}

}

#endif /* TBBLAS_DEEPLEARN_JOINT_CNN_MODEL_HPP_ */
