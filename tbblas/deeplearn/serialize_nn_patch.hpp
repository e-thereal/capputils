/*
 * serialize_nn_patch.hpp
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_NN_PATCH_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_NN_PATCH_HPP_

#include <tbblas/deeplearn/nn_patch_model.hpp>
#include <tbblas/deeplearn/serialize_nn.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::nn_patch_model<T, dim>& model, std::ostream& out) {
  serialize(model.model(), out);

  const typename tbblas::deeplearn::nn_patch_model<T, dim>::dim_t& patch_size = model.patch_size();
  out.write((char*)&patch_size, sizeof(patch_size));

  T threshold = model.threshold();
  out.write((char*)&threshold, sizeof(threshold));
}

template<class T, unsigned dim>
void serialize(const tbblas::deeplearn::nn_patch_model<T, dim>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T, unsigned dim>
void deserialize(std::istream& in, tbblas::deeplearn::nn_patch_model<T, dim>& model) {

  nn_model<T> nn_model;
  deserialize(in, nn_model);
  model.set_model(nn_model);

  typename tbblas::deeplearn::nn_patch_model<T, dim>::dim_t patch_size;
  in.read((char*)&patch_size, sizeof(patch_size));
  model.set_patch_size(patch_size);

  T threshold;
  in.read((char*)&threshold, sizeof(threshold));
  model.set_threshold(threshold);
}

template<class T, unsigned dim>
void deserialize(const std::string& filename, tbblas::deeplearn::nn_patch_model<T, dim>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_NN_PATCH_HPP_ */
