/*
 * serialize_dbn.hpp
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_
#define TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_

#include <tbblas/deeplearn/dbn_model.hpp>
#include <tbblas/deeplearn/serialize_conv_rbm.hpp>
#include <tbblas/deeplearn/serialize_rbm.hpp>
#include <iostream>
#include <fstream>

namespace tbblas {

namespace deeplearn {

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::dbn_model<T, dims>& model, std::ostream& out) {
  unsigned count = 0;

  count = model.crbms().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.crbms()[i], out);

  count = model.rbms().size();
  out.write((char*)&count, sizeof(count));
  for (size_t i = 0; i < count; ++i)
    serialize(*model.rbms()[i], out);
}

template<class T, unsigned dims>
void serialize(const tbblas::deeplearn::dbn_model<T, dims>& model, const std::string& filename) {
  std::ofstream out(filename.c_str(), std::ios_base::binary);
  serialize(model, out);
}

template<class T, unsigned dims>
void deserialize(std::istream& in, tbblas::deeplearn::dbn_model<T, dims>& model) {

  typedef conv_rbm_model<T, dims> crbm_t;
  typedef std::vector<boost::shared_ptr<crbm_t> > v_crbm_t;

  typedef rbm_model<T> rbm_t;
  typedef std::vector<boost::shared_ptr<rbm_t> > v_rbm_t;

  unsigned count = 0;
  crbm_t crbm;

  in.read((char*)&count, sizeof(count));
  v_crbm_t crbms(count);
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, crbm);
    crbms[i] = boost::make_shared<crbm_t>(crbm);
  }
  model.set_crbms(crbms);

  rbm_t rbm;

  in.read((char*)&count, sizeof(count));
  v_rbm_t rbms(count);
  for (size_t i = 0; i < count; ++i) {
    deserialize(in, rbm);
    rbms[i] = boost::make_shared<rbm_t>(rbm);
  }
  model.set_rbms(rbms);
}

template<class T, unsigned dims>
void deserialize(const std::string& filename, tbblas::deeplearn::dbn_model<T, dims>& model) {
  std::ifstream in(filename.c_str(), std::ios_base::binary);
  deserialize(in, model);
}

}

}

#endif /* TBBLAS_DEEPLEARN_SERIALIZE_DBN_HPP_ */
