/*
 * encodertest.cu
 *
 *  Created on: Apr 15, 2015
 *      Author: tombr
 */

#include "tests.h"

#include <tbblas/io.hpp>
#include <tbblas/linalg.hpp>
#include <tbblas/random.hpp>

//#include <tbblas/deeplearn/encoder_base.hpp>
#include <tbblas/deeplearn/encoder.hpp>
//#include <tbblas/deeplearn/opt/classic_momentum.hpp>
//#include <tbblas/deeplearn/opt/nesterov_momentum.hpp>
//#include <tbblas/deeplearn/opt/adadelta.hpp>
//#include <tbblas/deeplearn/opt/adam.hpp>
//#include <tbblas/deeplearn/opt/adam2.hpp>
//#include <tbblas/deeplearn/opt/rms_prop.hpp>
//#include <tbblas/deeplearn/opt/vsgd_fd.hpp>
//#include <tbblas/deeplearn/opt/vsgd_fd_v2.hpp>
//#include <tbblas/deeplearn/opt/adagrad.hpp>

#include <tbblas/deeplearn/opt/hessian_free.hpp>
//#include <tbblas/deeplearn/opt/hessian_free2.hpp>

#include <tbblas/deeplearn/nn.hpp>

//typedef tbblas::deeplearn::cnn_layer<float, 4> layer_t;

//void print_flags(int flags) {
//  std::cout << "Flag " << flags << ":";
//
//  if (flags & layer_t::APPLY_BIAS)
//      std::cout << " apply bias";
//
//  if (flags & layer_t::APPLY_NONLINEARITY)
//      std::cout << " apply non-linearity";
//
//  if (flags & layer_t::DROPOUT)
//      std::cout << " dropout";
//
//  if (flags & layer_t::ACCUMULATE)
//      std::cout << " accumulate";
//
//  std::cout << std::endl;
//}

void encodertest() {
  using namespace tbblas;
  using namespace tbblas::deeplearn;
  using namespace tbblas::deeplearn::opt;

  std::cout << "Encoder test" << std::endl;

//  print_flags(layer_t::DEFAULT);
//  print_flags(layer_t::DEFAULT | layer_t::DROPOUT);
//  print_flags(layer_t::APPLY_BIAS);
//  print_flags(layer_t::ACCUMULATE);

  encoder_model<float, 4> model;

//  encoder<float, 4, nesterov_momentum<float> > trainer(model);
//  trainer.set_learning_rate(0.0001);
//  trainer.set_momentum(0.9);
//  trainer.update_model(0.0002);
//
//  encoder<float, 4, vsgd_fd_v2<float> > trainer2(model);
//
//
//  nn_base<float>& nn_base = nn;

  std::vector<boost::shared_ptr<tensor<float, 4> > > batch, labels;

  encoder<float, 4> enn(model);
  opt::hessian_free<float, 4> train(enn);

//  train.set_learning_rate(0.001);
//  train.update(batch, labels);
//  train.check_loss(*batch[0], *labels[0]);
  train.check_gradient(batch, labels);
  train.check_Gv(batch, labels, false);

//  tensor<float, 1, true> v;
//  tensor<float, 4, true> target;
//  enn.update_gv(v, target);

//  nn_model<float> nn_model;
//  nn<float> nn(nn_model);
//
//  opt::hessian_free2<tbblas::deeplearn::nn<float> > train(nn);
//  tensor<float, 2, true> samples, labels;
//  train.check_gradient(samples, labels, 0.0001);
//  train.check_Gv(samples, labels, 0.001);
}
