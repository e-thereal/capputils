/*
 * hessian_free2.hpp
 *
 *  Created on: Jun 3, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_HESSIAN_FREE2_HPP_
#define TBBLAS_DEEPLEARN_OPT_HESSIAN_FREE2_HPP_

#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/random.hpp>
#include <fstream>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class Network>
class hessian_free2
{
  typedef typename Network::value_t value_t;
  typedef tbblas::tensor<value_t, 1, true> vector_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  typedef Network nn_t;

protected:
  nn_t& _nn;
  value_t _lambda, _weightcost, _zeta;
  vector_t gv, _delta, Jv;
  value_t error, oldError;

  // temporary variables used by pcg
  vector_t r, y, p, x, Ap;
  value_t val;

  // Temporary variables used by update
  vector_t b, b2, parameters, precon;
  int _iteration_count;

public:
  hessian_free2(nn_t& nn) : _nn(nn), _lambda(45), _weightcost(0.0002), _zeta(0.9), error(0), oldError(0), _iteration_count(10) {
    _delta = zeros<value_t>(_nn._model.parameter_count());
    _nn.set_is_encoder(false);
  }

  void set_lambda(value_t lambda) {
    _lambda = lambda;
  }

  value_t lambda() const {
    return _lambda;
  }

  void set_weightcost(value_t cost) {
    _weightcost = cost;
  }

  value_t weightcost() const {
    return _weightcost;
  }

  void set_iteration_count(int maxiters) {
    _iteration_count = maxiters;
  }

  int iteration_count() const {
    return _iteration_count;
  }

  void set_zeta(value_t zeta) {
    _zeta = zeta;
  }

  value_t zeta() const {
    return _zeta;
  }

//  vector_t& computeGV(v_host_tensor_t& samples, v_host_tensor_t& labels, vector_t& v, bool damping = true) {
//
//    // Do the computation
//    _encoder.reset_gradient();
//    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
//      sample = *samples[iSample];
//      label = *labels[iSample];
//      _encoder.inputs() = sample;
//      _encoder.update_gv(v, label);
//
//      gv = _encoder.gradient();
//    }
//
//    // TODO: incorporate weight mask
//    gv = _encoder.gradient() + _weightcost * v;
//    if (damping)
//      gv += _lambda * v;
//
//    return gv;
//  }

  // Preconditioned conjugate gradients
//  vector_t& pcg(v_host_tensor_t& samples, v_host_tensor_t& labels, vector_t& b, vector_t& x0, vector_t& Mdiag) {
//    value_t pAp, alpha, beta;
//
//    x = x0;
//    r = computeGV(samples, labels, x) - b;
//    y = r / Mdiag;
//    p = -y;
//
//    val = 0.5 * dot(-b + r, x);
//
//    for (size_t i = 0; i < _iteration_count; ++i) {
//      tbblas_print(val);
//
//      Ap = computeGV(samples, labels, p);
//      pAp = dot(p, Ap);
//
//      if (pAp <= 0) {
//        std::cout << "Negative curvature! " << pAp << std::endl;
//        break;
//      }
//
//      alpha = dot(r, y) / pAp;
//      beta = value_t(1) / dot(r, y);
//
//      x = x + alpha * p;
//      r = r + alpha * Ap;
//      y = r / Mdiag;
//
//      beta *= dot(r, y);
//      p = -y + beta * p;
//
//      val = 0.5 * dot(-b + r, x);
//    }
//
//    return x;
//  }

  void check_gradient(matrix_t& samples, matrix_t& labels, value_t epsilon) {
    value_t dloss = 0;

    parameters = _nn.parameters();
    _nn.visibles() = samples;

    tbblas_print(_nn.loss(labels));

    // Estimate the gradient using finite differences

    vector_t dparam, grad(parameters.size());
    tbblas_print(epsilon);
    for (int iParam = 0; iParam < parameters.count(); ++iParam) {

      // Forward loss
      dparam = parameters;
      dparam[seq(iParam)] = dparam[seq(iParam)] + epsilon;
      _nn.set_parameters(dparam);
      dloss = _nn.loss(labels);

      // Backward loss
      dparam = parameters;
      dparam[seq(iParam)] = dparam[seq(iParam)] - epsilon;
      _nn.set_parameters(dparam);
      dloss -= _nn.loss(labels);
      grad[seq(iParam)] = dloss / (2.0 * epsilon);
    }

    // Calculate gradient
    _nn.set_parameters(parameters);
    _nn.reset_gradient();
    _nn.update_gradient(labels);
    b = _nn.gradient(0);

    tbblas_print(cor(grad, b));
    tbblas_print(::sqrt(dot(grad - b, grad - b) / parameters.count()));

    std::ofstream file("nn_grad.csv");
    file << "Pos,FD,Grad\n";
    for (int i = 0; i < grad.count(); ++i)
      file << i << "," << grad[seq(i)] << "," << b[seq(i)] << "\n";
    file.close();
  }

  // Check the Gv product using Gv = J'(H_L(Jv))
  void check_Gv(matrix_t& samples, matrix_t& labels, value_t epsilon, int count, bool randomVector) {
    vector_t dparam;

    vector_t v, Jv, HJv, JHJv, e, Je;
    parameters = _nn.parameters();

    tbblas_print(dot(samples, samples));

    random_tensor2<value_t, 1, true, normal<value_t> > randn(parameters.size());

    // Check for all basis vectors
    for (int iTrial = 0; iTrial < count && iTrial < parameters.count(); ++iTrial) {
      tbblas_print(iTrial);

      if (randomVector) {
        v = randn();
      } else {
        v = zeros<value_t>(parameters.count());
        v[seq(iTrial)] = 1;
      }

      Je = Jv = zeros<value_t>(_nn._model.hiddens_count());

      _nn.visibles() = samples;

      // Jv = (f(x, theta + e*v) - f(x, theta - e*v)) / 2e;

      // Forward prediction
      dparam = parameters + epsilon * v;
      _nn.set_parameters(dparam);
      _nn.infer_hiddens();
      Jv = reshape(_nn.hiddens(), Jv.size());
      tbblas_print(dot(Jv, Jv));

      // Backward prediction
      dparam = parameters - epsilon * v;
      _nn.set_parameters(dparam);
      _nn.infer_hiddens();
      Jv -= reshape(_nn.hiddens(), Jv.size());
      Jv = Jv / (value_t)(2.0 * epsilon);
      tbblas_print(dot(Jv, Jv));

      // For SSD, H_L = I -> Jv = HJv
      HJv = Jv;

      // J'w = [(Je_1)'w (Je_2)'w ... (Je_n)'w]
      JHJv = zeros<value_t>(parameters.count());
      for (int i = 0; i < JHJv.count(); ++i) {
        e = zeros<value_t>(JHJv.count());
        e[seq(i)] = 1;

        // Forward prediction
        dparam = parameters + epsilon * e;
        _nn.set_parameters(dparam);
        _nn.infer_hiddens();
        Je = reshape(_nn.hiddens(), Je.size());

        // Backward prediction
        dparam = parameters - epsilon * e;
        _nn.set_parameters(dparam);
        _nn.infer_hiddens();
        Je -= reshape(_nn.hiddens(), Je.size());
        Je = Je / (value_t)(2.0 * epsilon);

        JHJv[seq(i)] = dot(Je, HJv);
  //      tbblas_print((float)i / (float)JHJv.count());
      }

      // Calculate Gv
      _nn.set_parameters(parameters);
      _nn.reset_Rgradient();
      _nn.update_Gv(v, labels);
      gv = _nn.Rgradient(0);

      assert(gv.size() == JHJv.size());

      tbblas_print(::sqrt(dot(JHJv - gv, JHJv - gv) / parameters.count()));
      tbblas_print(cor(JHJv, gv));
    }

    std::ofstream file("Gv.csv");
    file << "Pos,JHJv,Gv\n";
    for (int i = 0; i < gv.count(); ++i)
      file << i << "," << JHJv[seq(i)] << "," << gv[seq(i)] << "\n";
    file.close();
  }

//  void update(v_host_tensor_t& samples, v_host_tensor_t& labels) {
//    parameters = _encoder.parameters();
//    b = zeros<value_t>(_encoder._model.parameter_count());
//    b2 = zeros<value_t>(_encoder._model.parameter_count());
//
//    error = 0;
//
//    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
//      sample = *samples[iSample];
//      label = *labels[iSample];
//      _encoder.inputs() = sample;
//
//      _encoder.reset_gradient();
//      error += _encoder.update_gradient(label);
//      b += _encoder.gradient(_weightcost);
//      b2 += b * b;
//    }
//    b = -b / samples.size();
//    error = error / samples.size() + 0.5 * weightcost() * dot(parameters, parameters);
//
//    tbblas_print(error);
//
//    // Update lambda
//    if (oldError > 0) {
//      value_t rho = (error - oldError) / val;
//      tbblas_print(error - oldError);
//      tbblas_print(val);
//      tbblas_print(rho);
//      if (rho > 0.75) {
//        _lambda *= 2./3.;
//      }
//      if (rho < 0.25) {
//        _lambda *= 3./2.;
//      }
//    }
//
//    precon = pow(b2 + _lambda, 0.75);
//    _delta = _zeta * _delta;
//    _delta = pcg(samples, labels, b, _delta, precon);
//    _encoder.update_model(_delta);
//
//    // TODO: Choose a step size
//
//    oldError = error;
//    val = 0.5 * dot(_delta, computeGV(samples, labels, _delta, true)) - dot(b, _delta);
//  }
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_HESSIAN_FREE_HPP_ */
