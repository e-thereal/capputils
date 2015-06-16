/*
 * hessian_free.hpp
 *
 *  Created on: Jun 3, 2015
 *      Author: tombr
 */

#ifndef TBBLAS_DEEPLEARN_OPT_HESSIAN_FREE_HPP_
#define TBBLAS_DEEPLEARN_OPT_HESSIAN_FREE_HPP_

#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/random.hpp>

#include <tbblas/deeplearn/encoder.hpp>
#include <tbblas/deeplearn/opt/trainer_base.hpp>
#ifdef COMPARE_TO_NN
#include <tbblas/deeplearn/nn.hpp>
#endif
#include <fstream>

namespace tbblas {

namespace deeplearn {

namespace opt {

template<class T, unsigned dims>
class hessian_free : public trainer_base<T, dims>
{
  typedef trainer_base<T, dims> base_t;

  typedef typename base_t::value_t value_t;
  typedef typename base_t::vector_t vector_t;
  typedef typename base_t::tensor_t tensor_t;
  typedef typename base_t::host_tensor_t host_tensor_t;

  typedef typename base_t::v_host_tensor_t v_host_tensor_t;

  typedef typename base_t::encoder_t encoder_t;

protected:
  using base_t::_encoder;
  using base_t::_weightcost;
#ifdef COMPARE_TO_NN
  typedef nn<T> nn_t;
  typedef nn_model<T> nn_model_t;
#endif

protected:
#ifdef COMPARE_TO_NN
  boost::shared_ptr<nn_t> _nn;
  nn_model_t _nn_model;
#endif
  value_t _lambda, _zeta;
  tensor_t sample, label;
  vector_t gv, _delta, Jv, nn_gv;
  value_t error, oldError;

  // temporary variables used by pcg
  vector_t r, y, p, x, Ap;
  value_t val;

  // Temporary variables used by update
  vector_t b, b2, parameters, precon, weight_mask;
#ifdef COMPARE_TO_NN
  vector_t nn_parameters, conv_parameters;
#endif
  int _iteration_count;

public:
  hessian_free(encoder_t& encoder) : base_t(encoder), _lambda(45), _zeta(0.9), error(0), oldError(0), _iteration_count(10) {
    parameters = _delta = zeros<value_t>(_encoder._model.parameter_count());

#ifdef COMPARE_TO_NN
    encoder_model<value_t, 4>& model = _encoder._model;

    // Create a dummy model with the right dimensions but with different initial parameters
    for (size_t iLayer = 0; iLayer < model.cnn_encoders().size(); ++iLayer) {
      cnn_layer_model<value_t, 4>& clayer = *model.cnn_encoders()[iLayer];

      const size_t visibleCount = clayer.visibles_count();
      const size_t hiddenCount = clayer.hiddens_count();

      typename nn_model_t::nn_layer_t layer;
      layer.set_activation_function(clayer.activation_function());

      matrix_t W = ones<value_t>(visibleCount, hiddenCount);
      matrix_t b = zeros<value_t>(1, hiddenCount);
      layer.set_weights(W);
      layer.set_bias(b);

      matrix_t means = clayer.mean() * ones<value_t>(1, visibleCount);
      matrix_t stddev = clayer.stddev() * ones<value_t>(1, visibleCount);
      layer.set_mean(means);
      layer.set_stddev(stddev);

      _nn_model.append_layer(layer);
    }

    // Create a dummy model with the right dimensions but with different initial parameters
    for (size_t iLayer = 0; iLayer < model.dnn_decoders().size(); ++iLayer) {
      dnn_layer_model<value_t, 4>& clayer = *model.dnn_decoders()[iLayer];

      const size_t visibleCount = clayer.hiddens_count();
      const size_t hiddenCount = clayer.visibles_count();

      typename nn_model_t::nn_layer_t layer;
      layer.set_activation_function(clayer.activation_function());

      matrix_t W = ones<value_t>(visibleCount, hiddenCount);
      matrix_t b = zeros<value_t>(1, hiddenCount);
      layer.set_weights(W);
      layer.set_bias(b);

      matrix_t means = zeros<value_t>(1, visibleCount);
      matrix_t stddev = ones<value_t>(1, visibleCount);
      layer.set_mean(means);
      layer.set_stddev(stddev);

      _nn_model.append_layer(layer);
    }

    _nn = boost::make_shared<nn_t>(boost::ref(_nn_model));
    _nn->set_is_encoder(true);
#endif
  }

  void set_lambda(value_t lambda) {
    _lambda = lambda;
  }

  value_t lambda() const {
    return _lambda;
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

  template<class T2>
  vector_t& computeGV(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels, vector_t& v, bool damping = true) {

    // Do the computation
    _encoder.reset_gradient();
    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      sample = *samples[iSample];
      label = *labels[iSample];
      _encoder.inputs() = sample;
      _encoder.update_gv(v, label);
    }

    // TODO: incorporate weight mask
    gv = _encoder.gradient() + _weightcost * weight_mask * v;
    if (damping)
      gv += _lambda * v;

    return gv;
  }

  // Preconditioned conjugate gradients
  template<class T2>
  vector_t& pcg(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels, vector_t& b, vector_t& x0, vector_t& Mdiag) {
    value_t pAp, alpha, beta;

    x = x0;
    r = computeGV(samples, labels, x) - b;
    y = r / Mdiag;
    p = -y;

    val = 0.5 * dot(-b + r, x);

    for (size_t i = 0; i < _iteration_count; ++i) {
      tbblas_print(val);

      Ap = computeGV(samples, labels, p);
      pAp = dot(p, Ap);

      if (pAp <= 0) {
        std::cout << "Negative curvature! " << pAp << std::endl;
        break;
      }

      alpha = dot(r, y) / pAp;
      beta = value_t(1) / dot(r, y);

      x = x + alpha * p;
      r = r + alpha * Ap;
      y = r / Mdiag;

      beta *= dot(r, y);
      p = -y + beta * p;

      val = 0.5 * dot(-b + r, x);
    }

    return x;
  }

#ifdef COMPARE_TO_NN
  vector_t& convert_to_nn(vector_t& parameters) {
    nn_parameters.resize(seq(_nn_model.parameter_count()));
    nn_parameters[seq(0), parameters.size()] = parameters;
    nn_parameters[parameters.size(), nn_parameters.size() - parameters.size()] =
        ones<value_t>(nn_parameters.size() - parameters.size()) * parameters[parameters.size() - 1];
    return nn_parameters;
  }

  vector_t& convert_to_conv(vector_t& parameters) {
    conv_parameters.resize(this->parameters.size());
    conv_parameters[seq(0), conv_parameters.size() - 1] = parameters[seq(0), conv_parameters.size() - 1];
    conv_parameters[conv_parameters.size() - 1] = sum(parameters[conv_parameters.size() - 1, nn_parameters.size() - conv_parameters.size() + 1]);

    return conv_parameters;
  }

  template<class T2>
  void check_loss(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels) {
    parameters = _encoder.parameters();

    sample = *samples[0];
    label = *labels[0];

    _nn->set_parameters(convert_to_nn(parameters));

    matrix_t m_samples = reshape(sample, seq(1, (int)sample.count()));
    matrix_t m_labels = reshape(label, seq(1, (int)label.count()));

    _encoder.inputs() = sample;
    tbblas_print(_encoder.loss(sample));

    _nn->visibles() = m_samples;
    tbblas_print(_nn->loss(m_samples));
  }
#endif

  template<class T2>
  void check_gradient(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels, value_t epsilon) {
    value_t dloss = 0;
#ifdef COMPARE_TO_NN
    value_t nn_dloss = 0;
    matrix_t m_samples, m_labels;
#endif

    // Estimate the gradient using finite differences

    parameters = _encoder.parameters();

    vector_t dparam, grad(parameters.size()), nn_grad(parameters.size());
    for (int iParam = 0; iParam < parameters.count(); ++iParam) {

      // Forward loss
      dparam = parameters;
      dparam[seq(iParam)] = dparam[seq(iParam)] + epsilon;
      _encoder.set_parameters(dparam);
      dloss = 0;

#ifdef COMPARE_TO_NN
      _nn->set_parameters(convert_to_nn(dparam));
      nn_dloss = 0;
#endif
      for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
        sample = *samples[iSample];
        label = *labels[iSample];
        _encoder.inputs() = sample;
        dloss += _encoder.loss(label);

#ifdef COMPARE_TO_NN
        m_samples = reshape(sample, seq(1, (int)sample.count()));
        m_labels = reshape(label, seq(1, (int)label.count()));
        _nn->visibles() = m_samples;
        nn_dloss += _nn->loss(m_labels);
#endif
      }

      // Backward loss
      dparam = parameters;
      dparam[seq(iParam)] = dparam[seq(iParam)] - epsilon;
      _encoder.set_parameters(dparam);

#ifdef COMPARE_TO_NN
      _nn->set_parameters(convert_to_nn(dparam));
#endif

      for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
        sample = *samples[iSample];
        label = *labels[iSample];
        _encoder.inputs() = sample;
        dloss -= _encoder.loss(label);

#ifdef COMPARE_TO_NN
        m_samples = reshape(sample, seq(1, (int)sample.count()));
        m_labels = reshape(label, seq(1, (int)label.count()));
        _nn->visibles() = m_samples;
        nn_dloss -= _nn->loss(m_labels);
#endif
      }
      grad[seq(iParam)] = dloss / ((value_t)samples.size() * 2.0 * epsilon);
#ifdef COMPARE_TO_NN
      nn_grad[seq(iParam)] = nn_dloss / ((value_t)samples.size() * 2.0 * epsilon);
#endif

//      tbblas_print((float)iParam / (float)parameters.count());
    }

    // Calculate gradient
    _encoder.set_parameters(parameters);
    _encoder.reset_gradient();

#ifdef COMPARE_TO_NN
    _nn->set_parameters(convert_to_nn(parameters));
    _nn->reset_gradient();
#endif
    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      sample = *samples[iSample];
      label = *labels[iSample];
      _encoder.inputs() = sample;
      _encoder.update_gradient(label);

#ifdef COMPARE_TO_NN
      m_samples = reshape(sample, seq(1, (int)sample.count()));
      m_labels = reshape(label, seq(1, (int)label.count()));
      _nn->visibles() = m_samples;
      _nn->update_gradient(m_labels);
#endif
    }
    b = _encoder.gradient(0);
    tbblas_print(cor(grad, b));
    tbblas_print(::sqrt(dot(grad - b, grad - b) / parameters.count()));

//    tbblas_print(::sqrt(dot(grad - b, grad - b) / parameters.count()));

    std::ofstream file("grad.csv");
    file << "Pos,FD,Grad\n";
    for (int i = 0; i < grad.count(); ++i)
      file << i << "," << grad[seq(i)] << "," << b[seq(i)] << "\n";
    file.close();

#ifdef COMPARE_TO_NN
    vector_t nn_b = convert_to_conv(_nn->gradient(0));
    tbblas_print(cor(grad, nn_grad));
    tbblas_print(cor(nn_grad, nn_b));

    std::ofstream nn_file("nn_grad.csv");
    nn_file << "Pos,FD,Grad\n";
    for (int i = 0; i < nn_grad.count(); ++i)
      nn_file << i << "," << nn_grad[seq(i)] << "," << nn_b[seq(i)] << "\n";
    nn_file.close();
#endif
  }

  // Check the Gv product using Gv = J'(H_L(Jv))
  template<class T2>
  void check_Gv(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels, value_t epsilon, bool randomVector) {
    vector_t dparam;

    vector_t v, Jv, HJv, JHJv, e, Je;
#ifdef COMPARE_TO_NN
    vector_t nn_Jv, nn_HJv, nn_JHJv, nn_Je;
    matrix_t m_samples, m_labels;
#endif
    parameters = _encoder.parameters();

    random_tensor2<value_t, 1, true, normal<value_t> > randn(parameters.size());

    // Check for all basis vectors
    if (randomVector) {
      v = randn();
    } else {
      v = zeros<value_t>(parameters.count());
      v[seq(0)] = 1;
    }

    Je = Jv = zeros<value_t>(_encoder._model.outputs_count());
#ifdef COMPARE_TO_NN
    nn_Je = nn_Jv = Je;
#endif

    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      sample = *samples[iSample];
      label = *labels[iSample];
      _encoder.inputs() = sample;

#ifdef COMPARE_TO_NN
      m_samples = reshape(sample, seq(1, (int)sample.count()));
      m_labels = reshape(label, seq(1, (int)label.count()));
      _nn->visibles() = m_samples;
#endif

      // Jv = (f(x, theta + e*v) - f(x, theta - e*v)) / 2e;

      // Forward prediction
      dparam = parameters + epsilon * v;
      _encoder.set_parameters(dparam);
      _encoder.infer_outputs();
      Jv = reshape(_encoder.outputs(), Jv.size());

#ifdef COMPARE_TO_NN
      _nn->set_parameters(convert_to_nn(dparam));
      _nn->infer_hiddens();
      nn_Jv = reshape(_nn->hiddens(), nn_Jv.size());

      tbblas_print(dot(Jv, Jv));
      tbblas_print(dot(nn_Jv, nn_Jv));
      tbblas_print(cor(Jv, nn_Jv));
#endif

      // Backward prediction
      dparam = parameters - epsilon * v;
      _encoder.set_parameters(dparam);
      _encoder.infer_outputs();
      Jv -= reshape(_encoder.outputs(), Jv.size());
      Jv = Jv / (value_t)(2.0 * epsilon);

#ifdef COMPARE_TO_NN
      _nn->set_parameters(convert_to_nn(dparam));
      _nn->infer_hiddens();
      nn_Jv -= reshape(_nn->hiddens(), nn_Jv.size());
      nn_Jv = nn_Jv / (value_t)(2.0 * epsilon);

      tbblas_print(dot(Jv, Jv));
      tbblas_print(dot(nn_Jv, nn_Jv));
      tbblas_print(cor(Jv, nn_Jv));
#endif

      // For SSD, H_L = I -> Jv = HJv
      HJv = Jv / (value_t)label.count();

      if (_encoder.objective_function() == objective_function::SenSpe) {
        const value_t positive_ratio = sum(label) / (value_t)label.count();
        const value_t alpha = 2 * _encoder.sensitivity_ratio() / (positive_ratio + value_t(1e-8));
        const value_t beta =  2 * (value_t(1) - _encoder.sensitivity_ratio()) / (value_t(1) - positive_ratio + value_t(1e-8));

        HJv = (alpha * reshape(label, HJv.size()) + beta * (1 - reshape(label, HJv.size()))) * HJv;
      }

#ifdef COMPARE_TO_NN
      nn_HJv = nn_Jv;
#endif

      // J'w = [(Je_1)'w (Je_2)'w ... (Je_n)'w]
      JHJv = zeros<value_t>(parameters.count());
#ifdef COMPARE_TO_NN
      nn_JHJv = JHJv;
#endif
      for (int i = 0; i < JHJv.count(); ++i) {
        e = zeros<value_t>(JHJv.count());
        e[seq(i)] = 1;

        // Forward prediction
        dparam = parameters + epsilon * e;
        _encoder.set_parameters(dparam);
        _encoder.infer_outputs();
        Je = reshape(_encoder.outputs(), Je.size());

#ifdef COMPARE_TO_NN
        _nn->set_parameters(convert_to_nn(dparam));
        _nn->infer_hiddens();
        nn_Je = reshape(_nn->hiddens(), nn_Je.size());
#endif

        // Backward prediction
        dparam = parameters - epsilon * e;
        _encoder.set_parameters(dparam);
        _encoder.infer_outputs();
        Je -= reshape(_encoder.outputs(), Je.size());
        Je = Je / (value_t)(2.0 * epsilon);
        JHJv[seq(i)] = dot(Je, HJv);

#ifdef COMPARE_TO_NN
        _nn->set_parameters(convert_to_nn(dparam));
        _nn->infer_hiddens();
        nn_Je -= reshape(_nn->hiddens(), nn_Je.size());
        nn_Je = nn_Je / (value_t)(2.0 * epsilon);
        nn_JHJv[seq(i)] = dot(nn_Je, nn_HJv);

        if (i == 0) {
          tbblas_print(dot(Je, Je));
          tbblas_print(dot(nn_Je, nn_Je));
          tbblas_print(cor(Je, nn_Je));
        }
#endif

//        tbblas_print((float)i / (float)JHJv.count());
      }

      // Calculate Gv
      _encoder.set_parameters(parameters);
      _encoder.reset_gradient();
      _encoder.update_gv(v, label);
      gv = _encoder.gradient(0);
      assert(gv.size() == JHJv.size());

      tbblas_print(cor(JHJv, gv));
      tbblas_print(::sqrt(dot(JHJv - gv, JHJv - gv) / parameters.count()));

      std::ofstream file("Gv.csv");
      file << "Pos,JHJv,Gv\n";
      for (int i = 0; i < gv.count(); ++i)
        file << i << "," << JHJv[seq(i)] << "," << gv[seq(i)] << "\n";
      file.close();

#ifdef COMPARE_TO_NN
      tbblas_print(dot(JHJv, JHJv));
      tbblas_print(dot(nn_JHJv, nn_JHJv));
      tbblas_print(cor(JHJv, nn_JHJv));

      _nn->set_parameters(convert_to_nn(parameters));
      _nn->reset_Rgradient();
      _nn->update_Gv(convert_to_nn(v), m_labels);
      nn_gv = convert_to_conv(_nn->Rgradient(0));
      assert(nn_gv.size() == nn_JHJv.size());

      tbblas_print(cor(nn_JHJv, nn_gv));
      tbblas_print(::sqrt(dot(nn_JHJv - nn_gv, nn_JHJv - nn_gv) / parameters.count()));

      std::ofstream nn_file("nn_Gv.csv");
      nn_file << "Pos,JHJv,Gv\n";
      for (int i = 0; i < nn_gv.count(); ++i)
        nn_file << i << "," << nn_JHJv[seq(i)] << "," << nn_gv[seq(i)] << "\n";
      nn_file.close();
#endif
      break;
    }
  }

#ifdef COMPARE_TO_NN
  template<class T2>
  void check_Gv2(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels, value_t epsilon, bool randomVector) {
    vector_t dparam;

    vector_t v;
    matrix_t m_samples, m_labels;
    parameters = _encoder.parameters();

    random_tensor2<value_t, 1, true, normal<value_t> > randn(parameters.size());

    // Check for all basis vectors
    if (randomVector) {
      v = randn();
    } else {
      v = zeros<value_t>(parameters.count());
      v[seq(0)] = 1;
    }

    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      sample = *samples[iSample];
      label = *labels[iSample];
      m_samples = reshape(sample, seq(1, (int)sample.count()));
      m_labels = reshape(label, seq(1, (int)label.count()));

      _encoder.inputs() = sample;
      _nn->visibles() = m_samples;

      // Calculate Gv
      _encoder.set_parameters(parameters);
      _encoder.reset_gradient();
      _encoder.update_gv(v, label);
      gv = _encoder.gradient(0);

      _nn->set_parameters(convert_to_nn(parameters));
      _nn->reset_Rgradient();
      _nn->update_Gv(convert_to_nn(v), m_labels);
      nn_gv = convert_to_conv(_nn->Rgradient(0));
      assert(nn_gv.size() == gv.size());

      tbblas_print(cor(_encoder._Rs1, _nn->_Rs1));
      tbblas_print(cor(_encoder._Rs2, _nn->_Rs2));
      tbblas_print(cor(_encoder._Ra1, _nn->_Ra1));
      tbblas_print(cor(_encoder._a1, _nn->_a1));
      tbblas_print(cor(_encoder._a2, _nn->_a2));

      tbblas_print(cor(gv, nn_gv));
      tbblas_print(::sqrt(dot(nn_gv - gv, nn_gv - gv) / parameters.count()));

      std::ofstream file("Gv_comp.csv");
      file << "Pos,nn_Gv,Gv\n";
      for (int i = 0; i < gv.count(); ++i)
        file << i << "," << nn_gv[seq(i)] << "," << gv[seq(i)] << "\n";
      file.close();
      break;
    }
  }
#endif

//  template<class T2>
//  virtual void update(std::vector<boost::shared_ptr<tensor<T2, dims> > >& samples, std::vector<boost::shared_ptr<tensor<T2, dims> > >& labels) {
  virtual void update(v_host_tensor_t& samples, v_host_tensor_t& labels) {
    parameters = _encoder.parameters();
    weight_mask = _encoder.weight_mask();
    b = zeros<value_t>(_encoder._model.parameter_count());
    b2 = zeros<value_t>(_encoder._model.parameter_count());

    error = 0;

    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      sample = *samples[iSample];
      label = *labels[iSample];
      _encoder.inputs() = sample;

      _encoder.reset_gradient();
      error += _encoder.update_gradient(label);
      b += _encoder.gradient(_weightcost);
      b2 += b * b;
    }
    b = -b / samples.size();
    error = error / samples.size() + 0.5 * _weightcost * dot(weight_mask * parameters, weight_mask * parameters);

    tbblas_print(error);

#ifdef OLD_LAMBDA_UPDATE
    // Update lambda
    if (oldError > 0) {
      value_t rho = (error - oldError) / val;
      tbblas_print(error - oldError);
      tbblas_print(val);
      tbblas_print(rho);
      if (rho > 0.75) {
        _lambda *= 2./3.;
        tbblas_print(_lambda);
      } else if (rho < 0.25) {
        _lambda *= 3./2.;
        tbblas_print(_lambda);
      }
    }
#endif

    precon = pow(b2 + _lambda, 0.75);
    _delta = _zeta * _delta;
    _delta = pcg(samples, labels, b, _delta, precon);
    _encoder.update_model(_delta);
    parameters = parameters + _delta;

    oldError = error;
    val = 0.5 * dot(_delta, computeGV(samples, labels, _delta, false)) - dot(b, _delta);

    // Calculate new error
    value_t newError = 0;
    for (size_t iSample = 0; iSample < samples.size(); ++iSample) {
      sample = *samples[iSample];
      label = *labels[iSample];
      _encoder.inputs() = sample;

      newError += _encoder.loss(label);
    }
    newError = newError / samples.size() + 0.5 * _weightcost * dot(weight_mask * parameters, weight_mask * parameters);

    value_t rho = (newError - error) / val;
    tbblas_print(newError - error);
    tbblas_print(val);
    tbblas_print(rho);
    if (rho > 0.75) {
      _lambda *= 2./3.;
      tbblas_print(_lambda);
    } else if (rho < 0.25) {
      _lambda *= 3./2.;
      tbblas_print(_lambda);
    }

//    _lambda = max(_lambda, minLambda);
  }
};

}

}

}

#endif /* TBBLAS_DEEPLEARN_OPT_HESSIAN_FREE_HPP_ */
