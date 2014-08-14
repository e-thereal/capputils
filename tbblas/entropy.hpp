/*
 * entropy.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_ENTROPY_HPP_
#define TBBLAS_ENTROPY_HPP_

#include <tbblas/tensor.hpp>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace tbblas {

struct entropy_float {
  __host__ __device__
  float operator()(const float& x) const {
    if (x > 0.f)
      return x * logf(x);
    else
      return 0.f;
  }
};

struct entropy_float_norm {

  entropy_float_norm(float count) : count(count) { }

  __host__ __device__
  float operator()(const float& x) const {
    if (x > 0.f)
      return x / count * logf(x / count);
    else
      return 0.f;
  }

private:
  float count;
};

struct entropy_double {
  __host__ __device__
  double operator()(const double& x) const {
    if (x > 0.0)
      return x * logf(x);
    else
      return 0.0;
  }
};

struct entropy_double_norm {

  entropy_double_norm(double count) : count(count) { }

  __host__ __device__
  double operator()(const double& x) const {
    if (x > 0.0)
      return x / count * logf(x / count);
    else
      return 0.0;
  }

private:
  double count;
};

template<unsigned dim, bool device>
float entropy(const tensor<float, dim, device>& tensor, float totalSum = 1.f) {
  if (totalSum == 1.f) {
    return -tbblas::detail::transform_reduce(
        typename tbblas::detail::select_system<device>::system(),
        tensor.begin(), tensor.end(),
        entropy_float(), 0.f, thrust::plus<float>());
  } else {
    return -tbblas::detail::transform_reduce(
        typename tbblas::detail::select_system<device>::system(),
        tensor.begin(), tensor.end(),
        entropy_float_norm(totalSum), 0.f, thrust::plus<float>());
  }
}

template<unsigned dim, bool device>
double entropy(const tensor<double, dim, device>& tensor, double totalSum = 1.0) {
  if (totalSum == 1.0) {
    return -tbblas::detail::transform_reduce(
        typename tbblas::detail::select_system<device>::system(),
        tensor.begin(), tensor.end(),
        entropy_double(), 0.0, thrust::plus<double>());
  } else {
    return -tbblas::detail::transform_reduce(
        typename tbblas::detail::select_system<device>::system(),
        tensor.begin(), tensor.end(),
        entropy_double_norm(totalSum), 0.0, thrust::plus<double>());
  }
}

}

#endif /* TBBLAS_ENTROPY_HPP_ */
