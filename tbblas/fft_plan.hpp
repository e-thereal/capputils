/*
 * fft_plan.hpp
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_FFT_PLAN_HPP_
#define TBBLAS_FFT_PLAN_HPP_

#include <tbblas/sequence.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>

#include <cufft.h>

#include <cassert>

namespace tbblas {

template<unsigned dim>
class fft_plan_handle : boost::noncopyable {

  typedef sequence<unsigned, dim> dim_t;

private:
  cufftHandle plan;
  cufftType_t type;
  dim_t size;
  bool hasPlan;

public:
  fft_plan_handle() : hasPlan(false) {
  }

  ~fft_plan_handle() {
    if (hasPlan) {
      cufftDestroy(plan);
    }
  }

public:
  cufftHandle create(const dim_t& size, cufftType_t type) {
    if (hasPlan && this->size == size && this->type == type)
      return plan;

    if (hasPlan)
      cufftDestroy(plan);

    this->size = size;
    this->type = type;
    this->hasPlan = true;

    int n[dim];
    for (unsigned i = 0; i < dim; ++i)
      n[i] = size[dim - i - 1];

    assert(cufftPlanMany(&plan, dim, n,
        0, 0, 0,
        0, 0, 0,
        type, 1) == 0);

    return plan;
  }
};

template<unsigned dim>
class fft_plan {

  typedef sequence<unsigned, dim> dim_t;

private:
  boost::shared_ptr<fft_plan_handle<dim> > handle;

public:
  fft_plan() : handle(new fft_plan_handle<dim>()) { }

  cufftHandle create(const dim_t& size, cufftType_t type) const {
    return handle->create(size, type);
  }
};

}

#endif /* TBBLAS_FFT_PLAN_HPP_ */
