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

    unsigned rank = dim;
    for (; rank > 0; --rank) {
      if (size[rank-1] > 1)
        break;
    }

    int n[dim]; // doesn't hurt to be a little bit bigger than necessary and MSVC complains about non constant rank
    for (unsigned i = 0; i < rank; ++i)
      n[i] = size[rank - i - 1];

    assert(cufftPlanMany(&plan, rank, n,
        0, 0, 0,
        0, 0, 0,
        type, 1) == CUFFT_SUCCESS);

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
