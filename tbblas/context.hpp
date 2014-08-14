/*
 * context.hpp
 *
 *  Created on: Jul 30, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_CONTEXT_HPP_
#define TBBLAS_CONTEXT_HPP_

#include <string>

#include <cuda_runtime_api.h>

#include <tbblas/tbblas.hpp>

#ifdef TBBLAS_HAVE_CUBLAS
#include <cublas_v2.h>
#endif

namespace tbblas {

class context_manager;
class new_context;
class change_stream;

struct context {
  cudaStream_t stream;
  cublasHandle_t cublasHandle;

  friend class context_manager;
  friend class new_context;
  friend class change_stream;

private:
  context();

public:
  static context& get();
};

} /* namespace tbblas */

#endif /* TBBLAS_CONTEXT_HPP_ */
