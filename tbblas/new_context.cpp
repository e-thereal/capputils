/*
 * newcontext.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: tombr
 */

#include "new_context.hpp"

#include <iostream>

#include "context_manager.hpp"

namespace tbblas {

new_context::new_context() : _context(new tbblas::context()) {
  cudaStreamCreate(&_context->stream);

#ifdef TBBLAS_HAVE_CUBLAS
  cublasCreate(&_context->cublasHandle);
#endif

  context_manager::get().add(_context);
}

new_context::~new_context() {
  cudaStreamSynchronize(_context->stream);
  cudaStreamDestroy(_context->stream);

#ifdef TBBLAS_HAVE_CUBLAS
  cublasDestroy(_context->cublasHandle);
#endif

  context_manager::get().remove(_context);
}

} /* namespace tbblas */
