/*
 * changestream.cpp
 *
 *  Created on: Aug 4, 2014
 *      Author: tombr
 */

#include "change_stream.hpp"

#include "context_manager.hpp"

namespace tbblas {

change_stream::change_stream(cudaStream_t stream) : _context(new tbblas::context()) {
  _context->stream = stream;
#ifdef TBBLAS_HAVE_CUBLAS
  _context->cublasHandle = context::get().cublasHandle;
#endif
  context_manager::get().add(_context);
}

change_stream::~change_stream() {
  context_manager::get().remove(_context);
}

} /* namespace tbblas */
