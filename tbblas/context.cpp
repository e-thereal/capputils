/*
 * context.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: tombr
 */

#include "context.hpp"

#include <tbblas/context_manager.hpp>

namespace tbblas {

context::context() : stream(0)
#ifdef TBBLAS_HAVE_CUBLAS
, cublasHandle(0) 
#endif
{ }

context& context::get() {
  return context_manager::get().current();
}

} /* namespace tbblas */
