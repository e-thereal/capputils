/*
 * newcontext.hpp
 *
 *  Created on: Jul 30, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_NEWCONTEXT_HPP_
#define TBBLAS_NEWCONTEXT_HPP_

#include <string>
#include <boost/shared_ptr.hpp>

#include <cuda_runtime_api.h>

namespace tbblas {

struct context;

class new_context {
private:
  boost::shared_ptr<tbblas::context> _context;

public:
  new_context();
  ~new_context();
};

} /* namespace tbblas */

#endif /* TBBLAS_NEWCONTEXT_HPP_ */
