/*
 * changestream.hpp
 *
 *  Created on: Aug 4, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_CHANGE_STREAM_HPP_
#define TBBLAS_CHANGE_STREAM_HPP_

#include <boost/shared_ptr.hpp>

#include <cuda_runtime_api.h>

namespace tbblas {

struct context;

class change_stream {
private:
  boost::shared_ptr<tbblas::context> _context;

public:
  change_stream(cudaStream_t stream);
  ~change_stream();
};

} /* namespace tbblas */

#endif /* TBBLAS_CHANGE_STREAM_HPP_ */
