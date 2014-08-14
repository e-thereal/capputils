/*
 * contextmanager.hpp
 *
 *  Created on: Jul 30, 2014
 *      Author: tombr
 */

#ifndef TBBLAS_CONTEXTMANAGER_HPP_
#define TBBLAS_CONTEXTMANAGER_HPP_

#include <tbblas/context.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include <stack>
#include <map>

namespace tbblas {

class new_context;
class change_stream;

class context_manager {

  friend class new_context;
  friend class change_stream;

private:
  typedef std::stack<boost::shared_ptr<tbblas::context> > stack_t;
  typedef std::map<boost::thread::id, stack_t> map_t;

  map_t contexts;
  tbblas::context default_context;
  boost::mutex mtx;

private:
  context_manager();

public:
  ~context_manager();

  static context_manager& get();

  tbblas::context& current();

private:
  void add(boost::shared_ptr<tbblas::context> context);
  void remove(boost::shared_ptr<tbblas::context> context);
};

} /* namespace tbblas */

#endif /* TBBLAS_CONTEXTMANAGER_HPP_ */
