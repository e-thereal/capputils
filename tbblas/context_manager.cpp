/*
 * contextmanager.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: tombr
 */

#include "context_manager.hpp"

#include <boost/thread/locks.hpp>

#include <iostream>

namespace tbblas {

context_manager::context_manager() {
  default_context.stream = 0;
  cublasCreate(&default_context.cublasHandle);
}

context_manager::~context_manager() {
  cublasDestroy(default_context.cublasHandle);
}

context_manager& context_manager::get() {
  // should be thread-safe as be the c++11 standard
  static context_manager instance;
  return instance;
}

tbblas::context& context_manager::current() {
  boost::lock_guard<boost::mutex> guard(mtx);

  if (contexts.find(boost::this_thread::get_id()) != contexts.end()) {
    stack_t& s = contexts[boost::this_thread::get_id()];
    if (!s.empty())
      return *s.top();
  }

  return default_context;
}

void context_manager::add(boost::shared_ptr<tbblas::context> context) {
  boost::lock_guard<boost::mutex> guard(mtx);

  if (contexts.find(boost::this_thread::get_id()) != contexts.end()) {
    contexts[boost::this_thread::get_id()].push(context);
//    std::cout << "added context" << std::endl;
  } else {
    stack_t s;
    s.push(context);
    contexts[boost::this_thread::get_id()] = s;
//    std::cout << "created new context stack" << std::endl;
  }
}

void context_manager::remove(boost::shared_ptr<tbblas::context> context) {
  boost::lock_guard<boost::mutex> guard(mtx);

  assert(contexts.find(boost::this_thread::get_id()) != contexts.end());

  stack_t& s = contexts[boost::this_thread::get_id()];

  assert (context.get() == s.top().get());
  s.pop();

  if (s.empty()) {
    contexts.erase(boost::this_thread::get_id());
//    std::cout << "Deleted context stack" << std::endl;
  }
}

} /* namespace tbblas */
