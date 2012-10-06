/*
 * fill.hpp
 *
 *  Created on: Oct 5, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_FILL_HPP_
#define TBBLAS_FILL_HPP_

#include <boost/utility/enable_if.hpp>

namespace tbblas {

// works also with a tensor
template<class Proxy>
struct proxy_filler {

  typedef typename Proxy::iterator iterator;
  typedef proxy_filler<Proxy> filler_t;
  typedef typename Proxy::value_t value_t;

  proxy_filler(const Proxy& proxy) : curr(proxy.begin()), end(proxy.end()) { }

  filler_t& operator,(const value_t& value) {
    if (curr != end) {
      *curr = value;
      ++curr;
    }
    return *this;
  }

private:
  iterator curr, end;
};

}

#endif /* TBBLAS_FILL_HPP_ */
