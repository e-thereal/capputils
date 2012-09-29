/*
 * flip.hpp
 *
 *  Created on: Sep 26, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_FLIP_HPP_
#define TBBLAS_FLIP_HPP_

#include <tbblas/proxy.hpp>
#include <cassert>

namespace tbblas {

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(const proxy<tensor<T, dim, device> >& p) {
  proxy<tensor<T, dim, device> > proxy = p;
  for (unsigned i = 0; i < dim; ++i)
    proxy.flipped[i] = !proxy.flipped[i];
  return proxy;
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(const tensor<T, dim, device>& t) {
  proxy<tensor<T, dim, device> > proxy(t);
  for (unsigned i = 0; i < dim; ++i)
    proxy.flipped[i] = !proxy.flipped[i];
  return proxy;
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(const proxy<tensor<T, dim, device> >& p, size_t idx) {
  assert(idx < dim);

  proxy<tensor<T, dim, device> > proxy = p;
  proxy.flipped[idx] = !proxy.flipped[idx];
  return proxy;
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(const tensor<T, dim, device>& t, size_t idx) {
  assert(idx < dim);

  proxy<tensor<T, dim, device> > proxy(t);
  proxy.flipped[idx] = !proxy.flipped[idx];
  return proxy;
}

}

#endif /* TBBLAS_FLIP_HPP_ */
