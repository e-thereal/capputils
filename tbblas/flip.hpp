/*
 * flip.hpp
 *
 *  Created on: Sep 26, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_FLIP_HPP_
#define TBBLAS_FLIP_HPP_

#include <tbblas/proxy.hpp>
#include <tbblas/assert.hpp>

namespace tbblas {

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(const proxy<tensor<T, dim, device> >& p) {
  proxy<tensor<T, dim, device> > proxy = p;
  sequence<bool, dim> flipped = proxy.flipped();
  for (unsigned i = 0; i < dim; ++i)
    flipped[i] = !flipped[i];
  proxy.set_flipped(flipped);
  return proxy;
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(tensor<T, dim, device>& t) {
  proxy<tensor<T, dim, device> > proxy(t);
  sequence<bool, dim> flipped = proxy.flipped();
  for (unsigned i = 0; i < dim; ++i)
    flipped[i] = !flipped[i];
  proxy.set_flipped(flipped);
  return proxy;
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(const proxy<tensor<T, dim, device> >& p, size_t idx) {
  tbblas_assert(idx < dim);

  proxy<tensor<T, dim, device> > proxy = p;
  sequence<bool, dim> flipped = proxy.flipped();
  flipped[(int)idx] = !flipped[(int)idx];
  proxy.set_flipped(flipped);
  return proxy;
}

template<class T, unsigned dim, bool device>
proxy<tensor<T, dim, device> > flip(tensor<T, dim, device>& t, size_t idx) {
  tbblas_assert(idx < dim);

  proxy<tensor<T, dim, device> > proxy(t);
  sequence<bool, dim> flipped = proxy.flipped();
  flipped[(int)idx] = !flipped[(int)idx];
  proxy.set_flipped(flipped);
  return proxy;
}

}

#endif /* TBBLAS_FLIP_HPP_ */
