/*
 * trans.hpp
 *
 *  Created on: Nov 20, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_TRANS_HPP_
#define TBBLAS_TRANS_HPP_

#include <tbblas/proxy.hpp>

namespace tbblas {

template<class T, bool device>
proxy<tensor<T, 2, device> > trans(const proxy<tensor<T, 2, device> >& p) {
  proxy<tensor<T, 2, device> > proxy = p;
  typename tbblas::proxy<tensor<T, 2, device> >::dim_t order = seq(1, 0);
  proxy.reorder(order);
  return proxy;
}

template<class T, bool device>
proxy<tensor<T, 2, device> > trans(tensor<T, 2, device>& t) {
  proxy<tensor<T, 2, device> > proxy(t);
  typename tbblas::proxy<tensor<T, 2, device> >::dim_t order = seq(1, 0);
  proxy.reorder(order);
  return proxy;
}

}

#endif /* TBBLAS_TRANS_HPP_ */
