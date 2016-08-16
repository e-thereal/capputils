/*
* trans.hpp
*
*  Created on: July 28, 2016
*      Author: tombr
*/

#ifndef TBBLAS_REORDER_HPP_
#define TBBLAS_REORDER_HPP_

#include <tbblas/proxy.hpp>

namespace tbblas {

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
	proxy<Tensor>
>::type
reorder(Tensor& t, const typename Tensor::dim_t& order) {
	proxy<Tensor> p(t);
	p.reorder(order);
	return p;
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
	proxy<Tensor>
>::type
reorder(proxy<Tensor>& p, const typename Tensor::dim_t& order) {
	proxy<Tensor> p2(p);
	p2.reorder(order);
	return p2;
}

template<class Tensor>
typename boost::enable_if<is_tensor<Tensor>,
	const proxy<Tensor>
>::type
reorder(const proxy<Tensor>& p, const typename Tensor::dim_t& order) {
	proxy<Tensor> p2(p);
	p2.reorder(order);
	return p2;
}

//template<class T, bool device>
//proxy<tensor<T, 2, device> > reorder(tensor<T, 2, device>& t) {
//	proxy<tensor<T, 2, device> > proxy(t);
//	typename tbblas::proxy<tensor<T, 2, device> >::dim_t order = seq(1, 0);
//	proxy.reorder(order);
//	return proxy;
//}
//
//template<class T, bool device>
//const proxy<tensor<T, 2, device> > trans(const proxy<tensor<T, 2, device> >& p) {
//	proxy<tensor<T, 2, device> > proxy = p;
//	typename tbblas::proxy<tensor<T, 2, device> >::dim_t order = seq(1, 0);
//	proxy.reorder(order);
//	return proxy;
//}
//
//template<class T, bool device>
//const proxy<tensor<T, 2, device> > trans(const tensor<T, 2, device>& t) {
//	proxy<tensor<T, 2, device> > proxy(t);
//	typename tbblas::proxy<tensor<T, 2, device> >::dim_t order = seq(1, 0);
//	proxy.reorder(order);
//	return proxy;
//}

}

#endif /* TBBLAS_TRANS_HPP_ */
