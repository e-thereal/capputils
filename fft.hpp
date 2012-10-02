/*
 * fft.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_FFT_HPP_
#define TBBLAS_FFT_HPP_

#include <tbblas/type_traits.hpp>

#include <cassert>

namespace tbblas {

template<class T, unsigned dim, bool device>
void fft(const tensor_base<T, dim, device>& dt, const size_t (&size)[dim],
    typename vector_type<typename complex_type<T>::type, device>::vector_t& ftdata)
{
  assert(0);
}

template<class T, unsigned dim, bool device>
void ifft(typename vector_type<typename complex_type<T>::type, device>::vector_t& ftdata,
    const size_t (&size)[dim], tensor_base<T, dim, device>& dt)
{
  assert(0);
}

}

#include <tbblas/device/fft.hpp>

#endif /* TBBLAS_FFT_HPP_ */
