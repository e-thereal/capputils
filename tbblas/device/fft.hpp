/*
 * fft.hpp
 *
 *  Created on: Aug 30, 2012
 *      Author: tombr
 */

#ifndef TBBLAS_DEVICE_FFT_HPP_
#define TBBLAS_DEVICE_FFT_HPP_

#include <tbblas/tensor_base.hpp>

namespace tbblas {

// Forward declarations

template<class T, unsigned dim, bool device>
void fft(const tensor_base<T, dim, device>& dt, const size_t (&size)[dim],
    typename vector_type<typename complex_type<T>::complex_t, device>::vector_t& ftdata);

template<class T, unsigned dim, bool device>
void ifft(typename vector_type<typename complex_type<T>::complex_t, device>::vector_t& ftdata,
    const size_t (&size)[dim], tensor_base<T, dim, device>& dt);

// Template specializations

// 2D cases

template<>
void fft(const tensor_base<float, 2, true>& dt, const size_t (&size)[2],
    thrust::device_vector<complex_type<float>::complex_t>& ftdata);

template<>
void ifft(thrust::device_vector<complex_type<float>::complex_t>& ftdata,
    const size_t (&size)[2], tensor_base<float, 2, true>& dt);

// 3D cases

template<>
void fft(const tensor_base<float, 3, true>& dt, const size_t (&size)[3],
    thrust::device_vector<complex_type<float>::complex_t>& ftdata);

template<>
void ifft(thrust::device_vector<complex_type<float>::complex_t>& ftdata,
    const size_t (&size)[3], tensor_base<float, 3, true>& dt);

template<>
void fft(const tensor_base<double, 3, true>& dt, const size_t (&size)[3],
    thrust::device_vector<complex_type<double>::complex_t>& ftdata);

template<>
void ifft(thrust::device_vector<complex_type<double>::complex_t>& ftdata,
    const size_t (&size)[3], tensor_base<double, 3, true>& dt);

}

#endif /* TBBLAS_DEVICE_FFT_HPP_ */
