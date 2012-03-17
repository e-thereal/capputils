/*
 * device_tensor_gpu.cu
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#include "device_tensor.hpp"

namespace tbblas {

template<>
void fft<float, 3, true>(const tensor_base<float, 3, true>& dt, const size_t (&size)[3],
    thrust::device_vector<complex_type<float>::complex_t>& ftdata)
{
  typedef device_tensor<float, 3> tensor_t;
  typedef tensor_proxy<tensor_t::iterator, 3> proxy_t;

  const size_t start[3] = {0, 0, 0};

  tensor_t padded(size);
  proxy_t proxy = subrange(padded, start, dt.size());
  thrust::copy(dt.begin(), dt.end(), proxy.begin());

  cufftHandle plan;
  cufftResult res;

  res = cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_R2C);
  cufftExecR2C(plan, padded.data().data().get(), ftdata.data().get());
  cufftDestroy(plan);
}

template<>
void ifft<float, 3, true>(thrust::device_vector<complex_type<float>::complex_t>& ftdata,
    const size_t (&size)[3], tensor_base<float, 3, true>& dt)
{
  const size_t count = size[0] * size[1] * size[2];
  typedef device_tensor<float, 3> tensor_t;
  typedef tensor_proxy<tensor_t::const_iterator, 3> const_proxy_t;

  cufftHandle plan;
  cufftResult res;

  tensor_t padded(size);
  res = cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_C2R);
  cufftExecC2R(plan, ftdata.data().get(), padded.data().data().get());
  cufftDestroy(plan);

  size_t start[3];
  for (int i = 0; i < 3; ++i)
    start[i]= abs((int)size[i] - (int)dt.size()[i]);

  const_proxy_t proxy = subrange(padded / (float)count, start, dt.size());
  thrust::copy(proxy.begin(), proxy.end(), dt.begin());
}

template<>
void fft<double, 3, true>(const tensor_base<double, 3, true>& dt, const size_t (&size)[3],
    thrust::device_vector<complex_type<double>::complex_t>& ftdata)
{
  typedef double value_t;
  typedef device_tensor<value_t, 3> tensor_t;
  typedef tensor_proxy<tensor_t::iterator, 3> proxy_t;

  const size_t start[3] = {0, 0, 0};

  tensor_t padded(size);
  proxy_t proxy = subrange(padded, start, dt.size());
  thrust::copy(dt.begin(), dt.end(), proxy.begin());

  cufftHandle plan;
  cufftResult res;

  res = cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_D2Z);
  cufftExecD2Z(plan, padded.data().data().get(), ftdata.data().get());
  cufftDestroy(plan);
}

template<>
void ifft<double, 3, true>(thrust::device_vector<complex_type<double>::complex_t>& ftdata,
    const size_t (&size)[3], tensor_base<double, 3, true>& dt)
{
  typedef double value_t;
  typedef device_tensor<value_t, 3> tensor_t;
  typedef tensor_proxy<tensor_t::const_iterator, 3> const_proxy_t;

  const size_t count = size[0] * size[1] * size[2];

  cufftHandle plan;
  cufftResult res;

  tensor_t padded(size);
  res = cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_Z2D);
  cufftExecZ2D(plan, ftdata.data().get(), padded.data().data().get());
  cufftDestroy(plan);

  size_t start[3];
  for (int i = 0; i < 3; ++i)
    start[i]= abs((int)size[i] - (int)dt.size()[i]);

  const_proxy_t proxy = subrange(padded / (value_t)count, start, dt.size());
  thrust::copy(proxy.begin(), proxy.end(), dt.begin());
}

}
