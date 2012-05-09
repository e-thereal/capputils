/*
 * device_tensor_gpu.cu
 *
 *  Created on: Mar 2, 2012
 *      Author: tombr
 */

#include "device_tensor.hpp"

#include <culib/CulibException.h>

#include <iostream>

namespace tbblas {

#define CHECK_CUFFT_ERROR(a) \
  if ((res = a) != CUFFT_SUCCESS) { \
    std::cout << "Error in file " << __FILE__ << ", line " << __LINE__ << ": " << res << std::endl; \
  }

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

  // TODO: potential problem with the size order. Find a way to test it!
  CHECK_CUFFT_ERROR(cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_R2C));
  CHECK_CUFFT_ERROR(cufftExecR2C(plan, padded.data().data().get(), ftdata.data().get()));
  CHECK_CUFFT_ERROR(cufftDestroy(plan));
  CULIB_SAFE_CALL(cudaThreadSynchronize());
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
  CHECK_CUFFT_ERROR(cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_C2R));
  CHECK_CUFFT_ERROR(cufftExecC2R(plan, ftdata.data().get(), padded.data().data().get()));
  CHECK_CUFFT_ERROR(cufftDestroy(plan));

  size_t start[3];
  for (int i = 0; i < 3; ++i)
    start[i]= abs((int)size[i] - (int)dt.size()[i]);

  const_proxy_t proxy = subrange(padded / (float)count, start, dt.size());
  thrust::copy(proxy.begin(), proxy.end(), dt.begin());
  CULIB_SAFE_CALL(cudaThreadSynchronize());
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

//  std::cout << "Size: " << dt.size()[0] << ", " << dt.size()[1] << ", " << dt.size()[2] << std::endl;
//  std::cout << "Padded Size: " << size[0] << ", " << size[1] << ", " << size[2] << std::endl;
//  std::cout << padded.data().size() << " == " << ftdata.size() << std::endl;

  assert(size[0] > 1);

  if (size[1] > 1) {
    if (size[2] > 1) {
      CHECK_CUFFT_ERROR(cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_D2Z));
    } else {
      CHECK_CUFFT_ERROR(cufftPlan2d(&plan, size[1], size[0], CUFFT_D2Z));
    }
  } else {
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, size[0], CUFFT_D2Z, 1));
  }
  CHECK_CUFFT_ERROR(cufftExecD2Z(plan, padded.data().data().get(), ftdata.data().get()));
  CHECK_CUFFT_ERROR(cufftDestroy(plan));
  CULIB_SAFE_CALL(cudaThreadSynchronize());
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

  assert(size[0] > 1);

  if (size[1] > 1) {
    if (size[2] > 1) {
      CHECK_CUFFT_ERROR(cufftPlan3d(&plan, size[2], size[1], size[0], CUFFT_Z2D));
    } else {
      CHECK_CUFFT_ERROR(cufftPlan2d(&plan, size[1], size[0], CUFFT_Z2D));
    }
  } else {
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, size[0], CUFFT_Z2D, 1));
  }
  CHECK_CUFFT_ERROR(cufftExecZ2D(plan, ftdata.data().get(), padded.data().data().get()));
  CHECK_CUFFT_ERROR(cufftDestroy(plan));

  size_t start[3];
  for (int i = 0; i < 3; ++i)
    start[i]= abs((int)size[i] - (int)dt.size()[i]);

  const_proxy_t proxy = subrange(padded / (value_t)count, start, dt.size());
  thrust::copy(proxy.begin(), proxy.end(), dt.begin());
  CULIB_SAFE_CALL(cudaThreadSynchronize());
}

}
