/*
 * warp_gpu.cu
 *
 *  Created on: Nov 7, 2014
 *      Author: tombr
 */

#include <tbblas/imgproc/warp.hpp>

namespace tbblas {

namespace imgproc {

texture<float, 3, cudaReadModeElementType> warp_tex;  // 3D texture

__global__ void warp3DKernel(float *d_result, tensor<float, 3>::dim_t dimension, float* d_deformation, sequence<float, 3> voxelSize, uint k) {
  uint i = (blockDim.x * blockIdx.x + threadIdx.x);
  uint j = (blockDim.y * blockIdx.y + threadIdx.y);

  if (i >= dimension[0] || j >= dimension[1] || k >= dimension[2])
    return;

  // t    - coordinate in target space
  // s    - coordinate in source space
  // u(t) - deformation field defined in target space
  // s = t + u(t)

  const int t = (k * dimension[1] + j) * dimension[0] + i;
  const int size = dimension[0] * dimension[1] * dimension[2];

  d_result[t] = tex3D(warp_tex,
      (float)i + d_deformation[t] / voxelSize[0] + 0.5f,
      (float)j + d_deformation[t + size] / voxelSize[1] + 0.5f,
      (float)k + d_deformation[t + 2 * size] / voxelSize[2] + 0.5f);
}

template<>
void warp(tensor<float, 3, true>& input, const sequence<float, 3>& voxel_size, tensor<float, 4, true>& deformation, tensor<float, 3, true>& output) {
  const int BlockWidth = 16;
  const int BlockHeight = 16;

  assert(deformation.size()[0] == input.size()[0]);
  assert(deformation.size()[1] == input.size()[1]);
  assert(deformation.size()[2] == input.size()[2]);
  assert(deformation.size()[3] == 3);

  cudaArray *d_cudaArray;

  const cudaExtent volumeSize = make_cudaExtent(input.size()[0], input.size()[1], input.size()[2]);
  cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&d_cudaArray, &texDesc, volumeSize);

  cudaMemcpy3DParms cpy_params = {0};
  cpy_params.extent = volumeSize;
  cpy_params.kind = cudaMemcpyDeviceToDevice;
  cpy_params.dstArray = d_cudaArray;
  cpy_params.srcPtr = make_cudaPitchedPtr(input.data().data().get(), input.size()[0] * sizeof(float),
      input.size()[0], input.size()[1]);
  cudaMemcpy3D(&cpy_params);

  // set texture parameters
  warp_tex.normalized = false;                    // access with normalized texture coordinates
  warp_tex.filterMode = cudaFilterModeLinear;     // linear interpolation

  warp_tex.addressMode[0] = cudaAddressModeClamp;
  warp_tex.addressMode[1] = cudaAddressModeClamp;
  warp_tex.addressMode[2] = cudaAddressModeClamp;

  // bind array to 3D texture
  cudaBindTextureToArray(warp_tex, d_cudaArray, texDesc);

  // Start kernel
  dim3 gridDim((output.size()[0] + BlockWidth - 1) / BlockWidth,
      (output.size()[1] + BlockHeight - 1) / BlockHeight);
  dim3 blockDim(BlockWidth, BlockHeight);

  for (uint k = 0; k < output.size()[2]; ++k) {
    warp3DKernel<<<gridDim, blockDim>>>(output.data().data().get(), output.size(), deformation.data().data().get(), voxel_size, k);
  }

  cudaUnbindTexture(warp_tex);
  cudaFreeArray(d_cudaArray);
}

template<>
void warp(tensor<float, 4, true>& input, const sequence<float, 3>& voxel_size, tensor<float, 4, true>& deformation, tensor<float, 4, true>& output) {
  const int BlockWidth = 16;
  const int BlockHeight = 16;

  assert(deformation.size()[0] == input.size()[0]);
  assert(deformation.size()[1] == input.size()[1]);
  assert(deformation.size()[2] == input.size()[2]);
  assert(deformation.size()[3] == 3);

  cudaArray *d_cudaArray;

  const cudaExtent volumeSize = make_cudaExtent(input.size()[0], input.size()[1], input.size()[2]);
  cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&d_cudaArray, &texDesc, volumeSize);

  for (int iChannel = 0; iChannel < input.size()[3]; ++iChannel) {

    const int offset = iChannel * input.size()[0] * input.size()[1] * input.size()[2];

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent = volumeSize;
    cpy_params.kind = cudaMemcpyDeviceToDevice;
    cpy_params.dstArray = d_cudaArray;
    cpy_params.srcPtr = make_cudaPitchedPtr(input.data().data().get() + offset, input.size()[0] * sizeof(float),
        input.size()[0], input.size()[1]);
    cudaMemcpy3D(&cpy_params);

    // set texture parameters
    warp_tex.normalized = false;                    // access with normalized texture coordinates
    warp_tex.filterMode = cudaFilterModeLinear;     // linear interpolation

    warp_tex.addressMode[0] = cudaAddressModeClamp;
    warp_tex.addressMode[1] = cudaAddressModeClamp;
    warp_tex.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaBindTextureToArray(warp_tex, d_cudaArray, texDesc);

    // Start kernel
    dim3 gridDim((output.size()[0] + BlockWidth - 1) / BlockWidth,
        (output.size()[1] + BlockHeight - 1) / BlockHeight);
    dim3 blockDim(BlockWidth, BlockHeight);

    for (uint k = 0; k < output.size()[2]; ++k) {
      warp3DKernel<<<gridDim, blockDim>>>(output.data().data().get() + offset, seq(output.size()[0], output.size()[1], output.size()[2]), deformation.data().data().get(), voxel_size, k);
    }

    cudaUnbindTexture(warp_tex);
  }
  cudaFreeArray(d_cudaArray);
}

}

}
