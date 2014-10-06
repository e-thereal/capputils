/*
 * transform_gpu.cu
 *
 *  Created on: 2014-10-05
 *      Author: tombr
 */

#include <tbblas/transform/transform.hpp>

namespace tbblas {

namespace transform {

texture<float, 3, cudaReadModeElementType> tex;  // 3D texture

__global__ void transform3DKernel(float *d_result, tensor<float, 3>::dim_t dimension, fmatrix4 transMatrix, uint k) {
  uint i = (blockDim.x * blockIdx.x + threadIdx.x);
  uint j = (blockDim.y * blockIdx.y + threadIdx.y);

  if (i >= dimension[0] || j >= dimension[1] || k >= dimension[2])
    return;

//  float4 v = transMatrix * make_float4(i + 0.5, j + 0.5, k + 0.5, 1);
//  d_result[(k * dimension[1] + j) * dimension[0] + i] = tex3D(tex, get_x(v), get_y(v), get_z(v));

  float4 v = transMatrix * make_float4(i, j, k, 1);
  d_result[(k * dimension[1] + j) * dimension[0] + i] = tex3D(tex, get_x(v) + 0.5, get_y(v) + 0.5, get_z(v) + 0.5);
}

template<>
void transform(tensor<float, 3, true>& input, const fmatrix4& matrix, tensor<float, 3, true>& output) {
  const int BlockWidth = 16;
  const int BlockHeight = 16;

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
  tex.normalized = false;                    // access with normalized texture coordinates
  tex.filterMode = cudaFilterModeLinear;    // linear interpolation

  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.addressMode[2] = cudaAddressModeClamp;

  // bind array to 3D texture
  cudaBindTextureToArray(tex, d_cudaArray, texDesc);

  // Start kernel
//  dim3 gridDim((output.size()[0] / skip.x+BlockWidth-1)/BlockWidth,
//      (dimension.y/skip.y+BlockHeight-1)/BlockHeight);
  dim3 gridDim((output.size()[0] + BlockWidth - 1) / BlockWidth,
      (output.size()[1] + BlockHeight - 1) / BlockHeight);
  dim3 blockDim(BlockWidth, BlockHeight);

  for (uint k = 0; k < output.size()[2]; ++k) {
    transform3DKernel<<<gridDim, blockDim>>>(output.data().data().get(), output.size(), matrix, k);
  }

  cudaUnbindTexture(tex);
  cudaFreeArray(d_cudaArray);
}

}

}
