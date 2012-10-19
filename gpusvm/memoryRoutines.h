#ifndef GPUSVM_MEMORY_ROUTINES
#define GPUSVM_MEMORY_ROUTINES

namespace gpusvm {

__device__ void coopExtractRowVector(float* data, int dataPitch, int index, int dimension, float* destination) {
  float* xiRowPtr = data + (index * dataPitch) + threadIdx.x;
  for(int currentDim = threadIdx.x; currentDim < dimension; currentDim += blockDim.x) {
    destination[currentDim] = *xiRowPtr;
    xiRowPtr += blockDim.x;
  }
}

}

#endif
