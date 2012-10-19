#ifndef GPUSVM_DEVICESELECT
#define GPUSVM_DEVICESELECT

#include <cuda.h>
#include <cutil.h>
#include <stdio.h>

namespace gpusvm {

void chooseLargestGPU(bool verbose = true);

}

#endif
