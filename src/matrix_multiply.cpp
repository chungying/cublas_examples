#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

static __inline__ void dotProdoct (
  cublasHandle_t handle,//status
  int array_size,
  float* x,
  float* y,
  float* result)
{
  cublasSdot(handle, array_size, x, 1, y, 1, result);
}

static void 
