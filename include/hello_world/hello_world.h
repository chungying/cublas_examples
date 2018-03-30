#pragma once
#if defined(CUDA_ENABLE)
#define __CUDA_DEVICE__ __device__
#define __CUDA_HOST_DEVICE__ __host__ __device__
#define __CUDA_GLOBAL__ __global__

#else
#define __CUDA_DEVICE__ 
#define __CUDA_HOST_DEVICE__ 
#define __CUDA_GLOBAL__ 

#endif

#include <stdio.h>

#if defined(CUDA_ENABLE)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

const int N = 7;
const int blocksize = 7;

__CUDA_HOST_DEVICE__ char sharedFunction(char a, int b);

int testmainCPU();

#if defined(CUDA_ENABLE)
int testmainCUDA();
__global__ void hello(char *a, int *b);
#endif
