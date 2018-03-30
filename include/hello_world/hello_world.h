#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int N = 7;
const int blocksize = 7;

int testmain();
int testmain2();
__global__ void hello(char *a, int *b);
