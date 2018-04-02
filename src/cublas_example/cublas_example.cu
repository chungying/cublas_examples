//Example 2. Application Using C and CUBLAS: 0-based indexing
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#define M 6
//#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

static __inline__ void modify (
  cublasHandle_t handle,//status
  double *m,//device array pointer 
  int ldm,//first dimension, 6
  int n,//second dimenstion, 5
  int p,//1 
  int q,//2
  double alpha,//scalar
  double beta)//scalar
{
  printf ("element_size: %d, scalar: %f, strid: %d\n", n-p, alpha, ldm);
  cublasDscal (handle, n-q, &alpha, m+IDX2C(p,q,ldm), ldm);
  cublasDscal (handle, ldm-p, &beta, m+IDX2C(p,q,ldm), 1);
  //cublasDdot(handle, ldm, m+IDX2C(0,0,ldm), 1, m+IDX2C(0,1,ldm), 1, m+IDX2C(0,2,ldm));
}

int main (void){
    int M = 6;
    int N = 5;
    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    double* devPtrA;
    double* a = 0;
    a = (double *)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            a[IDX2C(i,j,M)] = (double)(IDX2C(i,j,M));
            //printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        //printf ("\n");
    }
    printf("original a\n");
    for (i = 0 ; i < N*M ; i++)
    {
      printf("%7.0f", a[i]);
      if(i%M==M-1)
        printf("\n");
    }
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed\n");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    printf ("pointers: %d, %d, %d\n",IDX2C(0,0,M),IDX2C(0,1,M),IDX2C(0,2,M));
    printf ("pointers: %p, %p, %p\n",a+IDX2C(0,0,M),a+IDX2C(0,1,M),a+IDX2C(0,2,M));
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);

    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    printf("modified a\n");
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}
