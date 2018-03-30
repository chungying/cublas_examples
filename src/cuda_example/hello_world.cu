#include "hello_world/hello_world.h"

__global__
void hello(char *a, int *b)
{
  //a[threadIdx.x] += b[threadIdx.x];
  a[threadIdx.x] = sharedFunction(a[threadIdx.x], b[threadIdx.x]);

}

int testmainCUDA()
{
  char a[N] = "Hello ";
  int b[N] = {15, 10, 6, 0, -11, 1, 0}; 
  char *ad;
  int *bd;
  const int csize = N*sizeof(char);
  const int isize = N*sizeof(int);
  
  printf("%s", a);
  
  if ( cudaSuccess != cudaMalloc( (void**)&ad, csize ) )
    printf( "cannot allocate device memory to ad\n");
  if ( cudaSuccess != cudaMalloc( (void**)&bd, isize ) )
    printf( "cannot allocate device memory to bd\n");
  if ( cudaSuccess != cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ) )
    printf( "cannot copy memory to device\n");
  if ( cudaSuccess != cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ) )
    printf( "cannot copy memory to device\n");
  
  dim3 dimBlock( blocksize, 1 );
  dim3 dimGrid( 1, 1 );
  hello<<<dimGrid, dimBlock>>>(ad, bd);
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    fprintf( stderr, "cudaCheckError() %d:%s\n", err, cudaGetErrorString( err ) );
    exit( -1 );
  }
  if ( cudaSuccess != cudaGetLastError() )
        printf( "Kernel Error!\n" );
  cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
  cudaFree( ad );
  
  printf("%s from CUDA\n", a);
  return EXIT_SUCCESS;
}
