#include "hello_world/hello_world.h"

int testmainCPU()
{
  char a[N] = "Hello ";
  int b[N] = {15, 10, 6, 0, -11, 1, 0};
  printf("%s", a);
  for ( int i = 0 ; i < 6 ; i++)
  {
    //a[i] += b[i];
    a[i] = sharedFunction(a[i], b[i]);
  }

  printf("%s from CPU\n", a);
}

__CUDA_HOST_DEVICE__
char sharedFunction(char a, int b)
{
  return a+b;
}
