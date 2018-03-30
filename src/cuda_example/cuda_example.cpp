#include "hello_world/hello_world.h"
int main()
{
#if defined(CUDA_ENABLE)
  testmainCUDA();
#else
  testmainCPU();
#endif 
  return 0;
}
