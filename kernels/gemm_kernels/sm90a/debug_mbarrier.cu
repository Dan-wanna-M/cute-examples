#include "cute/util/debug.hpp"
#include "debug_mbarrier.cuh"
#include "cutlass/cluster_launch.hpp"
void debug_mbarrier() {
  // Launch parameter setup
  dim3 dimBlock(256);
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(132);
  int smemBytes = sizeof(DebugSharedStorage);
  auto *kernel_ptr = MWE;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
   int32_t *ptr = nullptr;
   cudaMalloc(&ptr, sizeof(int32_t));
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, ptr);
  CUTE_CHECK_LAST();
}