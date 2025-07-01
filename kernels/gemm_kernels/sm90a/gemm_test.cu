#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/cluster_launch.hpp"
#include "gemm.cuh"
#include <math.h>
#include <torch/extension.h>
#include <torch/torch.h>
namespace test {
torch::Tensor gemm_tn_test(const torch::Tensor A, const torch::Tensor B, torch::Tensor C) {
  // Ensure the tensors are on the same device
  TORCH_CHECK(A.device() == B.device(),
              "All tensors must be on the same device");
  // Ensure the tensors are contiguous
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
              "Both tensors must be 2D (matrix) tensors");
  // Ensure the inner dimensions match
  TORCH_CHECK(A.size(1) == B.size(1),
              "Inner dimensions of A and B must match (A: %d, B: %d)",
              A.size(1), B.size(1));
  TORCH_CHECK(A.scalar_type() == torch::kBFloat16 &&
                  B.scalar_type() == torch::kBFloat16,
              "Both tensors must be of type bfloat16");
  int M = A.size(0);
  int N = B.size(0);
  int K = A.size(1);
  // Assert divisble blocks
  TORCH_CHECK(M % (2 * test::DefaultConfig::bM::value) == 0,
              "M(%d) must be divisible by 2*bM(%d) for multicasting", 2 * M,
              test::DefaultConfig::bM::value);
  TORCH_CHECK(N % test::DefaultConfig::bN::value == 0,
              "N(%d) must be divisible by bN(%d)", N,
              test::DefaultConfig::bN::value);
  TORCH_CHECK(K % test::DefaultConfig::bK::value == 0,
              "K(%d) must be divisible by bK(%d)", K,
              test::DefaultConfig::bK::value);
  auto shape_MNK =
      cute::Shape<int32_t, int32_t, int32_t>(A.size(0), B.size(0), A.size(1));
  auto A_stride = cute::make_stride((int)A.stride(0), (int)A.stride(1));
  auto B_stride = cute::make_stride((int)B.stride(0), (int)B.stride(1));
  auto C_stride = cute::make_stride((int)C.stride(0), (int)C.stride(1));
  cute::Tensor mA = cute::make_tensor(
      cute::make_gmem_ptr<cute::bfloat16_t>(A.const_data_ptr()),
      cute::make_shape(M, K), A_stride);
  cute::Tensor mB = cute::make_tensor(
      cute::make_gmem_ptr<cute::bfloat16_t>(B.const_data_ptr()),
      cute::make_shape(N, K), B_stride);
  cute::Tensor mC = cute::make_tensor(
      cute::make_gmem_ptr<cute::bfloat16_t>(C.mutable_data_ptr()),
      cute::make_shape(M, N), C_stride);
  auto warpgroup_tiler =
      cute::Shape<typename DefaultConfig::half_bM, typename DefaultConfig::bN,
                  typename DefaultConfig::bK>{};
  auto cluster_shape = cute::Shape<cute::_2, cute::_1, cute::_1>{};
  cute::TiledCopy tma_A = cute::make_tma_copy_A_sm90(
      cute::SM90_TMA_LOAD{}, mA,
      typename DefaultConfig::SmemLayoutA{}(cute::_, cute::_, 0),
      warpgroup_tiler, cluster_shape);
  cute::TiledCopy tma_B = cute::make_tma_copy_B_sm90(
      cute::SM90_TMA_LOAD_MULTICAST{}, mB,
      typename DefaultConfig::SmemLayoutB{}(cute::_, cute::_, 0),
      warpgroup_tiler, cluster_shape);
  cute::TiledCopy tma_C = cute::make_tma_copy_C_sm90(
      cute::SM90_TMA_STORE{}, mC, typename DefaultConfig::SmemLayoutC{},
      warpgroup_tiler);
  // Launch parameter setup
  dim3 dimBlock(test::Gemm<DefaultConfig>::threads_per_block);
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(std::min(132, (M / (2 * DefaultConfig::bM::value)) * N /
                                 DefaultConfig::bN::value));
  int smemBytes = sizeof(Gemm<DefaultConfig>::SharedStorage);
  auto *kernel_ptr = gemm_kernel<DefaultConfig, decltype(tma_A),
                                 decltype(tma_B), decltype(tma_C)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, shape_MNK, tma_A, tma_B, tma_C);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
  return C;
}

} // namespace test