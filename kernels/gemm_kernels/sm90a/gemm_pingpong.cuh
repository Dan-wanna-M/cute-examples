#pragma once
#include "cute/algorithm/tuple_algorithms.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/arch/copy_sm90_desc.hpp"
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/config.hpp"
#include "cute/layout.hpp"
#include "cute/layout_composed.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cstdint>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
namespace test {
// Assume aligned block size
template <typename Config> struct Gemm {
  constexpr static int32_t cluster_size =
      2; // Assume 2 CTAs in a cluster(it does not work well with more than 2
         // CTAs anyway)
  constexpr static int32_t TMA_multicast_num = cluster_size;
  constexpr static int32_t warpgroup_size = 128;         // SM90 warpgroup size
  constexpr static int32_t consumer_warpgroup_count = 2; // Ping-pong
  constexpr static int32_t producer_warpgroup_count = 1; // TMA
  constexpr static uint32_t ProducerRegisters = 40;
  constexpr static uint32_t ConsumerRegisters = 232;
  constexpr static int32_t threads_per_block =
      warpgroup_size * consumer_warpgroup_count +
      producer_warpgroup_count * warpgroup_size; // 384 threads per block
  // Adapted from https://github.com/rchardx/cuda-gemm/
  struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;
    uint32_t num_aligned_n_blocks;
    uint32_t num_blocks;

    __device__ explicit Scheduler(
        cute::Shape<int32_t, int32_t, int32_t> shape_MNK) {
      constexpr auto cta_tiler = typename Config::CtaTiler{};
      constexpr uint32_t blockM = cute::size<0>(cta_tiler);
      constexpr uint32_t blockN = cute::size<1>(cta_tiler);
      num_aligned_m_blocks = cute::size<0>(shape_MNK) / blockM;
      num_aligned_n_blocks = cute::size<1>(shape_MNK) / blockN;
      num_blocks = num_aligned_m_blocks * num_aligned_n_blocks;
    }

    __device__ void get_swizzled_block_idx(int block_idx, uint32_t &m_block_idx,
                                           uint32_t &n_block_idx) {
      const auto num_blocks_per_group =
          num_aligned_n_blocks * Config::num_blocks_per_group;
      const auto group_idx = block_idx / num_blocks_per_group;
      const auto in_group_idx = block_idx % num_blocks_per_group;

      const auto first_m_block_idx = group_idx * Config::num_blocks_per_group;
      const auto num_m_blocks_in_group =
          min(Config::num_blocks_per_group,
              num_aligned_m_blocks - first_m_block_idx);
      m_block_idx = first_m_block_idx + in_group_idx % num_m_blocks_in_group;
      n_block_idx = in_group_idx / num_m_blocks_in_group;
    }

    __device__ __forceinline__ uint32_t
    get_peer_in_pair_idx(const uint32_t idx) {
      return idx ^ 1;
    }

    __device__ bool get_next_block(uint32_t &m_block_idx,
                                   uint32_t &n_block_idx) {
      const auto next_block_idx = (++current_iter) * gridDim.x + blockIdx.x;
      if (next_block_idx >= num_blocks)
        return false;
      get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
      return true;
    }
  };
  struct SharedStorage {
    alignas(1024) cute::ArrayEngine<
        typename Config::TA,
        cute::cosize_v<typename Config::SmemLayoutA>> A; // (M/2,K,P)
    // A1 and A2 are used by two consumer warpgroups
    alignas(1024) cute::ArrayEngine<
        typename Config::TB,
        cute::cosize_v<typename Config::SmemLayoutB>> B; // (N,K,P)
    alignas(1024) cute::ArrayEngine<
        typename Config::TC,
        cute::cosize_v<typename Config::SmemLayoutC>> C1; // (M,N,P)
    alignas(1024) cute::ArrayEngine<
        typename Config::TC,
        cute::cosize_v<typename Config::SmemLayoutC>> C2; // (M,N,P)
    // C1 and C2 are used by two consumer warpgroups
    // pipeline sized barriers
    uint64_t tma_barrier[cute::size<2>(typename Config::SmemLayoutA{})];
    uint64_t mma_barrier[cute::size<2>(typename Config::SmemLayoutA{})];
    uint64_t mma_ordering_barrier;
  };
  enum class WarpgroupRole {
    Producer = 0,
    Consumer1 = 1,
    Consumer2 = 2,
  };
  enum class ProducerWarpRole { Reader = 0, Writer = 1 };
  static __device__ __forceinline__ bool
  is_reader_thread(const uint32_t warp_idx, const uint32_t lane_predicate) {
    return warp_idx == static_cast<uint32_t>(ProducerWarpRole::Reader) &&
           lane_predicate;
  }
  static __device__ __forceinline__ bool
  is_writer_thread(const uint32_t warp_idx, const uint32_t lane_predicate) {
    return warp_idx == static_cast<uint32_t>(ProducerWarpRole::Writer) &&
           lane_predicate;
  }
  template <typename TmaA, typename TmaB, typename TmaC>
  __device__ __forceinline__ void
  operator()(cute::Shape<int32_t, int32_t, int32_t> shape_MNK,
             TmaA const &tma_a, TmaB const &tma_b, TmaC const &tma_c,
             void *shared_memory) const {
    constexpr auto cta_tiler = typename Config::CtaTiler{};
    constexpr auto mma = typename Config::TiledMma{};
    auto tile_scheduler = Scheduler{shape_MNK};
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = cutlass::canonical_warp_group_idx();
    // Preconditions
    CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{}); // (M, N, K)
    CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) ==
                         cute::Int<3>{}); // (BLK_M, BLK_N, BLK_K)
    static_assert(cute::is_static<typename Config::SmemLayoutA>::value);
    static_assert(cute::is_static<typename Config::SmemLayoutB>::value);
    // Initialize persistent states
    auto [M, N, K] = shape_MNK;
    constexpr int32_t blockM = cute::size<0>(cta_tiler);
    constexpr int32_t blockN = cute::size<1>(cta_tiler);
    constexpr int32_t blockK = cute::size<2>(cta_tiler);
    cute::Tensor gmem_A = tma_a.get_tma_tensor(cute::make_shape(M, K));
    cute::Tensor gmem_B = tma_b.get_tma_tensor(cute::make_shape(N, K));
    cute::Tensor gmem_C = tma_c.get_tma_tensor(cute::make_shape(M, N));
    constexpr auto pipe_length = cute::size<2>(typename Config::SmemLayoutA{});
    SharedStorage &smem = *reinterpret_cast<SharedStorage *>(shared_memory);
    cute::Tensor smem_A_tile = cute::make_tensor(
        cute::make_smem_ptr(smem.A.begin()),
        typename Config::SmemLayoutA{}); // (BLK_M/2,BLK_K,PIPE)
    // smem_A1_tile and smem_A2_tile are used
    cute::Tensor smem_B_tile =
        cute::make_tensor(cute::make_smem_ptr(smem.B.begin()),
                          typename Config::SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    cute::Tensor smem_C1_tile =
        cute::make_tensor(cute::make_smem_ptr(smem.C1.begin()),
                          typename Config::SmemLayoutC{}); // (BLK_M/2,BLK_N)
    cute::Tensor smem_C2_tile =
        cute::make_tensor(cute::make_smem_ptr(smem.C2.begin()),
                          typename Config::SmemLayoutC{}); // (BLK_M/2,BLK_N)
    // Initialize Barriers
    const auto lane_predicate = cute::elect_one_sync();
    uint64_t *producer_mbar = smem.tma_barrier;
    uint64_t *consumer_mbar = smem.mma_barrier;
    uint64_t *consumer_ordering_mbar =
        &smem.mma_ordering_barrier; // for ordering
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA
    if (is_reader_thread(warp_idx, lane_predicate)) {
      cutlass::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
      cutlass::prefetch_tma_descriptor(tma_b.get_tma_descriptor());
      cutlass::prefetch_tma_descriptor(tma_c.get_tma_descriptor());
      CUTE_UNROLL
      for (int32_t pipe = 0; pipe < pipe_length; ++pipe) {
        ProducerBarType::init(&producer_mbar[pipe], 1);
        ConsumerBarType::init(&consumer_mbar[pipe],
                              warpgroup_size * cluster_size);
      }
      // Initialize ordering barrier
      ConsumerBarType::init(consumer_ordering_mbar, warpgroup_size);
      // Make sure barriers on shared memory are visible to TMA
      cutlass::arch::fence_view_async_shared();
      // Make sure barriers on shared memory are visible to another CTA
      cutlass::arch::fence_barrier_init();
    }
    // Wait for all CTAs to initialize barriers
    cute::cluster_sync();
    uint32_t m_block_idx = 0;
    uint32_t n_block_idx = 0;
    auto producer_state = cutlass::PipelineState<pipe_length>();
    auto consumer_state = cutlass::PipelineState<pipe_length>(0, 1, 0);
    auto this_consumer_warpgroup =
        warpgroup_idx == static_cast<uint32_t>(WarpgroupRole::Consumer1);
    auto ordering_phase = 0;
    const uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    while (tile_scheduler.get_next_block(m_block_idx, n_block_idx)) {
      auto cta_coord = cute::make_coord(m_block_idx, n_block_idx,
                                        cute::_); // (m,n,k)
      cute::Tensor gmem_A_tile = cute::local_tile(
          gmem_A, cta_tiler, cta_coord,
          cute::Step<cute::_1, cute::X, cute::_1>{}); // (BLK_M,BLK_K,k)
      cute::Tensor gmem_B_tile = cute::local_tile(
          gmem_B, cta_tiler, cta_coord,
          cute::Step<cute::X, cute::_1, cute::_1>{}); // (BLK_N,BLK_K,k)
      constexpr uint32_t A_transaction_bytes =
          sizeof(Config::TA) * blockM * blockK;
      constexpr uint32_t B_transaction_bytes =
          sizeof(Config::TB) * blockN * blockK;
      constexpr uint32_t transaction_bytes =
          A_transaction_bytes + B_transaction_bytes;
      // For the tile being multicasted, each CTA only issues the TMA to copy
      // half of the tile. The other tile is distinct for each CTA in the
      // cluster so each CTA must issue the TMA to copy its own, full tile.
      static_assert(!Config::kIsTMAMulticastOnA);
      const uint16_t tma_mcast_mask = 0b11;
      auto tma_a_per_cta = tma_a.get_slice(cta_rank_in_cluster);
      auto tma_b_per_cta = tma_b.get_slice(cta_rank_in_cluster);
      if (warpgroup_idx != static_cast<int>(WarpgroupRole::Producer)) {
        cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();
        auto tma_c_per_cta = tma_c.get_slice(cta_rank_in_cluster);
        auto thread_id_within_warpgroup =
            threadIdx.x - warpgroup_idx * warpgroup_size; // [0,warpgroup_size)
        // This is actually "get warpgroup slice"
        cute::ThrMMA thr_mma = mma.get_thread_slice(thread_id_within_warpgroup);
        cute::Tensor smem_A_per_thread =
            thr_mma.partition_A(smem_A_tile); // (MMA,MMA_M,MMA_K,PIPE)
        cute::Tensor smem_B_per_thread =
            thr_mma.partition_B(smem_B_tile); // (MMA,MMA_N,MMA_K,PIPE)
        decltype(smem_C1_tile) smem_C_tile;
        // Each consumer warpgroup has its own C tile
        if (warpgroup_idx == static_cast<uint32_t>(WarpgroupRole::Consumer1)) {
          smem_C_tile = smem_C1_tile;
        } else {
          smem_C_tile = smem_C2_tile;
        }
        // Allocate accumulators and clear them
        cute::Tensor rmem_C_per_thread =
            thr_mma.partition_fragment_C(smem_C_tile); // (MMA,MMA_M,MMA_N)
        cute::clear(rmem_C_per_thread);
        // Allocate "fragments"
        cute::Tensor descriptor_A_per_thread = thr_mma.make_fragment_A(
            smem_A_per_thread); // (MMA,MMA_M,MMA_K,PIPE)
        cute::Tensor descriptor_B_per_thread = thr_mma.make_fragment_B(
            smem_B_per_thread); // (MMA,MMA_N,MMA_K,PIPE)
        // Wait until current consumer finishes its MMA
        if (!this_consumer_warpgroup) {
          ConsumerBarType::wait(consumer_ordering_mbar, ordering_phase);
          ordering_phase ^= 1;
          this_consumer_warpgroup = true;
          producer_state.advance(cute::size<2>(gmem_A_tile));
          if (!tile_scheduler.get_next_block(m_block_idx, n_block_idx)) {
            ConsumerBarType::arrive(consumer_ordering_mbar);
            return;
          }
          cta_coord = cute::make_coord(
              m_block_idx, n_block_idx,
              cute::_); // (m,n,k) - new cta coord for the next tile
        }
        CUTE_NO_UNROLL
        for (int32_t computed_k_tile_idx = 0;
             computed_k_tile_idx < cute::size<2>(gmem_A_tile);
             ++computed_k_tile_idx) {
          // Wait for Producer to complete
          int pipe = producer_state.index();
          ProducerBarType::wait(&producer_mbar[pipe], producer_state.phase());
          // MMAs to cover 1 K_TILE
          cute::warpgroup_arrive();
          cute::gemm(mma,
                     descriptor_A_per_thread(cute::_, cute::_, cute::_, pipe),
                     descriptor_B_per_thread(cute::_, cute::_, cute::_, pipe),
                     rmem_C_per_thread); // (V,M) x (V,N) => (V,M,N)
          cute::warpgroup_commit_batch();
          // Wait for all MMAs in a K_TILE to complete
          cute::warpgroup_wait<0>();
          // Notify that consumption is done
          ConsumerBarType::arrive(&consumer_mbar[pipe]);
          ConsumerBarType::arrive(
              &consumer_mbar[pipe],
              tile_scheduler.get_peer_in_pair_idx(cta_rank_in_cluster), 1);
          ++producer_state;
        }
        // Signals that the current warpgroup has finished the MMA
        if (this_consumer_warpgroup) {
          ConsumerBarType::arrive(consumer_ordering_mbar);
          ordering_phase ^= 1;
          this_consumer_warpgroup = false;
        }
        cute::Tensor gmem_C_tile = cute::local_tile(
            gmem_C, cta_tiler, cta_coord,
            cute::Step<cute::_1, cute::_1, cute::X>{}); // (BLK_M,BLK_N,k)
        auto rmem_C_per_thread_x2 =
            cute::recast<cutlass::Array<float, 2>>(rmem_C_per_thread);
        auto downcasted_rmem_C = cute::make_tensor_like<typename Config::TC>(
            rmem_C_per_thread.layout());
        auto downcasted_rmem_Cx2 =
            cute::recast<cutlass::Array<typename Config::TC, 2>>(
                downcasted_rmem_C);
        cute::transform(rmem_C_per_thread_x2, downcasted_rmem_Cx2,
                        cutlass::NumericArrayConverter<typename Config::TC,
                                                       float, 2>::convert);
        auto tiled_copy = typename Config::R2SCopy{};
        auto thread_copy =
            tiled_copy.get_thread_slice(thread_id_within_warpgroup);
        auto rmem_C_per_thread_copy_view =
            thread_copy.retile_D(downcasted_rmem_C);
        auto smem_C_per_thread_copy_view = thread_copy.partition_D(smem_C_tile);
        // Wait last TMA store
        cute::tma_store_wait<0>();
        cutlass::arch::NamedBarrier::arrive_and_wait(warpgroup_size,
                                                     warpgroup_idx);
        cute::copy(tiled_copy, rmem_C_per_thread_copy_view,
                   smem_C_per_thread_copy_view);
        cutlass::tma_store_fence();
        cutlass::arch::NamedBarrier::arrive_and_wait(warpgroup_size,
                                                     warpgroup_idx);
        if (thread_id_within_warpgroup == 0) {
          cute::copy(tma_c, tma_c_per_cta.partition_D(smem_C_tile),
                     tma_c_per_cta.partition_D(gmem_C_tile));
          cute::tma_store_arrive();
        }
        __syncwarp();
      } else {
        cutlass::arch::warpgroup_reg_dealloc<ProducerRegisters>();
        if (is_reader_thread(warp_idx, lane_predicate)) {
          CUTE_NO_UNROLL
          for (int32_t loaded_k_tile_idx = 0;
               loaded_k_tile_idx < cute::size<2>(gmem_A_tile);
               ++loaded_k_tile_idx) {
            auto pipe = consumer_state.index();
            // Wait for the current warpgroup to finish
            ConsumerBarType::wait(&consumer_mbar[pipe], consumer_state.phase());
            // Arrive for current CTA
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe],
                                                  transaction_bytes);
            cute::copy(
                tma_a.with(producer_mbar[pipe], tma_mcast_mask),
                tma_a_per_cta.partition_S(
                    gmem_A_tile(cute::_, cute::_, loaded_k_tile_idx)),
                tma_a_per_cta.partition_D(smem_A_tile(cute::_, cute::_, pipe)));
            cute::copy(
                tma_b.with(producer_mbar[pipe], tma_mcast_mask),
                tma_b_per_cta.partition_S(
                    gmem_B_tile(cute::_, cute::_, loaded_k_tile_idx)),
                tma_b_per_cta.partition_D(smem_B_tile(cute::_, cute::_, pipe)));
            ++consumer_state;
          }
        }
      }
    }
  }
};
struct DefaultConfig {
  using bM = cute::Int<64>;
  using bN = cute::Int<256>;
  using bK = cute::Int<64>;
  using bP = cute::Int<4>;
  using TA = cute::bfloat16_t;      // Type of A tile
  using TB = cute::bfloat16_t;      // Type of B tile
  using TC = cute::bfloat16_t;      // Type of C tile
  using CtaTiler = cute::Shape<bM,  // BLK_M
                               bN,  // BLK_N
                               bK>; // BLK_K
  // for each warpgroup
  using SmemLayoutA = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
      cute::make_shape(bM{}, bK{}, bP{}),
      cute::Step<cute::_2, cute::_1, cute::_3>{}));
  using SmemLayoutB = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
      cute::make_shape(bN{}, bK{}, bP{}),
      cute::Step<cute::_2, cute::_1, cute::_3>{}));
  using SmemLayoutC = decltype(cute::tile_to_shape(
      cute::GMMA::Layout_K_SW128_Atom<cute::bfloat16_t>{},
      cute::make_shape(bM{}, bN{}),
      cute::Step<cute::_2, cute::_1>{})); // no pipe
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::MMA_Atom<
          cute::MMA_Traits<cute::SM90::GMMA::MMA_64x256x16_F32BF16BF16_SS<
              cute::GMMA::Major::K, cute::GMMA::Major::K>>>{}));
  using R2SCopyAtom =
      cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cute::bfloat16_t>;
  using R2SCopy = decltype(cute::make_tiled_copy_C(R2SCopyAtom{}, TiledMma{}));
  static constexpr uint32_t num_blocks_per_group = 16;
  static constexpr bool kIsTMAMulticastOnA = false;
};
template <typename Config, typename TmaA, typename TmaB, typename TmaC>
__launch_bounds__(test::Gemm<Config>::threads_per_block, 1) __global__
    void gemm_kernel(cute::Shape<int32_t, int32_t, int32_t> shape_MNK,
                     CUTLASS_GRID_CONSTANT TmaA const tma_a,
                     CUTLASS_GRID_CONSTANT TmaB const tma_b,
                     CUTLASS_GRID_CONSTANT TmaC const tma_c) {
  extern __shared__ char shared_memory[];
  const test::Gemm<Config> gemm{};
  gemm(shape_MNK, tma_a, tma_b, tma_c,
       shared_memory); // (M,N,K)
}
}; // namespace test