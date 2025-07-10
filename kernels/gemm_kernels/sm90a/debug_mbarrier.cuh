#pragma once
#include "cute/arch/cluster_sm90.hpp"
#include "cute/config.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

template <int32_t pipeline_length> struct TmaLoadPipeline {
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA
  uint64_t *producer_mbarriers;
  uint64_t *consumer_mbarriers;
  cutlass::PipelineState<pipeline_length> pipeline_state;
  // to make array happy
  CUTE_DEVICE TmaLoadPipeline() : producer_mbarriers{}, consumer_mbarriers{} {};
  CUTE_DEVICE
  TmaLoadPipeline(cute::array<uint64_t, pipeline_length> &producer_mbarriers,
                  cute::array<uint64_t, pipeline_length> &consumer_mbarriers,
                  bool is_producer_warpgroup)
      : producer_mbarriers(producer_mbarriers.data()),
        consumer_mbarriers(consumer_mbarriers.data()),
        pipeline_state(0, (uint32_t)is_producer_warpgroup, 0) {};
  CUTE_DEVICE
  TmaLoadPipeline(const TmaLoadPipeline &other)
      : producer_mbarriers(other.producer_mbarriers),
        consumer_mbarriers(other.consumer_mbarriers),
        pipeline_state(other.pipeline_state) {}
  // This function should be called by an appropriate leader thread in a CTA
  // and followed by appropriate fence and sync.
  CUTE_DEVICE void init_mbarriers(const uint32_t producer_thread_count,
                                  const uint32_t consumer_thread_count) {
    // Initialize the mbarriers
    CUTE_UNROLL
    for (int32_t pipe = 0; pipe < pipeline_length; ++pipe) {
      ProducerBarType::init(&producer_mbarriers[pipe], producer_thread_count);
      ConsumerBarType::init(&consumer_mbarriers[pipe], consumer_thread_count);
    }
  }

  CUTE_DEVICE int32_t index() const { return pipeline_state.index(); }

  // Wait for the consumer to release the buffer.
  CUTE_DEVICE void producer_acquire() {
    ConsumerBarType::wait(&consumer_mbarriers[pipeline_state.index()],
                          pipeline_state.phase());
  }
  CUTE_DEVICE bool producer_try_acquire() {
    return ConsumerBarType::try_wait(
        &consumer_mbarriers[pipeline_state.index()], pipeline_state.phase());
  }
  // Signal that the producer starts committing a transaction.
  CUTE_DEVICE void producer_commit_start(const uint32_t transaction_bytes) {
    ProducerBarType::arrive_and_expect_tx(
        &producer_mbarriers[pipeline_state.index()], transaction_bytes);
  }
  // Signal that the producer has finished committing a transaction.
  CUTE_DEVICE void producer_commit_end() { ++pipeline_state; }
  // Wait for the producer to release the buffer.
  CUTE_DEVICE void consumer_acquire() {
    ProducerBarType::wait(&producer_mbarriers[pipeline_state.index()],
                          pipeline_state.phase());
  }
  CUTE_DEVICE bool consumer_try_acquire() {
    return ProducerBarType::try_wait(
        &producer_mbarriers[pipeline_state.index()], pipeline_state.phase());
  }
  // Signal that the consumer committed a transaction for local cta.
  CUTE_DEVICE void consumer_cta_commit() {
    ConsumerBarType::arrive(&consumer_mbarriers[pipeline_state.index()]);
    ++pipeline_state;
  }
  // Signal that the consumer committed a transaction for both local and remote
  // cta.
  CUTE_DEVICE void consumer_cluster_commit() {
    const uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    ConsumerBarType::arrive(&consumer_mbarriers[pipeline_state.index()]);
    ConsumerBarType::arrive(&consumer_mbarriers[pipeline_state.index()],
                            cta_rank_in_cluster ^ 1, // peer cta id
                            1);
    ++pipeline_state;
  }
};
struct DebugSharedStorage {
  cute::array<uint64_t, 1> tma_barrier;
  cute::array<uint64_t, 1> mma_barrier;
  cute::array<uint64_t, 1> debug_p_barrier;
  cute::array<uint64_t, 1> debug_c_barrier;
};
__launch_bounds__(256, 1) __global__ void MWE(int32_t *ptr) {
  extern __shared__ char shared_memory[];
  DebugSharedStorage &smem =
      *reinterpret_cast<DebugSharedStorage *>(shared_memory);
  auto &producer_mbar = smem.tma_barrier;
  auto &consumer_mbar = smem.mma_barrier;
  auto &debug_p_barrier = smem.debug_p_barrier;
  auto &debug_c_barrier = smem.debug_c_barrier;
  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier; // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;            // MMA
  const auto warpgroup_idx = cutlass::canonical_warp_group_idx();
  if (threadIdx.x == 0) {
    for (int i = 0; i < (int)producer_mbar.size(); ++i) {
      ProducerBarType::init(&producer_mbar[i], 1);
      ConsumerBarType::init(&consumer_mbar[i], 1);
      ProducerBarType::init(&debug_p_barrier[i], 1);
      ConsumerBarType::init(&debug_c_barrier[i], 1);
    }
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::fence_barrier_init();
  }
  auto pipeline = TmaLoadPipeline<1>{smem.tma_barrier, smem.mma_barrier,
                                     warpgroup_idx == static_cast<int>(0)};
  auto debug_pipeline =
      TmaLoadPipeline<1>{smem.debug_p_barrier, smem.debug_c_barrier,
                         warpgroup_idx == static_cast<int>(0)};
  // Wait for all CTAs to initialize barriers
  cute::cluster_sync();
  if (warpgroup_idx == static_cast<int>(0)) {
    CUTE_NO_UNROLL
    for (int32_t loaded_k_tile_idx = 0; loaded_k_tile_idx < 100;
         ++loaded_k_tile_idx) {
      pipeline.producer_acquire();
      pipeline.producer_commit_start(0);
      pipeline.producer_commit_end();
      debug_pipeline.producer_acquire();
      debug_pipeline.producer_commit_start(0);
      debug_pipeline.producer_commit_end();
    }
  } else {
    CUTE_NO_UNROLL
    for (int32_t computed_k_tile_idx = 0; computed_k_tile_idx < 100;
         ++computed_k_tile_idx) {
      pipeline.consumer_acquire();
      pipeline.consumer_cluster_commit();
      debug_pipeline.consumer_acquire();
      debug_pipeline.consumer_cluster_commit();
    }
  }
  if (threadIdx.x == 0) {
    *ptr = 1;
  }
}