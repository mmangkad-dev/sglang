/**
 * \file topk_v2.cuh
 * \brief TopK kernel for DeepSeek v4.
 * Adapted from
 * 1:
 *   https://github.com/vllm-project/vllm/blob/a8c6ee9b787d273916206a29b77feebadb80c368/csrc/persistent_topk.cuh
 * 2:
 *   https://github.com/flashinfer-ai/flashinfer/blob/c2b4db2b1a84448d802f0e6ac445243312bd6a4c/include/flashinfer/topk.cuh
 * DarkSharpness never took a detailed look at these 2 implementation, but his claude code did.
 * So we add credit to the reference implementations.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <iterator>

namespace sglang {

namespace impl = device::topk;
using impl::TopKProblem;

using Register2 = impl::TopKRegister<2>;  // <= 8192, register-resident, 1 read
using Register4 = impl::TopKRegister<4>;  // <= 16384, register-resident, 1 read
using Streaming = impl::TopKStreaming;
using Cluster = impl::TopKCluster<8>;

// Elements per vectorized load in the impls; the ragged window's misalignment is
// measured against it.
constexpr uint32_t kVecElems = Register2::kVecSize;
constexpr uint32_t kBlockSize = impl::TopKConfig::kBlockSize;
constexpr uint32_t kOccupancy = impl::TopKConfig::kOccupancy;
constexpr uint32_t kMaxTopK = impl::TopKConfig::kMaxTopK;
constexpr uint32_t kClusterSize = Cluster::kClusterSize;
constexpr uint32_t kReg2MaxSeqLen = Register2::kMaxSeqLen;  // 8192
constexpr uint32_t kReg4MaxSeqLen = Register4::kMaxSeqLen;  // 16384

#define TOPK_KERNEL __global__ __launch_bounds__(kBlockSize, kOccupancy)
#define CLUSTER_TOPK_KERNEL TOPK_KERNEL __cluster_dims__(1, kClusterSize, 1)

constexpr uint32_t kClusterFloor = 65536;
constexpr uint32_t kClusterMaxBatch = 512;
constexpr uint32_t kNumPersistentClusters = 15 * kOccupancy;

/// Metadata tensor rows (each 8 B / 2 int32). Row 0 is the global plan result;
/// rows 1..N are the (batch_id, seq_len) of items routed to the cluster pool.
struct alignas(8) GlobalMetadata {
  uint32_t cluster_threshold;
  uint32_t num_cluster_items;  // N = number of items routed to the cluster pool
};
struct alignas(8) PlanItem {
  uint32_t batch_id;
  uint32_t seq_len;
};
static_assert(sizeof(GlobalMetadata) == 2 * sizeof(int32_t) && sizeof(PlanItem) == sizeof(GlobalMetadata));

struct TopKLaunchParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;  // nullptr selects the ragged output transform
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;      // optional raw (pre-transform) indices output; nullptr if unused
  const PlanItem* __restrict__ metadata;  // [0]=GlobalMetadata, [1+i]=PlanItem
  // Extend contract only; all nullptr for the decode contract.
  const int32_t* __restrict__ row_starts;    // per-row start column inside the score row
  const int32_t* __restrict__ out_offsets;   // ragged output: added to each selected index
  const int32_t* __restrict__ row_to_batch;  // paged output: row -> page_table row
  int64_t score_stride;
  int64_t page_table_stride;
  uint32_t topk;
  uint32_t page_bits;
  uint32_t cluster_floor;  // seq_len > this routes to the cluster path (batch-aware, host-set)

  SGL_DEVICE const GlobalMetadata& global() const {
    return *reinterpret_cast<const GlobalMetadata*>(metadata);
  }
  SGL_DEVICE uint32_t cluster_threshold() const {
    return global().cluster_threshold;
  }
  SGL_DEVICE const PlanItem& item(uint32_t i) const {
    return metadata[1 + i];
  }
  SGL_DEVICE int32_t* get_output_ptr(uint32_t batch_id) const {
    return page_indices + batch_id * static_cast<int64_t>(topk);
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id, uint32_t seq_len) const {
    const auto k = static_cast<int64_t>(topk);
    // Ragged rows start at an arbitrary column, so the window base can sit off
    // the 16B boundary the vectorized loads need; `head` names how many leading
    // elements the impls must scalar-load to realign. The score row base is 16B
    // aligned (data_ptr alignment + score_stride % 4 == 0 check on the host), so
    // the offset alone decides the misalignment.
    const uint32_t row_start = row_starts != nullptr ? static_cast<uint32_t>(row_starts[batch_id]) : 0u;
    const uint32_t head = (kVecElems - (row_start % kVecElems)) % kVecElems;
    // Decode maps row -> page-table row identically; extend has many query rows
    // per request, so the mapping is supplied.
    const uint32_t pt_row = row_to_batch != nullptr ? static_cast<uint32_t>(row_to_batch[batch_id]) : batch_id;
    return TopKProblem{
        .in = scores + batch_id * score_stride + row_start,
        .out = page_indices + batch_id * k,
        .raw_out = raw_indices != nullptr ? raw_indices + batch_id * k : nullptr,
        .page_table = page_table != nullptr ? page_table + pt_row * page_table_stride : nullptr,
        .topk = topk,
        .seq_len = seq_len,
        .page_bits = page_bits,
        // A trivial row (seq_len <= topk) never enters the vectorized impls, so
        // its head is irrelevant; clamping keeps `seq_len - head` well-defined.
        .head = seq_len > head ? head : 0u,
        .out_offset = out_offsets != nullptr ? out_offsets[batch_id] : 0,
    };
  }
  SGL_DEVICE TopKProblem problem(uint32_t batch_id) const {
    return this->problem(batch_id, static_cast<uint32_t>(seq_lens[batch_id]));
  }
};

/**
 * \brief Persistent cluster kernel for the long items. It will handle long inputs.
 * The short items are handled by the separate topk_kernel.
 */
template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_persistent_cluster_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  __shared__ impl::MaxSmem<Cluster::Smem> smem;
  const uint32_t num_cluster_items = params.global().num_cluster_items;
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
#pragma unroll 1
  for (uint32_t w = blockIdx.x; w < num_cluster_items; w += kNumPersistentClusters) {
    const auto it = params.item(w);
    const auto problem = params.problem(it.batch_id, it.seq_len);
    Cluster::forward<false>(problem, &smem);
    __syncthreads();
  }
}

template <typename F>
SGL_DEVICE void for_each_item(uint32_t topk, const F& f) {
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
#pragma unroll
  for (uint32_t i = 0; i < kNumElems; ++i) {
    if (const auto tx = i * kBlockSize + threadIdx.x; tx < topk) {
      __builtin_assume(tx < kMaxTopK);
      f(tx, i);
    }
  }
}

/// Output-completeness invariant, relied on by callers to allocate the output
/// buffer uninitialized: BOTH writers below cover slot 0..topk-1 of every row.
/// `trivial_transform` writes each slot directly (-1 past seq_len), and
/// `problem_transform` -- the single exit of every non-trivial path, including the
/// cluster epilogue in topk_main_kernel<...,3> -- re-writes all of them from the
/// staged raw indices, which `handle_tie` has itself filled up to topk. No slot is
/// left at its incoming value, so no caller needs to pre-fill -1.
template <bool kPDL>
SGL_DEVICE void trivial_transform(const TopKProblem& problem) {
  device::PDLWaitPrimary<kPDL>();
  device::PDLTriggerSecondary<kPDL>();
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t) {
    problem.transform_output(tx, tx < problem.seq_len ? static_cast<int32_t>(tx) : -1);
  });
}

SGL_DEVICE void problem_transform(TopKProblem& problem, int32_t* output_ptr) {
  static_assert(kMaxTopK % kBlockSize == 0);
  constexpr uint32_t kNumElems = kMaxTopK / kBlockSize;
  int32_t source_index[kNumElems];
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { source_index[i] = problem.out[tx]; });
  problem.out = output_ptr;
  for_each_item(problem.topk, [&](uint32_t tx, uint32_t i) { problem.transform_output(tx, source_index[i]); });
}

/**
 * \brief Main kernel for the short items and epilogue of long items.
 * \tparam kPDL whether to use PDL to synchronize with the cluster kernel (if any)
 * \tparam kLevel:
 * - Level 0: max_seq_len <= 8192           -> trivial + register<2>
 * - Level 1: max_seq_len <= 16384          -> trivial + register<4>
 * - Level 2: max_seq_len <= cluster_floor  -> trivial + register<4> + streaming
 * - Level 3: max_seq_len > cluster_floor   -> + epilogue process of cluster path
 */
template <bool kPDL, int kLevel>
TOPK_KERNEL void topk_main_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
  __shared__ impl::MaxSmem<Register2::Smem, Register4::Smem, Streaming::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  constexpr bool kHandleCluster = (kLevel == 3);
  // non-trivial path: dispatch based on level and seq_len
  const auto cluster_threshold = kHandleCluster ? params.cluster_threshold() : kU32Max;
  if constexpr (kLevel == 0) {
    __builtin_assume(problem.seq_len <= kReg2MaxSeqLen);
    Register2::forward<kPDL>(problem, &smem);
  } else if constexpr (kLevel == 1) {
    __builtin_assume(problem.seq_len <= kReg4MaxSeqLen);
    Register4::forward<kPDL>(problem, &smem);  // max_seq_len <= 16384 guarantees seq <= 16384
  } else {
    static_assert(kLevel == 2 || kLevel == 3, "we only support level = 0,1,2,3 now");
    // if using cluster, we can delay the PDL wait
    constexpr bool kPDLEarly = kPDL && !kHandleCluster;
    constexpr bool kPDLFinal = kPDL && kHandleCluster;
    if (problem.seq_len <= kReg4MaxSeqLen) {
      Register4::forward<kPDLEarly>(problem, &smem);
    } else if (problem.seq_len <= cluster_threshold) {
      Streaming::forward<kPDLEarly>(problem, &smem);
    } else {  // cluster path do nothing here
      problem.out = params.get_output_ptr(blockIdx.x);
    }
    device::PDLWaitPrimary<kPDLFinal>();
  }

  // page-table transform pass (gathers kept out of the hot scatter loop),
  // then trigger the dependent kernel only after the full output is written.
  device::PDLTriggerSecondary<kPDL>();
  __syncthreads();
  problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

template <bool kPDL>
CLUSTER_TOPK_KERNEL void topk_small_batch_kernel(const __grid_constant__ TopKLaunchParams params) {
  device::enable_smem_spilling();
  auto problem = params.problem(blockIdx.x);
  __shared__ impl::MaxSmem<Streaming::Smem, Cluster::Smem> smem;
  if (problem.seq_len <= problem.topk) return trivial_transform<kPDL>(problem);
  __shared__ int32_t topk_indices[kMaxTopK];
  problem.out = topk_indices;

  // randomly elect one worker rank to avoid workload imbalance
  const auto worker_rank = blockIdx.x % kClusterSize;

  // for small batch, we will fuse in the cluster case
  if (problem.seq_len <= kReg4MaxSeqLen) {
    if (blockIdx.y != worker_rank) return;
    Register4::forward<kPDL>(problem, &smem);
    device::PDLWaitPrimary<kPDL>();
    __syncthreads();
  } else if (problem.seq_len <= params.cluster_floor) {
    if (blockIdx.y != worker_rank) return;
    Streaming::forward<kPDL>(problem, &smem);
    device::PDLWaitPrimary<kPDL>();
    __syncthreads();
  } else {
    auto cluster = cooperative_groups::this_cluster();
    problem.out = cluster.map_shared_rank(topk_indices, worker_rank);
    Cluster::forward<kPDL>(problem, &smem);  // write to peer's output shared memory
    device::PDLWaitPrimary<kPDL>();
    cluster.sync();
    if (blockIdx.y != worker_rank) return;
  }

  // Only the elected worker reaches here, and it mapped `topk_indices` to
  // itself, so `problem.out` is this block's own buffer. Stating that keeps the
  // shared::cluster address out of the load problem_transform issues -- which is
  // load-bearing, not an optimization: without it cicc segfaults on CUDA 13.1+
  // for sm_90a (issue #32830, previously worked around by copying `problem` in
  // #32910). Verified: dropping this line reproduces the crash on 13.1/13.2/13.3.
  __builtin_assume(problem.out == topk_indices);
  problem_transform(problem, params.get_output_ptr(blockIdx.x));
}

// --- Plan: choose cluster_threshold from the seq_len distribution -----------
__global__ __launch_bounds__(kBlockSize, 1) void topk_plan(
    const uint32_t* __restrict__ seq_lens,
    PlanItem* __restrict__ metadata,  // [0]=GlobalMetadata, [1+i]=PlanItem
    const uint32_t batch_size,
    const uint32_t static_cluster_threshold) {
  // Candidate (threshold T_j, cap_j) pairs, T strictly increasing. The plan lowers
  // cluster_threshold to T_j while #(items with seq_len > T_j) <= cap_j, so cap_j
  // bounds how many long items go to the persistent pool. The pool runs N items in
  // ceil(N / kNumPersistentClusters) waves; the longer the seq the more waves pay
  // off (streaming a single block over a long item is very slow), so cap_j is the
  // measured cluster-vs-streaming crossover (B200, occ2) and GROWS with T -- a flat
  // cap = pool size only fits the shortest (~98K, one-wave) bucket. (Plan is tunable.)
  struct Pair {
    uint32_t threshold;
    uint32_t max_batch_size;
  };
  constexpr Pair kCandidates[] = {
      {65536, 30},    // (65536,98304]:    ~1 pool wave, streams beyond 30
      {98304, 48},    // (98304,131072]
      {131072, 60},   // (131072,196608]
      {196608, 80},   // (196608,262144]
      {262144, 112},  // (262144,393216]
      {393216, 128},  // (393216,inf):     longest -- worth many pool waves; a top
                      // threshold here lets overloaded ~280-393K batches still stream
  };
  constexpr uint32_t kNumCandidates = std::size(kCandidates);
  static_assert(kCandidates[0].threshold == kClusterFloor);

  __shared__ uint32_t s_counts[kNumCandidates];
  __shared__ uint32_t s_threshold;
  __shared__ uint32_t s_count;

  const auto tx = threadIdx.x;
  if (tx < kNumCandidates) s_counts[tx] = 0;
  if (tx == 0) s_count = 0;
  __syncthreads();

  if (static_cluster_threshold > 0) {
    if (tx == 0) s_threshold = static_cluster_threshold;
  } else {
    for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
      const uint32_t sl = seq_lens[i];
      uint32_t count = 0;
#pragma unroll
      for (uint32_t j = 0; j < kNumCandidates; ++j) {
        count += (sl > kCandidates[j].threshold ? 1 : 0);
      }
      if (count > 0) atomicAdd(&s_counts[count - 1], 1);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t accum = 0;
      uint32_t chosen = kCandidates[kNumCandidates - 1].threshold;
#pragma unroll
      for (uint32_t i = 0; i < kNumCandidates; ++i) {
        const auto j = kNumCandidates - 1 - i;
        accum += s_counts[j];  // # items with seq_len > kCandidates[j].threshold
        if (accum > kCandidates[j].max_batch_size) break;
        chosen = kCandidates[j].threshold;
      }
      s_threshold = chosen;
    }
  }
  __syncthreads();
  const auto cluster_threshold = max(s_threshold, kClusterFloor);

  // Compact items with seq_len > threshold into metadata[1..N]: their batch ids
  // are the work list the persistent cluster pool fetches.
  for (uint32_t i = tx; i < batch_size; i += kBlockSize) {
    const uint32_t sl = seq_lens[i];
    if (sl > cluster_threshold) {
      const auto pos = atomicAdd(&s_count, 1);
      metadata[1 + pos] = {i, sl};
    }
  }
  __syncthreads();
  if (tx == 0) {
    auto* g = reinterpret_cast<GlobalMetadata*>(metadata);
    *g = {.cluster_threshold = cluster_threshold, .num_cluster_items = s_count};
  }
}

struct TopKKernel {
  static void plan(  //
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView metadata,
      const uint32_t static_cluster_threshold) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto device = device_.unwrap();
    LaunchKernel(1, kBlockSize, device)(  //
        topk_plan,
        static_cast<const uint32_t*>(seq_lens.data_ptr()),
        static_cast<PlanItem*>(metadata.data_ptr()),
        batch_size,
        static_cluster_threshold);
  }

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::TensorView metadata,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, -1})  // page_table
        .with_strides({P, 1})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_table);
    TensorMatcher({B, K})  // page_indices
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(page_indices);
    TensorMatcher({Bp1, 2})  // metadata: [0]=GlobalMetadata, [1..N]=PlanItem(batch_id, seq_len)
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(metadata);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device_).verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto max_seq_len = static_cast<uint32_t>(L.unwrap());
    const auto device = device_.unwrap();

    // The fused kernel runs one 8-block cluster per batch element, and B200 fits one
    // wave of exactly 15 such clusters (occ2). For batch <= 15 it stays latency-bound,
    // so the 8-way split beats streaming from a much lower seq (measured crossover
    // ~36-40K); batch 16 spills into a 2nd wave (+25%) and keeps the 64K floor.
    // The floor is chosen on the host per launch.
    constexpr uint32_t kClusterFloorSmall = 32768;
    constexpr uint32_t kSmallBatchLowFloor = 15;
    const auto params = TopKLaunchParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .row_starts = nullptr,
        .out_offsets = nullptr,
        .row_to_batch = nullptr,
        .score_stride = S.unwrap(),
        .page_table_stride = P.unwrap(),
        .topk = topk,
        .page_bits = page_bits,
        .cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor,
    };

    launch(params, batch_size, max_seq_len, device);
  }

  /// Extend (prefill / draft-extend) variant. Each row's window is
  /// `scores[b, row_starts[b] : row_starts[b] + seq_lens[b]]` -- unlike decode,
  /// the window does not start at column 0 and there are many query rows per
  /// request. The selected window-relative positions go through one of two output
  /// transforms, exactly one of which must be supplied:
  ///   * paged (`page_table` + `row_to_batch`): the same page-table transform as
  ///     `transform`, reading row `row_to_batch[b]` of the compact page table.
  ///     Replaces sgl_kernel's `fast_topk_transform_fused` prefill kernel, which
  ///     instead gathers a page_size=1 table.
  ///   * ragged (`out_offsets`): `position + out_offsets[b]`, an index into the
  ///     concatenated ragged KV buffer. Replaces
  ///     `fast_topk_transform_ragged_fused`.
  /// \param max_seq_len upper bound on ``seq_lens``, used to pick the level. Unlike
  /// decode -- where the score width IS the longest row -- an extend score matrix
  /// is as wide as the batch's concatenated KV, which can be far longer than any
  /// single row, so the caller supplies the real bound. It MUST be >= every entry
  /// of ``seq_lens``: too low picks a level whose fixed-unrolled loop cannot cover
  /// a row and the top-k comes back wrong.
  static void transform_extend(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView row_starts,
      const tvm::ffi::TensorView topk_indices,
      const tvm::ffi::TensorView metadata,
      const uint32_t page_size,
      const uint32_t max_seq_len,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::Optional<tvm::ffi::TensorView> row_to_batch,
      const tvm::ffi::Optional<tvm::ffi::TensorView> out_offsets) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto Bp1 = SymbolicSize{"batch_size_plus_1"};
    auto L = SymbolicSize{"max_seq_len"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, L})  // score
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device_)
        .verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(seq_lens);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_starts);
    TensorMatcher({B, K}).with_dtype<int32_t>().with_device(device_).verify(topk_indices);
    TensorMatcher({Bp1, 2}).with_dtype<int32_t>().with_device(device_).verify(metadata);

    RuntimeCheck(
        page_table.has_value() != out_offsets.has_value(),
        "transform_extend takes exactly one of page_table (paged) / out_offsets (ragged)");
    const int32_t* page_table_ptr = nullptr;
    const int32_t* row_to_batch_ptr = nullptr;
    const int32_t* out_offsets_ptr = nullptr;
    int64_t page_table_stride = 0;
    uint32_t page_bits = 0;
    if (page_table.has_value()) {
      TensorMatcher({-1, -1})
          .with_strides({P, 1})
          .with_dtype<int32_t>()
          .with_device(device_)
          .verify(page_table.value());
      RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
      RuntimeCheck(row_to_batch.has_value(), "paged transform_extend requires row_to_batch");
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(row_to_batch.value());
      page_table_ptr = static_cast<const int32_t*>(page_table.value().data_ptr());
      row_to_batch_ptr = static_cast<const int32_t*>(row_to_batch.value().data_ptr());
      page_table_stride = P.unwrap();
      page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    } else {
      RuntimeCheck(!row_to_batch.has_value(), "ragged transform_extend takes no row_to_batch");
      TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(out_offsets.value());
      out_offsets_ptr = static_cast<const int32_t*>(out_offsets.value().data_ptr());
    }

    RuntimeCheck(S.unwrap() % 4 == 0, "score_stride must be a multiple of 4 (16-byte vectorized load)");
    RuntimeCheck(Bp1.unwrap() == B.unwrap() + 1, "invalid metadata shape");
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    // A row may not read past the score matrix, so the bound cannot exceed its
    // width; that also catches a caller passing 0 (which would pick level 0 for
    // arbitrarily long rows).
    RuntimeCheck(
        max_seq_len > 0 && max_seq_len <= static_cast<uint32_t>(L.unwrap()),
        "max_seq_len must be in (0, scores.shape[1]]");
    const auto device = device_.unwrap();
    constexpr uint32_t kClusterFloorSmall = 32768;
    constexpr uint32_t kSmallBatchLowFloor = 15;
    const auto params = TopKLaunchParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .seq_lens = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .page_table = page_table_ptr,
        .page_indices = static_cast<int32_t*>(topk_indices.data_ptr()),
        .raw_indices = nullptr,
        .metadata = static_cast<const PlanItem*>(metadata.data_ptr()),
        .row_starts = static_cast<const int32_t*>(row_starts.data_ptr()),
        .out_offsets = out_offsets_ptr,
        .row_to_batch = row_to_batch_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = page_table_stride,
        .topk = topk,
        .page_bits = page_bits,
        .cluster_floor = (batch_size <= kSmallBatchLowFloor) ? kClusterFloorSmall : kClusterFloor,
    };
    launch(params, batch_size, max_seq_len, device);
  }

 private:
  static void launch(
      const TopKLaunchParams& params, uint32_t batch_size, uint32_t max_seq_len, DLDevice device) {
    using namespace host;
    const bool use_cluster = (max_seq_len > params.cluster_floor) && (batch_size <= kClusterMaxBatch);
    constexpr bool kUsePDL = true;
    if (use_cluster) {
      if (batch_size <= kNumPersistentClusters) {
        LaunchKernel({batch_size, kClusterSize}, kBlockSize, device)
            .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
            .launch(topk_small_batch_kernel<kUsePDL>, params);
      } else {
        const uint32_t num_clusters = std::min(batch_size, kNumPersistentClusters);
        LaunchKernel({num_clusters, kClusterSize}, kBlockSize, device)
            .config({.use_pdl = kUsePDL, .cluster_dim = dim3{1, kClusterSize}})
            .launch(topk_persistent_cluster_kernel<kUsePDL>, params);
        LaunchKernel(batch_size, kBlockSize, device)
            .config({.use_pdl = kUsePDL})
            .launch(topk_main_kernel<kUsePDL, /*kLevel=*/3>, params);
      }
    } else if (max_seq_len <= kReg2MaxSeqLen) {
      LaunchKernel(batch_size, kBlockSize, device)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/0>, params);
    } else if (max_seq_len <= kReg4MaxSeqLen) {
      LaunchKernel(batch_size, kBlockSize, device)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/1>, params);
    } else {
      LaunchKernel(batch_size, kBlockSize, device)
          .config({.use_pdl = kUsePDL})
          .launch(topk_main_kernel<kUsePDL, /*kLevel=*/2>, params);
    }
  }
};

}  // namespace sglang
