/**
 * @file
 * @brief Workgroup-cluster primitives for gfx1250.
 *
 * The HIP compute hierarchy is:
 * **Grid -> Cluster -> Workgroup -> Wave -> Thread**.
 *
 * The on-chip cache hierarchy visible to the shader is two levels:
 * **L1 (per-WGP) -> L2 (chip-wide)**.
 *
 * gfx1250 supports CUDA thread block clusters (known as workgroup clusters)
 * where workgroups dispatched together can share a cluster-wide split barrier
 * and use multicast loads. When multiple workgroups in a cluster request the
 * same line, the fabric coalesces their requests and a single L2 return
 * broadcasts to up to 5 workgroups in one cycle. The multicast loads
 * force-miss the L1, so plan locality assuming no L1 hit on those lines.
 *
 * The runtime side (HIP launch API) is still landing; in the meantime
 * this header provides the **device-side** primitives that take a `M0`
 * multicast mask and route through the same async-load builtins. Outside a
 * cluster (`workgroup_mask == 0`) the multicast-aware load reduces to a
 * non-multicast `cluster_load_async_to_lds_*`, so kernels can be authored
 * once and run in either mode.
 */

#pragma once


#include "../../../common/common.cuh"
#include "../sync/barrier.cuh"

namespace kittens {
namespace cluster {

/**
 * @brief Build the `M0` mask for a cluster multicast load.
 *
 * @param wg_bits        16-bit mask, bit `i` set ⇒ deliver result to WG `i` of the cluster.
 * @param early_timeout  If true, set bit 21 -- the load returns to whichever requesters
 *                       have already joined as soon as the L2 returns; late joiners
 *                       issue a follow-up transaction. Useful when a few stragglers
 *                       would otherwise stall fast workgroups.
 *
 * @return The `M0` value to pass as the `cluster_mask` argument of
 *         `kittens::load_async`/`kittens::tdm::load_async`.
 */
static constexpr uint32_t EARLY_TIMEOUT_BIT = 1u << 21;
__device__ __host__ __forceinline__ constexpr uint32_t mask(
    uint16_t wg_bits,
    bool early_timeout = false)
{
    return static_cast<uint32_t>(wg_bits)
         | (early_timeout ? EARLY_TIMEOUT_BIT : 0u);
}

/**
 * @brief Signal the cluster-wide split barrier.
 *
 */
__device__ __forceinline__ void arrive() {
    __builtin_amdgcn_s_barrier_signal(-3);
}

/**
 * @brief Wait on the cluster-wide split barrier.
 *
 */
__device__ __forceinline__ void wait() {
    __builtin_amdgcn_s_barrier_wait(-3);
}

/**
 * @brief Cluster-wide barrier: workgroup rendezvous, one signal per workgroup, then both waits.
 *
 * The cluster barrier's member count is the number of workgroups in the cluster, and completion
 * compares it against a signal count that records how many signals arrived rather than which waves
 * sent them. Exactly one warp per workgroup may therefore signal it. If every warp
 * signalled, a single workgroup would satisfy the count on its own and release the cluster before
 * its peers arrived, and would do so without raising a fault. The wait is unconditional, since
 * completion is reported to every wave in the cluster.
 */
__device__ __forceinline__ void sync() {
    ::kittens::sync::arrive();
    if (::kittens::warpid() == 0) arrive();
    ::kittens::sync::wait();
    wait();
}

} // namespace cluster
} // namespace kittens

