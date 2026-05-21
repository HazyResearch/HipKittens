/**
 * @file
 * @brief Workgroup-cluster primitives for gfx1250 (CGA).
 *
 * gfx1250 supports CUDA-Graph-style workgroup clusters (CGA) where up to 5
 * workgroups dispatched together can share a cluster-wide split barrier and
 * coalesce GL1 line requests via multicast.
 *
 * The runtime side (HIP CGA launch API) is still landing; in the meantime
 * this header provides the **device-side** primitives that take a `M0`
 * multicast mask and route through the same async-load builtins. Outside a
 * cluster (`workgroup_mask == 0`) the multicast-aware load reduces to a
 * non-multicast `cluster_load_async_to_lds_*`, so kernels can be authored
 * once and run in either mode.
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "../../../common/common.cuh"
#include "../sync/barrier.cuh"

namespace kittens {
namespace cluster {

/**
 * @brief Build the `M0` mask for a cluster multicast load.
 *
 * @param wg_bits        16-bit mask, bit `i` set ⇒ deliver result to WG `i` of the cluster.
 * @param early_timeout  If true, set bit 16 -- the load returns to whichever waves
 *                       have already joined as soon as GL2 returns; late joiners
 *                       issue a follow-up transaction. Useful when a few stragglers
 *                       would otherwise stall fast WGs.
 *
 * @return The `M0` value to pass as the `mask` argument of
 *         `kittens::tdm::cluster::load_async` / `kittens::async::cluster::load`.
 *         Note: the TDM path consumes only the low 16 bits; the `early_timeout`
 *         flag at bit 16 is honored by the async (per-lane) path but dropped by
 *         the TDM (descriptor-driven) path.
 */
__device__ __host__ __forceinline__ constexpr uint32_t mask(
    uint16_t wg_bits,
    bool early_timeout = false)
{
    return static_cast<uint32_t>(wg_bits) | (static_cast<uint32_t>(early_timeout) << 16);
}

/**
 * @brief Cluster-wide split barrier.
 *
 * Outside a CGA cluster this lowers to a workgroup-wide `sync::sync()`. Inside
 * a cluster the same `s_barrier_signal -1 / s_barrier_wait -1` pair extends to
 * every workgroup in the cluster by hardware-managed forwarding.
 */
__device__ __forceinline__ void sync() {
    ::kittens::sync::sync();
}

} // namespace cluster
} // namespace kittens

#endif // KITTENS_UDNA1
