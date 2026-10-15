/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}

/* ---------- register staging ---------- */

// RDNA has no global->LDS DMA, so a prefetch has to sit in VGPRs between the
// global read and the LDS write. The warp-level pair is the only asynchronous
// path the architecture offers; a group needs it at least as much, since a
// whole-workgroup tile is what a GEMM actually prefetches. These just thread
// GROUP_THREADS through, so the two halves agree on the slot -> chunk mapping.

/// How many float4 slots a lane must reserve to stage one ST across the group.
template<ducks::st::all ST> static constexpr int stage_calls = kittens::stage_calls<ST, GROUP_THREADS>;

template<int axis=2, bool assume_aligned=false,
         ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_global_to_register_buffer(float4* reg_buffer, const int buffer_size,
                                                             const GL& src, const COORD& idx,
                                                             const ST& dst_template) {
    kittens::load_global_to_register_buffer<axis, assume_aligned, GROUP_THREADS, ST, GL, COORD>(
        reg_buffer, buffer_size, src, idx, dst_template);
}

template<bool wait = true, ducks::st::all ST>
__device__ static inline void store_register_buffer_to_shared(ST& dst, const float4* reg_buffer,
                                                              const int buffer_size = stage_calls<ST>) {
    kittens::store_register_buffer_to_shared<GROUP_THREADS, wait, ST>(dst, reg_buffer, buffer_size);
}