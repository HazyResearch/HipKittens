/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<GROUP_THREADS, axis, assume_aligned>(dst, src, idx);
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>> // default case
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<GROUP_THREADS>(dst, src, idx);
}
