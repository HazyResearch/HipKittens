/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * Identical lane mapping to the shared-memory vector path next door; see the
 * comment there for the three layouts. Global memory just replaces src.data[]
 * with a raw pointer, and lanes that map to the same entry coalesce instead of
 * broadcasting.
 */

 /**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = typename RV::dtype;
    using U  = typename GL::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    const U *src_ptr = (const U*)&src[(idx.template unit_coord<-1, 3>())];
    const int lane = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            #pragma unroll
            for(int e = 0; e < rv_align_held<T>(); e++) {
                const int p = rv_align_entry<T>(e, lane);
                const T val = base_types::convertor<T, U>::convert(src_ptr[w*kittens::TILE_ROW_DIM<T> + p]);
                if(e & 1) dst.data[w][e>>1].y = val;
                else      dst.data[w][e>>1].x = val;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            dst[w][0] = base_types::convertor<T, U>::convert(
                src_ptr[w*kittens::TILE_ROW_DIM<T> + (lane & 15)]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            const int i = w*WARP_THREADS + lane;
            if(i < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[i]);
            }
        }
    }
}

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = typename RV::dtype;
    using U  = typename GL::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    const int lane = ::kittens::laneid();

    // As in the shared path, replicated layouts have several lanes writing the
    // same entry with the same value.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            #pragma unroll
            for(int e = 0; e < rv_align_held<T>(); e++) {
                const int p = rv_align_entry<T>(e, lane);
                const T val = (e & 1) ? src.data[w][e>>1].y : src.data[w][e>>1].x;
                dst_ptr[w*kittens::TILE_ROW_DIM<T> + p] = base_types::convertor<U, T>::convert(val);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            dst_ptr[w*kittens::TILE_ROW_DIM<T> + (lane & 15)] =
                base_types::convertor<U, T>::convert(src[w][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            const int i = w*WARP_THREADS + lane;
            if(i < src.length) {
                dst_ptr[i] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}

/**
 * @brief Store *one* element of a register vector to one element of a global.
 *
 * The folded-reduction counterpart to store() above, and only correct when
 * every entry of `src` holds the same value -- which is what
 * lang.collective.fold_rows produces and what hk's RegVecType.uniform tracks.
 * Writing a single entry of a vector whose entries differ picks whichever one
 * happens to live in lane 0 and is wrong on the other fifteen.
 *
 * Why it exists: a kernel that has folded a row onto a tile's sixteen rows has
 * one workgroup per row of the real tensor, so its per-row output -- a
 * quantization scale, say -- is a single number with a single slot, not a
 * sixteen-wide run. store() would write sixteen consecutive rows' worth.
 *
 * The address is computed exactly as store() computes it, so this writes the
 * first of the elements store() would have written. Lane 0 of every warp
 * writes; a multi-warp workgroup therefore issues several identical stores to
 * the same address, which is the same benign duplicate a backed-up block
 * already relies on and cheaper than branching the workgroup on warp 0.
 */
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store_scalar(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = typename RV::dtype;
    using U  = typename GL::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    if(::kittens::laneid() != 0) return;
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    // data[0][0] under any of the three layouts is an entry this lane really
    // holds; which one it is differs per layout and does not matter here,
    // because the precondition is that they are all equal. For align_l the
    // storage is packed, so the cast takes the low half of the pair.
    *dst_ptr = base_types::convertor<U, T>::convert(
        reinterpret_cast<const T*>(&src.data[0][0])[0]);
}

/**
 * @brief One element of a global, as a wave-uniform integer.
 *
 * The read counterpart of store_scalar, and the primitive a paged kernel is
 * built on: `block_table[req][i]`, `seq_lens[req]`, `slot_mapping[t]` are all
 * "fetch an integer from memory and index with it". Without this the IR can
 * only index by block_idx and arithmetic on it, which cannot express a page
 * table.
 *
 * Every lane loads the same address, and the result goes through
 * `readfirstlane`. The load is not the point; declaring the *value* uniform
 * is, so that the address arithmetic built on it lands in SGPRs instead of
 * being carried per lane. A page index that is accidentally divergent costs a
 * VGPR for every value derived from it, which on this architecture is how a
 * kernel that fits becomes a kernel that spills.
 *
 * When the address is itself uniform -- a page table indexed by blockIdx --
 * the compiler goes one better and drops the vector load entirely: the ISA
 * for a kernel doing exactly that contains no `v_readfirstlane_b32` and one
 * `s_load_b32` into an SGPR. The builtin is a floor, not a cost.
 *
 * No bounds check: a page table is produced by the framework and read in the
 * kernel's innermost loop. The caller clamps the *index* (see how the
 * attention kernel backs up its last block) rather than paying for a branch
 * per element here.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int load_scalar(const GL &src, const COORD &idx) {
    using U = typename GL::dtype;
    const U *src_ptr = (const U*)&src[(idx.template unit_coord<-1, 3>())];
    return __builtin_amdgcn_readfirstlane((int)(*src_ptr));
}

/**
 * @brief Count this workgroup's arrival at the counter `cnt[idx]`, device-wide.
 *
 * Returns, in every thread, how many workgroups arrived before this one -- so
 * the workgroup that sees `expected - 1` is the last, and can finish a
 * reduction the others contributed to. That one also puts the counter back
 * to 0, so a counter allocated zeroed stays zeroed across launches and graph
 * replays without a memset.
 *
 * The fences are what make it a reduction primitive rather than a counter:
 * every global store any thread issued before the call is visible to the
 * last arrival once its call returns (release before the increment, acquire
 * after). It contains barriers, so every thread of the workgroup must reach it.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int wg_arrive(const GL &cnt, const COORD &idx, int expected) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "wg_arrive counts in an int32 tensor");
    __shared__ int arrived_before;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        int *p = (int*)&cnt[(idx.template unit_coord<-1, 3>())];
        const int old = atomicAdd(p, 1);
        if (old == expected - 1) atomicExch(p, 0);
        arrived_before = old;
    }
    __syncthreads();
    const int arrived = __builtin_amdgcn_readfirstlane(arrived_before);
    __threadfence();
    return arrived;
}
} // namespace kittens
