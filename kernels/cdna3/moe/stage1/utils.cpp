#include "kittens.cuh"

using namespace kittens;

constexpr int NT_LOAD_BITS = 0b10;

extern "C" __device__ inline float
llvm_amdgcn_raw_buffer_load_f32(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ inline void store_shared_f32(uint32_t lds_off, float val) {
    asm volatile(
        "ds_write_b32 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

/**
 * @brief Loads token IDs and corresponding Top-K slot from permuted+sorted token mapping array.
 *        Assumes that subsequent functions that actually load activations matrix also map lanes to 
 *        data in a simple row-major fashion (i.e. lanes vary fastest along columns) AND no lane is
 *        mapped to the same token more than once (i.e. memcpy_per_row <= N_THREADS)
 *
 * @tparam N_THREADS            The number of threads used.
 * @param dst[out]              Register buffer to hold token IDs.
 * @param sorted_token_ids[in]  Array mapping each permuted-space row index to its
 *                              corresponding bit-packed int, which encodes the token index in 
 *                              src. in the lower 24-bits, and its top-K slot in the upper 8.
 *                              Padding tokens are marked as M, as valid tokens range from [0, M - 1].
 * @param idx[in]               Coordinate of this block's assigned tile. N-coord. is ignored
 *                              since only M-coord is needed for tokens.
 * @param st[in]                Shared tile for activations matrix; only used for its shape to compute
 *                              thread-value mappings.
 */
template<int N_THREADS,
        ducks::gl::all GL,
        ducks::st::all ST,
        ducks::coord::tile COORD=coord<ST>
>
__device__ inline void gather_tokens(int* dst, const GL& sorted_token_ids, const COORD& idx, const ST& st) {
    using T = typename ST::dtype;
    constexpr int axis = 2;

    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    coord<> tm_coord(0, 0, 0, unit_coord.r);
    typename GL::dtype *tm_ptr = (typename GL::dtype*)&sorted_token_ids[tm_coord];
    const int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for (int i = 0; i < total_calls; ++i) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;

        int packed = tm_ptr[row];
        dst[i] = packed & 0x00FFFFFF;
    }
}

/**
 * @brief Gathers non-contiguous rows from a global tile into a shared tile selected by a mapping.
 *
 * @tparam N_THREADS  The number of threads used.
 * @param dst[out]    The destination shared tile.
 * @param src[in]     The source global tensor (unpermuted activations), shape [M, d_hidden].
 * @param idx[in]     Tile coordinate (m_tile, k_tile): m_tile is the M-tile index into the
 *                    PERMUTED row space; k_tile is the column-block index into src's actual K axis.
 * @param token_array[in]  Token IDs assigned to this lane
 */
template<int N_THREADS,
        ducks::st::all ST,
        ducks::gl::all GL,
        ducks::coord::tile COORD = coord<ST>
>
__device__ inline void gather_load(
    ST& dst, const GL& src, const COORD& idx, const int* token_array
) {
    using T = typename ST::dtype;
    constexpr int axis = 2;

    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    coord<> k_coord(0, 0, 0, unit_coord.c);

    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[k_coord];
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    float4    buf[total_calls];
    const int total_bytes = src.rows() * row_stride * sizeof(T);
    i32x4 srsrc = make_srsrc(src_ptr, total_bytes, row_stride_bytes);

    #pragma unroll
    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
        int token_id = token_array[i];
        int flat_offset = token_id * row_stride + col;
        int byte_offset = flat_offset * sizeof(T);
        if (row < ST::rows) {
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
            buf[i] = *reinterpret_cast<float4*>(&raw);
        }
    }
    #pragma unroll
    for (int i = 0; i < total_calls; i++) {
        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
        if (row < ST::rows) {
            store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf[i].x, buf[i].y});
            store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf[i].z, buf[i].w});
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
}

/**
 * @brief Gathers non-contiguous rows from a global tile into register buffers selected by a mapping.
 *
 * @tparam N_THREADS  The number of threads used.
 * @param reg_buffer[out]  The destination register buffers.
 * @param buffer_size[in]  Size of register buffers, in units of float4's (i.e. 128-bits)
 * @param src[in]   The source global tensor (unpermuted activations), shape [M, d_hidden].
 * @param idx[in]   Tile coordinate (m_tile, k_tile): m_tile is the M-tile index into the
 *                  PERMUTED row space; k_tile is the column-block index into src's actual K axis.
 * @param token_array[in]  Token IDs assigned to this lane
 */
template<int N_THREADS,
        ducks::st::all ST, 
        ducks::gl::all GL,
        ducks::coord::tile COORD = coord<ST>
>
__device__ inline void gather_load_global_to_register_buffer(
    float4* reg_buffer, const int buffer_size, const GL& src, const COORD& idx, const int* token_array, const ST& dst_template
) {
    using T = typename ST::dtype;
    constexpr int axis = 2;
    
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 4;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    coord<> k_coord(0, 0, 0, unit_coord.c);
    T* base_ptr = (T*)&src[k_coord];
    const int laneid = threadIdx.x % N_THREADS;

    const int total_bytes = row_stride * src.rows() * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    int buf_idx = 0;
    for (int i = 0; i < big_calls && buf_idx < buffer_size; ++i) {
        const int offset = i * small_calls;
        #pragma unroll
        for (int j = 0; j < small_calls; ++j) {
            const int chunk_idx = (offset + j) * N_THREADS + laneid;
            if (chunk_idx < total_chunks && buf_idx < buffer_size) {
                int row = chunk_idx / memcpy_per_row;
                int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
                int token_id = token_array[buf_idx];
                int flat_offset = token_id * row_stride + col;
                int byte_offset = flat_offset * sizeof(T);
                __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
                reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                buf_idx++;
            }
        }
    }
}

/**
 * @brief Near-exact copy from HK's global_to_shared; only diff. is we're using non-temporal loads here
 */
template<int axis=2, bool assume_aligned=false,
        int N_THREADS = WARP_THREADS,
        ducks::st::all ST, 
        ducks::gl::all GL,
        ducks::coord::tile COORD = coord<ST>
>
__device__ inline void load_global_to_register_buffer_nt(float4* reg_buffer, const int buffer_size, const GL& src, const COORD& idx, const ST& dst_template) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* base_ptr = (T*)&src[unit_coord];  // global memory pointer
    const int laneid = threadIdx.x % N_THREADS;

    // buffer resource
    const int total_bytes = row_stride * ST::rows * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    int buf_idx = 0;
    for (int i = 0; i < big_calls && buf_idx < buffer_size; ++i) {
        const int offset = i * small_calls;
        #pragma unroll
        for (int j = 0; j < small_calls; ++j) {
            const int chunk_idx = (offset + j) * N_THREADS + laneid;
            if (chunk_idx < total_chunks && buf_idx < buffer_size) {
                int row = chunk_idx / memcpy_per_row;
                int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
                int flat_offset = row * row_stride + col;
                int byte_offset = flat_offset * sizeof(T);
                __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, NT_LOAD_BITS);
                reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                buf_idx++;
            }
        }
    }
}

/**
 * @brief Gathers PT (per-token) f32 scale factors.
 *
 * @tparam N_THREADS  The number of threads used.
 * @param dst[out]    Destination shared vector.
 * @param src[in]     The source global tile.
 * @param idx[in]     Coord. of [m_tile_index]
 * @param sorted_token_ids[in]  Array mapping each permuted-space row index to its token + top-K slot
 */
template<int N_THREADS,
        ducks::sv::all SV, 
        ducks::gl::all GL,
        ducks::gl::all GL_IDX,
        ducks::coord::vec COORD=coord<SV>
>
__device__ inline void gather_f32_sf_a(
    SV& dst, const GL& src, const COORD& idx, const GL_IDX& sorted_token_ids
) {
    // in practice, BLOCK_M (i.e. dst.length) is probably less than N_THREADS, so total_calls = 1
    constexpr int total_calls = (SV::length + N_THREADS - 1) / N_THREADS;
    float buf[total_calls];
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    float* base_ptr = (float*)&src[0];  // index into dense scale factor vector only via token IDs routed to this block
    typename GL_IDX::dtype* tm_ptr = (typename GL_IDX::dtype*)&sorted_token_ids[unit_coord];
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    const int total_bytes = src.cols() * sizeof(float);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, sizeof(float));
    
    #pragma unroll
    for (int i = 0; i < total_calls; ++i) {
        if (laneid < SV::length) {  // one lane per row in BLOCK_M, mask out lanes that exceed this block's segment
            int token_id = tm_ptr[laneid + i * N_THREADS] & 0xFFFFFF;
            int byte_offset = token_id * sizeof(float);
            buf[i] = llvm_amdgcn_raw_buffer_load_f32(srsrc, byte_offset, 0, 0);
            // buf[i] = (token_id != src.cols()) ? base_ptr[token_id] : 1.0f;

            store_shared_f32(dst.idx(dst_ptr, laneid), buf[i]);
        }
    }
    asm volatile("s_waitcnt lgkmcnt(0)");
}

/**
 * @brief Load data from SV to RV.
 *        - For align layout RVs, each element broadcasted to lanes across columns of 16x16 MFMA accum. tile (i.e. lanes 0-15 share same 4 elements, 16-31 share, and so on.)
 *        - For ortho layout RVs, each element broadcasted to lanes across rows (i.e. lanes 0+16+32+48 share the same 1 element, and so on).
 * 
 * @param dst[out]  The destination register vector.
 * @param src[in]   The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load_sv_to_rv(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);
    
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            if(idx < dst.length) {
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src.data[idx]);
            }
        }
        #pragma unroll
        for (auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid / 16 * 2) + 8 * (w % 8);
            T2 tmp = dst[w][w % 2];
            dst[w][0] = packed_shfl(MASK_ALL, tmp, leader);
            dst[w][1] = packed_shfl(MASK_ALL, tmp, leader + 1);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim + 3) / 4; w++) {
            int idx = 64 * w + laneid;
            int o_dim = 4 * w + laneid / 16;
            if (idx < dst.length) {
                T tmp = base_types::convertor<T, U>::convert(src.data[idx]);
                dst[o_dim][0] = tmp;
            }
        }
        #pragma unroll
        for (auto w = 0; w < dst.outer_dim; w++) {
            int leader = laneid % 16 + 16 * w;
            dst[w][0] = packed_shfl(MASK_ALL, dst[w][0], leader);
        }
    }
}

/**
 * @brief Apply per-token (generally--per-row) scale factors to an accumulator tile for dequant.
 */
template<ducks::rt::col_layout T, ducks::rv::align_layout V>
__device__ static inline void apply_row_sf(T &dst, const T &src, const V &row_values) {
    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(std::is_same_v<RT2, typename T::dtype>);
    static_assert(V::outer_dim == T::height);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        RT2 packed0 = row_values[i][0];
        RT2 packed1 = row_values[i][1];
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            dst.tiles[i][j].data[0] = base_ops::mul::op(src.tiles[i][j].data[0], packed0);
            dst.tiles[i][j].data[1] = base_ops::mul::op(src.tiles[i][j].data[1], packed1);
        }
    }
}

/**
 * @brief Apply per-channel (generally--per-col.) scale factors to an accumulator tile for dequant.
 */
template<ducks::rt::col_layout T, ducks::rv::ortho_layout V>
__device__ static inline void apply_col_sf(T& dst, const T &src, const V &col_values) {
    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(std::is_same_v<RT2, typename T::dtype>);
    static_assert(V::outer_dim == T::width);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            RT unpacked = col_values[j][0];
            dst.tiles[i][j].data[0].x = base_ops::mul::op(src.tiles[i][j].data[0].x, unpacked);
            dst.tiles[i][j].data[0].y = base_ops::mul::op(src.tiles[i][j].data[0].y, unpacked);
            dst.tiles[i][j].data[1].x = base_ops::mul::op(src.tiles[i][j].data[1].x, unpacked);
            dst.tiles[i][j].data[1].y = base_ops::mul::op(src.tiles[i][j].data[1].y, unpacked);
        }
    }
}

/**
 * @brief Scatters a warp's accumulator tile to the routed output rows using
 *        non-temporal global stores
 *
 * @tparam TOP_K    Mixture-of-Experts Top-K parameter
 * @param dst[out]  Destination global tile
 * @param src[in]   Source register tile
 * @param idx[in]   Coord. of warp's register tile
 * @param sorted_token_ids[in]  Array mapping each permuted-space row index to its token + top-K slot
 */
template<int TOP_K,
        ducks::rt::all RT,
        ducks::gl::all GL,
        ducks::gl::all GL_IDX,
        ducks::coord::tile COORD=coord<RT>
>
__device__  inline void scatter_store(
    GL& dst, const RT& src, const COORD& idx, const GL_IDX& sorted_token_ids
) {
    using T = bf16;
    using U = float;
    constexpr int axis = 2;
    coord<> unit_coord = idx.template unit_coord<axis, 3>();

    coord<> tm_coord(0, 0, 0, unit_coord.r);
    coord<> n_coord(0, 0, 0, unit_coord.c);
    T* base_ptr = (T*)&dst[n_coord];  // block+warp col. offset only!
    typename GL_IDX::dtype* tm_ptr = (typename GL_IDX::dtype*)&sorted_token_ids[tm_coord];  // block+warp offset

    const int row_stride = dst.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    const int total_bytes = row_stride * dst.rows() * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    const int sentinel = dst.rows() / TOP_K;  // since dst.rows = M * TOP_K
    #pragma unroll
    for (int i = 0; i < RT::height; ++i) {
        #pragma unroll
        for (int j = 0; j < RT::width; ++j) {
            const float* flat = reinterpret_cast<const float*>(src.tiles[i][j].data);  // avoid dealing w/ named attributes
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int row_offset = (i * 16) + (kittens::laneid() / 16 * 4) + k;
                int col_offset = (j * 16) + (kittens::laneid() % 16);
                int packed = tm_ptr[row_offset];  // no raw buffer load here b/c row_offset guaranteed to land in valid tiles
                int token_id = packed & 0x00FFFFFF;
                int topk_slot = (packed & 0xFF000000) >> 24;
                int flat_offset = (token_id * TOP_K + topk_slot) * row_stride + col_offset;

                uint16_t store = __builtin_bit_cast(uint16_t, __float2bfloat16(flat[k]));
                llvm_amdgcn_raw_buffer_store_b16(store, srsrc, flat_offset * sizeof(T), 0, NT_LOAD_BITS);
            }
        }
    }
}
