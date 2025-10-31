


template <typename T, typename ST, int N_THREADS>
__device__ inline void buffer_load_lds(int i, const T* lds_base, i32x4 srsrc, int row_stride) {
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(fp8e4m3) / (16 * N_THREADS); // 8
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    constexpr int elem_per_thread = 16 / sizeof(fp8e4m3);
    constexpr int col_dim_threads = ST::cols / elem_per_thread;
    int row_offset = i * elem_per_thread * N_THREADS / ST::cols + threadIdx.x / col_dim_threads;
    int col_offset = threadIdx.x % col_dim_threads * elem_per_thread;

    uintptr_t offset_in_st = (row_offset * ST::underlying_cols + col_offset) * sizeof(T);
    offset_in_st ^= (((offset_in_st % (256*8)) >> 8) << 4);
 
    row_offset = offset_in_st / (ST::underlying_cols * sizeof(T));
    col_offset = offset_in_st % (ST::underlying_cols * sizeof(T)) / sizeof(T);

    uintptr_t offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

    const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
    uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

    llvm_amdgcn_raw_buffer_load_lds(
        srsrc, // buffer resource
        lds_ptr,
        16, // 16 bytes
        offset_in_global,
        0, 
        0, // instruction offset
        static_cast<int>(coherency::cache_all)); // cache coherency
}

template<int axis, bool assume_aligned,
         ducks::rt_layout::all RT_layout, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_gl_to_st(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); 
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; 
    const int laneid = kittens::laneid();
    const int warp_id = warpid();
    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));
    const T* lds_base = &dst.data[0] + (warp_id * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        constexpr int col_dim_threads = ST::cols / elem_per_thread;
        int row_offset = i * elem_per_thread * N_THREADS / ST::cols + threadIdx.x / col_dim_threads;
        int col_offset = threadIdx.x % col_dim_threads * elem_per_thread;

        uintptr_t offset_in_st = (row_offset * ST::underlying_cols + col_offset) * sizeof(T);
        offset_in_st ^= (((offset_in_st % (256*8)) >> 8) << 4);
        offset_in_st /= sizeof(T);

        row_offset = offset_in_st / (ST::underlying_cols);
        col_offset = offset_in_st % (ST::underlying_cols);

        uintptr_t offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            16, // 16 bytes
            offset_in_global,
            0, 
            0, // instruction offset
            static_cast<int>(coherency::cache_all)); // cache coherency
    }
}

/************************************************************************/

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets_fp8(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); 
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; 
    const int laneid = kittens::laneid();
    const int warp_id = warpid();
    const int row_stride = src.template stride<axis>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        constexpr int col_dim_threads = ST::cols / elem_per_thread;
        int row_offset = i * elem_per_thread * N_THREADS / ST::cols + threadIdx.x / col_dim_threads;
        int col_offset = threadIdx.x % col_dim_threads * elem_per_thread;

        uintptr_t offset_in_st = (row_offset * ST::underlying_cols + col_offset) * sizeof(T);
        offset_in_st ^= (((offset_in_st % (256*8)) >> 8) << 4);
        offset_in_st /= sizeof(T);

        row_offset = offset_in_st / (ST::underlying_cols);
        col_offset = offset_in_st % (ST::underlying_cols);

        uintptr_t offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);
        swizzled_offsets[i] = offset_in_global;
    }
}

template<int axis, bool assume_aligned,
         ducks::rt_layout::all RT_layout, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_gl_to_st(ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); 
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; 
    const int laneid = kittens::laneid();
    const int warp_id = warpid();
    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));
    const T* lds_base = &dst.data[0] + (warp_id * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            16, // 16 bytes
            swizzled_offsets[i],
            0, 
            0, // instruction offset
            static_cast<int>(coherency::cache_all)); // cache coherency
    }
}

/********************************/

template<typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_base_wrapper(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma, int n, int m, int k) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
                  std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>);
    
    mma_ABt_base(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m]
    );
}

template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load_st_to_rt(RT &dst, const ST &src) {

    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols  == ST::cols,  "register tile and shared tile must match cols");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2 || sizeof(U) == 1, "only supporting 16 and 8-bit dtypes");
    static_assert((!std::is_same_v<T, fp8e4m3>) || std::is_same_v<U, T>, "global and shared tile must have the same dtype if fp8");

    constexpr int subtile_stride = RT::base_tile_cols * sizeof(U) / 2;
    const int tile_stride = subtile_stride * 2;
    constexpr int row_stride = RT::base_tile_rows * ST::underlying_cols * sizeof(U);

    const int elem_per_thread = 16 / sizeof(U); // 8 if bf16, 16 if fp8e4m3
    uint32_t st_offset = (laneid() % RT::base_tile_rows) * ST::underlying_cols + (laneid() / RT::base_tile_rows * 16 / sizeof(U));
    uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[st_offset]);
    uint32_t addr0 = base_addr;
    addr0 ^= (((addr0 % (256*8)) >> 8) << 4);
    uint32_t addr1 = base_addr + subtile_stride;
    addr1 ^= (((addr1 % (256*8)) >> 8) << 4);

    constexpr int num_rts = RT::rows / RT::base_tile_rows;

    #pragma unroll
    for(int i = 0; i < num_rts; i++) {

        // tile 0
        asm volatile(
            "ds_read_b128 %0, %1 offset:%2\n"
            : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][0].data[0]))
            : "v"(addr0), "i"(i * row_stride)
            : "memory"
        );

        asm volatile(
            "ds_read_b128 %0, %1 offset:%2\n"
            : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][0].data[4]))
            : "v"(addr1), "i"(i * row_stride)
            : "memory"
        );
    }
}

