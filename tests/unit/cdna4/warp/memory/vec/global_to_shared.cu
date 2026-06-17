#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED

template<typename T>
struct shared_vec_load_store {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW> using valid = std::bool_constant<
        NW == 1 && S<=64 
        && S*ST_SHAPE::cols*sizeof(T) <= kittens::MAX_SHARED_MEMORY
        && (S*ST_SHAPE::cols*sizeof(T)) % (kittens::WARP_THREADS * 4) == 0
    >;
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "shared_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "shared_vec_loadstore_gmem=half" :
                                                                                             "shared_vec_loadstore_gmem=float";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, kittens::ducks::gl::all GL> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<ST_SHAPE::cols*S> al((int*)&__shm[0]); 
        kittens::sv<dtype, ST_SHAPE::cols*S> &shared_vec = al.template allocate<kittens::sv<dtype, ST_SHAPE::cols*S>>();
        kittens::load(shared_vec, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::store(output, shared_vec, {});
    }
};

void warp::memory::vec::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    using DEFAULT_ST_SHAPE = kittens::ducks::st_shape::st_16x16;
    using DEFAULT_RT_SHAPE = kittens::ducks::rt_shape::rt_16x16;
    sweep_gmem_type_1d_warp<shared_vec_load_store, DEFAULT_RT_SHAPE, DEFAULT_ST_SHAPE, SIZE>::run(results);
}

#endif