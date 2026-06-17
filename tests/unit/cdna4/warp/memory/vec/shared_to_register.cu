#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER

template<typename T>
struct vec_load_store {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L> using valid = std::bool_constant<
        (NW == 1 && S<=64)  
        && S*RT_SHAPE::cols*sizeof(T) <= kittens::MAX_SHARED_MEMORY
        && (S*RT_SHAPE::cols*sizeof(T)) % (kittens::WARP_THREADS * 4) == 0
    >; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "shared_reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "shared_reg_vec_loadstore_gmem=half" :
                                                                                             "shared_reg_vec_loadstore_gmem=float";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = i_ref[i] + 1.f; // just a dummy op to prevent optimization away
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<RT_SHAPE::cols*S> al((int*)&__shm[0]); 
        kittens::sv<dtype, RT_SHAPE::cols*S> &shared_vec = al.template allocate<kittens::sv<dtype, RT_SHAPE::cols*S>>();
        kittens::rv<dtype, RT_SHAPE::cols*S, RT_SHAPE::cols, RT_SHAPE, L> reg_vec;
        kittens::load(shared_vec, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::load(reg_vec, shared_vec);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::add(reg_vec, reg_vec, dtype(1.)); // TODO: CHANGE HOST TOO
        kittens::store(shared_vec, reg_vec);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::store(output, shared_vec, {});
    }
};

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE=kittens::ducks::st_shape::st_16x16>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_gmem_type_1d_warp<vec_load_store, RT_SHAPE, ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<vec_load_store, RT_SHAPE, ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<vec_load_store, RT_SHAPE, ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);
}

void warp::memory::vec::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/shared_to_register tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}

#endif