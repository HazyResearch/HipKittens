#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_VEC_CONVERSIONS

template<typename T>
struct test_subvec {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, typename _SV_S> using valid = std::bool_constant<
        NW == 1 && S<=64 
        && S % _SV_S::value == 0 
        && S*ST_SHAPE::cols*sizeof(T) <= kittens::MAX_SHARED_MEMORY
        && (S*ST_SHAPE::cols*sizeof(T)) % (kittens::WARP_THREADS * 4) == 0
        && sizeof(dtype) != 1
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_vec_convert_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_vec_convert_gmem=half" :
                                                                                         "shared_vec_convert_gmem=float";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, gl_t GL, typename _SV_S>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int SV_S = _SV_S::value;
        constexpr int subvec_size = SV_S * ST_SHAPE::cols;
        for(int i = 0; i < S*ST_SHAPE::cols; i++) {
            int subvec_idx = i / subvec_size;  // Which subvector does element i belong to?
            o_ref[i] = i_ref[i] * float(subvec_idx);
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, gl_t GL, typename _SV_S> __device__ static void device_func(const GL &input, const GL &output) {
        constexpr int SV_S = _SV_S::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<ST_SHAPE::cols*S> al((int*)&__shm[0]); 
        kittens::sv<dtype, ST_SHAPE::cols*S> &shared_vec = al.template allocate<kittens::sv<dtype, ST_SHAPE::cols*S>>();
        kittens::load(shared_vec, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int i = 0; i < S/SV_S; i++) {
            auto ref = kittens::subvec_inplace<ST_SHAPE::cols*SV_S>(shared_vec, i);
            kittens::rv<dtype, ST_SHAPE::cols*SV_S, RT_SHAPE::cols, RT_SHAPE, kittens::ducks::rv_layout::naive> reg;
            kittens::load(reg, ref);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            kittens::mul(reg, reg, dtype(i));
            kittens::store(output, reg, {0, 0, 0, i});
        }

    }
};

void warp::shared::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    using DEFAULT_ST_SHAPE = kittens::ducks::st_shape::st_16x16;
    using DEFAULT_RT_SHAPE = kittens::ducks::rt_shape::rt_16x16;
    sweep_gmem_type_1d_warp<test_subvec, DEFAULT_RT_SHAPE, DEFAULT_ST_SHAPE, SIZE, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_1d_warp<test_subvec, DEFAULT_RT_SHAPE, DEFAULT_ST_SHAPE, SIZE, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_1d_warp<test_subvec, DEFAULT_RT_SHAPE, DEFAULT_ST_SHAPE, SIZE, std::integral_constant<int, 4>>::run(results);
}

#endif