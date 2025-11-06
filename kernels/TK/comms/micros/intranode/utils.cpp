#include "kittens.cuh"
using namespace kittens;

namespace kittens {
    namespace ducks {
        namespace pgl {
        
        struct identifier {};
        
        template<typename T>
        concept all = requires {
            typename T::identifier;
        } && std::is_same_v<typename T::identifier, identifier>;
        
        } // namespace pgl
    } // namespace ducks
} // namespace kittens

// TK-style: safe dimension helper
template<int D>
auto tk_dim_arg(int val) {
    if constexpr (D > 0) return nullptr;
    else return kittens::ducks::gl::make_arg_t<D>(val);
}

template<kittens::ducks::gl::all GL, int NUM_DEVICES = 8>
struct pgl_amd {
    using dtype = typename GL::dtype;
    using identifier = kittens::ducks::pgl::identifier;
    static constexpr int num_devices = NUM_DEVICES;

    GL gls[NUM_DEVICES];
    dtype* ptrs[NUM_DEVICES];
    int device_ids[NUM_DEVICES];

    // Main constructor
    __host__ inline pgl_amd(int* dev_ids, dtype** data_ptrs, int b, int d, int r, int c)
        : pgl_amd(std::make_index_sequence<NUM_DEVICES>{}, dev_ids, data_ptrs, b, d, r, c) {}

    // Compile-time constructor using index sequence
    template<size_t... I>
    __host__ inline pgl_amd(std::index_sequence<I...>, int* dev_ids, dtype** data_ptrs, int b, int d, int r, int c)
        : gls{
            GL(data_ptrs[I],
               tk_dim_arg<GL::__b__>(b),
               tk_dim_arg<GL::__d__>(d),
               tk_dim_arg<GL::__r__>(r),
               tk_dim_arg<GL::__c__>(c))...
          },
          ptrs{ data_ptrs[I]... },
          device_ids{ dev_ids[I]... }
    {}

    __host__ __device__ const GL& operator[](int i) const { return gls[i]; }

    __device__ inline dtype* ptr_at(int dev_idx, int idx) const {
        return &ptrs[dev_idx][idx];
    }
};

template<kittens::ducks::gl::all GL, int NUM_DEVICES = 8>
struct pgl_manager_amd {
    using T = typename GL::dtype;

    int device_ids[NUM_DEVICES];
    T* device_ptrs[NUM_DEVICES];
    GL gls[NUM_DEVICES];

    __host__ pgl_manager_amd(int* _device_ids, T** _data,
                             kittens::ducks::gl::make_arg_t<GL::__b__> _b,
                             kittens::ducks::gl::make_arg_t<GL::__d__> _d,
                             kittens::ducks::gl::make_arg_t<GL::__r__> _r,
                             kittens::ducks::gl::make_arg_t<GL::__c__> _c)
        : pgl_manager_amd(std::make_index_sequence<NUM_DEVICES>{}, _device_ids, _data, _b, _d, _r, _c) {}

    template<size_t... I>
    __host__ pgl_manager_amd(std::index_sequence<I...>,
                             int* _device_ids, T** _data,
                             kittens::ducks::gl::make_arg_t<GL::__b__> _b,
                             kittens::ducks::gl::make_arg_t<GL::__d__> _d,
                             kittens::ducks::gl::make_arg_t<GL::__r__> _r,
                             kittens::ducks::gl::make_arg_t<GL::__c__> _c)
        : gls{ GL(_data[I], _b, _d, _r, _c)... },
          device_ptrs{ _data[I]... },
          device_ids{ _device_ids[I]... }
    {}

    __host__ __device__ const GL& get_gl(int i) const {
        return gls[i];
    }

    __host__ __device__ auto get_pgl_obj(int i) const {
        using dtype = typename GL::dtype;
        return pgl_amd<GL, NUM_DEVICES>(
            const_cast<int*>(device_ids),
            const_cast<dtype**>(device_ptrs),
            gls[i].batch(),
            gls[i].depth(),
            gls[i].rows(),
            gls[i].cols());
    }
};


