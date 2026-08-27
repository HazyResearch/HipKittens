/**
 * @file pyext.h
 * @brief pybind11 module for one rung of the gfx1250 GEMM ladder.
 *
 * Included by `harness.h` under `-DHARNESS_PYEXT`, once per rung: each rung declares its own tile
 * geometry before including `common.h` and each names its entry point `dispatch`, so one module
 * cannot serve all twelve. The module's name must match the .so basename, so both come from
 * `-DHK_PYEXT_NAME`; `make KERNEL=00_gemm_naive` builds the dynamically imported
 * `00_gemm_naive` module.
 *
 * The tensor contract, which is the whole of what is easy to get wrong here because getting it
 * wrong produces a wrong answer rather than an error:
 *
 *   a  [M, K] row-major, K contiguous
 *   b  [N, K] row-major, K contiguous       the kernel computes C = a . b^T via `mma_ABt`
 *   c  [M, N] COLUMN-major, so element (r, c) lives at c*M + r
 *
 * `c` is column-major because `gl_c` carries `gl_layout::col_major`, which makes the row axis the
 * unit-stride one and lets an accumulator whose register-contiguous axis is M leave as a wide store
 * instead of a scatter; the epilogue rungs exist to exploit that. Torch allocates row-major, so a
 * caller has to ask: `torch.empty_strided((M, N), (1, M), ...)`. The strides are the contract and
 * so are what this file checks; `pyutils::from_object<GL>` cannot, since it takes the extents from
 * the shape, requires C-contiguity and never consults `GL::layout`.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <sstream>
#include <string>
#include <vector>

#ifndef HK_PYEXT_NAME
#error "HARNESS_PYEXT needs -DHK_PYEXT_NAME=<module>, matching the .so basename"
#endif

namespace hk_pyext {

namespace pyb = pybind11;

template<typename T> struct torch_dtype_name;
template<> struct torch_dtype_name<__hip_bfloat16> {
    static constexpr const char* value = "torch.bfloat16";
};
template<> struct torch_dtype_name<__half> {
    static constexpr const char* value = "torch.float16";
};

/* What this file needs off a torch tensor, read through the Python object rather than through
 * libtorch: the module then links against nothing but pybind11 and the HIP runtime, so a rung still
 * builds with one hipcc invocation and no torch headers. */
struct tensor_view {
    uint64_t ptr = 0;
    std::vector<int64_t> shape, stride;
};

[[noreturn]] static void fail(const std::string& who, const std::string& msg)
{
    throw std::invalid_argument(who + ": " + msg);
}

static std::string dims_str(const std::vector<int64_t>& v)
{
    std::ostringstream os;
    os << "(";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
    os << ")";
    return os.str();
}

static tensor_view inspect(const pyb::object& t, const std::string& who)
{
    if (!pyb::hasattr(t, "data_ptr") || !pyb::hasattr(t, "stride"))
        fail(who, "expected a torch.Tensor");
    if (t.attr("device").attr("type").cast<std::string>() == "cpu")
        fail(who, "tensor must be on the GPU, not the CPU");

    const std::string dt = pyb::str(t.attr("dtype")).cast<std::string>();
    if (dt != torch_dtype_name<dev_elem>::value)
        fail(who, "dtype must be " + std::string(torch_dtype_name<dev_elem>::value)
                  + ", got " + dt);

    tensor_view v;
    v.ptr = t.attr("data_ptr")().cast<uint64_t>();
    for (auto d : t.attr("shape").cast<pyb::tuple>()) v.shape.push_back(d.cast<int64_t>());
    for (auto s : t.attr("stride")().cast<pyb::tuple>()) v.stride.push_back(s.cast<int64_t>());
    if (v.shape.size() != 2)
        fail(who, "expected a 2-D tensor, got " + std::to_string(v.shape.size()) + " dims");
    return v;
}

static void require_row_major(const tensor_view& v, const std::string& who)
{
    if (v.stride[1] != 1 || v.stride[0] != v.shape[1])
        fail(who, "must be row-major, i.e. stride " + dims_str({v.shape[1], 1})
                  + " for shape " + dims_str(v.shape) + "; got stride " + dims_str(v.stride));
}

static void require_col_major(const tensor_view& v, const std::string& who)
{
    if (v.stride[0] != 1 || v.stride[1] != v.shape[0])
        fail(who, "must be COLUMN-major, i.e. stride " + dims_str({1, v.shape[0]})
                  + " for shape " + dims_str(v.shape) + "; got stride " + dims_str(v.stride)
                  + ". Allocate it as torch.empty_strided((M, N), (1, M), ...)");
}

/* M, N and K are each pinned by two of the three tensors, and disagreement is refused rather than
 * resolved: a caller who allocated `c` the wrong way round is what this catches, and at M == N the
 * swap is invisible to every other check. */
static gfx1250_gemm::gemm_globals make_globals(const pyb::object& a_obj,
                                               const pyb::object& b_obj,
                                               const pyb::object& c_obj)
{
    const tensor_view a = inspect(a_obj, "a");
    const tensor_view b = inspect(b_obj, "b");
    const tensor_view c = inspect(c_obj, "c");
    require_row_major(a, "a");
    require_row_major(b, "b");
    require_col_major(c, "c");

    const int64_t M = a.shape[0], K = a.shape[1];
    const int64_t N = b.shape[0];
    if (b.shape[1] != K)
        fail("b", "is [N, K] with K contiguous, so b.shape[1] must equal a.shape[1]=" +
                  std::to_string(K) + "; got " + dims_str(b.shape));
    if (c.shape[0] != M || c.shape[1] != N)
        fail("c", "must be [M, N] = " + dims_str({M, N}) + "; got " + dims_str(c.shape));

    gfx1250_gemm::gl_e a_gl(reinterpret_cast<dev_elem*>(a.ptr),
                            size_t(1), size_t(1), size_t(M), size_t(K));
    gfx1250_gemm::gl_e b_gl(reinterpret_cast<dev_elem*>(b.ptr),
                            size_t(1), size_t(1), size_t(N), size_t(K));
    gfx1250_gemm::gl_c c_gl(reinterpret_cast<dev_elem*>(c.ptr),
                            size_t(1), size_t(1), size_t(M), size_t(N));

    return gfx1250_gemm::gemm_globals{a_gl, b_gl, c_gl};
}

static std::string protocol_str()
{
    std::ostringstream os;
    os << HK_WARMUP_ITERS << " warmup; "
       << (HK_FLUSH_BETWEEN_ITERS ? "cache flush before every launch" : "no cache flush")
       << "; one event pair per launch";
    return os.str();
}

static void check_launch(const char* what)
{
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

static void py_dispatch(const pyb::object& a, const pyb::object& b, const pyb::object& c)
{
    gfx1250_gemm::gemm_globals g = make_globals(a, b, c);
    const gfx1250_gemm::launch_config launch(g, /*stream=*/ 0);
    HIP_OK(hipDeviceSynchronize());
    dispatch(g, launch);
    check_launch("dispatch");
    HIP_OK(hipDeviceSynchronize());
}

static pyb::dict py_bench(const pyb::object& a, const pyb::object& b, const pyb::object& c,
                          int iters)
{
    gfx1250_gemm::gemm_globals g = make_globals(a, b, c);
    const gfx1250_gemm::launch_config launch(g, /*stream=*/ 0);
    const int M = g.M(), N = g.N(), K = g.K();
    HIP_OK(hipDeviceSynchronize());
    const hk_timing t = hk_run_protocol(g, launch, iters);
    check_launch("bench");

    pyb::dict d;
    d["ms_per_iter"] = t.ms_per;
    d["ms_min"]      = t.ms_min;
    d["ms_max"]      = t.ms_max;
    d["spread_pct"]  = t.spread_pct();
    d["gflops"]      = t.gflops(M, N, K);
    d["tflops"]      = t.gflops(M, N, K) / 1000.0;
    d["measured"]    = t.measured;
    d["warmup"]      = HK_WARMUP_ITERS;
    d["flush_mb"]    = t.flush_mb;
    d["l2_mb"]       = t.l2_mb;
    d["protocol"]    = protocol_str();
    return d;
}

} // namespace hk_pyext

// Two levels, so the argument is expanded to its value before being turned into a string.
#define HK_STRINGIFY_(x) #x
#define HK_STRINGIFY(x) HK_STRINGIFY_(x)

/* `PYBIND11_MODULE` pastes its name argument into the init symbol with `##`, which suppresses macro
 * expansion of its operands. Routing the name through a wrapper expands it one level earlier, where
 * it is an ordinary argument substitution. */
#define HK_PYBIND_MODULE_(nm) PYBIND11_MODULE(nm, m)
#define HK_PYBIND_MODULE HK_PYBIND_MODULE_(HK_PYEXT_NAME)

HK_PYBIND_MODULE {
    namespace pyb = pybind11;
    using namespace hk_pyext;

    m.doc() = "HipKittens gfx1250 GEMM rung " HK_STRINGIFY(HK_PYEXT_NAME)
              ". C = a @ b.t(), with a [M,K] and b [N,K] row-major and c [M,N] COLUMN-major "
              "(stride (1, M)).";

    m.def("dispatch", &py_dispatch, pyb::arg("a"), pyb::arg("b"), pyb::arg("c"),
          "One launch of this rung. Blocks until c is written.");
    m.def("bench", &py_bench, pyb::arg("a"), pyb::arg("b"), pyb::arg("c"),
          pyb::arg("iters") = HK_MEASURED_ITERS,
          "Run the measurement protocol: " HK_STRINGIFY(HK_WARMUP_ITERS) " warmup iterations "
          "then `iters` measured ones, each preceded by a cache flush and timed individually.");

    m.attr("rung")        = HK_STRINGIFY(HK_PYEXT_NAME);
    m.attr("dtype")       = torch_dtype_name<dev_elem>::value;
    m.attr("BLOCK_M")     = BLOCK_M;
    m.attr("BLOCK_N")     = BLOCK_N;
    m.attr("BLOCK_K")     = BLOCK_K;
    m.attr("WARPS_M")     = WARPS_M;
    m.attr("WARPS_N")     = WARPS_N;
    m.attr("warmup_iters") = HK_WARMUP_ITERS;
    m.attr("c_layout")    = "column-major";
    m.attr("benchmark_protocol") = protocol_str();
}
