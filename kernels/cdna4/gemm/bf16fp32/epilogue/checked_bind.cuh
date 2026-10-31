#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include "pyutils/pyutils.cuh"

// Like kittens::py::bind_function, but raises unless each tensor argument has the expected torch dtype and at
// most two dimensions, and takes the HIP stream to launch on as a trailing integer (torch's .cuda_stream).
// Without the dtype check a bf16 tensor passed where fp32 is expected (or the reverse) is silently reinterpreted:
// from_object reads only the data pointer and the shape. Without the rank check from_object pads a 3-D [B, M, K]
// tensor to 4-D and the kernel silently computes only its first [M, K] slice. The tk_* modules are internal: call
// them through hk.py, which also launches on the tensors' GPU.
enum class tensor_dtype { bf16, fp32, int_scalar };   // int_scalar: a Python int (an `int` Globals member)

// torch.bfloat16 and torch.float32 are singleton objects: look each up once, then compare pointers.
// release() keeps the references alive for the whole process, so nothing is freed after Python shuts down.
inline PyObject* torch_dtype(tensor_dtype d) {
    static PyObject* bf16 = pybind11::object(pybind11::module_::import("torch").attr("bfloat16")).release().ptr();
    static PyObject* fp32 = pybind11::object(pybind11::module_::import("torch").attr("float32")).release().ptr();
    return d == tensor_dtype::bf16 ? bf16 : fp32;
}

inline void require_dtype(const pybind11::object& t, tensor_dtype expected, size_t index) {
    if (expected == tensor_dtype::int_scalar) {
        if (!pybind11::isinstance<pybind11::int_>(t))
            throw std::runtime_error("argument " + std::to_string(index) + " must be a Python int");
        return;
    }
    if (const int dims = t.attr("dim")().cast<int>(); dims > 2)
        throw std::runtime_error("argument " + std::to_string(index) + " must be 1-D or 2-D, got " +
                                 std::to_string(dims) + "-D");
    const pybind11::object got = t.attr("dtype");
    if (got.ptr() != torch_dtype(expected))
        throw std::runtime_error("argument " + std::to_string(index) + " must be " +
                                 (expected == tensor_dtype::bf16 ? "torch.bfloat16" : "torch.float32") +
                                 ", got " + std::string(pybind11::str(got)));
}

template<auto function, typename TGlobal, typename... MT>
void bind_checked(pybind11::module_& m, const char* name,
                  std::array<tensor_dtype, sizeof...(MT)> dtypes, MT TGlobal::*... members) {
    m.def(name, [dtypes, members...](kittens::py::object<MT>... args, std::uintptr_t stream) {
        size_t i = 0;
        ((require_dtype(args, dtypes[i], i), ++i), ...);
        const std::tuple<MT...> made{kittens::py::from_object<MT>::make(args)...};
        // Aggregate-initialise in listed order (a member listed out of declaration order with a different type is a
        // compile error), then route every argument through its member pointer, so the pointers and not the order
        // decide which field gets which tensor (residual_rms_gamma has three bf16 gls; QKV has b_k and b_v).
        // Unlisted trailing members (QKV's q_blocks / qk_blocks) stay zero until launch() sets them.
        TGlobal g = std::apply([](const MT&... x) { return TGlobal{x...}; }, made);
        std::apply([&](const MT&... x) { ((g.*members = x), ...); }, made);
        function(g, reinterpret_cast<hipStream_t>(stream));
    });
}
