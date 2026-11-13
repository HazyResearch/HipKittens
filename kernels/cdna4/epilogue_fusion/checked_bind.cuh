#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include "pyutils/pyutils.cuh"

// kittens::py::bind_function plus dtype and rank (<= 2) checks, and a trailing HIP stream int.
// from_object ignores dtype, and pads 3-D tensors to 4-D (only the first slice would be computed).
// Internal: call through hk.py.
enum class tensor_dtype { bf16, fp32, int_scalar };   // int_scalar: Python int

// Torch dtypes are singletons: compare pointers. release() leaks them so nothing is freed after shutdown.
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
        // Aggregate-init, then assign through member pointers so the pointers, not the order, pick each field.
        // Unlisted trailing members (q_blocks, qk_blocks) stay zero until launch().
        TGlobal g = std::apply([](const MT&... x) { return TGlobal{x...}; }, made);
        std::apply([&](const MT&... x) { ((g.*members = x), ...); }, made);
        function(g, reinterpret_cast<hipStream_t>(stream));
    });
}
