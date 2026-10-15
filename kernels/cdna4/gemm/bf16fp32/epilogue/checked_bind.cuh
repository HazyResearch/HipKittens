#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "pyutils/pyutils.cuh"

// Like kittens::py::bind_function, but raises unless each tensor argument has the expected torch
// dtype, and takes the HIP stream to launch on as a trailing integer (torch's .cuda_stream). Without it a bf16 tensor passed where fp32 is expected (or the reverse) is silently
// reinterpreted: from_object reads only the data pointer and the shape.
enum class tensor_dtype { bf16, fp32 };

// torch.bfloat16 and torch.float32 are singleton objects: look each up once, then compare pointers.
// release() keeps the references alive for the whole process, so nothing is freed after Python shuts down.
inline PyObject* torch_dtype(tensor_dtype d) {
    static PyObject* bf16 = pybind11::object(pybind11::module_::import("torch").attr("bfloat16")).release().ptr();
    static PyObject* fp32 = pybind11::object(pybind11::module_::import("torch").attr("float32")).release().ptr();
    return d == tensor_dtype::bf16 ? bf16 : fp32;
}

inline void require_dtype(const pybind11::object& t, tensor_dtype expected, size_t index) {
    const pybind11::object got = t.attr("dtype");
    if (got.ptr() != torch_dtype(expected))
        throw std::runtime_error("argument " + std::to_string(index) + " must be " +
                                 (expected == tensor_dtype::bf16 ? "torch.bfloat16" : "torch.float32") +
                                 ", got " + std::string(pybind11::str(got)));
}

template<auto function, typename TGlobal, typename... MT>
void bind_checked(pybind11::module_& m, const char* name,
                  std::array<tensor_dtype, sizeof...(MT)> dtypes, MT TGlobal::*... members) {
    m.def(name, [dtypes](kittens::py::object<MT>... args, std::uintptr_t stream) {
        size_t i = 0;
        ((require_dtype(args, dtypes[i], i), ++i), ...);
        TGlobal g{kittens::py::from_object<MT>::make(args)...};
        function(g, reinterpret_cast<hipStream_t>(stream));
    });
}
