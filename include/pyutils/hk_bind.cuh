#pragma once
/*
 * A pybind entry point that spends its time on the launch.
 *
 * `pyutils.cuh`'s `from_object` is written for a benchmark harness, where the
 * kernel runs for milliseconds and the wrapper is free. It is not free for the
 * Phase 2 ops in python/hk/ops: a 4096x4096 quantize is sixty microseconds of
 * GPU, and these kernels take three and four tensors.
 *
 * Per tensor, the upstream converter does eleven Python operations and builds
 * three std::strings: `hasattr(__class__)`, `__class__.__name__` cast to a
 * std::string and compared, `is_contiguous()`, `device` then `.type` cast to a
 * std::string and compared, `shape`, a cast per dimension, and `data_ptr()`.
 * Every `obj.attr("name")` also builds a fresh Python str for the name.
 *
 * This one does five, with the attribute names interned once:
 *
 *   - `shape`, read straight out of the tuple (`torch.Size` is a tuple
 *     subclass, so PyTuple_GET_ITEM applies);
 *   - `is_cuda`, which is an attribute, where `device.type` is an attribute
 *     that builds a `torch.device` plus a second attribute plus a string
 *     compare;
 *   - `is_contiguous()`;
 *   - `data_ptr()`.
 *
 * **Which two checks were dropped, and why those two.** The class-name test
 * goes because every later step is itself a test: a non-tensor has no `.shape`
 * tuple and raises the same error one attribute later. The `device.type`
 * string goes because `is_cuda` answers the same question without the
 * allocation.
 *
 * **Which check was kept, and why.** Contiguity. A non-contiguous tensor has a
 * valid pointer and a shape that looks right, and `gl` is a base pointer plus
 * strides it derives from that shape -- so dropping this check does not fail,
 * it reads the wrong elements and returns a plausible wrong answer. That is
 * the class of bug this whole toolchain exists to make unrepresentable, and
 * it is worth a method call. The same reasoning says `is_cuda` stays too,
 * even though a host pointer would fault rather than lie: it costs an
 * attribute load and the error message is the difference between "tensor must
 * be on the GPU" and a memory access fault with no context.
 */

#include "util.cuh"
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

namespace kittens {
namespace py {
namespace fast {

/* Attribute names, interned once per process and never released -- they live
 * as long as the module does. Constructed on the first launch, which is under
 * the GIL. */
struct attr_keys {
    PyObject *shape, *data_ptr, *is_contiguous, *is_cuda, *stride;
    attr_keys()
        : shape(PyUnicode_InternFromString("shape")),
          data_ptr(PyUnicode_InternFromString("data_ptr")),
          is_contiguous(PyUnicode_InternFromString("is_contiguous")),
          is_cuda(PyUnicode_InternFromString("is_cuda")),
          stride(PyUnicode_InternFromString("stride")) {}
};
inline const attr_keys &keys() { static const attr_keys k; return k; }

[[noreturn]] inline void fail(const char *what) {
    // The pending Python error, if any, is less informative than `what` --
    // "AttributeError: 'int' object has no attribute 'shape'" against "hk
    // expected a torch.Tensor". Clear it so pybind reports the throw.
    PyErr_Clear();
    throw std::runtime_error(what);
}

inline PyObject *call0(PyObject *obj, PyObject *name, const char *what) {
    PyObject *fn = PyObject_GetAttr(obj, name);
    if (fn == nullptr) fail(what);
    PyObject *r = PyObject_CallNoArgs(fn);
    Py_DECREF(fn);
    if (r == nullptr) fail(what);
    return r;
}

inline bool truthy(PyObject *obj, PyObject *name, const char *what) {
    PyObject *v = PyObject_GetAttr(obj, name);
    if (v == nullptr) fail(what);
    const int t = PyObject_IsTrue(v);
    Py_DECREF(v);
    if (t < 0) fail(what);
    return t == 1;
}

static constexpr const char *NOT_A_TENSOR =
    "hk: expected a torch.Tensor on the GPU";

/* The row stride of a non-contiguous tensor `gl` can describe, or a throw.
 *
 * A gl is a base pointer, the four extents, and one row stride: elements are
 * contiguous along the last axis, rows are `row stride` apart, and the outer
 * axes are compact over the rows. That is exactly a slice of the last axis --
 * x[..., I:] of an (M, 2I) tensor -- which is the view an engine hands over
 * without a copy. Anything else would be read as if it were that, wrongly,
 * so it is refused here rather than returned as plausible numbers. Only the
 * non-contiguous path pays for the `stride()` call. */
inline int strided_rows(PyObject *o, Py_ssize_t dims, const int s[4]) {
    PyObject *st = call0(o, keys().stride, NOT_A_TENSOR);
    if (!PyTuple_Check(st) || PyTuple_GET_SIZE(st) != dims) {
        Py_DECREF(st);
        fail(NOT_A_TENSOR);
    }
    long t[4] = {0, 0, 0, 1};
    for (Py_ssize_t i = 0; i < dims; ++i)
        t[4 - dims + i] = PyLong_AsLong(PyTuple_GET_ITEM(st, i));
    Py_DECREF(st);
    if (PyErr_Occurred()) fail(NOT_A_TENSOR);
    const long row = dims >= 2 ? t[2] : s[3];
    const bool ok = (s[3] == 1 || t[3] == 1)
                 && row >= s[3]
                 && (dims < 3 || s[1] == 1 || t[1] == s[2] * row)
                 && (dims < 4 || s[0] == 1 || t[0] == s[1] * t[1]);
    if (!ok)
        fail("hk: a non-contiguous tensor is taken only as a slice of its last "
             "axis -- contiguous along it, rows evenly strided, outer axes "
             "compact over the rows (x[..., a:b] of a contiguous tensor). "
             "This one is not; call .contiguous().");
    return (int)row;
}

template<ducks::gl::all GL> GL to_gl(pybind11::handle h) {
    PyObject *o = h.ptr();
    const attr_keys &k = keys();

    PyObject *shape = PyObject_GetAttr(o, k.shape);
    if (shape == nullptr || !PyTuple_Check(shape)) {
        Py_XDECREF(shape);
        fail(NOT_A_TENSOR);
    }
    const Py_ssize_t dims = PyTuple_GET_SIZE(shape);
    if (dims > 4) {
        Py_DECREF(shape);
        fail("hk: expected Tensor.ndim <= 4");
    }
    // Leading axes default to 1, so a 2D tensor is a (1, 1, rows, cols)
    // global -- the same convention pyutils.cuh uses.
    int s[4] = {1, 1, 1, 1};
    for (Py_ssize_t i = 0; i < dims; ++i)
        s[4 - dims + i] = (int)PyLong_AsLong(PyTuple_GET_ITEM(shape, i));
    Py_DECREF(shape);
    if (PyErr_Occurred()) fail(NOT_A_TENSOR);

    if (!truthy(o, k.is_cuda, NOT_A_TENSOR))
        fail("hk: tensor must be on the GPU");

    PyObject *cont = call0(o, k.is_contiguous, NOT_A_TENSOR);
    const int ok = PyObject_IsTrue(cont);
    Py_DECREF(cont);
    const int row_stride = ok == 1 ? -1 : strided_rows(o, dims, s);

    PyObject *p = call0(o, k.data_ptr, NOT_A_TENSOR);
    const uint64_t ptr = (uint64_t)PyLong_AsUnsignedLongLong(p);
    Py_DECREF(p);
    if (PyErr_Occurred()) fail(NOT_A_TENSOR);

    return make_gl<GL>(ptr, s[0], s[1], s[2], s[3], row_stride);
}

template<typename T> struct from_object {
    static T make(pybind11::handle h) { return pybind11::cast<T>(h); }
};
template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::handle h) { return to_gl<GL>(h); }
};

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> {
    using member_type = MT;
    using type = T;
};
/* `handle` and not `object`: pybind borrows the argument either way, and
 * `object` costs an incref/decref pair per tensor per launch for a reference
 * the caller already holds for the duration of the call. */
template<typename> using handle = pybind11::handle;

template<auto function, typename TGlobal>
static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](handle<decltype(member_ptrs)>... args) {
        TGlobal g {
            from_object<typename trait<decltype(member_ptrs)>::member_type>
                ::make(args)...
        };
        function(g);
    });
}

} // namespace fast
} // namespace py
} // namespace kittens
