#include <cstdint>
#include "gemm_base.cuh"
#include "rope.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RopeGlobals g, hipStream_t stream) {
    const int N = g.b.rows(), dh = g.cos_sin.cols();
    if (g.cos_sin.rows() != g.a.rows() || dh <= 0 || dh % 2 || N % dh)
        throw std::runtime_error("rope: cos_sin must be [M, Dh] with Dh even and N % Dh == 0 (M = a.rows(), N = b.rows())");
    if (g.rope_cols < 0 || g.rope_cols > N || g.rope_cols % BLOCK_SIZE)
        throw std::runtime_error("rope: rope_cols must be a multiple of 256 in [0, N]");
    if (reinterpret_cast<uintptr_t>(g.c.raw_ptr) % alignof(bf16_2) != 0)
        throw std::runtime_error("rope: output must have 4-byte alignment");
    launch<RopeEpilogue, RopeGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RoPE: c = RoPE(a @ b.T) on columns [0, rope_cols), interleaved pairs, natural column order; "
              "a = [M,K], b = [N,K], cos_sin = [M,Dh] one head's natural table (see ops/rotary.cuh), "
              "c = [M,N] 4-byte aligned (all bf16)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16,
                                           tensor_dtype::int_scalar},
        &RopeGlobals::a, &RopeGlobals::b, &RopeGlobals::c, &RopeGlobals::cos_sin, &RopeGlobals::rope_cols);
}
