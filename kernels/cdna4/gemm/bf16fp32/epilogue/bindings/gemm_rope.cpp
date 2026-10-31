#include <cstdint>
#include "gemm_base.cuh"
#include "rope.cuh"
#include "rope_args.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RopeGlobals g, hipStream_t stream) {
    check_rope_args("rope", g, g.b.rows(), {g.b.rows()});
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
