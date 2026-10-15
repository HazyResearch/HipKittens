#include "gemm_base.cuh"
#include "swiglu.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(gemm_args_base g, hipStream_t stream) { launch<SwigluEpilogue, gemm_args_base>(g, stream); }

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + SwiGLU: c = silu(gate) * value, a = [M,K], b = [N,K] with rows permuted "
              "(see epilogues/swiglu.cuh), c = [M,N/2] (bf16)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16},
        &gemm_args_base::a, &gemm_args_base::b, &gemm_args_base::c);
}
