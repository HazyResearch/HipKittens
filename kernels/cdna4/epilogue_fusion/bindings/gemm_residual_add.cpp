#include "gemm_base.cuh"
#include "residual_add.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(ResidualAddGlobals g, hipStream_t stream) {
    if (g.residual.rows() != g.a.rows() || g.residual.cols() != g.b.rows())
        throw std::runtime_error("residual_add: residual must be [M, N] (M = a.rows(), N = b.rows())");
    launch<ResidualAddEpilogue, ResidualAddGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + residual add: c = a @ b.T + residual, a = [M,K], b = [N,K], c and residual = [M,N] (bf16)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16},
        &ResidualAddGlobals::a, &ResidualAddGlobals::b, &ResidualAddGlobals::c, &ResidualAddGlobals::residual);
}
