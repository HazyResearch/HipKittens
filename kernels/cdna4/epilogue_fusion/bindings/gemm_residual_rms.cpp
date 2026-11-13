#include "gemm_base.cuh"
#include "residual_rms.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(ResidualRMSGlobals g, hipStream_t stream) {
    const int M = g.a.rows(), N = g.b.rows();
    if (g.residual.rows() != M || g.residual.cols() != N)
        throw std::runtime_error("residual_rms: residual must be [M, N] (M = a.rows(), N = b.rows())");
    if (g.partials.cols() != M || g.partials.rows() != N / REG_BLOCK_N)
        throw std::runtime_error("residual_rms: partials must be [N/64, M] fp32 (N = b.rows(), M = a.rows())");
    launch<ResidualRMSEpilogue, ResidualRMSGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + residual + partial RMS: h = a @ b.T + residual stored in c, partials[g, m] = sum of "
              "h[m, cols of group g]^2, a = [M,K], b = [N,K], c and residual = [M,N] (bf16), partials = [N/64, M] (fp32)";
    bind_checked<dispatch>(m, "dispatch",
        {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32},
        &ResidualRMSGlobals::a, &ResidualRMSGlobals::b, &ResidualRMSGlobals::c,
        &ResidualRMSGlobals::residual, &ResidualRMSGlobals::partials);
}
