#include "gemm_base.cuh"
#include "residual_rms_gamma.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(ResidualRMSGammaGlobals g, hipStream_t stream) {
    const int M = g.a.rows(), N = g.b.rows();
    if (g.residual.rows() != M || g.residual.cols() != N)
        throw std::runtime_error("residual_rms_gamma: residual must be [M, N] (M = a.rows(), N = b.rows())");
    if (g.partials.cols() != M || g.partials.rows() != N / REG_BLOCK_N)
        throw std::runtime_error("residual_rms_gamma: partials must be [N/64, M] fp32");
    if (g.gamma.rows() != 1 || g.gamma.cols() != N)
        throw std::runtime_error("residual_rms_gamma: gamma must be [N] bf16");
    if (g.hg.rows() != M || g.hg.cols() != N)
        throw std::runtime_error("residual_rms_gamma: hg must be [M, N] bf16");
    launch<ResidualRMSGammaEpilogue, ResidualRMSGammaGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + residual + partial RMS + producer-side gamma: h = a @ b.T + residual stored in c, "
              "partials[g, m] = sum of h[m, cols of group g]^2 (unscaled h), hg = h * gamma[None, :]. "
              "a = [M,K], b = [N,K], c, residual and hg = [M,N] (bf16), partials = [N/64, M] (fp32), gamma = [N] (bf16)";
    bind_checked<dispatch>(m, "dispatch",
        {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32,
         tensor_dtype::bf16, tensor_dtype::bf16},
        &ResidualRMSGammaGlobals::a, &ResidualRMSGammaGlobals::b, &ResidualRMSGammaGlobals::c,
        &ResidualRMSGammaGlobals::residual, &ResidualRMSGammaGlobals::partials,
        &ResidualRMSGammaGlobals::gamma, &ResidualRMSGammaGlobals::hg);
}
