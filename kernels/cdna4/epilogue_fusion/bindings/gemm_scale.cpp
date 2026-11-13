#include "gemm_base.cuh"
#include "scale.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(ScaleGlobals g, hipStream_t stream) { launch<ScaleEpilogue, ScaleGlobals>(g, stream); }

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + scale: c = alpha * (a @ b.T), with a = [M,K], b = [N,K], c = [M,N] (bf16), "
              "alpha = 1-element fp32 CUDA tensor";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32},
        &ScaleGlobals::a, &ScaleGlobals::b, &ScaleGlobals::c, &ScaleGlobals::alpha);
}
