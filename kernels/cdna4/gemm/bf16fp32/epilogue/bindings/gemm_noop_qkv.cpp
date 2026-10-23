#include "gemm_base.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

// The noop GEMM over Q, K and V as ONE launch with three separate weights (no concatenated copy): output block
// columns [0, q_blocks) read Wq, [q_blocks, qk_blocks) read Wk, the rest Wv (gemm_kernel picks the B tensor per
// block). Used by the fusion-only control, so its QKV stage runs the same three-weight GEMM as hk.qkv.
struct NoopQkvGlobals {
    _gl_A a; _gl_B b; _gl_C c;          // b = Wq [Nq, K], c = [M, Nq + 2 Nkv]
    _gl_B b_k, b_v;                     // Wk, Wv [Nkv, K]
    int q_blocks, qk_blocks;            // 256-column block boundaries, set by launch()
};

void dispatch(NoopQkvGlobals g, hipStream_t stream) {
    launch<NoOpEpilogue, NoopQkvGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM over [Wq; Wk; Wv] without concatenating them: c = a @ [Wq; Wk; Wv].T; a = [M,K], wq = [Nq,K], "
              "wk = wv = [Nkv,K], c = [M, Nq + 2 Nkv] (bf16)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16,
                                           tensor_dtype::bf16, tensor_dtype::bf16},
        &NoopQkvGlobals::a, &NoopQkvGlobals::b, &NoopQkvGlobals::c, &NoopQkvGlobals::b_k, &NoopQkvGlobals::b_v);
}
