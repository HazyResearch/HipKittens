#include <cstdint>
#include "gemm_base.cuh"
#include "rmsnorm_rope.cuh"
#include "rope_args.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

// "rmsnorm_rope" over Q, K and V as ONE launch with three separate weights (no concatenated copy): output block
// columns [0, q_blocks) read Wq, [q_blocks, qk_blocks) read Wk, the rest Wv. gemm_kernel picks the B tensor per
// block (gemm_base.cuh); the epilogue is RmsnormRopeEpilogue unchanged. q_blocks / qk_blocks are set by launch().
struct RmsnormRopeQkvGlobals {
    _gl_A a; _gl_B b; _gl_C c;          // b = Wq [Nq, K]
    gl<float, -1, -1, -1, -1> r;        // [M] fp32
    gl<bf16, -1, -1, -1, -1> cos_sin;   // [M, Dh] bf16, one head's natural table
    int rope_cols;                      // columns [0, rope_cols) are rotated (Q and K), the rest (V) stored as is
    _gl_B b_k, b_v;                     // Wk, Wv [Nkv, K]
    int q_blocks, qk_blocks;            // 256-column block boundaries, set by launch()
};

void dispatch(RmsnormRopeQkvGlobals g, hipStream_t stream) {
    const int N = g.b.rows() + g.b_k.rows() + g.b_v.rows();
    if (g.r.rows() != 1 || g.r.cols() != g.a.rows())
        throw std::runtime_error("rmsnorm_rope_qkv: r must be [M] (M = a.rows())");
    check_rope_args("rmsnorm_rope_qkv", g, N, {g.b.rows(), g.b_k.rows()});
    launch<RmsnormRopeEpilogue, RmsnormRopeQkvGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RMSNorm + RoPE over [Wq; Wk; Wv] without concatenating them: c = RoPE(r[:, None] * (a @ [Wq; Wk; Wv].T)) "
              "on columns [0, rope_cols); a = [M,K] the RMSNorm input times its gamma (hg from residual_rms_gamma), "
              "wq = [Nq,K], wk = wv = [Nkv,K] plain weights, c = [M, Nq + 2 Nkv]";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32,
                                           tensor_dtype::bf16, tensor_dtype::int_scalar, tensor_dtype::bf16, tensor_dtype::bf16},
        &RmsnormRopeQkvGlobals::a, &RmsnormRopeQkvGlobals::b, &RmsnormRopeQkvGlobals::c,
        &RmsnormRopeQkvGlobals::r, &RmsnormRopeQkvGlobals::cos_sin, &RmsnormRopeQkvGlobals::rope_cols,
        &RmsnormRopeQkvGlobals::b_k, &RmsnormRopeQkvGlobals::b_v);
}
