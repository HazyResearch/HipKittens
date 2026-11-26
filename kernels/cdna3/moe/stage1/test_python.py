import statistics

import torch
import aiter
from aiter.fused_moe_bf16_asm import moe_sorting_ck
from aiter.ops.shuffle import shuffle_weight
import tk_kernel

torch.manual_seed(0)

perf_benchmark = True
inter_dim = 2048
model_dim = 7168
topk = 8

num_tokens = 256
num_experts = 256
block_m = 32
block_n = 64
WEIGHT_SWIZZLE_GRANULARITY = block_n // 2
fp8 = torch.float8_e4m3fnuz


def interleave_gate_up(w_gate: torch.Tensor, w_up: torch.Tensor, block_size: int) -> torch.Tensor:
    assert w_gate.shape == w_up.shape, "gate and up must have the same shape"
    num_experts, inter_dim, model_dim = w_gate.shape
    assert inter_dim % block_size == 0, "inter_dim must be divisible by block_size"

    num_blocks = inter_dim // block_size
    # Reshape into blocks: (num_experts, num_blocks, block_size, model_dim)
    gate_blocks = w_gate.view(num_experts, num_blocks, block_size, model_dim)
    up_blocks = w_up.view(num_experts, num_blocks, block_size, model_dim)
    # Stack gate/up along a new axis right after num_blocks:
    # (num_experts, num_blocks, 2, block_size, model_dim)
    interleaved = torch.stack((gate_blocks, up_blocks), dim=2)
    # Flatten (num_blocks, 2, block_size) -> (2 * inter_dim)
    interleaved = interleaved.reshape(num_experts, 2 * inter_dim, model_dim)
    return interleaved


def moe_stage1_reference(
    hidden_states_fp8: torch.Tensor,  # [M, model_dim] fp8
    a1_scale: torch.Tensor,           # [M] float32, per-token activation scale
    w1_gate_fp8: torch.Tensor,        # [E, inter_dim, model_dim] fp8
    w1_up_fp8: torch.Tensor,          # [E, inter_dim, model_dim] fp8
    w1_scale: torch.Tensor,           # [E, 2*inter_dim] float32, per-channel weight scale
    topk_ids: torch.Tensor,           # [M, topk] int
) -> torch.Tensor:
    """Correctness-only reference for the a8w8 per-token/per-channel quantized
    fused MoE stage-1 kernel w/ SwiGLU epilogue.

    fp8 x fp8 matmul with fp32 accumulation,
    per-token row dequant and per-channel column dequant, then silu(gate) * up.
    Weight scale layout matches sf_B: gate = w1_scale[:, :inter_dim],
    up = w1_scale[:, inter_dim:]. Output rows are token_id * topk + slot.
    """
    M = hidden_states_fp8.shape[0]
    E, inter_dim, _ = w1_gate_fp8.shape
    topk = topk_ids.shape[1]

    # fp8 values are exactly representable in fp32, so upcasting reproduces the
    # MFMA fp8 x fp8 -> fp32 products and fp32 accumulation.
    x = hidden_states_fp8.float()                      # [M, D]
    wg = w1_gate_fp8.float()                            # [E, I, D]
    wu = w1_up_fp8.float()                              # [E, I, D]

    gate_scale = w1_scale[:, :inter_dim]               # [E, I]
    up_scale = w1_scale[:, inter_dim:]                 # [E, I]

    e = topk_ids.reshape(-1).long()                    # [M*topk]
    xt = x.repeat_interleave(topk, dim=0)              # [M*topk, D]
    a = a1_scale.repeat_interleave(topk, dim=0).unsqueeze(1)  # [M*topk, 1]

    gate_raw = torch.einsum("nd,nid->ni", xt, wg[e])   # [M*topk, I]
    up_raw = torch.einsum("nd,nid->ni", xt, wu[e])     # [M*topk, I]

    gate = gate_raw * a * gate_scale[e]
    up = up_raw * a * up_scale[e]
    act = torch.nn.functional.silu(gate) * up
    return act.to(torch.bfloat16)


hidden_states = torch.randn(num_tokens, model_dim, dtype=torch.bfloat16, device="cuda")
w1_gate = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.bfloat16, device="cuda")
w1_up = torch.randn(num_experts, inter_dim, model_dim, dtype=torch.bfloat16, device="cuda")
w2 = torch.randn(num_experts, model_dim, inter_dim, dtype=torch.bfloat16, device="cuda")
router_logits = torch.randn(num_tokens, num_experts, device="cuda")

hidden_states_fp8 = hidden_states.to(fp8)
w1_gate_fp8 = w1_gate.to(fp8)
w1_up_fp8 = w1_up.to(fp8)
w1_fp8 = torch.concat((w1_gate_fp8, w1_up_fp8), dim=1)
# CK reuqires B in MFMA-shuffled layout
w1_fp8_aiter = shuffle_weight(w1_fp8, layout=(16,32))
w2_fp8 = w2.to(fp8)

topk_weights, topk_ids = torch.topk(router_logits.softmax(dim=-1), k=topk, dim=-1)
topk_ids = topk_ids.to(torch.int32)

sorted_ids, _sorted_weights, sorted_expert_ids, num_valid_ids, _moe_buf = (
    moe_sorting_ck(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        torch.bfloat16,
        block_size=block_m,
        expert_mask=None,
    )
)
torch.cuda.synchronize()

out_ref = torch.empty((num_tokens * topk, inter_dim), dtype=torch.bfloat16, device="cuda")
out_test = torch.empty((num_tokens * topk, inter_dim), dtype=torch.bfloat16, device="cuda")

a1_scale = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")
w1_scale = torch.rand(num_experts, 1, inter_dim * 2, dtype=torch.float32, device="cuda")
a1_scale_1d = a1_scale.reshape((num_tokens))
w1_scale_2d = w1_scale.reshape((num_experts, inter_dim * 2))

interleaved = interleave_gate_up(w1_gate_fp8, w1_up_fp8, WEIGHT_SWIZZLE_GRANULARITY)

tk_kernel.call(
    hidden_states_fp8,
    a1_scale_1d,
    interleaved,
    w1_scale_2d,
    out_test,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids
)
torch.cuda.synchronize()

aiter.ck_moe_stage1_fwd(
    hidden_states=hidden_states_fp8,
    w1=w1_fp8_aiter,
    w2=w2_fp8,
    sorted_token_ids=sorted_ids,
    sorted_expert_ids=sorted_expert_ids,
    num_valid_ids=num_valid_ids,
    out=out_ref,
    topk=topk,
    kernelName="",
    w1_scale=w1_scale,
    a1_scale=a1_scale,
    block_m=32,
    sorted_weights=None,
    quant_type=aiter.QuantType.per_Token,
    activation=aiter.ActivationType.Silu
)
torch.cuda.synchronize()

# out_torch = moe_stage1_reference(
#     hidden_states_fp8,
#     a1_scale.reshape((num_tokens)),
#     w1_gate_fp8,
#     w1_up_fp8,
#     w1_scale.reshape((num_experts, inter_dim * 2)),
#     topk_ids,
# )

# print("out_test:\n", out_test)
# print("out_ref:\n", out_ref)
# print("out_torch:\n", out_torch)
max_abs_err = (out_test.float() - out_ref.float()).abs().max().item()
print("max abs err:", max_abs_err)
print("allclose:", torch.allclose(out_test.float(), out_ref.float(), atol=1e-2, rtol=1e-2))

if perf_benchmark:
    def run_tk():
        tk_kernel.call(
            hidden_states_fp8,
            a1_scale_1d,
            interleaved,
            w1_scale_2d,
            out_test,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids
        )

    def run_aiter():
        aiter.ck_moe_stage1_fwd(
            hidden_states=hidden_states_fp8,
            w1=w1_fp8_aiter,
            w2=w2_fp8,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=out_ref,
            topk=topk,
            kernelName="",
            w1_scale=w1_scale,
            a1_scale=a1_scale,
            block_m=32,
            sorted_weights=None,
            quant_type=aiter.QuantType.per_Token,
            activation=aiter.ActivationType.Silu
        )

    def time_once(fn, num_warmup=5, num_iters=100):
        for _ in range(num_warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters

    def report(name, times):
        print(
            f"{name} perf. (ms): "
            f"min={min(times):.5f} "
            f"median={statistics.median(times):.5f} "
            f"mean={statistics.fmean(times):.5f} "
            f"std={statistics.pstdev(times) if len(times) > 1 else 0.0:.5f} "
            f"(n={len(times)})"
        )

    num_trials = 20
    time_once(run_tk, num_warmup=10, num_iters=1)
    time_once(run_aiter, num_warmup=10, num_iters=1)

    tk_times, aiter_times = [], []
    for t in range(num_trials):
        if t % 2 == 0:
            tk_times.append(time_once(run_tk))
            aiter_times.append(time_once(run_aiter))
        else:
            aiter_times.append(time_once(run_aiter))
            tk_times.append(time_once(run_tk))

    report("TK", tk_times)
    report("AITER", aiter_times)
