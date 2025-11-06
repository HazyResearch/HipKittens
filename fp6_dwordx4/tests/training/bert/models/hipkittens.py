import math
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.autograd import Function

import tk_kernel_fwd
import tk_kernel_bkwd


def segfault_handler(signum, frame):
    """Handle segmentation faults and memory access violations."""
    print(f"\n!!! MEMORY ACCESS FAULT DETECTED (signal {signum}) !!!")
    print(f"Frame: {frame}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    import traceback
    traceback.print_stack()
    breakpoint()

# Register signal handlers for memory faults
signal.signal(signal.SIGSEGV, segfault_handler)  # Segmentation fault
signal.signal(signal.SIGBUS, segfault_handler)   # Bus error



class HipKittensFlashAttnFn(Function):
    """
    Inputs/outputs are BNHD (batch, seq, heads, dim), like your harness.
    Forward:  O, L  via tk_kernel_fwd.dispatch_fwd
    Backward: dQ,dK,dV via tk_kernel_bkwd.{dispatch_prep,dispatch_bwd_combined,dispatch_dq_shuffle}
    Compute in bf16, save L and O for backward, return O in input dtype.
    """

    @staticmethod
    def forward(ctx, q_bnhd: torch.Tensor, k_bnhd: torch.Tensor, v_bnhd: torch.Tensor):
        B, N, H, D = q_bnhd.shape
        HKV = k_bnhd.shape[2]
        dev = q_bnhd.device
        out_dtype = q_bnhd.dtype  

        # Validate input tensor shapes
        assert q_bnhd.shape == (B, N, H, D), f"Q shape mismatch: expected ({B}, {N}, {H}, {D}), got {q_bnhd.shape}"
        assert k_bnhd.shape == (B, N, HKV, D), f"K shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {k_bnhd.shape}"
        assert v_bnhd.shape == (B, N, HKV, D), f"V shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {v_bnhd.shape}"
        
        # Validate tensor properties
        assert q_bnhd.is_cuda and k_bnhd.is_cuda and v_bnhd.is_cuda, "All tensors must be on CUDA device"
        assert q_bnhd.device == k_bnhd.device == v_bnhd.device, "All tensors must be on same device"
        
        # Validate GQA constraints
        assert H % HKV == 0, f"H ({H}) must be divisible by HKV ({HKV}) for GQA"
        assert HKV <= H, f"HKV ({HKV}) cannot exceed H ({H})"

        q = q_bnhd.to(torch.bfloat16).contiguous()
        k = k_bnhd.to(torch.bfloat16).contiguous()
        v = v_bnhd.to(torch.bfloat16).contiguous()

        # Validate contiguity after conversion
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Tensors must be contiguous after dtype conversion"

        O = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  
        L = torch.empty((B, H, 1, N), dtype=torch.float32,  device=dev).contiguous()    

        # Validate output tensor allocation
        assert O.is_contiguous() and L.is_contiguous(), "Output tensors must be contiguous"
        assert O.dtype == torch.bfloat16 and L.dtype == torch.float32, "Output tensor dtypes incorrect"

        # Safely dispatch forward kernel with error handling
        try:
            # print(f"Forward dispatch: q={q.shape}, k={k.shape}, v={v.shape}, O={O.shape}, L={L.shape}")
            torch.cuda.synchronize()  # Ensure GPU is ready
            tk_kernel_fwd.dispatch_fwd(q, k, v, O, L)
            torch.cuda.synchronize()  # Wait for kernel completion
        except RuntimeError as e:
            print(f"CUDA kernel error in forward dispatch: {e}")
            print(f"Tensor shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
            print(f"Tensor devices - Q: {q.device}, K: {k.device}, V: {v.device}")
            print(f"Tensor dtypes - Q: {q.dtype}, K: {k.dtype}, V: {v.dtype}")
            breakpoint()
            raise
        except Exception as e:
            print(f"Unexpected error in forward dispatch: {e}")
            breakpoint()
            raise

        if O.isnan().any():
            print("O is nan")
            breakpoint()
        if L.isnan().any():
            print("L is nan")
            breakpoint()

        ctx.save_for_backward(q, k, v, O, L)
        return O.to(out_dtype)

    # @staticmethod
    # def forward(ctx, q_bnhd: torch.Tensor, k_bnhd: torch.Tensor, v_bnhd: torch.Tensor):
    #     B, N, H, D = q_bnhd.shape
    #     HKV = k_bnhd.shape[2]
    #     dev = q_bnhd.device
    #     out_dtype = q_bnhd.dtype  

    #     # Validate input tensor shapes
    #     assert q_bnhd.shape == (B, N, H, D), f"Q shape mismatch: expected ({B}, {N}, {H}, {D}), got {q_bnhd.shape}"
    #     assert k_bnhd.shape == (B, N, HKV, D), f"K shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {k_bnhd.shape}"
    #     assert v_bnhd.shape == (B, N, HKV, D), f"V shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {v_bnhd.shape}"
        
    #     # Validate tensor properties
    #     assert q_bnhd.is_cuda and k_bnhd.is_cuda and v_bnhd.is_cuda, "All tensors must be on CUDA device"
    #     assert q_bnhd.device == k_bnhd.device == v_bnhd.device, "All tensors must be on same device"
        
    #     # Validate GQA constraints
    #     assert H % HKV == 0, f"H ({H}) must be divisible by HKV ({HKV}) for GQA"
    #     assert HKV <= H, f"HKV ({HKV}) cannot exceed H ({H})"

    #     q = q_bnhd.to(torch.float32).transpose(1, 2).contiguous()
    #     k = k_bnhd.to(torch.float32).transpose(1, 2).contiguous()
    #     v = v_bnhd.to(torch.float32).transpose(1, 2).contiguous()

    #     # Validate contiguity after conversion
    #     assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Tensors must be contiguous after dtype conversion"

    #     O = torch.empty((B, H, N, D), dtype=torch.bfloat16, device=dev).contiguous()  
    #     L = torch.empty((B, H, N, 1), dtype=torch.float32,  device=dev).contiguous()    

    #     # Validate output tensor allocation
    #     assert O.is_contiguous() and L.is_contiguous(), "Output tensors must be contiguous"
    #     assert O.dtype == torch.bfloat16 and L.dtype == torch.float32, "Output tensor dtypes incorrect"

    #     # Safely dispatch forward kernel with error handling
    #     try:
    #         # Expand K,V to match Q heads for GQA computation
    #         group_size = H // HKV
    #         # Repeat each KV head group_size times: (B, h_kv, N, D) -> (B, h_q, N, D)
    #         k_expanded = k.clone().repeat_interleave(group_size, dim=1)  
    #         v_expanded = v.clone().repeat_interleave(group_size, dim=1)

    #         # Manual pytorch implementation of scaled dot product attention
    #         QK = torch.matmul(q, k_expanded.transpose(-2, -1))
    #         QK /= (q.size(-1) ** 0.5)

    #         # Compute LSE before softmax
    #         L = torch.logsumexp(QK, dim=-1)  # (batch, heads, seq)

    #         QK = torch.nn.functional.softmax(QK, dim=-1)
    #         O = torch.matmul(QK, v_expanded).to(torch.bfloat16)

    #     except RuntimeError as e:
    #         print(f"CUDA kernel error in forward dispatch: {e}")
    #         print(f"Tensor shapes - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    #         print(f"Tensor devices - Q: {q.device}, K: {k.device}, V: {v.device}")
    #         print(f"Tensor dtypes - Q: {q.dtype}, K: {k.dtype}, V: {v.dtype}")
    #         breakpoint()
    #         raise
    #     except Exception as e:
    #         print(f"Unexpected error in forward dispatch: {e}")
    #         breakpoint()
    #         raise

    #     if O.isnan().any():
    #         print("O is nan")
    #         breakpoint()
    #     if L.isnan().any():
    #         print("L is nan")
    #         breakpoint()

    #     q = q.transpose(1, 2).contiguous()
    #     k = k.transpose(1, 2).contiguous()
    #     v = v.transpose(1, 2).contiguous()
    #     O = O.transpose(1, 2).contiguous()
    #     L = L.unsqueeze(-1).transpose(-1, -2).contiguous()

    #     ctx.save_for_backward(q, k, v, O, L)
    #     return O.to(out_dtype)

    @staticmethod
    def backward(ctx, dO_bnhd: torch.Tensor):
        q, k, v, O, L = ctx.saved_tensors
        # print(f"DEBUG backward - O.shape: {O.shape}, L.shape: {L.shape}")
        # print(f"DEBUG backward - q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        B, N, H, D = O.shape
        HKV = k.shape[2]
        dev = dO_bnhd.device

        # Validate saved tensors
        assert q.shape == (B, N, H, D), f"Saved Q shape mismatch: expected ({B}, {N}, {H}, {D}), got {q.shape}"
        assert k.shape == (B, N, HKV, D), f"Saved K shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {k.shape}"
        assert v.shape == (B, N, HKV, D), f"Saved V shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {v.shape}"
        assert O.shape == (B, N, H, D), f"Saved O shape mismatch: expected ({B}, {N}, {H}, {D}), got {O.shape}"
        assert L.shape == (B, H, 1, N), f"Saved L shape mismatch: expected ({B}, {H}, 1, {N}), got {L.shape}"
        
        # Validate gradient input
        assert dO_bnhd.shape == (B, N, H, D), f"dO shape mismatch: expected ({B}, {N}, {H}, {D}), got {dO_bnhd.shape}"
        assert dO_bnhd.is_cuda and dO_bnhd.device == dev, "dO must be on correct CUDA device"
        
        # Validate GQA constraints
        assert H % HKV == 0, f"H ({H}) must be divisible by HKV ({HKV}) for GQA"

        # Cast grad to bf16 for kernels
        dO = dO_bnhd.to(torch.bfloat16).contiguous()
        assert dO.is_contiguous(), "dO must be contiguous after conversion"

        # Allocate grads and workspaces
        dQ_in = torch.zeros((B, H, N, D), dtype=torch.bfloat16, device=dev).contiguous()  # BHND (pre-shuffle)
        dQ    = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHD
        dK    = torch.empty((B, N, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHKVD
        dV    = torch.empty((B, N, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()  # BNHKVD
        delta = torch.empty((B, H, 1, N), dtype=torch.float32,  device=dev).contiguous() 

        # Validate gradient tensor allocation
        assert all(t.is_contiguous() for t in [dQ_in, dQ, dK, dV, delta]), "All gradient tensors must be contiguous"
        assert dQ_in.dtype == torch.bfloat16 and dQ.dtype == torch.bfloat16, "dQ tensors must be bfloat16"
        assert dK.dtype == torch.bfloat16 and dV.dtype == torch.bfloat16, "dK, dV tensors must be bfloat16"
        assert delta.dtype == torch.float32, "delta tensor must be float32"

        if dO.isnan().any():
            print("dO is nan")
            breakpoint()

        # Backward kernels
        try:
            # print(f"Backward prep dispatch: O={O.shape}, dO={dO.shape}, delta={delta.shape}")
            torch.cuda.synchronize()
            tk_kernel_bkwd.dispatch_prep(O, dO, delta)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"CUDA kernel error in dispatch_prep: {e}")
            print(f"Tensor shapes - O: {O.shape}, dO: {dO.shape}, delta: {delta.shape}")
            breakpoint()
            raise
        except Exception as e:
            print(f"Unexpected error in dispatch_prep: {e}")
            breakpoint()
            raise
            
        if delta.isnan().any():
            print("delta is nan")
            breakpoint()

        # Validate tensors before dispatch_bwd_combined
        assert all(t.is_contiguous() for t in [q, k, v, O, dO, dQ_in, dK, dV, L, delta]), "All tensors must be contiguous before dispatch_bwd_combined"
        
        try:
            # print(f"Backward combined dispatch: q={q.shape}, k={k.shape}, v={v.shape}")
            # print(f"                          O={O.shape}, dO={dO.shape}, dQ_in={dQ_in.shape}")
            # print(f"                          dK={dK.shape}, dV={dV.shape}, L={L.shape}, delta={delta.shape}")
            torch.cuda.synchronize()
            tk_kernel_bkwd.dispatch_bwd_combined(q, k, v, O, dO, dQ_in, dK, dV, L, delta)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"CUDA kernel error in dispatch_bwd_combined: {e}")
            print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            breakpoint()
            raise
        except Exception as e:
            print(f"Unexpected error in dispatch_bwd_combined: {e}")
            breakpoint()
            raise
        if dQ_in.isnan().any():
            print("dQ_in is nan")
            breakpoint()
        if dK.isnan().any():
            print("dK is nan")
            breakpoint()
        if dV.isnan().any():
            print("dV is nan")
            breakpoint()

        # Validate tensors before dispatch_dq_shuffle
        assert dQ_in.is_contiguous() and dQ.is_contiguous(), "dQ_in and dQ must be contiguous before shuffle"
        assert dQ_in.shape == (B, H, N, D), f"dQ_in shape mismatch before shuffle: expected ({B}, {H}, {N}, {D}), got {dQ_in.shape}"
        assert dQ.shape == (B, N, H, D), f"dQ shape mismatch before shuffle: expected ({B}, {N}, {H}, {D}), got {dQ.shape}"
        
        try:
            # print(f"dQ shuffle dispatch: dQ_in={dQ_in.shape} -> dQ={dQ.shape}")
            torch.cuda.synchronize()
            tk_kernel_bkwd.dispatch_dq_shuffle(dQ_in, dQ)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f"CUDA kernel error in dispatch_dq_shuffle: {e}")
            print(f"Tensor shapes - dQ_in: {dQ_in.shape}, dQ: {dQ.shape}")
            print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            breakpoint()
            raise
        except Exception as e:
            print(f"Unexpected error in dispatch_dq_shuffle: {e}")
            breakpoint()
            raise
            
        if dQ.isnan().any():
            print("dQ is nan")
            breakpoint()

        # Final validation before returning
        assert dQ.shape == (B, N, H, D), f"Final dQ shape mismatch: expected ({B}, {N}, {H}, {D}), got {dQ.shape}"
        assert dK.shape == (B, N, HKV, D), f"Final dK shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {dK.shape}"
        assert dV.shape == (B, N, HKV, D), f"Final dV shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {dV.shape}"

        return dQ.to(dO_bnhd.dtype), dK.to(dO_bnhd.dtype), dV.to(dO_bnhd.dtype)


    # @staticmethod
    # def backward(ctx, dO_bnhd: torch.Tensor):
    #     # Saved tensors from forward
    #     q_bnhd, k_bnhkd, v_bnhkd, O_bnhd, L = ctx.saved_tensors
    #     B, N, H, D = q_bnhd.shape
    #     HKV = k_bnhkd.shape[2]
    #     assert H % HKV == 0
    #     G = H // HKV
    #     dev = dO_bnhd.device

    #     # Work in fp32 for stability; go to BHND for math
    #     q = q_bnhd.permute(0, 2, 1, 3).to(torch.float32).contiguous()        # [B,H,N,D]
    #     k = k_bnhkd.permute(0, 2, 1, 3).to(torch.float32).contiguous()        # [B,HKV,N,D]
    #     v = v_bnhkd.permute(0, 2, 1, 3).to(torch.float32).contiguous()        # [B,HKV,N,D]
    #     dO = dO_bnhd.permute(0, 2, 1, 3).to(torch.float32).contiguous()       # [B,H,N,D]

    #     # Expand KV to H heads (autograd sum is irrelevant here—we're in backward)
    #     k_rep = k.repeat_interleave(G, dim=1)                                 # [B,H,N,D]
    #     v_rep = v.repeat_interleave(G, dim=1)                                 # [B,H,N,D]

    #     # Recompute softmax probs
    #     scale = 1.0 / math.sqrt(D)
    #     S = torch.matmul(q, k_rep.transpose(-1, -2)) * scale                  # [B,H,N,N]
    #     P = torch.softmax(S, dim=-1)                                          # [B,H,N,N]

    #     # Forward context (for Delta term)
    #     O_rep = torch.matmul(P, v_rep)                                        # [B,H,N,D]

    #     # Backprop
    #     Delta = (dO * O_rep).sum(dim=-1, keepdim=True)                        # [B,H,N,1]
    #     dS = P * (torch.matmul(dO, v_rep.transpose(-1, -2)) - Delta)          # [B,H,N,N]

    #     dQ_bhnd = torch.matmul(dS, k_rep) * scale                             # [B,H,N,D]
    #     dK_rep  = torch.matmul(dS.transpose(-1, -2), q) * scale               # [B,H,N,D]
    #     dV_rep  = torch.matmul(P.transpose(-1, -2), dO)                       # [B,H,N,D]

    #     # Reduce H → HKV by summing groups
    #     dK_bhkvnd = dK_rep.view(B, HKV, G, N, D).sum(dim=2)                   # [B,HKV,N,D]
    #     dV_bhkvnd = dV_rep.view(B, HKV, G, N, D).sum(dim=2)                   # [B,HKV,N,D]

    #     # Return grads in BNHD/BNHKVD and original dtype
    #     dQ = dQ_bhnd.permute(0, 2, 1, 3).to(dO_bnhd.dtype).contiguous()       # [B,N,H,D]
    #     dK = dK_bhkvnd.permute(0, 2, 1, 3).to(dO_bnhd.dtype).contiguous()     # [B,N,HKV,D]
    #     dV = dV_bhkvnd.permute(0, 2, 1, 3).to(dO_bnhd.dtype).contiguous()     # [B,N,HKV,D]

    #     if dQ.isnan().any():
    #         print("dQ is nan")
    #         breakpoint()
    #     if dK.isnan().any():
    #         print("dK is nan")
    #         breakpoint()
    #     if dV.isnan().any():
    #         print("dV is nan")
    #         breakpoint()
    #     return dQ, dK, dV


class HipKittensBertSelfAttention(nn.Module):
    """
    Uses HipKittensFlashAttnFn when there is NO padding.
    Falls back to MHA-style expansion if num_key_value_heads < num_attention_heads (GQA).
    Expects HF additive mask: [B,1,1,N] (0 keep, -inf mask)
    """
    def __init__(self, config, layer_idx=None, deterministic: bool = False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError("hidden_size must be multiple of num_attention_heads")
        
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads                                        # h_q
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)  # h_kv
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        
        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size)
        self.key   = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.deterministic = deterministic
        self.is_causal = False

        print(f"HipKittens BertSelfAttention layer {layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H = self.num_attention_heads
        HKV = self.num_key_value_heads
        D = self.attention_head_size

        q = self.query(hidden_states).view(B, N, H, D).to(torch.bfloat16).contiguous()
        k = self.key(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()
        v = self.value(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()

        if q.isnan().any():
            print("q is nan")
            breakpoint()
        if k.isnan().any():
            print("k is nan")
            breakpoint()
        if v.isnan().any():
            print("v is nan")
            breakpoint()

        out_bnhd = HipKittensFlashAttnFn.apply(q, k, v)  # BNHD
        ctx = out_bnhd.to(q.dtype).contiguous().view(B, N, H * D)
        return ctx, None

