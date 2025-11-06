import torch
import torch.nn as nn
from torch.autograd import Function

import llama.models.attentions.tk_fwd_causal_kernel as tk_fwd_causal_kernel
import llama.models.attentions.tk_kernel_causal_bkwd as tk_kernel_causal_bkwd

class HipAttnFunction(Function):
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

        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Tensors must be contiguous after dtype conversion"

        O = torch.empty((B, N, H, D), dtype=torch.bfloat16, device=dev).contiguous()  
        L = torch.empty((B, H, 1, N), dtype=torch.float32,  device=dev).contiguous()    

        # Validate output tensor allocation
        assert O.is_contiguous() and L.is_contiguous(), "Output tensors must be contiguous"
        assert O.dtype == torch.bfloat16 and L.dtype == torch.float32, "Output tensor dtypes incorrect"

        # Safely dispatch forward kernel with error handling
        tk_fwd_causal_kernel.dispatch_fwd(q, k, v, O, L)

        if O.isnan().any():
            print("O is nan")
            breakpoint()
        if L.isnan().any():
            print("L is nan")
            breakpoint()

        ctx.save_for_backward(q, k, v, O, L)
        return O.to(out_dtype)


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
        tk_kernel_causal_bkwd.dispatch_prep(O, dO, delta)
        tk_kernel_causal_bkwd.dispatch_bwd_combined(q, k, v, O, dO, dQ_in, dK, dV, L, delta)
        tk_kernel_causal_bkwd.dispatch_dq_shuffle(dQ_in, dQ)

        # Final validation before returning
        assert dQ.shape == (B, N, H, D), f"Final dQ shape mismatch: expected ({B}, {N}, {H}, {D}), got {dQ.shape}"
        assert dK.shape == (B, N, HKV, D), f"Final dK shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {dK.shape}"
        assert dV.shape == (B, N, HKV, D), f"Final dV shape mismatch: expected ({B}, {N}, {HKV}, {D}), got {dV.shape}"

        return dQ.to(dO_bnhd.dtype), dK.to(dO_bnhd.dtype), dV.to(dO_bnhd.dtype)



class HipSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        deterministic=False,
    ):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        out_bnhd = HipAttnFunction.apply(q, k, v)  
        ctx = out_bnhd.to(q.dtype).contiguous()
        return ctx
        

class HipCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, deterministic=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.deterministic = deterministic

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        k, v = kv.unbind(dim=2)
        out_bnhd = HipAttnFunction.apply(q, k, v)  
        ctx = out_bnhd.to(q.dtype).contiguous()
        return ctx

