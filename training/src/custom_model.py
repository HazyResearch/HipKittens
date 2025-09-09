import torch
import torch.nn as nn
from torch.autograd import Function
import wandb
import os

import tk_mha_fwd
import tk_mha_bwd


class AttentionFunction(Function):
    def forward(ctx, q, k, v, is_causal):        
        
        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()

        o, l_vec = tk_mha_fwd(q, k, v, is_causal) 

        ctx.save_for_backward(q, k, v, o, l_vec)
        ctx.is_causal = is_causal
        return o.to(torch.float32)

    def backward(ctx, grad_o):        
        q, k, v, o, l_vec = ctx.saved_tensors
        is_causal = ctx.is_causal

        l_vec = l_vec.contiguous()
        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()
        o = o.to(torch.bfloat16).contiguous()
        grad_o = grad_o.to(torch.bfloat16).contiguous()

        grad_q, grad_k, grad_v = tk_mha_bwd(
            q, k, v, o, 
            l_vec, 
            grad_o, is_causal
        )   

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


class CustomAttention(nn.Module):
    def __init__(self, config):
        super(CustomAttention, self).__init__()

        # dimensions
        self.b = config.batch_size
        self.h = config.n_head
        self.n = config.block_size
        self.d = config.n_embd
        self.is_causal = config.causal 

        print(f"Using CustomAttention -- Causal = {self.is_causal}")

        # weights
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)


    def forward(self, x):
        B, T, C = x.size() 
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d, dim=2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2).contiguous() # (B, nh, T, hs)

        output = AttentionFunction.apply( q, k, v, self.is_causal )

        y = output.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y