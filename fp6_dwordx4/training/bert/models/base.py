import torch
import torch.nn as nn
from typing import Optional
import math

class BertSelfAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.layer_idx = layer_idx

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads "
                f"({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads                      # H (query heads)
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)  # HKV
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_key_value_heads}) for GQA."
            )
        self.attention_head_size = config.hidden_size // self.num_attention_heads  # D
        self.all_head_size = self.num_attention_heads * self.attention_head_size   # H*D

        # Q projects to H*D; K/V project to HKV*D for GQA
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = hidden_states.shape
        H   = self.num_attention_heads
        HKV = self.num_key_value_heads
        D   = self.attention_head_size
        group_size = H // HKV

        q = self.query(hidden_states).view(B, N, H, D).transpose(1, 2)  # [B, H, N, D]
        k = self.key(hidden_states).view(B, N, HKV, D).transpose(1, 2)  # [B, HKV, N, D]
        v = self.value(hidden_states).view(B, N, HKV, D).transpose(1, 2)  # [B, HKV, N, D]

        if HKV != H:
            k = k.repeat_interleave(group_size, dim=1).contiguous()  # [B, H, N, D]
            v = v.repeat_interleave(group_size, dim=1).contiguous()  # [B, H, N, D]
        else:
            pass  # k,v already [B, H, N, D]

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        return context, attn_probs

        