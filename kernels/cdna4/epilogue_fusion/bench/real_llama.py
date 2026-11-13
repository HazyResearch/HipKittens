"""One decoder layer of a real Hugging Face Llama checkpoint, in the layout bench_layer.Layer uses.

HF Llama differs from this benchmark's layer in three ways, all undone here, once, for every arm:
  1. Wq, Wk rows are permuted for HF's rotate_half RoPE; we unpermute them to interleaved (2k, 2k+1) pairs.
  2. HK's RoPE computes (x cos + y sin, y cos - x sin); HF rotates the other way, so the table stores -sin.
  3. Frequencies come from the checkpoint's own rotary embedding (rope_theta, rope_scaling), not base 10000.
The HF layer output is captured from HF's own forward, so a mistake in 1-3 shows up as a mismatch with HF.

Input: real text (the repository's Markdown files, in sorted order, repeated if short) cut to M tokens.
"""
import torch

import common


def unpermute(w, heads):
    """Undo HF's q/k row permutation: per head, rows [first halves | second halves] -> interleaved pairs."""
    n, k = w.shape
    return w.view(heads, 2, n // heads // 2, k).transpose(1, 2).reshape(n, k).contiguous()


def text_ids(tokenizer, M):
    """[1, M] token ids of real text."""
    text = "\n\n".join(p.read_text(errors="ignore") for p in sorted(common.REPO.rglob("*.md")))
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) < M:
        ids = ids.repeat(M // len(ids) + 1)
    return ids[:M].unsqueeze(0).cuda()


class RealLlama:
    """Loads the checkpoint once; layer(M) returns one layer's weights, inputs and HF output for M tokens."""

    def __init__(self, path, layer, cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=common.BF16, attn_implementation="eager")
        self.model = self.model.cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        c = self.model.config
        want = dict(d=c.hidden_size, d_ff=c.intermediate_size, H=c.num_attention_heads,
                    H_KV=c.num_key_value_heads, Dh=c.hidden_size // c.num_attention_heads)
        if any(cfg[k] != v for k, v in want.items()) or c.rms_norm_eps != common.EPS:
            raise SystemExit(f"--config widths {cfg} do not match {path}: {want}, eps {c.rms_norm_eps}")
        if not 0 <= layer < c.num_hidden_layers:
            raise SystemExit(f"--layer must be in [0, {c.num_hidden_layers})")
        self.index, self.cfg = layer, cfg

    @torch.no_grad()
    def layer(self, M):
        cfg, block = self.cfg, self.model.model.layers[self.index]
        seen = {}

        def hook(_, args, kwargs, out):
            seen["x"] = (args[0] if args else kwargs["hidden_states"])[0]
            seen["out"] = (out[0] if isinstance(out, tuple) else out)[0]

        handle = block.register_forward_hook(hook, with_kwargs=True)
        try:
            ids = text_ids(self.tokenizer, M)
            self.model.model(input_ids=ids)
            positions = torch.arange(M, device="cuda")[None]
            cos, sin = self.model.model.rotary_emb(seen["x"][None], positions)   # [1, M, Dh], halves repeated
        finally:
            handle.remove()
        half = cfg["Dh"] // 2
        cs = torch.empty(M, cfg["Dh"], dtype=common.BF16, device="cuda")
        cs[:, 0::2], cs[:, 1::2] = cos[0, :, :half], -sin[0, :, :half]
        a, m = block.self_attn, block.mlp
        plain = lambda p: p.detach().contiguous()                  # HK bindings take torch.Tensor, not Parameter
        W = dict(q=unpermute(plain(a.q_proj.weight), cfg["H"]), k=unpermute(plain(a.k_proj.weight), cfg["H_KV"]),
                 v=plain(a.v_proj.weight), o=plain(a.o_proj.weight),
                 gu=torch.cat((m.gate_proj.weight, m.up_proj.weight)).detach().contiguous(),
                 dn=plain(m.down_proj.weight))
        return dict(W=W, g_attn=plain(block.input_layernorm.weight), g_mlp=plain(block.post_attention_layernorm.weight),
                    x=seen["x"].contiguous(), cs=cs, hf_out=seen["out"].contiguous())
