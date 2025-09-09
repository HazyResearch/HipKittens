import os
import sys
import torch
import tiktoken
from contextlib import nullcontext

# --- paths & imports ---
sys.path.append('/workdir/AMD-benchmarking-harness/training')
from src.model import GPTConfig, GPT

# --- settings ---
device = 'cuda'
fpath = '/workdir/AMD-benchmarking-harness/training/ref_non_causal_ckpts/'
filename = 'ckpt.pt'
MASK_ID = 50303 

# --- load checkpoint & model ---
ckpt_path = os.path.join(fpath, filename)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# strip unwanted prefix (e.g., from DDP/compile)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device).eval().to(torch.float32)

# --- tokenizer ---
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# --- dtype & autocast ---
dtype = torch.float32
ptdtype = torch.float32
ctx = nullcontext() if device != 'cuda' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

@torch.no_grad()
def simple_prefix_decode(sentence: str, steps: int = 20, temperature: float = 1.0) -> str:
    """
    Re-encode from scratch each step, append one MASK, sample the masked token, repeat.
    No filtering, no repetition penalty.
    """
    text = sentence
    for _ in range(steps):
        # 1) encode from scratch
        ids = encode(text)
        seq = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # [1, T]

        # 2) append a single MASK
        seq = torch.cat([seq, torch.tensor([[MASK_ID]], device=device)], dim=1)  # [1, T+1]
        mask_pos = seq.shape[1] - 1

        # 3) force full logits by passing dummy targets
        targets = torch.full_like(seq, -100)
        with ctx:
            logits, _ = model(seq, targets)  # logits: [1, T+1, V]

        # 4) sample next token for the masked position (temperature only)
        EOT_ID = 50256  # gpt2 eot
        row = logits[0, mask_pos] / max(temperature, 1e-6)
        row[MASK_ID] = -float('inf')
        row[EOT_ID]  = -float('inf')
        probs = torch.softmax(row, dim=-1)
        token = torch.multinomial(probs, num_samples=1)

        # 5) fill the mask and decode to a fresh string for the next iteration
        seq[0, mask_pos] = token
        text = decode(seq[0].tolist())

    return text

# --- run ---
if __name__ == "__main__":
    test_sentences = [
        "The house is nice and the garden is",
        "Machine learning is a subset of",
        "The weather today is quite",
        "Python is a popular",
        "The quick brown fox jumps over"
    ]
    for s in test_sentences:
        print(f"Prefix: {s!r}")
        for i in range(3):
            print("Sample", i+1, ":", simple_prefix_decode(s, steps=20, temperature=1.0))
        print("---------------")
