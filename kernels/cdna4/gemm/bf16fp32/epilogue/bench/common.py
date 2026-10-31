"""Shared bench helpers: configs, cold-cache timing, checks, extension loading, table / JSON output."""
import argparse
import dataclasses
import importlib.util
import json
import pathlib
import platform
import subprocess
import sys

import torch

BENCH = pathlib.Path(__file__).resolve().parent
EPI = BENCH.parent                 # hk.py and tk_<name> modules
BASE = EPI.parent                  # upstream base GEMM
REPO = EPI.parents[4]
if str(EPI) not in sys.path:
    sys.path.insert(0, str(EPI))

BF16, FP32 = torch.bfloat16, torch.float32
EPS = 1e-5                         # matches rms_reduce

# Widths from each model's Hugging Face config.json; H * Dh == d, no biases.
CONFIGS = {
    "llama3.2-1b": dict(d=2048, d_ff=8192, H=32, H_KV=8, Dh=64),     # Llama-3.2-1B
    "llama3.2-3b": dict(d=3072, d_ff=8192, H=24, H_KV=8, Dh=128),    # Llama-3.2-3B
    "llama3-8b": dict(d=4096, d_ff=14336, H=32, H_KV=8, Dh=128),    # Llama-3.1-8B
    "llama3-70b": dict(d=8192, d_ff=28672, H=64, H_KV=8, Dh=128),   # Llama-3.1-70B
    "llama3-405b": dict(d=16384, d_ff=53248, H=128, H_KV=8, Dh=128),  # Llama-3.1-405B
    "pr1": dict(d=4096, d_ff=11008, H=32, H_KV=8, Dh=128),          # Llama-2-7B widths with GQA; not a real model
}
M_VALUES = (2048, 4096, 8192)

COMPILE = dict(mode="max-autotune-no-cudagraphs", dynamic=False)

# Last-level (Infinity) cache size; torch reports only L2.
LLC_BYTES = {"gfx950": 256 * 2**20}

STAGE_TOL = dict(rtol=1e-2, atol=1e-1)     # bf16 [M, N] outputs
VECTOR_TOL = dict(rtol=1e-2, atol=1e-3)    # fp32 per-row inverse-RMS vectors
LAYER_REL = 2e-2                           # full layer, normwise
# Fusion-only control: its truncated bf16 GEMM output (2^-7) compounds to ~2^-6 in SwiGLU.
CONTROL_TOL = dict(rtol=2.5e-2, atol=1e-1)


def control_tols(tols):
    """STAGE_TOL -> CONTROL_TOL."""
    return tuple(CONTROL_TOL if t is STAGE_TOL else t for t in tols)


def layer_gemms(cfg):
    """name -> (K, N); weights are stored [N, K]."""
    d, d_ff, kv = cfg["d"], cfg["d_ff"], cfg["H_KV"] * cfg["Dh"]
    return {"Wq": (d, d), "Wk": (d, kv), "Wv": (d, kv), "Wo": (d, d), "Wgu": (d, 2 * d_ff), "Wd": (d_ff, d)}


# ---- inputs -------------------------------------------------------------------------------------

def activation(M, K):
    """bf16 [M, K] standard-normal activations."""
    return torch.randn(M, K, device="cuda").to(BF16)


def weight(K, N):
    """bf16 [N, K] weight scaled by 1/sqrt(K). Drawn [K, N] and transposed to keep the values reproducible."""
    return (torch.randn(K, N, device="cuda") * K ** -0.5).to(BF16).t().contiguous()


def gamma(n):
    """bf16 RMSNorm gain [n], near 1."""
    return (1 + 0.1 * torch.randn(n, device="cuda")).to(BF16)


def inv_rms(t):
    """fp32 [M] 1 / rms(t) per row."""
    return torch.rsqrt(t.float().pow(2).mean(-1) + EPS)


def rope_table(M, Dh, base=10000.0):
    """bf16 [M, Dh]: [m, 2k] = cos(m t_k), [m, 2k+1] = sin(m t_k)."""
    k = torch.arange(Dh // 2, device="cuda")
    angle = torch.arange(M, device="cuda").float()[:, None] * base ** (-2.0 * k / Dh)[None, :]
    table = torch.empty(M, Dh, device="cuda")
    table[:, 0::2], table[:, 1::2] = torch.cos(angle), torch.sin(angle)
    return table.to(BF16)


def rope(t, cos_sin):
    """(x, y) -> (x cos + y sin, y cos - x sin) on the last dim, in t's dtype."""
    x, y = t[..., 0::2], t[..., 1::2]
    c, s = cos_sin[..., 0::2], cos_sin[..., 1::2]
    return torch.stack((x * c + y * s, y * c - x * s), dim=-1).flatten(-2)


# ---- timing -------------------------------------------------------------------------------------

@dataclasses.dataclass
class Stats:
    median: float
    p10: float
    p90: float


class Timer:
    """Cold-cache GPU timing: flush 2x LLC before each call; events bracket only the call."""

    def __init__(self, warmup=10, iters=50):
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch not in LLC_BYTES:
            raise RuntimeError(f"no last-level-cache size for {arch}; add it to LLC_BYTES in common.py")
        self.llc_bytes = LLC_BYTES[arch]
        self.flush_bytes = 2 * self.llc_bytes
        self._flush = torch.empty(self.flush_bytes // 4, dtype=torch.int32, device="cuda")
        self.warmup, self.iters = warmup, iters

    def __call__(self, fn, cold=True):
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        start = [torch.cuda.Event(enable_timing=True) for _ in range(self.iters)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(self.iters)]
        for i in range(self.iters):
            if cold:
                self._flush.zero_()
            start[i].record()
            fn()
            end[i].record()
        torch.cuda.synchronize()
        ms = sorted(s.elapsed_time(e) for s, e in zip(start, end))
        pick = lambda q: ms[round(q * (len(ms) - 1))]
        return Stats(pick(0.5), pick(0.1), pick(0.9))


def ms_fields(arm, stats):
    """<arm>_ms (median), <arm>_p10, <arm>_p90."""
    return {f"{arm}_ms": stats.median, f"{arm}_p10": stats.p10, f"{arm}_p90": stats.p90}


# ---- correctness --------------------------------------------------------------------------------

def as_tuple(t):
    return t if isinstance(t, tuple) else (t,)


def compare(got, want, tol):
    """(ok, normwise rel error); ok iff finite and within atol + rtol * |want|."""
    want = want.float()
    got = got.float().reshape(want.shape)
    rel = ((got - want).norm() / want.norm()).item()
    ok = bool(torch.isfinite(got).all()) and bool(((got - want).abs() <= tol["atol"] + tol["rtol"] * want.abs()).all())
    return ok, rel


def check(outputs, refs, tols):
    """(all ok, rel error of the first output)."""
    results = [compare(g, w, t) for g, w, t in zip(as_tuple(outputs), as_tuple(refs), tols)]
    return all(ok for ok, _ in results), results[0][1]


def layer_check(got, want):
    """(ok, worst rel): finite and normwise rel < LAYER_REL."""
    rels = [((g.float().reshape(w.shape) - w.float()).norm() / w.float().norm()).item() for g, w in zip(got, want)]
    ok = all(bool(torch.isfinite(g).all()) for g in got) and max(rels) < LAYER_REL
    return ok, max(rels)


# ---- extensions ---------------------------------------------------------------------------------

def require_epilogue_modules(names):
    """Exit with build commands if any tk_<name> is missing."""
    missing = [n for n in names if importlib.util.find_spec(f"tk_{n}") is None]
    if missing:
        raise SystemExit(f"missing epilogue modules; from {EPI} run:\n"
                         + "\n".join(f"  make KERNEL={n}" for n in missing))


def find_extension(directory, name, build_hint):
    hits = sorted(pathlib.Path(directory).glob(f"{name}*.so"))
    if not hits:
        raise SystemExit(f"{name} is not built in {directory}; build it with:\n  {build_hint}")
    return hits[0]


def load_extension(path, name="tk_kernel"):
    """Load a module from a .so path (several builds share the name tk_kernel)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- output -------------------------------------------------------------------------------------

def parse_args(description, add=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="llama3-8b", choices=sorted(CONFIGS))
    ap.add_argument("--M", default=",".join(map(str, M_VALUES)), help="comma-separated sequence lengths")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json", default=None, help="also write rows and provenance to this path")
    ap.add_argument("--markdown", action="store_true", help="print tables as Markdown")
    if add:
        add(ap)
    args = ap.parse_args()
    args.M = [int(m) for m in args.M.split(",")]
    args.cfg = CONFIGS[args.config]
    return args


def print_table(title, columns, rows, markdown=False):
    """columns: [(header, key, format)]; missing values print '-'."""
    head = [h for h, _, _ in columns]
    body = [["-" if r.get(k) is None else format(r[k], f) for _, k, f in columns] for r in rows]
    print(f"\n{title}")
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for line in body:
            print("| " + " | ".join(line) + " |")
        return
    width = [max(len(c) for c in col) for col in zip(head, *body)]
    for line in [head, *body]:
        print("  ".join(c.rjust(w) for c, w in zip(line, width)))


def provenance(config):
    p = torch.cuda.get_device_properties(0)
    stamp = REPO / "SOURCE_COMMIT"
    if stamp.exists():
        commit = stamp.read_text().strip()
    else:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip() or "unknown"
    return {"host": platform.node(), "gpu": p.name, "arch": p.gcnArchName, "gpu_uuid": str(getattr(p, "uuid", "unknown")),
            "hip": torch.version.hip, "torch": torch.__version__, "commit": commit, "config": config}


def write_json(path, script, config, rows, extra=None):
    doc = {"script": script, "provenance": provenance(config), "rows": rows, **(extra or {})}
    pathlib.Path(path).write_text(json.dumps(doc, indent=2))


def exit_code(rows):
    """0 all PASS, 2 worst RERUN, 1 any FAIL."""
    statuses = {r["status"] for r in rows}
    return 1 if "FAIL" in statuses else 2 if "RERUN" in statuses else 0
