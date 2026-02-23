#!/usr/bin/env python3
"""
Test harness for CDNA3 Flash Attention Forward (Non-Causal).
Compares kernel output against PyTorch's scaled_dot_product_attention.
"""

import torch
import argparse
import sys
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Test CDNA3 Flash Attention Forward")
    parser.add_argument("--B", type=int, default=4, help="Batch size")
    parser.add_argument("--H", type=int, default=32, help="Number of Q heads")
    parser.add_argument("--H_KV", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--N", type=int, default=1024, help="Sequence length")
    parser.add_argument("--D", type=int, default=128, help="Head dimension (64 or 128)")
    parser.add_argument("--profile", action="store_true", help="Run profiling iterations")
    parser.add_argument("--num_warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num_iters", type=int, default=20, help="Timing iterations")
    return parser.parse_args()

def reference_attention(Q, K, V, is_causal=False):
    """PyTorch reference: scaled_dot_product_attention."""
    # Q: [B, H, N, D], K: [B, H_KV, N, D], V: [B, H_KV, N, D]
    B, H, N, D = Q.shape
    H_KV = K.shape[1]
    group_size = H // H_KV

    # Expand KV heads for GQA
    K_expanded = K.repeat_interleave(group_size, dim=1)  # [B, H, N, D]
    V_expanded = V.repeat_interleave(group_size, dim=1)  # [B, H, N, D]

    return torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_expanded.float(), V_expanded.float(), is_causal=is_causal
    ).to(torch.bfloat16)

def test_attention(args):
    B, H, H_KV, N, D = args.B, args.H, args.H_KV, args.N, args.D

    # Kernel tile alignment: Q_ROWS=16, NUM_WARPS=4 → 64 rows per block
    # Store uses raw pointers (no OOB protection), so pad output tensors
    BLOCK_SIZE = 64  # Q_ROWS * NUM_WARPS
    N_pad = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    print(f"Testing Flash Attention Forward (Non-Causal)")
    print(f"  B={B}, H={H}, H_KV={H_KV}, N={N}, D={D}" + (f" (padded to {N_pad})" if N_pad != N else ""))
    print(f"  GQA ratio: {H // H_KV}")
    print()

    # Import the appropriate kernel module
    if D == 128:
        import attn_fwd as kernel_mod
    elif D == 64:
        import attn_fwd_d64 as kernel_mod
    else:
        raise ValueError(f"Unsupported D={D}. Use 64 or 128.")

    torch.manual_seed(42)

    # Create inputs [B, H, N, D] in bf16 — padded to N_pad with zeros
    Q = torch.zeros(B, H, N_pad, D, dtype=torch.bfloat16, device='cuda')
    K = torch.zeros(B, H_KV, N_pad, D, dtype=torch.bfloat16, device='cuda')
    V = torch.zeros(B, H_KV, N_pad, D, dtype=torch.bfloat16, device='cuda')
    Q[:, :, :N, :] = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda') / (D ** 0.25)
    K[:, :, :N, :] = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device='cuda') / (D ** 0.25)
    V[:, :, :N, :] = torch.randn(B, H_KV, N, D, dtype=torch.bfloat16, device='cuda') / (D ** 0.25)

    # Output buffers (padded)
    O = torch.zeros(B, H, N_pad, D, dtype=torch.bfloat16, device='cuda')
    L = torch.zeros(B, H, N_pad, 1, dtype=torch.float32, device='cuda')

    # Reference output (use only valid rows)
    O_ref = reference_attention(Q[:, :, :N, :], K[:, :, :N, :], V[:, :, :N, :], is_causal=False)

    # Run kernel
    kernel_mod.attention(Q, K, V, O, L, N)
    torch.cuda.synchronize()

    # Compare only valid rows
    O_float = O[:, :, :N, :].float()
    O_ref_float = O_ref.float()

    diff = (O_float - O_ref_float).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()

    # Cosine similarity (per-sample)
    cos_sim = torch.nn.functional.cosine_similarity(
        O_float.reshape(B, -1), O_ref_float.reshape(B, -1), dim=1
    ).mean().item()

    print(f"=== Correctness ===")
    print(f"  Max error:  {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Cosine sim: {cos_sim:.6f}")

    passed = cos_sim > 0.999
    print(f"  Status:     {'PASS' if passed else 'FAIL'} (threshold: cosine_sim > 0.999)")
    print()

    # Profiling
    if args.profile:
        # Warmup
        for _ in range(args.num_warmup):
            kernel_mod.attention(Q, K, V, O, L, N)
        torch.cuda.synchronize()

        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        timings = []
        for _ in range(args.num_iters):
            torch.cuda.synchronize()
            start_event.record()
            kernel_mod.attention(Q, K, V, O, L, N)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))

        avg_time = sum(timings) / len(timings)
        # FLOPs: 2 * B * H * N * N * D (QK) + 2 * B * H * N * N * D (PV) = 4 * B * H * N^2 * D
        flops = 4 * B * H * N * N * D
        tflops = flops / (avg_time * 1e9)

        print(f"=== Performance ===")
        print(f"  Avg time:   {avg_time:.4f} ms")
        print(f"  TFLOPS:     {tflops:.2f}")
        print()

        # Reference timing
        timings_ref = []
        for _ in range(args.num_warmup):
            _ = reference_attention(Q, K, V)
        for _ in range(args.num_iters):
            torch.cuda.synchronize()
            start_event.record()
            O_ref = reference_attention(Q, K, V)
            end_event.record()
            torch.cuda.synchronize()
            timings_ref.append(start_event.elapsed_time(end_event))

        avg_time_ref = sum(timings_ref) / len(timings_ref)
        tflops_ref = flops / (avg_time_ref * 1e9)
        print(f"  PyTorch ref avg time: {avg_time_ref:.4f} ms")
        print(f"  PyTorch ref TFLOPS:   {tflops_ref:.2f}")
        print(f"  Speedup:              {avg_time_ref / avg_time:.2f}x")

    return passed

def run_shape_sweep():
    """Test multiple shapes for robustness."""
    shapes = [
        # (B, H, H_KV, N, D)
        (1, 8, 8, 128, 128),     # MHA, small
        (4, 32, 8, 1024, 128),   # GQA-4, medium
        (2, 16, 2, 2048, 128),   # GQA-8, large
        (1, 32, 4, 4096, 128),   # GQA-8, very large
    ]

    # Only test D=64 shapes if the module exists
    try:
        import attn_fwd_d64
        shapes += [
            (1, 8, 8, 128, 64),
            (4, 32, 8, 1024, 64),
        ]
    except ImportError:
        print("Note: attn_fwd_d64 not built, skipping D=64 tests")

    all_passed = True
    for B, H, H_KV, N, D in shapes:
        test_args = argparse.Namespace(B=B, H=H, H_KV=H_KV, N=N, D=D, profile=False, num_warmup=1, num_iters=1)
        try:
            passed = test_attention(test_args)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"  FAIL: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    args = parse_args()

    if args.B == 0:
        # Special mode: run shape sweep
        all_passed = run_shape_sweep()
        sys.exit(0 if all_passed else 1)
    else:
        passed = test_attention(args)
        sys.exit(0 if passed else 1)
