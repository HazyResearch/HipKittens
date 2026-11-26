import torch
import tk_kernel
import random
from utils import init_randn, init_empty

# N must be a multiple of the block tile (128) and K of K_STEP (32). M is free:
# the last M block backs up to M - BLOCK_M and recomputes the overlap, which is
# idempotent, so any M >= BLOCK_M works. The second group exercises that.
test_shapes = [
    (4096, 4096, 4096), # (m, n, k)
    (2048, 4096, 2048),
    (4096, 2048, 2048),
    (4096, 1024, 4096),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (2048, 1024, 512),
    (1024, 1024, 128),
    # M remainder.
    (3008, 5120, 6144),
    (2049, 5120, 6144),
    (1500, 5120, 6144),
    (7777, 5120, 6144),
    (129, 1024, 1024),
    (255, 1024, 1024),
    (4097, 1024, 1024),
    # Decode. M < BLOCK_M(128) routes to the thin configs; M <= 16 takes the
    # 16-row one with split-K, above that the 32-row one without. The Qwen3
    # TP=2 shapes are the K_local = 3072 (attn_out) and 8704 (mlp_down) rows.
    (1, 5120, 3072),
    (4, 5120, 3072),
    (8, 5120, 8704),
    (16, 5120, 8704),
    (17, 5120, 3072),
    (20, 5120, 8704),
    (31, 5120, 3072),
    (32, 5120, 3072),
    (64, 5120, 8704),
]

torch.manual_seed(0)
random.seed(0)
dtype = torch.bfloat16
device = "cuda:0"

if __name__ == "__main__":
    for test_shape in test_shapes:
        m, n, k = test_shape
        A = init_randn((m, k), dtype, device)
        B = init_randn((k, n), dtype, device)
        Bt = B.t().contiguous()
        C = init_empty((m, n), dtype, device)

        C_ref = torch.matmul(A, B)
        tk_kernel.dispatch_micro(A, Bt, C)
        torch.cuda.synchronize()

        is_valid = torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2)
        result = "TEST PASSED" if is_valid else "TEST FAILED"
        if not is_valid:
            err = (C.float() - C_ref.float()).abs().max().item()
            result += f" (max abs err {err:.4g})"
        print(f"{test_shape}".ljust(20) + f" | {result}")
