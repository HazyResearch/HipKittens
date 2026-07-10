import torch
import tk_kernel
import random
from utils import init_randn, init_empty

test_shapes = [
    (8192, 8192, 8192), # (m, n, k)
    (4096, 8192, 2048),
    (8192, 4096, 2048),
    (8192, 2048, 4096),
    (512, 1024, 1024),
    (512, 1024, 2048),
    (2048, 1024, 512),
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

        is_valid = torch.allclose(C, C_ref, rtol=1e-2)
        result = "TEST PASSED" if is_valid else "TEST FAILED"
        print(f"{test_shape}".ljust(18) + f" | {result}")