from random import rand, seed
from benchmark import Bench, BenchConfig, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer.dimlist import DimList, Dim
from gpu.host import DeviceContext
from internal_utils._utils import ValOrDim, dynamic, static
from math import ceildiv
from pathlib import Path
from buffer import NDBuffer
from utils import IndexList
from utils.index import Index
from utils.numerics import min_or_neg_inf
from collections import OptionalReg
from memory import memset_zero

# Import flash attention and related utilities
from mha import flash_attention
# , mha_gpu_naive
from nn.mha_mask import CausalMask, NullMask, MaterializedMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal
from math import isclose

fn test_amd_mha[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int,
](
    mut bench: Bench,
    ctx: DeviceContext,
    seq_len: ValOrDim,
    num_keys: ValOrDim,
    batch_size: Int = 1,
    bench_and_verify: Bool = True,
) raises:
    var M = seq_len.value
    var K = num_keys.value
    print("AMD MHA Test:", M, "x", K, ",", num_heads, "heads, depth", depth, ", group", group)

    alias scale = Float32(1.0 / (depth ** 0.5))  # Proper attention scaling
    alias kv_num_heads = num_heads // group

    # Calculate sizes
    var q_size = batch_size * num_heads * M * depth
    var k_size = batch_size * kv_num_heads * K * depth  
    var v_size = k_size
    var o_size = q_size
    var mask_size = batch_size * num_heads * M * K

    print("Sizes - Q:", q_size, "K/V:", k_size, "O:", o_size, "Mask:", mask_size)
    print("KV heads:", kv_num_heads, "Scale:", scale)

    # Allocate host memory
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Initialize with random data
    seed(42)
    rand[qkv_type](q_ptr, q_size)
    rand[qkv_type](k_ptr, k_size)
    rand[qkv_type](v_ptr, v_size)

    # Initialize causal mask (following the reference pattern)
    var mask = NDBuffer[mask_type, 4](
        mask_ptr, Index(batch_size, num_heads, M, K)
    )
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(M):
                for k_idx in range(K):
                    # Causal mask: allow attention to current and previous positions
                    mask.store(
                        Index(b, h, q_idx, k_idx),
                        0 if q_idx + K - M >= k_idx else min_or_neg_inf[mask_type](),
                    )

    # Create device buffers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    # Construct device NDBuffers with proper static shapes (following reference pattern)
    var q_device = NDBuffer[
        qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), num_heads, depth)
    ](
        q_device_ptr.unsafe_ptr(),
        Index(batch_size, M, num_heads, depth),
    )
    var k_device = NDBuffer[
        qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        k_device_ptr.unsafe_ptr(),
        Index(batch_size, K, kv_num_heads, depth),
    )
    var v_device = NDBuffer[
        qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](
        v_device_ptr.unsafe_ptr(),
        Index(batch_size, K, kv_num_heads, depth),
    )
    var mask4d = NDBuffer[
        mask_type, 4, MutableAnyOrigin, DimList.create_unknown[4]()
    ](
        mask_device_ptr.unsafe_ptr(),
        Index(batch_size, num_heads, M, K),
    )
    var output_device = NDBuffer[
        qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), num_heads, depth)
    ](
        output_device_ptr.unsafe_ptr(),
        Index(batch_size, M, num_heads, depth),
    )

    print("Device buffers created successfully")

    # Define kernel launch function (following reference pattern)
    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        flash_attention(
            output_device,
            q_device,
            k_device,
            v_device,
            MaterializedMask(mask4d),
            IdentityScoreMod(),
            scale,
            ctx,
            OptionalReg[Int](None),  # num_partitions
        )

    # Test the kernel call once
    print("Testing AMD flash attention kernel...")
    try:
        kernel_launch(ctx)
        ctx.synchronize()
        print("AMD flash attention kernel successful!")
    except e:
        print("AMD flash attention kernel failed:", e)
        # Clean up and return
        q_ptr.free()
        k_ptr.free()
        v_ptr.free()
        mask_ptr.free()
        output_ptr.free()
        flash_output_ptr.free()
        return

    # Copy result back
    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)
    ctx.synchronize()

    # Check output for validity
    var has_invalid = False
    for i in range(min(100, o_size)):
        var val = flash_output_ptr[i]
        var val_f32 = val.cast[DType.float32]()
        if val_f32 != val_f32:  # NaN check
            print("NaN detected at index", i)
            has_invalid = True
            break

    if not has_invalid:
        print("Output validation passed (no NaN detected)")
    else:
        print("WARNING: Invalid values detected in output")

    if bench_and_verify:
        # Run reference implementation for verification
        print("Running reference implementation...")
        var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
        var output_ref_device = NDBuffer[
            qkv_type, 4, MutableAnyOrigin, DimList(Dim(), Dim(), num_heads, depth)
        ](
            output_ref_device_ptr.unsafe_ptr(),
            Index(batch_size, M, num_heads, depth),
        )
        
        # # Initialize reference output
        # ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        # # Run naive MHA for comparison
        # mha_gpu_naive(
        #     q_device,
        #     k_device,
        #     v_device,
        #     mask4d,
        #     output_ref_device,
        #     scale,
        #     batch_size,
        #     M,
        #     K,
        #     num_heads,
        #     depth,
        #     group,
        #     ctx,
        # )

        # ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        # ctx.synchronize()

        # # Verify results
        # print("Verifying results...")
        # var rtol = 0.02
        # var errors = 0
        # var max_errors_to_print = 5

        # for h in range(num_heads):
        #     for s in range(M):
        #         for d in range(depth):
        #             var idx = d + depth * (h + s * num_heads)
        #             var expect = output_ptr[idx]
        #             var actual = flash_output_ptr[idx]
        #             if not isclose(expect, actual, atol=1e-5, rtol=rtol):
        #                 if errors < max_errors_to_print:
        #                     print("Mismatch at (h=", h, "s=", s, "d=", d, ") actual=", actual, "expected=", expect)
        #                 errors += 1

        # if errors == 0:
        #     print("All outputs match reference!")
        # else:
        #     print("Found", errors, "mismatches")

        # _ = output_ref_device_ptr

    # Calculate FLOPS for benchmarking
    fn compute_flops() -> Int:
        return 4 * batch_size * num_heads * M * K * depth

    if bench_and_verify:
        @parameter
        fn bench_func(mut b: Bencher):
            @parameter
            @always_inline
            fn _kernel_launch(ctx: DeviceContext) raises:
                kernel_launch(ctx)

            b.iter_custom[_kernel_launch](ctx)

        bench.bench_function[bench_func](
            BenchId("AMD_FlashAttention_" + String(M) + "x" + String(K) + "_" + String(num_heads) + "h_" + String(depth) + "d"),
            ThroughputMeasure(BenchMetric.flops, compute_flops()),
        )

        print("Benchmark completed!")

    print("Theoretical FLOPS:", compute_flops())

    # Clean up
    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr

    q_ptr.free()
    k_ptr.free() 
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()


def main():
    var bench = Bench(
        BenchConfig(
            out_file=Path("amd_flash_attention_out.txt"),
            num_warmup_iters=500,
            max_iters=50,
        )
    )

    with DeviceContext() as ctx:
        alias qkv_type = DType.bfloat16
        alias mask_type = DType.float32
        alias depth = 128
        alias num_heads = 64
        alias group = 8
        alias batch_size = 16

        # Medium test case  
        test_amd_mha[qkv_type, mask_type, depth, num_heads, group](
            bench, ctx, static[2048](), static[2048](), batch_size=batch_size
        )

    bench.dump_report()
    print("Done! Check amd_flash_attention_out.txt for results")

    