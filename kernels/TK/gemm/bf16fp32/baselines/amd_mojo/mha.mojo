# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import ceildiv, recip
from math.constants import log2e
from sys import (
    CompilationTarget,
    alignof,
    has_amd_gpu_accelerator,
    is_amd_gpu,
    simdwidthof,
    sizeof,
)

import gpu.warp as warp
from algorithm import elementwise
from algorithm.functional import tile_and_unswitch, unswitch, vectorize
from bit import next_power_of_two
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext
from gpu.host import Dim as LaunchDim
from gpu.host import FuncAttribute
from gpu.host.info import A100, H100, B200
from gpu.memory import (
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from kv_cache.types import KVCacheT
from layout import Layout
from layout.int_tuple import IntTuple
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_local_to_shared,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_builder import static
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg._multistage_gemm_gpu import multistage_mma
from linalg.bmm import batched_matmul
from linalg.transpose import transpose
from memory import stack_allocation
from nn.mha_amd import (
    # mha_decoding_single_batch_amd,
    mha_single_batch_amd
)
from nn.mha_mask import MaterializedMask, MHAMask, TileMaskStatus
from nn.mha_operand import KVCacheMHAOperand, MHAOperand, NDBufferMHAOperand
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_sm90 import mha_sm90_dispatch
from nn.mha_sm100 import mha_sm100_dispatch
from nn.mha_utils import (
    DynamicInt,
    FlashAttentionAlgorithm,
    MHAConfig,
    NoPartition,
    SplitKPartition,
    StaticInt,
    _copy_frag_to_smem,
    _kernel_mask,
)
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from tensor_internal import IOUnknown, ManagedTensorSlice
from tensor_internal.managed_tensor_slice import StaticTensorSpec

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple

from nn.softmax import (
    _exp2_concrete,
    _exp_concrete,
    _online_softmax_iter_for_mma_output,
    _online_softmax_iter_for_mma_output_split_warp_reduce,
    _softmax_gpu,
    softmax,
)

# ===-----------------------------------------------------------------------===#
# Flash attention
# ===-----------------------------------------------------------------------===#


fn flash_attention[
    rank: Int,
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(type, q_shape.get[2](), q_shape.get[3]()),
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: NDBuffer[_, rank, *_],
    v: NDBuffer[_, rank, *_],
    mask: NDBuffer,
    scale: Float32,
    context: DeviceContextPtr = DeviceContextPtr(),
    num_partitions: OptionalReg[Int] = None,
) raises:
    # TODO docstring
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("k", k),
            trace_arg("v", v),
            trace_arg("output", output),
        )

    var ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target = ctx.default_device_info.api](
        "flash_attention",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
    ):
        return flash_attention[
            use_score_mod=use_score_mod,
            config=config,
            decoding_warp_split_k=decoding_warp_split_k,
            naive_kernel=naive_kernel,
        ](
            output,
            q,
            k,
            v,
            MaterializedMask(mask),
            IdentityScoreMod(),
            scale,
            context.get_device_context(),
            num_partitions,
        )

fn flash_attention_hw_supported[qkv_type: DType]() -> Bool:
    return (
        has_amd_gpu_accelerator() and qkv_type is DType.bfloat16
    )


# Entry point for flash_attention with batch_size > 1.
@always_inline
fn flash_attention[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        type, q_shape.get[rank - 2](), q_shape.get[rank - 1]()
    ),
    ragged: Bool = False,
    decoding_warp_split_k: Bool = False,
    naive_kernel: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ] = None,
    num_partitions: OptionalReg[Int] = None,
) raises:
    """Flash attention 2 algorithm.
    Compute:
        (1) Transpose (Q) BSHD -> BHSD;
        (2) Transpose (K) BSHD -> BHSD;
        (3) Transpose (V) BSHD -> BHSD;
        (4) P = Bmm(Q, K), P is also called "score";
        (5) P = P * scale + mask;
        (6) P = softmax(P);
        (7) O = Bmm(P, V)
        (8) Output = Transpose(O).

    B, S, H, D denote batch size, sequence length, head count and depth, respectively.
    (1), (2), (3) happens while loading the data into shared memory.
    (8) happens when writing output to global memory.

    All inputs (query, key, and value) must have BSHD layout. The mask can be
    BSS or BHSS.

    This kernel also handles grouped attention optimization. In this case the shape of
    K and V are BShD where h = H / num_groups.

    This kernels handles batches with different valid lengths (i.e., before the
    padding). Such lengths are passed in valid_length argument.
    """
    constrained[
        ragged or rank == 4, "only support rank 4 inputs for non-ragged inputs."
    ]()
    constrained[
        not ragged or rank == 3, "only support rank 3 inputs for ragged inputs."
    ]()
    constrained[
        q.dtype == cache_t.dtype == output.type,
        "Q, K, V, output should have same type.",
    ]()
    constrained[
        q.dtype is DType.float32 or q.dtype.is_half_float(),
        "Only support single and half precision.",
    ]()

    # TODO docstring
    @always_inline
    @parameter
    fn description_fn() -> String:
        return String(";").join(
            trace_arg("q", q),
            trace_arg("output", output),
        )

    with Trace[TraceLevel.OP, target = ctx.default_device_info.api](
        "flash_attention",
        Trace[
            TraceLevel.OP, target = ctx.default_device_info.api
        ]._get_detail_str[description_fn](),
    ):
        # TODO: This helps differentiate between CE/TG. Not batch-specific.
        #       We'll just implement a flag on the cache object which is true
        #       when the batch contains all cache_lens == 0. Remove this when
        #       such flag (part of ContiguousKVCache) is implemented.
        var is_token_generation = (
            k.max_prompt_length() == 1 and not k.empty_cache()
        )

        var max_prompt_len: Int
        var num_keys = Int(k.max_context_length())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k.max_prompt_length())

        # Whether head and depth are static. With BSHD, B and S are dynamic.
        # H and D are always known for opaque KVCache types, we only check Q.
        # fmt: off
        alias head_depth_known = q.shape.all_known[rank-2, rank]()
        alias is_sm90or100 = (ctx.default_device_info is H100) or (ctx.default_device_info is B200)
        alias head_depth_supported = q.shape.get[rank-1]() == 128 or (q.shape.get[rank-1]() == 64 and (is_sm90or100 or ctx.default_device_info is A100 or has_amd_gpu_accelerator())) or (q.shape.get[rank-1]() == 256 and (has_amd_gpu_accelerator() or (is_sm90or100 and mask_t.mask_safe_out_of_bounds)))
        alias flash_attention_applicable = flash_attention_hw_supported[type]() and head_depth_known and head_depth_supported and not naive_kernel
        # fmt: on
        alias kv_num_heads = cache_t.kv_params.num_heads

        var k_operand = KVCacheMHAOperand(k)
        var v_operand = KVCacheMHAOperand(v)

        flash_attention_dispatch[
            kv_num_heads=kv_num_heads,
            use_score_mod=use_score_mod,
            config=config,
            ragged=ragged,
            _is_flash_attention_applicable=flash_attention_applicable,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            output,
            q,
            k_operand,
            v_operand,
            mask_functor,
            score_mod_functor,
            valid_length,
            max_prompt_len,
            num_keys,
            scale,
            is_token_generation,
            ctx,
            kv_input_row_offsets,
            num_partitions,
        )


@always_inline
fn q_num_matrix_view_rows[
    type: DType, rank: Int, q_shape: DimList, //, *, decoding: Bool, depth: Int
](q: NDBuffer[type, rank, _, q_shape, *_]) -> Int:
    # for tma if decoding, we view q as a rows x depth matrix
    # otherwise, we view q as a rows x (depth*num_heads) matrix
    var num_rows: Int = q.dim[0]()

    @parameter
    for i in range(1, rank - 1 if decoding else rank - 2):
        num_rows *= q.dim[i]()
    return num_rows * (depth // 64) if decoding else num_rows


@always_inline
fn flash_attention_dispatch[
    rank: Int,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    kv_num_heads: Int,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(
        type, q_shape.get[rank - 2](), q_shape.get[rank - 1]()
    ),
    ragged: Bool = False,
    _is_flash_attention_applicable: Bool = True,
    # Work arounds to unify KVCache and NDBuffer inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV NBuffer's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for homogeneous
    # NDBuffer inputs to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    # we might also want to use valid length for padded NDBuffer inputs
    _padded_ndbuffer: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    output: NDBuffer[_, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    is_token_generation: Bool,
    ctx: DeviceContext,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ] = None,
    num_partitions: OptionalReg[Int] = None,
) raises:
    alias num_heads = config.num_heads
    alias depth = config.depth
    alias group = config.num_heads // kv_num_heads

    # K V smem is only separate for GPUs with shared memory greater or equal to A100's.
    alias is_shared_kv = ctx.default_device_info.shared_memory_per_multiprocessor < A100.shared_memory_per_multiprocessor

    constrained[depth == q.shape.get[rank - 1]()]()
    constrained[num_heads == q.shape.get[rank - 2]()]()
    var batch_size: Int

    @parameter
    if ragged:
        batch_size = valid_length.shape()[0] - 1
    # This branch holds for both KVCache and NDBuffer inputs.
    # Q is BSHD, S is either homogeneous or padded to same length.
    else:
        batch_size = q.dim[0]()

    alias q_half_float = type in (DType.float16, DType.bfloat16)
    alias q_half_float_or_fp32 = type is DType.float32 or q_half_float

    @parameter
    if _is_flash_attention_applicable:
        alias is_sm90 = ctx.default_device_info is H100
        alias is_sm100 = ctx.default_device_info is B200
        if not is_token_generation:
            # TODO note that we have to handle mask tensor alignment here.
            # Choose matmul parameters based on dtype.
            @parameter
            if (
                (is_sm90 or is_sm100)
                and q_half_float
                and (ragged or not _use_valid_length)
                and config.algorithm == FlashAttentionAlgorithm(3)
            ):
                num_rows_q = q_num_matrix_view_rows[
                    decoding=False, depth=depth
                ](q)

                @parameter
                if is_sm90:
                    mha_sm90_dispatch[
                        config=config,
                        group=group,
                        use_score_mod=use_score_mod,
                        ragged=ragged,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        output.data,
                        q.data,
                        k,
                        rebind[k_t](v),
                        num_rows_q,
                        mask_functor,
                        score_mod_functor,
                        valid_length,
                        DynamicInt(max_prompt_len),
                        max_cache_valid_length,
                        scale,
                        kv_input_row_offsets,
                        batch_size,
                        NoPartition[get_accum_type[q.dtype]()](),
                        ctx,
                    )
                else:
                    constrained[is_sm100]()
                    mha_sm100_dispatch[
                        config=config,
                        group=group,
                        use_score_mod=use_score_mod,
                        ragged=ragged,
                        _is_cache_length_accurate=_is_cache_length_accurate,
                    ](
                        output.data,
                        q.data,
                        k,
                        rebind[k_t](v),
                        num_rows_q,
                        mask_functor,
                        score_mod_functor,
                        valid_length,
                        DynamicInt(max_prompt_len),
                        max_cache_valid_length,
                        scale,
                        kv_input_row_offsets,
                        batch_size,
                        NoPartition[get_accum_type[q.dtype]()](),
                        ctx,
                    )

            else:
                alias BM = config.block_m()
                alias smem_use = config.shared_mem_bytes[is_shared_kv]()
                alias kernel = mha[
                    config.type,
                    k_t,
                    v_t,
                    output.type,
                    mask_t,
                    score_mod_t,
                    config,
                    group=group,
                    use_score_mod=use_score_mod,
                    ragged=ragged,
                    is_shared_kv=is_shared_kv,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    _padded_ndbuffer=_padded_ndbuffer,
                ]
                ctx.enqueue_function[kernel](
                    q.data,
                    k,
                    v,
                    output.data,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length,
                    kv_input_row_offsets,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=(
                        Int(ceildiv(max_prompt_len, BM)),
                        Int(config.num_heads),
                        Int(batch_size),
                    ),
                    block_dim=(Int(config.num_threads()), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )

        # Not supported by contexting and decoding, e.g cross-attention or depth != 128
        else:
            # Assumes BSHD.
            mha_gpu_naive[
                ragged=ragged,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q,
                k,
                v,
                mask_functor,
                output,
                valid_length,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                num_heads,
                depth,
                group,
                ctx,
            )

    # Not supported by fast flash attention kernel.
    else:
        # Assumes BSHD.
        mha_gpu_naive[
            ragged=ragged,
            _use_valid_length=_use_valid_length,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](
            q,
            k,
            v,
            mask_functor,
            output,
            valid_length,
            scale,
            batch_size,
            max_prompt_len,
            max_cache_valid_length,
            num_heads,
            depth,
            group,
            ctx,
        )


fn flash_attention[
    rank: Int,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    q_shape: DimList, //,
    use_score_mod: Bool = False,
    config: MHAConfig = MHAConfig(type, q_shape.get[2](), q_shape.get[3]()),
    decoding_warp_split_k: Bool = False,
    _use_valid_length: Bool = False,
    _padded_ndbuffer: Bool = False,
    naive_kernel: Bool = False,
](
    output: NDBuffer[mut=True, _, rank, *_],
    q: NDBuffer[type, rank, _, q_shape, *_],
    k: NDBuffer[_, rank, *_],
    v: NDBuffer[_, rank, *_],
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    scale: Float32,
    ctx: DeviceContext,
    # if not set, we select num_partitions based on heuristics
    num_partitions: OptionalReg[Int] = None,
    valid_length: OptionalReg[
        ManagedTensorSlice[
            IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ]
    ] = None,
) raises:
    # See the kV cache overloads for comments.

    constrained[rank == 4, "only support rank 4 inputs."]()

    # Runtime dimensions.
    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()

    # Whether head and depth are static. With BSHD, B and S are dynamic.
    # H and D are always known.
    # fmt: off
    alias head_depth_known = q.shape.all_known[2, 4]() and k.shape.has_value[2]()
    alias is_sm90or100 = (ctx.default_device_info is H100) or (ctx.default_device_info is B200)
    alias head_depth_supported = q.shape.get[rank-1]() == 128 or (q.shape.get[rank-1]() == 64 and (is_sm90or100 or ctx.default_device_info is A100 or has_amd_gpu_accelerator())) or (q.shape.get[rank-1]() == 256 and (has_amd_gpu_accelerator() or (is_sm90or100 and mask_t.mask_safe_out_of_bounds)))
    alias flash_attention_applicable = flash_attention_hw_supported[type]() and head_depth_known and head_depth_supported and not naive_kernel

    alias q_half_float = q.dtype in (DType.float16, DType.bfloat16)
    alias kv_num_heads = k.shape.get[2]()
    # fmt: on

    var is_token_generation = seq_len == 1 and num_keys > seq_len

    var k_operand = NDBufferMHAOperand(k)
    var v_operand = NDBufferMHAOperand(v)

    flash_attention_dispatch[
        kv_num_heads=kv_num_heads,
        use_score_mod=use_score_mod,
        config=config,
        ragged=False,
        _is_flash_attention_applicable=flash_attention_applicable,
        _is_cache_length_accurate=True,
        _use_valid_length=_use_valid_length,
        _padded_ndbuffer=_padded_ndbuffer,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        v_operand,
        mask_functor,
        score_mod_functor,
        valid_length.or_else(valid_length.T(UnsafePointer[UInt32](), [1], [1])),
        q.dim[1](),
        num_keys,
        scale,
        is_token_generation,
        ctx,
        None,
        num_partitions,
    )


# ===-----------------------------------------------------------------------===#
# Flash attention for context encoding
# ===-----------------------------------------------------------------------===#


# for depth = 128 we want waves_per_eu = 2 and for depth = 256 we want waves_per_eu = 1
# this heuristic may not be valid for other depths
@__llvm_metadata(`rocdl.waves_per_eu`=Int(256 // config.depth))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn mha[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    is_shared_kv: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    _padded_ndbuffer: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: Int,
    seq_len_arg: Int,
    num_keys_arg: Int,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    mask: mask_t,
    score_mod: score_mod_t,
):
    alias depth = config.depth
    alias num_heads = config.num_heads
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = seq_len_arg
    var num_keys: Int
    var mask_tensor_col = num_keys_arg
    var start_pos: UInt32 = 0

    @parameter
    if ragged:
        # treat valid_lengths as a input_row_offsets
        start_of_seq = Int(valid_length[batch_idx])
        end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq

        if seq_len < block_idx.x * config.block_m():
            return

        start_pos = k.cache_length(batch_idx)

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[batch_idx])
            kv_seq_end = Int(kv_row_offsets[batch_idx + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + Int(start_pos)
        else:
            num_keys = seq_len + Int(start_pos)

        q_batch_offset = start_of_seq * config.depth * config.num_heads

    # KVCache inputs, prompt lengths are all padded to the max in batch.
    elif _use_valid_length and not _padded_ndbuffer:
        # treat valid_lengths as valid lengths
        seq_len = Int(valid_length[batch_idx])

        if seq_len < block_idx.x * config.block_m():
            return

        @parameter
        if not _is_cache_length_accurate:
            var cache_length = k.cache_length(batch_idx)
            start_pos = cache_length

        num_keys = seq_len + k.cache_length(batch_idx)
        q_batch_offset = (
            config.depth * config.num_heads * max_seq_len * batch_idx
        )
    # NDBuffer inputs, homogeneous and padded batching.
    else:

        @parameter
        if _padded_ndbuffer:
            seq_len = Int(valid_length[batch_idx])
            num_keys = seq_len
        else:
            seq_len = seq_len_arg
            num_keys = num_keys_arg

        if seq_len < block_idx.x * config.block_m():
            return
        q_batch_offset = (
            config.depth * config.num_heads * max_seq_len * batch_idx
        )

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len

    @parameter
    if is_amd_gpu():
        constrained[
            use_score_mod == False,
            "use_score_mod must be False for AMD flash attention",
        ]()
        mha_single_batch_amd[group=group, config=config](
            output_ptr.offset(q_batch_offset),
            q_ptr.offset(q_batch_offset),
            k,
            v,
            seq_len,
            num_keys,
            scale,
            batch_idx,
            Int(start_pos),
            mask,
        )
    else:
        return CompilationTarget.unsupported_target_error[operation="mha"]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn mha_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask_tensor_col: Int,  # second dimension of mask tensor
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias k_type = k_t.dtype
    alias v_type = v_t.dtype
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    constrained[
        num_warps_m * num_warps_n == (num_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = thread_idx.x
    var warp_id: UInt32 = warp.broadcast(tid // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = config.k_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    alias v_smem_size = config.v_smem_size()
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    var v_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, BN),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](v_smem, v_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var q_tile_num_rows = min(BM, UInt(seq_len) - q_tile_idx * BM)
    var q_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2
    alias p_frag_align = alignof[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[alignment=p_frag_align]()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    alias row_alignment = alignof[SIMD[accum_type, simdwidthof[accum_type]()]]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    @parameter
    for i in range(0, Int(WM), 2):
        rowmax.store(i, SIMD[accum_type, 2](min_or_neg_inf[accum_type]()))
        rowsum.store(i, SIMD[accum_type, 2](0))

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    var p_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_y * WM
    var mask_warp_col = warp_x * WN

    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias num_pipeline_stages = config.num_pipeline_stages

    alias q_num_vecs = BM * BK // simd_size

    alias async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK, BK // simd_size
    )

    @parameter
    for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(q_id)[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        # we `async_copy_commit_group()` and after we finish copying `k`.

        q_gmem_iter._incr()

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(kv_tile_start_row, seq_len, tile_size):
    #     if i + tile_size >= seq_len:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @__copy_capture(seq_len, max_seq_len, num_keys, start_pos)
    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        if (
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * BM + start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += BN
            return

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[kv_gmem_layout](
            {kv_tile_num_rows, depth},
            {kv_num_heads * depth, 1},
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        @parameter
        fn _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: __type_of(tensor)
        ):
            return __type_of(tensor)(
                tensor.ptr,
                __type_of(tensor.runtime_layout)(
                    __type_of(tensor.runtime_layout.shape)(
                        num_rows, tensor.dim[1]()
                    ),
                    tensor.runtime_layout.stride,
                ),
            )

        alias kv_num_vecs = BN * BK // simd_size
        alias async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        @parameter
        for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(k_id)[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )

            k_gmem_iter._incr()

        async_copy_commit_group()
        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters = Int(depth // BK),
            k_group_size = config.k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            depth // BK,
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        fn _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[accum_type]() if use_score_mod
                or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
                * log2e
            )

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + m_mma * MMA_M
                    var mask_frag_col = mask_warp_col + n_mma * MMA_N

                    # Offset to current thread's fragment
                    mask_frag_row += lane // (MMA_N // p_frag_simdwidth)
                    mask_frag_col += lane * p_frag_simdwidth % MMA_N

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row + mask_frag_row + i * MMA_M // 2
                        )
                        var score_col = mask_frag_col

                        score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32,](
                                    Int(block_idx.z),
                                    Int(block_idx.y),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        @parameter
                        if use_score_mod:
                            p_reg_vec2[mma_id, i] = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32,](
                                        Int(block_idx.z),
                                        Int(block_idx.y),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p_reg_vec2[mma_id, i],
                                    max_seq_len,
                                )
                                * log2e
                            )
                        elif mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                IndexList[2, element_type = DType.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * BM + start_pos),
                    kv_tile_start_row,
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        alias reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        alias async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // v_smem_iter.layout.stride[0].value(),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load V tile into smem
        @parameter
        for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(v_id)[]

            @parameter
            if not not_last_iter:
                var num_rows_bound = min(
                    Int(BK), end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle = v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            v_gmem_iter._incr()

        async_copy_commit_group()

        @parameter
        if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM, BN, BK, WM, WN, MMA_M, MMA_N, p_frag_simdwidth
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters = Int(BN // BK),
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                BN // BK,
            )

        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters = Int(BN // BK),
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_reg_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                BN // BK,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    # Apply softmax denumerator.
    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            Int(warp_y), Int(warp_x)
        )
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                num_threads * simd_size // depth, depth // simd_size
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )
    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn mha_single_batch_pipelined[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    *,
    config: MHAConfig,
    group: Int = 1,
    use_score_mod: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    num_keys: Int,
    mask_tensor_col: Int,  # second dimension of mask tensor
    mask: mask_t,
    score_mod: score_mod_t,
    batch_idx: Int,
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias k_type = k_t.dtype
    alias v_type = v_t.dtype
    constrained[q_type == k_type and k_type == v_type]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_warps_n = config.num_warps_n()
    alias num_threads = config.num_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias BK = config.block_k()
    alias num_heads = config.num_heads
    alias depth = config.depth

    constrained[
        num_warps_m * num_warps_n == (num_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var tid: UInt32 = thread_idx.x
    var warp_id: UInt32 = warp.broadcast(tid // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size()
    var q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
    ]()
    var q_smem_iter = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    alias k_smem_size = config.kv_smem_size()
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    var k_smem_iter = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](k_smem, k_smem_size)

    var head_idx: UInt32 = block_idx.y
    var q_tile_idx: UInt32 = block_idx.x

    # Query global memory iterator
    alias q_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var q_tile_num_rows = min(BM, UInt(seq_len) - q_tile_idx * BM)
    var q_offset = depth * (head_idx + num_heads * q_tile_idx * BM)
    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[q_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]
    alias WM = config.WM
    alias WN = config.WN
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    alias accum_type = get_accum_type[q_type]()
    alias frag_size = get_fragment_size[mma_shape]()
    alias p_frag_size = frag_size[2]
    alias p_frag_simdwidth = p_frag_size // 2
    alias p_frag_align = alignof[SIMD[accum_type, p_frag_size]]()

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation[alignment=p_frag_align]()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation[alignment=p_frag_align]()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    alias row_alignment = alignof[SIMD[accum_type, simdwidthof[accum_type]()]]()
    var rowmax = stack_allocation[WM, accum_type, alignment=row_alignment]()
    var rowsum = stack_allocation[WM, accum_type, alignment=row_alignment]()

    @parameter
    for i in range(0, Int(WM), p_frag_simdwidth):
        rowmax.store(
            i, SIMD[accum_type, p_frag_simdwidth](min_or_neg_inf[accum_type]())
        )
        rowsum.store(i, SIMD[accum_type, p_frag_simdwidth](0))

    # Shared memory for P = Q * K^t
    # Only use BN/BK tiles. Setting circular so that the prefetch in matmul
    # doesn't go OOB at the last tile.
    var p_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    var p_smem_iter = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        address_space = AddressSpace.SHARED,
        circular=True,
    ](p_smem, BM * BN)

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(p_frag_simdwidth * num_warps_n, BM),
        address_space = AddressSpace.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row: UInt32 = q_tile_idx * BM
    var mask_warp_row = warp_y * WM
    var mask_warp_col = warp_x * WN

    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias num_pipeline_stages = config.num_pipeline_stages
    var is_first_iter = True

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(start, end, tile_size):
    #     if i + tile_size >= end:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @always_inline
    @parameter
    fn loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int):
        if (
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * BM + start_pos), Int(kv_tile_start_row)
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.FULL_MASK
        ):
            mask_warp_col += BN
            return

        alias kv_gmem_layout = Layout(
            IntTuple(Int(BN), Int(depth)),
            IntTuple(Int(kv_num_heads * depth), 1),
        )
        var kv_tile_num_rows = min(Int(tile_size), end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type = DType.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type = DType.int32](
                kv_num_heads * depth, 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked = not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                batch_idx, kv_tile_start_row, Int(head_idx // group), 0
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, BN, axis=0](0, 0)

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        var num_b_rows = None if not_last_iter else OptionalReg[Int](
            kv_tile_num_rows
        )

        # First iteration load q from global memory to shared memory.
        if is_first_iter:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
                swizzle_a=True,
                continue_prefetch_b=True,
                b_next_smem_layout = Layout.row_major(BK, BN),
                next_op_b_iter_masked = __type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type = __type_of(
                    v_gmem_iter
                ).layout_int_type,
                next_op_b_linear_idx_type = __type_of(
                    v_gmem_iter
                ).linear_idx_type,
                k_group_size = config.k_group_size,
            ](
                p_reg_tile,
                q_gmem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )
            is_first_iter = False
        # Subsequent iterations just use q in share memory.
        # TODO: Figure out a better function interface instead of passing in
        # shared memory iterator twice.
        else:
            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                True,  # transpose_b
                swizzle_a=True,
                continue_prefetch_b=True,
                b_next_smem_layout = Layout.row_major(BK, BN),
                next_op_b_iter_masked = __type_of(v_gmem_iter).masked,
                next_op_b_layout_int_type = __type_of(
                    v_gmem_iter
                ).layout_int_type,
                next_op_b_linear_idx_type = __type_of(
                    v_gmem_iter
                ).linear_idx_type,
                k_group_size = config.k_group_size,
            ](
                p_reg_tile,
                # Pass shared memory iterator to hint not loading from global memory.
                q_smem_iter,
                k_gmem_iter,
                q_smem_iter,
                k_smem_iter,
                depth // BK,
                next_op_b_iter=v_gmem_iter.bitcast[k_type](),
                num_b_rows=num_b_rows,
            )

        # Increment V iterator since it's prefetched inside 1st matmul.
        v_gmem_iter += num_pipeline_stages - 1

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        @parameter
        fn _apply_mask[masked: Bool]():
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[accum_type]() if use_score_mod
                or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
                * log2e
            )

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + m_mma * MMA_M
                    var mask_frag_col = mask_warp_col + n_mma * MMA_N

                    mask_frag_row += lane // (MMA_N // p_frag_simdwidth)
                    mask_frag_col += lane * p_frag_simdwidth % MMA_N

                    @parameter
                    for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row + mask_frag_row + (i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        var score_row_with_start_pos = score_row + start_pos

                        @parameter
                        if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type = DType.uint32,](
                                    Int(block_idx.z),
                                    Int(block_idx.y),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )

                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        @parameter
                        if use_score_mod:
                            p_reg_vec2[mma_id, i] = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32,](
                                        Int(block_idx.z),
                                        Int(block_idx.y),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p_reg_vec2[mma_id, i],
                                    max_seq_len,
                                )
                                * log2e
                            )
                        elif mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[
                                    2,
                                    element_type = DType.uint32,
                                ](Int(score_row), Int(score_col)),
                                IndexList[
                                    2,
                                    element_type = DType.uint32,
                                ](seq_len, num_keys),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch[_apply_mask](
            mask.status(
                Index[dtype = DType.uint32](
                    Int(q_tile_idx * BM + start_pos), kv_tile_start_row
                ),
                Index[dtype = DType.uint32](Int(BM), Int(BN)),
            )
            == TileMaskStatus.PARTIAL_MASK
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        alias reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, p_frag_simdwidth
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[
                1, p_frag_simdwidth
            ](),
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        # V reuse K's smem iterator. They has same smem footage expect for different layouts.
        var v_smem_iter = k_smem_iter.reshape[
            Layout.row_major(BK, BN)
        ]().bitcast[v_type]()

        @parameter
        if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM, BN, BK, WM, WN, MMA_M, MMA_N, p_frag_simdwidth
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)
            barrier()

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_smem_iter,
                v_gmem_iter,
                p_smem_iter,
                v_smem_iter,
                BN // BK,
                num_b_rows=num_b_rows,
            )
        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            multistage_mma[
                BM,
                BN,
                BK,
                WM,
                WN,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                static_num_iters = Int(BN // BK),
                prefetch_init=False,
                k_group_size = config.k_group_size,
            ](
                output_reg_tile,
                p_reg_iter,
                v_gmem_iter,
                p_smem_iter,
                v_smem_iter,
                BN // BK,
                num_b_rows=num_b_rows,
            )

    tile_and_unswitch[loop_over_kvcache, VariadicList[Int](BN)](0, num_keys)

    @parameter
    for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    alias output_gmem_layout = Layout(
        IntTuple(Int(BM), Int(depth)), IntTuple(Int(num_heads * depth), 1)
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
        masked=True,
    ](
        output_ptr + Int(q_offset),
        RuntimeLayout[
            element_type = DType.int32, linear_idx_type = DType.int32
        ](
            RuntimeTuple[output_gmem_layout.shape, element_type = DType.int32](
                Int(q_tile_num_rows), depth
            ),
            RuntimeTuple[output_gmem_layout.stride, element_type = DType.int32](
                num_heads * depth, 1
            ),
        ),
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN](
        Int(warp_y), Int(warp_x)
    )

    # Write to global memory.
    @parameter
    if output_type.is_half_float():
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space = AddressSpace.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN](
            Int(warp_y), Int(warp_x)
        )

        alias swizzle = make_swizzle[
            num_rows = MMA_M // 2, row_size=WN, access_size=MMA_N
        ]()
        copy_local_to_shared[
            thread_layout = Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
        barrier()
        copy_sram_to_dram[
            thread_layout = Layout.row_major(
                num_threads * simd_size // depth, depth // simd_size
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )

        # Guard writing to shared memory.

        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.

    else:
        copy_local_to_dram[dst_thread_layout = Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions and
# batch_size > 1.
# ===-----------------------------------------------------------------------===#

alias _NAIVE_BMM_BLOCK_DIM = LaunchDim(32, 16, 1)
alias _NAIVE_BMM_BLOCK_TUPLE = StaticTuple[Int32, 1](
    _NAIVE_BMM_BLOCK_DIM.x()
    * _NAIVE_BMM_BLOCK_DIM.y()
    * _NAIVE_BMM_BLOCK_DIM.z()
)


fn mha_gpu_naive[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    rank: Int, //,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: NDBuffer[_, rank, *_],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: NDBuffer[output_type, rank, *_],
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
) raises:
    alias q_type = q.dtype
    alias k_type = k_t.dtype
    alias v_type = k_type

    var num_keys = max_cache_size

    alias p_type = get_accum_type[q_type]()
    var p_device = ctx.enqueue_create_buffer[p_type](
        batch_size * num_heads * max_prompt_len * num_keys
    )
    # FIXME: RUNP-356 Direct access to CUDA within DeviceContext
    var p_buffer = NDBuffer[p_type, 3](
        p_device._unsafe_ptr(),
        Index(batch_size * num_heads, max_prompt_len, num_keys),
    )
    var q_ptr = q.data
    alias kernel = _bmm0_bs[
        q_type,
        k_t,
        mask_t,
        p_type,
        ragged=ragged,
        _use_valid_length=_use_valid_length,
    ]

    ctx.enqueue_function[kernel](
        p_device,
        q_ptr,
        k,
        managed_tensor_slice_to_ndbuffer(valid_length),
        scale,
        batch_size,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        mask_functor,
        grid_dim=(
            ceildiv(num_keys, 32),
            ceildiv(max_prompt_len, 16),
            num_heads * batch_size,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
    )

    @parameter
    @__copy_capture(p_buffer)
    fn input_fn_device[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[p_type, _simd_width]:
        return p_buffer.load[width=_simd_width](rebind[IndexList[3]](coords))

    _softmax_gpu[p_type, 1, 3, DimList.create_unknown[3](), input_fn_device](
        Index(batch_size * num_heads, max_prompt_len, num_keys),
        p_buffer,
        2,
        ctx,
    )
    ctx.enqueue_function[
        _bmm1_bs[
            output_type,
            p_type,
            v_t,
            ragged=ragged,
            _use_valid_length=_use_valid_length,
        ]
    ](
        output.data,
        p_device,
        v,
        valid_length,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        grid_dim=(
            ceildiv(depth, 32),
            ceildiv(max_prompt_len, 16),
            num_heads * batch_size,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
    )

    _ = p_device^


@always_inline
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=_NAIVE_BMM_BLOCK_TUPLE)
fn _bmm0_bs[
    q_type: DType,
    k_t: MHAOperand,
    mask_t: MHAMask,
    p_type: DType,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
](
    p_ptr: UnsafePointer[Scalar[p_type]],
    q_ptr: UnsafePointer[Scalar[q_type]],
    k: k_t,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    mask_functor: mask_t,
):
    # In the num_keys dim.
    var x = global_idx.x
    # In the prompt length dim.
    var y = global_idx.y

    alias k_type = k_t.dtype

    var batch_head = block_idx.z
    var batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var q_offset: Int
    var cur_cache_len: Int
    var padded_num_keys = max_cache_size
    var p_offset = batch_head * max_prompt_len * padded_num_keys

    @parameter
    if ragged:
        seq_start = Int(valid_length[batch])
        seq_end = Int(valid_length[batch + 1])
        cur_query_len = seq_end - seq_start
        q_offset = Int((seq_start * num_heads + head) * depth)
        cur_cache_len = k.cache_length(batch) + cur_query_len
    elif _use_valid_length:
        cur_query_len = Int(valid_length[batch])
        q_offset = Int(depth * (head + num_heads * max_prompt_len * batch))
        cur_cache_len = k.cache_length(batch) + cur_query_len
    # When inputs are all NDBuffers i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = max_prompt_len
        q_offset = Int(depth * (head + num_heads * max_prompt_len * batch))
        cur_cache_len = max_cache_size
        p_offset = batch_head * max_prompt_len * max_cache_size

    debug_assert(cur_query_len <= max_prompt_len, "Invalid cur_query_len")
    debug_assert(
        cur_cache_len <= padded_num_keys,
        "Invalid cur_cache_len",
    )

    if x >= padded_num_keys or y >= max_prompt_len:
        return

    var q = q_ptr + q_offset

    var kv_head = Int(head // group)

    var p = p_ptr + Int(p_offset)

    var accum = SIMD[p_type, 1](0.0)

    if x < cur_cache_len and y < cur_query_len:
        var k_ptr = k.block_paged_ptr[1](batch, x, kv_head, 0)

        # TODO: The AMD-specific path is to handle Llama shapes, similar
        #       to how things were before #53433. Once flash attention is
        #       supported on AMD, this stopgap AMD path should be eliminated to
        #       function as a generic fall-back (i.e., without vectorization).
        #       REL: KERN-1343.
        @parameter
        if is_amd_gpu():
            var accum_vec = SIMD[p_type, simdwidthof[p_type]()](0)

            @parameter
            fn accum_fn[width: Int](offset: Int):
                alias alignment = alignof[SIMD[p_type, width]]()
                var q_val = q.load[width=width, alignment=alignment](
                    y * num_heads * depth + offset
                ).cast[k_type]()
                var k_val = k_ptr.load[width=width, alignment=alignment](offset)
                var qk_val = (q_val * k_val).cast[p_type]()

                @parameter
                if width == 1:
                    accum += rebind[__type_of(accum)](qk_val)
                else:
                    accum_vec += rebind[__type_of(accum_vec)](qk_val)

            vectorize[accum_fn, simdwidthof[p_type]()](depth)
            accum += accum_vec.reduce_add()
        else:
            for d in range(depth):
                var q_val = q[y * num_heads * depth + d]
                var k_val = k_ptr[d]
                accum += q_val.cast[p_type]() * k_val.cast[p_type]()

    var score_row = y + cur_cache_len - cur_query_len
    var score_col = x
    p[y * padded_num_keys + x] = mask_functor.mask(
        Index(
            Int(batch),
            Int(head),
            Int(score_row),
            Int(score_col),
        ),
        accum * scale.cast[p_type](),
    )

    if x >= cur_cache_len or y >= cur_query_len:
        p[y * padded_num_keys + x] = min_or_neg_inf[p_type]()


@always_inline
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=_NAIVE_BMM_BLOCK_TUPLE)
fn _bmm1_bs[
    output_type: DType,
    p_type: DType,
    v_t: MHAOperand,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
](
    output_ptr: UnsafePointer[Scalar[output_type]],
    p_ptr: UnsafePointer[Scalar[p_type]],
    v: v_t,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    alias v_type = v_t.dtype

    # In the depth dim.
    var x = global_idx.x
    # IN the sequence length dim.
    var y = global_idx.y

    var batch_head = block_idx.z
    var batch, head = divmod(batch_head, UInt(num_heads))

    var cur_query_len: Int
    var output_offset: Int
    var cur_cache_len: Int
    var padded_num_keys = max_cache_size
    var p_offset = batch_head * max_prompt_len * padded_num_keys

    @parameter
    if ragged:
        seq_start = Int(valid_length[batch])
        seq_end = Int(valid_length[batch + 1])
        cur_query_len = seq_end - seq_start
        output_offset = Int((seq_start * num_heads + head) * depth)
        cur_cache_len = cur_query_len + v.cache_length(batch)
    elif _use_valid_length:
        cur_query_len = Int(valid_length[batch])
        output_offset = depth * (head + num_heads * max_prompt_len * batch)
        cur_cache_len = cur_query_len + v.cache_length(batch)
    # When inputs are all NDBuffers i.e. all sequences in batch have the same
    # length and same cache length
    else:
        cur_query_len = max_prompt_len
        output_offset = depth * (head + num_heads * max_prompt_len * batch)
        cur_cache_len = max_cache_size
        p_offset = batch_head * max_prompt_len * max_cache_size

    debug_assert(cur_query_len <= max_prompt_len, "Invalid cur_query_len")

    if x >= depth or y >= cur_query_len:
        return

    var p = p_ptr + p_offset

    var kv_head = Int(head // group)
    var output = output_ptr + Int(output_offset)

    var accum = SIMD[DType.float32, 1](0.0)

    for i in range(cur_cache_len):
        var v_ptr = v.block_paged_ptr[1](batch, i, kv_head, x)
        accum += (
            p[y * padded_num_keys + i].cast[DType.float32]()
            * v_ptr[0].cast[DType.float32]()
        )

    output[y * num_heads * depth + x] = accum.cast[output_type]()


# ===-----------------------------------------------------------------------===#
# Naive GPU multihead attention supporting flexible dimensions.
# ===-----------------------------------------------------------------------===#


fn mha_gpu_naive[
    q_type: DType,
    k_type: DType,
    v_type: DType,
    output_type: DType,
    rank: Int,
    mask_type: DType,
    mask_rank: Int, //,
](
    q: NDBuffer[q_type, rank, *_],
    k: NDBuffer[k_type, rank, *_],
    v: NDBuffer[v_type, rank, *_],
    mask: NDBuffer[mask_type, mask_rank, *_, **_],
    output: NDBuffer[mut=True, output_type, rank, *_],
    scale: Float32,
    batch_size: Int,
    seq_len: Int,
    num_keys: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
) raises:
    var k_operand = NDBufferMHAOperand(k)
    var v_operand = NDBufferMHAOperand(v)
    var null_valid_length = ManagedTensorSlice[
        IOUnknown,
        static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
    ](UnsafePointer[UInt32](), IndexList[1](0), IndexList[1](0))

    mha_gpu_naive[_is_cache_length_accurate=True](
        q,
        k_operand,
        v_operand,
        MaterializedMask(mask),
        output,
        null_valid_length,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )


fn mha_gpu_naive[
    q_type: DType,
    output_type: DType,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    rank: Int, //,
    ragged: Bool = False,
](
    q: NDBuffer[q_type, rank, *_],
    k: cache_t,
    v: cache_t,
    mask_functor: mask_t,
    output: NDBuffer[mut=True, output_type, rank, *_],
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
) raises:
    var k_operand = KVCacheMHAOperand(k)
    var v_operand = KVCacheMHAOperand(v)

    mha_gpu_naive[_use_valid_length=True, _is_cache_length_accurate=False,](
        q,
        k_operand,
        v_operand,
        mask_functor,
        output,
        valid_length,
        scale,
        batch_size,
        max_prompt_len,
        max_cache_size,
        num_heads,
        depth,
        group,
        ctx,
    )


# ===-----------------------------------------------------------------------===#
# Naive CPU MHA as reference
# ===-----------------------------------------------------------------------===#


fn _naive_attention_with_transpose[
    type: DType,
    transpose_k: Bool = False,
](
    output: NDBuffer[mut=True, type, 4],
    q: NDBuffer[type, 4],
    k: NDBuffer[type, 4],
    v: NDBuffer[type, 4],
    mask: NDBuffer[type, 2],
    scale: Float32,
) raises:
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    Layouts:
        q: BSHD
        k, v: BKHD
        output: BSHD
        mask: SK
    B, S, K, H, D stand for batch size, sequence length, number of keys,
    number of heads, and depth per head, respectively.
    """
    alias simd_size = simdwidthof[type]()

    var batch_size = q.dim[0]()
    var seq_len = q.dim[1]()
    var num_keys = k.dim[1]()
    var num_heads = q.dim[2]()
    var depth = q.dim[3]()

    # Q, K, V transposed
    var qt_ptr = UnsafePointer[Scalar[type]].alloc(q.num_elements())
    var kt_ptr = UnsafePointer[Scalar[type]].alloc(k.num_elements())
    var vt_ptr = UnsafePointer[Scalar[type]].alloc(v.num_elements())
    # Score = softmax(Q * K)
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = UnsafePointer[Scalar[type]].alloc(score_size)
    # O = Score * V. It's transposed and will be transposed back to output.
    var ot_ptr = UnsafePointer[Scalar[type]].alloc(output.num_elements())

    var qt = NDBuffer[type, 4](
        qt_ptr, Index(batch_size, num_heads, seq_len, depth)
    )
    var kt = NDBuffer[type, 4](
        kt_ptr, Index(batch_size, num_heads, depth, num_keys)
    )
    var vt = NDBuffer[type, 4](
        vt_ptr, Index(batch_size, num_heads, num_keys, depth)
    )
    var ot = NDBuffer[type, 4](
        ot_ptr, Index(batch_size, num_heads, seq_len, depth)
    )

    # BSHD -> BHSD
    var q_perm = NDBuffer[
        DType.index, 1, MutableAnyOrigin, 4
    ].stack_allocation()
    q_perm[0] = 0
    q_perm[1] = 2
    q_perm[2] = 1
    q_perm[3] = 3

    # BSHD -> BHDS
    var k_perm = NDBuffer[
        DType.index, 1, MutableAnyOrigin, 4
    ].stack_allocation()
    k_perm[0] = 0
    k_perm[1] = 2
    k_perm[2] = 3
    k_perm[3] = 1

    # BHSD -> BSHD
    var o_perm = NDBuffer[
        DType.index, 1, MutableAnyOrigin, 4
    ].stack_allocation()
    o_perm[0] = 0
    o_perm[1] = 2
    o_perm[2] = 1
    o_perm[3] = 3

    transpose(qt, q, q_perm.data)
    transpose(kt, k, k_perm.data)
    transpose(vt, v, q_perm.data)

    _naive_attention[type, transpose_k](ot, qt, kt, vt, mask, scale)

    transpose(output, ot, o_perm.data)

    qt_ptr.free()
    kt_ptr.free()
    vt_ptr.free()
    score_ptr.free()
    ot_ptr.free()


fn _naive_attention[
    type: DType,
    transpose_k: Bool = False,
](
    output: NDBuffer[mut=True, type, 4],
    q: NDBuffer[type, 4],
    k: NDBuffer[type, 4],
    v: NDBuffer[type, 4],
    mask: NDBuffer[type, 2],
    scale: Float32,
) raises:
    """This kernel provides reference values for flash attention in llama 2.
    It can't be used in any model.
    """
    alias simd_size = simdwidthof[type]()

    var batch_size = q.dim[0]()
    var num_heads = q.dim[1]()
    var seq_len = q.dim[2]()
    var num_keys = v.dim[2]()

    # Allocate intermediate memory buffer.
    var score_size = batch_size * num_heads * seq_len * num_keys
    var score_ptr = UnsafePointer[Scalar[type]].alloc(score_size)
    var score = NDBuffer[type, 4](
        score_ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )

    batched_matmul[transpose_b=transpose_k](score, q, k)

    @__copy_capture(score)
    @parameter
    @always_inline
    fn scale_and_mask[
        width: Int, _rank: Int, alignment: Int = 1
    ](coords: IndexList[_rank]):
        var vec = score.load[width=width](rebind[IndexList[4]](coords))
        vec = vec * scale.cast[type]()
        vec = vec + mask.load[width=width](
            Index(coords[_rank - 2], coords[_rank - 1])
        )
        score.store[width=width](rebind[IndexList[4]](coords), vec)

    elementwise[scale_and_mask, simd_size](score.get_shape())

    softmax[type, simd_size, 4](
        score,
        score,
        axis=3,
    )

    batched_matmul[transpose_b=False](output, score, v)

    score_ptr.free()


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    spec: StaticTensorSpec, //
](tensor: ManagedTensorSlice[static_spec=spec]) -> NDBuffer[
    spec.dtype,
    spec.rank,
    MutableAnyOrigin,
    spec.shape,
    spec.strides,
    alignment = spec.alignment,
    address_space = spec.address_space,
    exclusive = spec.exclusive,
]:
    var ptr = tensor._ptr.address_space_cast[spec.address_space]()
    return NDBuffer[
        spec.dtype,
        spec.rank,
        _,
        spec.shape,
        spec.strides,
        alignment = spec.alignment,
        address_space = spec.address_space,
        exclusive = spec.exclusive,
    ](ptr, tensor.shape(), tensor._runtime_strides)


