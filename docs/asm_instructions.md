# 1. Data Movement and Conversion Instructions

- `ds_read_b128`

    Load 128 bits of data from a data share into a vector register.
    ```
    addr = CalcDsAddr(ADDR.b32, 0x0, 0x0);
    RETURN_DATA[31 : 0] = MEM[addr + OFFSET.u32].b32;
    RETURN_DATA[63 : 32] = MEM[addr + OFFSET.u32 + 4U].b32;
    RETURN_DATA[95 : 64] = MEM[addr + OFFSET.u32 + 8U].b32;
    RETURN_DATA[127 : 96] = MEM[addr + OFFSET.u32 + 12U].b32
    ```

- `ds_read_b64_tr_b16`

    Read 64 bits of data per lane from data share. Interpret the data as a matrix with 16 bit elements and transpose
    the matrix. Store the result into vector registers.

    > Note: `ds_read_b64_tr_b16` used for either column major matrix A or row major matrix B data load to 2 VGPRs. Element size is 16b. Two instruction load a complete matrix. The first loads K=0..3 and K=8..11 into two VGPRs, and the next loads K=4..7 and 12..15. Each lane (one VGPR) holds 4 consecutive M or N values.

- `ds_write_b64`

    Store 64 bits of data from a vector input register into a data share.

    ```
    addr = CalcDsAddr(ADDR.b32, 0x0, 0x0);
    MEM[addr + OFFSET.u32].b32 = DATA[31 : 0];
    MEM[addr + OFFSET.u32 + 4U].b32 = DATA[63 : 32]
    ```

- `buffer_store_dword`

    Store 32 bits of data from vector input registers into a buffer surface.

    ```
    addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
    MEM[addr].b32 = VDATA[31 : 0]
    ```

- `buffer_store_dwordx2`

    Store 64 bits of data from vector input registers into a buffer surface.

    ```
    addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
    MEM[addr].b32 = VDATA[31 : 0];
    MEM[addr + 4U].b32 = VDATA[63 : 32]
    ```

- `buffer_store_dwordx4`

    Store 128 bits of data from vector input registers into a buffer surface.

    ```
    addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
    MEM[addr].b32 = VDATA[31 : 0];
    MEM[addr + 4U].b32 = VDATA[63 : 32];
    MEM[addr + 8U].b32 = VDATA[95 : 64];
    MEM[addr + 12U].b32 = VDATA[127 : 96]
    ```

- `buffer_load_dwordx4`

    Load 128 bits of data from a buffer surface into a vector register.
    ```
    addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
    VDATA[31 : 0] = MEM[addr].b32;
    VDATA[63 : 32] = MEM[addr + 4U].b32;
    VDATA[95 : 64] = MEM[addr + 8U].b32;
    VDATA[127 : 96] = MEM[addr + 12U].b32
    ```

- `buffer_load_dwordx2`

    Load 64 bits of data from a buffer surface into a vector register.

    ```
    addr = CalcBufferAddr(VADDR.b32, SRSRC.b32, SOFFSET.b32, OFFSET.b32);
    VDATA[31 : 0] = MEM[addr].b32;
    VDATA[63 : 32] = MEM[addr + 4U].b32
    ```

- `v_permlane16_swap_b32`

    Swap data between two vector registers. Odd rows of the first operand are swapped with even rows of the second operand (one row is 16 lanes).

    ```
    for pass in 0 : 1 do
        for lane in 0 : 15 do
            tmp = VGPR[pass * 32 + lane][SRC0.u32];
            VGPR[pass * 32 + lane][SRC0.u32] = VGPR[pass * 32 + lane + 16][VDST.u32];
            VGPR[pass * 32 + lane + 16][VDST.u32] = tmp
        endfor
    endfor
    ```
- `v_accvgpr_read_b32`

    Move 32 bits of data from an accumulator vector register into an architectural vector register

- `v_mov_b32`

    Move 32-bit data from a vector input into a vector register.
    ```
    D0.b32 = S0.b32
    ```
    > Note: Floating-point modifiers are valid for this instruction if S0 is a 32-bit floating point value. This instruction is suitable for negating or taking the absolute value of a floating-point value.
    
    Functional examples:
    ```
    mov_b32 v0, v1 // Move into v0 from v1
    v_mov_b32 v0, -v1 // Set v0 to the negation of v1
    v_mov_b32 v0, abs(v1) // Set v0 to the absolute value of v1
    ```
- `v_cndmask_b32`

    Copy data from one of two inputs based on the per-lane condition code and store the result into a vector register.

    ```
    D0.u32 = VCC.u64[laneId] ? S1.u32 : S0.u32
    ```

- `v_cvt_pk_bf15_f32`

    Convert from two single-precision float inputs to a packed BF16 value and store the result into a vector register.
    ```
    prev_mode = ROUND_MODE;
    ROUND_MODE = ROUND_NEAREST_EVEN;
    tmp[15 : 0].bf16 = f32_to_bf16(S0.f32);
    tmp[31 : 16].bf16 = f32_to_bf16(S1.f32);
    D0 = tmp.b32;
    ROUND_MODE = prev_mode
    ```

# 2. Arithmetic instructions
- `buffer_atomic_pk_add_bf16`

    Add a packed 2-component BF16 float value in the data register to a location in a buffer surface. Store the original value from buffer surface into a vector register iff the SC0 bit is set.
    ```
    tmp = MEM[ADDR];
    src = DATA;
    dst[31 : 16].bf16 = tmp[31 : 16].bf16 + src[31 : 16].bf16;
    dst[15 : 0].bf16 = tmp[15 : 0].bf16 + src[15 : 0].bf16;
    MEM[ADDR] = dst.b32;
    RETURN_DATA = tmp
    ```
    > Note: Floating-point addition handles NAN/INF/denorm.

- `v_mfma_f32_16x16x32_bf16`

    Multiply the 16x32 matrix in the first input by the 32x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add. Store the resulting matrix into vector registers.
    ```
    D = A (16x32) * B (32x16) + C (16x16)
    ```
    Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher performance. Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.
    >Note: NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not supported. Round toward nearest even semantics.

- `v_mfma_f32_32x32x16_bf16`
    Multiply the 32x16 matrix in the first input by the 16x32 matrix in the second input and add the 32x32 matrix in the third input using fused multiply add. Store the resulting matrix into vector registers.
    ```
    D = A (32x16) * B (16x32) + C (32x32)
    ```
    Each operand contains a single matrix whose elements are distributed across all lanes of the wave. A single matrix multiply is computed and the row-column dot products are distributed across the vector ALU for higher performance. Matrices A and B are BF16 float format. Matrices C and D are single-precision float format.
    >Note: NEG[1:0] and ABS[1:0] must be zero. NEG[2] and ABS[2] may be used to control matrix C. CLAMP is not supported. Round toward nearest even semantics.

- `v_subrev_f32`

    Subtract the first floating point input from the second input and store the result into a vector register.
    ```
    D0.f32 = S1.f32 - S0.f32
    ```
    > Note: 0.5ULP precision, denormals are supported
    - Modifiers
        - quad_perm: [{0..3},{0..3},{0..3},{0..3}] Full permute of 4 threads.
        - row_mask: Controls which rows are enabled for data sharing. By default, all rows are enabled. row_mask:{0..15} specifies a row mask as a positive integer number or an absolute expression. Each of the 4 bits in the mask controls one row (0 - disabled, 1 - enabled). In wave32 mode, the values shall be limited to {0..7}.
        - bank_mask: Controls which banks are enabled for data sharing. By default, all banks are enabled. bank_mask:{0..15} specifies a bank mask as a positive integer number or an absolute expression. Each of the 4 bits in the mask controls one bank (0 - disabled, 1 - enabled).
        - >Note: the lanes of a wavefront are organized in four rows and four banks.

- `v_mul_f32`

    Multiply two floating point inputs and store the result into a vector register.

    ```
    D0.f32 = S0.f32 * S1.f32
    ```
    > Note: 0.5ULP precision, denormals are supported.

- `v_exp_f32`

    Calculate 2 raised to the power of the single-precision float input and store the result into a vector register
    ```
    D0.f32 = pow(2.0F, S0.f32)
    ```
    > Note: 1ULP accuracy, denormals are flushed

    Examples:
    ```
    V_EXP_F32(0xff800000) => 0x00000000 // exp(-INF) = 0
    V_EXP_F32(0x80000000) => 0x3f800000 // exp(-0.0) = 1
    V_EXP_F32(0x7f800000) => 0x7f800000 // exp(+INF) = +IN
    ```
