	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:                                ; %entry
	s_load_dwordx8 s[4:11], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_load_dword s9, s[0:1], 0x20
	s_load_dwordx2 s[2:3], s[0:1], 0x30
	s_load_dword s11, s[0:1], 0x50
	v_lshrrev_b32_e32 v3, 2, v0
	v_and_b32_e32 v2, 15, v0
	v_and_b32_e32 v4, 12, v3
	s_mul_i32 s0, s6, s8
	s_mul_i32 s0, s0, s10
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s0, s9
	s_lshl_b32 s6, s0, 1
	v_mad_u64_u32 v[6:7], s[0:1], v2, s9, v[4:5]
	s_mov_b32 s7, 0x20000
	v_lshlrev_b32_e32 v1, 1, v6
	s_lshl_b32 s0, s9, 4
	v_add_lshl_u32 v5, v6, s0, 1
	buffer_load_dwordx2 v[6:7], v1, s[4:7], 0 offen
	buffer_load_dwordx2 v[8:9], v5, s[4:7], 0 offen
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	; sched_barrier mask(0x00000000)
	v_lshl_or_b32 v1, v0, 4, v3
	v_and_b32_e32 v5, 12, v0
	s_movk_i32 s0, 0xfc
	s_cmp_lg_u32 0, -1
	v_bitop3_b32 v1, v1, v5, s0 bitop3:0x6c
	s_cselect_b32 s0, 0, 0
	s_add_i32 s0, s0, 15
	s_and_b32 s0, s0, -16
	v_lshl_add_u32 v1, v1, 1, s0
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v1, v[6:7] offset:0

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v1, v[8:9] offset:0x200

	;;#ASMEND
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v1, 3, v0
	v_lshrrev_b32_e32 v0, 1, v0
	v_and_b32_e32 v0, 24, v0
	v_xad_u32 v0, v1, v0, s0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[0:1], v0 offset:0
ds_read_b64_tr_b16 v[6:7], v0 offset:0x200

	;;#ASMEND
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	; sched_barrier mask(0x00000000)
	v_mad_u64_u32 v[4:5], s[0:1], v4, s11, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[8:9], v[4:5], 1, s[2:3]
	v_add_u32_e32 v4, s11, v4
	v_ashrrev_i32_e32 v5, 31, v4
	global_store_short v[8:9], v0, off
	v_lshl_add_u64 v[8:9], v[4:5], 1, s[2:3]
	v_add_u32_e32 v4, s11, v4
	v_ashrrev_i32_e32 v5, 31, v4
	global_store_short_d16_hi v[8:9], v0, off
	v_lshl_add_u64 v[8:9], v[4:5], 1, s[2:3]
	v_or_b32_e32 v0, 3, v3
	global_store_short v[8:9], v1, off
	v_mad_u64_u32 v[8:9], s[0:1], v0, s11, v[2:3]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[2:3]
	global_store_short_d16_hi v[8:9], v1, off
	v_mad_u64_u32 v[0:1], s[0:1], s11, 14, v[4:5]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[4:5], v[0:1], 1, s[2:3]
	v_add_u32_e32 v0, s11, v0
	v_ashrrev_i32_e32 v1, 31, v0
	global_store_short v[4:5], v6, off
	v_lshl_add_u64 v[4:5], v[0:1], 1, s[2:3]
	v_add_u32_e32 v0, s11, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	global_store_short_d16_hi v[4:5], v6, off
	global_store_short v[0:1], v7, off
	v_or_b32_e32 v0, 19, v3
	v_mad_u64_u32 v[0:1], s[0:1], v0, s11, v[2:3]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	global_store_short_d16_hi v[0:1], v7, off
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tk13micro_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 12
		.amdhsa_accum_offset 12
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z8micro_tk13micro_globals, .Lfunc_end0-_Z8micro_tk13micro_globals
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 492
; NumSgprs: 18
; NumVgprs: 10
; NumAgprs: 0
; TotalNumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 10
; AccumOffset: 12
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 2
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_271c5135db511982,@object ; @__hip_cuid_271c5135db511982
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_271c5135db511982
__hip_cuid_271c5135db511982:
	.byte	0                               ; 0x0
	.size	__hip_cuid_271c5135db511982, 1

	.ident	"AMD clang version 19.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25164 2b159522a6e9b34fe13b1d7b4c4ae751ef122765)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_271c5135db511982
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z8micro_tk13micro_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
