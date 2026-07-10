	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z11attn_kernel12attn_globalsi ; -- Begin function _Z11attn_kernel12attn_globalsi
	.globl	_Z11attn_kernel12attn_globalsi
	.p2align	8
	.type	_Z11attn_kernel12attn_globalsi,@function
_Z11attn_kernel12attn_globalsi:         ; @_Z11attn_kernel12attn_globalsi
; %bb.0:
	s_load_dwordx2 s[8:9], s[0:1], 0xb0
	s_load_dwordx4 s[12:15], s[0:1], 0xa0
	v_and_b32_e32 v2, 0x180, v0
	v_lshlrev_b32_e32 v4, 4, v0
	v_and_b32_e32 v1, 32, v0
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s5, s14, s8
	v_lshrrev_b32_e32 v11, 2, v0
	v_lshrrev_b32_e32 v7, 1, v0
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[6:7], vcc
; %bb.1:
	v_or_b32_e32 v2, v1, v11
	v_and_b32_e32 v3, 62, v7
	v_bitop3_b32 v2, v2, 48, v4 bitop3:0x48
	v_mad_u64_u32 v[2:3], s[10:11], v3, s5, v[2:3]
; %bb.2:                                ; %_ZN7kittens5groupILi8EE24prefill_swizzled_offsetsILi1ELb0ETkNS_5ducks2st3allENS_2stI14__hip_bfloat16Li32ELi32ENS3_8st_shape8st_32x32EEETkNS3_2gl3allENS_2glIS6_Lin1ELin1ELin1ELin1EJEEEEEvRT1_RKT2_Pj.exit
	s_or_b64 exec, exec, s[6:7]
	s_load_dword s6, s[0:1], 0x130
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s6, 1
	s_cbranch_scc1 .LBB0_7
; %bb.3:                                ; %.lr.ph
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s7, 0, 0
	s_and_b32 s9, s7, -16
	s_load_dwordx2 s[34:35], s[0:1], 0x50
	s_load_dwordx4 s[20:23], s[0:1], 0x40
	s_load_dwordx4 s[16:19], s[0:1], 0x10
	s_load_dwordx2 s[24:25], s[0:1], 0x0
	s_load_dwordx2 s[30:31], s[0:1], 0x20
	s_load_dwordx2 s[26:27], s[0:1], 0x30
	s_load_dwordx2 s[28:29], s[0:1], 0x90
	s_mov_b32 s1, 0
	s_and_b32 s0, s7, 15
	s_add_i32 s9, s9, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s7, s7, s9
	s_add_i32 s9, s7, 0x4000
	s_and_b32 s10, s9, -16
	s_and_b32 s0, s9, 15
	s_add_i32 s10, s10, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s9, s9, s10
	s_add_i32 s10, s9, 0x4000
	s_and_b32 s11, s10, -16
	s_and_b32 s0, s10, 15
	s_add_i32 s11, s11, 16
	s_cmp_eq_u64 s[0:1], 0
	s_cselect_b32 s0, s10, s11
	s_add_i32 s1, s6, 31
	s_ashr_i32 s6, s1, 31
	v_lshrrev_b32_e32 v6, 6, v0
	s_lshr_b32 s6, s6, 27
	v_lshlrev_b32_e32 v5, 4, v6
	s_add_i32 s1, s1, s6
	v_and_b32_e32 v3, 0x3c0, v4
	v_lshlrev_b32_e32 v8, 10, v6
	s_movk_i32 s6, 0x400
	v_and_b32_e32 v9, 16, v5
	v_and_or_b32 v3, v8, s6, v3
	v_bitop3_b32 v1, v9, v4, v1 bitop3:0x36
	v_cmp_gt_u32_e32 vcc, 2, v6
	v_add_u32_e32 v6, s0, v8
	v_lshlrev_b32_e32 v8, 6, v0
	v_lshrrev_b32_e32 v3, 6, v3
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v5, 0x60, v5
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s11, s18, s30
	s_mul_i32 s13, s22, s34
	v_and_b32_e32 v8, 0x7c0, v8
	v_and_b32_e32 v7, 16, v7
	v_lshlrev_b32_e32 v12, 2, v0
	v_and_b32_e32 v13, 16, v0
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_or_b32 v5, v1, 24, v5
	v_mul_lo_u32 v1, v3, s11
	v_mul_lo_u32 v3, v3, s13
	v_or_b32_e32 v9, v7, v8
	v_and_or_b32 v10, v12, 32, v13
	v_and_b32_e32 v0, 4, v0
	v_add_lshl_u32 v1, v1, v5, 1
	v_add_lshl_u32 v3, v3, v5, 1
	v_and_b32_e32 v5, 0x1c00, v4
	v_bitop3_b32 v14, v7, v10, v8 bitop3:0x36
	v_bitop3_b32 v10, v9, v10, 32 bitop3:0x36
	v_and_or_b32 v0, v11, 3, v0
	v_and_or_b32 v11, v12, 12, v13
	v_add_u32_e32 v4, s7, v5
	v_add_u32_e32 v7, s7, v14
	v_add_u32_e32 v8, s7, v10
	v_add_u32_e32 v9, s9, v14
	v_lshlrev_b32_e32 v13, 6, v0
	v_lshlrev_b32_e32 v14, 1, v11
	s_movk_i32 s7, 0x220
	v_bitop3_b32 v11, v13, s7, v14 bitop3:0x36
	s_movk_i32 s7, 0x410
	v_bitop3_b32 v12, v13, s7, v14 bitop3:0x36
	s_movk_i32 s7, 0x630
	v_or_b32_e32 v0, v13, v14
	v_bitop3_b32 v13, v13, s7, v14 bitop3:0x36
	v_add_u32_e32 v0, s0, v0
	v_add_u32_e32 v11, s0, v11
	v_add_u32_e32 v12, s0, v12
	v_add_u32_e32 v13, s0, v13
	s_mul_i32 s0, s4, s12
	s_mul_i32 s0, s0, s14
	s_add_i32 s0, s3, s0
	s_lshl_b32 s15, s2, 5
	s_mul_i32 s0, s0, s8
	s_add_i32 s12, s0, s15
	s_mul_i32 s0, s4, s20
	s_mul_i32 s0, s0, s22
	s_add_i32 s0, s3, s0
	s_mul_i32 s14, s0, s34
	s_mul_i32 s0, s4, s16
	s_mul_i32 s0, s0, s18
	s_ashr_i32 s1, s1, 5
	s_add_i32 s0, s3, s0
	s_lshl_b32 s6, s11, 6
	s_lshl_b32 s10, s13, 6
	v_add_u32_e32 v5, s9, v5
	s_lshl_b32 s2, s5, 6
	v_add_u32_e32 v10, s9, v10
	s_max_i32 s19, s1, 1
	s_lshl_b32 s21, s5, 5
	s_lshl_b32 s20, s13, 5
	s_mul_i32 s16, s0, s30
	s_lshl_b32 s18, s11, 5
	s_mov_b32 s7, 0x110000
	s_branch .LBB0_5
.LBB0_4:                                ; %_ZN7kittens5groupILi8EE4loadILi1ELb0ETkNS_5ducks2st3allENS_2stI14__hip_bfloat16Li32ELi32ENS3_8st_shape8st_32x32EEETkNS3_2gl3allENS_2glIS6_Lin1ELin1ELin1ELin1EJEEETkNS3_5coord4tileENS_5coordIS9_EEEEvRT1_RKT2_RKT3_PKj.exit
                                        ;   in Loop: Header=BB0_5 Depth=1
	s_or_b64 exec, exec, s[4:5]
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v7 offset:0

	;;#ASMEND
	s_add_i32 s19, s19, -1
	;;#ASMSTART
	ds_read_b128 v[14:17], v7 offset:0x800

	;;#ASMEND
	s_add_i32 s12, s12, s21
	;;#ASMSTART
	ds_read_b128 v[14:17], v7 offset:0x1000

	;;#ASMEND
	s_add_i32 s14, s14, s20
	;;#ASMSTART
	ds_read_b128 v[14:17], v7 offset:0x1800

	;;#ASMEND
	s_add_i32 s16, s16, s18
	;;#ASMSTART
	ds_read_b128 v[14:17], v8 offset:0

	;;#ASMEND
	s_cmp_eq_u32 s19, 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v8 offset:0x800

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v8 offset:0x1000

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v8 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(1)
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v9 offset:0

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v9 offset:0x800

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v9 offset:0x1000

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v9 offset:0x1800

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v10 offset:0

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v10 offset:0x800

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v10 offset:0x1000

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b128 v[14:17], v10 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v0 offset:0

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v11 offset:0

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v12 offset:0

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_read_b64_tr_b16 v[14:15], v13 offset:0

	;;#ASMEND
	s_cbranch_scc1 .LBB0_7
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[0:1], s[16:17], 1
	s_add_u32 s4, s24, s0
	s_addc_u32 s5, s25, s1
	v_readfirstlane_b32 s0, v4
	s_ashr_i32 s15, s14, 31
	s_mov_b32 m0, s0
	s_lshl_b64 s[0:1], s[14:15], 1
	s_add_u32 s8, s26, s0
	v_readfirstlane_b32 s0, v5
	buffer_load_dwordx4 v1, s[4:7], 0 offen lds
	s_addc_u32 s9, s27, s1
	s_mov_b32 s11, s7
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v3, s[8:11], 0 offen lds
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_4
; %bb.6:                                ;   in Loop: Header=BB0_5 Depth=1
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[0:1], s[12:13], 1
	s_add_u32 s0, s28, s0
	v_readfirstlane_b32 s8, v6
	s_addc_u32 s1, s29, s1
	s_mov_b32 s3, s7
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v2, s[0:3], 0 offen lds
	s_branch .LBB0_4
.LBB0_7:                                ; %._crit_edge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11attn_kernel12attn_globalsi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 308
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 18
		.amdhsa_next_free_sgpr 36
		.amdhsa_accum_offset 20
		.amdhsa_reserve_vcc 1
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
	.size	_Z11attn_kernel12attn_globalsi, .Lfunc_end0-_Z11attn_kernel12attn_globalsi
                                        ; -- End function
	.set _Z11attn_kernel12attn_globalsi.num_vgpr, 18
	.set _Z11attn_kernel12attn_globalsi.num_agpr, 0
	.set _Z11attn_kernel12attn_globalsi.numbered_sgpr, 36
	.set _Z11attn_kernel12attn_globalsi.private_seg_size, 0
	.set _Z11attn_kernel12attn_globalsi.uses_vcc, 1
	.set _Z11attn_kernel12attn_globalsi.uses_flat_scratch, 0
	.set _Z11attn_kernel12attn_globalsi.has_dyn_sized_stack, 0
	.set _Z11attn_kernel12attn_globalsi.has_recursion, 0
	.set _Z11attn_kernel12attn_globalsi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1236
; TotalNumSgprs: 42
; NumVgprs: 18
; NumAgprs: 0
; TotalNumVgprs: 18
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 42
; NumVGPRsForWavesPerEU: 18
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_ec7c5fe0bf1ad600,@object ; @__hip_cuid_ec7c5fe0bf1ad600
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ec7c5fe0bf1ad600
__hip_cuid_ec7c5fe0bf1ad600:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ec7c5fe0bf1ad600, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_ec7c5fe0bf1ad600
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           304
        .value_kind:     by_value
      - .offset:         304
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 308
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z11attn_kernel12attn_globalsi
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         _Z11attn_kernel12attn_globalsi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     18
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
