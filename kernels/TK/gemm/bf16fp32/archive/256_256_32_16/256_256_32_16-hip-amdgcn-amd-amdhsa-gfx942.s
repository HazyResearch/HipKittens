	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s18, 0, 0
	s_and_b32 s8, s18, -16
	s_load_dwordx2 s[22:23], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x20
	s_load_dwordx2 s[14:15], s[0:1], 0x30
	s_load_dwordx2 s[4:5], s[0:1], 0x50
	s_add_i32 s19, s8, 16
	s_load_dwordx2 s[8:9], s[0:1], 0x60
	s_load_dwordx2 s[10:11], s[0:1], 0x80
	s_load_dword s20, s[0:1], 0x90
	s_mov_b32 s17, 0
	s_and_b32 s16, s18, 15
	s_cmp_eq_u64 s[16:17], 0
	s_cselect_b32 s21, s18, s19
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s20, s3
	s_add_i32 s0, s0, s2
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 29
	s_add_i32 s1, s0, s1
	s_ashr_i32 s1, s1, 3
	s_lshl_b32 s0, s0, 7
	s_mulk_i32 s1, 0xfc01
	s_add_i32 s0, s1, s0
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 23
	s_add_i32 s1, s0, s1
	s_ashr_i32 s2, s1, 9
	s_lshl_b32 s2, s2, 4
	s_sub_i32 s3, 32, s2
	s_min_i32 s3, s3, 16
	s_abs_i32 s18, s3
	v_cvt_f32_u32_e32 v1, s18
	s_sub_i32 s20, 0, s18
	s_add_i32 s24, s21, 0x4000
	s_and_b32 s1, s1, 0xfffffe00
	v_rcp_iflag_f32_e32 v1, v1
	s_and_b32 s19, s24, -16
	s_sub_i32 s0, s0, s1
	s_add_i32 s25, s19, 16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_abs_i32 s19, s0
	s_xor_b32 s1, s0, s3
	s_and_b32 s16, s24, 15
	v_readfirstlane_b32 s26, v1
	s_mul_i32 s20, s20, s26
	s_mul_hi_u32 s20, s26, s20
	s_add_i32 s26, s26, s20
	s_mul_hi_u32 s20, s19, s26
	s_mul_i32 s26, s20, s18
	s_sub_i32 s19, s19, s26
	s_ashr_i32 s1, s1, 31
	s_add_i32 s26, s20, 1
	s_sub_i32 s27, s19, s18
	s_cmp_ge_u32 s19, s18
	s_cselect_b32 s20, s26, s20
	s_cselect_b32 s19, s27, s19
	s_add_i32 s26, s20, 1
	s_cmp_ge_u32 s19, s18
	s_cselect_b32 s18, s26, s20
	s_xor_b32 s18, s18, s1
	s_sub_i32 s18, s18, s1
	s_mul_i32 s1, s18, s3
	s_sub_i32 s0, s0, s1
	s_add_i32 s2, s2, s0
	s_lshl_b32 s20, s2, 8
	s_ashr_i32 s19, s20, 31
	s_mul_i32 s0, s6, s19
	s_mul_hi_u32 s1, s6, s20
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s7, s20
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s6, s20
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s22, s0
	s_addc_u32 s1, s23, s1
	s_lshl_b32 s2, s18, 8
	s_ashr_i32 s3, s2, 31
	s_mul_i32 s3, s4, s3
	s_mul_hi_u32 s7, s4, s2
	s_add_i32 s3, s7, s3
	s_mul_i32 s5, s5, s2
	s_add_i32 s3, s3, s5
	s_mul_i32 s2, s4, s2
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_u32 s14, s14, s2
	v_and_b32_e32 v16, 24, v2
	v_lshrrev_b32_e32 v11, 2, v0
	s_addc_u32 s15, s15, s3
	v_mad_u64_u32 v[6:7], s[2:3], v11, s6, v[16:17]
	v_or_b32_e32 v17, 0x80, v11
	v_ashrrev_i32_e32 v7, 31, v6
	v_mad_u64_u32 v[8:9], s[2:3], v17, s6, v[16:17]
	v_lshlrev_b32_e32 v10, 4, v0
	v_lshlrev_b32_e32 v20, 1, v16
	v_lshl_add_u64 v[2:3], v[6:7], 1, s[0:1]
	v_ashrrev_i32_e32 v9, 31, v8
	v_and_b32_e32 v7, 0x1fc0, v10
	v_add_u32_e32 v18, s21, v20
	v_lshl_add_u64 v[12:13], v[8:9], 1, s[0:1]
	v_add_u32_e32 v9, v18, v7
	v_or_b32_e32 v21, 8, v20
	v_lshrrev_b32_e32 v22, 4, v9
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_add_u32_e32 v19, s21, v21
	v_and_b32_e32 v22, 56, v22
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[12:13], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v9, v22, v9
	;;#ASMSTART
	ds_write_b64 v9, v[2:3]

	;;#ASMEND
	v_add_u32_e32 v2, v19, v7
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	v_or_b32_e32 v9, 0x2000, v7
	;;#ASMSTART
	ds_write_b64 v2, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v2, v18, v9
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v2, v19, v9
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[14:15]

	;;#ASMEND
	v_mad_u64_u32 v[2:3], s[2:3], v11, s4, v[16:17]
	s_cmp_eq_u64 s[16:17], 0
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[14:15]
	s_cselect_b32 s5, s24, s25
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[2:3], v17, s4, v[16:17]
	v_add_u32_e32 v3, s5, v20
	v_ashrrev_i32_e32 v5, 31, v4
	v_add_u32_e32 v20, v3, v7
	v_lshl_add_u64 v[16:17], v[4:5], 1, s[14:15]
	v_add_u32_e32 v5, s5, v21
	v_lshrrev_b32_e32 v21, 4, v20
	v_and_b32_e32 v21, 56, v21
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v20, v21, v20
	;;#ASMSTART
	ds_write_b64 v20, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v12, v5, v7
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 56, v13
	v_xor_b32_e32 v12, v13, v12
	v_add_u32_e32 v3, v3, v9
	;;#ASMSTART
	ds_write_b64 v12, v[14:15]

	;;#ASMEND
	v_lshrrev_b32_e32 v12, 4, v3
	v_and_b32_e32 v12, 56, v12
	v_xor_b32_e32 v3, v12, v3
	;;#ASMSTART
	ds_write_b64 v3, v[16:17]

	;;#ASMEND
	v_add_u32_e32 v3, v5, v9
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v3, v5, v3
	;;#ASMSTART
	ds_write_b64 v3, v[18:19]

	;;#ASMEND
	v_lshrrev_b32_e32 v1, 8, v0
	s_mov_b64 s[12:13], 0
	v_cmp_eq_u32_e32 vcc, 1, v1
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[2:3]
	v_bfe_u32 v3, v0, 6, 2
	v_lshlrev_b32_e32 v147, 5, v3
	v_and_b32_e32 v146, 15, v0
	v_and_b32_e32 v196, 12, v11
	v_or_b32_e32 v3, v147, v146
	v_lshlrev_b32_e32 v5, 1, v196
	v_lshl_add_u32 v3, v3, 6, s5
	v_add_u32_e32 v11, v3, v5
	v_lshlrev_b32_e32 v199, 1, v6
	v_or_b32_e32 v6, 16, v146
	v_lshrrev_b32_e32 v12, 4, v11
	v_lshlrev_b32_e32 v198, 1, v8
	v_or_b32_e32 v8, v147, v6
	v_and_b32_e32 v12, 56, v12
	v_lshl_add_u32 v8, v8, 6, s5
	v_xor_b32_e32 v197, v12, v11
	v_add_u32_e32 v11, v8, v5
	v_lshrrev_b32_e32 v12, 4, v11
	v_and_b32_e32 v12, 56, v12
	v_xor_b32_e32 v200, v12, v11
	v_or_b32_e32 v11, 0x80, v147
	v_or_b32_e32 v12, v11, v146
	v_lshl_add_u32 v12, v12, 6, s5
	v_add_u32_e32 v13, v12, v5
	v_lshrrev_b32_e32 v14, 4, v13
	v_or_b32_e32 v11, v11, v6
	v_and_b32_e32 v14, 56, v14
	v_lshl_add_u32 v11, v11, 6, s5
	v_xor_b32_e32 v201, v14, v13
	v_add_u32_e32 v13, v11, v5
	v_lshrrev_b32_e32 v14, 4, v13
	v_and_b32_e32 v14, 56, v14
	v_xor_b32_e32 v202, v14, v13
	v_lshlrev_b32_e32 v13, 5, v1
	v_lshlrev_b32_e32 v212, 1, v2
	v_or_b32_e32 v2, 32, v5
	v_or_b32_e32 v14, v13, v146
	v_add_u32_e32 v3, v3, v2
	v_lshl_add_u32 v14, v14, 6, s21
	v_lshlrev_b32_e32 v211, 1, v4
	v_lshrrev_b32_e32 v4, 4, v3
	v_add_u32_e32 v15, v14, v5
	v_and_b32_e32 v4, 56, v4
	v_lshrrev_b32_e32 v16, 4, v15
	v_xor_b32_e32 v213, v4, v3
	v_add_u32_e32 v3, v8, v2
	v_and_b32_e32 v16, 56, v16
	v_lshrrev_b32_e32 v4, 4, v3
	v_xor_b32_e32 v203, v16, v15
	v_or_b32_e32 v15, v6, v13
	v_and_b32_e32 v4, 56, v4
	v_lshl_add_u32 v15, v15, 6, s21
	v_xor_b32_e32 v214, v4, v3
	v_add_u32_e32 v3, v12, v2
	v_add_u32_e32 v16, v15, v5
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshrrev_b32_e32 v17, 4, v16
	v_and_b32_e32 v4, 56, v4
	v_and_b32_e32 v17, 56, v17
	v_xor_b32_e32 v215, v4, v3
	v_add_u32_e32 v3, v11, v2
	v_xor_b32_e32 v204, v17, v16
	v_or_b32_e32 v16, 64, v13
	v_lshrrev_b32_e32 v4, 4, v3
	v_or_b32_e32 v17, v16, v146
	v_and_b32_e32 v4, 56, v4
	v_lshl_add_u32 v17, v17, 6, s21
	v_xor_b32_e32 v216, v4, v3
	v_add_u32_e32 v3, v14, v2
	v_add_u32_e32 v18, v17, v5
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshrrev_b32_e32 v19, 4, v18
	v_or_b32_e32 v16, v16, v6
	v_and_b32_e32 v4, 56, v4
	v_and_b32_e32 v19, 56, v19
	v_lshl_add_u32 v16, v16, 6, s21
	v_xor_b32_e32 v217, v4, v3
	v_add_u32_e32 v3, v15, v2
	v_xor_b32_e32 v205, v19, v18
	v_add_u32_e32 v18, v16, v5
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshrrev_b32_e32 v19, 4, v18
	v_and_b32_e32 v4, 56, v4
	v_and_b32_e32 v19, 56, v19
	v_xor_b32_e32 v218, v4, v3
	v_add_u32_e32 v3, v17, v2
	v_xor_b32_e32 v206, v19, v18
	v_or_b32_e32 v18, 0x80, v13
	v_lshrrev_b32_e32 v4, 4, v3
	v_or_b32_e32 v19, v18, v146
	v_and_b32_e32 v4, 56, v4
	v_lshl_add_u32 v19, v19, 6, s21
	v_xor_b32_e32 v219, v4, v3
	v_add_u32_e32 v3, v16, v2
	v_add_u32_e32 v20, v19, v5
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshrrev_b32_e32 v21, 4, v20
	v_or_b32_e32 v18, v18, v6
	v_and_b32_e32 v4, 56, v4
	v_and_b32_e32 v21, 56, v21
	v_lshl_add_u32 v18, v18, 6, s21
	v_xor_b32_e32 v220, v4, v3
	v_add_u32_e32 v3, v19, v2
	v_xor_b32_e32 v207, v21, v20
	v_add_u32_e32 v20, v18, v5
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshrrev_b32_e32 v21, 4, v20
	v_and_b32_e32 v4, 56, v4
	v_and_b32_e32 v21, 56, v21
	v_or_b32_e32 v13, 0xc0, v13
	v_xor_b32_e32 v221, v4, v3
	v_add_u32_e32 v3, v18, v2
	v_xor_b32_e32 v208, v21, v20
	v_or_b32_e32 v20, v13, v146
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshl_add_u32 v20, v20, 6, s21
	v_and_b32_e32 v4, 56, v4
	v_or_b32_e32 v6, v13, v6
	v_xor_b32_e32 v222, v4, v3
	v_add_u32_e32 v3, v20, v2
	v_lshl_add_u32 v6, v6, 6, s21
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 56, v4
	v_add_u32_e32 v2, v6, v2
	v_xor_b32_e32 v223, v4, v3
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 56, v3
	v_xor_b32_e32 v224, v3, v2
	v_and_b32_e32 v2, 48, v10
	v_add_u32_e32 v3, s21, v2
	v_add_u32_e32 v13, v6, v5
	v_add_u32_e32 v6, v3, v7
	v_or_b32_e32 v4, 8, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_add_u32_e32 v21, v20, v5
	v_add_u32_e32 v5, s21, v4
	v_and_b32_e32 v8, 56, v8
	v_xor_b32_e32 v225, v8, v6
	v_add_u32_e32 v6, v5, v7
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 56, v8
	v_add_u32_e32 v3, v3, v9
	v_xor_b32_e32 v226, v8, v6
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 56, v6
	v_xor_b32_e32 v227, v6, v3
	v_add_u32_e32 v3, v5, v9
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 56, v5
	v_add_u32_e32 v2, s5, v2
	v_xor_b32_e32 v228, v5, v3
	v_add_u32_e32 v3, s5, v4
	v_add_u32_e32 v4, v2, v7
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 56, v5
	v_xor_b32_e32 v229, v5, v4
	v_add_u32_e32 v4, v3, v7
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 56, v5
	v_add_u32_e32 v2, v2, v9
	v_xor_b32_e32 v230, v5, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 56, v4
	s_lshl_b32 s2, s6, 9
	v_lshrrev_b32_e32 v22, 4, v21
	s_lshl_b32 s6, s4, 9
	v_xor_b32_e32 v231, v4, v2
	v_add_u32_e32 v2, v3, v9
	v_and_b32_e32 v22, 56, v22
	v_lshrrev_b32_e32 v3, 4, v2
	s_add_u32 s14, s14, 64
	v_xor_b32_e32 v209, v22, v21
	v_lshrrev_b32_e32 v21, 4, v13
	v_and_b32_e32 v3, 56, v3
	s_addc_u32 s15, s15, 0
	v_and_b32_e32 v21, 56, v21
	v_xor_b32_e32 v232, v3, v2
	s_add_u32 s16, s0, 64
	v_mov_b32_e32 v2, 0
	v_mov_b64_e32 v[140:141], 0
	v_xor_b32_e32 v210, v21, v13
	s_addc_u32 s17, s1, 0
	s_mov_b32 s3, 0x110000
	s_mov_b32 s21, 0xffff
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v70, v2
	v_mov_b32_e32 v71, v2
	v_mov_b32_e32 v72, v2
	v_mov_b32_e32 v73, v2
	v_mov_b32_e32 v74, v2
	v_mov_b32_e32 v75, v2
	v_mov_b32_e32 v76, v2
	v_mov_b32_e32 v77, v2
	v_mov_b32_e32 v78, v2
	v_mov_b32_e32 v79, v2
	v_mov_b32_e32 v80, v2
	v_mov_b32_e32 v81, v2
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v2
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v85, v2
	v_mov_b32_e32 v86, v2
	v_mov_b32_e32 v87, v2
	v_mov_b32_e32 v88, v2
	v_mov_b32_e32 v89, v2
	v_mov_b32_e32 v90, v2
	v_mov_b32_e32 v91, v2
	v_mov_b32_e32 v92, v2
	v_mov_b32_e32 v93, v2
	v_mov_b32_e32 v94, v2
	v_mov_b32_e32 v95, v2
	v_mov_b32_e32 v96, v2
	v_mov_b32_e32 v97, v2
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v99, v2
	v_mov_b32_e32 v100, v2
	v_mov_b32_e32 v101, v2
	v_mov_b32_e32 v102, v2
	v_mov_b32_e32 v103, v2
	v_mov_b32_e32 v104, v2
	v_mov_b32_e32 v105, v2
	v_mov_b32_e32 v106, v2
	v_mov_b32_e32 v107, v2
	v_mov_b32_e32 v108, v2
	v_mov_b32_e32 v109, v2
	v_mov_b32_e32 v110, v2
	v_mov_b32_e32 v111, v2
	v_mov_b32_e32 v112, v2
	v_mov_b32_e32 v113, v2
	v_mov_b32_e32 v114, v2
	v_mov_b32_e32 v115, v2
	v_mov_b32_e32 v116, v2
	v_mov_b32_e32 v117, v2
	v_mov_b32_e32 v118, v2
	v_mov_b32_e32 v119, v2
	v_mov_b32_e32 v120, v2
	v_mov_b32_e32 v121, v2
	v_mov_b32_e32 v122, v2
	v_mov_b32_e32 v123, v2
	v_mov_b32_e32 v124, v2
	v_mov_b32_e32 v125, v2
	v_mov_b32_e32 v134, v2
	v_mov_b32_e32 v135, v2
	v_mov_b32_e32 v136, v2
	v_mov_b32_e32 v137, v2
	v_mov_b64_e32 v[128:129], v[140:141]
	v_mov_b64_e32 v[144:145], v[140:141]
	v_mov_b64_e32 v[132:133], v[140:141]
	s_branch .LBB0_4
.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v178, s21, v178, v178
	v_bfi_b32 v176, s21, v176, v176
	v_bfi_b32 v174, s21, v174, v174
	v_bfi_b32 v172, s21, v172, v172
	v_bfi_b32 v194, s21, v194, v194
	v_bfi_b32 v192, s21, v192, v192
	v_bfi_b32 v190, s21, v190, v190
	v_bfi_b32 v188, s21, v188, v188
	v_bfi_b32 v186, s21, v186, v186
	v_bfi_b32 v184, s21, v184, v184
	v_bfi_b32 v182, s21, v182, v182
	v_bfi_b32 v180, s21, v180, v180
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[134:137], v[194:195], v[178:179], v[134:137]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[194:195], v[176:177], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[192:193], v[178:179], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[192:193], v[176:177], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[194:195], v[174:175], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[194:195], v[172:173], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[192:193], v[174:175], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[192:193], v[172:173], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[190:191], v[178:179], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[190:191], v[176:177], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[188:189], v[178:179], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[188:189], v[176:177], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[190:191], v[174:175], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[190:191], v[172:173], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[188:189], v[174:175], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[188:189], v[172:173], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[186:187], v[178:179], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[186:187], v[176:177], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[184:185], v[178:179], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[184:185], v[176:177], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[186:187], v[174:175], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[186:187], v[172:173], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[184:185], v[174:175], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[184:185], v[172:173], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[182:183], v[178:179], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[182:183], v[176:177], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[180:181], v[178:179], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[180:181], v[176:177], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[182:183], v[174:175], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[182:183], v[172:173], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[180:181], v[174:175], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[180:181], v[172:173], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s12, 64
	s_addc_u32 s13, s13, 0
	s_cmpk_eq_i32 s12, 0x4000
	s_cbranch_scc1 .LBB0_10
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_cmpk_lg_i32 s12, 0x3fc0
	s_cselect_b64 s[4:5], -1, 0
	s_cmpk_eq_i32 s12, 0x3fc0
	s_cbranch_scc1 .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s0, s16, s12
	s_addc_u32 s1, s17, s13
	buffer_load_dwordx4 v[126:129], v199, s[0:3], 0 offen
	buffer_load_dwordx4 v[138:141], v198, s[0:3], 0 offen
.LBB0_6:                                ;   in Loop: Header=BB0_4 Depth=1
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[154:155], v197

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v201

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v202

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[170:171], v203

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[168:169], v204

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[166:167], v205

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v206

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v207

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v208

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v209

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[156:157], v210

	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cndmask_b32_e64 v172, 0, 1, s[4:5]
	v_cmp_ne_u32_e64 s[0:1], 1, v172
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s4, s14, s12
	s_addc_u32 s5, s15, s13
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v[130:133], v212, s[4:7], 0 offen
	buffer_load_dwordx4 v[142:145], v211, s[4:7], 0 offen
.LBB0_8:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v154, s21, v154, v154
	v_bfi_b32 v152, s21, v152, v152
	v_bfi_b32 v150, s21, v150, v150
	v_bfi_b32 v148, s21, v148, v148
	v_bfi_b32 v170, s21, v170, v170
	v_bfi_b32 v168, s21, v168, v168
	v_bfi_b32 v166, s21, v166, v166
	v_bfi_b32 v164, s21, v164, v164
	v_bfi_b32 v162, s21, v162, v162
	v_bfi_b32 v160, s21, v160, v160
	v_bfi_b32 v158, s21, v158, v158
	v_bfi_b32 v156, s21, v156, v156
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[178:179], v213

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v214

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[174:175], v215

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[172:173], v216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v217

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v218

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[190:191], v219

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v220

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[186:187], v221

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[184:185], v222

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[182:183], v223

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[180:181], v224

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[134:137], v[170:171], v[154:155], v[134:137]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[170:171], v[152:153], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[168:169], v[154:155], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[168:169], v[152:153], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[170:171], v[150:151], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[170:171], v[148:149], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[168:169], v[150:151], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[168:169], v[148:149], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[166:167], v[154:155], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[166:167], v[152:153], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[164:165], v[154:155], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[164:165], v[152:153], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[166:167], v[150:151], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[166:167], v[148:149], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[164:165], v[150:151], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[164:165], v[148:149], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[162:163], v[154:155], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[162:163], v[152:153], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[160:161], v[154:155], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[160:161], v[152:153], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[162:163], v[150:151], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[162:163], v[148:149], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[160:161], v[150:151], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[160:161], v[148:149], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[158:159], v[154:155], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[158:159], v[152:153], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[156:157], v[154:155], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[156:157], v[152:153], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[158:159], v[150:151], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[158:159], v[148:149], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[156:157], v[150:151], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[156:157], v[148:149], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_3
; %bb.9:                                ;   in Loop: Header=BB0_4 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v225, v[126:127]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v226, v[128:129]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v227, v[138:139]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v228, v[140:141]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v229, v[130:131]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v230, v[132:133]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v231, v[142:143]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v232, v[144:145]

	;;#ASMEND
	s_branch .LBB0_3
.LBB0_10:
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	s_barrier
.LBB0_12:
	s_or_b64 exec, exec, s[0:1]
	v_mul_lo_u32 v0, v196, s10
	v_lshl_or_b32 v172, v1, 5, s20
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v126, s10, v0
	v_lshl_or_b32 v162, s18, 8, v147
	s_mul_i32 s2, s10, s19
	v_mul_lo_u32 v1, s11, v172
	v_mad_u64_u32 v[164:165], s[0:1], s10, v172, 0
	v_add_u32_e32 v128, s10, v126
	v_add_u32_e32 v132, v0, v146
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v138, v126, v146
	v_add3_u32 v165, v165, s2, v1
	v_ashrrev_i32_e32 v163, 31, v162
	v_add_u32_e32 v130, s10, v128
	v_ashrrev_i32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v139, 31, v138
	v_lshl_add_u64 v[164:165], v[164:165], 2, s[8:9]
	v_lshlrev_b64 v[162:163], 2, v[162:163]
	v_add_u32_e32 v140, v128, v146
	v_add_u32_e32 v142, v130, v146
	v_lshl_add_u64 v[164:165], v[164:165], 0, v[162:163]
	v_lshlrev_b64 v[132:133], 2, v[132:133]
	v_lshlrev_b64 v[138:139], 2, v[138:139]
	v_ashrrev_i32_e32 v141, 31, v140
	v_ashrrev_i32_e32 v143, 31, v142
	v_lshl_add_u64 v[166:167], v[164:165], 0, v[132:133]
	v_lshl_add_u64 v[168:169], v[164:165], 0, v[138:139]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mov_b32_e32 v147, 0
	global_store_dword v[166:167], v134, off
	global_store_dword v[168:169], v135, off
	v_lshlrev_b64 v[134:135], 2, v[140:141]
	v_lshlrev_b64 v[142:143], 2, v[142:143]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[146:147]
	v_ashrrev_i32_e32 v127, 31, v126
	v_lshl_add_u64 v[140:141], v[164:165], 0, v[134:135]
	v_lshl_add_u64 v[170:171], v[164:165], 0, v[142:143]
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_lshl_add_u64 v[126:127], v[126:127], 0, v[146:147]
	global_store_dword v[140:141], v136, off
	global_store_dword v[170:171], v137, off
	v_lshl_add_u64 v[136:137], v[164:165], 0, v[0:1]
	v_lshlrev_b64 v[126:127], 2, v[126:127]
	global_store_dword v[136:137], v122, off offset:64
	v_lshl_add_u64 v[136:137], v[164:165], 0, v[126:127]
	v_ashrrev_i32_e32 v129, 31, v128
	global_store_dword v[136:137], v123, off offset:64
	v_lshl_add_u64 v[122:123], v[128:129], 0, v[146:147]
	v_lshlrev_b64 v[122:123], 2, v[122:123]
	v_mad_u64_u32 v[144:145], s[0:1], s10, 13, v[130:131]
	v_lshl_add_u64 v[128:129], v[164:165], 0, v[122:123]
	v_ashrrev_i32_e32 v131, 31, v130
	v_add_u32_e32 v148, s10, v144
	global_store_dword v[128:129], v124, off offset:64
	v_lshl_add_u64 v[128:129], v[130:131], 0, v[146:147]
	v_add_u32_e32 v150, s10, v148
	v_add_u32_e32 v154, v144, v146
	v_add_u32_e32 v156, v148, v146
	v_lshlrev_b64 v[128:129], 2, v[128:129]
	v_add_u32_e32 v152, s10, v150
	v_ashrrev_i32_e32 v155, 31, v154
	v_ashrrev_i32_e32 v157, 31, v156
	v_lshl_add_u64 v[130:131], v[164:165], 0, v[128:129]
	v_add_u32_e32 v158, v150, v146
	v_add_u32_e32 v160, v152, v146
	global_store_dword v[130:131], v125, off offset:64
	v_lshlrev_b64 v[124:125], 2, v[154:155]
	v_lshlrev_b64 v[136:137], 2, v[156:157]
	v_ashrrev_i32_e32 v159, 31, v158
	v_ashrrev_i32_e32 v161, 31, v160
	v_lshl_add_u64 v[130:131], v[164:165], 0, v[124:125]
	v_lshl_add_u64 v[154:155], v[164:165], 0, v[136:137]
	global_store_dword v[130:131], v118, off
	global_store_dword v[154:155], v119, off
	v_lshlrev_b64 v[118:119], 2, v[158:159]
	v_lshlrev_b64 v[158:159], 2, v[160:161]
	v_lshl_add_u64 v[156:157], v[164:165], 0, v[118:119]
	v_lshl_add_u64 v[160:161], v[164:165], 0, v[158:159]
	v_ashrrev_i32_e32 v145, 31, v144
	global_store_dword v[156:157], v120, off
	global_store_dword v[160:161], v121, off
	v_lshl_add_u64 v[120:121], v[144:145], 0, v[146:147]
	v_lshlrev_b64 v[120:121], 2, v[120:121]
	v_lshl_add_u64 v[144:145], v[164:165], 0, v[120:121]
	v_ashrrev_i32_e32 v149, 31, v148
	global_store_dword v[144:145], v114, off offset:64
	v_lshl_add_u64 v[144:145], v[148:149], 0, v[146:147]
	v_lshlrev_b64 v[144:145], 2, v[144:145]
	v_lshl_add_u64 v[148:149], v[164:165], 0, v[144:145]
	v_ashrrev_i32_e32 v151, 31, v150
	global_store_dword v[148:149], v115, off offset:64
	v_lshl_add_u64 v[114:115], v[150:151], 0, v[146:147]
	v_ashrrev_i32_e32 v153, 31, v152
	v_lshlrev_b64 v[114:115], 2, v[114:115]
	v_lshl_add_u64 v[146:147], v[152:153], 0, v[146:147]
	v_lshl_add_u64 v[148:149], v[164:165], 0, v[114:115]
	v_lshlrev_b64 v[146:147], 2, v[146:147]
	global_store_dword v[148:149], v116, off offset:64
	v_lshl_add_u64 v[148:149], v[164:165], 0, v[146:147]
	s_mov_b64 s[0:1], 0x200
	global_store_dword v[148:149], v117, off offset:64
	v_lshl_add_u64 v[116:117], v[164:165], 0, s[0:1]
	global_store_dword v[166:167], v110, off offset:512
	global_store_dword v[168:169], v111, off offset:512
	global_store_dword v[140:141], v112, off offset:512
	global_store_dword v[170:171], v113, off offset:512
	v_lshl_add_u64 v[110:111], v[116:117], 0, v[0:1]
	global_store_dword v[110:111], v106, off offset:64
	v_lshl_add_u64 v[110:111], v[116:117], 0, v[126:127]
	global_store_dword v[110:111], v107, off offset:64
	v_lshl_add_u64 v[106:107], v[116:117], 0, v[122:123]
	global_store_dword v[106:107], v108, off offset:64
	v_lshl_add_u64 v[106:107], v[116:117], 0, v[128:129]
	global_store_dword v[106:107], v109, off offset:64
	global_store_dword v[130:131], v102, off offset:512
	global_store_dword v[154:155], v103, off offset:512
	global_store_dword v[156:157], v104, off offset:512
	global_store_dword v[160:161], v105, off offset:512
	v_lshl_add_u64 v[102:103], v[116:117], 0, v[120:121]
	global_store_dword v[102:103], v98, off offset:64
	v_lshl_add_u64 v[102:103], v[116:117], 0, v[144:145]
	global_store_dword v[102:103], v99, off offset:64
	v_lshl_add_u64 v[98:99], v[116:117], 0, v[114:115]
	global_store_dword v[98:99], v100, off offset:64
	v_lshl_add_u64 v[98:99], v[116:117], 0, v[146:147]
	global_store_dword v[98:99], v101, off offset:64
	v_or_b32_e32 v98, 64, v172
	v_mul_lo_u32 v100, s11, v98
	v_mad_u64_u32 v[98:99], s[4:5], s10, v98, 0
	v_add3_u32 v99, v99, s2, v100
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[162:163]
	v_lshl_add_u64 v[100:101], v[98:99], 0, v[132:133]
	v_lshl_add_u64 v[102:103], v[98:99], 0, v[138:139]
	global_store_dword v[100:101], v94, off
	global_store_dword v[102:103], v95, off
	v_lshl_add_u64 v[94:95], v[98:99], 0, v[134:135]
	v_lshl_add_u64 v[104:105], v[98:99], 0, v[142:143]
	global_store_dword v[94:95], v96, off
	global_store_dword v[104:105], v97, off
	v_lshl_add_u64 v[96:97], v[98:99], 0, v[0:1]
	global_store_dword v[96:97], v90, off offset:64
	v_lshl_add_u64 v[96:97], v[98:99], 0, v[126:127]
	global_store_dword v[96:97], v91, off offset:64
	v_lshl_add_u64 v[90:91], v[98:99], 0, v[122:123]
	global_store_dword v[90:91], v92, off offset:64
	v_lshl_add_u64 v[90:91], v[98:99], 0, v[128:129]
	global_store_dword v[90:91], v93, off offset:64
	v_lshl_add_u64 v[90:91], v[98:99], 0, v[124:125]
	v_lshl_add_u64 v[92:93], v[98:99], 0, v[136:137]
	global_store_dword v[90:91], v86, off
	global_store_dword v[92:93], v87, off
	v_lshl_add_u64 v[86:87], v[98:99], 0, v[118:119]
	v_lshl_add_u64 v[96:97], v[98:99], 0, v[158:159]
	global_store_dword v[86:87], v88, off
	global_store_dword v[96:97], v89, off
	v_lshl_add_u64 v[88:89], v[98:99], 0, v[120:121]
	global_store_dword v[88:89], v82, off offset:64
	v_lshl_add_u64 v[88:89], v[98:99], 0, v[144:145]
	global_store_dword v[88:89], v83, off offset:64
	v_lshl_add_u64 v[82:83], v[98:99], 0, v[114:115]
	global_store_dword v[82:83], v84, off offset:64
	v_lshl_add_u64 v[82:83], v[98:99], 0, v[146:147]
	global_store_dword v[82:83], v85, off offset:64
	v_lshl_add_u64 v[82:83], v[98:99], 0, s[0:1]
	global_store_dword v[100:101], v78, off offset:512
	global_store_dword v[102:103], v79, off offset:512
	global_store_dword v[94:95], v80, off offset:512
	global_store_dword v[104:105], v81, off offset:512
	v_lshl_add_u64 v[78:79], v[82:83], 0, v[0:1]
	global_store_dword v[78:79], v74, off offset:64
	v_lshl_add_u64 v[78:79], v[82:83], 0, v[126:127]
	global_store_dword v[78:79], v75, off offset:64
	v_lshl_add_u64 v[74:75], v[82:83], 0, v[122:123]
	global_store_dword v[74:75], v76, off offset:64
	v_lshl_add_u64 v[74:75], v[82:83], 0, v[128:129]
	global_store_dword v[74:75], v77, off offset:64
	global_store_dword v[90:91], v70, off offset:512
	global_store_dword v[92:93], v71, off offset:512
	global_store_dword v[86:87], v72, off offset:512
	global_store_dword v[96:97], v73, off offset:512
	v_lshl_add_u64 v[70:71], v[82:83], 0, v[120:121]
	global_store_dword v[70:71], v66, off offset:64
	v_lshl_add_u64 v[70:71], v[82:83], 0, v[144:145]
	global_store_dword v[70:71], v67, off offset:64
	v_lshl_add_u64 v[66:67], v[82:83], 0, v[114:115]
	global_store_dword v[66:67], v68, off offset:64
	v_lshl_add_u64 v[66:67], v[82:83], 0, v[146:147]
	global_store_dword v[66:67], v69, off offset:64
	v_or_b32_e32 v66, 0x80, v172
	v_mul_lo_u32 v68, s11, v66
	v_mad_u64_u32 v[66:67], s[4:5], s10, v66, 0
	v_add3_u32 v67, v67, s2, v68
	v_lshl_add_u64 v[66:67], v[66:67], 2, s[8:9]
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[162:163]
	v_lshl_add_u64 v[68:69], v[66:67], 0, v[132:133]
	v_lshl_add_u64 v[70:71], v[66:67], 0, v[138:139]
	global_store_dword v[68:69], v62, off
	global_store_dword v[70:71], v63, off
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[134:135]
	v_lshl_add_u64 v[72:73], v[66:67], 0, v[142:143]
	global_store_dword v[62:63], v64, off
	global_store_dword v[72:73], v65, off
	v_lshl_add_u64 v[64:65], v[66:67], 0, v[0:1]
	global_store_dword v[64:65], v58, off offset:64
	v_lshl_add_u64 v[64:65], v[66:67], 0, v[126:127]
	global_store_dword v[64:65], v59, off offset:64
	v_lshl_add_u64 v[58:59], v[66:67], 0, v[122:123]
	global_store_dword v[58:59], v60, off offset:64
	v_lshl_add_u64 v[58:59], v[66:67], 0, v[128:129]
	global_store_dword v[58:59], v61, off offset:64
	v_lshl_add_u64 v[58:59], v[66:67], 0, v[124:125]
	v_lshl_add_u64 v[60:61], v[66:67], 0, v[136:137]
	global_store_dword v[58:59], v54, off
	global_store_dword v[60:61], v55, off
	v_lshl_add_u64 v[54:55], v[66:67], 0, v[118:119]
	v_lshl_add_u64 v[64:65], v[66:67], 0, v[158:159]
	global_store_dword v[54:55], v56, off
	global_store_dword v[64:65], v57, off
	v_lshl_add_u64 v[56:57], v[66:67], 0, v[120:121]
	global_store_dword v[56:57], v50, off offset:64
	v_lshl_add_u64 v[56:57], v[66:67], 0, v[144:145]
	global_store_dword v[56:57], v51, off offset:64
	v_lshl_add_u64 v[50:51], v[66:67], 0, v[114:115]
	global_store_dword v[50:51], v52, off offset:64
	v_lshl_add_u64 v[50:51], v[66:67], 0, v[146:147]
	global_store_dword v[50:51], v53, off offset:64
	v_lshl_add_u64 v[50:51], v[66:67], 0, s[0:1]
	global_store_dword v[68:69], v46, off offset:512
	global_store_dword v[70:71], v47, off offset:512
	global_store_dword v[62:63], v48, off offset:512
	global_store_dword v[72:73], v49, off offset:512
	v_lshl_add_u64 v[46:47], v[50:51], 0, v[0:1]
	global_store_dword v[46:47], v42, off offset:64
	v_lshl_add_u64 v[46:47], v[50:51], 0, v[126:127]
	global_store_dword v[46:47], v43, off offset:64
	v_lshl_add_u64 v[42:43], v[50:51], 0, v[122:123]
	global_store_dword v[42:43], v44, off offset:64
	v_lshl_add_u64 v[42:43], v[50:51], 0, v[128:129]
	global_store_dword v[42:43], v45, off offset:64
	global_store_dword v[58:59], v38, off offset:512
	global_store_dword v[60:61], v39, off offset:512
	global_store_dword v[54:55], v40, off offset:512
	global_store_dword v[64:65], v41, off offset:512
	v_lshl_add_u64 v[38:39], v[50:51], 0, v[120:121]
	global_store_dword v[38:39], v34, off offset:64
	v_lshl_add_u64 v[38:39], v[50:51], 0, v[144:145]
	global_store_dword v[38:39], v35, off offset:64
	v_lshl_add_u64 v[34:35], v[50:51], 0, v[114:115]
	global_store_dword v[34:35], v36, off offset:64
	v_lshl_add_u64 v[34:35], v[50:51], 0, v[146:147]
	global_store_dword v[34:35], v37, off offset:64
	v_or_b32_e32 v34, 0xc0, v172
	v_mul_lo_u32 v36, s11, v34
	v_mad_u64_u32 v[34:35], s[4:5], s10, v34, 0
	v_add3_u32 v35, v35, s2, v36
	v_lshl_add_u64 v[34:35], v[34:35], 2, s[8:9]
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[162:163]
	v_lshl_add_u64 v[36:37], v[34:35], 0, v[132:133]
	v_lshl_add_u64 v[38:39], v[34:35], 0, v[138:139]
	global_store_dword v[36:37], v30, off
	global_store_dword v[38:39], v31, off
	v_lshl_add_u64 v[30:31], v[34:35], 0, v[134:135]
	v_lshl_add_u64 v[40:41], v[34:35], 0, v[142:143]
	global_store_dword v[30:31], v32, off
	global_store_dword v[40:41], v33, off
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[0:1]
	global_store_dword v[32:33], v26, off offset:64
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[126:127]
	global_store_dword v[32:33], v27, off offset:64
	v_lshl_add_u64 v[26:27], v[34:35], 0, v[122:123]
	global_store_dword v[26:27], v28, off offset:64
	v_lshl_add_u64 v[26:27], v[34:35], 0, v[128:129]
	global_store_dword v[26:27], v29, off offset:64
	v_lshl_add_u64 v[26:27], v[34:35], 0, v[124:125]
	v_lshl_add_u64 v[28:29], v[34:35], 0, v[136:137]
	global_store_dword v[26:27], v22, off
	global_store_dword v[28:29], v23, off
	v_lshl_add_u64 v[22:23], v[34:35], 0, v[118:119]
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[158:159]
	global_store_dword v[22:23], v24, off
	global_store_dword v[32:33], v25, off
	v_lshl_add_u64 v[24:25], v[34:35], 0, v[120:121]
	global_store_dword v[24:25], v18, off offset:64
	v_lshl_add_u64 v[24:25], v[34:35], 0, v[144:145]
	global_store_dword v[24:25], v19, off offset:64
	v_lshl_add_u64 v[18:19], v[34:35], 0, v[114:115]
	global_store_dword v[18:19], v20, off offset:64
	v_lshl_add_u64 v[18:19], v[34:35], 0, v[146:147]
	global_store_dword v[18:19], v21, off offset:64
	v_lshl_add_u64 v[18:19], v[34:35], 0, s[0:1]
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[0:1]
	global_store_dword v[36:37], v14, off offset:512
	global_store_dword v[38:39], v15, off offset:512
	global_store_dword v[30:31], v16, off offset:512
	global_store_dword v[40:41], v17, off offset:512
	global_store_dword v[0:1], v10, off offset:64
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[126:127]
	global_store_dword v[0:1], v11, off offset:64
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[122:123]
	global_store_dword v[0:1], v12, off offset:64
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[128:129]
	global_store_dword v[0:1], v13, off offset:64
	global_store_dword v[26:27], v6, off offset:512
	global_store_dword v[28:29], v7, off offset:512
	global_store_dword v[22:23], v8, off offset:512
	global_store_dword v[32:33], v9, off offset:512
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[120:121]
	global_store_dword v[0:1], v2, off offset:64
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[144:145]
	global_store_dword v[0:1], v3, off offset:64
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[114:115]
	global_store_dword v[0:1], v4, off offset:64
	v_lshl_add_u64 v[0:1], v[18:19], 0, v[146:147]
	global_store_dword v[0:1], v5, off offset:64
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tk13micro_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 400
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
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 233
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 236
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
	.size	_Z8micro_tk13micro_globals, .Lfunc_end0-_Z8micro_tk13micro_globals
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5708
; NumSgprs: 34
; NumVgprs: 233
; NumAgprs: 0
; TotalNumVgprs: 233
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 233
; AccumOffset: 236
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 58
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_445f97fa9acb2968,@object ; @__hip_cuid_445f97fa9acb2968
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_445f97fa9acb2968
__hip_cuid_445f97fa9acb2968:
	.byte	0                               ; 0x0
	.size	__hip_cuid_445f97fa9acb2968, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_445f97fa9acb2968
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           144
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         148
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         152
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         156
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         158
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         160
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         162
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         164
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         166
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         184
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         192
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         200
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         208
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         264
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 400
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z8micro_tk13micro_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     233
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
