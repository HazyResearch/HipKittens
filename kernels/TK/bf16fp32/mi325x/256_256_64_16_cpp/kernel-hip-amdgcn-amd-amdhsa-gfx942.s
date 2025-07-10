	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:
	s_load_dwordx2 s[18:19], s[0:1], 0x0
	s_load_dwordx2 s[14:15], s[0:1], 0x20
	s_load_dwordx2 s[20:21], s[0:1], 0x30
	s_load_dwordx2 s[12:13], s[0:1], 0x50
	s_load_dwordx8 s[4:11], s[0:1], 0x60
	s_load_dwordx2 s[16:17], s[0:1], 0x80
	s_load_dwordx2 s[26:27], s[0:1], 0x90
	s_cmp_lg_u32 0, -1
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s7, 0, 0
	s_and_b32 s9, s7, -16
	s_mov_b32 s23, 0
	s_and_b32 s22, s7, 15
	s_add_i32 s9, s9, 16
	s_cmp_eq_u64 s[22:23], 0
	s_mul_i32 s1, s26, s3
	s_cselect_b32 s24, s7, s9
	s_add_i32 s1, s1, s2
	s_ashr_i32 s3, s1, 31
	s_lshr_b32 s3, s3, 29
	s_add_i32 s3, s1, s3
	s_mul_i32 s2, s27, s26
	s_ashr_i32 s7, s3, 3
	s_and_b32 s3, s3, -8
	s_sub_i32 s1, s1, s3
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 29
	s_add_i32 s2, s2, s3
	s_ashr_i32 s2, s2, 3
	s_mul_i32 s1, s1, s2
	s_add_i32 s2, s1, s7
	s_ashr_i32 s1, s2, 31
	s_lshr_b32 s1, s1, 23
	s_add_i32 s3, s2, s1
	s_ashr_i32 s1, s3, 9
	s_lshl_b32 s7, s1, 4
	s_sub_i32 s1, 32, s7
	s_min_i32 s9, s1, 16
	s_abs_i32 s11, s9
	v_cvt_f32_u32_e32 v1, s11
	s_sub_i32 s26, 0, s11
	s_and_b32 s3, s3, 0xfffffe00
	s_sub_i32 s2, s2, s3
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s25, s2
	s_add_i32 s0, s24, 0x8000
	s_and_b32 s1, s0, -16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s3, s2, s9
	s_and_b32 s22, s0, 15
	s_add_i32 s1, s1, 16
	v_readfirstlane_b32 s27, v1
	s_mul_i32 s26, s26, s27
	s_mul_hi_u32 s26, s27, s26
	s_add_i32 s27, s27, s26
	s_mul_hi_u32 s26, s25, s27
	s_mul_i32 s27, s26, s11
	s_sub_i32 s25, s25, s27
	s_ashr_i32 s3, s3, 31
	s_add_i32 s27, s26, 1
	s_sub_i32 s28, s25, s11
	s_cmp_ge_u32 s25, s11
	s_cselect_b32 s26, s27, s26
	s_cselect_b32 s25, s28, s25
	s_add_i32 s27, s26, 1
	s_cmp_ge_u32 s25, s11
	s_cselect_b32 s11, s27, s26
	s_xor_b32 s11, s11, s3
	s_sub_i32 s11, s11, s3
	s_mul_i32 s3, s11, s9
	s_sub_i32 s2, s2, s3
	s_add_i32 s7, s7, s2
	s_lshl_b32 s7, s7, 8
	s_ashr_i32 s2, s7, 31
	s_mul_i32 s2, s14, s2
	s_mul_hi_u32 s3, s14, s7
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s15, s7
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s14, s7
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s18, s18, s2
	s_addc_u32 s19, s19, s3
	s_lshl_b32 s11, s11, 8
	s_ashr_i32 s9, s11, 31
	s_mul_i32 s2, s12, s9
	s_mul_hi_u32 s3, s12, s11
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s13, s11
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s12, s11
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_u32 s20, s20, s2
	v_and_b32_e32 v32, 56, v2
	v_lshrrev_b32_e32 v28, 3, v0
	s_addc_u32 s21, s21, s3
	v_mad_u64_u32 v[2:3], s[2:3], v28, s14, v[32:33]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[18:19]
	v_or_b32_e32 v29, 64, v28
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[2:3], v29, s14, v[32:33]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[6:7], v[4:5], 1, s[18:19]
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[6:7], off

	;;#ASMEND
	v_lshl_add_u32 v6, s14, 7, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[8:9], v[6:7], 1, s[18:19]
	v_or_b32_e32 v33, 0xc0, v28
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[8:9], off

	;;#ASMEND
	v_mad_u64_u32 v[8:9], s[2:3], v33, s14, v[32:33]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[18:19], v[8:9], 1, s[18:19]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[18:19], off

	;;#ASMEND
	v_lshlrev_b32_e32 v18, 4, v0
	v_lshlrev_b32_e32 v19, 1, v32
	v_and_b32_e32 v3, 0x1f80, v18
	v_add_u32_e32 v30, s24, v19
	v_add_u32_e32 v5, v30, v3
	v_lshrrev_b32_e32 v7, 4, v5
	v_or_b32_e32 v36, 8, v19
	v_and_b32_e32 v7, 0x78, v7
	v_add_u32_e32 v31, s24, v36
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v5, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v5, v31, v3
	v_lshrrev_b32_e32 v7, 4, v5
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v5, v7, v5
	;;#ASMSTART
	ds_write_b64 v5, v[12:13]

	;;#ASMEND
	v_or_b32_e32 v5, 0x2000, v3
	v_add_u32_e32 v7, v30, v5
	v_lshrrev_b32_e32 v9, 4, v7
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	ds_write_b64 v7, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v7, v31, v5
	v_lshrrev_b32_e32 v9, 4, v7
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	ds_write_b64 v7, v[16:17]

	;;#ASMEND
	v_or_b32_e32 v7, 0x4000, v3
	v_add_u32_e32 v9, v30, v7
	v_lshrrev_b32_e32 v10, 4, v9
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v9, v10, v9
	;;#ASMSTART
	ds_write_b64 v9, v[20:21]

	;;#ASMEND
	v_add_u32_e32 v9, v31, v7
	v_lshrrev_b32_e32 v10, 4, v9
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v9, v10, v9
	;;#ASMSTART
	ds_write_b64 v9, v[22:23]

	;;#ASMEND
	v_or_b32_e32 v9, 0x6000, v3
	v_add_u32_e32 v10, v30, v9
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v10, v11, v10
	;;#ASMSTART
	ds_write_b64 v10, v[24:25]

	;;#ASMEND
	v_add_u32_e32 v10, v31, v9
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v10, v11, v10
	;;#ASMSTART
	ds_write_b64 v10, v[26:27]

	;;#ASMEND
	v_mad_u64_u32 v[10:11], s[2:3], v28, s12, v[32:33]
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[12:13], v[10:11], 1, s[20:21]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[12:13], off

	;;#ASMEND
	v_mad_u64_u32 v[12:13], s[2:3], v29, s12, v[32:33]
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[14:15], v[12:13], 1, s[20:21]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[14:15], off

	;;#ASMEND
	v_lshl_add_u32 v14, s12, 7, v10
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[16:17], v[14:15], 1, s[20:21]
	s_cmp_eq_u64 s[22:23], 0
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[16:17], off

	;;#ASMEND
	v_mad_u64_u32 v[16:17], s[2:3], v33, s12, v[32:33]
	s_cselect_b32 s3, s0, s1
	s_nop 0
	v_add_u32_e32 v11, s3, v19
	v_ashrrev_i32_e32 v17, 31, v16
	v_add_u32_e32 v15, v11, v3
	v_lshl_add_u64 v[32:33], v[16:17], 1, s[20:21]
	v_lshrrev_b32_e32 v17, 4, v15
	v_and_b32_e32 v17, 0x78, v17
	v_add_u32_e32 v13, s3, v36
	v_xor_b32_e32 v15, v17, v15
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[32:33], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v15, v[20:21]

	;;#ASMEND
	v_add_u32_e32 v15, v13, v3
	v_lshrrev_b32_e32 v17, 4, v15
	v_and_b32_e32 v17, 0x78, v17
	v_xor_b32_e32 v15, v17, v15
	;;#ASMSTART
	ds_write_b64 v15, v[22:23]

	;;#ASMEND
	v_add_u32_e32 v15, v11, v5
	v_lshrrev_b32_e32 v17, 4, v15
	v_and_b32_e32 v17, 0x78, v17
	v_xor_b32_e32 v15, v17, v15
	;;#ASMSTART
	ds_write_b64 v15, v[24:25]

	;;#ASMEND
	v_add_u32_e32 v15, v13, v5
	v_lshrrev_b32_e32 v17, 4, v15
	v_and_b32_e32 v17, 0x78, v17
	v_xor_b32_e32 v15, v17, v15
	;;#ASMSTART
	ds_write_b64 v15, v[26:27]

	;;#ASMEND
	v_add_u32_e32 v15, v11, v7
	v_lshrrev_b32_e32 v17, 4, v15
	v_and_b32_e32 v17, 0x78, v17
	v_xor_b32_e32 v15, v17, v15
	;;#ASMSTART
	ds_write_b64 v15, v[28:29]

	;;#ASMEND
	v_add_u32_e32 v15, v13, v7
	v_lshrrev_b32_e32 v17, 4, v15
	v_and_b32_e32 v17, 0x78, v17
	v_xor_b32_e32 v15, v17, v15
	v_add_u32_e32 v11, v11, v9
	;;#ASMSTART
	ds_write_b64 v15, v[30:31]

	;;#ASMEND
	v_lshrrev_b32_e32 v15, 4, v11
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v11, v15, v11
	;;#ASMSTART
	ds_write_b64 v11, v[32:33]

	;;#ASMEND
	v_add_u32_e32 v11, v13, v9
	v_lshrrev_b32_e32 v13, 4, v11
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v11, v13, v11
	;;#ASMSTART
	ds_write_b64 v11, v[34:35]

	;;#ASMEND
	v_lshrrev_b32_e32 v1, 8, v0
	v_cmp_eq_u32_e32 vcc, 1, v1
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v161, 1, v2
	v_and_b32_e32 v147, 15, v0
	v_lshrrev_b32_e32 v2, 2, v0
	v_bfe_u32 v148, v0, 6, 2
	v_and_b32_e32 v146, 12, v2
	v_lshlrev_b32_e32 v2, 7, v147
	v_lshlrev_b32_e32 v162, 1, v4
	v_lshl_or_b32 v4, v148, 13, v2
	v_lshlrev_b32_e32 v163, 1, v6
	v_lshlrev_b32_e32 v6, 1, v146
	v_add_u32_e32 v4, s3, v4
	v_lshlrev_b32_e32 v164, 1, v8
	v_add_u32_e32 v8, v4, v6
	v_lshrrev_b32_e32 v11, 4, v8
	v_lshl_or_b32 v2, v1, 13, v2
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v2, s24, v2
	v_lshlrev_b32_e32 v149, 6, v1
	v_xor_b32_e32 v160, v11, v8
	v_add_u32_e32 v8, v2, v6
	v_or_b32_e32 v1, v149, v147
	v_lshrrev_b32_e32 v11, 4, v8
	v_lshl_add_u32 v1, v1, 7, s24
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v1, 0x4000, v1
	v_xor_b32_e32 v159, v11, v8
	v_add_u32_e32 v8, v1, v6
	v_lshrrev_b32_e32 v11, 4, v8
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v158, v11, v8
	v_or_b32_e32 v8, 32, v6
	v_lshlrev_b32_e32 v165, 1, v10
	v_add_u32_e32 v10, v4, v8
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v157, v11, v10
	v_add_u32_e32 v10, v2, v8
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v8, v1, v8
	v_xor_b32_e32 v156, v11, v10
	v_lshrrev_b32_e32 v10, 4, v8
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v155, v10, v8
	v_or_b32_e32 v8, 64, v6
	v_add_u32_e32 v10, v4, v8
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v154, v11, v10
	v_add_u32_e32 v10, v2, v8
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v8, v1, v8
	v_xor_b32_e32 v153, v11, v10
	v_lshrrev_b32_e32 v10, 4, v8
	v_or_b32_e32 v6, 0x60, v6
	v_and_b32_e32 v10, 0x78, v10
	v_add_u32_e32 v4, v4, v6
	v_xor_b32_e32 v152, v10, v8
	v_lshrrev_b32_e32 v8, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v2, v2, v6
	v_xor_b32_e32 v151, v8, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_add_u32_e32 v1, v1, v6
	v_xor_b32_e32 v150, v4, v2
	v_lshrrev_b32_e32 v2, 4, v1
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v1, v2, v1
	v_and_b32_e32 v2, 0x70, v18
	v_add_u32_e32 v4, s24, v2
	v_add_u32_e32 v10, v4, v3
	v_or_b32_e32 v6, 8, v2
	v_lshrrev_b32_e32 v11, 4, v10
	v_add_u32_e32 v8, s24, v6
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v169, v11, v10
	v_add_u32_e32 v10, v8, v3
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v170, v11, v10
	v_add_u32_e32 v10, v4, v5
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v171, v11, v10
	v_add_u32_e32 v10, v8, v5
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v172, v11, v10
	v_add_u32_e32 v10, v4, v7
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v173, v11, v10
	v_add_u32_e32 v10, v8, v7
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v4, v4, v9
	v_xor_b32_e32 v174, v11, v10
	v_lshrrev_b32_e32 v10, 4, v4
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v175, v10, v4
	v_add_u32_e32 v4, v8, v9
	v_lshrrev_b32_e32 v8, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v2, s3, v2
	v_xor_b32_e32 v176, v8, v4
	v_add_u32_e32 v4, s3, v6
	v_add_u32_e32 v6, v2, v3
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v3, v4, v3
	v_xor_b32_e32 v177, v8, v6
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v178, v6, v3
	v_add_u32_e32 v3, v2, v5
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v179, v6, v3
	v_add_u32_e32 v3, v4, v5
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v180, v5, v3
	v_add_u32_e32 v3, v2, v7
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v181, v5, v3
	v_add_u32_e32 v3, v4, v7
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v2, v2, v9
	v_xor_b32_e32 v182, v5, v3
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v183, v3, v2
	v_add_u32_e32 v2, v4, v9
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v184, v3, v2
	v_mov_b32_e32 v2, 0
	s_lshl_b32 s2, s14, 9
	s_lshl_b32 s14, s12, 9
	v_lshlrev_b32_e32 v166, 1, v12
	v_lshlrev_b32_e32 v167, 1, v14
	v_lshlrev_b32_e32 v168, 1, v16
	s_mov_b64 s[22:23], 0x80
	s_mov_b32 s3, 0x20000
	s_mov_b32 s24, 0xffff
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
	v_mov_b32_e32 v126, v2
	v_mov_b32_e32 v127, v2
	v_mov_b32_e32 v128, v2
	v_mov_b32_e32 v129, v2
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_add_u32 s0, s18, s22
	s_addc_u32 s1, s19, s23
	buffer_load_dwordx4 v[130:133], v161, s[0:3], 0 offen
	buffer_load_dwordx4 v[134:137], v162, s[0:3], 0 offen
	buffer_load_dwordx4 v[138:141], v163, s[0:3], 0 offen
	buffer_load_dwordx4 v[142:145], v164, s[0:3], 0 offen
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[186:187], v160 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[190:191], v160 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v160 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[198:199], v160 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v159 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v159 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[196:197], v159 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[200:201], v159 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v158 offset:0
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v158 offset:0x800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v158 offset:0x1000
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v158 offset:0x1800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v186, s24, v186, v186
	v_bfi_b32 v190, s24, v190, v190
	v_bfi_b32 v194, s24, v194, v194
	v_bfi_b32 v198, s24, v198, v198
	v_bfi_b32 v188, s24, v188, v188
	v_bfi_b32 v192, s24, v192, v192
	v_bfi_b32 v196, s24, v196, v196
	v_bfi_b32 v200, s24, v200, v200
	v_bfi_b32 v202, s24, v202, v202
	v_bfi_b32 v204, s24, v204, v204
	v_bfi_b32 v206, s24, v206, v206
	v_bfi_b32 v208, s24, v208, v208
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[188:189], v[186:187], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[188:189], v[190:191], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[188:189], v[194:195], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[188:189], v[198:199], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[192:193], v[186:187], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[192:193], v[190:191], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[192:193], v[194:195], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[192:193], v[198:199], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[196:197], v[186:187], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[196:197], v[190:191], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[196:197], v[194:195], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[196:197], v[198:199], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[200:201], v[186:187], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[200:201], v[190:191], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[200:201], v[194:195], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[200:201], v[198:199], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[202:203], v[186:187], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[202:203], v[190:191], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[202:203], v[194:195], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[202:203], v[198:199], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[204:205], v[186:187], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[204:205], v[190:191], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[204:205], v[194:195], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[204:205], v[198:199], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[206:207], v[186:187], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[206:207], v[190:191], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[206:207], v[194:195], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[206:207], v[198:199], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[186:189], v[208:209], v[186:187], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[190:193], v[208:209], v[190:191], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[194:197], v[208:209], v[194:195], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[198:201], v[208:209], v[198:199], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s20, s22
	s_addc_u32 s13, s21, s23
	s_mov_b32 s15, s3
	buffer_load_dwordx4 v[2:5], v165, s[12:15], 0 offen
	buffer_load_dwordx4 v[6:9], v166, s[12:15], 0 offen
	buffer_load_dwordx4 v[10:13], v167, s[12:15], 0 offen
	buffer_load_dwordx4 v[14:17], v168, s[12:15], 0 offen
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[202:203], v157 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v157 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v157 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v157 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v156 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v156 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[214:215], v156 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[216:217], v156 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[218:219], v155 offset:0
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[220:221], v155 offset:0x800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v155 offset:0x1000
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v155 offset:0x1800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v202, s24, v202, v202
	v_bfi_b32 v204, s24, v204, v204
	v_bfi_b32 v206, s24, v206, v206
	v_bfi_b32 v208, s24, v208, v208
	v_bfi_b32 v210, s24, v210, v210
	v_bfi_b32 v212, s24, v212, v212
	v_bfi_b32 v214, s24, v214, v214
	v_bfi_b32 v216, s24, v216, v216
	v_bfi_b32 v218, s24, v218, v218
	v_bfi_b32 v220, s24, v220, v220
	v_bfi_b32 v222, s24, v222, v222
	v_bfi_b32 v224, s24, v224, v224
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[210:211], v[202:203], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[210:211], v[204:205], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[210:211], v[206:207], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[210:211], v[208:209], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[212:213], v[202:203], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[212:213], v[204:205], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[212:213], v[206:207], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[212:213], v[208:209], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[214:215], v[202:203], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[214:215], v[204:205], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[214:215], v[206:207], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[214:215], v[208:209], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[216:217], v[202:203], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[216:217], v[204:205], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[216:217], v[206:207], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[216:217], v[208:209], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[218:219], v[202:203], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[218:219], v[204:205], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[218:219], v[206:207], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[218:219], v[208:209], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[220:221], v[202:203], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[220:221], v[204:205], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[220:221], v[206:207], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[220:221], v[208:209], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[222:223], v[202:203], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[222:223], v[204:205], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[222:223], v[206:207], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[222:223], v[208:209], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[186:189], v[224:225], v[202:203], v[186:189]
	v_mfma_f32_16x16x16_bf16 v[190:193], v[224:225], v[204:205], v[190:193]
	v_mfma_f32_16x16x16_bf16 v[194:197], v[224:225], v[206:207], v[194:197]
	v_mfma_f32_16x16x16_bf16 v[198:201], v[224:225], v[208:209], v[198:201]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[202:203], v154 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v154 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v154 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v154 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v153 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v153 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[214:215], v153 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[216:217], v153 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[218:219], v152 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[220:221], v152 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v152 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v152 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[226:227], v151 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[228:229], v151 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[230:231], v151 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[232:233], v151 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[234:235], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[236:237], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[238:239], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[240:241], v150 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[242:243], v1 offset:0
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[244:245], v1 offset:0x800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[246:247], v1 offset:0x1000
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[248:249], v1 offset:0x1800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v202, s24, v202, v202
	v_bfi_b32 v204, s24, v204, v204
	v_bfi_b32 v206, s24, v206, v206
	v_bfi_b32 v208, s24, v208, v208
	v_bfi_b32 v210, s24, v210, v210
	v_bfi_b32 v212, s24, v212, v212
	v_bfi_b32 v214, s24, v214, v214
	v_bfi_b32 v216, s24, v216, v216
	v_bfi_b32 v218, s24, v218, v218
	v_bfi_b32 v220, s24, v220, v220
	v_bfi_b32 v222, s24, v222, v222
	v_bfi_b32 v224, s24, v224, v224
	v_bfi_b32 v226, s24, v226, v226
	v_bfi_b32 v228, s24, v228, v228
	v_bfi_b32 v230, s24, v230, v230
	v_bfi_b32 v232, s24, v232, v232
	v_bfi_b32 v234, s24, v234, v234
	v_bfi_b32 v236, s24, v236, v236
	v_bfi_b32 v238, s24, v238, v238
	v_bfi_b32 v240, s24, v240, v240
	v_bfi_b32 v242, s24, v242, v242
	v_bfi_b32 v244, s24, v244, v244
	v_bfi_b32 v246, s24, v246, v246
	v_bfi_b32 v248, s24, v248, v248
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[210:211], v[202:203], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[210:211], v[204:205], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[210:211], v[206:207], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[210:211], v[208:209], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[212:213], v[202:203], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[212:213], v[204:205], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[212:213], v[206:207], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[212:213], v[208:209], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[214:215], v[202:203], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[214:215], v[204:205], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[214:215], v[206:207], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[214:215], v[208:209], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[216:217], v[202:203], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[216:217], v[204:205], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[216:217], v[206:207], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[216:217], v[208:209], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[218:219], v[202:203], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[218:219], v[204:205], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[218:219], v[206:207], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[218:219], v[208:209], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[220:221], v[202:203], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[220:221], v[204:205], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[220:221], v[206:207], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[220:221], v[208:209], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[222:223], v[202:203], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[222:223], v[204:205], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[222:223], v[206:207], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[222:223], v[208:209], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[186:189], v[224:225], v[202:203], v[186:189]
	v_mfma_f32_16x16x16_bf16 v[190:193], v[224:225], v[204:205], v[190:193]
	v_mfma_f32_16x16x16_bf16 v[194:197], v[224:225], v[206:207], v[194:197]
	v_mfma_f32_16x16x16_bf16 v[198:201], v[224:225], v[208:209], v[198:201]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(7)
	;;#ASMSTART
	ds_write_b64 v169, v[130:131]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v170, v[132:133]

	;;#ASMEND
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	ds_write_b64 v171, v[134:135]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v172, v[136:137]

	;;#ASMEND
	s_waitcnt vmcnt(5)
	;;#ASMSTART
	ds_write_b64 v173, v[138:139]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v174, v[140:141]

	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	ds_write_b64 v175, v[142:143]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v176, v[144:145]

	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	ds_write_b64 v177, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v178, v[4:5]

	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_write_b64 v179, v[6:7]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v180, v[8:9]

	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v181, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v182, v[12:13]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v183, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v184, v[16:17]

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[234:235], v[226:227], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[234:235], v[228:229], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[234:235], v[230:231], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[234:235], v[232:233], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[236:237], v[226:227], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[236:237], v[228:229], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[236:237], v[230:231], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[236:237], v[232:233], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[238:239], v[226:227], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[238:239], v[228:229], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[238:239], v[230:231], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[238:239], v[232:233], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[240:241], v[226:227], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[240:241], v[228:229], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[240:241], v[230:231], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[240:241], v[232:233], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[242:243], v[226:227], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[242:243], v[228:229], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[242:243], v[230:231], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[242:243], v[232:233], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[244:245], v[226:227], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[244:245], v[228:229], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[244:245], v[230:231], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[244:245], v[232:233], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[246:247], v[226:227], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[246:247], v[228:229], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[246:247], v[230:231], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[246:247], v[232:233], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[248:249], v[226:227], v[186:189]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[248:249], v[228:229], v[190:193]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[248:249], v[230:231], v[194:197]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[248:249], v[232:233], v[198:201]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s22, s22, 0x80
	s_addc_u32 s23, s23, 0
	s_cmpk_eq_i32 s22, 0x4000
	s_cbranch_scc0 .LBB0_3
; %bb.4:
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[130:131], v160 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v160 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v160 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v160 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v159 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v159 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v159 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v159 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v158 offset:0
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v158 offset:0x800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v158 offset:0x1000
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_mov_b32 s0, 0xffff
	;;#ASMSTART
	ds_read_b64 v[158:159], v158 offset:0x1800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v130, s0, v130, v130
	v_bfi_b32 v132, s0, v132, v132
	v_bfi_b32 v134, s0, v134, v134
	v_bfi_b32 v136, s0, v136, v136
	v_bfi_b32 v138, s0, v138, v138
	v_bfi_b32 v140, s0, v140, v140
	v_bfi_b32 v142, s0, v142, v142
	v_bfi_b32 v144, s0, v144, v144
	v_bfi_b32 v160, s0, v160, v160
	v_bfi_b32 v162, s0, v162, v162
	v_bfi_b32 v164, s0, v164, v164
	v_bfi_b32 v158, s0, v158, v158
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[138:139], v[130:131], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[138:139], v[132:133], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[138:139], v[134:135], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[138:139], v[136:137], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[140:141], v[130:131], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[140:141], v[132:133], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[140:141], v[134:135], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[140:141], v[136:137], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[142:143], v[130:131], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[142:143], v[132:133], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[142:143], v[134:135], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[142:143], v[136:137], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[144:145], v[130:131], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[144:145], v[132:133], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[144:145], v[134:135], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[144:145], v[136:137], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[160:161], v[130:131], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[160:161], v[132:133], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[160:161], v[134:135], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[160:161], v[136:137], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[162:163], v[130:131], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[162:163], v[132:133], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[162:163], v[134:135], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[162:163], v[136:137], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[164:165], v[130:131], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[164:165], v[132:133], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[164:165], v[134:135], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[164:165], v[136:137], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[158:159], v[130:131], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[158:159], v[132:133], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[158:159], v[134:135], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[158:159], v[136:137], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[130:131], v157 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v157 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v157 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v157 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v156 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v156 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v156 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v156 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[156:157], v155 offset:0
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v155 offset:0x800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v155 offset:0x1000
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v155 offset:0x1800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v130, s0, v130, v130
	v_bfi_b32 v132, s0, v132, v132
	v_bfi_b32 v134, s0, v134, v134
	v_bfi_b32 v136, s0, v136, v136
	v_bfi_b32 v138, s0, v138, v138
	v_bfi_b32 v140, s0, v140, v140
	v_bfi_b32 v142, s0, v142, v142
	v_bfi_b32 v144, s0, v144, v144
	v_bfi_b32 v156, s0, v156, v156
	v_bfi_b32 v158, s0, v158, v158
	v_bfi_b32 v160, s0, v160, v160
	v_bfi_b32 v162, s0, v162, v162
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[138:139], v[130:131], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[138:139], v[132:133], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[138:139], v[134:135], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[138:139], v[136:137], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[140:141], v[130:131], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[140:141], v[132:133], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[140:141], v[134:135], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[140:141], v[136:137], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[142:143], v[130:131], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[142:143], v[132:133], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[142:143], v[134:135], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[142:143], v[136:137], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[144:145], v[130:131], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[144:145], v[132:133], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[144:145], v[134:135], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[144:145], v[136:137], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[156:157], v[130:131], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[156:157], v[132:133], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[156:157], v[134:135], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[156:157], v[136:137], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[158:159], v[130:131], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[158:159], v[132:133], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[158:159], v[134:135], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[158:159], v[136:137], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[160:161], v[130:131], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[160:161], v[132:133], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[160:161], v[134:135], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[160:161], v[136:137], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[162:163], v[130:131], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[162:163], v[132:133], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[162:163], v[134:135], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[162:163], v[136:137], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[190:191], v154 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v154 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[198:199], v154 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v154 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v153 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v153 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v153 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v153 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v152 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[170:171], v152 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[186:187], v152 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v152 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v151 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v151 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v151 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v151 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[214:215], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[216:217], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[218:219], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[220:221], v150 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v1 offset:0
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v1 offset:0x800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[226:227], v1 offset:0x1000
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[228:229], v1 offset:0x1800
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v190, s0, v190, v190
	v_bfi_b32 v194, s0, v194, v194
	v_bfi_b32 v198, s0, v198, v198
	v_bfi_b32 v202, s0, v202, v202
	v_bfi_b32 v130, s0, v130, v130
	v_bfi_b32 v132, s0, v132, v132
	v_bfi_b32 v134, s0, v134, v134
	v_bfi_b32 v136, s0, v136, v136
	v_bfi_b32 v154, s0, v154, v154
	v_bfi_b32 v170, s0, v170, v170
	v_bfi_b32 v186, s0, v186, v186
	v_bfi_b32 v204, s0, v204, v204
	v_bfi_b32 v206, s0, v206, v206
	v_bfi_b32 v208, s0, v208, v208
	v_bfi_b32 v210, s0, v210, v210
	v_bfi_b32 v212, s0, v212, v212
	v_bfi_b32 v214, s0, v214, v214
	v_bfi_b32 v216, s0, v216, v216
	v_bfi_b32 v218, s0, v218, v218
	v_bfi_b32 v220, s0, v220, v220
	v_bfi_b32 v222, s0, v222, v222
	v_bfi_b32 v224, s0, v224, v224
	v_bfi_b32 v226, s0, v226, v226
	v_bfi_b32 v228, s0, v228, v228
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[130:131], v[190:191], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[130:131], v[194:195], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[130:131], v[198:199], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[130:131], v[202:203], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[132:133], v[190:191], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[132:133], v[194:195], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[132:133], v[198:199], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[132:133], v[202:203], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[134:135], v[190:191], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[134:135], v[194:195], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[134:135], v[198:199], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[134:135], v[202:203], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[136:137], v[190:191], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[136:137], v[194:195], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[204:205], v[202:203], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[130:133], v[136:137], v[198:199], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[134:137], v[136:137], v[202:203], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[138:141], v[154:155], v[190:191], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[142:145], v[154:155], v[194:195], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[150:153], v[154:155], v[198:199], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[154:157], v[154:155], v[202:203], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[158:161], v[170:171], v[190:191], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[162:165], v[170:171], v[194:195], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[166:169], v[170:171], v[198:199], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[170:173], v[170:171], v[202:203], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[174:177], v[186:187], v[190:191], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[178:181], v[186:187], v[194:195], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[182:185], v[186:187], v[198:199], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[186:189], v[186:187], v[202:203], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[190:193], v[204:205], v[190:191], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[194:197], v[204:205], v[194:195], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[198:201], v[204:205], v[198:199], v[6:9]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[68:71], v[214:215], v[206:207], v[126:129]
	v_mfma_f32_16x16x16_bf16 v[124:127], v[214:215], v[208:209], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[214:215], v[210:211], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[214:215], v[212:213], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[216:217], v[206:207], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[216:217], v[208:209], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[216:217], v[210:211], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[216:217], v[212:213], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[218:219], v[206:207], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[218:219], v[208:209], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[218:219], v[210:211], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[218:219], v[212:213], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[220:221], v[206:207], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[220:221], v[208:209], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[220:221], v[210:211], v[130:133]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[220:221], v[212:213], v[134:137]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[222:223], v[206:207], v[138:141]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[222:223], v[208:209], v[142:145]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[222:223], v[210:211], v[150:153]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[222:223], v[212:213], v[154:157]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[224:225], v[206:207], v[158:161]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[224:225], v[208:209], v[162:165]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[224:225], v[210:211], v[166:169]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[224:225], v[212:213], v[170:173]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[226:227], v[206:207], v[174:177]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[226:227], v[208:209], v[178:181]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[226:227], v[210:211], v[182:185]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[226:227], v[212:213], v[186:189]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[228:229], v[206:207], v[190:193]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[228:229], v[208:209], v[194:197]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[228:229], v[210:211], v[198:201]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[228:229], v[212:213], v[0:3]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v128, 0x7f800000, v68
	v_cmp_ne_u32_e32 vcc, s0, v128
                                        ; implicit-def: $vgpr128
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.7:
	v_bfe_u32 v128, v68, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v128, v68, v128, s2
; %bb.8:                                ; %Flow4585
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.9:
	v_mov_b32_e32 v128, 0
	v_or_b32_e32 v129, 0x10000, v68
	v_cmp_eq_u32_sdwa vcc, v68, v128 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v128, v129, v68, vcc
; %bb.10:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v69
	v_cmp_ne_u32_e32 vcc, s0, v68
                                        ; implicit-def: $vgpr132
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.11:
	v_bfe_u32 v68, v69, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v132, v69, v68, s2
; %bb.12:                               ; %Flow4584
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.13:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v129, 0x10000, v69
	v_cmp_eq_u32_sdwa vcc, v69, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v132, v129, v69, vcc
; %bb.14:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v70
	v_cmp_ne_u32_e32 vcc, s0, v68
                                        ; implicit-def: $vgpr133
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.15:
	v_bfe_u32 v68, v70, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v133, v70, v68, s2
; %bb.16:                               ; %Flow4583
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.17:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v69, 0x10000, v70
	v_cmp_eq_u32_sdwa vcc, v70, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v133, v69, v70, vcc
; %bb.18:                               ; %_ZL16__float2bfloat16f.exit.i.i46.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v71
	v_cmp_ne_u32_e32 vcc, s0, v68
                                        ; implicit-def: $vgpr134
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.19:
	v_bfe_u32 v68, v71, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v134, v71, v68, s2
                                        ; implicit-def: $vgpr68_vgpr69_vgpr70_vgpr71
; %bb.20:                               ; %Flow4582
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.21:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v69, 0x10000, v71
	v_cmp_eq_u32_sdwa vcc, v71, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v134, v69, v71, vcc
; %bb.22:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshl_or_b32 v70, v148, 6, s11
	v_mul_lo_u32 v129, s17, v70
	v_mad_u64_u32 v[70:71], s[0:1], s16, v70, 0
	s_mul_i32 s0, s16, s9
	s_nop 0
	v_add3_u32 v71, v71, s0, v129
	s_mul_i32 s0, s8, s6
	v_lshrrev_b32_e32 v135, 16, v128
	v_mov_b32_e32 v68, s4
	v_mov_b32_e32 v69, s5
	v_or_b32_e32 v128, s7, v149
	s_mul_i32 s0, s0, s10
	v_ashrrev_i32_e32 v129, 31, v128
	v_lshl_add_u64 v[130:131], v[70:71], 1, v[68:69]
	s_mul_i32 s0, s0, s16
	v_lshl_add_u64 v[68:69], v[128:129], 1, v[130:131]
	s_lshl_b32 s8, s0, 1
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v129, 0xffff0000, v134
	v_mul_lo_u32 v134, v147, s16
	v_mov_b32_e32 v70, s8
	v_mov_b32_e32 v71, 0x20000
	v_or_b32_sdwa v133, v129, v133 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v132, v132, s0, v135
	v_add_lshl_u32 v129, v134, v146, 1
	s_mov_b64 s[2:3], exec
.LBB0_23:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[132:133], v129, s[4:7], 0 offen
                                        ; implicit-def: $vgpr132_vgpr133
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_23
; %bb.24:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v132, 0x7f800000, v124
	v_cmp_ne_u32_e32 vcc, s0, v132
                                        ; implicit-def: $vgpr132
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.25:
	v_bfe_u32 v132, v124, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v132, v124, v132, s2
; %bb.26:                               ; %Flow4581
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.27:
	v_mov_b32_e32 v132, 0
	v_or_b32_e32 v133, 0x10000, v124
	v_cmp_eq_u32_sdwa vcc, v124, v132 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v132, v133, v124, vcc
; %bb.28:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v125
	v_cmp_ne_u32_e32 vcc, s0, v124
                                        ; implicit-def: $vgpr133
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.29:
	v_bfe_u32 v124, v125, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v133, v125, v124, s2
; %bb.30:                               ; %Flow4580
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.31:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v133, 0x10000, v125
	v_cmp_eq_u32_sdwa vcc, v125, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v133, v133, v125, vcc
; %bb.32:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v126
	v_cmp_ne_u32_e32 vcc, s0, v124
                                        ; implicit-def: $vgpr135
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.33:
	v_bfe_u32 v124, v126, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v135, v126, v124, s2
; %bb.34:                               ; %Flow4579
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.35:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v125, 0x10000, v126
	v_cmp_eq_u32_sdwa vcc, v126, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v135, v125, v126, vcc
; %bb.36:                               ; %_ZL16__float2bfloat16f.exit.i.i46.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v127
	v_cmp_ne_u32_e32 vcc, s0, v124
                                        ; implicit-def: $vgpr136
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.37:
	v_bfe_u32 v124, v127, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v136, v127, v124, s2
                                        ; implicit-def: $vgpr124_vgpr125_vgpr126_vgpr127
; %bb.38:                               ; %Flow4578
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.39:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v125, 0x10000, v127
	v_cmp_eq_u32_sdwa vcc, v127, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v136, v125, v127, vcc
; %bb.40:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_lshl_b32 s9, s16, 4
	v_lshrrev_b32_e32 v124, 16, v132
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v125, 0xffff0000, v136
	v_add_u32_e32 v127, s9, v134
	v_or_b32_sdwa v125, v125, v135 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v124, v133, s0, v124
	v_add_lshl_u32 v126, v127, v146, 1
	s_mov_b64 s[2:3], exec
.LBB0_41:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[124:125], v126, s[4:7], 0 offen
                                        ; implicit-def: $vgpr124_vgpr125
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_41
; %bb.42:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v120
	v_cmp_ne_u32_e32 vcc, s0, v124
                                        ; implicit-def: $vgpr124
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.43:
	v_bfe_u32 v124, v120, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v124, v120, v124, s2
; %bb.44:                               ; %Flow4577
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.45:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v125, 0x10000, v120
	v_cmp_eq_u32_sdwa vcc, v120, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v124, v125, v120, vcc
; %bb.46:                               ; %_ZL16__float2bfloat16f.exit.i.i.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v121
	v_cmp_ne_u32_e32 vcc, s0, v120
                                        ; implicit-def: $vgpr125
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.47:
	v_bfe_u32 v120, v121, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v125, v121, v120, s2
; %bb.48:                               ; %Flow4576
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.49:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v125, 0x10000, v121
	v_cmp_eq_u32_sdwa vcc, v121, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v125, v125, v121, vcc
; %bb.50:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v122
	v_cmp_ne_u32_e32 vcc, s0, v120
                                        ; implicit-def: $vgpr132
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.51:
	v_bfe_u32 v120, v122, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v132, v122, v120, s2
; %bb.52:                               ; %Flow4575
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.53:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v121, 0x10000, v122
	v_cmp_eq_u32_sdwa vcc, v122, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v132, v121, v122, vcc
; %bb.54:                               ; %_ZL16__float2bfloat16f.exit.i.i46.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v123
	v_cmp_ne_u32_e32 vcc, s0, v120
                                        ; implicit-def: $vgpr133
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.55:
	v_bfe_u32 v120, v123, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v133, v123, v120, s2
                                        ; implicit-def: $vgpr120_vgpr121_vgpr122_vgpr123
; %bb.56:                               ; %Flow4574
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.57:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v121, 0x10000, v123
	v_cmp_eq_u32_sdwa vcc, v123, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v133, v121, v123, vcc
; %bb.58:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v120, 16, v124
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v121, 0xffff0000, v133
	v_add_u32_e32 v123, s9, v127
	v_or_b32_sdwa v121, v121, v132 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v120, v125, s0, v120
	v_add_lshl_u32 v122, v123, v146, 1
	s_mov_b64 s[2:3], exec
.LBB0_59:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[120:121], v122, s[4:7], 0 offen
                                        ; implicit-def: $vgpr120_vgpr121
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_59
; %bb.60:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v116
	v_cmp_ne_u32_e32 vcc, s0, v120
                                        ; implicit-def: $vgpr120
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.61:
	v_bfe_u32 v120, v116, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v116, v120, s2
; %bb.62:                               ; %Flow4573
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.63:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v121, 0x10000, v116
	v_cmp_eq_u32_sdwa vcc, v116, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v121, v116, vcc
; %bb.64:                               ; %_ZL16__float2bfloat16f.exit.i.i.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v117
	v_cmp_ne_u32_e32 vcc, s0, v116
                                        ; implicit-def: $vgpr121
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.65:
	v_bfe_u32 v116, v117, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v121, v117, v116, s2
; %bb.66:                               ; %Flow4572
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.67:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v121, 0x10000, v117
	v_cmp_eq_u32_sdwa vcc, v117, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v121, v121, v117, vcc
; %bb.68:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v118
	v_cmp_ne_u32_e32 vcc, s0, v116
                                        ; implicit-def: $vgpr124
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.69:
	v_bfe_u32 v116, v118, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v124, v118, v116, s2
; %bb.70:                               ; %Flow4571
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.71:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v117, 0x10000, v118
	v_cmp_eq_u32_sdwa vcc, v118, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v124, v117, v118, vcc
; %bb.72:                               ; %_ZL16__float2bfloat16f.exit.i.i46.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v119
	v_cmp_ne_u32_e32 vcc, s0, v116
                                        ; implicit-def: $vgpr125
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.73:
	v_bfe_u32 v116, v119, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v125, v119, v116, s2
                                        ; implicit-def: $vgpr116_vgpr117_vgpr118_vgpr119
; %bb.74:                               ; %Flow4570
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.75:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v117, 0x10000, v119
	v_cmp_eq_u32_sdwa vcc, v119, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v125, v117, v119, vcc
; %bb.76:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v116, 16, v120
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v117, 0xffff0000, v125
	v_add_u32_e32 v119, s9, v123
	v_or_b32_sdwa v117, v117, v124 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v116, v121, s0, v116
	v_add_lshl_u32 v118, v119, v146, 1
	s_mov_b64 s[2:3], exec
.LBB0_77:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[116:117], v118, s[4:7], 0 offen
                                        ; implicit-def: $vgpr116_vgpr117
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_77
; %bb.78:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v112
	v_cmp_ne_u32_e32 vcc, s0, v116
                                        ; implicit-def: $vgpr116
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.79:
	v_bfe_u32 v116, v112, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v112, v116, s2
; %bb.80:                               ; %Flow4569
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.81:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v117, 0x10000, v112
	v_cmp_eq_u32_sdwa vcc, v112, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v117, v112, vcc
; %bb.82:                               ; %_ZL16__float2bfloat16f.exit.i.i.167.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v113
	v_cmp_ne_u32_e32 vcc, s0, v112
                                        ; implicit-def: $vgpr117
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.83:
	v_bfe_u32 v112, v113, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v117, v113, v112, s2
; %bb.84:                               ; %Flow4568
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.85:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v117, 0x10000, v113
	v_cmp_eq_u32_sdwa vcc, v113, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v117, v117, v113, vcc
; %bb.86:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.174.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v114
	v_cmp_ne_u32_e32 vcc, s0, v112
                                        ; implicit-def: $vgpr120
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.87:
	v_bfe_u32 v112, v114, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v114, v112, s2
; %bb.88:                               ; %Flow4567
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.89:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v113, 0x10000, v114
	v_cmp_eq_u32_sdwa vcc, v114, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v113, v114, vcc
; %bb.90:                               ; %_ZL16__float2bfloat16f.exit.i.i46.178.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v115
	v_cmp_ne_u32_e32 vcc, s0, v112
                                        ; implicit-def: $vgpr121
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.91:
	v_bfe_u32 v112, v115, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v121, v115, v112, s2
                                        ; implicit-def: $vgpr112_vgpr113_vgpr114_vgpr115
; %bb.92:                               ; %Flow4566
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.93:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v113, 0x10000, v115
	v_cmp_eq_u32_sdwa vcc, v115, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v121, v113, v115, vcc
; %bb.94:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.189.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v112, 16, v116
	v_or_b32_e32 v115, 16, v146
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v113, 0xffff0000, v121
	v_or_b32_sdwa v113, v113, v120 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v112, v117, s0, v112
	v_add_lshl_u32 v114, v134, v115, 1
	s_mov_b64 s[2:3], exec
.LBB0_95:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[112:113], v114, s[4:7], 0 offen
                                        ; implicit-def: $vgpr112_vgpr113
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_95
; %bb.96:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v108
	v_cmp_ne_u32_e32 vcc, s0, v112
                                        ; implicit-def: $vgpr112
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.97:
	v_bfe_u32 v112, v108, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v112, v108, v112, s2
; %bb.98:                               ; %Flow4565
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.99:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v113, 0x10000, v108
	v_cmp_eq_u32_sdwa vcc, v108, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v112, v113, v108, vcc
; %bb.100:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v109
	v_cmp_ne_u32_e32 vcc, s0, v108
                                        ; implicit-def: $vgpr113
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.101:
	v_bfe_u32 v108, v109, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v113, v109, v108, s2
; %bb.102:                              ; %Flow4564
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.103:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v113, 0x10000, v109
	v_cmp_eq_u32_sdwa vcc, v109, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v113, v113, v109, vcc
; %bb.104:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v110
	v_cmp_ne_u32_e32 vcc, s0, v108
                                        ; implicit-def: $vgpr116
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.105:
	v_bfe_u32 v108, v110, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v110, v108, s2
; %bb.106:                              ; %Flow4563
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.107:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v109, 0x10000, v110
	v_cmp_eq_u32_sdwa vcc, v110, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v109, v110, vcc
; %bb.108:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v111
	v_cmp_ne_u32_e32 vcc, s0, v108
                                        ; implicit-def: $vgpr117
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.109:
	v_bfe_u32 v108, v111, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v117, v111, v108, s2
                                        ; implicit-def: $vgpr108_vgpr109_vgpr110_vgpr111
; %bb.110:                              ; %Flow4562
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.111:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v109, 0x10000, v111
	v_cmp_eq_u32_sdwa vcc, v111, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v117, v109, v111, vcc
; %bb.112:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v108, 16, v112
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v109, 0xffff0000, v117
	v_or_b32_sdwa v109, v109, v116 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v108, v113, s0, v108
	v_add_lshl_u32 v110, v127, v115, 1
	s_mov_b64 s[2:3], exec
.LBB0_113:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[108:109], v110, s[4:7], 0 offen
                                        ; implicit-def: $vgpr108_vgpr109
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_113
; %bb.114:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v104
	v_cmp_ne_u32_e32 vcc, s0, v108
                                        ; implicit-def: $vgpr108
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.115:
	v_bfe_u32 v108, v104, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v108, v104, v108, s2
; %bb.116:                              ; %Flow4561
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.117:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v109, 0x10000, v104
	v_cmp_eq_u32_sdwa vcc, v104, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v108, v109, v104, vcc
; %bb.118:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v105
	v_cmp_ne_u32_e32 vcc, s0, v104
                                        ; implicit-def: $vgpr109
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.119:
	v_bfe_u32 v104, v105, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v109, v105, v104, s2
; %bb.120:                              ; %Flow4560
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.121:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v109, 0x10000, v105
	v_cmp_eq_u32_sdwa vcc, v105, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v109, v109, v105, vcc
; %bb.122:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v106
	v_cmp_ne_u32_e32 vcc, s0, v104
                                        ; implicit-def: $vgpr111
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.123:
	v_bfe_u32 v104, v106, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v111, v106, v104, s2
; %bb.124:                              ; %Flow4559
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.125:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v105, 0x10000, v106
	v_cmp_eq_u32_sdwa vcc, v106, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v111, v105, v106, vcc
; %bb.126:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v107
	v_cmp_ne_u32_e32 vcc, s0, v104
                                        ; implicit-def: $vgpr112
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.127:
	v_bfe_u32 v104, v107, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v112, v107, v104, s2
                                        ; implicit-def: $vgpr104_vgpr105_vgpr106_vgpr107
; %bb.128:                              ; %Flow4558
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.129:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v105, 0x10000, v107
	v_cmp_eq_u32_sdwa vcc, v107, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v112, v105, v107, vcc
; %bb.130:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v104, 16, v108
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v105, 0xffff0000, v112
	v_or_b32_sdwa v105, v105, v111 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v104, v109, s0, v104
	v_add_lshl_u32 v106, v123, v115, 1
	s_mov_b64 s[2:3], exec
.LBB0_131:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[104:105], v106, s[4:7], 0 offen
                                        ; implicit-def: $vgpr104_vgpr105
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_131
; %bb.132:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v100
	v_cmp_ne_u32_e32 vcc, s0, v104
                                        ; implicit-def: $vgpr104
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.133:
	v_bfe_u32 v104, v100, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v100, v104, s2
; %bb.134:                              ; %Flow4557
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.135:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v105, 0x10000, v100
	v_cmp_eq_u32_sdwa vcc, v100, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v105, v100, vcc
; %bb.136:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v101
	v_cmp_ne_u32_e32 vcc, s0, v100
                                        ; implicit-def: $vgpr105
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.137:
	v_bfe_u32 v100, v101, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v105, v101, v100, s2
; %bb.138:                              ; %Flow4556
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.139:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v105, 0x10000, v101
	v_cmp_eq_u32_sdwa vcc, v101, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v105, v105, v101, vcc
; %bb.140:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v102
	v_cmp_ne_u32_e32 vcc, s0, v100
                                        ; implicit-def: $vgpr107
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.141:
	v_bfe_u32 v100, v102, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v107, v102, v100, s2
; %bb.142:                              ; %Flow4555
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.143:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v101, 0x10000, v102
	v_cmp_eq_u32_sdwa vcc, v102, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v107, v101, v102, vcc
; %bb.144:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v103
	v_cmp_ne_u32_e32 vcc, s0, v100
                                        ; implicit-def: $vgpr108
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.145:
	v_bfe_u32 v100, v103, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v108, v103, v100, s2
                                        ; implicit-def: $vgpr100_vgpr101_vgpr102_vgpr103
; %bb.146:                              ; %Flow4554
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.147:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v101, 0x10000, v103
	v_cmp_eq_u32_sdwa vcc, v103, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v108, v101, v103, vcc
; %bb.148:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v100, 16, v104
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v101, 0xffff0000, v108
	v_or_b32_sdwa v101, v101, v107 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v100, v105, s0, v100
	v_add_lshl_u32 v102, v119, v115, 1
	s_mov_b64 s[2:3], exec
.LBB0_149:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[100:101], v102, s[4:7], 0 offen
                                        ; implicit-def: $vgpr100_vgpr101
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_149
; %bb.150:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v96
	v_cmp_ne_u32_e32 vcc, s0, v100
                                        ; implicit-def: $vgpr100
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.151:
	v_bfe_u32 v100, v96, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v96, v100, s2
; %bb.152:                              ; %Flow4553
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.153:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v101, 0x10000, v96
	v_cmp_eq_u32_sdwa vcc, v96, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v101, v96, vcc
; %bb.154:                              ; %_ZL16__float2bfloat16f.exit.i.i.296.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v97
	v_cmp_ne_u32_e32 vcc, s0, v96
                                        ; implicit-def: $vgpr101
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.155:
	v_bfe_u32 v96, v97, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v101, v97, v96, s2
; %bb.156:                              ; %Flow4552
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.157:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v101, 0x10000, v97
	v_cmp_eq_u32_sdwa vcc, v97, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v101, v101, v97, vcc
; %bb.158:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2103.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v98
	v_cmp_ne_u32_e32 vcc, s0, v96
                                        ; implicit-def: $vgpr103
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.159:
	v_bfe_u32 v96, v98, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v103, v98, v96, s2
; %bb.160:                              ; %Flow4551
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.161:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v97, 0x10000, v98
	v_cmp_eq_u32_sdwa vcc, v98, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v103, v97, v98, vcc
; %bb.162:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2107.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v99
	v_cmp_ne_u32_e32 vcc, s0, v96
                                        ; implicit-def: $vgpr104
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.163:
	v_bfe_u32 v96, v99, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v99, v96, s2
                                        ; implicit-def: $vgpr96_vgpr97_vgpr98_vgpr99
; %bb.164:                              ; %Flow4550
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.165:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v97, 0x10000, v99
	v_cmp_eq_u32_sdwa vcc, v99, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v97, v99, vcc
; %bb.166:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2118.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v96, 16, v100
	v_or_b32_e32 v99, 32, v146
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v97, 0xffff0000, v104
	v_or_b32_sdwa v97, v97, v103 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v96, v101, s0, v96
	v_add_lshl_u32 v98, v134, v99, 1
	s_mov_b64 s[2:3], exec
.LBB0_167:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[96:97], v98, s[4:7], 0 offen
                                        ; implicit-def: $vgpr96_vgpr97
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_167
; %bb.168:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v92
	v_cmp_ne_u32_e32 vcc, s0, v96
                                        ; implicit-def: $vgpr96
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.169:
	v_bfe_u32 v96, v92, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v96, v92, v96, s2
; %bb.170:                              ; %Flow4549
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.171:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v97, 0x10000, v92
	v_cmp_eq_u32_sdwa vcc, v92, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v96, v97, v92, vcc
; %bb.172:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v93
	v_cmp_ne_u32_e32 vcc, s0, v92
                                        ; implicit-def: $vgpr97
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.173:
	v_bfe_u32 v92, v93, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v97, v93, v92, s2
; %bb.174:                              ; %Flow4548
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.175:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v97, 0x10000, v93
	v_cmp_eq_u32_sdwa vcc, v93, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v97, v97, v93, vcc
; %bb.176:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v94
	v_cmp_ne_u32_e32 vcc, s0, v92
                                        ; implicit-def: $vgpr100
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.177:
	v_bfe_u32 v92, v94, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v94, v92, s2
; %bb.178:                              ; %Flow4547
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.179:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v93, 0x10000, v94
	v_cmp_eq_u32_sdwa vcc, v94, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v93, v94, vcc
; %bb.180:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v95
	v_cmp_ne_u32_e32 vcc, s0, v92
                                        ; implicit-def: $vgpr101
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.181:
	v_bfe_u32 v92, v95, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v101, v95, v92, s2
                                        ; implicit-def: $vgpr92_vgpr93_vgpr94_vgpr95
; %bb.182:                              ; %Flow4546
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.183:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v93, 0x10000, v95
	v_cmp_eq_u32_sdwa vcc, v95, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v101, v93, v95, vcc
; %bb.184:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v92, 16, v96
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v93, 0xffff0000, v101
	v_or_b32_sdwa v93, v93, v100 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v92, v97, s0, v92
	v_add_lshl_u32 v94, v127, v99, 1
	s_mov_b64 s[2:3], exec
.LBB0_185:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[92:93], v94, s[4:7], 0 offen
                                        ; implicit-def: $vgpr92_vgpr93
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_185
; %bb.186:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v88
	v_cmp_ne_u32_e32 vcc, s0, v92
                                        ; implicit-def: $vgpr92
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.187:
	v_bfe_u32 v92, v88, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v92, v88, v92, s2
; %bb.188:                              ; %Flow4545
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.189:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v93, 0x10000, v88
	v_cmp_eq_u32_sdwa vcc, v88, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v92, v93, v88, vcc
; %bb.190:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v89
	v_cmp_ne_u32_e32 vcc, s0, v88
                                        ; implicit-def: $vgpr93
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.191:
	v_bfe_u32 v88, v89, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v93, v89, v88, s2
; %bb.192:                              ; %Flow4544
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.193:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v93, 0x10000, v89
	v_cmp_eq_u32_sdwa vcc, v89, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v89, vcc
; %bb.194:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v90
	v_cmp_ne_u32_e32 vcc, s0, v88
                                        ; implicit-def: $vgpr95
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.195:
	v_bfe_u32 v88, v90, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v95, v90, v88, s2
; %bb.196:                              ; %Flow4543
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.197:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v89, 0x10000, v90
	v_cmp_eq_u32_sdwa vcc, v90, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v95, v89, v90, vcc
; %bb.198:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v91
	v_cmp_ne_u32_e32 vcc, s0, v88
                                        ; implicit-def: $vgpr96
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.199:
	v_bfe_u32 v88, v91, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v96, v91, v88, s2
                                        ; implicit-def: $vgpr88_vgpr89_vgpr90_vgpr91
; %bb.200:                              ; %Flow4542
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.201:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v89, 0x10000, v91
	v_cmp_eq_u32_sdwa vcc, v91, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v96, v89, v91, vcc
; %bb.202:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v88, 16, v92
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v89, 0xffff0000, v96
	v_or_b32_sdwa v89, v89, v95 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v88, v93, s0, v88
	v_add_lshl_u32 v90, v123, v99, 1
	s_mov_b64 s[2:3], exec
.LBB0_203:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[88:89], v90, s[4:7], 0 offen
                                        ; implicit-def: $vgpr88_vgpr89
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_203
; %bb.204:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v84
	v_cmp_ne_u32_e32 vcc, s0, v88
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.205:
	v_bfe_u32 v88, v84, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v88, v84, v88, s2
; %bb.206:                              ; %Flow4541
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.207:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v89, 0x10000, v84
	v_cmp_eq_u32_sdwa vcc, v84, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v89, v84, vcc
; %bb.208:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v85
	v_cmp_ne_u32_e32 vcc, s0, v84
                                        ; implicit-def: $vgpr89
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.209:
	v_bfe_u32 v84, v85, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v89, v85, v84, s2
; %bb.210:                              ; %Flow4540
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.211:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v89, 0x10000, v85
	v_cmp_eq_u32_sdwa vcc, v85, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v89, v89, v85, vcc
; %bb.212:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v86
	v_cmp_ne_u32_e32 vcc, s0, v84
                                        ; implicit-def: $vgpr91
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.213:
	v_bfe_u32 v84, v86, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v91, v86, v84, s2
; %bb.214:                              ; %Flow4539
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.215:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v85, 0x10000, v86
	v_cmp_eq_u32_sdwa vcc, v86, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v91, v85, v86, vcc
; %bb.216:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v87
	v_cmp_ne_u32_e32 vcc, s0, v84
                                        ; implicit-def: $vgpr92
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.217:
	v_bfe_u32 v84, v87, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v92, v87, v84, s2
                                        ; implicit-def: $vgpr84_vgpr85_vgpr86_vgpr87
; %bb.218:                              ; %Flow4538
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.219:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v85, 0x10000, v87
	v_cmp_eq_u32_sdwa vcc, v87, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v92, v85, v87, vcc
; %bb.220:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v84, 16, v88
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v85, 0xffff0000, v92
	v_or_b32_sdwa v85, v85, v91 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v84, v89, s0, v84
	v_add_lshl_u32 v86, v119, v99, 1
	s_mov_b64 s[2:3], exec
.LBB0_221:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[84:85], v86, s[4:7], 0 offen
                                        ; implicit-def: $vgpr84_vgpr85
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_221
; %bb.222:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v80
	v_cmp_ne_u32_e32 vcc, s0, v84
                                        ; implicit-def: $vgpr84
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.223:
	v_bfe_u32 v84, v80, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v80, v84, s2
; %bb.224:                              ; %Flow4537
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.225:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v85, 0x10000, v80
	v_cmp_eq_u32_sdwa vcc, v80, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v84, v85, v80, vcc
; %bb.226:                              ; %_ZL16__float2bfloat16f.exit.i.i.3125.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v81
	v_cmp_ne_u32_e32 vcc, s0, v80
                                        ; implicit-def: $vgpr85
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.227:
	v_bfe_u32 v80, v81, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v85, v81, v80, s2
; %bb.228:                              ; %Flow4536
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.229:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v85, 0x10000, v81
	v_cmp_eq_u32_sdwa vcc, v81, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v81, vcc
; %bb.230:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3132.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v82
	v_cmp_ne_u32_e32 vcc, s0, v80
                                        ; implicit-def: $vgpr87
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.231:
	v_bfe_u32 v80, v82, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v87, v82, v80, s2
; %bb.232:                              ; %Flow4535
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.233:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v81, 0x10000, v82
	v_cmp_eq_u32_sdwa vcc, v82, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v87, v81, v82, vcc
; %bb.234:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3136.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v83
	v_cmp_ne_u32_e32 vcc, s0, v80
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.235:
	v_bfe_u32 v80, v83, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v88, v83, v80, s2
                                        ; implicit-def: $vgpr80_vgpr81_vgpr82_vgpr83
; %bb.236:                              ; %Flow4534
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.237:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v81, 0x10000, v83
	v_cmp_eq_u32_sdwa vcc, v83, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v81, v83, vcc
; %bb.238:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3147.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v80, 16, v84
	v_or_b32_e32 v83, 48, v146
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v81, 0xffff0000, v88
	v_or_b32_sdwa v81, v81, v87 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v80, v85, s0, v80
	v_add_lshl_u32 v82, v134, v83, 1
	s_mov_b64 s[2:3], exec
.LBB0_239:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[80:81], v82, s[4:7], 0 offen
                                        ; implicit-def: $vgpr80_vgpr81
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_239
; %bb.240:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v76
	v_cmp_ne_u32_e32 vcc, s0, v80
                                        ; implicit-def: $vgpr80
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.241:
	v_bfe_u32 v80, v76, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v80, v76, v80, s2
; %bb.242:                              ; %Flow4533
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.243:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v81, 0x10000, v76
	v_cmp_eq_u32_sdwa vcc, v76, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v80, v81, v76, vcc
; %bb.244:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v77
	v_cmp_ne_u32_e32 vcc, s0, v76
                                        ; implicit-def: $vgpr81
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.245:
	v_bfe_u32 v76, v77, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v81, v77, v76, s2
; %bb.246:                              ; %Flow4532
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.247:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v81, 0x10000, v77
	v_cmp_eq_u32_sdwa vcc, v77, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v77, vcc
; %bb.248:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v78
	v_cmp_ne_u32_e32 vcc, s0, v76
                                        ; implicit-def: $vgpr84
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.249:
	v_bfe_u32 v76, v78, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v78, v76, s2
; %bb.250:                              ; %Flow4531
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.251:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v77, 0x10000, v78
	v_cmp_eq_u32_sdwa vcc, v78, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v84, v77, v78, vcc
; %bb.252:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v79
	v_cmp_ne_u32_e32 vcc, s0, v76
                                        ; implicit-def: $vgpr85
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.253:
	v_bfe_u32 v76, v79, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v85, v79, v76, s2
                                        ; implicit-def: $vgpr76_vgpr77_vgpr78_vgpr79
; %bb.254:                              ; %Flow4530
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.255:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v77, 0x10000, v79
	v_cmp_eq_u32_sdwa vcc, v79, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v85, v77, v79, vcc
; %bb.256:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v76, 16, v80
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v77, 0xffff0000, v85
	v_or_b32_sdwa v77, v77, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v76, v81, s0, v76
	v_add_lshl_u32 v78, v127, v83, 1
	s_mov_b64 s[2:3], exec
.LBB0_257:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[76:77], v78, s[4:7], 0 offen
                                        ; implicit-def: $vgpr76_vgpr77
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_257
; %bb.258:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v72
	v_cmp_ne_u32_e32 vcc, s0, v76
                                        ; implicit-def: $vgpr76
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.259:
	v_bfe_u32 v76, v72, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v76, v72, v76, s2
; %bb.260:                              ; %Flow4529
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.261:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v77, 0x10000, v72
	v_cmp_eq_u32_sdwa vcc, v72, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v76, v77, v72, vcc
; %bb.262:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v73
	v_cmp_ne_u32_e32 vcc, s0, v72
                                        ; implicit-def: $vgpr77
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.263:
	v_bfe_u32 v72, v73, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v77, v73, v72, s2
; %bb.264:                              ; %Flow4528
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.265:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v77, 0x10000, v73
	v_cmp_eq_u32_sdwa vcc, v73, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v73, vcc
; %bb.266:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v74
	v_cmp_ne_u32_e32 vcc, s0, v72
                                        ; implicit-def: $vgpr79
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.267:
	v_bfe_u32 v72, v74, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v79, v74, v72, s2
; %bb.268:                              ; %Flow4527
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.269:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v73, 0x10000, v74
	v_cmp_eq_u32_sdwa vcc, v74, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v79, v73, v74, vcc
; %bb.270:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v75
	v_cmp_ne_u32_e32 vcc, s0, v72
                                        ; implicit-def: $vgpr80
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.271:
	v_bfe_u32 v72, v75, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v80, v75, v72, s2
                                        ; implicit-def: $vgpr72_vgpr73_vgpr74_vgpr75
; %bb.272:                              ; %Flow4526
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.273:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v73, 0x10000, v75
	v_cmp_eq_u32_sdwa vcc, v75, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v80, v73, v75, vcc
; %bb.274:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v72, 16, v76
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v73, 0xffff0000, v80
	v_or_b32_sdwa v73, v73, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v72, v77, s0, v72
	v_add_lshl_u32 v74, v123, v83, 1
	s_mov_b64 s[2:3], exec
.LBB0_275:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[72:73], v74, s[4:7], 0 offen
                                        ; implicit-def: $vgpr72_vgpr73
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_275
; %bb.276:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v64
	v_cmp_ne_u32_e32 vcc, s0, v72
                                        ; implicit-def: $vgpr72
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.277:
	v_bfe_u32 v72, v64, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v72, v64, v72, s2
; %bb.278:                              ; %Flow4525
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.279:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v73, 0x10000, v64
	v_cmp_eq_u32_sdwa vcc, v64, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v72, v73, v64, vcc
; %bb.280:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v65
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr73
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.281:
	v_bfe_u32 v64, v65, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v73, v65, v64, s2
; %bb.282:                              ; %Flow4524
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.283:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v73, 0x10000, v65
	v_cmp_eq_u32_sdwa vcc, v65, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v73, v73, v65, vcc
; %bb.284:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v66
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr75
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.285:
	v_bfe_u32 v64, v66, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v75, v66, v64, s2
; %bb.286:                              ; %Flow4523
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.287:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v66
	v_cmp_eq_u32_sdwa vcc, v66, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v75, v65, v66, vcc
; %bb.288:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v67
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr76
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.289:
	v_bfe_u32 v64, v67, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v76, v67, v64, s2
                                        ; implicit-def: $vgpr64_vgpr65_vgpr66_vgpr67
; %bb.290:                              ; %Flow4522
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.291:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v67
	v_cmp_eq_u32_sdwa vcc, v67, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v76, v65, v67, vcc
; %bb.292:                              ; %_ZL16store_transposedITkN7kittens5ducks2rt3allENS0_2rtIfLi64ELi64ENS1_9rt_layout3colEEETkNS1_2gl3allENS0_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS6_EEEvRKT0_RKT_RKT1_.exit
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v64, 16, v72
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v65, 0xffff0000, v76
	v_or_b32_sdwa v65, v65, v75 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v64, v73, s0, v64
	v_add_lshl_u32 v66, v119, v83, 1
	s_mov_b64 s[2:3], exec
.LBB0_293:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v68
	v_readfirstlane_b32 s5, v69
	v_readfirstlane_b32 s6, v70
	v_readfirstlane_b32 s7, v71
	v_cmp_eq_u64_e32 vcc, s[4:5], v[68:69]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[70:71]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[64:65], v66, s[4:7], 0 offen
                                        ; implicit-def: $vgpr68_vgpr69_vgpr70_vgpr71
                                        ; implicit-def: $vgpr64_vgpr65
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_293
; %bb.294:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v60
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr64
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.295:
	v_bfe_u32 v64, v60, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v64, v60, v64, s2
; %bb.296:                              ; %Flow4521
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.297:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v60
	v_cmp_eq_u32_sdwa vcc, v60, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v64, v65, v60, vcc
; %bb.298:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.i1753
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v61
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr67
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.299:
	v_bfe_u32 v60, v61, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v61, v60, s2
; %bb.300:                              ; %Flow4520
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.301:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v65, 0x10000, v61
	v_cmp_eq_u32_sdwa vcc, v61, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v61, vcc
; %bb.302:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i1756
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v62
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr65
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.303:
	v_bfe_u32 v60, v62, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v65, v62, v60, s2
; %bb.304:                              ; %Flow4519
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.305:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v61, 0x10000, v62
	v_cmp_eq_u32_sdwa vcc, v62, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v65, v61, v62, vcc
; %bb.306:                              ; %_ZL16__float2bfloat16f.exit.i.i46.i.i1762
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v63
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.307:
	v_bfe_u32 v60, v63, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v63, v60, s2
                                        ; implicit-def: $vgpr60_vgpr61_vgpr62_vgpr63
; %bb.308:                              ; %Flow4518
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.309:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v61, 0x10000, v63
	v_cmp_eq_u32_sdwa vcc, v63, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v61, v63, vcc
; %bb.310:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.i.i1765
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v60, 0x80, v128
	v_lshrrev_b32_e32 v64, 16, v64
	v_ashrrev_i32_e32 v61, 31, v60
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v68, 0xffff0000, v68
	v_lshl_add_u64 v[60:61], v[60:61], 1, v[130:131]
	v_mov_b32_e32 v62, s8
	v_mov_b32_e32 v63, 0x20000
	v_or_b32_sdwa v65, v68, v65 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v64, v67, s0, v64
	s_mov_b64 s[2:3], exec
.LBB0_311:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[64:65], v129, s[4:7], 0 offen
                                        ; implicit-def: $vgpr64_vgpr65
                                        ; implicit-def: $vgpr129
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_311
; %bb.312:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v56
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr64
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.313:
	v_bfe_u32 v64, v56, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v64, v56, v64, s2
; %bb.314:                              ; %Flow4517
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.315:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v56
	v_cmp_eq_u32_sdwa vcc, v56, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v64, v65, v56, vcc
; %bb.316:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i1778
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v57
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr65
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.317:
	v_bfe_u32 v56, v57, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v65, v57, v56, s2
; %bb.318:                              ; %Flow4516
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.319:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v65, 0x10000, v57
	v_cmp_eq_u32_sdwa vcc, v57, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v57, vcc
; %bb.320:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.i.i1781
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v58
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr67
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.321:
	v_bfe_u32 v56, v58, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v58, v56, s2
; %bb.322:                              ; %Flow4515
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.323:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v58
	v_cmp_eq_u32_sdwa vcc, v58, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v67, v57, v58, vcc
; %bb.324:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.i.i1787
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v59
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.325:
	v_bfe_u32 v56, v59, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v59, v56, s2
                                        ; implicit-def: $vgpr56_vgpr57_vgpr58_vgpr59
; %bb.326:                              ; %Flow4514
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.327:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v59
	v_cmp_eq_u32_sdwa vcc, v59, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v57, v59, vcc
; %bb.328:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.i.i1790
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v56, 16, v64
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v57, 0xffff0000, v68
	v_or_b32_sdwa v57, v57, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v56, v65, s0, v56
	s_mov_b64 s[2:3], exec
.LBB0_329:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[56:57], v126, s[4:7], 0 offen
                                        ; implicit-def: $vgpr56_vgpr57
                                        ; implicit-def: $vgpr126
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_329
; %bb.330:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v52
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.331:
	v_bfe_u32 v56, v52, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v56, v52, v56, s2
; %bb.332:                              ; %Flow4513
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.333:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v52
	v_cmp_eq_u32_sdwa vcc, v52, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v56, v57, v52, vcc
; %bb.334:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.i.i1803
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v53
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr57
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.335:
	v_bfe_u32 v52, v53, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v57, v53, v52, s2
; %bb.336:                              ; %Flow4512
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.337:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v57, 0x10000, v53
	v_cmp_eq_u32_sdwa vcc, v53, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v53, vcc
; %bb.338:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.i.i1806
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v54
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr58
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.339:
	v_bfe_u32 v52, v54, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v58, v54, v52, s2
; %bb.340:                              ; %Flow4511
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.341:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v53, 0x10000, v54
	v_cmp_eq_u32_sdwa vcc, v54, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v58, v53, v54, vcc
; %bb.342:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.i.i1812
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v55
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr59
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.343:
	v_bfe_u32 v52, v55, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v59, v55, v52, s2
                                        ; implicit-def: $vgpr52_vgpr53_vgpr54_vgpr55
; %bb.344:                              ; %Flow4510
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.345:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v53, 0x10000, v55
	v_cmp_eq_u32_sdwa vcc, v55, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v59, v53, v55, vcc
; %bb.346:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.i.i1815
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v52, 16, v56
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v53, 0xffff0000, v59
	v_or_b32_sdwa v53, v53, v58 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v52, v57, s0, v52
	s_mov_b64 s[2:3], exec
.LBB0_347:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[52:53], v122, s[4:7], 0 offen
                                        ; implicit-def: $vgpr52_vgpr53
                                        ; implicit-def: $vgpr122
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_347
; %bb.348:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v48
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.349:
	v_bfe_u32 v52, v48, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v48, v52, s2
; %bb.350:                              ; %Flow4509
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.351:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v53, 0x10000, v48
	v_cmp_eq_u32_sdwa vcc, v48, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v53, v48, vcc
; %bb.352:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.i.i1828
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v49
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr53
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.353:
	v_bfe_u32 v48, v49, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v53, v49, v48, s2
; %bb.354:                              ; %Flow4508
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.355:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v53, 0x10000, v49
	v_cmp_eq_u32_sdwa vcc, v49, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v49, vcc
; %bb.356:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.i.i1831
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v50
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr54
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.357:
	v_bfe_u32 v48, v50, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v54, v50, v48, s2
; %bb.358:                              ; %Flow4507
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.359:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v49, 0x10000, v50
	v_cmp_eq_u32_sdwa vcc, v50, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v54, v49, v50, vcc
; %bb.360:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.i.i1837
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v51
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr55
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.361:
	v_bfe_u32 v48, v51, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v55, v51, v48, s2
                                        ; implicit-def: $vgpr48_vgpr49_vgpr50_vgpr51
; %bb.362:                              ; %Flow4506
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.363:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v49, 0x10000, v51
	v_cmp_eq_u32_sdwa vcc, v51, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v55, v49, v51, vcc
; %bb.364:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3.i.i1840
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v48, 16, v52
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v49, 0xffff0000, v55
	v_or_b32_sdwa v49, v49, v54 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v48, v53, s0, v48
	s_mov_b64 s[2:3], exec
.LBB0_365:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[48:49], v118, s[4:7], 0 offen
                                        ; implicit-def: $vgpr48_vgpr49
                                        ; implicit-def: $vgpr118
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_365
; %bb.366:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v44
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr48
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.367:
	v_bfe_u32 v48, v44, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v48, v44, v48, s2
; %bb.368:                              ; %Flow4505
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.369:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v49, 0x10000, v44
	v_cmp_eq_u32_sdwa vcc, v44, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v48, v49, v44, vcc
; %bb.370:                              ; %_ZL16__float2bfloat16f.exit.i.i.167.i.i1853
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v45
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr49
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.371:
	v_bfe_u32 v44, v45, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v49, v45, v44, s2
; %bb.372:                              ; %Flow4504
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.373:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v49, 0x10000, v45
	v_cmp_eq_u32_sdwa vcc, v45, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v45, vcc
; %bb.374:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.174.i.i1856
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v46
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr50
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.375:
	v_bfe_u32 v44, v46, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v50, v46, v44, s2
; %bb.376:                              ; %Flow4503
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.377:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v45, 0x10000, v46
	v_cmp_eq_u32_sdwa vcc, v46, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v50, v45, v46, vcc
; %bb.378:                              ; %_ZL16__float2bfloat16f.exit.i.i46.178.i.i1862
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v47
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr51
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.379:
	v_bfe_u32 v44, v47, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v47, v44, s2
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
; %bb.380:                              ; %Flow4502
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.381:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v45, 0x10000, v47
	v_cmp_eq_u32_sdwa vcc, v47, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v51, v45, v47, vcc
; %bb.382:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.189.i.i1865
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v44, 16, v48
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v45, 0xffff0000, v51
	v_or_b32_sdwa v45, v45, v50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v44, v49, s0, v44
	s_mov_b64 s[2:3], exec
.LBB0_383:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[44:45], v114, s[4:7], 0 offen
                                        ; implicit-def: $vgpr44_vgpr45
                                        ; implicit-def: $vgpr114
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_383
; %bb.384:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v40
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr44
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.385:
	v_bfe_u32 v44, v40, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v44, v40, v44, s2
; %bb.386:                              ; %Flow4501
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.387:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v45, 0x10000, v40
	v_cmp_eq_u32_sdwa vcc, v40, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v44, v45, v40, vcc
; %bb.388:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.1.i.i1878
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v41
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr45
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.389:
	v_bfe_u32 v40, v41, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v45, v41, v40, s2
; %bb.390:                              ; %Flow4500
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.391:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v45, 0x10000, v41
	v_cmp_eq_u32_sdwa vcc, v41, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v41, vcc
; %bb.392:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.1.i.i1881
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v42
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr46
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.393:
	v_bfe_u32 v40, v42, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v46, v42, v40, s2
; %bb.394:                              ; %Flow4499
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.395:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v41, 0x10000, v42
	v_cmp_eq_u32_sdwa vcc, v42, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v46, v41, v42, vcc
; %bb.396:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.1.i.i1887
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v43
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr47
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.397:
	v_bfe_u32 v40, v43, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v47, v43, v40, s2
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
; %bb.398:                              ; %Flow4498
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.399:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v41, 0x10000, v43
	v_cmp_eq_u32_sdwa vcc, v43, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v47, v41, v43, vcc
; %bb.400:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.1.i.i1890
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v40, 16, v44
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v41, 0xffff0000, v47
	v_or_b32_sdwa v41, v41, v46 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v40, v45, s0, v40
	s_mov_b64 s[2:3], exec
.LBB0_401:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[40:41], v110, s[4:7], 0 offen
                                        ; implicit-def: $vgpr40_vgpr41
                                        ; implicit-def: $vgpr110
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_401
; %bb.402:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v36
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.403:
	v_bfe_u32 v40, v36, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v36, v40, s2
; %bb.404:                              ; %Flow4497
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.405:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v41, 0x10000, v36
	v_cmp_eq_u32_sdwa vcc, v36, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v41, v36, vcc
; %bb.406:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.1.i.i1903
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v37
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.407:
	v_bfe_u32 v36, v37, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v41, v37, v36, s2
; %bb.408:                              ; %Flow4496
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.409:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v41, 0x10000, v37
	v_cmp_eq_u32_sdwa vcc, v37, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v37, vcc
; %bb.410:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.1.i.i1906
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v38
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr42
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.411:
	v_bfe_u32 v36, v38, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v42, v38, v36, s2
; %bb.412:                              ; %Flow4495
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.413:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v37, 0x10000, v38
	v_cmp_eq_u32_sdwa vcc, v38, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v42, v37, v38, vcc
; %bb.414:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.1.i.i1912
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v39
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr43
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.415:
	v_bfe_u32 v36, v39, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v43, v39, v36, s2
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
; %bb.416:                              ; %Flow4494
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.417:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v37, 0x10000, v39
	v_cmp_eq_u32_sdwa vcc, v39, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v43, v37, v39, vcc
; %bb.418:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.1.i.i1915
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v36, 16, v40
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v37, 0xffff0000, v43
	v_or_b32_sdwa v37, v37, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v36, v41, s0, v36
	s_mov_b64 s[2:3], exec
.LBB0_419:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[36:37], v106, s[4:7], 0 offen
                                        ; implicit-def: $vgpr36_vgpr37
                                        ; implicit-def: $vgpr106
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_419
; %bb.420:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v32
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.421:
	v_bfe_u32 v36, v32, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v32, v36, s2
; %bb.422:                              ; %Flow4493
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.423:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v37, 0x10000, v32
	v_cmp_eq_u32_sdwa vcc, v32, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v37, v32, vcc
; %bb.424:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.1.i.i1928
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v33
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr37
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.425:
	v_bfe_u32 v32, v33, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v37, v33, v32, s2
; %bb.426:                              ; %Flow4492
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.427:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v37, 0x10000, v33
	v_cmp_eq_u32_sdwa vcc, v33, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v33, vcc
; %bb.428:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.1.i.i1931
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v34
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr38
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.429:
	v_bfe_u32 v32, v34, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v38, v34, v32, s2
; %bb.430:                              ; %Flow4491
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.431:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v33, 0x10000, v34
	v_cmp_eq_u32_sdwa vcc, v34, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v38, v33, v34, vcc
; %bb.432:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.1.i.i1937
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v35
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr39
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.433:
	v_bfe_u32 v32, v35, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v39, v35, v32, s2
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
; %bb.434:                              ; %Flow4490
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.435:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v33, 0x10000, v35
	v_cmp_eq_u32_sdwa vcc, v35, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v39, v33, v35, vcc
; %bb.436:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3.1.i.i1940
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v32, 16, v36
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v33, 0xffff0000, v39
	v_or_b32_sdwa v33, v33, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v32, v37, s0, v32
	s_mov_b64 s[2:3], exec
.LBB0_437:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[32:33], v102, s[4:7], 0 offen
                                        ; implicit-def: $vgpr32_vgpr33
                                        ; implicit-def: $vgpr102
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_437
; %bb.438:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v28
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr32
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.439:
	v_bfe_u32 v32, v28, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v32, v28, v32, s2
; %bb.440:                              ; %Flow4489
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.441:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v33, 0x10000, v28
	v_cmp_eq_u32_sdwa vcc, v28, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v32, v33, v28, vcc
; %bb.442:                              ; %_ZL16__float2bfloat16f.exit.i.i.296.i.i1953
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v29
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr33
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.443:
	v_bfe_u32 v28, v29, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v33, v29, v28, s2
; %bb.444:                              ; %Flow4488
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.445:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v33, 0x10000, v29
	v_cmp_eq_u32_sdwa vcc, v29, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v33, v33, v29, vcc
; %bb.446:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2103.i.i1956
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v30
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.447:
	v_bfe_u32 v28, v30, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v34, v30, v28, s2
; %bb.448:                              ; %Flow4487
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.449:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v29, 0x10000, v30
	v_cmp_eq_u32_sdwa vcc, v30, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v34, v29, v30, vcc
; %bb.450:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2107.i.i1962
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v31
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr35
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.451:
	v_bfe_u32 v28, v31, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v31, v28, s2
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
; %bb.452:                              ; %Flow4486
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.453:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v29, 0x10000, v31
	v_cmp_eq_u32_sdwa vcc, v31, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v35, v29, v31, vcc
; %bb.454:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2118.i.i1965
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v28, 16, v32
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v29, 0xffff0000, v35
	v_or_b32_sdwa v29, v29, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v28, v33, s0, v28
	s_mov_b64 s[2:3], exec
.LBB0_455:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[28:29], v98, s[4:7], 0 offen
                                        ; implicit-def: $vgpr28_vgpr29
                                        ; implicit-def: $vgpr98
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_455
; %bb.456:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v24
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr28
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.457:
	v_bfe_u32 v28, v24, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v28, v24, v28, s2
; %bb.458:                              ; %Flow4485
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.459:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v29, 0x10000, v24
	v_cmp_eq_u32_sdwa vcc, v24, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v28, v29, v24, vcc
; %bb.460:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.2.i.i1978
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v25
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr29
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.461:
	v_bfe_u32 v24, v25, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v29, v25, v24, s2
; %bb.462:                              ; %Flow4484
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.463:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v29, 0x10000, v25
	v_cmp_eq_u32_sdwa vcc, v25, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v25, vcc
; %bb.464:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.2.i.i1981
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v26
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr30
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.465:
	v_bfe_u32 v24, v26, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v30, v26, v24, s2
; %bb.466:                              ; %Flow4483
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.467:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v25, 0x10000, v26
	v_cmp_eq_u32_sdwa vcc, v26, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v30, v25, v26, vcc
; %bb.468:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.2.i.i1987
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v27
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr31
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.469:
	v_bfe_u32 v24, v27, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v31, v27, v24, s2
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
; %bb.470:                              ; %Flow4482
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.471:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v25, 0x10000, v27
	v_cmp_eq_u32_sdwa vcc, v27, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v31, v25, v27, vcc
; %bb.472:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.2.i.i1990
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v24, 16, v28
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v25, 0xffff0000, v31
	v_or_b32_sdwa v25, v25, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v24, v29, s0, v24
	s_mov_b64 s[2:3], exec
.LBB0_473:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[24:25], v94, s[4:7], 0 offen
                                        ; implicit-def: $vgpr24_vgpr25
                                        ; implicit-def: $vgpr94
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_473
; %bb.474:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v20
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.475:
	v_bfe_u32 v24, v20, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v20, v24, s2
; %bb.476:                              ; %Flow4481
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.477:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v25, 0x10000, v20
	v_cmp_eq_u32_sdwa vcc, v20, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v25, v20, vcc
; %bb.478:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.2.i.i2003
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v21
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr25
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.479:
	v_bfe_u32 v20, v21, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v25, v21, v20, s2
; %bb.480:                              ; %Flow4480
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.481:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v25, 0x10000, v21
	v_cmp_eq_u32_sdwa vcc, v21, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v25, v25, v21, vcc
; %bb.482:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.2.i.i2006
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v22
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr26
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.483:
	v_bfe_u32 v20, v22, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v26, v22, v20, s2
; %bb.484:                              ; %Flow4479
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.485:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v21, 0x10000, v22
	v_cmp_eq_u32_sdwa vcc, v22, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v26, v21, v22, vcc
; %bb.486:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.2.i.i2012
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v23
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr27
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.487:
	v_bfe_u32 v20, v23, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v27, v23, v20, s2
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
; %bb.488:                              ; %Flow4478
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.489:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v21, 0x10000, v23
	v_cmp_eq_u32_sdwa vcc, v23, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v27, v21, v23, vcc
; %bb.490:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.2.i.i2015
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v20, 16, v24
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v21, 0xffff0000, v27
	v_or_b32_sdwa v21, v21, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v20, v25, s0, v20
	s_mov_b64 s[2:3], exec
.LBB0_491:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[20:21], v90, s[4:7], 0 offen
                                        ; implicit-def: $vgpr20_vgpr21
                                        ; implicit-def: $vgpr90
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_491
; %bb.492:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v16
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.493:
	v_bfe_u32 v20, v16, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v16, v20, s2
; %bb.494:                              ; %Flow4477
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.495:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v21, 0x10000, v16
	v_cmp_eq_u32_sdwa vcc, v16, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v21, v16, vcc
; %bb.496:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.2.i.i2028
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v17
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr21
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.497:
	v_bfe_u32 v16, v17, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v21, v17, v16, s2
; %bb.498:                              ; %Flow4476
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.499:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v21, 0x10000, v17
	v_cmp_eq_u32_sdwa vcc, v17, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v17, vcc
; %bb.500:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.2.i.i2031
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v18
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.501:
	v_bfe_u32 v16, v18, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v22, v18, v16, s2
; %bb.502:                              ; %Flow4475
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.503:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v17, 0x10000, v18
	v_cmp_eq_u32_sdwa vcc, v18, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v22, v17, v18, vcc
; %bb.504:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.2.i.i2037
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr23
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.505:
	v_bfe_u32 v16, v19, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v23, v19, v16, s2
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
; %bb.506:                              ; %Flow4474
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.507:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v17, 0x10000, v19
	v_cmp_eq_u32_sdwa vcc, v19, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v23, v17, v19, vcc
; %bb.508:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3.2.i.i2040
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v16, 16, v20
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v17, 0xffff0000, v23
	v_or_b32_sdwa v17, v17, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v16, v21, s0, v16
	s_mov_b64 s[2:3], exec
.LBB0_509:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[16:17], v86, s[4:7], 0 offen
                                        ; implicit-def: $vgpr16_vgpr17
                                        ; implicit-def: $vgpr86
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_509
; %bb.510:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v12
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr16
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.511:
	v_bfe_u32 v16, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v16, v12, v16, s2
; %bb.512:                              ; %Flow4473
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.513:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v17, 0x10000, v12
	v_cmp_eq_u32_sdwa vcc, v12, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v16, v17, v12, vcc
; %bb.514:                              ; %_ZL16__float2bfloat16f.exit.i.i.3125.i.i2053
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v13
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr17
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.515:
	v_bfe_u32 v12, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v17, v13, v12, s2
; %bb.516:                              ; %Flow4472
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.517:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v17, 0x10000, v13
	v_cmp_eq_u32_sdwa vcc, v13, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v17, v17, v13, vcc
; %bb.518:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3132.i.i2056
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v14
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.519:
	v_bfe_u32 v12, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v14, v12, s2
; %bb.520:                              ; %Flow4471
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.521:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v13, 0x10000, v14
	v_cmp_eq_u32_sdwa vcc, v14, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v18, v13, v14, vcc
; %bb.522:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3136.i.i2062
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr19
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.523:
	v_bfe_u32 v12, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v19, v15, v12, s2
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
; %bb.524:                              ; %Flow4470
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.525:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v13, 0x10000, v15
	v_cmp_eq_u32_sdwa vcc, v15, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v19, v13, v15, vcc
; %bb.526:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.3147.i.i2065
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v12, 16, v16
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v13, 0xffff0000, v19
	v_or_b32_sdwa v13, v13, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v12, v17, s0, v12
	s_mov_b64 s[2:3], exec
.LBB0_527:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[12:13], v82, s[4:7], 0 offen
                                        ; implicit-def: $vgpr12_vgpr13
                                        ; implicit-def: $vgpr82
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_527
; %bb.528:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v8
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.529:
	v_bfe_u32 v12, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v12, v8, v12, s2
; %bb.530:                              ; %Flow4469
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.531:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v13, 0x10000, v8
	v_cmp_eq_u32_sdwa vcc, v8, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v12, v13, v8, vcc
; %bb.532:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.3.i.i2078
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v9
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr13
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.533:
	v_bfe_u32 v8, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v13, v9, v8, s2
; %bb.534:                              ; %Flow4468
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.535:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v13, 0x10000, v9
	v_cmp_eq_u32_sdwa vcc, v9, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v13, v13, v9, vcc
; %bb.536:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.3.i.i2081
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v10
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.537:
	v_bfe_u32 v8, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v14, v10, v8, s2
; %bb.538:                              ; %Flow4467
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.539:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v9, 0x10000, v10
	v_cmp_eq_u32_sdwa vcc, v10, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v14, v9, v10, vcc
; %bb.540:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.3.i.i2087
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v11
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr15
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.541:
	v_bfe_u32 v8, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v15, v11, v8, s2
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
; %bb.542:                              ; %Flow4466
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.543:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v9, 0x10000, v11
	v_cmp_eq_u32_sdwa vcc, v11, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v15, v9, v11, vcc
; %bb.544:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.3.i.i2090
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v8, 16, v12
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v9, 0xffff0000, v15
	v_or_b32_sdwa v9, v9, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v8, v13, s0, v8
	s_mov_b64 s[2:3], exec
.LBB0_545:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[8:9], v78, s[4:7], 0 offen
                                        ; implicit-def: $vgpr8_vgpr9
                                        ; implicit-def: $vgpr78
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_545
; %bb.546:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v4
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.547:
	v_bfe_u32 v8, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v8, v4, v8, s2
; %bb.548:                              ; %Flow4465
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.549:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v9, 0x10000, v4
	v_cmp_eq_u32_sdwa vcc, v4, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v8, v9, v4, vcc
; %bb.550:                              ; %_ZL16__float2bfloat16f.exit.i.i.2.3.i.i2103
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v5
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.551:
	v_bfe_u32 v4, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v9, v5, v4, s2
; %bb.552:                              ; %Flow4464
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.553:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v9, 0x10000, v5
	v_cmp_eq_u32_sdwa vcc, v5, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v9, v9, v5, vcc
; %bb.554:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.2.3.i.i2106
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v6
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.555:
	v_bfe_u32 v4, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v10, v6, v4, s2
; %bb.556:                              ; %Flow4463
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.557:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v5, 0x10000, v6
	v_cmp_eq_u32_sdwa vcc, v6, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v10, v5, v6, vcc
; %bb.558:                              ; %_ZL16__float2bfloat16f.exit.i.i46.2.3.i.i2112
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v7
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr11
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.559:
	v_bfe_u32 v4, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v11, v7, v4, s2
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
; %bb.560:                              ; %Flow4462
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.561:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v5, 0x10000, v7
	v_cmp_eq_u32_sdwa vcc, v7, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v11, v5, v7, vcc
; %bb.562:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.2.3.i.i2115
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v4, 16, v8
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v5, 0xffff0000, v11
	v_or_b32_sdwa v5, v5, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v4, v9, s0, v4
	s_mov_b64 s[2:3], exec
.LBB0_563:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v74, s[4:7], 0 offen
                                        ; implicit-def: $vgpr4_vgpr5
                                        ; implicit-def: $vgpr74
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_563
; %bb.564:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v0
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.565:
	v_bfe_u32 v4, v0, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v0, v4, s2
; %bb.566:                              ; %Flow4461
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.567:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v5, 0x10000, v0
	v_cmp_eq_u32_sdwa vcc, v0, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v4, v5, v0, vcc
; %bb.568:                              ; %_ZL16__float2bfloat16f.exit.i.i.3.3.i.i2128
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.569:
	v_bfe_u32 v0, v1, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v5, v1, v0, s2
; %bb.570:                              ; %Flow4460
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.571:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v5, 0x10000, v1
	v_cmp_eq_u32_sdwa vcc, v1, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v1, vcc
; %bb.572:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.3.3.i.i2131
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v2
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.573:
	v_bfe_u32 v0, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v6, v2, v0, s2
; %bb.574:                              ; %Flow4459
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.575:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v2
	v_cmp_eq_u32_sdwa vcc, v2, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v6, v1, v2, vcc
; %bb.576:                              ; %_ZL16__float2bfloat16f.exit.i.i46.3.3.i.i2136
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v3
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr7
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.577:
	v_bfe_u32 v0, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v7, v3, v0, s2
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
; %bb.578:                              ; %Flow
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.579:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v3
	v_cmp_eq_u32_sdwa vcc, v3, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v7, v1, v3, vcc
; %bb.580:                              ; %_ZL16store_transposedITkN7kittens5ducks2rt3allENS0_2rtIfLi64ELi64ENS1_9rt_layout3colEEETkNS1_2gl3allENS0_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS6_EEEvRKT0_RKT_RKT1_.exit2213
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 16, v4
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v1, 0xffff0000, v7
	v_or_b32_sdwa v1, v1, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v0, v5, s0, v0
	s_mov_b64 s[0:1], exec
.LBB0_581:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v60
	v_readfirstlane_b32 s5, v61
	v_readfirstlane_b32 s6, v62
	v_readfirstlane_b32 s7, v63
	v_cmp_eq_u64_e32 vcc, s[4:5], v[60:61]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[62:63]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v66, s[4:7], 0 offen
                                        ; implicit-def: $vgpr60_vgpr61_vgpr62_vgpr63
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr66
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_581
; %bb.582:
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
		.amdhsa_next_free_vgpr 250
		.amdhsa_next_free_sgpr 29
		.amdhsa_accum_offset 252
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
; codeLenInByte = 21088
; NumSgprs: 35
; NumVgprs: 250
; NumAgprs: 0
; TotalNumVgprs: 250
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 35
; NumVGPRsForWavesPerEU: 250
; AccumOffset: 252
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 62
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_3c031c907996160d,@object ; @__hip_cuid_3c031c907996160d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_3c031c907996160d
__hip_cuid_3c031c907996160d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_3c031c907996160d, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_3c031c907996160d
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
    .sgpr_count:     35
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     250
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
