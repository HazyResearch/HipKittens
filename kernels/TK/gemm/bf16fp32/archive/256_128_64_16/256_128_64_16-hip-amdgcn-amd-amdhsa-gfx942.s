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
	s_cselect_b32 s22, 0, 0
	s_and_b32 s8, s22, -16
	s_load_dwordx2 s[18:19], s[0:1], 0x0
	s_load_dwordx2 s[14:15], s[0:1], 0x20
	s_load_dwordx2 s[4:5], s[0:1], 0x30
	s_load_dwordx2 s[6:7], s[0:1], 0x50
	s_add_i32 s23, s8, 16
	s_load_dwordx2 s[8:9], s[0:1], 0x60
	s_load_dwordx2 s[10:11], s[0:1], 0x80
	s_load_dwordx2 s[20:21], s[0:1], 0x90
	s_mov_b32 s17, 0
	s_and_b32 s16, s22, 15
	s_cmp_eq_u64 s[16:17], 0
	s_cselect_b32 s23, s22, s23
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s20, s3
	s_add_i32 s0, s0, s2
	s_ashr_i32 s2, s0, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s0, s2
	s_mul_i32 s1, s21, s20
	s_ashr_i32 s3, s2, 3
	s_and_b32 s2, s2, -8
	s_sub_i32 s0, s0, s2
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s1, s1, s2
	s_ashr_i32 s1, s1, 3
	s_mul_i32 s0, s0, s1
	s_add_i32 s0, s0, s3
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 23
	s_add_i32 s1, s0, s1
	s_ashr_i32 s2, s1, 9
	s_lshl_b32 s2, s2, 4
	s_sub_i32 s3, 64, s2
	s_min_i32 s3, s3, 16
	s_abs_i32 s20, s3
	v_cvt_f32_u32_e32 v1, s20
	s_sub_i32 s22, 0, s20
	s_add_i32 s24, s23, 0x4000
	s_and_b32 s1, s1, 0xfffffe00
	v_rcp_iflag_f32_e32 v1, v1
	s_and_b32 s21, s24, -16
	s_sub_i32 s0, s0, s1
	s_add_i32 s25, s21, 16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_abs_i32 s21, s0
	s_xor_b32 s1, s0, s3
	s_and_b32 s16, s24, 15
	v_readfirstlane_b32 s26, v1
	s_mul_i32 s22, s22, s26
	s_mul_hi_u32 s22, s26, s22
	s_add_i32 s26, s26, s22
	s_mul_hi_u32 s22, s21, s26
	s_mul_i32 s26, s22, s20
	s_sub_i32 s21, s21, s26
	s_ashr_i32 s1, s1, 31
	s_add_i32 s26, s22, 1
	s_sub_i32 s27, s21, s20
	s_cmp_ge_u32 s21, s20
	s_cselect_b32 s22, s26, s22
	s_cselect_b32 s21, s27, s21
	s_add_i32 s26, s22, 1
	s_cmp_ge_u32 s21, s20
	s_cselect_b32 s20, s26, s22
	s_xor_b32 s20, s20, s1
	s_sub_i32 s20, s20, s1
	s_mul_i32 s1, s20, s3
	s_sub_i32 s0, s0, s1
	s_add_i32 s2, s2, s0
	s_lshl_b32 s22, s2, 7
	s_ashr_i32 s21, s22, 31
	s_mul_i32 s0, s14, s21
	s_mul_hi_u32 s1, s14, s22
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s15, s22
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s14, s22
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s18, s0
	s_addc_u32 s1, s19, s1
	s_lshl_b32 s2, s20, 8
	s_ashr_i32 s3, s2, 31
	s_mul_i32 s3, s6, s3
	s_mul_hi_u32 s15, s6, s2
	s_add_i32 s3, s15, s3
	s_mul_i32 s7, s7, s2
	s_add_i32 s3, s3, s7
	s_mul_i32 s2, s6, s2
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v2, 3, v0
	s_add_u32 s4, s4, s2
	v_and_b32_e32 v28, 56, v2
	v_lshrrev_b32_e32 v15, 3, v0
	s_addc_u32 s5, s5, s3
	v_mad_u64_u32 v[2:3], s[2:3], v15, s14, v[28:29]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[0:1]
	v_or_b32_e32 v20, 64, v15
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[2:3], v20, s14, v[28:29]
	v_lshlrev_b32_e32 v14, 4, v0
	v_lshlrev_b32_e32 v32, 1, v28
	v_ashrrev_i32_e32 v5, 31, v4
	v_and_b32_e32 v3, 0x1f80, v14
	v_add_u32_e32 v16, s23, v32
	v_lshl_add_u64 v[10:11], v[4:5], 1, s[0:1]
	v_add_u32_e32 v5, v16, v3
	v_lshrrev_b32_e32 v18, 4, v5
	v_or_b32_e32 v33, 8, v32
	v_and_b32_e32 v18, 0x78, v18
	v_add_u32_e32 v17, s23, v33
	v_xor_b32_e32 v5, v18, v5
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v5, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v5, v17, v3
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v5, v6, v5
	;;#ASMSTART
	ds_write_b64 v5, v[8:9]

	;;#ASMEND
	v_or_b32_e32 v5, 0x2000, v3
	v_add_u32_e32 v6, v16, v5
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v6, v17, v5
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[12:13]

	;;#ASMEND
	v_mad_u64_u32 v[6:7], s[2:3], v15, s6, v[28:29]
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[8:9], v[6:7], 1, s[4:5]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[8:9], off

	;;#ASMEND
	v_mad_u64_u32 v[8:9], s[2:3], v20, s6, v[28:29]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[10:11], v[8:9], 1, s[4:5]
	s_cmp_eq_u64 s[16:17], 0
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[10:11], off

	;;#ASMEND
	v_lshl_add_u32 v10, s6, 7, v6
	v_ashrrev_i32_e32 v11, 31, v10
	s_cselect_b32 s7, s24, s25
	v_lshl_add_u64 v[12:13], v[10:11], 1, s[4:5]
	v_or_b32_e32 v7, 0xc0, v15
	v_add_u32_e32 v11, s7, v32
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[12:13], off

	;;#ASMEND
	v_mad_u64_u32 v[12:13], s[2:3], v7, s6, v[28:29]
	v_add_u32_e32 v7, v11, v3
	v_lshrrev_b32_e32 v9, 4, v7
	v_ashrrev_i32_e32 v13, 31, v12
	v_and_b32_e32 v9, 0x78, v9
	v_lshl_add_u64 v[28:29], v[12:13], 1, s[4:5]
	v_add_u32_e32 v13, s7, v33
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v7, v[16:17]

	;;#ASMEND
	v_add_u32_e32 v7, v13, v3
	v_lshrrev_b32_e32 v9, 4, v7
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	ds_write_b64 v7, v[18:19]

	;;#ASMEND
	v_add_u32_e32 v7, v11, v5
	v_lshrrev_b32_e32 v9, 4, v7
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	ds_write_b64 v7, v[20:21]

	;;#ASMEND
	v_add_u32_e32 v7, v13, v5
	v_lshrrev_b32_e32 v9, 4, v7
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	ds_write_b64 v7, v[22:23]

	;;#ASMEND
	v_or_b32_e32 v7, 0x4000, v3
	v_add_u32_e32 v9, v11, v7
	v_lshrrev_b32_e32 v15, 4, v9
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v9, v15, v9
	;;#ASMSTART
	ds_write_b64 v9, v[24:25]

	;;#ASMEND
	v_add_u32_e32 v9, v13, v7
	v_lshrrev_b32_e32 v15, 4, v9
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v9, v15, v9
	;;#ASMSTART
	ds_write_b64 v9, v[26:27]

	;;#ASMEND
	v_or_b32_e32 v9, 0x6000, v3
	v_add_u32_e32 v11, v11, v9
	v_lshrrev_b32_e32 v15, 4, v11
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v11, v15, v11
	;;#ASMSTART
	ds_write_b64 v11, v[28:29]

	;;#ASMEND
	v_add_u32_e32 v11, v13, v9
	v_lshrrev_b32_e32 v13, 4, v11
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v11, v13, v11
	;;#ASMSTART
	ds_write_b64 v11, v[30:31]

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
	v_bfe_u32 v11, v0, 6, 2
	v_lshlrev_b32_e32 v123, 5, v11
	v_and_b32_e32 v122, 15, v0
	v_lshrrev_b32_e32 v11, 2, v0
	v_and_b32_e32 v136, 12, v11
	v_or_b32_e32 v11, v123, v122
	v_lshlrev_b32_e32 v13, 1, v136
	v_lshl_add_u32 v11, v11, 7, s7
	v_add_u32_e32 v15, v11, v13
	v_lshrrev_b32_e32 v16, 4, v15
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v137, v16, v15
	v_or_b32_e32 v15, 16, v122
	v_or_b32_e32 v16, v123, v15
	v_lshl_add_u32 v16, v16, 7, s7
	v_add_u32_e32 v17, v16, v13
	v_lshrrev_b32_e32 v18, 4, v17
	v_and_b32_e32 v18, 0x78, v18
	v_xor_b32_e32 v138, v18, v17
	v_or_b32_e32 v17, 0x80, v123
	v_or_b32_e32 v18, v17, v122
	v_lshl_add_u32 v18, v18, 7, s7
	v_add_u32_e32 v19, v18, v13
	v_lshrrev_b32_e32 v20, 4, v19
	v_or_b32_e32 v17, v17, v15
	v_and_b32_e32 v20, 0x78, v20
	v_lshl_add_u32 v17, v17, 7, s7
	v_xor_b32_e32 v139, v20, v19
	v_add_u32_e32 v19, v17, v13
	v_lshrrev_b32_e32 v20, 4, v19
	v_and_b32_e32 v20, 0x78, v20
	v_xor_b32_e32 v140, v20, v19
	v_lshlrev_b32_e32 v19, 5, v1
	v_or_b32_e32 v20, v19, v122
	v_lshl_add_u32 v20, v20, 7, s23
	v_add_u32_e32 v21, v20, v13
	v_lshrrev_b32_e32 v22, 4, v21
	v_and_b32_e32 v22, 0x78, v22
	v_xor_b32_e32 v141, v22, v21
	v_or_b32_e32 v21, v15, v19
	v_lshl_add_u32 v21, v21, 7, s23
	v_add_u32_e32 v22, v21, v13
	v_lshrrev_b32_e32 v23, 4, v22
	v_and_b32_e32 v23, 0x78, v23
	v_or_b32_e32 v19, 64, v19
	v_xor_b32_e32 v142, v23, v22
	v_or_b32_e32 v22, v19, v122
	v_lshl_add_u32 v22, v22, 7, s23
	v_add_u32_e32 v23, v22, v13
	v_or_b32_e32 v15, v19, v15
	v_lshrrev_b32_e32 v24, 4, v23
	v_lshl_add_u32 v15, v15, 7, s23
	v_and_b32_e32 v24, 0x78, v24
	v_add_u32_e32 v19, v15, v13
	v_xor_b32_e32 v143, v24, v23
	v_lshrrev_b32_e32 v23, 4, v19
	v_lshlrev_b32_e32 v146, 1, v2
	v_or_b32_e32 v2, 32, v13
	v_and_b32_e32 v23, 0x78, v23
	v_lshlrev_b32_e32 v145, 1, v4
	v_add_u32_e32 v4, v11, v2
	v_xor_b32_e32 v144, v23, v19
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v147, v19, v4
	v_add_u32_e32 v4, v16, v2
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v148, v19, v4
	v_add_u32_e32 v4, v18, v2
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v149, v19, v4
	v_add_u32_e32 v4, v17, v2
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v150, v19, v4
	v_add_u32_e32 v4, v20, v2
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v151, v19, v4
	v_add_u32_e32 v4, v21, v2
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v152, v19, v4
	v_add_u32_e32 v4, v22, v2
	v_lshrrev_b32_e32 v19, 4, v4
	v_and_b32_e32 v19, 0x78, v19
	v_add_u32_e32 v2, v15, v2
	v_xor_b32_e32 v153, v19, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v154, v4, v2
	v_or_b32_e32 v2, 64, v13
	v_add_u32_e32 v4, v11, v2
	v_lshlrev_b32_e32 v155, 1, v6
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v159, v6, v4
	v_add_u32_e32 v4, v16, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v160, v6, v4
	v_add_u32_e32 v4, v18, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v161, v6, v4
	v_add_u32_e32 v4, v17, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v162, v6, v4
	v_add_u32_e32 v4, v20, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v163, v6, v4
	v_add_u32_e32 v4, v21, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v164, v6, v4
	v_add_u32_e32 v4, v22, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_add_u32_e32 v2, v15, v2
	v_xor_b32_e32 v165, v6, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v166, v4, v2
	v_or_b32_e32 v2, 0x60, v13
	v_add_u32_e32 v4, v11, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v167, v6, v4
	v_add_u32_e32 v4, v16, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v168, v6, v4
	v_add_u32_e32 v4, v18, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v169, v6, v4
	v_add_u32_e32 v4, v17, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v170, v6, v4
	v_add_u32_e32 v4, v20, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v171, v6, v4
	v_add_u32_e32 v4, v21, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v172, v6, v4
	v_add_u32_e32 v4, v22, v2
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_add_u32_e32 v2, v15, v2
	v_xor_b32_e32 v173, v6, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v174, v4, v2
	v_and_b32_e32 v2, 0x70, v14
	v_add_u32_e32 v4, s23, v2
	v_lshlrev_b32_e32 v157, 1, v10
	v_add_u32_e32 v10, v4, v3
	v_or_b32_e32 v6, 8, v2
	v_lshrrev_b32_e32 v11, 4, v10
	v_lshlrev_b32_e32 v156, 1, v8
	v_add_u32_e32 v8, s23, v6
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v175, v11, v10
	v_add_u32_e32 v10, v8, v3
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v4, v4, v5
	v_xor_b32_e32 v176, v11, v10
	v_lshrrev_b32_e32 v10, 4, v4
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v177, v10, v4
	v_add_u32_e32 v4, v8, v5
	v_lshrrev_b32_e32 v8, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v2, s7, v2
	v_xor_b32_e32 v178, v8, v4
	v_add_u32_e32 v4, s7, v6
	v_add_u32_e32 v6, v2, v3
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v3, v4, v3
	v_xor_b32_e32 v179, v8, v6
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v180, v6, v3
	v_add_u32_e32 v3, v2, v5
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v181, v6, v3
	v_add_u32_e32 v3, v4, v5
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v182, v5, v3
	v_add_u32_e32 v3, v2, v7
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v183, v5, v3
	v_add_u32_e32 v3, v4, v7
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v2, v2, v9
	v_xor_b32_e32 v184, v5, v3
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	s_lshl_b32 s2, s14, 8
	s_lshl_b32 s6, s6, 9
	v_xor_b32_e32 v185, v3, v2
	v_add_u32_e32 v2, v4, v9
	v_lshrrev_b32_e32 v3, 4, v2
	s_add_u32 s14, s4, 0x80
	v_and_b32_e32 v3, 0x78, v3
	s_addc_u32 s15, s5, 0
	v_xor_b32_e32 v186, v3, v2
	s_add_u32 s16, s0, 0x80
	v_mov_b32_e32 v2, 0
	v_mov_b64_e32 v[28:29], 0
	v_lshlrev_b32_e32 v158, 1, v12
	s_addc_u32 s17, s1, 0
	s_mov_b32 s3, 0x110000
	s_mov_b32 s18, 0xffff
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
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
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
	v_mov_b64_e32 v[16:17], v[28:29]
	v_mov_b64_e32 v[48:49], v[28:29]
	v_mov_b64_e32 v[40:41], v[28:29]
	v_mov_b64_e32 v[36:37], v[28:29]
	v_mov_b64_e32 v[24:25], v[28:29]
	s_branch .LBB0_4
.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v60, s18, v60, v60
	v_bfi_b32 v58, s18, v58, v58
	v_bfi_b32 v130, s18, v130, v130
	v_bfi_b32 v128, s18, v128, v128
	v_bfi_b32 v126, s18, v126, v126
	v_bfi_b32 v124, s18, v124, v124
	v_bfi_b32 v134, s18, v134, v134
	v_bfi_b32 v132, s18, v132, v132
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[86:89], v[60:61], v[130:131], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[60:61], v[128:129], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[58:59], v[130:131], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[58:59], v[128:129], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[60:61], v[126:127], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[60:61], v[124:125], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[58:59], v[126:127], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[58:59], v[124:125], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[134:135], v[130:131], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[134:135], v[128:129], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[132:133], v[130:131], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[132:133], v[128:129], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[134:135], v[126:127], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[134:135], v[124:125], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[132:133], v[126:127], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[132:133], v[124:125], v[118:121]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s12, 0x80
	s_addc_u32 s13, s13, 0
	s_cmpk_eq_i32 s12, 0x4000
	s_cbranch_scc1 .LBB0_10
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	ds_read_b64 v[96:97], v137

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[94:95], v138

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[92:93], v139

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[90:91], v140

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[104:105], v141

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[102:103], v142

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[100:101], v143

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[98:99], v144

	;;#ASMEND
	s_cmpk_lg_i32 s12, 0x3f80
	s_cselect_b64 s[4:5], -1, 0
	s_cmpk_eq_i32 s12, 0x3f80
	; sched_barrier mask(0x00000000)
	s_cbranch_scc1 .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s0, s16, s12
	s_addc_u32 s1, s17, s13
	buffer_load_dwordx4 v[14:17], v146, s[0:3], 0 offen
	buffer_load_dwordx4 v[26:29], v145, s[0:3], 0 offen
.LBB0_6:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v96, s18, v96, v96
	v_bfi_b32 v94, s18, v94, v94
	v_bfi_b32 v92, s18, v92, v92
	v_bfi_b32 v90, s18, v90, v90
	v_bfi_b32 v104, s18, v104, v104
	v_bfi_b32 v102, s18, v102, v102
	v_bfi_b32 v100, s18, v100, v100
	v_bfi_b32 v98, s18, v98, v98
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[86:89], v[104:105], v[96:97], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[104:105], v[94:95], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[102:103], v[96:97], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[102:103], v[94:95], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[104:105], v[92:93], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[104:105], v[90:91], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[102:103], v[92:93], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[102:103], v[90:91], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[100:101], v[96:97], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[100:101], v[94:95], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[98:99], v[96:97], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[98:99], v[94:95], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[100:101], v[92:93], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[100:101], v[90:91], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[98:99], v[92:93], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[98:99], v[90:91], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[96:97], v147

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[94:95], v148

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[92:93], v149

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[90:91], v150

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[104:105], v151

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[102:103], v152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[100:101], v153

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[98:99], v154

	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cndmask_b32_e64 v106, 0, 1, s[4:5]
	v_cmp_ne_u32_e64 s[0:1], 1, v106
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s4, s14, s12
	s_addc_u32 s5, s15, s13
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v[22:25], v155, s[4:7], 0 offen
	buffer_load_dwordx4 v[34:37], v156, s[4:7], 0 offen
	buffer_load_dwordx4 v[38:41], v157, s[4:7], 0 offen
	buffer_load_dwordx4 v[46:49], v158, s[4:7], 0 offen
.LBB0_8:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v96, s18, v96, v96
	v_bfi_b32 v94, s18, v94, v94
	v_bfi_b32 v92, s18, v92, v92
	v_bfi_b32 v90, s18, v90, v90
	v_bfi_b32 v104, s18, v104, v104
	v_bfi_b32 v102, s18, v102, v102
	v_bfi_b32 v100, s18, v100, v100
	v_bfi_b32 v98, s18, v98, v98
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[86:89], v[104:105], v[96:97], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[104:105], v[94:95], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[102:103], v[96:97], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[102:103], v[94:95], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[104:105], v[92:93], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[104:105], v[90:91], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[102:103], v[92:93], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[102:103], v[90:91], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[100:101], v[96:97], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[100:101], v[94:95], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[98:99], v[96:97], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[98:99], v[94:95], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[188:191], v[100:101], v[92:93], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[192:195], v[100:101], v[90:91], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[196:199], v[98:99], v[92:93], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[200:203], v[98:99], v[90:91], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[98:99], v159

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v160

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v161

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v162

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[42:43], v163

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[54:55], v164

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v165

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v166

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v167

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[128:129], v168

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v169

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v170

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[60:61], v171

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[58:59], v172

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v173

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v174

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v98, s18, v98, v98
	v_bfi_b32 v42, s18, v42, v42
	v_bfi_b32 v54, s18, v54, v54
	v_bfi_b32 v204, s18, v204, v204
	v_bfi_b32 v206, s18, v206, v206
	v_bfi_b32 v208, s18, v208, v208
	v_bfi_b32 v210, s18, v210, v210
	v_bfi_b32 v212, s18, v212, v212
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[2:5], v[42:43], v[98:99], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[42:43], v[204:205], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[54:55], v[98:99], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[54:55], v[204:205], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[42:43], v[206:207], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[42:43], v[208:209], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[54:55], v[206:207], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[54:55], v[208:209], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[210:211], v[98:99], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[210:211], v[204:205], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[212:213], v[98:99], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[212:213], v[204:205], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[210:211], v[206:207], v[188:191]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[210:211], v[208:209], v[192:195]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[212:213], v[206:207], v[196:199]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[212:213], v[208:209], v[200:203]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_3
; %bb.9:                                ;   in Loop: Header=BB0_4 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v175, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v176, v[16:17]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v177, v[26:27]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v178, v[28:29]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v179, v[22:23]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v180, v[24:25]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v181, v[34:35]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v182, v[36:37]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v183, v[38:39]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v184, v[40:41]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v185, v[46:47]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v186, v[48:49]

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
	v_mul_lo_u32 v0, v136, s10
	s_waitcnt vmcnt(1)
	v_add_u32_e32 v14, s10, v0
	v_lshl_or_b32 v106, v1, 5, s22
	v_add_u32_e32 v16, s10, v14
	v_lshl_or_b32 v96, s20, 8, v123
	s_mul_i32 s4, s10, s21
	v_mul_lo_u32 v1, s11, v106
	v_mad_u64_u32 v[98:99], s[0:1], s10, v106, 0
	v_add_u32_e32 v22, s10, v16
	v_add_u32_e32 v24, v0, v122
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v26, v14, v122
	v_add3_u32 v99, v99, s4, v1
	v_ashrrev_i32_e32 v97, 31, v96
	v_ashrrev_i32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v27, 31, v26
	v_add_u32_e32 v28, v16, v122
	v_add_u32_e32 v34, v22, v122
	v_lshl_add_u64 v[98:99], v[98:99], 2, s[8:9]
	v_lshlrev_b64 v[96:97], 2, v[96:97]
	v_ashrrev_i32_e32 v29, 31, v28
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[96:97]
	v_lshlrev_b64 v[24:25], 2, v[24:25]
	v_lshlrev_b64 v[26:27], 2, v[26:27]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mov_b32_e32 v123, 0
	v_lshl_add_u64 v[100:101], v[98:99], 0, v[24:25]
	v_lshl_add_u64 v[102:103], v[98:99], 0, v[26:27]
	v_lshlrev_b64 v[28:29], 2, v[28:29]
	v_lshlrev_b64 v[34:35], 2, v[34:35]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[122:123]
	v_ashrrev_i32_e32 v15, 31, v14
	global_store_dword v[100:101], v86, off
	global_store_dword v[102:103], v87, off
	v_lshl_add_u64 v[86:87], v[98:99], 0, v[28:29]
	v_lshl_add_u64 v[104:105], v[98:99], 0, v[34:35]
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[122:123]
	v_ashrrev_i32_e32 v17, 31, v16
	v_mad_u64_u32 v[36:37], s[0:1], s10, 13, v[22:23]
	global_store_dword v[86:87], v88, off
	global_store_dword v[104:105], v89, off
	v_lshl_add_u64 v[88:89], v[98:99], 0, v[0:1]
	v_lshlrev_b64 v[14:15], 2, v[14:15]
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[122:123]
	v_ashrrev_i32_e32 v23, 31, v22
	v_add_u32_e32 v38, s10, v36
	global_store_dword v[88:89], v82, off offset:64
	v_lshl_add_u64 v[88:89], v[98:99], 0, v[14:15]
	v_lshlrev_b64 v[16:17], 2, v[16:17]
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[122:123]
	v_add_u32_e32 v40, s10, v38
	v_add_u32_e32 v48, v36, v122
	v_add_u32_e32 v90, v38, v122
	global_store_dword v[88:89], v83, off offset:64
	v_lshl_add_u64 v[82:83], v[98:99], 0, v[16:17]
	v_lshlrev_b64 v[22:23], 2, v[22:23]
	v_add_u32_e32 v46, s10, v40
	v_ashrrev_i32_e32 v49, 31, v48
	v_ashrrev_i32_e32 v91, 31, v90
	global_store_dword v[82:83], v84, off offset:64
	v_lshl_add_u64 v[82:83], v[98:99], 0, v[22:23]
	v_add_u32_e32 v92, v40, v122
	v_add_u32_e32 v94, v46, v122
	global_store_dword v[82:83], v85, off offset:64
	v_lshlrev_b64 v[48:49], 2, v[48:49]
	v_lshlrev_b64 v[84:85], 2, v[90:91]
	v_ashrrev_i32_e32 v93, 31, v92
	v_ashrrev_i32_e32 v95, 31, v94
	v_lshl_add_u64 v[82:83], v[98:99], 0, v[48:49]
	v_lshl_add_u64 v[88:89], v[98:99], 0, v[84:85]
	v_ashrrev_i32_e32 v37, 31, v36
	global_store_dword v[82:83], v78, off
	global_store_dword v[88:89], v79, off
	v_lshlrev_b64 v[78:79], 2, v[92:93]
	v_lshlrev_b64 v[92:93], 2, v[94:95]
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[122:123]
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[90:91], v[98:99], 0, v[78:79]
	v_lshl_add_u64 v[94:95], v[98:99], 0, v[92:93]
	v_lshlrev_b64 v[36:37], 2, v[36:37]
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[122:123]
	v_ashrrev_i32_e32 v41, 31, v40
	global_store_dword v[90:91], v80, off
	global_store_dword v[94:95], v81, off
	v_lshl_add_u64 v[80:81], v[98:99], 0, v[36:37]
	v_lshlrev_b64 v[38:39], 2, v[38:39]
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[122:123]
	v_ashrrev_i32_e32 v47, 31, v46
	global_store_dword v[80:81], v74, off offset:64
	v_lshl_add_u64 v[80:81], v[98:99], 0, v[38:39]
	v_lshlrev_b64 v[40:41], 2, v[40:41]
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[122:123]
	global_store_dword v[80:81], v75, off offset:64
	v_lshl_add_u64 v[74:75], v[98:99], 0, v[40:41]
	v_lshlrev_b64 v[46:47], 2, v[46:47]
	global_store_dword v[74:75], v76, off offset:64
	v_lshl_add_u64 v[74:75], v[98:99], 0, v[46:47]
	s_mov_b64 s[0:1], 0x200
	global_store_dword v[74:75], v77, off offset:64
	v_lshl_add_u64 v[74:75], v[98:99], 0, s[0:1]
	global_store_dword v[100:101], v70, off offset:512
	global_store_dword v[102:103], v71, off offset:512
	global_store_dword v[86:87], v72, off offset:512
	global_store_dword v[104:105], v73, off offset:512
	v_lshl_add_u64 v[70:71], v[74:75], 0, v[0:1]
	global_store_dword v[70:71], v66, off offset:64
	v_lshl_add_u64 v[70:71], v[74:75], 0, v[14:15]
	global_store_dword v[70:71], v67, off offset:64
	v_lshl_add_u64 v[66:67], v[74:75], 0, v[16:17]
	global_store_dword v[66:67], v68, off offset:64
	v_lshl_add_u64 v[66:67], v[74:75], 0, v[22:23]
	global_store_dword v[66:67], v69, off offset:64
	global_store_dword v[82:83], v62, off offset:512
	global_store_dword v[88:89], v63, off offset:512
	global_store_dword v[90:91], v64, off offset:512
	global_store_dword v[94:95], v65, off offset:512
	v_lshl_add_u64 v[62:63], v[74:75], 0, v[36:37]
	global_store_dword v[62:63], v58, off offset:64
	v_lshl_add_u64 v[62:63], v[74:75], 0, v[38:39]
	global_store_dword v[62:63], v59, off offset:64
	v_lshl_add_u64 v[58:59], v[74:75], 0, v[40:41]
	global_store_dword v[58:59], v60, off offset:64
	v_lshl_add_u64 v[58:59], v[74:75], 0, v[46:47]
	global_store_dword v[58:59], v61, off offset:64
	v_or_b32_e32 v58, 64, v106
	v_mul_lo_u32 v60, s11, v58
	v_mad_u64_u32 v[58:59], s[2:3], s10, v58, 0
	v_add3_u32 v59, v59, s4, v60
	v_lshl_add_u64 v[58:59], v[58:59], 2, s[8:9]
	v_lshl_add_u64 v[58:59], v[58:59], 0, v[96:97]
	v_lshl_add_u64 v[24:25], v[58:59], 0, v[24:25]
	v_lshl_add_u64 v[26:27], v[58:59], 0, v[26:27]
	global_store_dword v[24:25], v54, off
	global_store_dword v[26:27], v55, off
	v_lshl_add_u64 v[28:29], v[58:59], 0, v[28:29]
	v_lshl_add_u64 v[34:35], v[58:59], 0, v[34:35]
	v_lshl_add_u64 v[54:55], v[58:59], 0, v[0:1]
	global_store_dword v[28:29], v56, off
	global_store_dword v[34:35], v57, off
	global_store_dword v[54:55], v50, off offset:64
	v_lshl_add_u64 v[54:55], v[58:59], 0, v[14:15]
	global_store_dword v[54:55], v51, off offset:64
	v_lshl_add_u64 v[50:51], v[58:59], 0, v[16:17]
	global_store_dword v[50:51], v52, off offset:64
	v_lshl_add_u64 v[50:51], v[58:59], 0, v[22:23]
	global_store_dword v[50:51], v53, off offset:64
	v_lshl_add_u64 v[48:49], v[58:59], 0, v[48:49]
	v_lshl_add_u64 v[50:51], v[58:59], 0, v[84:85]
	global_store_dword v[48:49], v42, off
	global_store_dword v[50:51], v43, off
	v_lshl_add_u64 v[42:43], v[58:59], 0, v[78:79]
	v_lshl_add_u64 v[52:53], v[58:59], 0, v[92:93]
	global_store_dword v[42:43], v44, off
	global_store_dword v[52:53], v45, off
	v_lshl_add_u64 v[44:45], v[58:59], 0, v[36:37]
	global_store_dword v[44:45], v30, off offset:64
	v_lshl_add_u64 v[44:45], v[58:59], 0, v[38:39]
	global_store_dword v[44:45], v31, off offset:64
	v_lshl_add_u64 v[30:31], v[58:59], 0, v[40:41]
	global_store_dword v[30:31], v32, off offset:64
	v_lshl_add_u64 v[30:31], v[58:59], 0, v[46:47]
	global_store_dword v[30:31], v33, off offset:64
	v_lshl_add_u64 v[30:31], v[58:59], 0, s[0:1]
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[0:1]
	global_store_dword v[24:25], v18, off offset:512
	global_store_dword v[26:27], v19, off offset:512
	global_store_dword v[28:29], v20, off offset:512
	global_store_dword v[34:35], v21, off offset:512
	global_store_dword v[0:1], v10, off offset:64
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[14:15]
	global_store_dword v[0:1], v11, off offset:64
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[16:17]
	global_store_dword v[0:1], v12, off offset:64
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[22:23]
	global_store_dword v[0:1], v13, off offset:64
	global_store_dword v[48:49], v6, off offset:512
	global_store_dword v[50:51], v7, off offset:512
	global_store_dword v[42:43], v8, off offset:512
	global_store_dword v[52:53], v9, off offset:512
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[36:37]
	global_store_dword v[0:1], v2, off offset:64
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[38:39]
	global_store_dword v[0:1], v3, off offset:64
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[40:41]
	global_store_dword v[0:1], v4, off offset:64
	v_lshl_add_u64 v[0:1], v[30:31], 0, v[46:47]
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
		.amdhsa_next_free_vgpr 214
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 216
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
; codeLenInByte = 5292
; NumSgprs: 34
; NumVgprs: 214
; NumAgprs: 0
; TotalNumVgprs: 214
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 26
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 214
; AccumOffset: 216
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 53
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_8818a49f16d72077,@object ; @__hip_cuid_8818a49f16d72077
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_8818a49f16d72077
__hip_cuid_8818a49f16d72077:
	.byte	0                               ; 0x0
	.size	__hip_cuid_8818a49f16d72077, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_8818a49f16d72077
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
    .vgpr_count:     214
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
