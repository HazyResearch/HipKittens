	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:
	s_load_dwordx2 s[12:13], s[0:1], 0x0
	s_load_dword s6, s[0:1], 0x20
	s_load_dwordx2 s[14:15], s[0:1], 0x30
	s_load_dword s5, s[0:1], 0x50
	s_load_dwordx2 s[8:9], s[0:1], 0x60
	s_load_dwordx2 s[10:11], s[0:1], 0x80
	s_load_dwordx2 s[20:21], s[0:1], 0x90
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s4, 0, 0
	s_mov_b32 s16, 0
	s_and_b32 s7, s4, -16
	s_and_b32 s18, s4, 15
	s_mov_b32 s19, s16
	s_add_i32 s7, s7, 16
	s_cmp_eq_u64 s[18:19], 0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s20, s3
	s_cselect_b32 s4, s4, s7
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
	s_lshr_b32 s1, s1, 25
	s_add_i32 s1, s0, s1
	s_ashr_i32 s2, s1, 7
	s_lshl_b32 s11, s2, 2
	s_sub_i32 s2, 32, s11
	s_min_i32 s3, s2, 4
	s_abs_i32 s17, s3
	v_cvt_f32_u32_e32 v1, s17
	s_sub_i32 s19, 0, s17
	s_and_b32 s1, s1, 0xffffff80
	s_sub_i32 s0, s0, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s18, s0
	s_add_i32 s7, s4, 0x8000
	s_and_b32 s2, s7, -16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s1, s0, s3
	s_and_b32 s22, s7, 15
	s_add_i32 s2, s2, 16
	v_readfirstlane_b32 s20, v1
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s18, s20
	s_mul_i32 s20, s19, s17
	s_sub_i32 s18, s18, s20
	s_ashr_i32 s1, s1, 31
	s_add_i32 s20, s19, 1
	s_sub_i32 s21, s18, s17
	s_cmp_ge_u32 s18, s17
	s_cselect_b32 s19, s20, s19
	s_cselect_b32 s18, s21, s18
	s_add_i32 s20, s19, 1
	s_cmp_ge_u32 s18, s17
	s_cselect_b32 s17, s20, s19
	s_xor_b32 s17, s17, s1
	s_sub_i32 s17, s17, s1
	s_mul_i32 s1, s17, s3
	s_sub_i32 s0, s0, s1
	s_add_i32 s11, s11, s0
	s_mul_i32 s0, s11, s6
	s_lshl_b32 s18, s0, 8
	s_ashr_i32 s19, s18, 31
	s_lshl_b64 s[0:1], s[18:19], 1
	s_add_u32 s24, s12, s0
	s_addc_u32 s25, s13, s1
	s_lshl_b32 s19, s17, 8
	s_mul_i32 s20, s19, s5
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[0:1], s[20:21], 1
	s_add_u32 s0, s14, s0
	v_lshlrev_b32_e32 v2, 3, v0
	s_mov_b32 s23, s16
	s_addc_u32 s1, s15, s1
	v_and_b32_e32 v32, 56, v2
	v_lshrrev_b32_e32 v28, 3, v0
	s_cmp_eq_u64 s[22:23], 0
	v_mad_u64_u32 v[2:3], s[22:23], v28, s6, v[32:33]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[24:25]
	v_or_b32_e32 v29, 64, v28
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[22:23], v29, s6, v[32:33]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[6:7], v[4:5], 1, s[24:25]
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[6:7], off

	;;#ASMEND
	v_lshl_add_u32 v6, s6, 7, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[8:9], v[6:7], 1, s[24:25]
	v_or_b32_e32 v33, 0xc0, v28
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[8:9], off

	;;#ASMEND
	v_mad_u64_u32 v[8:9], s[22:23], v33, s6, v[32:33]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[18:19], v[8:9], 1, s[24:25]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[18:19], off

	;;#ASMEND
	v_lshlrev_b32_e32 v18, 4, v0
	v_lshlrev_b32_e32 v19, 1, v32
	v_and_b32_e32 v3, 0x1f80, v18
	v_add_u32_e32 v30, s4, v19
	v_add_u32_e32 v5, v30, v3
	v_lshrrev_b32_e32 v7, 4, v5
	v_or_b32_e32 v36, 8, v19
	v_and_b32_e32 v7, 0x78, v7
	v_add_u32_e32 v31, s4, v36
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
	v_mad_u64_u32 v[10:11], s[22:23], v28, s5, v[32:33]
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[12:13], v[10:11], 1, s[0:1]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[12:13], off

	;;#ASMEND
	v_mad_u64_u32 v[12:13], s[22:23], v29, s5, v[32:33]
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[14:15], v[12:13], 1, s[0:1]
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[14:15], off

	;;#ASMEND
	v_lshl_add_u32 v14, s5, 7, v10
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[16:17], v[14:15], 1, s[0:1]
	s_cselect_b32 s3, s7, s2
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[16:17], off

	;;#ASMEND
	v_mad_u64_u32 v[16:17], s[22:23], v33, s5, v[32:33]
	v_add_u32_e32 v11, s3, v19
	v_ashrrev_i32_e32 v17, 31, v16
	v_add_u32_e32 v15, v11, v3
	v_lshl_add_u64 v[32:33], v[16:17], 1, s[0:1]
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
	s_mov_b32 s21, 64
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
	v_lshlrev_b32_e32 v146, 1, v2
	v_and_b32_e32 v134, 15, v0
	v_lshrrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v135, 12, v2
	v_lshlrev_b32_e32 v2, 7, v134
	v_lshlrev_b32_e32 v147, 1, v4
	v_lshl_or_b32 v4, v1, 13, v2
	v_lshlrev_b32_e32 v148, 1, v6
	v_lshlrev_b32_e32 v6, 1, v135
	v_add_u32_e32 v4, s4, v4
	v_lshlrev_b32_e32 v151, 1, v8
	v_add_u32_e32 v8, v4, v6
	v_lshrrev_b32_e32 v11, 4, v8
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v149, v11, v8
	v_add_u32_e32 v8, 0x4000, v4
	v_bfe_u32 v136, v0, 6, 2
	v_add_u32_e32 v11, v8, v6
	v_lshrrev_b32_e32 v13, 4, v11
	v_lshl_or_b32 v2, v136, 13, v2
	v_and_b32_e32 v13, 0x78, v13
	v_add_u32_e32 v2, s3, v2
	v_xor_b32_e32 v150, v13, v11
	v_add_u32_e32 v11, v2, v6
	v_lshrrev_b32_e32 v13, 4, v11
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v152, v13, v11
	v_or_b32_e32 v11, 32, v6
	v_add_u32_e32 v13, v2, v11
	v_lshrrev_b32_e32 v15, 4, v13
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v145, v15, v13
	v_add_u32_e32 v13, v4, v11
	v_lshrrev_b32_e32 v15, 4, v13
	v_and_b32_e32 v15, 0x78, v15
	v_add_u32_e32 v11, v8, v11
	v_xor_b32_e32 v144, v15, v13
	v_lshrrev_b32_e32 v13, 4, v11
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v143, v13, v11
	v_or_b32_e32 v11, 64, v6
	v_add_u32_e32 v13, v2, v11
	v_lshrrev_b32_e32 v15, 4, v13
	v_and_b32_e32 v15, 0x78, v15
	v_lshlrev_b32_e32 v153, 1, v10
	v_add_u32_e32 v10, v8, v11
	v_xor_b32_e32 v138, v15, v13
	v_add_u32_e32 v13, v4, v11
	v_lshrrev_b32_e32 v11, 4, v10
	v_or_b32_e32 v6, 0x60, v6
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v2, v2, v6
	v_xor_b32_e32 v141, v11, v10
	v_lshrrev_b32_e32 v10, 4, v2
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v142, v10, v2
	v_add_u32_e32 v2, v4, v6
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v140, v4, v2
	v_add_u32_e32 v2, v8, v6
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v139, v4, v2
	v_and_b32_e32 v2, 0x70, v18
	v_add_u32_e32 v4, s4, v2
	v_add_u32_e32 v10, v4, v3
	v_or_b32_e32 v6, 8, v2
	v_lshrrev_b32_e32 v11, 4, v10
	v_add_u32_e32 v8, s4, v6
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v157, v11, v10
	v_add_u32_e32 v10, v8, v3
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v158, v11, v10
	v_add_u32_e32 v10, v4, v5
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v159, v11, v10
	v_add_u32_e32 v10, v8, v5
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v160, v11, v10
	v_add_u32_e32 v10, v4, v7
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v161, v11, v10
	v_add_u32_e32 v10, v8, v7
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_add_u32_e32 v4, v4, v9
	v_xor_b32_e32 v162, v11, v10
	v_lshrrev_b32_e32 v10, 4, v4
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v163, v10, v4
	v_add_u32_e32 v4, v8, v9
	v_lshrrev_b32_e32 v8, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v2, s3, v2
	v_xor_b32_e32 v164, v8, v4
	v_add_u32_e32 v4, s3, v6
	v_add_u32_e32 v6, v2, v3
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v3, v4, v3
	v_xor_b32_e32 v165, v8, v6
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v166, v6, v3
	v_add_u32_e32 v3, v2, v5
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v167, v6, v3
	v_add_u32_e32 v3, v4, v5
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v168, v5, v3
	v_add_u32_e32 v3, v2, v7
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v169, v5, v3
	v_add_u32_e32 v3, v4, v7
	s_lshl_b32 s0, s6, 1
	v_lshrrev_b32_e32 v5, 4, v3
	s_lshl_b32 s2, s6, 9
	s_and_b32 s0, s0, 0x3ffe
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v2, v2, v9
	s_cmp_eq_u32 s0, 0
	v_xor_b32_e32 v170, v5, v3
	v_lshrrev_b32_e32 v3, 4, v2
	s_cselect_b64 s[22:23], -1, 0
	s_lshl_b32 s0, s6, 17
	v_and_b32_e32 v3, 0x78, v3
	s_or_b32 s17, s0, -2.0
	s_lshl_b32 s0, s5, 1
	v_xor_b32_e32 v171, v3, v2
	v_add_u32_e32 v2, v4, v9
	s_lshl_b32 s6, s5, 9
	s_and_b32 s0, s0, 0x3ffe
	v_lshrrev_b32_e32 v3, 4, v2
	v_lshrrev_b32_e32 v15, 4, v13
	s_cmp_eq_u32 s0, 0
	v_and_b32_e32 v3, 0x78, v3
	v_and_b32_e32 v15, 0x78, v15
	s_cselect_b64 s[24:25], -1, 0
	s_lshl_b32 s0, s5, 17
	v_xor_b32_e32 v172, v3, v2
	v_mov_b32_e32 v2, 0
	v_xor_b32_e32 v137, v15, v13
	s_or_b32 s27, s0, -2.0
	s_mov_b32 s26, s16
	v_lshlrev_b32_e32 v154, 1, v12
	v_lshlrev_b32_e32 v155, 1, v14
	v_lshlrev_b32_e32 v156, 1, v16
	s_mov_b32 s3, 0x110000
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
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
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
	s_add_i32 s0, s18, s21
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s12, s0
	s_addc_u32 s1, s13, s1
	s_or_b64 s[4:5], s[16:17], s[0:1]
	s_and_b64 s[28:29], s[22:23], exec
	s_cselect_b32 s1, s1, s5
	s_cselect_b32 s0, s0, s4
	buffer_load_dwordx4 v[130:133], v146, s[0:3], 0 offen
	buffer_load_dwordx4 v[174:177], v147, s[0:3], 0 offen
	buffer_load_dwordx4 v[178:181], v148, s[0:3], 0 offen
	buffer_load_dwordx4 v[182:185], v151, s[0:3], 0 offen
	;;#ASMSTART
	ds_read_b64 v[186:187], v149 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v149 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[190:191], v149 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v149 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[196:197], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[198:199], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[200:201], v150 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v152 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v152 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v152 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v152 offset:0x1800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[186:187], v[202:203], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[186:187], v[204:205], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[186:187], v[206:207], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[186:187], v[208:209], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[188:189], v[202:203], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[188:189], v[204:205], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[188:189], v[206:207], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[188:189], v[208:209], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[190:191], v[202:203], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[190:191], v[204:205], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[190:191], v[206:207], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[190:191], v[208:209], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[192:193], v[202:203], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[192:193], v[204:205], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[192:193], v[206:207], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[192:193], v[208:209], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[194:195], v[202:203], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[194:195], v[204:205], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[194:195], v[206:207], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[194:195], v[208:209], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[196:197], v[202:203], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[196:197], v[204:205], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[196:197], v[206:207], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[196:197], v[208:209], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[198:199], v[202:203], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[198:199], v[204:205], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[198:199], v[206:207], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[198:199], v[208:209], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[200:201], v[202:203], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[200:201], v[204:205], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[200:201], v[206:207], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[200:201], v[208:209], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[186:187], v145 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v145 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[190:191], v145 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v145 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v144 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[196:197], v144 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[198:199], v144 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[200:201], v144 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v143 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v143 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v143 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v143 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v138 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v138 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[214:215], v138 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[216:217], v138 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[218:219], v137 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[220:221], v137 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[222:223], v137 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[224:225], v137 offset:0x1800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[194:195], v[186:187], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[194:195], v[188:189], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[194:195], v[190:191], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[194:195], v[192:193], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[196:197], v[186:187], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[196:197], v[188:189], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[196:197], v[190:191], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[196:197], v[192:193], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[198:199], v[186:187], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[198:199], v[188:189], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[198:199], v[190:191], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[198:199], v[192:193], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[200:201], v[186:187], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[200:201], v[188:189], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[200:201], v[190:191], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[200:201], v[192:193], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[202:203], v[186:187], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[202:203], v[188:189], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[202:203], v[190:191], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[202:203], v[192:193], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[204:205], v[186:187], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[204:205], v[188:189], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[204:205], v[190:191], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[204:205], v[192:193], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[206:207], v[186:187], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[206:207], v[188:189], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[206:207], v[190:191], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[206:207], v[192:193], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[208:209], v[186:187], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[208:209], v[188:189], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[208:209], v[190:191], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[208:209], v[192:193], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s0, s20, s21
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_or_b64 s[4:5], s[26:27], s[0:1]
	s_and_b64 s[28:29], s[24:25], exec
	s_cselect_b32 s5, s1, s5
	s_cselect_b32 s4, s0, s4
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v[186:189], v153, s[4:7], 0 offen
	buffer_load_dwordx4 v[190:193], v154, s[4:7], 0 offen
	buffer_load_dwordx4 v[194:197], v155, s[4:7], 0 offen
	buffer_load_dwordx4 v[198:201], v156, s[4:7], 0 offen
	;;#ASMSTART
	ds_read_b64 v[202:203], v141 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v141 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v141 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v141 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[226:227], v142 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[228:229], v142 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[230:231], v142 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[232:233], v142 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[234:235], v140 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[236:237], v140 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[238:239], v140 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[240:241], v140 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[242:243], v139 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[244:245], v139 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[246:247], v139 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[248:249], v139 offset:0x1800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[218:219], v[210:211], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[218:219], v[212:213], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[218:219], v[214:215], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[218:219], v[216:217], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[220:221], v[210:211], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[220:221], v[212:213], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[220:221], v[214:215], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[220:221], v[216:217], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[222:223], v[210:211], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[222:223], v[212:213], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[222:223], v[214:215], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[222:223], v[216:217], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[224:225], v[210:211], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[224:225], v[212:213], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[224:225], v[214:215], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[224:225], v[216:217], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[202:203], v[210:211], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[202:203], v[212:213], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[202:203], v[214:215], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[202:203], v[216:217], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[204:205], v[210:211], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[204:205], v[212:213], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[204:205], v[214:215], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[204:205], v[216:217], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[206:207], v[210:211], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[206:207], v[212:213], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[206:207], v[214:215], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[206:207], v[216:217], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[208:209], v[210:211], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[208:209], v[212:213], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[208:209], v[214:215], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[208:209], v[216:217], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(7)
	;;#ASMSTART
	ds_write_b64 v157, v[130:131]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v158, v[132:133]

	;;#ASMEND
	s_waitcnt vmcnt(6)
	;;#ASMSTART
	ds_write_b64 v159, v[174:175]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v160, v[176:177]

	;;#ASMEND
	s_waitcnt vmcnt(5)
	;;#ASMSTART
	ds_write_b64 v161, v[178:179]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v162, v[180:181]

	;;#ASMEND
	s_waitcnt vmcnt(4)
	;;#ASMSTART
	ds_write_b64 v163, v[182:183]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v164, v[184:185]

	;;#ASMEND
	s_waitcnt vmcnt(3)
	;;#ASMSTART
	ds_write_b64 v165, v[186:187]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v166, v[188:189]

	;;#ASMEND
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_write_b64 v167, v[190:191]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v168, v[192:193]

	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v169, v[194:195]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v170, v[196:197]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v171, v[198:199]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v172, v[200:201]

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[234:235], v[226:227], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[234:235], v[228:229], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[234:235], v[230:231], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[234:235], v[232:233], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[236:237], v[226:227], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[236:237], v[228:229], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[236:237], v[230:231], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[236:237], v[232:233], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[238:239], v[226:227], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[238:239], v[228:229], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[238:239], v[230:231], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[238:239], v[232:233], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[240:241], v[226:227], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[240:241], v[228:229], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[240:241], v[230:231], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[240:241], v[232:233], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[242:243], v[226:227], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[242:243], v[228:229], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[242:243], v[230:231], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[242:243], v[232:233], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[244:245], v[226:227], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[244:245], v[228:229], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[244:245], v[230:231], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[244:245], v[232:233], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[246:247], v[226:227], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[246:247], v[228:229], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[246:247], v[230:231], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[246:247], v[232:233], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[248:249], v[226:227], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[248:249], v[228:229], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[248:249], v[230:231], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[248:249], v[232:233], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s21, s21, 64
	s_cmpk_eq_i32 s21, 0x2000
	s_cbranch_scc0 .LBB0_3
; %bb.4:
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[130:131], v152 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v152 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v152 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v152 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v149 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[156:157], v149 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v149 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v149 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v150 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[154:155], v[130:131], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[154:155], v[132:133], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[154:155], v[146:147], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[154:155], v[152:153], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[156:157], v[130:131], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[156:157], v[132:133], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[156:157], v[146:147], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[156:157], v[152:153], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[158:159], v[130:131], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[158:159], v[132:133], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[158:159], v[146:147], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[158:159], v[152:153], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[148:149], v[130:131], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[148:149], v[132:133], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[148:149], v[146:147], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[148:149], v[152:153], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[160:161], v[130:131], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[160:161], v[132:133], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[160:161], v[146:147], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[160:161], v[152:153], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[162:163], v[130:131], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[162:163], v[132:133], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[162:163], v[146:147], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[162:163], v[152:153], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[164:165], v[130:131], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[164:165], v[132:133], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[164:165], v[146:147], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[164:165], v[152:153], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[150:151], v[130:131], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[150:151], v[132:133], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[150:151], v[146:147], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[150:151], v[152:153], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[130:131], v145 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v145 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v145 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v145 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v144 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v144 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v144 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v144 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[156:157], v143 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v143 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v143 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v143 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[150:151], v[130:131], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[150:151], v[132:133], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[150:151], v[146:147], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[150:151], v[148:149], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[152:153], v[130:131], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[152:153], v[132:133], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[152:153], v[146:147], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[152:153], v[148:149], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[154:155], v[130:131], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[154:155], v[132:133], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[154:155], v[146:147], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[154:155], v[148:149], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[144:145], v[130:131], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[144:145], v[132:133], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[144:145], v[146:147], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[144:145], v[148:149], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[156:157], v[130:131], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[156:157], v[132:133], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[156:157], v[146:147], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[156:157], v[148:149], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[158:159], v[130:131], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[158:159], v[132:133], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[158:159], v[146:147], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[158:159], v[148:149], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[160:161], v[130:131], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[160:161], v[132:133], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[160:161], v[146:147], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[160:161], v[148:149], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[162:163], v[130:131], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[162:163], v[132:133], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[162:163], v[146:147], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[126:129], v[162:163], v[148:149], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[182:183], v138 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[186:187], v138 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[190:191], v138 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v138 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v137 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v137 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v137 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v137 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v141 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v141 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[178:179], v141 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[196:197], v141 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[198:199], v142 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[200:201], v142 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v142 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[204:205], v142 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[206:207], v140 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[208:209], v140 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[210:211], v140 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[212:213], v140 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[214:215], v139 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[216:217], v139 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[218:219], v139 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[220:221], v139 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[58:61], v[130:131], v[182:183], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[130:131], v[186:187], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[130:131], v[190:191], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[130:131], v[194:195], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[132:133], v[182:183], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[132:133], v[186:187], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[132:133], v[190:191], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[132:133], v[194:195], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[144:145], v[182:183], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[144:145], v[186:187], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[144:145], v[190:191], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[144:145], v[194:195], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[146:147], v[182:183], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[146:147], v[186:187], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[146:147], v[190:191], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[146:147], v[194:195], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[130:133], v[148:149], v[182:183], v[66:69]
	v_mfma_f32_16x16x16_bf16 v[138:141], v[148:149], v[186:187], v[70:73]
	v_mfma_f32_16x16x16_bf16 v[142:145], v[148:149], v[190:191], v[74:77]
	v_mfma_f32_16x16x16_bf16 v[146:149], v[148:149], v[194:195], v[78:81]
	v_mfma_f32_16x16x16_bf16 v[150:153], v[162:163], v[182:183], v[82:85]
	v_mfma_f32_16x16x16_bf16 v[154:157], v[162:163], v[186:187], v[86:89]
	v_mfma_f32_16x16x16_bf16 v[158:161], v[162:163], v[190:191], v[90:93]
	v_mfma_f32_16x16x16_bf16 v[162:165], v[162:163], v[194:195], v[94:97]
	v_mfma_f32_16x16x16_bf16 v[166:169], v[178:179], v[182:183], v[98:101]
	v_mfma_f32_16x16x16_bf16 v[170:173], v[178:179], v[186:187], v[102:105]
	v_mfma_f32_16x16x16_bf16 v[174:177], v[178:179], v[190:191], v[106:109]
	v_mfma_f32_16x16x16_bf16 v[178:181], v[178:179], v[194:195], v[110:113]
	v_mfma_f32_16x16x16_bf16 v[182:185], v[196:197], v[182:183], v[114:117]
	v_mfma_f32_16x16x16_bf16 v[186:189], v[196:197], v[186:187], v[118:121]
	v_mfma_f32_16x16x16_bf16 v[190:193], v[196:197], v[190:191], v[122:125]
	v_mfma_f32_16x16x16_bf16 v[194:197], v[196:197], v[194:195], v[126:129]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[126:129], v[206:207], v[198:199], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[122:125], v[206:207], v[200:201], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[118:121], v[206:207], v[202:203], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[206:207], v[204:205], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[110:113], v[208:209], v[198:199], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[106:109], v[208:209], v[200:201], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[102:105], v[208:209], v[202:203], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[98:101], v[208:209], v[204:205], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[94:97], v[210:211], v[198:199], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[90:93], v[210:211], v[200:201], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[86:89], v[210:211], v[202:203], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[82:85], v[210:211], v[204:205], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[78:81], v[212:213], v[198:199], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[74:77], v[212:213], v[200:201], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[70:73], v[212:213], v[202:203], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[66:69], v[212:213], v[204:205], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[214:215], v[198:199], v[130:133]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[214:215], v[200:201], v[138:141]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[214:215], v[202:203], v[142:145]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[214:215], v[204:205], v[146:149]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[216:217], v[198:199], v[150:153]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[216:217], v[200:201], v[154:157]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[216:217], v[202:203], v[158:161]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[216:217], v[204:205], v[162:165]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[218:219], v[198:199], v[166:169]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[218:219], v[200:201], v[170:173]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[218:219], v[202:203], v[174:177]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[218:219], v[204:205], v[178:181]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[220:221], v[198:199], v[182:185]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[220:221], v[200:201], v[186:189]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[220:221], v[202:203], v[190:193]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[220:221], v[204:205], v[194:197]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	v_lshl_or_b32 v1, s11, 2, v1
	v_lshl_or_b32 v0, v136, 6, s19
	v_mul_lo_u32 v130, s10, v1
	v_mad_u64_u32 v[132:133], s[0:1], v135, s10, v[134:135]
	v_lshl_add_u32 v130, v130, 6, v0
	v_ashrrev_i32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v131, 31, v130
	v_lshlrev_b64 v[136:137], 1, v[132:133]
	v_add_u32_e32 v132, s10, v132
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[8:9]
	v_ashrrev_i32_e32 v133, 31, v132
	v_lshl_add_u64 v[138:139], v[130:131], 0, v[136:137]
	v_lshlrev_b64 v[140:141], 1, v[132:133]
	global_store_short_d16_hi v[138:139], v126, off
	v_lshl_add_u64 v[142:143], v[130:131], 0, v[140:141]
	v_add_u32_e32 v126, s10, v132
	global_store_short_d16_hi v[142:143], v127, off
	v_ashrrev_i32_e32 v127, 31, v126
	v_lshlrev_b64 v[132:133], 1, v[126:127]
	v_add_u32_e32 v126, s10, v126
	v_ashrrev_i32_e32 v127, 31, v126
	v_lshlrev_b64 v[126:127], 1, v[126:127]
	v_lshl_add_u64 v[144:145], v[130:131], 0, v[132:133]
	v_lshl_add_u64 v[146:147], v[130:131], 0, v[126:127]
	global_store_short_d16_hi v[144:145], v128, off
	global_store_short_d16_hi v[146:147], v129, off
	global_store_short_d16_hi v[138:139], v122, off offset:32
	global_store_short_d16_hi v[142:143], v123, off offset:32
	global_store_short_d16_hi v[144:145], v124, off offset:32
	global_store_short_d16_hi v[146:147], v125, off offset:32
	global_store_short_d16_hi v[138:139], v118, off offset:64
	global_store_short_d16_hi v[142:143], v119, off offset:64
	global_store_short_d16_hi v[144:145], v120, off offset:64
	global_store_short_d16_hi v[146:147], v121, off offset:64
	global_store_short_d16_hi v[138:139], v114, off offset:96
	global_store_short_d16_hi v[142:143], v115, off offset:96
	global_store_short_d16_hi v[144:145], v116, off offset:96
	global_store_short_d16_hi v[146:147], v117, off offset:96
	v_or_b32_e32 v114, 16, v135
	v_mad_u64_u32 v[114:115], s[0:1], v114, s10, v[134:135]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshlrev_b64 v[116:117], 1, v[114:115]
	v_add_u32_e32 v114, s10, v114
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[118:119], v[130:131], 0, v[116:117]
	v_lshlrev_b64 v[120:121], 1, v[114:115]
	global_store_short_d16_hi v[118:119], v110, off
	v_lshl_add_u64 v[122:123], v[130:131], 0, v[120:121]
	v_add_u32_e32 v110, s10, v114
	global_store_short_d16_hi v[122:123], v111, off
	v_ashrrev_i32_e32 v111, 31, v110
	v_lshlrev_b64 v[114:115], 1, v[110:111]
	v_add_u32_e32 v110, s10, v110
	v_ashrrev_i32_e32 v111, 31, v110
	v_lshlrev_b64 v[110:111], 1, v[110:111]
	v_lshl_add_u64 v[124:125], v[130:131], 0, v[114:115]
	v_lshl_add_u64 v[128:129], v[130:131], 0, v[110:111]
	global_store_short_d16_hi v[124:125], v112, off
	global_store_short_d16_hi v[128:129], v113, off
	global_store_short_d16_hi v[118:119], v106, off offset:32
	global_store_short_d16_hi v[122:123], v107, off offset:32
	global_store_short_d16_hi v[124:125], v108, off offset:32
	global_store_short_d16_hi v[128:129], v109, off offset:32
	global_store_short_d16_hi v[118:119], v102, off offset:64
	global_store_short_d16_hi v[122:123], v103, off offset:64
	global_store_short_d16_hi v[124:125], v104, off offset:64
	global_store_short_d16_hi v[128:129], v105, off offset:64
	global_store_short_d16_hi v[118:119], v98, off offset:96
	global_store_short_d16_hi v[122:123], v99, off offset:96
	global_store_short_d16_hi v[124:125], v100, off offset:96
	global_store_short_d16_hi v[128:129], v101, off offset:96
	v_or_b32_e32 v98, 32, v135
	v_mad_u64_u32 v[98:99], s[0:1], v98, s10, v[134:135]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshlrev_b64 v[100:101], 1, v[98:99]
	v_add_u32_e32 v98, s10, v98
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[102:103], v[130:131], 0, v[100:101]
	v_lshlrev_b64 v[104:105], 1, v[98:99]
	global_store_short_d16_hi v[102:103], v94, off
	v_lshl_add_u64 v[106:107], v[130:131], 0, v[104:105]
	v_add_u32_e32 v94, s10, v98
	global_store_short_d16_hi v[106:107], v95, off
	v_ashrrev_i32_e32 v95, 31, v94
	v_lshlrev_b64 v[98:99], 1, v[94:95]
	v_add_u32_e32 v94, s10, v94
	v_ashrrev_i32_e32 v95, 31, v94
	v_lshlrev_b64 v[94:95], 1, v[94:95]
	v_lshl_add_u64 v[108:109], v[130:131], 0, v[98:99]
	v_lshl_add_u64 v[112:113], v[130:131], 0, v[94:95]
	global_store_short_d16_hi v[108:109], v96, off
	global_store_short_d16_hi v[112:113], v97, off
	global_store_short_d16_hi v[102:103], v90, off offset:32
	global_store_short_d16_hi v[106:107], v91, off offset:32
	global_store_short_d16_hi v[108:109], v92, off offset:32
	global_store_short_d16_hi v[112:113], v93, off offset:32
	global_store_short_d16_hi v[102:103], v86, off offset:64
	global_store_short_d16_hi v[106:107], v87, off offset:64
	global_store_short_d16_hi v[108:109], v88, off offset:64
	global_store_short_d16_hi v[112:113], v89, off offset:64
	global_store_short_d16_hi v[102:103], v82, off offset:96
	global_store_short_d16_hi v[106:107], v83, off offset:96
	global_store_short_d16_hi v[108:109], v84, off offset:96
	global_store_short_d16_hi v[112:113], v85, off offset:96
	v_or_b32_e32 v82, 48, v135
	v_mad_u64_u32 v[82:83], s[0:1], v82, s10, v[134:135]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshlrev_b64 v[84:85], 1, v[82:83]
	v_add_u32_e32 v82, s10, v82
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[86:87], v[130:131], 0, v[84:85]
	v_lshlrev_b64 v[88:89], 1, v[82:83]
	global_store_short_d16_hi v[86:87], v78, off
	v_lshl_add_u64 v[90:91], v[130:131], 0, v[88:89]
	v_add_u32_e32 v78, s10, v82
	global_store_short_d16_hi v[90:91], v79, off
	v_ashrrev_i32_e32 v79, 31, v78
	v_lshlrev_b64 v[82:83], 1, v[78:79]
	v_add_u32_e32 v78, s10, v78
	v_ashrrev_i32_e32 v79, 31, v78
	v_lshlrev_b64 v[78:79], 1, v[78:79]
	v_lshl_add_u64 v[92:93], v[130:131], 0, v[82:83]
	v_lshl_add_u64 v[96:97], v[130:131], 0, v[78:79]
	global_store_short_d16_hi v[92:93], v80, off
	global_store_short_d16_hi v[96:97], v81, off
	global_store_short_d16_hi v[86:87], v74, off offset:32
	global_store_short_d16_hi v[90:91], v75, off offset:32
	global_store_short_d16_hi v[92:93], v76, off offset:32
	global_store_short_d16_hi v[96:97], v77, off offset:32
	global_store_short_d16_hi v[86:87], v70, off offset:64
	global_store_short_d16_hi v[90:91], v71, off offset:64
	global_store_short_d16_hi v[92:93], v72, off offset:64
	global_store_short_d16_hi v[96:97], v73, off offset:64
	global_store_short_d16_hi v[86:87], v66, off offset:96
	global_store_short_d16_hi v[90:91], v67, off offset:96
	global_store_short_d16_hi v[92:93], v68, off offset:96
	global_store_short_d16_hi v[96:97], v69, off offset:96
	v_mov_b32_e32 v66, 0x80
	v_lshl_or_b32 v1, v1, 6, v66
	v_mad_u64_u32 v[0:1], s[0:1], v1, s10, v[0:1]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[8:9]
	v_lshl_add_u64 v[66:67], v[0:1], 0, v[136:137]
	v_lshl_add_u64 v[68:69], v[0:1], 0, v[140:141]
	global_store_short_d16_hi v[66:67], v62, off
	global_store_short_d16_hi v[68:69], v63, off
	v_lshl_add_u64 v[62:63], v[0:1], 0, v[132:133]
	v_lshl_add_u64 v[70:71], v[0:1], 0, v[126:127]
	global_store_short_d16_hi v[62:63], v64, off
	global_store_short_d16_hi v[70:71], v65, off
	global_store_short_d16_hi v[66:67], v58, off offset:32
	global_store_short_d16_hi v[68:69], v59, off offset:32
	global_store_short_d16_hi v[62:63], v60, off offset:32
	global_store_short_d16_hi v[70:71], v61, off offset:32
	global_store_short_d16_hi v[66:67], v54, off offset:64
	global_store_short_d16_hi v[68:69], v55, off offset:64
	global_store_short_d16_hi v[62:63], v56, off offset:64
	global_store_short_d16_hi v[70:71], v57, off offset:64
	global_store_short_d16_hi v[66:67], v50, off offset:96
	global_store_short_d16_hi v[68:69], v51, off offset:96
	global_store_short_d16_hi v[62:63], v52, off offset:96
	global_store_short_d16_hi v[70:71], v53, off offset:96
	v_lshl_add_u64 v[50:51], v[0:1], 0, v[116:117]
	v_lshl_add_u64 v[52:53], v[0:1], 0, v[120:121]
	global_store_short_d16_hi v[50:51], v46, off
	global_store_short_d16_hi v[52:53], v47, off
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[114:115]
	v_lshl_add_u64 v[54:55], v[0:1], 0, v[110:111]
	global_store_short_d16_hi v[46:47], v48, off
	global_store_short_d16_hi v[54:55], v49, off
	global_store_short_d16_hi v[50:51], v42, off offset:32
	global_store_short_d16_hi v[52:53], v43, off offset:32
	global_store_short_d16_hi v[46:47], v44, off offset:32
	global_store_short_d16_hi v[54:55], v45, off offset:32
	global_store_short_d16_hi v[50:51], v38, off offset:64
	global_store_short_d16_hi v[52:53], v39, off offset:64
	global_store_short_d16_hi v[46:47], v40, off offset:64
	global_store_short_d16_hi v[54:55], v41, off offset:64
	global_store_short_d16_hi v[50:51], v34, off offset:96
	global_store_short_d16_hi v[52:53], v35, off offset:96
	global_store_short_d16_hi v[46:47], v36, off offset:96
	global_store_short_d16_hi v[54:55], v37, off offset:96
	v_lshl_add_u64 v[34:35], v[0:1], 0, v[100:101]
	v_lshl_add_u64 v[36:37], v[0:1], 0, v[104:105]
	global_store_short_d16_hi v[34:35], v30, off
	global_store_short_d16_hi v[36:37], v31, off
	v_lshl_add_u64 v[30:31], v[0:1], 0, v[98:99]
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[94:95]
	global_store_short_d16_hi v[30:31], v32, off
	global_store_short_d16_hi v[38:39], v33, off
	global_store_short_d16_hi v[34:35], v26, off offset:32
	global_store_short_d16_hi v[36:37], v27, off offset:32
	global_store_short_d16_hi v[30:31], v28, off offset:32
	global_store_short_d16_hi v[38:39], v29, off offset:32
	global_store_short_d16_hi v[34:35], v22, off offset:64
	global_store_short_d16_hi v[36:37], v23, off offset:64
	global_store_short_d16_hi v[30:31], v24, off offset:64
	global_store_short_d16_hi v[38:39], v25, off offset:64
	global_store_short_d16_hi v[34:35], v18, off offset:96
	global_store_short_d16_hi v[36:37], v19, off offset:96
	global_store_short_d16_hi v[30:31], v20, off offset:96
	global_store_short_d16_hi v[38:39], v21, off offset:96
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[84:85]
	v_lshl_add_u64 v[20:21], v[0:1], 0, v[88:89]
	global_store_short_d16_hi v[18:19], v14, off
	global_store_short_d16_hi v[20:21], v15, off
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[82:83]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[78:79]
	global_store_short_d16_hi v[14:15], v16, off
	global_store_short_d16_hi v[0:1], v17, off
	global_store_short_d16_hi v[18:19], v10, off offset:32
	global_store_short_d16_hi v[20:21], v11, off offset:32
	global_store_short_d16_hi v[14:15], v12, off offset:32
	global_store_short_d16_hi v[0:1], v13, off offset:32
	global_store_short_d16_hi v[18:19], v6, off offset:64
	global_store_short_d16_hi v[20:21], v7, off offset:64
	global_store_short_d16_hi v[14:15], v8, off offset:64
	global_store_short_d16_hi v[0:1], v9, off offset:64
	global_store_short_d16_hi v[18:19], v2, off offset:96
	global_store_short_d16_hi v[20:21], v3, off offset:96
	global_store_short_d16_hi v[14:15], v4, off offset:96
	global_store_short_d16_hi v[0:1], v5, off offset:96
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
		.amdhsa_next_free_sgpr 30
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
; codeLenInByte = 7472
; NumSgprs: 36
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
; NumSGPRsForWavesPerEU: 36
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
	.type	__hip_cuid_97904f98876f312c,@object ; @__hip_cuid_97904f98876f312c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_97904f98876f312c
__hip_cuid_97904f98876f312c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_97904f98876f312c, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_97904f98876f312c
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
    .sgpr_count:     36
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
