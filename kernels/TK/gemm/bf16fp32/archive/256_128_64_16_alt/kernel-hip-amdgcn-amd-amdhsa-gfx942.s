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
	s_cselect_b32 s20, 0, 0
	s_and_b32 s8, s20, -16
	s_load_dwordx2 s[18:19], s[0:1], 0x0
	s_load_dwordx2 s[4:5], s[0:1], 0x20
	s_load_dwordx2 s[14:15], s[0:1], 0x30
	s_load_dwordx2 s[6:7], s[0:1], 0x50
	s_add_i32 s21, s8, 16
	s_load_dwordx2 s[10:11], s[0:1], 0x60
	s_load_dwordx2 s[8:9], s[0:1], 0x80
	s_load_dwordx2 s[24:25], s[0:1], 0x90
	s_mov_b32 s17, 0
	s_and_b32 s16, s20, 15
	s_cmp_eq_u64 s[16:17], 0
	s_cselect_b32 s21, s20, s21
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s24, s3
	s_add_i32 s0, s0, s2
	s_ashr_i32 s2, s0, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s0, s2
	s_mul_i32 s1, s25, s24
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
	s_sub_i32 s25, 0, s20
	s_and_b32 s1, s1, 0xfffffe00
	s_sub_i32 s0, s0, s1
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s24, s0
	s_add_i32 s22, s21, 0x4000
	s_and_b32 s23, s22, -16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_xor_b32 s1, s0, s3
	s_and_b32 s16, s22, 15
	s_add_i32 s23, s23, 16
	v_readfirstlane_b32 s26, v1
	s_mul_i32 s25, s25, s26
	s_mul_hi_u32 s25, s26, s25
	s_add_i32 s26, s26, s25
	s_mul_hi_u32 s25, s24, s26
	s_mul_i32 s26, s25, s20
	s_sub_i32 s24, s24, s26
	s_ashr_i32 s1, s1, 31
	s_add_i32 s26, s25, 1
	s_sub_i32 s27, s24, s20
	s_cmp_ge_u32 s24, s20
	s_cselect_b32 s25, s26, s25
	s_cselect_b32 s24, s27, s24
	s_add_i32 s26, s25, 1
	s_cmp_ge_u32 s24, s20
	s_cselect_b32 s20, s26, s25
	s_xor_b32 s20, s20, s1
	s_sub_i32 s24, s20, s1
	s_mul_i32 s1, s24, s3
	s_sub_i32 s0, s0, s1
	s_add_i32 s2, s2, s0
	s_lshl_b32 s20, s2, 7
	s_ashr_i32 s0, s20, 31
	s_mul_i32 s0, s4, s0
	s_mul_hi_u32 s1, s4, s20
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s5, s20
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s4, s20
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s18, s0
	s_addc_u32 s1, s19, s1
	s_lshl_b32 s18, s24, 8
	s_ashr_i32 s2, s18, 31
	s_mul_i32 s2, s6, s2
	s_mul_hi_u32 s3, s6, s18
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s7, s18
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s6, s18
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v1, 3, v0
	s_add_u32 s14, s14, s2
	v_and_b32_e32 v18, 56, v1
	v_lshrrev_b32_e32 v1, 3, v0
	s_addc_u32 s15, s15, s3
	v_mad_u64_u32 v[2:3], s[2:3], v1, s4, v[18:19]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[0:1]
	v_or_b32_e32 v3, 64, v1
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[2:3], v3, s4, v[18:19]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[10:11], v[4:5], 1, s[0:1]
	v_lshlrev_b32_e32 v5, 4, v0
	v_lshlrev_b32_e32 v31, 1, v18
	v_and_b32_e32 v30, 0x1f80, v5
	v_add_u32_e32 v14, s21, v31
	v_add_u32_e32 v16, v14, v30
	v_or_b32_e32 v32, 8, v31
	v_lshrrev_b32_e32 v17, 4, v16
	v_add_u32_e32 v15, s21, v32
	v_and_b32_e32 v17, 0x78, v17
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v16, v17, v16
	;;#ASMSTART
	ds_write_b64 v16, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v6, v15, v30
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	v_or_b32_e32 v33, 0x2000, v30
	;;#ASMSTART
	ds_write_b64 v6, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v6, v14, v33
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v6, v15, v33
	v_lshrrev_b32_e32 v7, 4, v6
	v_mad_u64_u32 v[22:23], s[2:3], v1, s6, v[18:19]
	v_or_b32_e32 v1, 0xc0, v1
	s_cmp_eq_u64 s[16:17], 0
	v_and_b32_e32 v7, 0x78, v7
	v_mad_u64_u32 v[24:25], s[2:3], v3, s6, v[18:19]
	v_mad_u64_u32 v[28:29], s[2:3], v1, s6, v[18:19]
	v_xor_b32_e32 v6, v7, v6
	v_ashrrev_i32_e32 v23, 31, v22
	s_cselect_b32 s2, s22, s23
	;;#ASMSTART
	ds_write_b64 v6, v[12:13]

	;;#ASMEND
	v_lshl_add_u64 v[6:7], v[22:23], 1, s[14:15]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u32 v26, s6, 7, v22
	v_add_u32_e32 v1, s2, v31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	v_lshl_add_u64 v[10:11], v[24:25], 1, s[14:15]
	v_ashrrev_i32_e32 v27, 31, v26
	v_add_u32_e32 v23, v1, v30
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	v_lshl_add_u64 v[14:15], v[26:27], 1, s[14:15]
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshrrev_b32_e32 v25, 4, v23
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[14:15], off

	;;#ASMEND
	v_lshl_add_u64 v[18:19], v[28:29], 1, s[14:15]
	v_add_u32_e32 v3, s2, v32
	v_and_b32_e32 v25, 0x78, v25
	;;#ASMSTART
	global_load_dwordx4 v[18:21], v[18:19], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v23, v25, v23
	;;#ASMSTART
	ds_write_b64 v23, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v6, v3, v30
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v6, v1, v33
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v6, v3, v33
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[12:13]

	;;#ASMEND
	v_or_b32_e32 v6, 0x4000, v30
	v_add_u32_e32 v7, v1, v6
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v7, v8, v7
	;;#ASMSTART
	ds_write_b64 v7, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v7, v3, v6
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v7, v8, v7
	;;#ASMSTART
	ds_write_b64 v7, v[16:17]

	;;#ASMEND
	v_or_b32_e32 v7, 0x6000, v30
	v_add_u32_e32 v1, v1, v7
	v_lshrrev_b32_e32 v8, 4, v1
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v1, v8, v1
	;;#ASMSTART
	ds_write_b64 v1, v[18:19]

	;;#ASMEND
	v_add_u32_e32 v1, v3, v7
	v_lshrrev_b32_e32 v3, 4, v1
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v1, v3, v1
	v_and_b32_e32 v121, 0xc0, v0
	;;#ASMSTART
	ds_write_b64 v1, v[20:21]

	;;#ASMEND
	v_and_b32_e32 v120, 15, v0
	v_lshrrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v0, 7, v0
	v_and_b32_e32 v150, 12, v1
	v_and_b32_e32 v0, 0x6780, v0
	v_lshlrev_b32_e32 v3, 1, v150
	v_add_u32_e32 v0, s2, v0
	v_add_u32_e32 v8, v0, v3
	v_lshrrev_b32_e32 v9, 4, v8
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v152, v9, v8
	v_or_b32_e32 v8, 16, v120
	v_or_b32_e32 v9, v121, v8
	v_lshl_add_u32 v9, v9, 7, s2
	v_add_u32_e32 v10, v9, v3
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v153, v11, v10
	v_or_b32_e32 v10, 32, v120
	v_or_b32_e32 v11, v121, v10
	v_lshl_add_u32 v11, v11, 7, s2
	v_add_u32_e32 v12, v11, v3
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v154, v13, v12
	v_or_b32_e32 v12, 48, v120
	v_or_b32_e32 v13, v121, v12
	v_lshl_add_u32 v13, v13, 7, s2
	v_add_u32_e32 v14, v13, v3
	v_and_b32_e32 v151, 64, v1
	v_lshrrev_b32_e32 v15, 4, v14
	v_or_b32_e32 v1, v151, v120
	v_and_b32_e32 v15, 0x78, v15
	v_lshl_add_u32 v1, v1, 7, s21
	v_xor_b32_e32 v155, v15, v14
	v_add_u32_e32 v14, v1, v3
	v_lshrrev_b32_e32 v15, 4, v14
	v_or_b32_e32 v8, v8, v151
	v_and_b32_e32 v15, 0x78, v15
	v_lshl_add_u32 v8, v8, 7, s21
	v_xor_b32_e32 v156, v15, v14
	v_add_u32_e32 v14, v8, v3
	v_lshrrev_b32_e32 v15, 4, v14
	v_or_b32_e32 v10, v10, v151
	v_and_b32_e32 v15, 0x78, v15
	v_lshl_add_u32 v10, v10, 7, s21
	v_xor_b32_e32 v157, v15, v14
	v_add_u32_e32 v14, v10, v3
	v_lshrrev_b32_e32 v15, 4, v14
	v_or_b32_e32 v12, v12, v151
	v_and_b32_e32 v15, 0x78, v15
	v_lshl_add_u32 v12, v12, 7, s21
	v_xor_b32_e32 v158, v15, v14
	v_add_u32_e32 v14, v12, v3
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v159, v15, v14
	v_or_b32_e32 v14, 32, v3
	v_lshlrev_b32_e32 v162, 1, v2
	v_add_u32_e32 v2, v9, v14
	v_lshlrev_b32_e32 v161, 1, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v163, v4, v2
	v_add_u32_e32 v2, v11, v14
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v164, v4, v2
	v_add_u32_e32 v2, v13, v14
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v165, v4, v2
	v_add_u32_e32 v2, v1, v14
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v166, v4, v2
	v_add_u32_e32 v2, v8, v14
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v167, v4, v2
	v_add_u32_e32 v2, v10, v14
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v168, v4, v2
	v_add_u32_e32 v2, v12, v14
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v169, v4, v2
	v_or_b32_e32 v2, 64, v3
	v_add_u32_e32 v4, v0, v2
	v_add_u32_e32 v15, v0, v14
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v174, v14, v4
	v_add_u32_e32 v4, v9, v2
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v175, v14, v4
	v_add_u32_e32 v4, v11, v2
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v176, v14, v4
	v_add_u32_e32 v4, v13, v2
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v177, v14, v4
	v_add_u32_e32 v4, v1, v2
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v178, v14, v4
	v_add_u32_e32 v4, v8, v2
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v179, v14, v4
	v_add_u32_e32 v4, v10, v2
	v_lshrrev_b32_e32 v14, 4, v4
	v_and_b32_e32 v14, 0x78, v14
	v_add_u32_e32 v2, v12, v2
	v_xor_b32_e32 v180, v14, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v181, v4, v2
	v_or_b32_e32 v2, 0x60, v3
	v_add_u32_e32 v0, v0, v2
	v_lshrrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v182, v3, v0
	v_add_u32_e32 v0, v9, v2
	v_lshrrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v183, v3, v0
	v_add_u32_e32 v0, v11, v2
	v_lshrrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v184, v3, v0
	v_add_u32_e32 v0, v13, v2
	v_lshrrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v185, v3, v0
	v_add_u32_e32 v0, v1, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v186, v1, v0
	v_add_u32_e32 v0, v8, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v187, v1, v0
	v_add_u32_e32 v0, v10, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v188, v1, v0
	v_add_u32_e32 v0, v12, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v189, v1, v0
	v_and_b32_e32 v0, 0x70, v5
	v_add_u32_e32 v1, s21, v0
	v_add_u32_e32 v4, v1, v30
	v_or_b32_e32 v2, 8, v0
	v_lshrrev_b32_e32 v5, 4, v4
	v_add_u32_e32 v3, s21, v2
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v190, v5, v4
	v_add_u32_e32 v4, v3, v30
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v1, v1, v33
	v_xor_b32_e32 v191, v5, v4
	v_lshrrev_b32_e32 v4, 4, v1
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v192, v4, v1
	v_add_u32_e32 v1, v3, v33
	v_lshrrev_b32_e32 v3, 4, v1
	v_and_b32_e32 v3, 0x78, v3
	v_add_u32_e32 v0, s2, v0
	v_xor_b32_e32 v193, v3, v1
	v_add_u32_e32 v1, s2, v2
	v_add_u32_e32 v2, v0, v30
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v194, v3, v2
	v_add_u32_e32 v2, v1, v30
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v195, v3, v2
	v_add_u32_e32 v2, v0, v33
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v196, v3, v2
	v_add_u32_e32 v2, v1, v33
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v197, v3, v2
	v_add_u32_e32 v2, v0, v6
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v198, v3, v2
	v_add_u32_e32 v2, v1, v6
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b64 v[122:123], v152

	;;#ASMEND
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_add_u32_e32 v0, v0, v7
	;;#ASMSTART
	ds_read_b64 v[124:125], v153

	;;#ASMEND
	v_xor_b32_e32 v199, v3, v2
	v_lshrrev_b32_e32 v2, 4, v0
	;;#ASMSTART
	ds_read_b64 v[126:127], v154

	;;#ASMEND
	v_and_b32_e32 v2, 0x78, v2
	;;#ASMSTART
	ds_read_b64 v[128:129], v155

	;;#ASMEND
	v_xor_b32_e32 v200, v2, v0
	v_add_u32_e32 v0, v1, v7
	;;#ASMSTART
	ds_read_b64 v[130:131], v156
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_lshl_b32 s2, s4, 8
	s_lshl_b32 s6, s6, 9
	v_lshrrev_b32_e32 v1, 4, v0
	;;#ASMSTART
	ds_read_b64 v[132:133], v157
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_add_u32 s14, s14, 0x80
	v_lshrrev_b32_e32 v16, 4, v15
	v_and_b32_e32 v1, 0x78, v1
	;;#ASMSTART
	ds_read_b64 v[134:135], v158
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_addc_u32 s15, s15, 0
	s_mov_b32 s16, 0xffff
	v_and_b32_e32 v16, 0x78, v16
	v_xor_b32_e32 v201, v1, v0
	;;#ASMSTART
	ds_read_b64 v[136:137], v159
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_add_u32 s17, s0, 0x80
	v_mov_b32_e32 v0, 0
	v_mov_b64_e32 v[66:67], 0
	s_mov_b64 s[12:13], 0
	v_bfi_b32 v122, s16, v122, v122
	v_xor_b32_e32 v160, v16, v15
	v_lshlrev_b32_e32 v170, 1, v22
	v_lshlrev_b32_e32 v171, 1, v24
	v_lshlrev_b32_e32 v172, 1, v26
	v_lshlrev_b32_e32 v173, 1, v28
	v_bfi_b32 v124, s16, v124, v124
	v_bfi_b32 v126, s16, v126, v126
	v_bfi_b32 v128, s16, v128, v128
	v_bfi_b32 v130, s16, v130, v130
	v_bfi_b32 v132, s16, v132, v132
	v_bfi_b32 v134, s16, v134, v134
	v_bfi_b32 v136, s16, v136, v136
	s_addc_u32 s19, s1, 0
	s_mov_b32 s3, 0x20000
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b64_e32 v[58:59], v[66:67]
	v_mov_b64_e32 v[78:79], v[66:67]
	v_mov_b64_e32 v[74:75], v[66:67]
	v_mov_b64_e32 v[70:71], v[66:67]
	v_mov_b64_e32 v[62:63], v[66:67]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	v_bfi_b32 v34, s16, v34, v34
	v_bfi_b32 v32, s16, v32, v32
	v_bfi_b32 v144, s16, v144, v144
	v_bfi_b32 v142, s16, v142, v142
	v_bfi_b32 v140, s16, v140, v140
	v_bfi_b32 v138, s16, v138, v138
	v_bfi_b32 v148, s16, v148, v148
	v_bfi_b32 v146, s16, v146, v146
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[84:87], v[34:35], v[144:145], v[0:3]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[34:35], v[142:143], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[34:35], v[140:141], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[34:35], v[138:139], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[32:33], v[144:145], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[32:33], v[142:143], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[32:33], v[140:141], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[32:33], v[138:139], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[148:149], v[144:145], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[148:149], v[142:143], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[148:149], v[140:141], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[148:149], v[138:139], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[146:147], v[144:145], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[146:147], v[142:143], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[146:147], v[140:141], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[146:147], v[138:139], v[116:119]
	s_setprio 0
	s_add_u32 s12, s12, 0x80
	s_addc_u32 s13, s13, 0
	s_cmpk_eq_i32 s12, 0x4000
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc1 .LBB0_10
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_cmpk_lg_i32 s12, 0x3f80
	s_cselect_b64 s[4:5], -1, 0
	s_cmpk_eq_i32 s12, 0x3f80
	s_cbranch_scc1 .LBB0_4
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	s_add_u32 s0, s17, s12
	s_addc_u32 s1, s19, s13
	buffer_load_dwordx4 v[56:59], v162, s[0:3], 0 offen
	buffer_load_dwordx4 v[64:67], v161, s[0:3], 0 offen
.LBB0_4:                                ;   in Loop: Header=BB0_2 Depth=1
	;;#ASMSTART
	ds_read_b64 v[94:95], v160

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[92:93], v163

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[90:91], v164

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[88:89], v165

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[102:103], v166

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[100:101], v167

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[98:99], v168

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[96:97], v169

	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[84:87], v[130:131], v[122:123], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[130:131], v[124:125], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[130:131], v[126:127], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[130:131], v[128:129], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[132:133], v[122:123], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[132:133], v[124:125], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[132:133], v[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[132:133], v[128:129], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[134:135], v[122:123], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[134:135], v[124:125], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[134:135], v[126:127], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[134:135], v[128:129], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[136:137], v[122:123], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[136:137], v[124:125], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[136:137], v[126:127], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[136:137], v[128:129], v[0:3]
	s_setprio 0
	v_cndmask_b32_e64 v104, 0, 1, s[4:5]
	v_cmp_ne_u32_e64 s[0:1], 1, v104
	s_andn2_b64 vcc, exec, s[4:5]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_vccnz .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
	s_add_u32 s4, s14, s12
	s_addc_u32 s5, s15, s13
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v[60:63], v170, s[4:7], 0 offen
	buffer_load_dwordx4 v[68:71], v171, s[4:7], 0 offen
	buffer_load_dwordx4 v[72:75], v172, s[4:7], 0 offen
	buffer_load_dwordx4 v[76:79], v173, s[4:7], 0 offen
.LBB0_6:                                ;   in Loop: Header=BB0_2 Depth=1
	;;#ASMSTART
	ds_read_b64 v[122:123], v174

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v175

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v176

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[128:129], v177

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v178
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v179
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v180
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v181
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v94, s16, v94, v94
	v_bfi_b32 v92, s16, v92, v92
	v_bfi_b32 v90, s16, v90, v90
	v_bfi_b32 v88, s16, v88, v88
	v_bfi_b32 v102, s16, v102, v102
	v_bfi_b32 v100, s16, v100, v100
	v_bfi_b32 v98, s16, v98, v98
	v_bfi_b32 v96, s16, v96, v96
	v_bfi_b32 v122, s16, v122, v122
	v_bfi_b32 v124, s16, v124, v124
	v_bfi_b32 v126, s16, v126, v126
	v_bfi_b32 v128, s16, v128, v128
	v_bfi_b32 v130, s16, v130, v130
	v_bfi_b32 v132, s16, v132, v132
	v_bfi_b32 v134, s16, v134, v134
	v_bfi_b32 v136, s16, v136, v136
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[84:87], v[102:103], v[94:95], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[102:103], v[92:93], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[102:103], v[90:91], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[102:103], v[88:89], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[100:101], v[94:95], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[100:101], v[92:93], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[100:101], v[90:91], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[100:101], v[88:89], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[98:99], v[94:95], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[98:99], v[92:93], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[98:99], v[90:91], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[98:99], v[88:89], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[202:205], v[96:97], v[94:95], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[206:209], v[96:97], v[92:93], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[210:213], v[96:97], v[90:91], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[214:217], v[96:97], v[88:89], v[0:3]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v182

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v183

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v184

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v185

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[34:35], v186

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[32:33], v187

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v188

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v189

	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[0:3], v[130:131], v[122:123], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[130:131], v[124:125], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[130:131], v[126:127], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[130:131], v[128:129], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[132:133], v[122:123], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[132:133], v[124:125], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[132:133], v[126:127], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[132:133], v[128:129], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[134:135], v[122:123], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[134:135], v[124:125], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[134:135], v[126:127], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[134:135], v[128:129], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[136:137], v[122:123], v[202:205]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[136:137], v[124:125], v[206:209]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[136:137], v[126:127], v[210:213]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[136:137], v[128:129], v[214:217]
	s_setprio 0
	s_and_b64 vcc, exec, s[0:1]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_vccnz .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_2 Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v190, v[56:57]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v191, v[58:59]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v192, v[64:65]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v193, v[66:67]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v194, v[60:61]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v195, v[62:63]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v196, v[68:69]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v197, v[70:71]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v198, v[72:73]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v199, v[74:75]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v200, v[76:77]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v201, v[78:79]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_8:                                ;   in Loop: Header=BB0_2 Depth=1
	s_and_b64 vcc, exec, s[0:1]
	s_barrier
	s_cbranch_vccnz .LBB0_1
; %bb.9:                                ;   in Loop: Header=BB0_2 Depth=1
	;;#ASMSTART
	ds_read_b64 v[122:123], v152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v153

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v154

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[128:129], v155

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v156
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v157
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v158
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v159
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v122, s16, v122, v122
	v_bfi_b32 v124, s16, v124, v124
	v_bfi_b32 v126, s16, v126, v126
	v_bfi_b32 v128, s16, v128, v128
	v_bfi_b32 v130, s16, v130, v130
	v_bfi_b32 v132, s16, v132, v132
	v_bfi_b32 v134, s16, v134, v134
	v_bfi_b32 v136, s16, v136, v136
	s_branch .LBB0_1
.LBB0_10:
	s_mov_b32 s0, 0x7f800000
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v56, 0x7f800000, v84
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.11:
	v_bfe_u32 v56, v84, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v56, v84, v56, s2
; %bb.12:                               ; %Flow1709
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.13:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v84
	v_cmp_eq_u32_sdwa vcc, v84, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v56, v57, v84, vcc
; %bb.14:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v57, 0x7f800000, v85
	v_cmp_ne_u32_e32 vcc, s0, v57
                                        ; implicit-def: $vgpr57
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.15:
	v_bfe_u32 v57, v85, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v57, v85, v57, s2
; %bb.16:                               ; %Flow1708
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.17:
	v_mov_b32_e32 v57, 0
	v_or_b32_e32 v58, 0x10000, v85
	v_cmp_eq_u32_sdwa vcc, v85, v57 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v57, v58, v85, vcc
; %bb.18:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v58, 0x7f800000, v86
	v_cmp_ne_u32_e32 vcc, s0, v58
                                        ; implicit-def: $vgpr58
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.19:
	v_bfe_u32 v58, v86, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v58, v86, v58, s2
; %bb.20:                               ; %Flow1707
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.21:
	v_mov_b32_e32 v58, 0
	v_or_b32_e32 v59, 0x10000, v86
	v_cmp_eq_u32_sdwa vcc, v86, v58 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v58, v59, v86, vcc
; %bb.22:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v59, 0x7f800000, v87
	v_cmp_ne_u32_e32 vcc, s0, v59
                                        ; implicit-def: $vgpr59
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.23:
	v_bfe_u32 v59, v87, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v59, v87, v59, s2
; %bb.24:                               ; %Flow1706
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.25:
	v_mov_b32_e32 v59, 0
	v_or_b32_e32 v60, 0x10000, v87
	v_cmp_eq_u32_sdwa vcc, v87, v59 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v59, v60, v87, vcc
; %bb.26:                               ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v80
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr60
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.27:
	v_bfe_u32 v60, v80, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v60, v80, v60, s2
; %bb.28:                               ; %Flow1705
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.29:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v61, 0x10000, v80
	v_cmp_eq_u32_sdwa vcc, v80, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v60, v61, v80, vcc
; %bb.30:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v61, 0x7f800000, v81
	v_cmp_ne_u32_e32 vcc, s0, v61
                                        ; implicit-def: $vgpr61
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.31:
	v_bfe_u32 v61, v81, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v61, v81, v61, s2
; %bb.32:                               ; %Flow1704
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.33:
	v_mov_b32_e32 v61, 0
	v_or_b32_e32 v62, 0x10000, v81
	v_cmp_eq_u32_sdwa vcc, v81, v61 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v61, v62, v81, vcc
; %bb.34:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v62, 0x7f800000, v82
	v_cmp_ne_u32_e32 vcc, s0, v62
                                        ; implicit-def: $vgpr62
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.35:
	v_bfe_u32 v62, v82, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v62, v82, v62, s2
; %bb.36:                               ; %Flow1703
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.37:
	v_mov_b32_e32 v62, 0
	v_or_b32_e32 v63, 0x10000, v82
	v_cmp_eq_u32_sdwa vcc, v82, v62 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v62, v63, v82, vcc
; %bb.38:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v63, 0x7f800000, v83
	v_cmp_ne_u32_e32 vcc, s0, v63
                                        ; implicit-def: $vgpr63
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.39:
	v_bfe_u32 v63, v83, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v63, v83, v63, s2
; %bb.40:                               ; %Flow1702
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_mov_b32_e32 v63, 0
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v64, 0x10000, v83
	v_cmp_eq_u32_sdwa vcc, v83, v63 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v63, v64, v83, vcc
.LBB0_42:                               ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v64, 0x7f800000, v52
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr64
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.43:
	v_bfe_u32 v64, v52, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v64, v52, v64, s2
; %bb.44:                               ; %Flow1701
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.45:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v52
	v_cmp_eq_u32_sdwa vcc, v52, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v64, v65, v52, vcc
; %bb.46:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v53
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.47:
	v_bfe_u32 v52, v53, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v53, v52, s2
; %bb.48:                               ; %Flow1700
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.49:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v65, 0x10000, v53
	v_cmp_eq_u32_sdwa vcc, v53, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v65, v53, vcc
; %bb.50:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v53, 0x7f800000, v54
	v_cmp_ne_u32_e32 vcc, s0, v53
                                        ; implicit-def: $vgpr53
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.51:
	v_bfe_u32 v53, v54, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v53, v54, v53, s2
; %bb.52:                               ; %Flow1699
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.53:
	v_mov_b32_e32 v53, 0
	v_or_b32_e32 v65, 0x10000, v54
	v_cmp_eq_u32_sdwa vcc, v54, v53 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v53, v65, v54, vcc
; %bb.54:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v54, 0x7f800000, v55
	v_cmp_ne_u32_e32 vcc, s0, v54
                                        ; implicit-def: $vgpr54
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.55:
	v_bfe_u32 v54, v55, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v54, v55, v54, s2
; %bb.56:                               ; %Flow1698
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.57:
	v_mov_b32_e32 v54, 0
	v_or_b32_e32 v65, 0x10000, v55
	v_cmp_eq_u32_sdwa vcc, v55, v54 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v54, v65, v55, vcc
; %bb.58:                               ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v55, 0x7f800000, v48
	v_cmp_ne_u32_e32 vcc, s0, v55
                                        ; implicit-def: $vgpr55
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.59:
	v_bfe_u32 v55, v48, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v55, v48, v55, s2
; %bb.60:                               ; %Flow1697
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.61:
	v_mov_b32_e32 v55, 0
	v_or_b32_e32 v65, 0x10000, v48
	v_cmp_eq_u32_sdwa vcc, v48, v55 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v55, v65, v48, vcc
; %bb.62:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v49
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr48
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.63:
	v_bfe_u32 v48, v49, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v48, v49, v48, s2
; %bb.64:                               ; %Flow1696
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.65:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v65, 0x10000, v49
	v_cmp_eq_u32_sdwa vcc, v49, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v48, v65, v49, vcc
; %bb.66:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v49, 0x7f800000, v50
	v_cmp_ne_u32_e32 vcc, s0, v49
                                        ; implicit-def: $vgpr49
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.67:
	v_bfe_u32 v49, v50, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v49, v50, v49, s2
; %bb.68:                               ; %Flow1695
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.69:
	v_mov_b32_e32 v49, 0
	v_or_b32_e32 v65, 0x10000, v50
	v_cmp_eq_u32_sdwa vcc, v50, v49 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v49, v65, v50, vcc
; %bb.70:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v50, 0x7f800000, v51
	v_cmp_ne_u32_e32 vcc, s0, v50
                                        ; implicit-def: $vgpr50
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.71:
	v_bfe_u32 v50, v51, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v50, v51, v50, s2
; %bb.72:                               ; %Flow1694
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.73:
	v_mov_b32_e32 v50, 0
	v_or_b32_e32 v65, 0x10000, v51
	v_cmp_eq_u32_sdwa vcc, v51, v50 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v50, v65, v51, vcc
; %bb.74:                               ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v51, 0x7f800000, v44
	v_cmp_ne_u32_e32 vcc, s0, v51
                                        ; implicit-def: $vgpr51
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.75:
	v_bfe_u32 v51, v44, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v44, v51, s2
; %bb.76:                               ; %Flow1693
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.77:
	v_mov_b32_e32 v51, 0
	v_or_b32_e32 v65, 0x10000, v44
	v_cmp_eq_u32_sdwa vcc, v44, v51 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v51, v65, v44, vcc
; %bb.78:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.120.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v45
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr44
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.79:
	v_bfe_u32 v44, v45, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v44, v45, v44, s2
; %bb.80:                               ; %Flow1692
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.81:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v65, 0x10000, v45
	v_cmp_eq_u32_sdwa vcc, v45, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v44, v65, v45, vcc
; %bb.82:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.126.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v45, 0x7f800000, v46
	v_cmp_ne_u32_e32 vcc, s0, v45
                                        ; implicit-def: $vgpr45
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.83:
	v_bfe_u32 v45, v46, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v45, v46, v45, s2
; %bb.84:                               ; %Flow1691
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.85:
	v_mov_b32_e32 v45, 0
	v_or_b32_e32 v65, 0x10000, v46
	v_cmp_eq_u32_sdwa vcc, v46, v45 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v45, v65, v46, vcc
; %bb.86:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.130.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v46, 0x7f800000, v47
	v_cmp_ne_u32_e32 vcc, s0, v46
                                        ; implicit-def: $vgpr46
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.87:
	v_bfe_u32 v46, v47, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v46, v47, v46, s2
; %bb.88:                               ; %Flow1690
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.89:
	v_mov_b32_e32 v46, 0
	v_or_b32_e32 v65, 0x10000, v47
	v_cmp_eq_u32_sdwa vcc, v47, v46 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v46, v65, v47, vcc
; %bb.90:                               ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.133.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v47, 0x7f800000, v40
	v_cmp_ne_u32_e32 vcc, s0, v47
                                        ; implicit-def: $vgpr47
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.91:
	v_bfe_u32 v47, v40, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v47, v40, v47, s2
; %bb.92:                               ; %Flow1689
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.93:
	v_mov_b32_e32 v47, 0
	v_or_b32_e32 v65, 0x10000, v40
	v_cmp_eq_u32_sdwa vcc, v40, v47 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v47, v65, v40, vcc
; %bb.94:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.1.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v41
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.95:
	v_bfe_u32 v40, v41, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v41, v40, s2
; %bb.96:                               ; %Flow1688
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.97:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v65, 0x10000, v41
	v_cmp_eq_u32_sdwa vcc, v41, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v65, v41, vcc
; %bb.98:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.1.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v41, 0x7f800000, v42
	v_cmp_ne_u32_e32 vcc, s0, v41
                                        ; implicit-def: $vgpr41
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.99:
	v_bfe_u32 v41, v42, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v41, v42, v41, s2
; %bb.100:                              ; %Flow1687
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.101:
	v_mov_b32_e32 v41, 0
	v_or_b32_e32 v65, 0x10000, v42
	v_cmp_eq_u32_sdwa vcc, v42, v41 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v41, v65, v42, vcc
; %bb.102:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.1.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v42, 0x7f800000, v43
	v_cmp_ne_u32_e32 vcc, s0, v42
                                        ; implicit-def: $vgpr42
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.103:
	v_bfe_u32 v42, v43, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v42, v43, v42, s2
; %bb.104:                              ; %Flow1686
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.105:
	v_mov_b32_e32 v42, 0
	v_or_b32_e32 v65, 0x10000, v43
	v_cmp_eq_u32_sdwa vcc, v43, v42 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v42, v65, v43, vcc
; %bb.106:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.1.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v43, 0x7f800000, v36
	v_cmp_ne_u32_e32 vcc, s0, v43
                                        ; implicit-def: $vgpr43
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.107:
	v_bfe_u32 v43, v36, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v43, v36, v43, s2
; %bb.108:                              ; %Flow1685
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.109:
	v_mov_b32_e32 v43, 0
	v_or_b32_e32 v65, 0x10000, v36
	v_cmp_eq_u32_sdwa vcc, v36, v43 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v43, v65, v36, vcc
; %bb.110:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.2.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v37
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.111:
	v_bfe_u32 v36, v37, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v37, v36, s2
; %bb.112:                              ; %Flow1684
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.113:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v65, 0x10000, v37
	v_cmp_eq_u32_sdwa vcc, v37, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v65, v37, vcc
; %bb.114:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.2.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v37, 0x7f800000, v38
	v_cmp_ne_u32_e32 vcc, s0, v37
                                        ; implicit-def: $vgpr37
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.115:
	v_bfe_u32 v37, v38, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v37, v38, v37, s2
; %bb.116:                              ; %Flow1683
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.117:
	v_mov_b32_e32 v37, 0
	v_or_b32_e32 v65, 0x10000, v38
	v_cmp_eq_u32_sdwa vcc, v38, v37 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v37, v65, v38, vcc
; %bb.118:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.2.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v38, 0x7f800000, v39
	v_cmp_ne_u32_e32 vcc, s0, v38
                                        ; implicit-def: $vgpr38
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.119:
	v_bfe_u32 v38, v39, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v38, v39, v38, s2
; %bb.120:                              ; %Flow1682
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.121:
	v_mov_b32_e32 v38, 0
	v_or_b32_e32 v65, 0x10000, v39
	v_cmp_eq_u32_sdwa vcc, v39, v38 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v38, v65, v39, vcc
; %bb.122:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.2.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v39, 0x7f800000, v32
	v_cmp_ne_u32_e32 vcc, s0, v39
                                        ; implicit-def: $vgpr39
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.123:
	v_bfe_u32 v39, v32, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v39, v32, v39, s2
; %bb.124:                              ; %Flow1681
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.125:
	v_mov_b32_e32 v39, 0
	v_or_b32_e32 v65, 0x10000, v32
	v_cmp_eq_u32_sdwa vcc, v32, v39 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v39, v65, v32, vcc
; %bb.126:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.3.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v33
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr32
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.127:
	v_bfe_u32 v32, v33, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v32, v33, v32, s2
; %bb.128:                              ; %Flow1680
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.129:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v65, 0x10000, v33
	v_cmp_eq_u32_sdwa vcc, v33, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v32, v65, v33, vcc
; %bb.130:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.3.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v33, 0x7f800000, v34
	v_cmp_ne_u32_e32 vcc, s0, v33
                                        ; implicit-def: $vgpr33
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.131:
	v_bfe_u32 v33, v34, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v33, v34, v33, s2
; %bb.132:                              ; %Flow1679
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.133:
	v_mov_b32_e32 v33, 0
	v_or_b32_e32 v65, 0x10000, v34
	v_cmp_eq_u32_sdwa vcc, v34, v33 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v33, v65, v34, vcc
; %bb.134:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.3.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v34, 0x7f800000, v35
	v_cmp_ne_u32_e32 vcc, s0, v34
                                        ; implicit-def: $vgpr34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.135:
	v_bfe_u32 v34, v35, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v34, v35, v34, s2
; %bb.136:                              ; %Flow1678
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.137:
	v_mov_b32_e32 v34, 0
	v_or_b32_e32 v65, 0x10000, v35
	v_cmp_eq_u32_sdwa vcc, v35, v34 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v34, v65, v35, vcc
; %bb.138:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.3.1.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v35, 0x7f800000, v28
	v_cmp_ne_u32_e32 vcc, s0, v35
                                        ; implicit-def: $vgpr35
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.139:
	v_bfe_u32 v35, v28, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v28, v35, s2
; %bb.140:                              ; %Flow1677
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.141:
	v_mov_b32_e32 v35, 0
	v_or_b32_e32 v65, 0x10000, v28
	v_cmp_eq_u32_sdwa vcc, v28, v35 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v35, v65, v28, vcc
; %bb.142:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.240.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v29
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr28
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.143:
	v_bfe_u32 v28, v29, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v28, v29, v28, s2
; %bb.144:                              ; %Flow1676
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.145:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v65, 0x10000, v29
	v_cmp_eq_u32_sdwa vcc, v29, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v28, v65, v29, vcc
; %bb.146:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.246.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v29, 0x7f800000, v30
	v_cmp_ne_u32_e32 vcc, s0, v29
                                        ; implicit-def: $vgpr29
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.147:
	v_bfe_u32 v29, v30, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v29, v30, v29, s2
; %bb.148:                              ; %Flow1675
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.149:
	v_mov_b32_e32 v29, 0
	v_or_b32_e32 v65, 0x10000, v30
	v_cmp_eq_u32_sdwa vcc, v30, v29 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v29, v65, v30, vcc
; %bb.150:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.250.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v30, 0x7f800000, v31
	v_cmp_ne_u32_e32 vcc, s0, v30
                                        ; implicit-def: $vgpr30
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.151:
	v_bfe_u32 v30, v31, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v30, v31, v30, s2
; %bb.152:                              ; %Flow1674
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.153:
	v_mov_b32_e32 v30, 0
	v_or_b32_e32 v65, 0x10000, v31
	v_cmp_eq_u32_sdwa vcc, v31, v30 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v30, v65, v31, vcc
; %bb.154:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.253.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v31, 0x7f800000, v24
	v_cmp_ne_u32_e32 vcc, s0, v31
                                        ; implicit-def: $vgpr31
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.155:
	v_bfe_u32 v31, v24, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v31, v24, v31, s2
; %bb.156:                              ; %Flow1673
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.157:
	v_mov_b32_e32 v31, 0
	v_or_b32_e32 v65, 0x10000, v24
	v_cmp_eq_u32_sdwa vcc, v24, v31 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v31, v65, v24, vcc
; %bb.158:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.1.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v25
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.159:
	v_bfe_u32 v24, v25, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v25, v24, s2
; %bb.160:                              ; %Flow1672
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.161:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v65, 0x10000, v25
	v_cmp_eq_u32_sdwa vcc, v25, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v65, v25, vcc
; %bb.162:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.1.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v25, 0x7f800000, v26
	v_cmp_ne_u32_e32 vcc, s0, v25
                                        ; implicit-def: $vgpr25
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.163:
	v_bfe_u32 v25, v26, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v25, v26, v25, s2
; %bb.164:                              ; %Flow1671
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.165:
	v_mov_b32_e32 v25, 0
	v_or_b32_e32 v65, 0x10000, v26
	v_cmp_eq_u32_sdwa vcc, v26, v25 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v25, v65, v26, vcc
; %bb.166:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.1.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v26, 0x7f800000, v27
	v_cmp_ne_u32_e32 vcc, s0, v26
                                        ; implicit-def: $vgpr26
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.167:
	v_bfe_u32 v26, v27, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v26, v27, v26, s2
; %bb.168:                              ; %Flow1670
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.169:
	v_mov_b32_e32 v26, 0
	v_or_b32_e32 v65, 0x10000, v27
	v_cmp_eq_u32_sdwa vcc, v27, v26 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v26, v65, v27, vcc
; %bb.170:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.1.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v27, 0x7f800000, v20
	v_cmp_ne_u32_e32 vcc, s0, v27
                                        ; implicit-def: $vgpr27
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.171:
	v_bfe_u32 v27, v20, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v27, v20, v27, s2
; %bb.172:                              ; %Flow1669
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.173:
	v_mov_b32_e32 v27, 0
	v_or_b32_e32 v65, 0x10000, v20
	v_cmp_eq_u32_sdwa vcc, v20, v27 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v27, v65, v20, vcc
; %bb.174:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.2.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v21
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.175:
	v_bfe_u32 v20, v21, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v21, v20, s2
; %bb.176:                              ; %Flow1668
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.177:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v65, 0x10000, v21
	v_cmp_eq_u32_sdwa vcc, v21, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v65, v21, vcc
; %bb.178:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.2.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v21, 0x7f800000, v22
	v_cmp_ne_u32_e32 vcc, s0, v21
                                        ; implicit-def: $vgpr21
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.179:
	v_bfe_u32 v21, v22, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v21, v22, v21, s2
; %bb.180:                              ; %Flow1667
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.181:
	v_mov_b32_e32 v21, 0
	v_or_b32_e32 v65, 0x10000, v22
	v_cmp_eq_u32_sdwa vcc, v22, v21 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v21, v65, v22, vcc
; %bb.182:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.2.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v22, 0x7f800000, v23
	v_cmp_ne_u32_e32 vcc, s0, v22
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.183:
	v_bfe_u32 v22, v23, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v22, v23, v22, s2
; %bb.184:                              ; %Flow1666
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.185:
	v_mov_b32_e32 v22, 0
	v_or_b32_e32 v65, 0x10000, v23
	v_cmp_eq_u32_sdwa vcc, v23, v22 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v22, v65, v23, vcc
; %bb.186:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.2.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v23, 0x7f800000, v16
	v_cmp_ne_u32_e32 vcc, s0, v23
                                        ; implicit-def: $vgpr23
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.187:
	v_bfe_u32 v23, v16, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v23, v16, v23, s2
; %bb.188:                              ; %Flow1665
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.189:
	v_mov_b32_e32 v23, 0
	v_or_b32_e32 v65, 0x10000, v16
	v_cmp_eq_u32_sdwa vcc, v16, v23 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v23, v65, v16, vcc
; %bb.190:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.3.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v17
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr16
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.191:
	v_bfe_u32 v16, v17, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v16, v17, v16, s2
; %bb.192:                              ; %Flow1664
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.193:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v65, 0x10000, v17
	v_cmp_eq_u32_sdwa vcc, v17, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v16, v65, v17, vcc
; %bb.194:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.3.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v17, 0x7f800000, v18
	v_cmp_ne_u32_e32 vcc, s0, v17
                                        ; implicit-def: $vgpr17
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.195:
	v_bfe_u32 v17, v18, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v17, v18, v17, s2
; %bb.196:                              ; %Flow1663
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.197:
	v_mov_b32_e32 v17, 0
	v_or_b32_e32 v65, 0x10000, v18
	v_cmp_eq_u32_sdwa vcc, v18, v17 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v17, v65, v18, vcc
; %bb.198:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.3.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v18, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc, s0, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.199:
	v_bfe_u32 v18, v19, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v19, v18, s2
; %bb.200:                              ; %Flow1662
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.201:
	v_mov_b32_e32 v18, 0
	v_or_b32_e32 v65, 0x10000, v19
	v_cmp_eq_u32_sdwa vcc, v19, v18 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v18, v65, v19, vcc
; %bb.202:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.3.2.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v19, 0x7f800000, v12
	v_cmp_ne_u32_e32 vcc, s0, v19
                                        ; implicit-def: $vgpr19
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.203:
	v_bfe_u32 v19, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v19, v12, v19, s2
; %bb.204:                              ; %Flow1661
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.205:
	v_mov_b32_e32 v19, 0
	v_or_b32_e32 v65, 0x10000, v12
	v_cmp_eq_u32_sdwa vcc, v12, v19 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v19, v65, v12, vcc
; %bb.206:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.360.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v13
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.207:
	v_bfe_u32 v12, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v12, v13, v12, s2
; %bb.208:                              ; %Flow1660
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.209:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v65, 0x10000, v13
	v_cmp_eq_u32_sdwa vcc, v13, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v12, v65, v13, vcc
; %bb.210:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.366.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v13, 0x7f800000, v14
	v_cmp_ne_u32_e32 vcc, s0, v13
                                        ; implicit-def: $vgpr13
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.211:
	v_bfe_u32 v13, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v13, v14, v13, s2
; %bb.212:                              ; %Flow1659
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.213:
	v_mov_b32_e32 v13, 0
	v_or_b32_e32 v65, 0x10000, v14
	v_cmp_eq_u32_sdwa vcc, v14, v13 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v13, v65, v14, vcc
; %bb.214:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.370.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v14, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc, s0, v14
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.215:
	v_bfe_u32 v14, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v14, v15, v14, s2
; %bb.216:                              ; %Flow1658
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.217:
	v_mov_b32_e32 v14, 0
	v_or_b32_e32 v65, 0x10000, v15
	v_cmp_eq_u32_sdwa vcc, v15, v14 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v14, v65, v15, vcc
; %bb.218:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.373.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v15, 0x7f800000, v8
	v_cmp_ne_u32_e32 vcc, s0, v15
                                        ; implicit-def: $vgpr15
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.219:
	v_bfe_u32 v15, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v15, v8, v15, s2
; %bb.220:                              ; %Flow1657
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.221:
	v_mov_b32_e32 v15, 0
	v_or_b32_e32 v65, 0x10000, v8
	v_cmp_eq_u32_sdwa vcc, v8, v15 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v15, v65, v8, vcc
; %bb.222:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.1.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v9
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.223:
	v_bfe_u32 v8, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v8, v9, v8, s2
; %bb.224:                              ; %Flow1656
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.225:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v65, 0x10000, v9
	v_cmp_eq_u32_sdwa vcc, v9, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v8, v65, v9, vcc
; %bb.226:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.1.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v9, 0x7f800000, v10
	v_cmp_ne_u32_e32 vcc, s0, v9
                                        ; implicit-def: $vgpr9
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.227:
	v_bfe_u32 v9, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v9, v10, v9, s2
; %bb.228:                              ; %Flow1655
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.229:
	v_mov_b32_e32 v9, 0
	v_or_b32_e32 v65, 0x10000, v10
	v_cmp_eq_u32_sdwa vcc, v10, v9 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v9, v65, v10, vcc
; %bb.230:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.1.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v10, 0x7f800000, v11
	v_cmp_ne_u32_e32 vcc, s0, v10
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.231:
	v_bfe_u32 v10, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v10, v11, v10, s2
; %bb.232:                              ; %Flow1654
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.233:
	v_mov_b32_e32 v10, 0
	v_or_b32_e32 v65, 0x10000, v11
	v_cmp_eq_u32_sdwa vcc, v11, v10 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v10, v65, v11, vcc
; %bb.234:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.1.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v11, 0x7f800000, v4
	v_cmp_ne_u32_e32 vcc, s0, v11
                                        ; implicit-def: $vgpr11
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.235:
	v_bfe_u32 v11, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v11, v4, v11, s2
; %bb.236:                              ; %Flow1653
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.237:
	v_mov_b32_e32 v11, 0
	v_or_b32_e32 v65, 0x10000, v4
	v_cmp_eq_u32_sdwa vcc, v4, v11 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v11, v65, v4, vcc
; %bb.238:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.2.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v5
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.239:
	v_bfe_u32 v4, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v5, v4, s2
; %bb.240:                              ; %Flow1652
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.241:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v65, 0x10000, v5
	v_cmp_eq_u32_sdwa vcc, v5, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v4, v65, v5, vcc
; %bb.242:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.2.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v5, 0x7f800000, v6
	v_cmp_ne_u32_e32 vcc, s0, v5
                                        ; implicit-def: $vgpr5
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.243:
	v_bfe_u32 v5, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v5, v6, v5, s2
; %bb.244:                              ; %Flow1651
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.245:
	v_mov_b32_e32 v5, 0
	v_or_b32_e32 v65, 0x10000, v6
	v_cmp_eq_u32_sdwa vcc, v6, v5 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v5, v65, v6, vcc
; %bb.246:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.2.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v6, 0x7f800000, v7
	v_cmp_ne_u32_e32 vcc, s0, v6
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.247:
	v_bfe_u32 v6, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v6, v7, v6, s2
; %bb.248:                              ; %Flow1650
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.249:
	v_mov_b32_e32 v6, 0
	v_or_b32_e32 v65, 0x10000, v7
	v_cmp_eq_u32_sdwa vcc, v7, v6 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v6, v65, v7, vcc
; %bb.250:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fTkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_7rt_baseIT_T1_EERKNS5_IT0_S7_EE.exit.2.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v7, 0x7f800000, v0
	v_cmp_ne_u32_e32 vcc, s0, v7
                                        ; implicit-def: $vgpr7
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.251:
	v_bfe_u32 v7, v0, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v7, v0, v7, s2
; %bb.252:                              ; %Flow1649
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.253:
	v_mov_b32_e32 v7, 0
	v_or_b32_e32 v65, 0x10000, v0
	v_cmp_eq_u32_sdwa vcc, v0, v7 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v7, v65, v0, vcc
; %bb.254:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.3.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.255:
	v_bfe_u32 v0, v1, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v1, v0, s2
; %bb.256:                              ; %Flow1648
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.257:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v65, 0x10000, v1
	v_cmp_eq_u32_sdwa vcc, v1, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v65, v1, vcc
; %bb.258:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.3.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v2
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.259:
	v_bfe_u32 v1, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v2, v1, s2
; %bb.260:                              ; %Flow1647
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.261:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v65, 0x10000, v2
	v_cmp_eq_u32_sdwa vcc, v2, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v65, v2, vcc
; %bb.262:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.3.3.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v2, 0x7f800000, v3
	v_cmp_ne_u32_e32 vcc, s0, v2
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.263:
	v_bfe_u32 v2, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v3, v2, s2
; %bb.264:                              ; %Flow
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.265:
	v_mov_b32_e32 v2, 0
	v_or_b32_e32 v65, 0x10000, v3
	v_cmp_eq_u32_sdwa vcc, v3, v2 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v2, v65, v3, vcc
; %bb.266:                              ; %_ZN7kittensL4copyI14__hip_bfloat16fLi64ELi64ETkNS_5ducks9rt_layout3allENS3_3colEEEvRNS_2rtIT_XT1_EXT2_ET3_EERKNS5_IT0_XT1_EXT2_ES7_EE.exit
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v3, s20, v151
	v_ashrrev_i32_e32 v65, 31, v3
	v_mul_lo_u32 v65, s8, v65
	v_mul_lo_u32 v67, s9, v3
	v_mad_u64_u32 v[68:69], s[0:1], s8, v3, 0
	v_or_b32_e32 v66, s18, v121
	v_add3_u32 v69, v69, v65, v67
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[68:69], v[68:69], 1, s[10:11]
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[68:69]
	v_mad_u64_u32 v[68:69], s[0:1], v150, s8, v[120:121]
	v_ashrrev_i32_e32 v69, 31, v68
	v_lshl_add_u64 v[70:71], v[68:69], 1, v[66:67]
	v_add_u32_e32 v68, s8, v68
	v_ashrrev_i32_e32 v69, 31, v68
	global_store_short_d16_hi v[70:71], v56, off
	v_lshl_add_u64 v[72:73], v[68:69], 1, v[66:67]
	v_add_u32_e32 v56, s8, v68
	global_store_short_d16_hi v[72:73], v57, off
	v_ashrrev_i32_e32 v57, 31, v56
	v_lshl_add_u64 v[68:69], v[56:57], 1, v[66:67]
	v_add_u32_e32 v56, s8, v56
	v_ashrrev_i32_e32 v57, 31, v56
	v_lshl_add_u64 v[56:57], v[56:57], 1, v[66:67]
	v_or_b32_e32 v3, 16, v150
	global_store_short_d16_hi v[68:69], v58, off
	global_store_short_d16_hi v[56:57], v59, off
	global_store_short_d16_hi v[70:71], v60, off offset:32
	global_store_short_d16_hi v[72:73], v61, off offset:32
	global_store_short_d16_hi v[68:69], v62, off offset:32
	global_store_short_d16_hi v[56:57], v63, off offset:32
	global_store_short_d16_hi v[70:71], v64, off offset:64
	global_store_short_d16_hi v[72:73], v52, off offset:64
	global_store_short_d16_hi v[68:69], v53, off offset:64
	global_store_short_d16_hi v[56:57], v54, off offset:64
	global_store_short_d16_hi v[70:71], v55, off offset:96
	global_store_short_d16_hi v[72:73], v48, off offset:96
	global_store_short_d16_hi v[68:69], v49, off offset:96
	global_store_short_d16_hi v[56:57], v50, off offset:96
	v_mad_u64_u32 v[48:49], s[0:1], v3, s8, v[120:121]
	v_ashrrev_i32_e32 v49, 31, v48
	v_lshl_add_u64 v[52:53], v[48:49], 1, v[66:67]
	v_add_u32_e32 v48, s8, v48
	v_ashrrev_i32_e32 v49, 31, v48
	global_store_short_d16_hi v[52:53], v51, off
	v_lshl_add_u64 v[50:51], v[48:49], 1, v[66:67]
	v_add_u32_e32 v48, s8, v48
	v_ashrrev_i32_e32 v49, 31, v48
	global_store_short_d16_hi v[50:51], v44, off
	v_lshl_add_u64 v[54:55], v[48:49], 1, v[66:67]
	v_add_u32_e32 v44, s8, v48
	global_store_short_d16_hi v[54:55], v45, off
	v_ashrrev_i32_e32 v45, 31, v44
	v_lshl_add_u64 v[44:45], v[44:45], 1, v[66:67]
	v_or_b32_e32 v3, 32, v150
	global_store_short_d16_hi v[44:45], v46, off
	global_store_short_d16_hi v[52:53], v47, off offset:32
	global_store_short_d16_hi v[50:51], v40, off offset:32
	global_store_short_d16_hi v[54:55], v41, off offset:32
	global_store_short_d16_hi v[44:45], v42, off offset:32
	global_store_short_d16_hi v[52:53], v43, off offset:64
	global_store_short_d16_hi v[50:51], v36, off offset:64
	global_store_short_d16_hi v[54:55], v37, off offset:64
	global_store_short_d16_hi v[44:45], v38, off offset:64
	global_store_short_d16_hi v[52:53], v39, off offset:96
	global_store_short_d16_hi v[50:51], v32, off offset:96
	global_store_short_d16_hi v[54:55], v33, off offset:96
	global_store_short_d16_hi v[44:45], v34, off offset:96
	v_mad_u64_u32 v[32:33], s[0:1], v3, s8, v[120:121]
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshl_add_u64 v[36:37], v[32:33], 1, v[66:67]
	v_add_u32_e32 v32, s8, v32
	v_ashrrev_i32_e32 v33, 31, v32
	global_store_short_d16_hi v[36:37], v35, off
	v_lshl_add_u64 v[34:35], v[32:33], 1, v[66:67]
	v_add_u32_e32 v32, s8, v32
	v_ashrrev_i32_e32 v33, 31, v32
	global_store_short_d16_hi v[34:35], v28, off
	v_lshl_add_u64 v[38:39], v[32:33], 1, v[66:67]
	v_add_u32_e32 v28, s8, v32
	global_store_short_d16_hi v[38:39], v29, off
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshl_add_u64 v[28:29], v[28:29], 1, v[66:67]
	v_or_b32_e32 v3, 48, v150
	global_store_short_d16_hi v[28:29], v30, off
	global_store_short_d16_hi v[36:37], v31, off offset:32
	global_store_short_d16_hi v[34:35], v24, off offset:32
	global_store_short_d16_hi v[38:39], v25, off offset:32
	global_store_short_d16_hi v[28:29], v26, off offset:32
	global_store_short_d16_hi v[36:37], v27, off offset:64
	global_store_short_d16_hi v[34:35], v20, off offset:64
	global_store_short_d16_hi v[38:39], v21, off offset:64
	global_store_short_d16_hi v[28:29], v22, off offset:64
	global_store_short_d16_hi v[36:37], v23, off offset:96
	global_store_short_d16_hi v[34:35], v16, off offset:96
	global_store_short_d16_hi v[38:39], v17, off offset:96
	global_store_short_d16_hi v[28:29], v18, off offset:96
	v_mad_u64_u32 v[16:17], s[0:1], v3, s8, v[120:121]
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[20:21], v[16:17], 1, v[66:67]
	v_add_u32_e32 v16, s8, v16
	v_ashrrev_i32_e32 v17, 31, v16
	global_store_short_d16_hi v[20:21], v19, off
	v_lshl_add_u64 v[18:19], v[16:17], 1, v[66:67]
	v_add_u32_e32 v16, s8, v16
	v_ashrrev_i32_e32 v17, 31, v16
	global_store_short_d16_hi v[18:19], v12, off
	v_lshl_add_u64 v[22:23], v[16:17], 1, v[66:67]
	v_add_u32_e32 v12, s8, v16
	global_store_short_d16_hi v[22:23], v13, off
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[12:13], v[12:13], 1, v[66:67]
	global_store_short_d16_hi v[12:13], v14, off
	global_store_short_d16_hi v[20:21], v15, off offset:32
	global_store_short_d16_hi v[18:19], v8, off offset:32
	global_store_short_d16_hi v[22:23], v9, off offset:32
	global_store_short_d16_hi v[12:13], v10, off offset:32
	global_store_short_d16_hi v[20:21], v11, off offset:64
	global_store_short_d16_hi v[18:19], v4, off offset:64
	global_store_short_d16_hi v[22:23], v5, off offset:64
	global_store_short_d16_hi v[12:13], v6, off offset:64
	global_store_short_d16_hi v[20:21], v7, off offset:96
	global_store_short_d16_hi v[18:19], v0, off offset:96
	global_store_short_d16_hi v[22:23], v1, off offset:96
	global_store_short_d16_hi v[12:13], v2, off offset:96
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
		.amdhsa_next_free_vgpr 218
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 220
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
; codeLenInByte = 10332
; NumSgprs: 34
; NumVgprs: 218
; NumAgprs: 0
; TotalNumVgprs: 218
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 27
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 218
; AccumOffset: 220
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 54
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_7ea7e7a68cf63f6c,@object ; @__hip_cuid_7ea7e7a68cf63f6c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7ea7e7a68cf63f6c
__hip_cuid_7ea7e7a68cf63f6c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7ea7e7a68cf63f6c, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_7ea7e7a68cf63f6c
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
    .vgpr_count:     218
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
