	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:
	s_load_dwordx2 s[24:25], s[0:1], 0x0
	s_load_dwordx2 s[20:21], s[0:1], 0x20
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx2 s[14:15], s[0:1], 0x50
	s_load_dwordx8 s[4:11], s[0:1], 0x60
	s_load_dwordx2 s[16:17], s[0:1], 0x80
	s_load_dwordx2 s[28:29], s[0:1], 0x90
	s_cmp_lg_u32 0, -1
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s7, 0, 0
	s_and_b32 s9, s7, -16
	s_mov_b32 s23, 0
	s_and_b32 s22, s7, 15
	s_add_i32 s9, s9, 16
	s_cmp_eq_u64 s[22:23], 0
	s_mul_i32 s0, s28, s3
	s_cselect_b32 s26, s7, s9
	s_add_i32 s0, s0, s2
	s_ashr_i32 s2, s0, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s0, s2
	s_mul_i32 s1, s29, s28
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
	s_lshl_b32 s7, s2, 4
	s_sub_i32 s2, 64, s7
	s_min_i32 s2, s2, 16
	s_abs_i32 s3, s2
	v_cvt_f32_u32_e32 v1, s3
	s_sub_i32 s11, 0, s3
	s_add_i32 s27, s26, 0x4000
	s_and_b32 s1, s1, 0xfffffe00
	v_rcp_iflag_f32_e32 v1, v1
	s_and_b32 s9, s27, -16
	s_sub_i32 s0, s0, s1
	s_add_i32 s28, s9, 16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_abs_i32 s9, s0
	s_xor_b32 s1, s0, s2
	s_and_b32 s22, s27, 15
	v_readfirstlane_b32 s29, v1
	s_mul_i32 s11, s11, s29
	s_mul_hi_u32 s11, s29, s11
	s_add_i32 s29, s29, s11
	s_mul_hi_u32 s11, s9, s29
	s_mul_i32 s29, s11, s3
	s_sub_i32 s9, s9, s29
	s_ashr_i32 s1, s1, 31
	s_add_i32 s29, s11, 1
	s_sub_i32 s30, s9, s3
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s11, s29, s11
	s_cselect_b32 s9, s30, s9
	s_add_i32 s29, s11, 1
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s3, s29, s11
	s_xor_b32 s3, s3, s1
	s_sub_i32 s3, s3, s1
	s_mul_i32 s1, s3, s2
	s_sub_i32 s0, s0, s1
	s_add_i32 s7, s7, s0
	s_lshl_b32 s0, s7, 7
	s_ashr_i32 s1, s0, 31
	s_mul_i32 s1, s20, s1
	s_mul_hi_u32 s2, s20, s0
	s_add_i32 s1, s2, s1
	s_mul_i32 s2, s21, s0
	s_add_i32 s1, s1, s2
	s_mul_i32 s0, s20, s0
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s24, s0
	s_addc_u32 s1, s25, s1
	s_lshl_b32 s11, s3, 8
	s_ashr_i32 s9, s11, 31
	s_mul_i32 s2, s14, s9
	s_mul_hi_u32 s3, s14, s11
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s15, s11
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s14, s11
	s_lshl_b64 s[2:3], s[2:3], 1
	v_lshlrev_b32_e32 v1, 3, v0
	s_add_u32 s12, s12, s2
	v_and_b32_e32 v12, 56, v1
	v_lshrrev_b32_e32 v1, 3, v0
	s_addc_u32 s13, s13, s3
	v_mad_u64_u32 v[6:7], s[2:3], v1, s20, v[12:13]
	v_or_b32_e32 v13, 64, v1
	v_ashrrev_i32_e32 v7, 31, v6
	v_mad_u64_u32 v[10:11], s[2:3], v13, s20, v[12:13]
	v_lshl_add_u64 v[2:3], v[6:7], 1, s[0:1]
	v_ashrrev_i32_e32 v11, 31, v10
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[8:9], v[10:11], 1, s[0:1]
	v_lshlrev_b32_e32 v14, 4, v0
	v_lshlrev_b32_e32 v32, 1, v12
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[8:9], off

	;;#ASMEND
	v_and_b32_e32 v7, 0x1f80, v14
	v_add_u32_e32 v8, s26, v32
	v_add_u32_e32 v11, v8, v7
	v_or_b32_e32 v33, 8, v32
	v_lshrrev_b32_e32 v20, 4, v11
	v_add_u32_e32 v9, s26, v33
	v_and_b32_e32 v20, 0x78, v20
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v11, v20, v11
	;;#ASMSTART
	ds_write_b64 v11, v[2:3]

	;;#ASMEND
	v_add_u32_e32 v2, v9, v7
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	v_or_b32_e32 v11, 0x2000, v7
	;;#ASMSTART
	ds_write_b64 v2, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v2, v8, v11
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[16:17]

	;;#ASMEND
	v_add_u32_e32 v2, v9, v11
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[18:19]

	;;#ASMEND
	v_mad_u64_u32 v[2:3], s[2:3], v1, s14, v[12:13]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[12:13]
	s_cmp_eq_u64 s[22:23], 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[2:3], v13, s14, v[12:13]
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v1, 0xc0, v1
	s_cselect_b32 s15, s27, s28
	v_lshl_add_u64 v[8:9], v[4:5], 1, s[12:13]
	v_mad_u64_u32 v[12:13], s[2:3], v1, s14, v[12:13]
	v_add_u32_e32 v1, s15, v32
	;;#ASMSTART
	global_load_dwordx4 v[20:23], v[8:9], off

	;;#ASMEND
	v_lshl_add_u32 v8, s14, 7, v2
	v_add_u32_e32 v3, v1, v7
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshrrev_b32_e32 v5, 4, v3
	v_lshl_add_u64 v[24:25], v[8:9], 1, s[12:13]
	v_ashrrev_i32_e32 v13, 31, v12
	v_and_b32_e32 v5, 0x78, v5
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off

	;;#ASMEND
	v_lshl_add_u64 v[28:29], v[12:13], 1, s[12:13]
	v_add_u32_e32 v9, s15, v33
	v_xor_b32_e32 v3, v5, v3
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v3, v[16:17]

	;;#ASMEND
	v_add_u32_e32 v3, v9, v7
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v3, v5, v3
	;;#ASMSTART
	ds_write_b64 v3, v[18:19]

	;;#ASMEND
	v_add_u32_e32 v3, v1, v11
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v3, v5, v3
	;;#ASMSTART
	ds_write_b64 v3, v[20:21]

	;;#ASMEND
	v_add_u32_e32 v3, v9, v11
	v_lshrrev_b32_e32 v5, 4, v3
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v3, v5, v3
	;;#ASMSTART
	ds_write_b64 v3, v[22:23]

	;;#ASMEND
	v_or_b32_e32 v3, 0x4000, v7
	v_add_u32_e32 v5, v1, v3
	v_lshrrev_b32_e32 v13, 4, v5
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v5, v13, v5
	;;#ASMSTART
	ds_write_b64 v5, v[24:25]

	;;#ASMEND
	v_add_u32_e32 v5, v9, v3
	v_lshrrev_b32_e32 v13, 4, v5
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v5, v13, v5
	;;#ASMSTART
	ds_write_b64 v5, v[26:27]

	;;#ASMEND
	v_or_b32_e32 v5, 0x6000, v7
	v_add_u32_e32 v1, v1, v5
	v_lshrrev_b32_e32 v13, 4, v1
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v1, v13, v1
	;;#ASMSTART
	ds_write_b64 v1, v[28:29]

	;;#ASMEND
	v_add_u32_e32 v1, v9, v5
	v_lshrrev_b32_e32 v9, 4, v1
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v1, v9, v1
	;;#ASMSTART
	ds_write_b64 v1, v[30:31]

	;;#ASMEND
	v_lshrrev_b32_e32 v15, 8, v0
	s_mov_b64 s[18:19], 0
	v_cmp_eq_u32_e32 vcc, 1, v15
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
	v_bfe_u32 v1, v0, 6, 2
	v_lshlrev_b32_e32 v9, 5, v1
	v_and_b32_e32 v155, 15, v0
	v_lshrrev_b32_e32 v13, 2, v0
	v_and_b32_e32 v154, 12, v13
	v_or_b32_e32 v13, v9, v155
	v_lshlrev_b32_e32 v16, 1, v154
	v_lshl_add_u32 v13, v13, 7, s15
	v_add_u32_e32 v17, v13, v16
	v_lshrrev_b32_e32 v18, 4, v17
	v_lshlrev_b32_e32 v159, 1, v6
	v_or_b32_e32 v6, 32, v16
	v_and_b32_e32 v18, 0x78, v18
	v_lshlrev_b32_e32 v158, 1, v10
	v_add_u32_e32 v10, v13, v6
	v_xor_b32_e32 v157, v18, v17
	v_lshrrev_b32_e32 v17, 4, v10
	v_and_b32_e32 v17, 0x78, v17
	v_xor_b32_e32 v160, v17, v10
	v_or_b32_e32 v10, 16, v155
	v_or_b32_e32 v17, v9, v10
	v_lshl_add_u32 v17, v17, 7, s15
	v_add_u32_e32 v18, v17, v16
	v_lshrrev_b32_e32 v19, 4, v18
	v_and_b32_e32 v19, 0x78, v19
	v_xor_b32_e32 v161, v19, v18
	v_add_u32_e32 v18, v17, v6
	v_lshrrev_b32_e32 v19, 4, v18
	v_and_b32_e32 v19, 0x78, v19
	v_or_b32_e32 v9, 0x80, v9
	v_xor_b32_e32 v162, v19, v18
	v_or_b32_e32 v18, v9, v155
	v_lshl_add_u32 v18, v18, 7, s15
	v_add_u32_e32 v19, v18, v16
	v_lshrrev_b32_e32 v20, 4, v19
	v_and_b32_e32 v20, 0x78, v20
	v_xor_b32_e32 v163, v20, v19
	v_add_u32_e32 v19, v18, v6
	v_lshrrev_b32_e32 v20, 4, v19
	v_or_b32_e32 v9, v9, v10
	v_and_b32_e32 v20, 0x78, v20
	v_lshl_add_u32 v9, v9, 7, s15
	v_xor_b32_e32 v164, v20, v19
	v_add_u32_e32 v19, v9, v16
	v_lshrrev_b32_e32 v20, 4, v19
	v_and_b32_e32 v20, 0x78, v20
	v_xor_b32_e32 v165, v20, v19
	v_add_u32_e32 v19, v9, v6
	v_lshlrev_b32_e32 v156, 5, v15
	v_lshrrev_b32_e32 v20, 4, v19
	v_or_b32_e32 v15, v156, v155
	v_and_b32_e32 v20, 0x78, v20
	v_lshl_add_u32 v15, v15, 7, s26
	v_xor_b32_e32 v166, v20, v19
	v_add_u32_e32 v19, v15, v16
	v_lshrrev_b32_e32 v20, 4, v19
	v_and_b32_e32 v20, 0x78, v20
	v_xor_b32_e32 v167, v20, v19
	v_add_u32_e32 v19, v15, v6
	v_lshrrev_b32_e32 v20, 4, v19
	v_and_b32_e32 v20, 0x78, v20
	v_xor_b32_e32 v168, v20, v19
	v_or_b32_e32 v19, v10, v156
	v_lshl_add_u32 v19, v19, 7, s26
	v_add_u32_e32 v20, v19, v16
	v_lshrrev_b32_e32 v21, 4, v20
	v_and_b32_e32 v21, 0x78, v21
	v_xor_b32_e32 v169, v21, v20
	v_add_u32_e32 v20, v19, v6
	v_lshrrev_b32_e32 v21, 4, v20
	v_and_b32_e32 v21, 0x78, v21
	v_xor_b32_e32 v170, v21, v20
	v_or_b32_e32 v20, 64, v156
	v_or_b32_e32 v21, v20, v155
	v_lshl_add_u32 v21, v21, 7, s26
	v_add_u32_e32 v22, v21, v16
	v_lshrrev_b32_e32 v23, 4, v22
	v_and_b32_e32 v23, 0x78, v23
	v_xor_b32_e32 v171, v23, v22
	v_add_u32_e32 v22, v21, v6
	v_or_b32_e32 v10, v20, v10
	v_lshrrev_b32_e32 v23, 4, v22
	v_lshl_add_u32 v10, v10, 7, s26
	v_and_b32_e32 v23, 0x78, v23
	v_add_u32_e32 v20, v10, v16
	v_xor_b32_e32 v172, v23, v22
	v_lshrrev_b32_e32 v22, 4, v20
	v_and_b32_e32 v22, 0x78, v22
	v_add_u32_e32 v6, v10, v6
	v_xor_b32_e32 v173, v22, v20
	v_lshrrev_b32_e32 v20, 4, v6
	v_lshlrev_b32_e32 v175, 1, v2
	v_or_b32_e32 v2, 64, v16
	v_and_b32_e32 v20, 0x78, v20
	v_lshlrev_b32_e32 v176, 1, v4
	v_add_u32_e32 v4, v13, v2
	v_xor_b32_e32 v174, v20, v6
	v_lshrrev_b32_e32 v6, 4, v4
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v179, v6, v4
	v_or_b32_e32 v4, 0x60, v16
	v_add_u32_e32 v6, v13, v4
	v_lshlrev_b32_e32 v177, 1, v8
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v180, v8, v6
	v_add_u32_e32 v6, v17, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v181, v8, v6
	v_add_u32_e32 v6, v17, v4
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v182, v8, v6
	v_add_u32_e32 v6, v18, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v183, v8, v6
	v_add_u32_e32 v6, v18, v4
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v184, v8, v6
	v_add_u32_e32 v6, v9, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v185, v8, v6
	v_add_u32_e32 v6, v9, v4
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v186, v8, v6
	v_add_u32_e32 v6, v15, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v187, v8, v6
	v_add_u32_e32 v6, v15, v4
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v188, v8, v6
	v_add_u32_e32 v6, v19, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v189, v8, v6
	v_add_u32_e32 v6, v19, v4
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v190, v8, v6
	v_add_u32_e32 v6, v21, v2
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v191, v8, v6
	v_add_u32_e32 v6, v21, v4
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v2, v10, v2
	v_xor_b32_e32 v192, v8, v6
	v_lshrrev_b32_e32 v6, 4, v2
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v193, v6, v2
	v_add_u32_e32 v2, v10, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v194, v4, v2
	v_and_b32_e32 v2, 0x70, v14
	v_add_u32_e32 v4, s26, v2
	v_add_u32_e32 v9, v4, v7
	v_or_b32_e32 v6, 8, v2
	v_lshrrev_b32_e32 v10, 4, v9
	v_add_u32_e32 v8, s26, v6
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v195, v10, v9
	v_add_u32_e32 v9, v8, v7
	v_lshrrev_b32_e32 v10, 4, v9
	v_and_b32_e32 v10, 0x78, v10
	v_add_u32_e32 v4, v4, v11
	v_xor_b32_e32 v196, v10, v9
	v_lshrrev_b32_e32 v9, 4, v4
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v197, v9, v4
	v_add_u32_e32 v4, v8, v11
	v_lshrrev_b32_e32 v8, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_add_u32_e32 v2, s15, v2
	v_xor_b32_e32 v198, v8, v4
	v_add_u32_e32 v4, s15, v6
	v_add_u32_e32 v6, v2, v7
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v199, v8, v6
	v_add_u32_e32 v6, v4, v7
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v200, v7, v6
	v_add_u32_e32 v6, v2, v11
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v201, v7, v6
	v_add_u32_e32 v6, v4, v11
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v202, v7, v6
	v_add_u32_e32 v6, v2, v3
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_add_u32_e32 v3, v4, v3
	v_xor_b32_e32 v203, v7, v6
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_add_u32_e32 v2, v2, v5
	v_xor_b32_e32 v204, v6, v3
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	s_lshl_b32 s2, s20, 8
	s_lshl_b32 s14, s14, 9
	v_xor_b32_e32 v205, v3, v2
	v_add_u32_e32 v2, v4, v5
	v_lshrrev_b32_e32 v3, 4, v2
	s_add_u32 s20, s0, 0x80
	v_and_b32_e32 v3, 0x78, v3
	s_addc_u32 s21, s1, 0
	v_xor_b32_e32 v206, v3, v2
	s_add_u32 s22, s12, 0x80
	v_mov_b32_e32 v2, 0
	v_mov_b64_e32 v[76:77], 0
	v_lshlrev_b32_e32 v178, 1, v12
	s_addc_u32 s23, s13, 0
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
	v_mov_b64_e32 v[68:69], v[76:77]
	v_mov_b64_e32 v[88:89], v[76:77]
	v_mov_b64_e32 v[84:85], v[76:77]
	v_mov_b64_e32 v[80:81], v[76:77]
	v_mov_b64_e32 v[72:73], v[76:77]
	s_branch .LBB0_4
.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v136, s24, v136, v136
	v_bfi_b32 v134, s24, v134, v134
	v_bfi_b32 v132, s24, v132, v132
	v_bfi_b32 v130, s24, v130, v130
	v_bfi_b32 v128, s24, v128, v128
	v_bfi_b32 v126, s24, v126, v126
	v_bfi_b32 v124, s24, v124, v124
	v_bfi_b32 v122, s24, v122, v122
	v_bfi_b32 v152, s24, v152, v152
	v_bfi_b32 v150, s24, v150, v150
	v_bfi_b32 v148, s24, v148, v148
	v_bfi_b32 v146, s24, v146, v146
	v_bfi_b32 v144, s24, v144, v144
	v_bfi_b32 v142, s24, v142, v142
	v_bfi_b32 v140, s24, v140, v140
	v_bfi_b32 v138, s24, v138, v138
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[62:65], v[152:153], v[136:137], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[152:153], v[132:133], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[148:149], v[136:137], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[148:149], v[132:133], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[152:153], v[128:129], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[152:153], v[124:125], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[148:149], v[128:129], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[148:149], v[124:125], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[144:145], v[136:137], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[144:145], v[132:133], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[140:141], v[136:137], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[140:141], v[132:133], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[144:145], v[128:129], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[144:145], v[124:125], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[140:141], v[128:129], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[140:141], v[124:125], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[150:151], v[134:135], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[150:151], v[130:131], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[146:147], v[134:135], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[146:147], v[130:131], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[150:151], v[126:127], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[150:151], v[122:123], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[146:147], v[126:127], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[146:147], v[122:123], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[142:143], v[134:135], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[142:143], v[130:131], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[138:139], v[134:135], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[138:139], v[130:131], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[142:143], v[126:127], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[142:143], v[122:123], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[138:139], v[126:127], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[138:139], v[122:123], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s18, s18, 0x80
	s_addc_u32 s19, s19, 0
	s_cmpk_eq_i32 s18, 0x4000
	s_cbranch_scc1 .LBB0_10
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_cmpk_lg_i32 s18, 0x3f80
	s_cselect_b64 s[12:13], -1, 0
	s_cmpk_eq_i32 s18, 0x3f80
	s_cbranch_scc1 .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s0, s20, s18
	s_addc_u32 s1, s21, s19
	buffer_load_dwordx4 v[66:69], v159, s[0:3], 0 offen
	buffer_load_dwordx4 v[74:77], v158, s[0:3], 0 offen
	; sched_barrier mask(0x00000000)
.LBB0_6:                                ;   in Loop: Header=BB0_4 Depth=1
	;;#ASMSTART
	ds_read_b64 v[104:105], v157

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[102:103], v160

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[100:101], v161

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[98:99], v162

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[96:97], v163

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[94:95], v164

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[92:93], v165

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[90:91], v166

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[120:121], v167

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[118:119], v168

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[116:117], v169

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[114:115], v170

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[110:111], v171

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[108:109], v172

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[106:107], v173

	;;#ASMEND
	v_cndmask_b32_e64 v112, 0, 1, s[12:13]
	v_cmp_ne_u32_e64 s[0:1], 1, v112
	;;#ASMSTART
	ds_read_b64 v[112:113], v174

	;;#ASMEND
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s12, s22, s18
	s_addc_u32 s13, s23, s19
	s_mov_b32 s15, s3
	buffer_load_dwordx4 v[70:73], v175, s[12:15], 0 offen
	buffer_load_dwordx4 v[78:81], v176, s[12:15], 0 offen
	buffer_load_dwordx4 v[82:85], v177, s[12:15], 0 offen
	buffer_load_dwordx4 v[86:89], v178, s[12:15], 0 offen
	; sched_barrier mask(0x00000000)
.LBB0_8:                                ;   in Loop: Header=BB0_4 Depth=1
	;;#ASMSTART
	ds_read_b64 v[136:137], v179

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v180

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v181

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v182

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[128:129], v183

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[126:127], v184

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v185

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[122:123], v186

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v187

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v188

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v189

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[146:147], v190

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v191

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v193

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v194

	;;#ASMEND
	v_bfi_b32 v104, s24, v104, v104
	v_bfi_b32 v102, s24, v102, v102
	v_bfi_b32 v100, s24, v100, v100
	v_bfi_b32 v98, s24, v98, v98
	v_bfi_b32 v96, s24, v96, v96
	v_bfi_b32 v94, s24, v94, v94
	v_bfi_b32 v92, s24, v92, v92
	v_bfi_b32 v90, s24, v90, v90
	v_bfi_b32 v120, s24, v120, v120
	v_bfi_b32 v118, s24, v118, v118
	v_bfi_b32 v116, s24, v116, v116
	v_bfi_b32 v114, s24, v114, v114
	v_bfi_b32 v110, s24, v110, v110
	v_bfi_b32 v108, s24, v108, v108
	v_bfi_b32 v106, s24, v106, v106
	v_bfi_b32 v112, s24, v112, v112
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[62:65], v[120:121], v[104:105], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[120:121], v[100:101], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[116:117], v[104:105], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[116:117], v[100:101], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[120:121], v[96:97], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[120:121], v[92:93], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[116:117], v[96:97], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[116:117], v[92:93], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[110:111], v[104:105], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[110:111], v[100:101], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[106:107], v[104:105], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[106:107], v[100:101], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[110:111], v[96:97], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[110:111], v[92:93], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[106:107], v[96:97], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[106:107], v[92:93], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[118:119], v[102:103], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[118:119], v[98:99], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[114:115], v[102:103], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[114:115], v[98:99], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[46:49], v[118:119], v[94:95], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[118:119], v[90:91], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[114:115], v[94:95], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[114:115], v[90:91], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[108:109], v[102:103], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[108:109], v[98:99], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[112:113], v[102:103], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[18:21], v[112:113], v[98:99], v[18:21]
	v_mfma_f32_16x16x16_bf16 v[14:17], v[108:109], v[94:95], v[14:17]
	v_mfma_f32_16x16x16_bf16 v[10:13], v[108:109], v[90:91], v[10:13]
	v_mfma_f32_16x16x16_bf16 v[6:9], v[112:113], v[94:95], v[6:9]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[112:113], v[90:91], v[2:5]
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
	ds_write_b64 v195, v[66:67]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v196, v[68:69]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v197, v[74:75]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v198, v[76:77]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v199, v[70:71]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v200, v[72:73]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v201, v[78:79]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v202, v[80:81]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v203, v[82:83]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v204, v[84:85]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v205, v[86:87]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v206, v[88:89]

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
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v62
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.13:
	v_bfe_u32 v0, v62, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v62, v0, s2
; %bb.14:                               ; %Flow1375
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_16
; %bb.15:
	v_mov_b32_e32 v0, 0
	s_waitcnt vmcnt(1)
	v_or_b32_e32 v66, 0x10000, v62
	v_cmp_eq_u32_sdwa vcc, v62, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v66, v62, vcc
.LBB0_16:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v62, 0x7f800000, v63
	v_cmp_ne_u32_e32 vcc, s0, v62
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_bfe_u32 v62, v63, 16, 1
	s_movk_i32 s2, 0x7fff
	s_waitcnt vmcnt(1)
	v_add3_u32 v68, v63, v62, s2
.LBB0_18:                               ; %Flow1374
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_mov_b32_e32 v62, 0
	s_waitcnt vmcnt(1)
	v_or_b32_e32 v66, 0x10000, v63
	v_cmp_eq_u32_sdwa vcc, v63, v62 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v66, v63, vcc
.LBB0_20:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v62, 0x7f800000, v64
	v_cmp_ne_u32_e32 vcc, s0, v62
                                        ; implicit-def: $vgpr69
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_bfe_u32 v62, v64, 16, 1
	s_movk_i32 s2, 0x7fff
	s_waitcnt vmcnt(1)
	v_add3_u32 v69, v64, v62, s2
.LBB0_22:                               ; %Flow1373
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_mov_b32_e32 v62, 0
	v_or_b32_e32 v63, 0x10000, v64
	v_cmp_eq_u32_sdwa vcc, v64, v62 src0_sel:WORD_0 src1_sel:DWORD
	s_waitcnt vmcnt(1)
	s_nop 0
	v_cndmask_b32_e32 v69, v63, v64, vcc
.LBB0_24:                               ; %_ZL16__float2bfloat16f.exit.i.i46.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v62, 0x7f800000, v65
	v_cmp_ne_u32_e32 vcc, s0, v62
                                        ; implicit-def: $vgpr73
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_bfe_u32 v62, v65, 16, 1
	s_movk_i32 s2, 0x7fff
	s_waitcnt vmcnt(3)
	v_add3_u32 v73, v65, v62, s2
.LBB0_26:                               ; %Flow1372
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_mov_b32_e32 v62, 0
	v_or_b32_e32 v63, 0x10000, v65
	v_cmp_eq_u32_sdwa vcc, v65, v62 src0_sel:WORD_0 src1_sel:DWORD
	s_waitcnt vmcnt(3)
	s_nop 0
	v_cndmask_b32_e32 v73, v63, v65, vcc
.LBB0_28:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mul_i32 s0, s8, s6
	s_mul_i32 s0, s0, s10
	s_mul_i32 s0, s0, s16
	s_waitcnt vmcnt(3)
	v_lshl_or_b32 v71, v1, 5, s11
	s_lshl_b32 s12, s0, 1
	s_mul_i32 s2, s16, s9
	v_mul_lo_u32 v1, s17, v71
	v_mad_u64_u32 v[62:63], s[0:1], s16, v71, 0
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v74, 16, v0
	v_lshl_or_b32 v0, s7, 7, v156
	v_add3_u32 v63, v63, s2, v1
	v_mul_lo_u32 v72, v155, s16
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[66:67], v[62:63], 1, s[4:5]
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v73, 0xffff0000, v73
	v_add_lshl_u32 v70, v72, v154, 1
	v_lshl_add_u64 v[62:63], v[0:1], 1, v[66:67]
	v_mov_b32_e32 v64, s12
	v_mov_b32_e32 v65, 0x20000
	v_or_b32_sdwa v69, v73, v69 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v68, v68, s0, v74
	s_mov_b64 s[2:3], exec
.LBB0_29:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v62
	v_readfirstlane_b32 s9, v63
	v_readfirstlane_b32 s10, v64
	v_readfirstlane_b32 s11, v65
	v_cmp_eq_u64_e32 vcc, s[8:9], v[62:63]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[64:65]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[68:69], v70, s[8:11], 0 offen
                                        ; implicit-def: $vgpr68_vgpr69
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_29
; %bb.30:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v58
	v_cmp_ne_u32_e32 vcc, s0, v68
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.31:
	v_bfe_u32 v68, v58, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v58, v68, s2
; %bb.32:                               ; %Flow1371
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.33:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v69, 0x10000, v58
	v_cmp_eq_u32_sdwa vcc, v58, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v69, v58, vcc
; %bb.34:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v58, 0x7f800000, v59
	v_cmp_ne_u32_e32 vcc, s0, v58
                                        ; implicit-def: $vgpr58
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.35:
	v_bfe_u32 v58, v59, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v58, v59, v58, s2
; %bb.36:                               ; %Flow1370
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.37:
	v_mov_b32_e32 v58, 0
	v_or_b32_e32 v69, 0x10000, v59
	v_cmp_eq_u32_sdwa vcc, v59, v58 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v58, v69, v59, vcc
; %bb.38:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v59, 0x7f800000, v60
	v_cmp_ne_u32_e32 vcc, s0, v59
                                        ; implicit-def: $vgpr59
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.39:
	v_bfe_u32 v59, v60, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v59, v60, v59, s2
; %bb.40:                               ; %Flow1369
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.41:
	v_mov_b32_e32 v59, 0
	v_or_b32_e32 v69, 0x10000, v60
	v_cmp_eq_u32_sdwa vcc, v60, v59 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v59, v69, v60, vcc
; %bb.42:                               ; %_ZL16__float2bfloat16f.exit.i.i46.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v61
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr69
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.43:
	v_bfe_u32 v60, v61, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v69, v61, v60, s2
; %bb.44:                               ; %Flow1368
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.45:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v69, 0x10000, v61
	v_cmp_eq_u32_sdwa vcc, v61, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v61, vcc
; %bb.46:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v68, 16, v68
	v_lshl_add_u32 v61, s16, 4, v72
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v69, 0xffff0000, v69
	v_add_lshl_u32 v60, v61, v154, 1
	v_or_b32_sdwa v59, v69, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v58, v58, s0, v68
	s_mov_b64 s[2:3], exec
.LBB0_47:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v62
	v_readfirstlane_b32 s9, v63
	v_readfirstlane_b32 s10, v64
	v_readfirstlane_b32 s11, v65
	v_cmp_eq_u64_e32 vcc, s[8:9], v[62:63]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[64:65]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[58:59], v60, s[8:11], 0 offen
                                        ; implicit-def: $vgpr58_vgpr59
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_47
; %bb.48:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v58, 0x7f800000, v54
	v_cmp_ne_u32_e32 vcc, s0, v58
                                        ; implicit-def: $vgpr58
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.49:
	v_bfe_u32 v58, v54, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v58, v54, v58, s2
; %bb.50:                               ; %Flow1367
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.51:
	v_mov_b32_e32 v58, 0
	v_or_b32_e32 v59, 0x10000, v54
	v_cmp_eq_u32_sdwa vcc, v54, v58 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v58, v59, v54, vcc
; %bb.52:                               ; %_ZL16__float2bfloat16f.exit.i.i.167.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v54, 0x7f800000, v55
	v_cmp_ne_u32_e32 vcc, s0, v54
                                        ; implicit-def: $vgpr54
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.53:
	v_bfe_u32 v54, v55, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v54, v55, v54, s2
; %bb.54:                               ; %Flow1366
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.55:
	v_mov_b32_e32 v54, 0
	v_or_b32_e32 v59, 0x10000, v55
	v_cmp_eq_u32_sdwa vcc, v55, v54 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v54, v59, v55, vcc
; %bb.56:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.174.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v55, 0x7f800000, v56
	v_cmp_ne_u32_e32 vcc, s0, v55
                                        ; implicit-def: $vgpr55
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.57:
	v_bfe_u32 v55, v56, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v55, v56, v55, s2
; %bb.58:                               ; %Flow1365
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.59:
	v_mov_b32_e32 v55, 0
	v_or_b32_e32 v59, 0x10000, v56
	v_cmp_eq_u32_sdwa vcc, v56, v55 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v55, v59, v56, vcc
; %bb.60:                               ; %_ZL16__float2bfloat16f.exit.i.i46.178.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v57
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr59
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.61:
	v_bfe_u32 v56, v57, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v59, v57, v56, s2
; %bb.62:                               ; %Flow1364
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.63:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v59, 0x10000, v57
	v_cmp_eq_u32_sdwa vcc, v57, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v57, vcc
; %bb.64:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.189.i.i
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v58, 16, v58
	v_or_b32_e32 v57, 16, v154
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v59, 0xffff0000, v59
	v_add_lshl_u32 v56, v72, v57, 1
	v_or_b32_sdwa v55, v59, v55 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v54, v54, s0, v58
	s_mov_b64 s[2:3], exec
.LBB0_65:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v62
	v_readfirstlane_b32 s9, v63
	v_readfirstlane_b32 s10, v64
	v_readfirstlane_b32 s11, v65
	v_cmp_eq_u64_e32 vcc, s[8:9], v[62:63]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[64:65]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[54:55], v56, s[8:11], 0 offen
                                        ; implicit-def: $vgpr54_vgpr55
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_65
; %bb.66:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v54, 0x7f800000, v50
	v_cmp_ne_u32_e32 vcc, s0, v54
                                        ; implicit-def: $vgpr55
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.67:
	v_bfe_u32 v54, v50, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v55, v50, v54, s2
; %bb.68:                               ; %Flow1363
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.69:
	v_mov_b32_e32 v54, 0
	v_or_b32_e32 v55, 0x10000, v50
	v_cmp_eq_u32_sdwa vcc, v50, v54 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v50, vcc
; %bb.70:                               ; %_ZL16__float2bfloat16f.exit.i.i.1.1.i.i
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
; %bb.72:                               ; %Flow1362
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.73:
	v_mov_b32_e32 v50, 0
	v_or_b32_e32 v54, 0x10000, v51
	v_cmp_eq_u32_sdwa vcc, v51, v50 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v50, v54, v51, vcc
; %bb.74:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v51, 0x7f800000, v52
	v_cmp_ne_u32_e32 vcc, s0, v51
                                        ; implicit-def: $vgpr51
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.75:
	v_bfe_u32 v51, v52, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v52, v51, s2
; %bb.76:                               ; %Flow1361
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.77:
	v_mov_b32_e32 v51, 0
	v_or_b32_e32 v54, 0x10000, v52
	v_cmp_eq_u32_sdwa vcc, v52, v51 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v51, v54, v52, vcc
; %bb.78:                               ; %_ZL16__float2bfloat16f.exit.i.i46.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v53
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.79:
	v_bfe_u32 v52, v53, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v53, v52, s2
; %bb.80:                               ; %Flow1360
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.81:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v54, 0x10000, v53
	v_cmp_eq_u32_sdwa vcc, v53, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v54, v53, vcc
; %bb.82:                               ; %_ZL16store_transposedITkN7kittens5ducks2rt3allENS0_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS0_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS6_EEEvRKT0_RKT_RKT1_.exit
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v53, 16, v55
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v52, 0xffff0000, v52
	v_add_lshl_u32 v54, v61, v57, 1
	v_or_b32_sdwa v51, v52, v51 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v50, v50, s0, v53
	s_mov_b64 s[2:3], exec
.LBB0_83:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v62
	v_readfirstlane_b32 s9, v63
	v_readfirstlane_b32 s10, v64
	v_readfirstlane_b32 s11, v65
	v_cmp_eq_u64_e32 vcc, s[8:9], v[62:63]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[64:65]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[50:51], v54, s[8:11], 0 offen
                                        ; implicit-def: $vgpr62_vgpr63_vgpr64_vgpr65
                                        ; implicit-def: $vgpr50_vgpr51
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_83
; %bb.84:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v50, 0x7f800000, v46
	v_cmp_ne_u32_e32 vcc, s0, v50
                                        ; implicit-def: $vgpr50
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.85:
	v_bfe_u32 v50, v46, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v50, v46, v50, s2
; %bb.86:                               ; %Flow1359
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.87:
	v_mov_b32_e32 v50, 0
	v_or_b32_e32 v51, 0x10000, v46
	v_cmp_eq_u32_sdwa vcc, v46, v50 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v50, v51, v46, vcc
; %bb.88:                               ; %_ZL16__float2bfloat16f.exit.i.i.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v46, 0x7f800000, v47
	v_cmp_ne_u32_e32 vcc, s0, v46
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.89:
	v_bfe_u32 v46, v47, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v47, v46, s2
; %bb.90:                               ; %Flow1358
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.91:
	v_mov_b32_e32 v46, 0
	v_or_b32_e32 v51, 0x10000, v47
	v_cmp_eq_u32_sdwa vcc, v47, v46 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v51, v47, vcc
; %bb.92:                               ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v46, 0x7f800000, v48
	v_cmp_ne_u32_e32 vcc, s0, v46
                                        ; implicit-def: $vgpr53
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.93:
	v_bfe_u32 v46, v48, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v53, v48, v46, s2
; %bb.94:                               ; %Flow1357
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.95:
	v_mov_b32_e32 v46, 0
	v_or_b32_e32 v47, 0x10000, v48
	v_cmp_eq_u32_sdwa vcc, v48, v46 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v53, v47, v48, vcc
; %bb.96:                               ; %_ZL16__float2bfloat16f.exit.i.i46.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v46, 0x7f800000, v49
	v_cmp_ne_u32_e32 vcc, s0, v46
                                        ; implicit-def: $vgpr55
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.97:
	v_bfe_u32 v46, v49, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v55, v49, v46, s2
; %bb.98:                               ; %Flow1356
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.99:
	v_mov_b32_e32 v46, 0
	v_or_b32_e32 v47, 0x10000, v49
	v_cmp_eq_u32_sdwa vcc, v49, v46 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v55, v47, v49, vcc
; %bb.100:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.i.i.1
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v46, 0x80, v71
	v_ashrrev_i32_e32 v47, 31, v71
	v_mul_lo_u32 v48, s17, v46
	v_mul_lo_u32 v49, s16, v47
	v_mad_u64_u32 v[46:47], s[0:1], s16, v46, 0
	v_add3_u32 v47, v47, v49, v48
	v_lshrrev_b32_e32 v57, 16, v50
	v_lshl_add_u64 v[50:51], v[46:47], 1, s[4:5]
	v_lshl_add_u64 v[46:47], v[0:1], 1, v[50:51]
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v1, 0xffff0000, v55
	v_mov_b32_e32 v48, s12
	v_mov_b32_e32 v49, 0x20000
	v_or_b32_sdwa v53, v1, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v52, v52, s0, v57
	s_mov_b64 s[2:3], exec
.LBB0_101:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v46
	v_readfirstlane_b32 s5, v47
	v_readfirstlane_b32 s6, v48
	v_readfirstlane_b32 s7, v49
	v_cmp_eq_u64_e32 vcc, s[4:5], v[46:47]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[48:49]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[52:53], v70, s[4:7], 0 offen
                                        ; implicit-def: $vgpr52_vgpr53
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_101
; %bb.102:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v42
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.103:
	v_bfe_u32 v1, v42, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v42, v1, s2
; %bb.104:                              ; %Flow1355
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.105:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v52, 0x10000, v42
	v_cmp_eq_u32_sdwa vcc, v42, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v52, v42, vcc
; %bb.106:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v42, 0x7f800000, v43
	v_cmp_ne_u32_e32 vcc, s0, v42
                                        ; implicit-def: $vgpr42
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.107:
	v_bfe_u32 v42, v43, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v42, v43, v42, s2
; %bb.108:                              ; %Flow1354
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.109:
	v_mov_b32_e32 v42, 0
	v_or_b32_e32 v52, 0x10000, v43
	v_cmp_eq_u32_sdwa vcc, v43, v42 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v42, v52, v43, vcc
; %bb.110:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v43, 0x7f800000, v44
	v_cmp_ne_u32_e32 vcc, s0, v43
                                        ; implicit-def: $vgpr43
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.111:
	v_bfe_u32 v43, v44, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v43, v44, v43, s2
; %bb.112:                              ; %Flow1353
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.113:
	v_mov_b32_e32 v43, 0
	v_or_b32_e32 v52, 0x10000, v44
	v_cmp_eq_u32_sdwa vcc, v44, v43 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v43, v52, v44, vcc
; %bb.114:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v45
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr44
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.115:
	v_bfe_u32 v44, v45, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v44, v45, v44, s2
; %bb.116:                              ; %Flow1352
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.117:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v52, 0x10000, v45
	v_cmp_eq_u32_sdwa vcc, v45, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v44, v52, v45, vcc
; %bb.118:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v1, 16, v1
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v44, 0xffff0000, v44
	v_or_b32_sdwa v43, v44, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v42, v42, s0, v1
	s_mov_b64 s[2:3], exec
.LBB0_119:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v46
	v_readfirstlane_b32 s5, v47
	v_readfirstlane_b32 s6, v48
	v_readfirstlane_b32 s7, v49
	v_cmp_eq_u64_e32 vcc, s[4:5], v[46:47]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[48:49]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[42:43], v60, s[4:7], 0 offen
                                        ; implicit-def: $vgpr42_vgpr43
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_119
; %bb.120:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v38
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.121:
	v_bfe_u32 v1, v38, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v38, v1, s2
; %bb.122:                              ; %Flow1351
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.123:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v42, 0x10000, v38
	v_cmp_eq_u32_sdwa vcc, v38, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v42, v38, vcc
; %bb.124:                              ; %_ZL16__float2bfloat16f.exit.i.i.167.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v38, 0x7f800000, v39
	v_cmp_ne_u32_e32 vcc, s0, v38
                                        ; implicit-def: $vgpr38
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.125:
	v_bfe_u32 v38, v39, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v38, v39, v38, s2
; %bb.126:                              ; %Flow1350
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.127:
	v_mov_b32_e32 v38, 0
	v_or_b32_e32 v42, 0x10000, v39
	v_cmp_eq_u32_sdwa vcc, v39, v38 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v38, v42, v39, vcc
; %bb.128:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.174.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v39, 0x7f800000, v40
	v_cmp_ne_u32_e32 vcc, s0, v39
                                        ; implicit-def: $vgpr39
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.129:
	v_bfe_u32 v39, v40, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v39, v40, v39, s2
; %bb.130:                              ; %Flow1349
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.131:
	v_mov_b32_e32 v39, 0
	v_or_b32_e32 v42, 0x10000, v40
	v_cmp_eq_u32_sdwa vcc, v40, v39 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v39, v42, v40, vcc
; %bb.132:                              ; %_ZL16__float2bfloat16f.exit.i.i46.178.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v41
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.133:
	v_bfe_u32 v40, v41, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v41, v40, s2
; %bb.134:                              ; %Flow1348
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.135:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v42, 0x10000, v41
	v_cmp_eq_u32_sdwa vcc, v41, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v42, v41, vcc
; %bb.136:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.189.i.i.1
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v1, 16, v1
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v40, 0xffff0000, v40
	v_or_b32_sdwa v39, v40, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v38, v38, s0, v1
	s_mov_b64 s[2:3], exec
.LBB0_137:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v46
	v_readfirstlane_b32 s5, v47
	v_readfirstlane_b32 s6, v48
	v_readfirstlane_b32 s7, v49
	v_cmp_eq_u64_e32 vcc, s[4:5], v[46:47]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[48:49]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[38:39], v56, s[4:7], 0 offen
                                        ; implicit-def: $vgpr38_vgpr39
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_137
; %bb.138:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v34
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.139:
	v_bfe_u32 v1, v34, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v34, v1, s2
; %bb.140:                              ; %Flow1347
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.141:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v38, 0x10000, v34
	v_cmp_eq_u32_sdwa vcc, v34, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v38, v34, vcc
; %bb.142:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v34, 0x7f800000, v35
	v_cmp_ne_u32_e32 vcc, s0, v34
                                        ; implicit-def: $vgpr34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.143:
	v_bfe_u32 v34, v35, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v34, v35, v34, s2
; %bb.144:                              ; %Flow1346
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.145:
	v_mov_b32_e32 v34, 0
	v_or_b32_e32 v38, 0x10000, v35
	v_cmp_eq_u32_sdwa vcc, v35, v34 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v34, v38, v35, vcc
; %bb.146:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v35, 0x7f800000, v36
	v_cmp_ne_u32_e32 vcc, s0, v35
                                        ; implicit-def: $vgpr35
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.147:
	v_bfe_u32 v35, v36, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v36, v35, s2
; %bb.148:                              ; %Flow1345
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.149:
	v_mov_b32_e32 v35, 0
	v_or_b32_e32 v38, 0x10000, v36
	v_cmp_eq_u32_sdwa vcc, v36, v35 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v35, v38, v36, vcc
; %bb.150:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.1.i.i.1
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v37
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.151:
	v_bfe_u32 v36, v37, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v37, v36, s2
; %bb.152:                              ; %Flow1344
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.153:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v38, 0x10000, v37
	v_cmp_eq_u32_sdwa vcc, v37, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v38, v37, vcc
; %bb.154:                              ; %_ZL16store_transposedITkN7kittens5ducks2rt3allENS0_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS0_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS6_EEEvRKT0_RKT_RKT1_.exit.1
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v1, 16, v1
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v36, 0xffff0000, v36
	v_or_b32_sdwa v35, v36, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v34, v34, s0, v1
	s_mov_b64 s[2:3], exec
.LBB0_155:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v46
	v_readfirstlane_b32 s5, v47
	v_readfirstlane_b32 s6, v48
	v_readfirstlane_b32 s7, v49
	v_cmp_eq_u64_e32 vcc, s[4:5], v[46:47]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[48:49]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[34:35], v54, s[4:7], 0 offen
                                        ; implicit-def: $vgpr46_vgpr47_vgpr48_vgpr49
                                        ; implicit-def: $vgpr34_vgpr35
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_155
; %bb.156:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v30
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.157:
	v_bfe_u32 v1, v30, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v30, v1, s2
; %bb.158:                              ; %Flow1343
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.159:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v34, 0x10000, v30
	v_cmp_eq_u32_sdwa vcc, v30, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v34, v30, vcc
; %bb.160:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v30, 0x7f800000, v31
	v_cmp_ne_u32_e32 vcc, s0, v30
                                        ; implicit-def: $vgpr34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.161:
	v_bfe_u32 v30, v31, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v34, v31, v30, s2
; %bb.162:                              ; %Flow1342
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.163:
	v_mov_b32_e32 v30, 0
	v_or_b32_e32 v34, 0x10000, v31
	v_cmp_eq_u32_sdwa vcc, v31, v30 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v34, v34, v31, vcc
; %bb.164:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v30, 0x7f800000, v32
	v_cmp_ne_u32_e32 vcc, s0, v30
                                        ; implicit-def: $vgpr35
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.165:
	v_bfe_u32 v30, v32, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v32, v30, s2
; %bb.166:                              ; %Flow1341
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.167:
	v_mov_b32_e32 v30, 0
	v_or_b32_e32 v31, 0x10000, v32
	v_cmp_eq_u32_sdwa vcc, v32, v30 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v35, v31, v32, vcc
; %bb.168:                              ; %_ZL16__float2bfloat16f.exit.i.i46.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v30, 0x7f800000, v33
	v_cmp_ne_u32_e32 vcc, s0, v30
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.169:
	v_bfe_u32 v30, v33, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v33, v30, s2
; %bb.170:                              ; %Flow1340
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.171:
	v_mov_b32_e32 v30, 0
	v_or_b32_e32 v31, 0x10000, v33
	v_cmp_eq_u32_sdwa vcc, v33, v30 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v31, v33, vcc
; %bb.172:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.i.i.2
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v0, 64, v0
	v_lshrrev_b32_e32 v37, 16, v1
	v_ashrrev_i32_e32 v1, 31, v0
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v36, 0xffff0000, v36
	v_lshl_add_u64 v[30:31], v[0:1], 1, v[66:67]
	v_mov_b32_e32 v32, s12
	v_mov_b32_e32 v33, 0x20000
	v_or_b32_sdwa v35, v36, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v34, v34, s0, v37
	s_mov_b64 s[2:3], exec
.LBB0_173:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v30
	v_readfirstlane_b32 s5, v31
	v_readfirstlane_b32 s6, v32
	v_readfirstlane_b32 s7, v33
	v_cmp_eq_u64_e32 vcc, s[4:5], v[30:31]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[32:33]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[34:35], v70, s[4:7], 0 offen
                                        ; implicit-def: $vgpr34_vgpr35
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_173
; %bb.174:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v34, 0x7f800000, v26
	v_cmp_ne_u32_e32 vcc, s0, v34
                                        ; implicit-def: $vgpr34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.175:
	v_bfe_u32 v34, v26, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v34, v26, v34, s2
; %bb.176:                              ; %Flow1339
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.177:
	v_mov_b32_e32 v34, 0
	v_or_b32_e32 v35, 0x10000, v26
	v_cmp_eq_u32_sdwa vcc, v26, v34 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v34, v35, v26, vcc
; %bb.178:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v26, 0x7f800000, v27
	v_cmp_ne_u32_e32 vcc, s0, v26
                                        ; implicit-def: $vgpr26
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.179:
	v_bfe_u32 v26, v27, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v26, v27, v26, s2
; %bb.180:                              ; %Flow1338
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.181:
	v_mov_b32_e32 v26, 0
	v_or_b32_e32 v35, 0x10000, v27
	v_cmp_eq_u32_sdwa vcc, v27, v26 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v26, v35, v27, vcc
; %bb.182:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v27, 0x7f800000, v28
	v_cmp_ne_u32_e32 vcc, s0, v27
                                        ; implicit-def: $vgpr27
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.183:
	v_bfe_u32 v27, v28, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v27, v28, v27, s2
; %bb.184:                              ; %Flow1337
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.185:
	v_mov_b32_e32 v27, 0
	v_or_b32_e32 v35, 0x10000, v28
	v_cmp_eq_u32_sdwa vcc, v28, v27 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v27, v35, v28, vcc
; %bb.186:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v29
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr28
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.187:
	v_bfe_u32 v28, v29, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v28, v29, v28, s2
; %bb.188:                              ; %Flow1336
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.189:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v35, 0x10000, v29
	v_cmp_eq_u32_sdwa vcc, v29, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v28, v35, v29, vcc
; %bb.190:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v29, 16, v34
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v28, 0xffff0000, v28
	v_or_b32_sdwa v27, v28, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v26, v26, s0, v29
	s_mov_b64 s[2:3], exec
.LBB0_191:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v30
	v_readfirstlane_b32 s5, v31
	v_readfirstlane_b32 s6, v32
	v_readfirstlane_b32 s7, v33
	v_cmp_eq_u64_e32 vcc, s[4:5], v[30:31]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[32:33]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[26:27], v60, s[4:7], 0 offen
                                        ; implicit-def: $vgpr26_vgpr27
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_191
; %bb.192:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v26, 0x7f800000, v22
	v_cmp_ne_u32_e32 vcc, s0, v26
                                        ; implicit-def: $vgpr26
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.193:
	v_bfe_u32 v26, v22, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v26, v22, v26, s2
; %bb.194:                              ; %Flow1335
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.195:
	v_mov_b32_e32 v26, 0
	v_or_b32_e32 v27, 0x10000, v22
	v_cmp_eq_u32_sdwa vcc, v22, v26 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v26, v27, v22, vcc
; %bb.196:                              ; %_ZL16__float2bfloat16f.exit.i.i.167.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v22, 0x7f800000, v23
	v_cmp_ne_u32_e32 vcc, s0, v22
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.197:
	v_bfe_u32 v22, v23, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v22, v23, v22, s2
; %bb.198:                              ; %Flow1334
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.199:
	v_mov_b32_e32 v22, 0
	v_or_b32_e32 v27, 0x10000, v23
	v_cmp_eq_u32_sdwa vcc, v23, v22 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v22, v27, v23, vcc
; %bb.200:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.174.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v23, 0x7f800000, v24
	v_cmp_ne_u32_e32 vcc, s0, v23
                                        ; implicit-def: $vgpr23
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.201:
	v_bfe_u32 v23, v24, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v23, v24, v23, s2
; %bb.202:                              ; %Flow1333
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.203:
	v_mov_b32_e32 v23, 0
	v_or_b32_e32 v27, 0x10000, v24
	v_cmp_eq_u32_sdwa vcc, v24, v23 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v23, v27, v24, vcc
; %bb.204:                              ; %_ZL16__float2bfloat16f.exit.i.i46.178.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v25
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.205:
	v_bfe_u32 v24, v25, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v25, v24, s2
; %bb.206:                              ; %Flow1332
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.207:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v27, 0x10000, v25
	v_cmp_eq_u32_sdwa vcc, v25, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v27, v25, vcc
; %bb.208:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.189.i.i.2
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v25, 16, v26
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v24, 0xffff0000, v24
	v_or_b32_sdwa v23, v24, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v22, v22, s0, v25
	s_mov_b64 s[2:3], exec
.LBB0_209:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v30
	v_readfirstlane_b32 s5, v31
	v_readfirstlane_b32 s6, v32
	v_readfirstlane_b32 s7, v33
	v_cmp_eq_u64_e32 vcc, s[4:5], v[30:31]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[32:33]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[22:23], v56, s[4:7], 0 offen
                                        ; implicit-def: $vgpr22_vgpr23
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_209
; %bb.210:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v22, 0x7f800000, v18
	v_cmp_ne_u32_e32 vcc, s0, v22
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.211:
	v_bfe_u32 v22, v18, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v22, v18, v22, s2
; %bb.212:                              ; %Flow1331
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.213:
	v_mov_b32_e32 v22, 0
	v_or_b32_e32 v23, 0x10000, v18
	v_cmp_eq_u32_sdwa vcc, v18, v22 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v22, v23, v18, vcc
; %bb.214:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v18, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc, s0, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.215:
	v_bfe_u32 v18, v19, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v19, v18, s2
; %bb.216:                              ; %Flow1330
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.217:
	v_mov_b32_e32 v18, 0
	v_or_b32_e32 v23, 0x10000, v19
	v_cmp_eq_u32_sdwa vcc, v19, v18 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v18, v23, v19, vcc
; %bb.218:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v19, 0x7f800000, v20
	v_cmp_ne_u32_e32 vcc, s0, v19
                                        ; implicit-def: $vgpr19
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.219:
	v_bfe_u32 v19, v20, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v19, v20, v19, s2
; %bb.220:                              ; %Flow1329
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.221:
	v_mov_b32_e32 v19, 0
	v_or_b32_e32 v23, 0x10000, v20
	v_cmp_eq_u32_sdwa vcc, v20, v19 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v19, v23, v20, vcc
; %bb.222:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.1.i.i.2
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v21
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.223:
	v_bfe_u32 v20, v21, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v21, v20, s2
; %bb.224:                              ; %Flow1328
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.225:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v23, 0x10000, v21
	v_cmp_eq_u32_sdwa vcc, v21, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v23, v21, vcc
; %bb.226:                              ; %_ZL16store_transposedITkN7kittens5ducks2rt3allENS0_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS0_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS6_EEEvRKT0_RKT_RKT1_.exit.2
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v21, 16, v22
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v20, 0xffff0000, v20
	v_or_b32_sdwa v19, v20, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v18, v18, s0, v21
	s_mov_b64 s[2:3], exec
.LBB0_227:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v30
	v_readfirstlane_b32 s5, v31
	v_readfirstlane_b32 s6, v32
	v_readfirstlane_b32 s7, v33
	v_cmp_eq_u64_e32 vcc, s[4:5], v[30:31]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[32:33]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[18:19], v54, s[4:7], 0 offen
                                        ; implicit-def: $vgpr30_vgpr31_vgpr32_vgpr33
                                        ; implicit-def: $vgpr18_vgpr19
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_227
; %bb.228:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v18, 0x7f800000, v14
	v_cmp_ne_u32_e32 vcc, s0, v18
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.229:
	v_bfe_u32 v18, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v14, v18, s2
; %bb.230:                              ; %Flow1327
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.231:
	v_mov_b32_e32 v18, 0
	v_or_b32_e32 v19, 0x10000, v14
	v_cmp_eq_u32_sdwa vcc, v14, v18 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v18, v19, v14, vcc
; %bb.232:                              ; %_ZL16__float2bfloat16f.exit.i.i.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v14, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc, s0, v14
                                        ; implicit-def: $vgpr19
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.233:
	v_bfe_u32 v14, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v19, v15, v14, s2
; %bb.234:                              ; %Flow1326
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.235:
	v_mov_b32_e32 v14, 0
	v_or_b32_e32 v19, 0x10000, v15
	v_cmp_eq_u32_sdwa vcc, v15, v14 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v19, v19, v15, vcc
; %bb.236:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v14, 0x7f800000, v16
	v_cmp_ne_u32_e32 vcc, s0, v14
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.237:
	v_bfe_u32 v14, v16, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v16, v14, s2
; %bb.238:                              ; %Flow1325
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.239:
	v_mov_b32_e32 v14, 0
	v_or_b32_e32 v15, 0x10000, v16
	v_cmp_eq_u32_sdwa vcc, v16, v14 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v15, v16, vcc
; %bb.240:                              ; %_ZL16__float2bfloat16f.exit.i.i46.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v14, 0x7f800000, v17
	v_cmp_ne_u32_e32 vcc, s0, v14
                                        ; implicit-def: $vgpr21
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.241:
	v_bfe_u32 v14, v17, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v21, v17, v14, s2
; %bb.242:                              ; %Flow1324
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.243:
	v_mov_b32_e32 v14, 0
	v_or_b32_e32 v15, 0x10000, v17
	v_cmp_eq_u32_sdwa vcc, v17, v14 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v21, v15, v17, vcc
; %bb.244:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.i.i.3
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v18, 16, v18
	v_lshl_add_u64 v[14:15], v[0:1], 1, v[50:51]
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v0, 0xffff0000, v21
	v_mov_b32_e32 v16, s12
	v_mov_b32_e32 v17, 0x20000
	v_or_b32_sdwa v1, v0, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v0, v19, s0, v18
	s_mov_b64 s[2:3], exec
.LBB0_245:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v14
	v_readfirstlane_b32 s5, v15
	v_readfirstlane_b32 s6, v16
	v_readfirstlane_b32 s7, v17
	v_cmp_eq_u64_e32 vcc, s[4:5], v[14:15]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[16:17]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v70, s[4:7], 0 offen
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr70
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_245
; %bb.246:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v10
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.247:
	v_bfe_u32 v0, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v10, v0, s2
; %bb.248:                              ; %Flow1323
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.249:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v10
	v_cmp_eq_u32_sdwa vcc, v10, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v1, v10, vcc
; %bb.250:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v11
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr10
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.251:
	v_bfe_u32 v1, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v10, v11, v1, s2
; %bb.252:                              ; %Flow1322
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.253:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v10, 0x10000, v11
	v_cmp_eq_u32_sdwa vcc, v11, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v11, vcc
; %bb.254:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v12
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.255:
	v_bfe_u32 v1, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v12, v1, s2
; %bb.256:                              ; %Flow1321
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.257:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v11, 0x10000, v12
	v_cmp_eq_u32_sdwa vcc, v12, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v11, v12, vcc
; %bb.258:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v11, 0x7f800000, v13
	v_cmp_ne_u32_e32 vcc, s0, v11
                                        ; implicit-def: $vgpr11
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.259:
	v_bfe_u32 v11, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v11, v13, v11, s2
; %bb.260:                              ; %Flow1320
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.261:
	v_mov_b32_e32 v11, 0
	v_or_b32_e32 v12, 0x10000, v13
	v_cmp_eq_u32_sdwa vcc, v13, v11 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v11, v12, v13, vcc
; %bb.262:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 16, v0
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v11, 0xffff0000, v11
	v_or_b32_sdwa v1, v11, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v0, v10, s0, v0
	s_mov_b64 s[2:3], exec
.LBB0_263:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v14
	v_readfirstlane_b32 s5, v15
	v_readfirstlane_b32 s6, v16
	v_readfirstlane_b32 s7, v17
	v_cmp_eq_u64_e32 vcc, s[4:5], v[14:15]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[16:17]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v60, s[4:7], 0 offen
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr60
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_263
; %bb.264:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v6
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.265:
	v_bfe_u32 v0, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v6, v0, s2
; %bb.266:                              ; %Flow1319
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.267:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v6
	v_cmp_eq_u32_sdwa vcc, v6, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v1, v6, vcc
; %bb.268:                              ; %_ZL16__float2bfloat16f.exit.i.i.167.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v7
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.269:
	v_bfe_u32 v1, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v6, v7, v1, s2
; %bb.270:                              ; %Flow1318
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.271:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v6, 0x10000, v7
	v_cmp_eq_u32_sdwa vcc, v7, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v7, vcc
; %bb.272:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.174.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v8
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.273:
	v_bfe_u32 v1, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v8, v1, s2
; %bb.274:                              ; %Flow1317
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.275:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v7, 0x10000, v8
	v_cmp_eq_u32_sdwa vcc, v8, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v7, v8, vcc
; %bb.276:                              ; %_ZL16__float2bfloat16f.exit.i.i46.178.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v7, 0x7f800000, v9
	v_cmp_ne_u32_e32 vcc, s0, v7
                                        ; implicit-def: $vgpr7
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.277:
	v_bfe_u32 v7, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v7, v9, v7, s2
; %bb.278:                              ; %Flow1316
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.279:
	v_mov_b32_e32 v7, 0
	v_or_b32_e32 v8, 0x10000, v9
	v_cmp_eq_u32_sdwa vcc, v9, v7 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v7, v8, v9, vcc
; %bb.280:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit52.189.i.i.3
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 16, v0
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v7, 0xffff0000, v7
	v_or_b32_sdwa v1, v7, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v0, v6, s0, v0
	s_mov_b64 s[2:3], exec
.LBB0_281:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v14
	v_readfirstlane_b32 s5, v15
	v_readfirstlane_b32 s6, v16
	v_readfirstlane_b32 s7, v17
	v_cmp_eq_u64_e32 vcc, s[4:5], v[14:15]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[16:17]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v56, s[4:7], 0 offen
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr56
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_281
; %bb.282:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v2
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.283:
	v_bfe_u32 v0, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v2, v0, s2
; %bb.284:                              ; %Flow1315
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.285:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v2
	v_cmp_eq_u32_sdwa vcc, v2, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v1, v2, vcc
; %bb.286:                              ; %_ZL16__float2bfloat16f.exit.i.i.1.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v3
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.287:
	v_bfe_u32 v1, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v2, v3, v1, s2
; %bb.288:                              ; %Flow1314
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.289:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v2, 0x10000, v3
	v_cmp_eq_u32_sdwa vcc, v3, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v2, v2, v3, vcc
; %bb.290:                              ; %_ZN7kittens10base_types9convertorI15__hip_bfloat16215HIP_vector_typeIfLj2EEE7convertERKS4_.exit.1.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v1, 0x7f800000, v4
	v_cmp_ne_u32_e32 vcc, s0, v1
                                        ; implicit-def: $vgpr1
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.291:
	v_bfe_u32 v1, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v1, v4, v1, s2
; %bb.292:                              ; %Flow1313
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.293:
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v3, 0x10000, v4
	v_cmp_eq_u32_sdwa vcc, v4, v1 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v1, v3, v4, vcc
; %bb.294:                              ; %_ZL16__float2bfloat16f.exit.i.i46.1.1.i.i.3
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v3, 0x7f800000, v5
	v_cmp_ne_u32_e32 vcc, s0, v3
                                        ; implicit-def: $vgpr3
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.295:
	v_bfe_u32 v3, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v5, v3, s2
; %bb.296:                              ; %Flow
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.297:
	v_mov_b32_e32 v3, 0
	v_or_b32_e32 v4, 0x10000, v5
	v_cmp_eq_u32_sdwa vcc, v5, v3 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v3, v4, v5, vcc
; %bb.298:                              ; %_ZL16store_transposedITkN7kittens5ducks2rt3allENS0_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS0_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS0_5coordIS6_EEEvRKT0_RKT_RKT1_.exit.3
	s_or_b64 exec, exec, s[0:1]
	v_lshrrev_b32_e32 v0, 16, v0
	s_mov_b32 s0, 0xffff0000
	v_and_b32_e32 v3, 0xffff0000, v3
	v_or_b32_sdwa v1, v3, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_and_or_b32 v0, v2, s0, v0
	s_mov_b64 s[0:1], exec
.LBB0_299:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v14
	v_readfirstlane_b32 s5, v15
	v_readfirstlane_b32 s6, v16
	v_readfirstlane_b32 s7, v17
	v_cmp_eq_u64_e32 vcc, s[4:5], v[14:15]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[16:17]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v54, s[4:7], 0 offen
                                        ; implicit-def: $vgpr14_vgpr15_vgpr16_vgpr17
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr54
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_299
; %bb.300:
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
		.amdhsa_next_free_vgpr 207
		.amdhsa_next_free_sgpr 31
		.amdhsa_accum_offset 208
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
; codeLenInByte = 11140
; NumSgprs: 37
; NumVgprs: 207
; NumAgprs: 0
; TotalNumVgprs: 207
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 25
; NumSGPRsForWavesPerEU: 37
; NumVGPRsForWavesPerEU: 207
; AccumOffset: 208
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 51
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_48f1801f2d44560f,@object ; @__hip_cuid_48f1801f2d44560f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_48f1801f2d44560f
__hip_cuid_48f1801f2d44560f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_48f1801f2d44560f, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_48f1801f2d44560f
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
    .sgpr_count:     37
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     207
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
