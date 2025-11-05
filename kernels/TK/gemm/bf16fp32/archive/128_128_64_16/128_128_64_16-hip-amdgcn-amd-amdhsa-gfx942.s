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
	s_lshl_b32 s2, s2, 3
	s_sub_i32 s3, 64, s2
	s_min_i32 s3, s3, 8
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
	s_lshl_b32 s2, s20, 7
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
	v_and_b32_e32 v16, 56, v2
	v_lshrrev_b32_e32 v17, 3, v0
	s_addc_u32 s5, s5, s3
	v_mad_u64_u32 v[2:3], s[2:3], v17, s14, v[16:17]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], v[2:3], 1, s[0:1]
	v_or_b32_e32 v18, 64, v17
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[4:5], off

	;;#ASMEND
	v_mad_u64_u32 v[4:5], s[2:3], v18, s14, v[16:17]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[10:11], v[4:5], 1, s[0:1]
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[10:11], off

	;;#ASMEND
	v_lshlrev_b32_e32 v10, 4, v0
	v_lshlrev_b32_e32 v11, 1, v16
	v_and_b32_e32 v3, 0x1f80, v10
	v_add_u32_e32 v19, s23, v11
	v_add_u32_e32 v5, v19, v3
	v_lshrrev_b32_e32 v22, 4, v5
	v_or_b32_e32 v20, 8, v11
	v_and_b32_e32 v22, 0x78, v22
	v_add_u32_e32 v21, s23, v20
	v_xor_b32_e32 v5, v22, v5
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v5, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v5, v21, v3
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v5, v6, v5
	;;#ASMSTART
	ds_write_b64 v5, v[8:9]

	;;#ASMEND
	v_or_b32_e32 v5, 0x2000, v3
	v_add_u32_e32 v6, v19, v5
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v6, v21, v5
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_xor_b32_e32 v6, v7, v6
	;;#ASMSTART
	ds_write_b64 v6, v[14:15]

	;;#ASMEND
	v_mad_u64_u32 v[6:7], s[2:3], v17, s6, v[16:17]
	s_cmp_eq_u64 s[16:17], 0
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[8:9], v[6:7], 1, s[4:5]
	s_cselect_b32 s7, s24, s25
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[12:15], v[8:9], off

	;;#ASMEND
	v_mad_u64_u32 v[8:9], s[2:3], v18, s6, v[16:17]
	v_add_u32_e32 v7, s7, v11
	v_ashrrev_i32_e32 v9, 31, v8
	v_add_u32_e32 v11, v7, v3
	v_lshl_add_u64 v[16:17], v[8:9], 1, s[4:5]
	v_add_u32_e32 v9, s7, v20
	v_lshrrev_b32_e32 v20, 4, v11
	v_and_b32_e32 v20, 0x78, v20
	v_xor_b32_e32 v11, v20, v11
	;;#ASMSTART
	global_load_dwordx4 v[16:19], v[16:17], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v11, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v11, v9, v3
	v_lshrrev_b32_e32 v12, 4, v11
	v_and_b32_e32 v12, 0x78, v12
	v_xor_b32_e32 v11, v12, v11
	v_add_u32_e32 v7, v7, v5
	;;#ASMSTART
	ds_write_b64 v11, v[14:15]

	;;#ASMEND
	v_lshrrev_b32_e32 v11, 4, v7
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v7, v11, v7
	;;#ASMSTART
	ds_write_b64 v7, v[16:17]

	;;#ASMEND
	v_add_u32_e32 v7, v9, v5
	v_lshrrev_b32_e32 v9, 4, v7
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v7, v9, v7
	;;#ASMSTART
	ds_write_b64 v7, v[18:19]

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
	v_and_b32_e32 v66, 15, v0
	v_lshrrev_b32_e32 v7, 2, v0
	v_bfe_u32 v67, v0, 6, 2
	v_and_b32_e32 v76, 12, v7
	v_lshlrev_b32_e32 v7, 7, v66
	v_lshl_or_b32 v9, v67, 11, v7
	v_lshlrev_b32_e32 v11, 1, v76
	v_add_u32_e32 v9, s7, v9
	v_add_u32_e32 v12, v9, v11
	v_lshrrev_b32_e32 v13, 4, v12
	v_lshlrev_b32_e32 v79, 1, v2
	v_add_u32_e32 v2, 0x2000, v9
	v_and_b32_e32 v13, 0x78, v13
	v_lshlrev_b32_e32 v78, 1, v4
	v_add_u32_e32 v4, v2, v11
	v_xor_b32_e32 v77, v13, v12
	v_lshrrev_b32_e32 v12, 4, v4
	v_and_b32_e32 v12, 0x78, v12
	v_xor_b32_e32 v80, v12, v4
	v_lshl_or_b32 v4, v1, 11, v7
	v_add_u32_e32 v4, s23, v4
	v_add_u32_e32 v7, v4, v11
	v_lshrrev_b32_e32 v12, 4, v7
	v_and_b32_e32 v12, 0x78, v12
	v_xor_b32_e32 v81, v12, v7
	v_add_u32_e32 v7, 0x1000, v4
	v_add_u32_e32 v12, v7, v11
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v82, v13, v12
	v_add_u32_e32 v12, 0x2000, v4
	v_add_u32_e32 v13, v12, v11
	v_lshrrev_b32_e32 v14, 4, v13
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v83, v14, v13
	v_add_u32_e32 v13, 0x3000, v4
	v_add_u32_e32 v14, v13, v11
	v_lshrrev_b32_e32 v15, 4, v14
	v_lshlrev_b32_e32 v86, 1, v6
	v_or_b32_e32 v6, 32, v11
	v_and_b32_e32 v15, 0x78, v15
	v_lshlrev_b32_e32 v85, 1, v8
	v_add_u32_e32 v8, v9, v6
	v_xor_b32_e32 v84, v15, v14
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v87, v14, v8
	v_add_u32_e32 v8, v2, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v88, v14, v8
	v_add_u32_e32 v8, v4, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v89, v14, v8
	v_add_u32_e32 v8, v7, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v90, v14, v8
	v_add_u32_e32 v8, v12, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_add_u32_e32 v6, v13, v6
	v_xor_b32_e32 v91, v14, v8
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v92, v8, v6
	v_or_b32_e32 v6, 64, v11
	v_add_u32_e32 v8, v9, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v93, v14, v8
	v_add_u32_e32 v8, v2, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v94, v14, v8
	v_add_u32_e32 v8, v4, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v95, v14, v8
	v_add_u32_e32 v8, v7, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v96, v14, v8
	v_add_u32_e32 v8, v12, v6
	v_lshrrev_b32_e32 v14, 4, v8
	v_and_b32_e32 v14, 0x78, v14
	v_add_u32_e32 v6, v13, v6
	v_xor_b32_e32 v97, v14, v8
	v_lshrrev_b32_e32 v8, 4, v6
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v98, v8, v6
	v_or_b32_e32 v6, 0x60, v11
	v_add_u32_e32 v8, v9, v6
	v_lshrrev_b32_e32 v9, 4, v8
	v_and_b32_e32 v9, 0x78, v9
	v_add_u32_e32 v2, v2, v6
	v_xor_b32_e32 v99, v9, v8
	v_lshrrev_b32_e32 v8, 4, v2
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v100, v8, v2
	v_add_u32_e32 v2, v4, v6
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v101, v4, v2
	v_add_u32_e32 v2, v7, v6
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v102, v4, v2
	v_add_u32_e32 v2, v12, v6
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v103, v4, v2
	v_add_u32_e32 v2, v13, v6
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v104, v4, v2
	v_and_b32_e32 v2, 0x70, v10
	v_add_u32_e32 v4, s23, v2
	v_add_u32_e32 v8, v4, v3
	v_or_b32_e32 v6, 8, v2
	v_lshrrev_b32_e32 v9, 4, v8
	v_add_u32_e32 v7, s23, v6
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v105, v9, v8
	v_add_u32_e32 v8, v7, v3
	v_lshrrev_b32_e32 v9, 4, v8
	v_and_b32_e32 v9, 0x78, v9
	v_add_u32_e32 v4, v4, v5
	v_xor_b32_e32 v106, v9, v8
	v_lshrrev_b32_e32 v8, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v107, v8, v4
	v_add_u32_e32 v4, v7, v5
	v_lshrrev_b32_e32 v7, 4, v4
	v_and_b32_e32 v7, 0x78, v7
	v_add_u32_e32 v2, s7, v2
	v_xor_b32_e32 v108, v7, v4
	v_add_u32_e32 v4, s7, v6
	v_add_u32_e32 v6, v2, v3
	v_lshrrev_b32_e32 v7, 4, v6
	v_and_b32_e32 v7, 0x78, v7
	v_add_u32_e32 v3, v4, v3
	v_xor_b32_e32 v109, v7, v6
	v_lshrrev_b32_e32 v6, 4, v3
	v_and_b32_e32 v6, 0x78, v6
	v_add_u32_e32 v2, v2, v5
	v_xor_b32_e32 v110, v6, v3
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	s_lshl_b32 s2, s14, 8
	s_lshl_b32 s6, s6, 8
	v_xor_b32_e32 v111, v3, v2
	v_add_u32_e32 v2, v4, v5
	v_lshrrev_b32_e32 v3, 4, v2
	s_add_u32 s14, s4, 0x80
	v_and_b32_e32 v3, 0x78, v3
	s_addc_u32 s15, s5, 0
	v_xor_b32_e32 v112, v3, v2
	s_add_u32 s16, s0, 0x80
	v_mov_b32_e32 v2, 0
	v_mov_b64_e32 v[16:17], 0
	s_addc_u32 s17, s1, 0
	s_mov_b32 s3, 0x110000
	s_mov_b32 s18, 0xffff
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
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
	v_mov_b64_e32 v[8:9], v[16:17]
	v_mov_b64_e32 v[20:21], v[16:17]
	v_mov_b64_e32 v[12:13], v[16:17]
	s_branch .LBB0_4
.LBB0_3:                                ;   in Loop: Header=BB0_4 Depth=1
	v_bfi_b32 v36, s18, v36, v36
	v_bfi_b32 v34, s18, v34, v34
	v_bfi_b32 v70, s18, v70, v70
	v_bfi_b32 v68, s18, v68, v68
	v_bfi_b32 v74, s18, v74, v74
	v_bfi_b32 v72, s18, v72, v72
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[46:49], v[36:37], v[70:71], v[2:5]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[36:37], v[68:69], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[34:35], v[70:71], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[34:35], v[68:69], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[74:75], v[70:71], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[74:75], v[68:69], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[72:73], v[70:71], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[72:73], v[68:69], v[62:65]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s12, s12, 0x80
	s_addc_u32 s13, s13, 0
	s_cmpk_eq_i32 s12, 0x4000
	s_cbranch_scc1 .LBB0_10
.LBB0_4:                                ; =>This Inner Loop Header: Depth=1
	s_cmpk_lg_i32 s12, 0x3f80
	s_cselect_b64 s[4:5], -1, 0
	s_cmpk_eq_i32 s12, 0x3f80
	s_cbranch_scc1 .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s0, s16, s12
	s_addc_u32 s1, s17, s13
	buffer_load_dwordx4 v[6:9], v79, s[0:3], 0 offen
	buffer_load_dwordx4 v[14:17], v78, s[0:3], 0 offen
.LBB0_6:                                ;   in Loop: Header=BB0_4 Depth=1
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[50:51], v77

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[52:53], v80

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[54:55], v81

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[56:57], v82

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[58:59], v83

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[60:61], v84
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v50, s18, v50, v50
	v_bfi_b32 v52, s18, v52, v52
	v_bfi_b32 v54, s18, v54, v54
	v_bfi_b32 v56, s18, v56, v56
	v_bfi_b32 v58, s18, v58, v58
	v_bfi_b32 v60, s18, v60, v60
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[46:49], v[54:55], v[50:51], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[54:55], v[52:53], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[56:57], v[50:51], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[34:37], v[56:57], v[52:53], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[58:59], v[50:51], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[58:59], v[52:53], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[60:61], v[50:51], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[2:5], v[60:61], v[52:53], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_cndmask_b32_e64 v50, 0, 1, s[4:5]
	v_cmp_ne_u32_e64 s[0:1], 1, v50
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_8
; %bb.7:                                ;   in Loop: Header=BB0_4 Depth=1
	s_add_u32 s4, s14, s12
	s_addc_u32 s5, s15, s13
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v[10:13], v86, s[4:7], 0 offen
	buffer_load_dwordx4 v[18:21], v85, s[4:7], 0 offen
.LBB0_8:                                ;   in Loop: Header=BB0_4 Depth=1
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[62:63], v87

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[68:69], v88

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[50:51], v89

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[52:53], v90

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[58:59], v91

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[70:71], v92
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v62, s18, v62, v62
	v_bfi_b32 v68, s18, v68, v68
	v_bfi_b32 v50, s18, v50, v50
	v_bfi_b32 v52, s18, v52, v52
	v_bfi_b32 v58, s18, v58, v58
	v_bfi_b32 v70, s18, v70, v70
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[46:49], v[50:51], v[62:63], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[42:45], v[50:51], v[68:69], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[38:41], v[52:53], v[62:63], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[52:53], v[68:69], v[34:37]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[58:59], v[62:63], v[30:33]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[58:59], v[68:69], v[26:29]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[70:71], v[62:63], v[22:25]
	v_mfma_f32_16x16x16_bf16 v[114:117], v[70:71], v[68:69], v[2:5]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[118:119], v93

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[120:121], v94

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[22:23], v95

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[30:31], v96

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[122:123], v97

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[124:125], v98

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[70:71], v99

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[68:69], v100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[36:37], v101

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[34:35], v102

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[74:75], v103

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[72:73], v104
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v22, s18, v22, v22
	v_bfi_b32 v30, s18, v30, v30
	v_bfi_b32 v118, s18, v118, v118
	v_bfi_b32 v120, s18, v120, v120
	v_bfi_b32 v122, s18, v122, v122
	v_bfi_b32 v124, s18, v124, v124
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[2:5], v[22:23], v[118:119], v[46:49]
	v_mfma_f32_16x16x16_bf16 v[22:25], v[22:23], v[120:121], v[42:45]
	v_mfma_f32_16x16x16_bf16 v[26:29], v[30:31], v[118:119], v[38:41]
	v_mfma_f32_16x16x16_bf16 v[30:33], v[30:31], v[120:121], v[50:53]
	v_mfma_f32_16x16x16_bf16 v[50:53], v[122:123], v[118:119], v[54:57]
	v_mfma_f32_16x16x16_bf16 v[54:57], v[122:123], v[120:121], v[58:61]
	v_mfma_f32_16x16x16_bf16 v[58:61], v[124:125], v[118:119], v[62:65]
	v_mfma_f32_16x16x16_bf16 v[62:65], v[124:125], v[120:121], v[114:117]
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
	ds_write_b64 v105, v[6:7]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v106, v[8:9]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v107, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v108, v[16:17]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v109, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v110, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v111, v[18:19]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v112, v[20:21]

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
	s_waitcnt vmcnt(1)
	v_mad_u64_u32 v[6:7], s[0:1], v76, s10, v[66:67]
	v_add_u32_e32 v8, s10, v6
	v_lshl_or_b32 v50, v1, 4, s22
	v_lshlrev_b32_e32 v0, 4, v67
	v_add_u32_e32 v10, s10, v8
	v_lshl_or_b32 v0, s20, 7, v0
	s_mul_i32 s2, s10, s21
	v_mul_lo_u32 v1, s11, v50
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[14:15], s[0:1], s10, v50, 0
	v_add_u32_e32 v12, s10, v10
	v_add3_u32 v15, v15, s2, v1
	v_ashrrev_i32_e32 v1, 31, v0
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[8:9]
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[0:1]
	v_lshlrev_b64 v[6:7], 2, v[6:7]
	v_lshlrev_b64 v[8:9], 2, v[8:9]
	v_lshlrev_b64 v[10:11], 2, v[10:11]
	v_lshlrev_b64 v[12:13], 2, v[12:13]
	v_lshl_add_u64 v[16:17], v[14:15], 0, v[6:7]
	v_lshl_add_u64 v[18:19], v[14:15], 0, v[8:9]
	v_lshl_add_u64 v[20:21], v[14:15], 0, v[10:11]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[12:13]
	global_store_dword v[16:17], v46, off
	global_store_dword v[18:19], v47, off
	global_store_dword v[20:21], v48, off
	global_store_dword v[14:15], v49, off
	global_store_dword v[16:17], v42, off offset:256
	global_store_dword v[18:19], v43, off offset:256
	global_store_dword v[20:21], v44, off offset:256
	global_store_dword v[14:15], v45, off offset:256
	v_or_b32_e32 v14, 32, v50
	v_mul_lo_u32 v16, s11, v14
	v_mad_u64_u32 v[14:15], s[0:1], s10, v14, 0
	v_add3_u32 v15, v15, s2, v16
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[8:9]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[0:1]
	v_lshl_add_u64 v[16:17], v[14:15], 0, v[6:7]
	v_lshl_add_u64 v[18:19], v[14:15], 0, v[8:9]
	v_lshl_add_u64 v[20:21], v[14:15], 0, v[10:11]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[12:13]
	global_store_dword v[16:17], v38, off
	global_store_dword v[18:19], v39, off
	global_store_dword v[20:21], v40, off
	global_store_dword v[14:15], v41, off
	global_store_dword v[16:17], v34, off offset:256
	global_store_dword v[18:19], v35, off offset:256
	global_store_dword v[20:21], v36, off offset:256
	global_store_dword v[14:15], v37, off offset:256
	v_or_b32_e32 v14, 64, v50
	v_mul_lo_u32 v16, s11, v14
	v_mad_u64_u32 v[14:15], s[0:1], s10, v14, 0
	v_add3_u32 v15, v15, s2, v16
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[8:9]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[0:1]
	v_lshl_add_u64 v[16:17], v[14:15], 0, v[6:7]
	v_lshl_add_u64 v[18:19], v[14:15], 0, v[8:9]
	v_lshl_add_u64 v[20:21], v[14:15], 0, v[10:11]
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[12:13]
	global_store_dword v[16:17], v30, off
	global_store_dword v[18:19], v31, off
	global_store_dword v[20:21], v32, off
	global_store_dword v[14:15], v33, off
	global_store_dword v[16:17], v26, off offset:256
	global_store_dword v[18:19], v27, off offset:256
	global_store_dword v[20:21], v28, off offset:256
	global_store_dword v[14:15], v29, off offset:256
	v_or_b32_e32 v14, 0x60, v50
	v_mul_lo_u32 v16, s11, v14
	v_mad_u64_u32 v[14:15], s[0:1], s10, v14, 0
	v_add3_u32 v15, v15, s2, v16
	v_lshl_add_u64 v[14:15], v[14:15], 2, s[8:9]
	v_lshl_add_u64 v[0:1], v[14:15], 0, v[0:1]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[6:7]
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[8:9]
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[12:13]
	global_store_dword v[6:7], v22, off
	global_store_dword v[8:9], v23, off
	global_store_dword v[10:11], v24, off
	global_store_dword v[0:1], v25, off
	global_store_dword v[6:7], v2, off offset:256
	global_store_dword v[8:9], v3, off offset:256
	global_store_dword v[10:11], v4, off offset:256
	global_store_dword v[0:1], v5, off offset:256
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
		.amdhsa_next_free_vgpr 126
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 128
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
; codeLenInByte = 3512
; NumSgprs: 34
; NumVgprs: 126
; NumAgprs: 0
; TotalNumVgprs: 126
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 126
; AccumOffset: 128
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_67315ad1895b224f,@object ; @__hip_cuid_67315ad1895b224f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_67315ad1895b224f
__hip_cuid_67315ad1895b224f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_67315ad1895b224f, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_67315ad1895b224f
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
    .vgpr_count:     126
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
