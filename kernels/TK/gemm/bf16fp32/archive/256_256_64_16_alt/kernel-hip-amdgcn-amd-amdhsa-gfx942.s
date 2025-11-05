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
	s_load_dwordx2 s[4:5], s[0:1], 0x20
	s_load_dwordx2 s[14:15], s[0:1], 0x30
	s_load_dwordx2 s[6:7], s[0:1], 0x50
	s_cmp_lg_u32 0, -1
	s_load_dwordx2 s[18:19], s[0:1], 0x90
	s_cselect_b32 s8, 0, 0
	s_and_b32 s9, s8, -16
	s_mov_b32 s17, 0
	s_and_b32 s16, s8, 15
	s_add_i32 s9, s9, 16
	s_cmp_eq_u64 s[16:17], 0
	s_cselect_b32 s21, s8, s9
	s_load_dwordx2 s[8:9], s[0:1], 0x60
	s_load_dwordx2 s[10:11], s[0:1], 0x80
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s18, s3
	s_add_i32 s0, s0, s2
	s_ashr_i32 s2, s0, 31
	s_lshr_b32 s2, s2, 29
	s_add_i32 s2, s0, s2
	s_mul_i32 s1, s19, s18
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
	s_sub_i32 s3, 32, s2
	s_min_i32 s3, s3, 16
	s_abs_i32 s18, s3
	v_cvt_f32_u32_e32 v1, s18
	s_sub_i32 s20, 0, s18
	s_add_i32 s22, s21, 0x8000
	s_and_b32 s1, s1, 0xfffffe00
	v_rcp_iflag_f32_e32 v1, v1
	s_and_b32 s19, s22, -16
	s_sub_i32 s0, s0, s1
	s_add_i32 s23, s19, 16
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_abs_i32 s19, s0
	s_xor_b32 s1, s0, s3
	s_and_b32 s16, s22, 15
	v_readfirstlane_b32 s24, v1
	s_mul_i32 s20, s20, s24
	s_mul_hi_u32 s20, s24, s20
	s_add_i32 s24, s24, s20
	s_mul_hi_u32 s20, s19, s24
	s_mul_i32 s24, s20, s18
	s_sub_i32 s19, s19, s24
	s_ashr_i32 s1, s1, 31
	s_add_i32 s24, s20, 1
	s_sub_i32 s25, s19, s18
	s_cmp_ge_u32 s19, s18
	s_cselect_b32 s20, s24, s20
	s_cselect_b32 s19, s25, s19
	s_add_i32 s24, s20, 1
	s_cmp_ge_u32 s19, s18
	s_cselect_b32 s18, s24, s20
	s_xor_b32 s18, s18, s1
	s_sub_i32 s20, s18, s1
	s_mul_i32 s1, s20, s3
	s_sub_i32 s0, s0, s1
	s_add_i32 s2, s2, s0
	s_lshl_b32 s19, s2, 8
	s_ashr_i32 s18, s19, 31
	s_mul_i32 s0, s4, s18
	s_mul_hi_u32 s1, s4, s19
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s5, s19
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s4, s19
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s12, s12, s0
	s_addc_u32 s13, s13, s1
	s_lshl_b32 s20, s20, 8
	s_ashr_i32 s0, s20, 31
	s_mul_i32 s0, s6, s0
	s_mul_hi_u32 s1, s6, s20
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s7, s20
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s6, s20
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshlrev_b32_e32 v1, 3, v0
	s_add_u32 s14, s14, s0
	v_and_b32_e32 v18, 56, v1
	v_lshrrev_b32_e32 v1, 3, v0
	s_addc_u32 s15, s15, s1
	v_mad_u64_u32 v[20:21], s[0:1], v1, s4, v[18:19]
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[2:3], v[20:21], 1, s[12:13]
	v_or_b32_e32 v19, 64, v1
	v_or_b32_e32 v21, 0xc0, v1
	v_mad_u64_u32 v[22:23], s[0:1], v19, s4, v[18:19]
	v_mad_u64_u32 v[26:27], s[0:1], v21, s4, v[18:19]
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u32 v24, s4, 7, v20
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[6:7], v[22:23], 1, s[12:13]
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u64 v[14:15], v[26:27], 1, s[12:13]
	v_lshlrev_b32_e32 v23, 4, v0
	v_lshlrev_b32_e32 v27, 1, v18
	v_lshl_add_u64 v[10:11], v[24:25], 1, s[12:13]
	v_and_b32_e32 v25, 0x1f80, v23
	v_add_u32_e32 v28, s21, v27
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_add_u32_e32 v30, v28, v25
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	v_or_b32_e32 v34, 8, v27
	v_lshrrev_b32_e32 v31, 4, v30
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	v_add_u32_e32 v29, s21, v34
	v_and_b32_e32 v31, 0x78, v31
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[14:15], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v30, v31, v30
	;;#ASMSTART
	ds_write_b64 v30, v[2:3]

	;;#ASMEND
	v_add_u32_e32 v2, v29, v25
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	v_or_b32_e32 v35, 0x2000, v25
	;;#ASMSTART
	ds_write_b64 v2, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v2, v28, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v2, v29, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	v_or_b32_e32 v36, 0x4000, v25
	;;#ASMSTART
	ds_write_b64 v2, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v2, v28, v36
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v2, v29, v36
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	v_or_b32_e32 v37, 0x6000, v25
	;;#ASMSTART
	ds_write_b64 v2, v[12:13]

	;;#ASMEND
	v_add_u32_e32 v2, v28, v37
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v2, v29, v37
	v_lshrrev_b32_e32 v3, 4, v2
	s_cmp_eq_u64 s[16:17], 0
	v_and_b32_e32 v3, 0x78, v3
	v_mad_u64_u32 v[28:29], s[0:1], v1, s6, v[18:19]
	v_mad_u64_u32 v[30:31], s[0:1], v19, s6, v[18:19]
	v_mad_u64_u32 v[18:19], s[0:1], v21, s6, v[18:19]
	v_xor_b32_e32 v2, v3, v2
	v_ashrrev_i32_e32 v29, 31, v28
	s_cselect_b32 s0, s22, s23
	;;#ASMSTART
	ds_write_b64 v2, v[16:17]

	;;#ASMEND
	v_lshl_add_u64 v[2:3], v[28:29], 1, s[14:15]
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u32 v32, s6, 7, v28
	v_add_u32_e32 v1, s0, v27
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[2:5], v[2:3], off

	;;#ASMEND
	v_lshl_add_u64 v[6:7], v[30:31], 1, s[14:15]
	v_ashrrev_i32_e32 v33, 31, v32
	v_add_u32_e32 v21, v1, v25
	;;#ASMSTART
	global_load_dwordx4 v[6:9], v[6:7], off

	;;#ASMEND
	v_lshl_add_u64 v[10:11], v[32:33], 1, s[14:15]
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshrrev_b32_e32 v27, 4, v21
	;;#ASMSTART
	global_load_dwordx4 v[10:13], v[10:11], off

	;;#ASMEND
	v_lshl_add_u64 v[14:15], v[18:19], 1, s[14:15]
	v_add_u32_e32 v19, s0, v34
	v_and_b32_e32 v27, 0x78, v27
	;;#ASMSTART
	global_load_dwordx4 v[14:17], v[14:15], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	v_xor_b32_e32 v21, v27, v21
	;;#ASMSTART
	ds_write_b64 v21, v[2:3]

	;;#ASMEND
	v_add_u32_e32 v2, v19, v25
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[4:5]

	;;#ASMEND
	v_add_u32_e32 v2, v1, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v2, v19, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[8:9]

	;;#ASMEND
	v_add_u32_e32 v2, v1, v36
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	;;#ASMSTART
	ds_write_b64 v2, v[10:11]

	;;#ASMEND
	v_add_u32_e32 v2, v19, v36
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v2, v3, v2
	v_add_u32_e32 v1, v1, v37
	;;#ASMSTART
	ds_write_b64 v2, v[12:13]

	;;#ASMEND
	v_lshrrev_b32_e32 v2, 4, v1
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v1, v2, v1
	;;#ASMSTART
	ds_write_b64 v1, v[14:15]

	;;#ASMEND
	v_add_u32_e32 v1, v19, v37
	v_lshrrev_b32_e32 v2, 4, v1
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v1, v2, v1
	v_and_b32_e32 v38, 0xc0, v0
	;;#ASMSTART
	ds_write_b64 v1, v[16:17]

	;;#ASMEND
	v_and_b32_e32 v16, 15, v0
	v_lshrrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v0, 7, v0
	v_and_b32_e32 v2, 12, v1
	v_and_b32_e32 v0, 0x6780, v0
	scratch_store_dword off, v2, off offset:104 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v2, 1, v2
	v_add_u32_e32 v0, s0, v0
	v_add_u32_e32 v3, v0, v2
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v17, v4, v3
	v_or_b32_e32 v3, 16, v16
	v_or_b32_e32 v4, v38, v3
	v_lshl_add_u32 v4, v4, 7, s0
	v_add_u32_e32 v5, v4, v2
	v_lshrrev_b32_e32 v6, 4, v5
	v_and_b32_e32 v6, 0x78, v6
	v_xor_b32_e32 v19, v6, v5
	v_or_b32_e32 v5, 32, v16
	v_or_b32_e32 v6, v38, v5
	v_lshl_add_u32 v6, v6, 7, s0
	v_add_u32_e32 v7, v6, v2
	v_lshrrev_b32_e32 v8, 4, v7
	v_and_b32_e32 v8, 0x78, v8
	v_xor_b32_e32 v21, v8, v7
	v_or_b32_e32 v7, 48, v16
	v_or_b32_e32 v8, v38, v7
	v_lshl_add_u32 v8, v8, 7, s0
	v_add_u32_e32 v9, v8, v2
	v_and_b32_e32 v14, 64, v1
	v_lshrrev_b32_e32 v10, 4, v9
	v_or_b32_e32 v1, v14, v16
	v_and_b32_e32 v10, 0x78, v10
	v_lshl_add_u32 v1, v1, 7, s21
	v_xor_b32_e32 v27, v10, v9
	v_add_u32_e32 v9, v1, v2
	v_lshrrev_b32_e32 v10, 4, v9
	v_and_b32_e32 v10, 0x78, v10
	v_xor_b32_e32 v29, v10, v9
	v_or_b32_e32 v9, v3, v14
	v_lshl_add_u32 v9, v9, 7, s21
	v_add_u32_e32 v10, v9, v2
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v201, v11, v10
	v_or_b32_e32 v10, v5, v14
	v_lshl_add_u32 v10, v10, 7, s21
	v_add_u32_e32 v11, v10, v2
	v_lshrrev_b32_e32 v12, 4, v11
	v_and_b32_e32 v12, 0x78, v12
	v_xor_b32_e32 v202, v12, v11
	v_or_b32_e32 v11, v7, v14
	v_lshl_add_u32 v11, v11, 7, s21
	v_add_u32_e32 v12, v11, v2
	v_lshrrev_b32_e32 v13, 4, v12
	v_and_b32_e32 v13, 0x78, v13
	v_xor_b32_e32 v203, v13, v12
	v_or_b32_e32 v12, 0x80, v14
	v_or_b32_e32 v13, v12, v16
	scratch_store_dword off, v14, off offset:112 ; 4-byte Folded Spill
	v_mov_b32_e32 v14, v16
	v_lshl_add_u32 v13, v13, 7, s21
	scratch_store_dwordx2 off, v[14:15], off offset:96 ; 8-byte Folded Spill
	v_add_u32_e32 v14, v13, v2
	v_lshrrev_b32_e32 v15, 4, v14
	v_or_b32_e32 v3, v12, v3
	v_and_b32_e32 v15, 0x78, v15
	v_lshl_add_u32 v3, v3, 7, s21
	v_xor_b32_e32 v204, v15, v14
	v_add_u32_e32 v14, v3, v2
	v_lshrrev_b32_e32 v15, 4, v14
	v_or_b32_e32 v5, v12, v5
	v_and_b32_e32 v15, 0x78, v15
	v_lshl_add_u32 v5, v5, 7, s21
	v_xor_b32_e32 v205, v15, v14
	v_add_u32_e32 v14, v5, v2
	v_or_b32_e32 v7, v12, v7
	v_lshrrev_b32_e32 v15, 4, v14
	v_lshl_add_u32 v7, v7, 7, s21
	v_and_b32_e32 v15, 0x78, v15
	v_add_u32_e32 v12, v7, v2
	v_xor_b32_e32 v208, v15, v14
	v_lshrrev_b32_e32 v14, 4, v12
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v215, v14, v12
	v_lshlrev_b32_e32 v12, 1, v20
	scratch_store_dword off, v12, off offset:84 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v12, 1, v22
	scratch_store_dword off, v12, off offset:88 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v12, 1, v24
	scratch_store_dword off, v12, off offset:92 ; 4-byte Folded Spill
	v_or_b32_e32 v12, 32, v2
	v_add_u32_e32 v14, v0, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v223, v15, v14
	v_add_u32_e32 v14, v4, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v222, v15, v14
	v_add_u32_e32 v14, v6, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v220, v15, v14
	v_add_u32_e32 v14, v8, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v216, v15, v14
	v_add_u32_e32 v14, v1, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v214, v15, v14
	v_add_u32_e32 v14, v9, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v213, v15, v14
	v_add_u32_e32 v14, v10, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v212, v15, v14
	v_add_u32_e32 v14, v11, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v211, v15, v14
	v_add_u32_e32 v14, v13, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v210, v15, v14
	v_add_u32_e32 v14, v3, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v209, v15, v14
	v_add_u32_e32 v14, v5, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_add_u32_e32 v12, v7, v12
	v_xor_b32_e32 v207, v15, v14
	v_lshrrev_b32_e32 v14, 4, v12
	v_and_b32_e32 v14, 0x78, v14
	v_xor_b32_e32 v206, v14, v12
	v_or_b32_e32 v12, 64, v2
	v_add_u32_e32 v14, v0, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v200, v15, v14
	v_add_u32_e32 v14, v4, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v199, v15, v14
	v_add_u32_e32 v14, v6, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v198, v15, v14
	v_add_u32_e32 v14, v8, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:80 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v1, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:76 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v9, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:72 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v10, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:68 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v11, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:64 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v13, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:60 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v3, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	scratch_store_dword off, v14, off offset:56 ; 4-byte Folded Spill
	v_add_u32_e32 v14, v5, v12
	v_lshrrev_b32_e32 v15, 4, v14
	v_and_b32_e32 v15, 0x78, v15
	v_xor_b32_e32 v14, v15, v14
	v_add_u32_e32 v12, v7, v12
	scratch_store_dword off, v14, off offset:52 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v14, 4, v12
	v_and_b32_e32 v14, 0x78, v14
	v_or_b32_e32 v2, 0x60, v2
	v_xor_b32_e32 v12, v14, v12
	v_add_u32_e32 v0, v0, v2
	scratch_store_dword off, v12, off offset:48 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v12, 4, v0
	v_and_b32_e32 v12, 0x78, v12
	v_xor_b32_e32 v0, v12, v0
	scratch_store_dword off, v0, off offset:44 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v4, v2
	v_lshrrev_b32_e32 v4, 4, v0
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v0, v4, v0
	scratch_store_dword off, v0, off offset:40 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v6, v2
	v_lshrrev_b32_e32 v4, 4, v0
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v0, v4, v0
	scratch_store_dword off, v0, off offset:36 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v8, v2
	v_lshrrev_b32_e32 v4, 4, v0
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v0, v4, v0
	scratch_store_dword off, v0, off offset:32 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v1, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:28 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v9, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:24 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v10, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:20 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v11, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:16 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v13, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:12 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v3, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:8 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v5, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off offset:4 ; 4-byte Folded Spill
	v_add_u32_e32 v0, v7, v2
	v_lshrrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v1, 0x78, v1
	v_xor_b32_e32 v0, v1, v0
	scratch_store_dword off, v0, off        ; 4-byte Folded Spill
	v_and_b32_e32 v0, 0x70, v23
	v_add_u32_e32 v1, s21, v0
	v_add_u32_e32 v2, v1, v25
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v228, v3, v2
	v_or_b32_e32 v2, 8, v0
	v_add_u32_e32 v3, s21, v2
	v_add_u32_e32 v4, v3, v25
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v229, v5, v4
	v_add_u32_e32 v4, v1, v35
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v230, v5, v4
	v_add_u32_e32 v4, v3, v35
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v231, v5, v4
	v_add_u32_e32 v4, v1, v36
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v232, v5, v4
	v_add_u32_e32 v4, v3, v36
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v1, v1, v37
	v_xor_b32_e32 v233, v5, v4
	v_lshrrev_b32_e32 v4, 4, v1
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v234, v4, v1
	v_add_u32_e32 v1, v3, v37
	v_lshrrev_b32_e32 v3, 4, v1
	v_and_b32_e32 v3, 0x78, v3
	v_add_u32_e32 v0, s0, v0
	v_xor_b32_e32 v235, v3, v1
	v_add_u32_e32 v1, v0, v25
	v_lshrrev_b32_e32 v3, 4, v1
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v236, v3, v1
	v_add_u32_e32 v1, s0, v2
	v_add_u32_e32 v2, v1, v25
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v237, v3, v2
	v_add_u32_e32 v2, v0, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v238, v3, v2
	v_add_u32_e32 v2, v1, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	scratch_store_dword off, v38, off offset:108 ; 4-byte Folded Spill
	v_xor_b32_e32 v239, v3, v2
	v_add_u32_e32 v2, v0, v36
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b64 v[146:147], v17

	;;#ASMEND
	v_lshrrev_b32_e32 v3, 4, v2
	;;#ASMSTART
	ds_read_b64 v[148:149], v19

	;;#ASMEND
	v_and_b32_e32 v3, 0x78, v3
	;;#ASMSTART
	ds_read_b64 v[150:151], v21

	;;#ASMEND
	v_xor_b32_e32 v240, v3, v2
	v_add_u32_e32 v2, v1, v36
	;;#ASMSTART
	ds_read_b64 v[152:153], v27

	;;#ASMEND
	v_lshrrev_b32_e32 v3, 4, v2
	;;#ASMSTART
	ds_read_b64 v[168:169], v29

	;;#ASMEND
	v_and_b32_e32 v3, 0x78, v3
	v_add_u32_e32 v0, v0, v37
	;;#ASMSTART
	ds_read_b64 v[166:167], v201

	;;#ASMEND
	v_xor_b32_e32 v241, v3, v2
	v_lshrrev_b32_e32 v2, 4, v0
	;;#ASMSTART
	ds_read_b64 v[164:165], v202

	;;#ASMEND
	v_and_b32_e32 v2, 0x78, v2
	;;#ASMSTART
	ds_read_b64 v[162:163], v203

	;;#ASMEND
	v_xor_b32_e32 v242, v2, v0
	v_add_u32_e32 v0, v1, v37
	;;#ASMSTART
	ds_read_b64 v[160:161], v204

	;;#ASMEND
	v_lshrrev_b32_e32 v1, 4, v0
	;;#ASMSTART
	ds_read_b64 v[158:159], v205

	;;#ASMEND
	v_and_b32_e32 v1, 0x78, v1
	;;#ASMSTART
	ds_read_b64 v[156:157], v208

	;;#ASMEND
	v_xor_b32_e32 v243, v1, v0
	s_mov_b32 s21, 0xffff
	;;#ASMSTART
	ds_read_b64 v[154:155], v215

	;;#ASMEND
	v_mov_b32_e32 v0, 0
	v_lshlrev_b32_e32 v221, 1, v26
	v_lshlrev_b32_e32 v224, 1, v28
	v_lshlrev_b32_e32 v225, 1, v30
	v_lshlrev_b32_e32 v226, 1, v32
	v_lshlrev_b32_e32 v227, 1, v18
	v_mov_b32_e32 v217, v17
	v_bfi_b32 v146, s21, v146, v146
	v_mov_b32_e32 v218, v19
	v_bfi_b32 v148, s21, v148, v148
	v_mov_b32_e32 v196, v21
	v_bfi_b32 v150, s21, v150, v150
	v_mov_b32_e32 v219, v27
	v_bfi_b32 v152, s21, v152, v152
	v_mov_b32_e32 v197, v29
	v_bfi_b32 v168, s21, v168, v168
	v_bfi_b32 v166, s21, v166, v166
	v_bfi_b32 v164, s21, v164, v164
	v_bfi_b32 v162, s21, v162, v162
	v_bfi_b32 v160, s21, v160, v160
	v_bfi_b32 v158, s21, v158, v158
	v_bfi_b32 v156, s21, v156, v156
	v_bfi_b32 v154, s21, v154, v154
	s_lshl_b32 s2, s4, 9
	s_lshl_b32 s6, s6, 9
	s_mov_b64 s[16:17], 0x80
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
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b32_e32 v88, v0
	v_mov_b32_e32 v89, v0
	v_mov_b32_e32 v90, v0
	v_mov_b32_e32 v91, v0
	v_mov_b32_e32 v92, v0
	v_mov_b32_e32 v93, v0
	v_mov_b32_e32 v94, v0
	v_mov_b32_e32 v95, v0
	v_mov_b32_e32 v96, v0
	v_mov_b32_e32 v97, v0
	v_mov_b32_e32 v98, v0
	v_mov_b32_e32 v99, v0
	v_mov_b32_e32 v100, v0
	v_mov_b32_e32 v101, v0
	v_mov_b32_e32 v102, v0
	v_mov_b32_e32 v103, v0
	v_mov_b32_e32 v104, v0
	v_mov_b32_e32 v105, v0
	v_mov_b32_e32 v106, v0
	v_mov_b32_e32 v107, v0
	v_mov_b32_e32 v108, v0
	v_mov_b32_e32 v109, v0
	v_mov_b32_e32 v110, v0
	v_mov_b32_e32 v111, v0
	v_mov_b32_e32 v112, v0
	v_mov_b32_e32 v113, v0
	v_mov_b32_e32 v114, v0
	v_mov_b32_e32 v115, v0
	v_mov_b32_e32 v116, v0
	v_mov_b32_e32 v117, v0
	v_mov_b32_e32 v118, v0
	v_mov_b32_e32 v119, v0
	v_mov_b32_e32 v120, v0
	v_mov_b32_e32 v121, v0
	v_mov_b32_e32 v122, v0
	v_mov_b32_e32 v123, v0
	v_mov_b32_e32 v124, v0
	v_mov_b32_e32 v125, v0
	v_mov_b32_e32 v126, v0
	v_mov_b32_e32 v127, v0
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	scratch_load_dword v128, off, off offset:84 ; 4-byte Folded Reload
	scratch_load_dword v132, off, off offset:88 ; 4-byte Folded Reload
	scratch_load_dword v136, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s0, s12, s16
	s_addc_u32 s1, s13, s17
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[128:131], v128, s[0:3], 0 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[132:135], v132, s[0:3], 0 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[136:139], v136, s[0:3], 0 offen
	s_nop 0
	buffer_load_dwordx4 v[140:143], v221, s[0:3], 0 offen
	;;#ASMSTART
	ds_read_b64 v[244:245], v223

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[246:247], v222

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[248:249], v220

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[250:251], v216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[252:253], v214

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[254:255], v213

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v212

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[170:171], v211

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[172:173], v210
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[174:175], v209
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v207
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[178:179], v206
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v244, s21, v244, v244
	v_bfi_b32 v246, s21, v246, v246
	v_bfi_b32 v248, s21, v248, v248
	v_bfi_b32 v250, s21, v250, v250
	v_bfi_b32 v252, s21, v252, v252
	v_bfi_b32 v254, s21, v254, v254
	v_bfi_b32 v144, s21, v144, v144
	v_bfi_b32 v170, s21, v170, v170
	v_bfi_b32 v172, s21, v172, v172
	v_bfi_b32 v174, s21, v174, v174
	v_bfi_b32 v176, s21, v176, v176
	v_bfi_b32 v178, s21, v178, v178
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[168:169], v[146:147], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[168:169], v[148:149], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[168:169], v[150:151], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[168:169], v[152:153], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[166:167], v[146:147], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[166:167], v[148:149], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[166:167], v[150:151], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[166:167], v[152:153], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[164:165], v[146:147], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[164:165], v[148:149], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[164:165], v[150:151], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[164:165], v[152:153], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[162:163], v[146:147], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[162:163], v[148:149], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[162:163], v[150:151], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[162:163], v[152:153], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[160:161], v[146:147], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[160:161], v[148:149], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[160:161], v[150:151], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[160:161], v[152:153], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[158:159], v[146:147], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[158:159], v[148:149], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[158:159], v[150:151], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[158:159], v[152:153], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[156:157], v[146:147], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[156:157], v[148:149], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[156:157], v[150:151], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[156:157], v[152:153], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[154:155], v[146:147], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[154:155], v[148:149], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[154:155], v[150:151], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[154:155], v[152:153], v[0:3]
	s_setprio 0
	s_add_u32 s4, s14, s16
	s_addc_u32 s5, s15, s17
	s_mov_b32 s7, s3
	buffer_load_dwordx4 v[146:149], v224, s[4:7], 0 offen
	buffer_load_dwordx4 v[150:153], v225, s[4:7], 0 offen
	buffer_load_dwordx4 v[154:157], v226, s[4:7], 0 offen
	buffer_load_dwordx4 v[158:161], v227, s[4:7], 0 offen
	;;#ASMSTART
	ds_read_b64 v[162:163], v200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v199

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[166:167], v198

	;;#ASMEND
	scratch_load_dword v168, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[168:169], v168

	;;#ASMEND
	scratch_load_dword v180, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[180:181], v180

	;;#ASMEND
	scratch_load_dword v182, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[182:183], v182

	;;#ASMEND
	scratch_load_dword v184, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[184:185], v184

	;;#ASMEND
	scratch_load_dword v186, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[186:187], v186

	;;#ASMEND
	scratch_load_dword v188, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[188:189], v188
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v190, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[190:191], v190
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v192, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[192:193], v192
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v194, off, off offset:48 ; 4-byte Folded Reload
	v_bfi_b32 v162, s21, v162, v162
	v_bfi_b32 v164, s21, v164, v164
	v_bfi_b32 v166, s21, v166, v166
	v_bfi_b32 v168, s21, v168, v168
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[194:195], v194
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v180, s21, v180, v180
	v_bfi_b32 v182, s21, v182, v182
	v_bfi_b32 v184, s21, v184, v184
	v_bfi_b32 v186, s21, v186, v186
	v_bfi_b32 v188, s21, v188, v188
	v_bfi_b32 v190, s21, v190, v190
	v_bfi_b32 v192, s21, v192, v192
	v_bfi_b32 v194, s21, v194, v194
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[252:253], v[244:245], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[252:253], v[246:247], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[252:253], v[248:249], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[252:253], v[250:251], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[254:255], v[244:245], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[254:255], v[246:247], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[254:255], v[248:249], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[254:255], v[250:251], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[144:145], v[244:245], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[144:145], v[246:247], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[144:145], v[248:249], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[144:145], v[250:251], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[170:171], v[244:245], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[170:171], v[246:247], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[170:171], v[248:249], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[170:171], v[250:251], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[172:173], v[244:245], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[172:173], v[246:247], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[172:173], v[248:249], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[172:173], v[250:251], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[174:175], v[244:245], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[174:175], v[246:247], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[174:175], v[248:249], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[174:175], v[250:251], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[176:177], v[244:245], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[176:177], v[246:247], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[176:177], v[248:249], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[176:177], v[250:251], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[178:179], v[244:245], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[178:179], v[246:247], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[178:179], v[248:249], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[178:179], v[250:251], v[0:3]
	s_setprio 0
	scratch_load_dword v144, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[144:145], v144

	;;#ASMEND
	scratch_load_dword v170, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[170:171], v170

	;;#ASMEND
	scratch_load_dword v172, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[172:173], v172

	;;#ASMEND
	scratch_load_dword v174, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[174:175], v174

	;;#ASMEND
	scratch_load_dword v176, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[176:177], v176

	;;#ASMEND
	scratch_load_dword v178, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[178:179], v178

	;;#ASMEND
	scratch_load_dword v244, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[244:245], v244

	;;#ASMEND
	scratch_load_dword v246, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[246:247], v246

	;;#ASMEND
	scratch_load_dword v248, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[248:249], v248
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v250, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[250:251], v250
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v252, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[252:253], v252
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v254, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[254:255], v254
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v144, s21, v144, v144
	v_bfi_b32 v170, s21, v170, v170
	v_bfi_b32 v172, s21, v172, v172
	v_bfi_b32 v174, s21, v174, v174
	v_bfi_b32 v176, s21, v176, v176
	v_bfi_b32 v178, s21, v178, v178
	v_bfi_b32 v244, s21, v244, v244
	v_bfi_b32 v246, s21, v246, v246
	v_bfi_b32 v248, s21, v248, v248
	v_bfi_b32 v250, s21, v250, v250
	v_bfi_b32 v252, s21, v252, v252
	v_bfi_b32 v254, s21, v254, v254
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[180:181], v[162:163], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[180:181], v[164:165], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[180:181], v[166:167], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[180:181], v[168:169], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[182:183], v[162:163], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[182:183], v[164:165], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[182:183], v[166:167], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[182:183], v[168:169], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[184:185], v[162:163], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[184:185], v[164:165], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[184:185], v[166:167], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[184:185], v[168:169], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[186:187], v[162:163], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[186:187], v[164:165], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[186:187], v[166:167], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[186:187], v[168:169], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[188:189], v[162:163], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[188:189], v[164:165], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[188:189], v[166:167], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[188:189], v[168:169], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[190:191], v[162:163], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[190:191], v[164:165], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[190:191], v[166:167], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[190:191], v[168:169], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[192:193], v[162:163], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[192:193], v[164:165], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[192:193], v[166:167], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[192:193], v[168:169], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[194:195], v[162:163], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[194:195], v[164:165], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[194:195], v[166:167], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[194:195], v[168:169], v[0:3]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v228, v[128:129]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v229, v[130:131]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v230, v[132:133]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v231, v[134:135]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v232, v[136:137]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v233, v[138:139]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v234, v[140:141]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v235, v[142:143]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v236, v[146:147]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v237, v[148:149]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v238, v[150:151]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v239, v[152:153]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v240, v[154:155]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v241, v[156:157]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v242, v[158:159]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v243, v[160:161]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b64 v[146:147], v217

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v218

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v196

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v219

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[168:169], v197

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[166:167], v201

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v202

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v203

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v204
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v205
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[156:157], v208
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v215
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_nop 0
	v_bfi_b32 v146, s21, v146, v146
	v_bfi_b32 v148, s21, v148, v148
	v_bfi_b32 v150, s21, v150, v150
	v_bfi_b32 v152, s21, v152, v152
	v_bfi_b32 v168, s21, v168, v168
	v_bfi_b32 v166, s21, v166, v166
	v_bfi_b32 v164, s21, v164, v164
	v_bfi_b32 v162, s21, v162, v162
	v_bfi_b32 v160, s21, v160, v160
	v_bfi_b32 v158, s21, v158, v158
	v_bfi_b32 v156, s21, v156, v156
	v_bfi_b32 v154, s21, v154, v154
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[176:177], v[144:145], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[176:177], v[170:171], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[176:177], v[172:173], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[176:177], v[174:175], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[178:179], v[144:145], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[178:179], v[170:171], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[178:179], v[172:173], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[178:179], v[174:175], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[244:245], v[144:145], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[244:245], v[170:171], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[244:245], v[172:173], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[244:245], v[174:175], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[246:247], v[144:145], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[246:247], v[170:171], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[246:247], v[172:173], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[246:247], v[174:175], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[248:249], v[144:145], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[248:249], v[170:171], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[248:249], v[172:173], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[248:249], v[174:175], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[250:251], v[144:145], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[250:251], v[170:171], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[250:251], v[172:173], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[250:251], v[174:175], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[252:253], v[144:145], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[252:253], v[170:171], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[252:253], v[172:173], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[252:253], v[174:175], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[254:255], v[144:145], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[254:255], v[170:171], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[254:255], v[172:173], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[254:255], v[174:175], v[0:3]
	s_setprio 0
	s_add_u32 s16, s16, 0x80
	s_addc_u32 s17, s17, 0
	s_cmpk_eq_i32 s16, 0x4000
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	;;#ASMSTART
	ds_read_b64 v[128:129], v223

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[130:131], v222

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v220

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v214

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v213

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v212

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v211

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[184:185], v210
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[202:203], v209
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	s_mov_b32 s0, 0xffff
	;;#ASMSTART
	ds_read_b64 v[204:205], v207
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v132, s0, v132, v132
	;;#ASMSTART
	ds_read_b64 v[206:207], v206
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v128, s0, v128, v128
	v_bfi_b32 v130, s0, v130, v130
	v_bfi_b32 v134, s0, v134, v134
	v_bfi_b32 v136, s0, v136, v136
	v_bfi_b32 v138, s0, v138, v138
	v_bfi_b32 v140, s0, v140, v140
	v_bfi_b32 v142, s0, v142, v142
	v_bfi_b32 v184, s0, v184, v184
	v_bfi_b32 v202, s0, v202, v202
	v_bfi_b32 v204, s0, v204, v204
	v_bfi_b32 v206, s0, v206, v206
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[168:169], v[146:147], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[168:169], v[148:149], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[168:169], v[150:151], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[168:169], v[152:153], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[166:167], v[146:147], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[166:167], v[148:149], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[166:167], v[150:151], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[166:167], v[152:153], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[164:165], v[146:147], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[164:165], v[148:149], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[164:165], v[150:151], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[164:165], v[152:153], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[162:163], v[146:147], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[162:163], v[148:149], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[162:163], v[150:151], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[162:163], v[152:153], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[160:161], v[146:147], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[160:161], v[148:149], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[160:161], v[150:151], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[160:161], v[152:153], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[158:159], v[146:147], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[158:159], v[148:149], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[158:159], v[150:151], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[158:159], v[152:153], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[156:157], v[146:147], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[156:157], v[148:149], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[156:157], v[150:151], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[156:157], v[152:153], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[154:155], v[146:147], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[154:155], v[148:149], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[154:155], v[150:151], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[154:155], v[152:153], v[0:3]
	s_setprio 0
	;;#ASMSTART
	ds_read_b64 v[146:147], v200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v199

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v198

	;;#ASMEND
	scratch_load_dword v144, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[152:153], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:76 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[154:155], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[156:157], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[158:159], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:64 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[160:161], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[162:163], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[164:165], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[166:167], v144

	;;#ASMEND
	scratch_load_dword v144, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[168:169], v144

	;;#ASMEND
	v_bfi_b32 v146, s0, v146, v146
	v_bfi_b32 v148, s0, v148, v148
	v_bfi_b32 v150, s0, v150, v150
	v_bfi_b32 v152, s0, v152, v152
	v_bfi_b32 v154, s0, v154, v154
	v_bfi_b32 v156, s0, v156, v156
	v_bfi_b32 v158, s0, v158, v158
	v_bfi_b32 v160, s0, v160, v160
	v_bfi_b32 v162, s0, v162, v162
	v_bfi_b32 v164, s0, v164, v164
	v_bfi_b32 v166, s0, v166, v166
	v_bfi_b32 v168, s0, v168, v168
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[136:137], v[128:129], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[136:137], v[130:131], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[136:137], v[132:133], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[136:137], v[134:135], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[138:139], v[128:129], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[138:139], v[130:131], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[138:139], v[132:133], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[138:139], v[134:135], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[140:141], v[128:129], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[140:141], v[130:131], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[140:141], v[132:133], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[140:141], v[134:135], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[142:143], v[128:129], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[142:143], v[130:131], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[142:143], v[132:133], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[142:143], v[134:135], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[184:185], v[128:129], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[184:185], v[130:131], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[184:185], v[132:133], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[184:185], v[134:135], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[202:203], v[128:129], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[202:203], v[130:131], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[202:203], v[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[202:203], v[134:135], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[204:205], v[128:129], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[204:205], v[130:131], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[204:205], v[132:133], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[204:205], v[134:135], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[206:207], v[128:129], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[206:207], v[130:131], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[206:207], v[132:133], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[206:207], v[134:135], v[0:3]
	s_setprio 0
	scratch_load_dword v128, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[128:129], v128

	;;#ASMEND
	scratch_load_dword v130, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[130:131], v130

	;;#ASMEND
	scratch_load_dword v132, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[132:133], v132

	;;#ASMEND
	scratch_load_dword v134, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[134:135], v134

	;;#ASMEND
	scratch_load_dword v136, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[136:137], v136

	;;#ASMEND
	scratch_load_dword v138, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[138:139], v138

	;;#ASMEND
	scratch_load_dword v140, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[140:141], v140

	;;#ASMEND
	scratch_load_dword v142, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[142:143], v142

	;;#ASMEND
	scratch_load_dword v144, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[176:177], v144
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v144, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[174:175], v144
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	scratch_load_dword v144, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[178:179], v144
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v132, s0, v132, v132
	scratch_load_dword v144, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64 v[172:173], v144
s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_bfi_b32 v128, s0, v128, v128
	v_bfi_b32 v130, s0, v130, v130
	v_bfi_b32 v134, s0, v134, v134
	v_bfi_b32 v136, s0, v136, v136
	v_bfi_b32 v138, s0, v138, v138
	v_bfi_b32 v140, s0, v140, v140
	v_bfi_b32 v142, s0, v142, v142
	v_bfi_b32 v176, s0, v176, v176
	v_bfi_b32 v174, s0, v174, v174
	v_bfi_b32 v178, s0, v178, v178
	v_bfi_b32 v172, s0, v172, v172
	;;#ASMSTART
	s_waitcnt lgkmcnt(12)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[154:155], v[146:147], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[154:155], v[148:149], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[154:155], v[150:151], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[154:155], v[152:153], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[156:157], v[146:147], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[156:157], v[148:149], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[156:157], v[150:151], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[156:157], v[152:153], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[158:159], v[146:147], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[158:159], v[148:149], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[158:159], v[150:151], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[158:159], v[152:153], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[160:161], v[146:147], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[160:161], v[148:149], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[160:161], v[150:151], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[160:161], v[152:153], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[162:163], v[146:147], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[162:163], v[148:149], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[162:163], v[150:151], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[162:163], v[152:153], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[164:165], v[146:147], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[164:165], v[148:149], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[164:165], v[150:151], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[164:165], v[152:153], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[166:167], v[146:147], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[166:167], v[148:149], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[166:167], v[150:151], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[166:167], v[152:153], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[168:169], v[146:147], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[168:169], v[148:149], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[168:169], v[150:151], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[168:169], v[152:153], v[0:3]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 3
	v_mfma_f32_16x16x16_bf16 v[124:127], v[136:137], v[128:129], v[124:127]
	v_mfma_f32_16x16x16_bf16 v[120:123], v[136:137], v[130:131], v[120:123]
	v_mfma_f32_16x16x16_bf16 v[116:119], v[136:137], v[132:133], v[116:119]
	v_mfma_f32_16x16x16_bf16 v[112:115], v[136:137], v[134:135], v[112:115]
	v_mfma_f32_16x16x16_bf16 v[108:111], v[138:139], v[128:129], v[108:111]
	v_mfma_f32_16x16x16_bf16 v[104:107], v[138:139], v[130:131], v[104:107]
	v_mfma_f32_16x16x16_bf16 v[100:103], v[138:139], v[132:133], v[100:103]
	v_mfma_f32_16x16x16_bf16 v[96:99], v[138:139], v[134:135], v[96:99]
	v_mfma_f32_16x16x16_bf16 v[92:95], v[140:141], v[128:129], v[92:95]
	v_mfma_f32_16x16x16_bf16 v[88:91], v[140:141], v[130:131], v[88:91]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[140:141], v[132:133], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[140:141], v[134:135], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[142:143], v[128:129], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[142:143], v[130:131], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[142:143], v[132:133], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[142:143], v[134:135], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[176:177], v[128:129], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[176:177], v[130:131], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[176:177], v[132:133], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[176:177], v[134:135], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[174:175], v[128:129], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[174:175], v[130:131], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[174:175], v[132:133], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[174:175], v[134:135], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[178:179], v[128:129], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[178:179], v[130:131], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[20:23], v[178:179], v[132:133], v[20:23]
	v_mfma_f32_16x16x16_bf16 v[16:19], v[178:179], v[134:135], v[16:19]
	v_mfma_f32_16x16x16_bf16 v[12:15], v[172:173], v[128:129], v[12:15]
	v_mfma_f32_16x16x16_bf16 v[8:11], v[172:173], v[130:131], v[8:11]
	v_mfma_f32_16x16x16_bf16 v[4:7], v[172:173], v[132:133], v[4:7]
	v_mfma_f32_16x16x16_bf16 v[0:3], v[172:173], v[134:135], v[0:3]
	s_setprio 0
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v128, 0x7f800000, v124
	v_cmp_ne_u32_e32 vcc, s0, v128
                                        ; implicit-def: $vgpr132
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.3:
	v_bfe_u32 v128, v124, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v132, v124, v128, s2
; %bb.4:                                ; %Flow4743
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[148:149], off, off offset:96 ; 8-byte Folded Reload
	scratch_load_dword v149, off, off offset:104 ; 4-byte Folded Reload
	s_xor_b64 exec, exec, s[0:1]
; %bb.5:
	v_mov_b32_e32 v128, 0
	v_or_b32_e32 v129, 0x10000, v124
	v_cmp_eq_u32_sdwa vcc, v124, v128 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v132, v129, v124, vcc
; %bb.6:                                ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v124, off, off offset:112 ; 4-byte Folded Reload
	s_mul_i32 s2, s10, s18
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v146, s19, v124
	scratch_load_dword v124, off, off offset:108 ; 4-byte Folded Reload
	v_mad_u64_u32 v[130:131], s[0:1], s10, v146, 0
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v128, s20, v124
	v_mul_lo_u32 v124, s11, v146
	v_add3_u32 v131, v131, s2, v124
	v_ashrrev_i32_e32 v129, 31, v128
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[8:9]
	v_lshl_add_u64 v[136:137], v[128:129], 1, v[130:131]
	v_mad_u64_u32 v[130:131], s[0:1], v149, s10, v[148:149]
	v_ashrrev_i32_e32 v131, 31, v130
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v125
	v_lshl_add_u64 v[138:139], v[130:131], 1, v[136:137]
	v_cmp_ne_u32_e32 vcc, s0, v124
	global_store_short_d16_hi v[138:139], v132, off
                                        ; implicit-def: $vgpr124
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.7:
	v_bfe_u32 v124, v125, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v124, v125, v124, s2
; %bb.8:                                ; %Flow4742
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.9:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v132, 0x10000, v125
	v_cmp_eq_u32_sdwa vcc, v125, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v124, v132, v125, vcc
; %bb.10:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v132, s10, v130
	v_ashrrev_i32_e32 v133, 31, v132
	v_lshl_add_u64 v[140:141], v[132:133], 1, v[136:137]
	global_store_short_d16_hi v[140:141], v124, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v126
	v_cmp_ne_u32_e32 vcc, s0, v124
                                        ; implicit-def: $vgpr124
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.11:
	v_bfe_u32 v124, v126, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v124, v126, v124, s2
; %bb.12:                               ; %Flow4741
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.13:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v125, 0x10000, v126
	v_cmp_eq_u32_sdwa vcc, v126, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v124, v125, v126, vcc
; %bb.14:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v134, s10, v132
	v_ashrrev_i32_e32 v135, 31, v134
	v_lshl_add_u64 v[142:143], v[134:135], 1, v[136:137]
	global_store_short_d16_hi v[142:143], v124, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v124, 0x7f800000, v127
	v_cmp_ne_u32_e32 vcc, s0, v124
                                        ; implicit-def: $vgpr147
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.15:
	v_bfe_u32 v124, v127, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v147, v127, v124, s2
                                        ; implicit-def: $vgpr124_vgpr125_vgpr126_vgpr127
; %bb.16:                               ; %Flow4740
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.17:
	v_mov_b32_e32 v124, 0
	v_or_b32_e32 v125, 0x10000, v127
	v_cmp_eq_u32_sdwa vcc, v127, v124 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v147, v125, v127, vcc
; %bb.18:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v124, s10, v134
	v_ashrrev_i32_e32 v125, 31, v124
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v144, 0x7f800000, v120
	v_lshl_add_u64 v[126:127], v[124:125], 1, v[136:137]
	v_cmp_ne_u32_e32 vcc, s0, v144
	global_store_short_d16_hi v[126:127], v147, off
                                        ; implicit-def: $vgpr147
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.19:
	v_bfe_u32 v144, v120, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v147, v120, v144, s2
; %bb.20:                               ; %Flow4739
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.21:
	v_mov_b32_e32 v144, 0
	v_or_b32_e32 v145, 0x10000, v120
	v_cmp_eq_u32_sdwa vcc, v120, v144 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v147, v145, v120, vcc
; %bb.22:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v121
	v_cmp_ne_u32_e32 vcc, s0, v120
	global_store_short_d16_hi v[138:139], v147, off offset:32
                                        ; implicit-def: $vgpr120
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.23:
	v_bfe_u32 v120, v121, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v121, v120, s2
; %bb.24:                               ; %Flow4738
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.25:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v144, 0x10000, v121
	v_cmp_eq_u32_sdwa vcc, v121, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v144, v121, vcc
; %bb.26:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[140:141], v120, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v122
	v_cmp_ne_u32_e32 vcc, s0, v120
                                        ; implicit-def: $vgpr120
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.27:
	v_bfe_u32 v120, v122, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v122, v120, s2
; %bb.28:                               ; %Flow4737
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.29:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v121, 0x10000, v122
	v_cmp_eq_u32_sdwa vcc, v122, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v121, v122, vcc
; %bb.30:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[142:143], v120, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v123
	v_cmp_ne_u32_e32 vcc, s0, v120
                                        ; implicit-def: $vgpr147
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.31:
	v_bfe_u32 v120, v123, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v147, v123, v120, s2
                                        ; implicit-def: $vgpr120_vgpr121_vgpr122_vgpr123
; %bb.32:                               ; %Flow4736
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.33:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v121, 0x10000, v123
	v_cmp_eq_u32_sdwa vcc, v123, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v147, v121, v123, vcc
; %bb.34:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v120, 0x7f800000, v116
	v_cmp_ne_u32_e32 vcc, s0, v120
	global_store_short_d16_hi v[126:127], v147, off offset:32
                                        ; implicit-def: $vgpr120
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.35:
	v_bfe_u32 v120, v116, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v116, v120, s2
; %bb.36:                               ; %Flow4735
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.37:
	v_mov_b32_e32 v120, 0
	v_or_b32_e32 v121, 0x10000, v116
	v_cmp_eq_u32_sdwa vcc, v116, v120 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v121, v116, vcc
; %bb.38:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v117
	v_cmp_ne_u32_e32 vcc, s0, v116
	global_store_short_d16_hi v[138:139], v120, off offset:64
                                        ; implicit-def: $vgpr116
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.39:
	v_bfe_u32 v116, v117, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v117, v116, s2
; %bb.40:                               ; %Flow4734
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.41:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v120, 0x10000, v117
	v_cmp_eq_u32_sdwa vcc, v117, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v120, v117, vcc
; %bb.42:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[140:141], v116, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v118
	v_cmp_ne_u32_e32 vcc, s0, v116
                                        ; implicit-def: $vgpr116
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.43:
	v_bfe_u32 v116, v118, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v118, v116, s2
; %bb.44:                               ; %Flow4733
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.45:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v117, 0x10000, v118
	v_cmp_eq_u32_sdwa vcc, v118, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v117, v118, vcc
; %bb.46:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[142:143], v116, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v119
	v_cmp_ne_u32_e32 vcc, s0, v116
                                        ; implicit-def: $vgpr120
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.47:
	v_bfe_u32 v116, v119, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v120, v119, v116, s2
                                        ; implicit-def: $vgpr116_vgpr117_vgpr118_vgpr119
; %bb.48:                               ; %Flow4732
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.49:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v117, 0x10000, v119
	v_cmp_eq_u32_sdwa vcc, v119, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v120, v117, v119, vcc
; %bb.50:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v116, 0x7f800000, v112
	v_cmp_ne_u32_e32 vcc, s0, v116
	global_store_short_d16_hi v[126:127], v120, off offset:64
                                        ; implicit-def: $vgpr116
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.51:
	v_bfe_u32 v116, v112, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v112, v116, s2
; %bb.52:                               ; %Flow4731
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.53:
	v_mov_b32_e32 v116, 0
	v_or_b32_e32 v117, 0x10000, v112
	v_cmp_eq_u32_sdwa vcc, v112, v116 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v117, v112, vcc
; %bb.54:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v113
	v_cmp_ne_u32_e32 vcc, s0, v112
	global_store_short_d16_hi v[138:139], v116, off offset:96
                                        ; implicit-def: $vgpr112
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.55:
	v_bfe_u32 v112, v113, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v112, v113, v112, s2
; %bb.56:                               ; %Flow4730
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.57:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v116, 0x10000, v113
	v_cmp_eq_u32_sdwa vcc, v113, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v112, v116, v113, vcc
; %bb.58:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[140:141], v112, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v114
	v_cmp_ne_u32_e32 vcc, s0, v112
                                        ; implicit-def: $vgpr112
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.59:
	v_bfe_u32 v112, v114, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v112, v114, v112, s2
; %bb.60:                               ; %Flow4729
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.61:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v113, 0x10000, v114
	v_cmp_eq_u32_sdwa vcc, v114, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v112, v113, v114, vcc
; %bb.62:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[142:143], v112, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v115
	v_cmp_ne_u32_e32 vcc, s0, v112
                                        ; implicit-def: $vgpr116
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.63:
	v_bfe_u32 v112, v115, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v115, v112, s2
                                        ; implicit-def: $vgpr112_vgpr113_vgpr114_vgpr115
; %bb.64:                               ; %Flow4728
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.65:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v113, 0x10000, v115
	v_cmp_eq_u32_sdwa vcc, v115, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v116, v113, v115, vcc
; %bb.66:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v112, 0x7f800000, v108
	v_cmp_ne_u32_e32 vcc, s0, v112
	global_store_short_d16_hi v[126:127], v116, off offset:96
                                        ; implicit-def: $vgpr114
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.67:
	v_bfe_u32 v112, v108, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v114, v108, v112, s2
; %bb.68:                               ; %Flow4727
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.69:
	v_mov_b32_e32 v112, 0
	v_or_b32_e32 v113, 0x10000, v108
	v_cmp_eq_u32_sdwa vcc, v108, v112 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v114, v113, v108, vcc
; %bb.70:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.190.i.i
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v108, 16, v149
	v_mad_u64_u32 v[112:113], s[0:1], v108, s10, v[148:149]
	v_ashrrev_i32_e32 v113, 31, v112
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v109
	v_lshl_add_u64 v[118:119], v[112:113], 1, v[136:137]
	v_cmp_ne_u32_e32 vcc, s0, v108
	global_store_short_d16_hi v[118:119], v114, off
                                        ; implicit-def: $vgpr108
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.71:
	v_bfe_u32 v108, v109, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v108, v109, v108, s2
; %bb.72:                               ; %Flow4726
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.73:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v114, 0x10000, v109
	v_cmp_eq_u32_sdwa vcc, v109, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v108, v114, v109, vcc
; %bb.74:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.195.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v114, s10, v112
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[120:121], v[114:115], 1, v[136:137]
	global_store_short_d16_hi v[120:121], v108, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v110
	v_cmp_ne_u32_e32 vcc, s0, v108
                                        ; implicit-def: $vgpr108
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.75:
	v_bfe_u32 v108, v110, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v108, v110, v108, s2
; %bb.76:                               ; %Flow4725
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.77:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v109, 0x10000, v110
	v_cmp_eq_u32_sdwa vcc, v110, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v108, v109, v110, vcc
; %bb.78:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1100.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v116, s10, v114
	v_ashrrev_i32_e32 v117, 31, v116
	v_lshl_add_u64 v[122:123], v[116:117], 1, v[136:137]
	global_store_short_d16_hi v[122:123], v108, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v108, 0x7f800000, v111
	v_cmp_ne_u32_e32 vcc, s0, v108
                                        ; implicit-def: $vgpr126
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.79:
	v_bfe_u32 v108, v111, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v126, v111, v108, s2
                                        ; implicit-def: $vgpr108_vgpr109_vgpr110_vgpr111
; %bb.80:                               ; %Flow4724
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.81:
	v_mov_b32_e32 v108, 0
	v_or_b32_e32 v109, 0x10000, v111
	v_cmp_eq_u32_sdwa vcc, v111, v108 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v126, v109, v111, vcc
; %bb.82:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1103.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v108, s10, v116
	v_ashrrev_i32_e32 v109, 31, v108
	v_lshl_add_u64 v[110:111], v[108:109], 1, v[136:137]
	global_store_short_d16_hi v[110:111], v126, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v126, 0x7f800000, v104
	v_cmp_ne_u32_e32 vcc, s0, v126
                                        ; implicit-def: $vgpr126
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.83:
	v_bfe_u32 v126, v104, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v126, v104, v126, s2
; %bb.84:                               ; %Flow4723
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.85:
	v_mov_b32_e32 v126, 0
	v_or_b32_e32 v127, 0x10000, v104
	v_cmp_eq_u32_sdwa vcc, v104, v126 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v126, v127, v104, vcc
; %bb.86:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v105
	v_cmp_ne_u32_e32 vcc, s0, v104
	global_store_short_d16_hi v[118:119], v126, off offset:32
                                        ; implicit-def: $vgpr104
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.87:
	v_bfe_u32 v104, v105, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v105, v104, s2
; %bb.88:                               ; %Flow4722
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.89:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v126, 0x10000, v105
	v_cmp_eq_u32_sdwa vcc, v105, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v126, v105, vcc
; %bb.90:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[120:121], v104, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v106
	v_cmp_ne_u32_e32 vcc, s0, v104
                                        ; implicit-def: $vgpr104
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.91:
	v_bfe_u32 v104, v106, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v106, v104, s2
; %bb.92:                               ; %Flow4721
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.93:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v105, 0x10000, v106
	v_cmp_eq_u32_sdwa vcc, v106, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v105, v106, vcc
; %bb.94:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[122:123], v104, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v107
	v_cmp_ne_u32_e32 vcc, s0, v104
                                        ; implicit-def: $vgpr126
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.95:
	v_bfe_u32 v104, v107, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v126, v107, v104, s2
                                        ; implicit-def: $vgpr104_vgpr105_vgpr106_vgpr107
; %bb.96:                               ; %Flow4720
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.97:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v105, 0x10000, v107
	v_cmp_eq_u32_sdwa vcc, v107, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v126, v105, v107, vcc
; %bb.98:                               ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v104, 0x7f800000, v100
	v_cmp_ne_u32_e32 vcc, s0, v104
	global_store_short_d16_hi v[110:111], v126, off offset:32
                                        ; implicit-def: $vgpr104
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.99:
	v_bfe_u32 v104, v100, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v100, v104, s2
; %bb.100:                              ; %Flow4719
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.101:
	v_mov_b32_e32 v104, 0
	v_or_b32_e32 v105, 0x10000, v100
	v_cmp_eq_u32_sdwa vcc, v100, v104 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v105, v100, vcc
; %bb.102:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v101
	v_cmp_ne_u32_e32 vcc, s0, v100
	global_store_short_d16_hi v[118:119], v104, off offset:64
                                        ; implicit-def: $vgpr100
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.103:
	v_bfe_u32 v100, v101, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v101, v100, s2
; %bb.104:                              ; %Flow4718
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.105:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v104, 0x10000, v101
	v_cmp_eq_u32_sdwa vcc, v101, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v104, v101, vcc
; %bb.106:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[120:121], v100, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v102
	v_cmp_ne_u32_e32 vcc, s0, v100
                                        ; implicit-def: $vgpr100
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.107:
	v_bfe_u32 v100, v102, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v102, v100, s2
; %bb.108:                              ; %Flow4717
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.109:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v101, 0x10000, v102
	v_cmp_eq_u32_sdwa vcc, v102, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v101, v102, vcc
; %bb.110:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[122:123], v100, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v103
	v_cmp_ne_u32_e32 vcc, s0, v100
                                        ; implicit-def: $vgpr104
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.111:
	v_bfe_u32 v100, v103, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v104, v103, v100, s2
                                        ; implicit-def: $vgpr100_vgpr101_vgpr102_vgpr103
; %bb.112:                              ; %Flow4716
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.113:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v101, 0x10000, v103
	v_cmp_eq_u32_sdwa vcc, v103, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v104, v101, v103, vcc
; %bb.114:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v100, 0x7f800000, v96
	v_cmp_ne_u32_e32 vcc, s0, v100
	global_store_short_d16_hi v[110:111], v104, off offset:64
                                        ; implicit-def: $vgpr100
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.115:
	v_bfe_u32 v100, v96, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v96, v100, s2
; %bb.116:                              ; %Flow4715
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.117:
	v_mov_b32_e32 v100, 0
	v_or_b32_e32 v101, 0x10000, v96
	v_cmp_eq_u32_sdwa vcc, v96, v100 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v101, v96, vcc
; %bb.118:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v97
	v_cmp_ne_u32_e32 vcc, s0, v96
	global_store_short_d16_hi v[118:119], v100, off offset:96
                                        ; implicit-def: $vgpr96
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.119:
	v_bfe_u32 v96, v97, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v96, v97, v96, s2
; %bb.120:                              ; %Flow4714
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.121:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v100, 0x10000, v97
	v_cmp_eq_u32_sdwa vcc, v97, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v96, v100, v97, vcc
; %bb.122:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[120:121], v96, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v98
	v_cmp_ne_u32_e32 vcc, s0, v96
                                        ; implicit-def: $vgpr96
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.123:
	v_bfe_u32 v96, v98, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v96, v98, v96, s2
; %bb.124:                              ; %Flow4713
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.125:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v97, 0x10000, v98
	v_cmp_eq_u32_sdwa vcc, v98, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v96, v97, v98, vcc
; %bb.126:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[122:123], v96, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v99
	v_cmp_ne_u32_e32 vcc, s0, v96
                                        ; implicit-def: $vgpr100
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.127:
	v_bfe_u32 v96, v99, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v100, v99, v96, s2
                                        ; implicit-def: $vgpr96_vgpr97_vgpr98_vgpr99
; %bb.128:                              ; %Flow4712
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.129:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v97, 0x10000, v99
	v_cmp_eq_u32_sdwa vcc, v99, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v100, v97, v99, vcc
; %bb.130:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3.1.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v96, 0x7f800000, v92
	v_cmp_ne_u32_e32 vcc, s0, v96
	global_store_short_d16_hi v[110:111], v100, off offset:96
                                        ; implicit-def: $vgpr98
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.131:
	v_bfe_u32 v96, v92, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v98, v92, v96, s2
; %bb.132:                              ; %Flow4711
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.133:
	v_mov_b32_e32 v96, 0
	v_or_b32_e32 v97, 0x10000, v92
	v_cmp_eq_u32_sdwa vcc, v92, v96 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v98, v97, v92, vcc
; %bb.134:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2110.i.i
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v92, 32, v149
	v_mad_u64_u32 v[96:97], s[0:1], v92, s10, v[148:149]
	v_ashrrev_i32_e32 v97, 31, v96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v93
	v_lshl_add_u64 v[102:103], v[96:97], 1, v[136:137]
	v_cmp_ne_u32_e32 vcc, s0, v92
	global_store_short_d16_hi v[102:103], v98, off
                                        ; implicit-def: $vgpr92
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.135:
	v_bfe_u32 v92, v93, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v92, v93, v92, s2
; %bb.136:                              ; %Flow4710
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.137:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v98, 0x10000, v93
	v_cmp_eq_u32_sdwa vcc, v93, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v92, v98, v93, vcc
; %bb.138:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2115.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v98, s10, v96
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[104:105], v[98:99], 1, v[136:137]
	global_store_short_d16_hi v[104:105], v92, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v94
	v_cmp_ne_u32_e32 vcc, s0, v92
                                        ; implicit-def: $vgpr92
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.139:
	v_bfe_u32 v92, v94, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v92, v94, v92, s2
; %bb.140:                              ; %Flow4709
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.141:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v93, 0x10000, v94
	v_cmp_eq_u32_sdwa vcc, v94, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v92, v93, v94, vcc
; %bb.142:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2120.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v100, s10, v98
	v_ashrrev_i32_e32 v101, 31, v100
	v_lshl_add_u64 v[106:107], v[100:101], 1, v[136:137]
	global_store_short_d16_hi v[106:107], v92, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v92, 0x7f800000, v95
	v_cmp_ne_u32_e32 vcc, s0, v92
                                        ; implicit-def: $vgpr110
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.143:
	v_bfe_u32 v92, v95, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v110, v95, v92, s2
                                        ; implicit-def: $vgpr92_vgpr93_vgpr94_vgpr95
; %bb.144:                              ; %Flow4708
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.145:
	v_mov_b32_e32 v92, 0
	v_or_b32_e32 v93, 0x10000, v95
	v_cmp_eq_u32_sdwa vcc, v95, v92 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v110, v93, v95, vcc
; %bb.146:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2123.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v92, s10, v100
	v_ashrrev_i32_e32 v93, 31, v92
	v_lshl_add_u64 v[94:95], v[92:93], 1, v[136:137]
	global_store_short_d16_hi v[94:95], v110, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v110, 0x7f800000, v88
	v_cmp_ne_u32_e32 vcc, s0, v110
                                        ; implicit-def: $vgpr110
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.147:
	v_bfe_u32 v110, v88, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v110, v88, v110, s2
; %bb.148:                              ; %Flow4707
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.149:
	v_mov_b32_e32 v110, 0
	v_or_b32_e32 v111, 0x10000, v88
	v_cmp_eq_u32_sdwa vcc, v88, v110 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v110, v111, v88, vcc
; %bb.150:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v89
	v_cmp_ne_u32_e32 vcc, s0, v88
	global_store_short_d16_hi v[102:103], v110, off offset:32
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.151:
	v_bfe_u32 v88, v89, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v88, v89, v88, s2
; %bb.152:                              ; %Flow4706
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.153:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v110, 0x10000, v89
	v_cmp_eq_u32_sdwa vcc, v89, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v110, v89, vcc
; %bb.154:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[104:105], v88, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v90
	v_cmp_ne_u32_e32 vcc, s0, v88
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.155:
	v_bfe_u32 v88, v90, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v88, v90, v88, s2
; %bb.156:                              ; %Flow4705
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.157:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v89, 0x10000, v90
	v_cmp_eq_u32_sdwa vcc, v90, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v89, v90, vcc
; %bb.158:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[106:107], v88, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v91
	v_cmp_ne_u32_e32 vcc, s0, v88
                                        ; implicit-def: $vgpr110
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.159:
	v_bfe_u32 v88, v91, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v110, v91, v88, s2
                                        ; implicit-def: $vgpr88_vgpr89_vgpr90_vgpr91
; %bb.160:                              ; %Flow4704
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.161:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v89, 0x10000, v91
	v_cmp_eq_u32_sdwa vcc, v91, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v110, v89, v91, vcc
; %bb.162:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v88, 0x7f800000, v84
	v_cmp_ne_u32_e32 vcc, s0, v88
	global_store_short_d16_hi v[94:95], v110, off offset:32
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.163:
	v_bfe_u32 v88, v84, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v88, v84, v88, s2
; %bb.164:                              ; %Flow4703
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.165:
	v_mov_b32_e32 v88, 0
	v_or_b32_e32 v89, 0x10000, v84
	v_cmp_eq_u32_sdwa vcc, v84, v88 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v89, v84, vcc
; %bb.166:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v85
	v_cmp_ne_u32_e32 vcc, s0, v84
	global_store_short_d16_hi v[102:103], v88, off offset:64
                                        ; implicit-def: $vgpr84
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.167:
	v_bfe_u32 v84, v85, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v85, v84, s2
; %bb.168:                              ; %Flow4702
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.169:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v88, 0x10000, v85
	v_cmp_eq_u32_sdwa vcc, v85, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v84, v88, v85, vcc
; %bb.170:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[104:105], v84, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v86
	v_cmp_ne_u32_e32 vcc, s0, v84
                                        ; implicit-def: $vgpr84
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.171:
	v_bfe_u32 v84, v86, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v86, v84, s2
; %bb.172:                              ; %Flow4701
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.173:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v85, 0x10000, v86
	v_cmp_eq_u32_sdwa vcc, v86, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v84, v85, v86, vcc
; %bb.174:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[106:107], v84, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v87
	v_cmp_ne_u32_e32 vcc, s0, v84
                                        ; implicit-def: $vgpr88
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.175:
	v_bfe_u32 v84, v87, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v88, v87, v84, s2
                                        ; implicit-def: $vgpr84_vgpr85_vgpr86_vgpr87
; %bb.176:                              ; %Flow4700
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.177:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v85, 0x10000, v87
	v_cmp_eq_u32_sdwa vcc, v87, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v88, v85, v87, vcc
; %bb.178:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v84, 0x7f800000, v80
	v_cmp_ne_u32_e32 vcc, s0, v84
	global_store_short_d16_hi v[94:95], v88, off offset:64
                                        ; implicit-def: $vgpr84
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.179:
	v_bfe_u32 v84, v80, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v80, v84, s2
; %bb.180:                              ; %Flow4699
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.181:
	v_mov_b32_e32 v84, 0
	v_or_b32_e32 v85, 0x10000, v80
	v_cmp_eq_u32_sdwa vcc, v80, v84 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v84, v85, v80, vcc
; %bb.182:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v81
	v_cmp_ne_u32_e32 vcc, s0, v80
	global_store_short_d16_hi v[102:103], v84, off offset:96
                                        ; implicit-def: $vgpr80
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.183:
	v_bfe_u32 v80, v81, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v80, v81, v80, s2
; %bb.184:                              ; %Flow4698
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.185:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v84, 0x10000, v81
	v_cmp_eq_u32_sdwa vcc, v81, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v80, v84, v81, vcc
; %bb.186:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[104:105], v80, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v82
	v_cmp_ne_u32_e32 vcc, s0, v80
                                        ; implicit-def: $vgpr80
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.187:
	v_bfe_u32 v80, v82, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v80, v82, v80, s2
; %bb.188:                              ; %Flow4697
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.189:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v81, 0x10000, v82
	v_cmp_eq_u32_sdwa vcc, v82, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v80, v81, v82, vcc
; %bb.190:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[106:107], v80, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v83
	v_cmp_ne_u32_e32 vcc, s0, v80
                                        ; implicit-def: $vgpr84
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.191:
	v_bfe_u32 v80, v83, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v83, v80, s2
                                        ; implicit-def: $vgpr80_vgpr81_vgpr82_vgpr83
; %bb.192:                              ; %Flow4696
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.193:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v81, 0x10000, v83
	v_cmp_eq_u32_sdwa vcc, v83, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v84, v81, v83, vcc
; %bb.194:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3.2.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v80, 0x7f800000, v76
	v_cmp_ne_u32_e32 vcc, s0, v80
	global_store_short_d16_hi v[94:95], v84, off offset:96
                                        ; implicit-def: $vgpr82
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.195:
	v_bfe_u32 v80, v76, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v82, v76, v80, s2
; %bb.196:                              ; %Flow4695
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.197:
	v_mov_b32_e32 v80, 0
	v_or_b32_e32 v81, 0x10000, v76
	v_cmp_eq_u32_sdwa vcc, v76, v80 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v82, v81, v76, vcc
; %bb.198:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3130.i.i
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v76, 48, v149
	v_mad_u64_u32 v[80:81], s[0:1], v76, s10, v[148:149]
	v_ashrrev_i32_e32 v81, 31, v80
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v77
	v_lshl_add_u64 v[86:87], v[80:81], 1, v[136:137]
	v_cmp_ne_u32_e32 vcc, s0, v76
	global_store_short_d16_hi v[86:87], v82, off
                                        ; implicit-def: $vgpr76
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.199:
	v_bfe_u32 v76, v77, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v76, v77, v76, s2
; %bb.200:                              ; %Flow4694
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.201:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v82, 0x10000, v77
	v_cmp_eq_u32_sdwa vcc, v77, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v76, v82, v77, vcc
; %bb.202:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3135.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v82, s10, v80
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[88:89], v[82:83], 1, v[136:137]
	global_store_short_d16_hi v[88:89], v76, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v78
	v_cmp_ne_u32_e32 vcc, s0, v76
                                        ; implicit-def: $vgpr76
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.203:
	v_bfe_u32 v76, v78, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v76, v78, v76, s2
; %bb.204:                              ; %Flow4693
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.205:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v77, 0x10000, v78
	v_cmp_eq_u32_sdwa vcc, v78, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v76, v77, v78, vcc
; %bb.206:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3140.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v84, s10, v82
	v_ashrrev_i32_e32 v85, 31, v84
	v_lshl_add_u64 v[90:91], v[84:85], 1, v[136:137]
	global_store_short_d16_hi v[90:91], v76, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v76, 0x7f800000, v79
	v_cmp_ne_u32_e32 vcc, s0, v76
                                        ; implicit-def: $vgpr94
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.207:
	v_bfe_u32 v76, v79, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v94, v79, v76, s2
                                        ; implicit-def: $vgpr76_vgpr77_vgpr78_vgpr79
; %bb.208:                              ; %Flow4692
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.209:
	v_mov_b32_e32 v76, 0
	v_or_b32_e32 v77, 0x10000, v79
	v_cmp_eq_u32_sdwa vcc, v79, v76 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v94, v77, v79, vcc
; %bb.210:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3143.i.i
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v76, s10, v84
	v_ashrrev_i32_e32 v77, 31, v76
	v_lshl_add_u64 v[78:79], v[76:77], 1, v[136:137]
	global_store_short_d16_hi v[78:79], v94, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v94, 0x7f800000, v72
	v_cmp_ne_u32_e32 vcc, s0, v94
                                        ; implicit-def: $vgpr94
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.211:
	v_bfe_u32 v94, v72, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v94, v72, v94, s2
; %bb.212:                              ; %Flow4691
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.213:
	v_mov_b32_e32 v94, 0
	v_or_b32_e32 v95, 0x10000, v72
	v_cmp_eq_u32_sdwa vcc, v72, v94 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v94, v95, v72, vcc
; %bb.214:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v73
	v_cmp_ne_u32_e32 vcc, s0, v72
	global_store_short_d16_hi v[86:87], v94, off offset:32
                                        ; implicit-def: $vgpr72
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.215:
	v_bfe_u32 v72, v73, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v72, v73, v72, s2
; %bb.216:                              ; %Flow4690
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.217:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v94, 0x10000, v73
	v_cmp_eq_u32_sdwa vcc, v73, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v72, v94, v73, vcc
; %bb.218:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[88:89], v72, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v74
	v_cmp_ne_u32_e32 vcc, s0, v72
                                        ; implicit-def: $vgpr72
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.219:
	v_bfe_u32 v72, v74, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v72, v74, v72, s2
; %bb.220:                              ; %Flow4689
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.221:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v73, 0x10000, v74
	v_cmp_eq_u32_sdwa vcc, v74, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v72, v73, v74, vcc
; %bb.222:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[90:91], v72, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v75
	v_cmp_ne_u32_e32 vcc, s0, v72
                                        ; implicit-def: $vgpr94
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.223:
	v_bfe_u32 v72, v75, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v94, v75, v72, s2
                                        ; implicit-def: $vgpr72_vgpr73_vgpr74_vgpr75
; %bb.224:                              ; %Flow4688
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.225:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v73, 0x10000, v75
	v_cmp_eq_u32_sdwa vcc, v75, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v94, v73, v75, vcc
; %bb.226:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v72, 0x7f800000, v68
	v_cmp_ne_u32_e32 vcc, s0, v72
	global_store_short_d16_hi v[78:79], v94, off offset:32
                                        ; implicit-def: $vgpr72
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.227:
	v_bfe_u32 v72, v68, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v72, v68, v72, s2
; %bb.228:                              ; %Flow4687
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.229:
	v_mov_b32_e32 v72, 0
	v_or_b32_e32 v73, 0x10000, v68
	v_cmp_eq_u32_sdwa vcc, v68, v72 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v72, v73, v68, vcc
; %bb.230:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v69
	v_cmp_ne_u32_e32 vcc, s0, v68
	global_store_short_d16_hi v[86:87], v72, off offset:64
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.231:
	v_bfe_u32 v68, v69, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v69, v68, s2
; %bb.232:                              ; %Flow4686
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.233:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v72, 0x10000, v69
	v_cmp_eq_u32_sdwa vcc, v69, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v72, v69, vcc
; %bb.234:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[88:89], v68, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v70
	v_cmp_ne_u32_e32 vcc, s0, v68
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.235:
	v_bfe_u32 v68, v70, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v70, v68, s2
; %bb.236:                              ; %Flow4685
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.237:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v69, 0x10000, v70
	v_cmp_eq_u32_sdwa vcc, v70, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v69, v70, vcc
; %bb.238:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[90:91], v68, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v71
	v_cmp_ne_u32_e32 vcc, s0, v68
                                        ; implicit-def: $vgpr72
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.239:
	v_bfe_u32 v68, v71, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v72, v71, v68, s2
                                        ; implicit-def: $vgpr68_vgpr69_vgpr70_vgpr71
; %bb.240:                              ; %Flow4684
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.241:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v69, 0x10000, v71
	v_cmp_eq_u32_sdwa vcc, v71, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v72, v69, v71, vcc
; %bb.242:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v68, 0x7f800000, v64
	v_cmp_ne_u32_e32 vcc, s0, v68
	global_store_short_d16_hi v[78:79], v72, off offset:64
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.243:
	v_bfe_u32 v68, v64, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v64, v68, s2
; %bb.244:                              ; %Flow4683
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.245:
	v_mov_b32_e32 v68, 0
	v_or_b32_e32 v69, 0x10000, v64
	v_cmp_eq_u32_sdwa vcc, v64, v68 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v69, v64, vcc
; %bb.246:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.3.i.i
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v65
	v_cmp_ne_u32_e32 vcc, s0, v64
	global_store_short_d16_hi v[86:87], v68, off offset:96
                                        ; implicit-def: $vgpr64
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.247:
	v_bfe_u32 v64, v65, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v64, v65, v64, s2
; %bb.248:                              ; %Flow4682
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.249:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v68, 0x10000, v65
	v_cmp_eq_u32_sdwa vcc, v65, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v64, v68, v65, vcc
; %bb.250:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[88:89], v64, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v66
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr64
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.251:
	v_bfe_u32 v64, v66, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v64, v66, v64, s2
; %bb.252:                              ; %Flow4681
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.253:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v66
	v_cmp_eq_u32_sdwa vcc, v66, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v64, v65, v66, vcc
; %bb.254:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.3.i.i
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[90:91], v64, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v67
	v_cmp_ne_u32_e32 vcc, s0, v64
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.255:
	v_bfe_u32 v64, v67, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v67, v64, s2
                                        ; implicit-def: $vgpr64_vgpr65_vgpr66_vgpr67
; %bb.256:                              ; %Flow4680
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.257:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v67
	v_cmp_eq_u32_sdwa vcc, v67, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v65, v67, vcc
; %bb.258:                              ; %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi64ELi64ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v64, 0x7f800000, v60
	v_cmp_ne_u32_e32 vcc, s0, v64
	global_store_short_d16_hi v[78:79], v68, off offset:96
                                        ; implicit-def: $vgpr68
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.259:
	v_bfe_u32 v64, v60, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v68, v60, v64, s2
; %bb.260:                              ; %Flow4679
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.261:
	v_mov_b32_e32 v64, 0
	v_or_b32_e32 v65, 0x10000, v60
	v_cmp_eq_u32_sdwa vcc, v60, v64 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v68, v65, v60, vcc
; %bb.262:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i1033
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v60, 0x80, v146
	v_ashrrev_i32_e32 v64, 31, v60
	v_mul_lo_u32 v66, s10, v64
	v_mul_lo_u32 v67, s11, v60
	v_mad_u64_u32 v[64:65], s[0:1], s10, v60, 0
	v_add3_u32 v65, v65, v66, v67
	v_lshl_add_u64 v[64:65], v[64:65], 1, s[8:9]
	v_lshl_add_u64 v[64:65], v[128:129], 1, v[64:65]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v61
	v_lshl_add_u64 v[66:67], v[130:131], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v60
	global_store_short_d16_hi v[66:67], v68, off
                                        ; implicit-def: $vgpr60
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.263:
	v_bfe_u32 v60, v61, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v60, v61, v60, s2
; %bb.264:                              ; %Flow4678
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.265:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v68, 0x10000, v61
	v_cmp_eq_u32_sdwa vcc, v61, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v60, v68, v61, vcc
; %bb.266:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.i.i1037
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[68:69], v[132:133], 1, v[64:65]
	global_store_short_d16_hi v[68:69], v60, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v62
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr60
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.267:
	v_bfe_u32 v60, v62, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v60, v62, v60, s2
; %bb.268:                              ; %Flow4677
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.269:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v61, 0x10000, v62
	v_cmp_eq_u32_sdwa vcc, v62, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v60, v61, v62, vcc
; %bb.270:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.i.i1041
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[70:71], v[134:135], 1, v[64:65]
	global_store_short_d16_hi v[70:71], v60, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v60, 0x7f800000, v63
	v_cmp_ne_u32_e32 vcc, s0, v60
                                        ; implicit-def: $vgpr72
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.271:
	v_bfe_u32 v60, v63, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v72, v63, v60, s2
                                        ; implicit-def: $vgpr60_vgpr61_vgpr62_vgpr63
; %bb.272:                              ; %Flow4676
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.273:
	v_mov_b32_e32 v60, 0
	v_or_b32_e32 v61, 0x10000, v63
	v_cmp_eq_u32_sdwa vcc, v63, v60 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v72, v61, v63, vcc
; %bb.274:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.i.i1045
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v62, 0x7f800000, v56
	v_lshl_add_u64 v[60:61], v[124:125], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v62
	global_store_short_d16_hi v[60:61], v72, off
                                        ; implicit-def: $vgpr62
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.275:
	v_bfe_u32 v62, v56, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v62, v56, v62, s2
; %bb.276:                              ; %Flow4675
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.277:
	v_mov_b32_e32 v62, 0
	v_or_b32_e32 v63, 0x10000, v56
	v_cmp_eq_u32_sdwa vcc, v56, v62 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v62, v63, v56, vcc
; %bb.278:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i1049
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v57
	v_cmp_ne_u32_e32 vcc, s0, v56
	global_store_short_d16_hi v[66:67], v62, off offset:32
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.279:
	v_bfe_u32 v56, v57, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v56, v57, v56, s2
; %bb.280:                              ; %Flow4674
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.281:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v62, 0x10000, v57
	v_cmp_eq_u32_sdwa vcc, v57, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v56, v62, v57, vcc
; %bb.282:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.i.i1053
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[68:69], v56, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v58
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.283:
	v_bfe_u32 v56, v58, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v56, v58, v56, s2
; %bb.284:                              ; %Flow4673
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.285:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v58
	v_cmp_eq_u32_sdwa vcc, v58, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v56, v57, v58, vcc
; %bb.286:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.i.i1057
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[70:71], v56, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v59
	v_cmp_ne_u32_e32 vcc, s0, v56
                                        ; implicit-def: $vgpr62
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.287:
	v_bfe_u32 v56, v59, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v62, v59, v56, s2
                                        ; implicit-def: $vgpr56_vgpr57_vgpr58_vgpr59
; %bb.288:                              ; %Flow4672
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.289:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v59
	v_cmp_eq_u32_sdwa vcc, v59, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v62, v57, v59, vcc
; %bb.290:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.i.i1061
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v56, 0x7f800000, v52
	v_cmp_ne_u32_e32 vcc, s0, v56
	global_store_short_d16_hi v[60:61], v62, off offset:32
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.291:
	v_bfe_u32 v56, v52, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v56, v52, v56, s2
; %bb.292:                              ; %Flow4671
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.293:
	v_mov_b32_e32 v56, 0
	v_or_b32_e32 v57, 0x10000, v52
	v_cmp_eq_u32_sdwa vcc, v52, v56 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v56, v57, v52, vcc
; %bb.294:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.i.i1065
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v53
	v_cmp_ne_u32_e32 vcc, s0, v52
	global_store_short_d16_hi v[66:67], v56, off offset:64
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.295:
	v_bfe_u32 v52, v53, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v53, v52, s2
; %bb.296:                              ; %Flow4670
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.297:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v56, 0x10000, v53
	v_cmp_eq_u32_sdwa vcc, v53, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v56, v53, vcc
; %bb.298:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.i.i1069
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[68:69], v52, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v54
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.299:
	v_bfe_u32 v52, v54, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v54, v52, s2
; %bb.300:                              ; %Flow4669
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.301:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v53, 0x10000, v54
	v_cmp_eq_u32_sdwa vcc, v54, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v53, v54, vcc
; %bb.302:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.i.i1073
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[70:71], v52, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v55
	v_cmp_ne_u32_e32 vcc, s0, v52
                                        ; implicit-def: $vgpr56
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.303:
	v_bfe_u32 v52, v55, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v56, v55, v52, s2
                                        ; implicit-def: $vgpr52_vgpr53_vgpr54_vgpr55
; %bb.304:                              ; %Flow4668
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.305:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v53, 0x10000, v55
	v_cmp_eq_u32_sdwa vcc, v55, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v56, v53, v55, vcc
; %bb.306:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.i.i1077
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v52, 0x7f800000, v48
	v_cmp_ne_u32_e32 vcc, s0, v52
	global_store_short_d16_hi v[60:61], v56, off offset:64
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.307:
	v_bfe_u32 v52, v48, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v48, v52, s2
; %bb.308:                              ; %Flow4667
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.309:
	v_mov_b32_e32 v52, 0
	v_or_b32_e32 v53, 0x10000, v48
	v_cmp_eq_u32_sdwa vcc, v48, v52 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v53, v48, vcc
; %bb.310:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.i.i1081
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v49
	v_cmp_ne_u32_e32 vcc, s0, v48
	global_store_short_d16_hi v[66:67], v52, off offset:96
                                        ; implicit-def: $vgpr48
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.311:
	v_bfe_u32 v48, v49, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v48, v49, v48, s2
; %bb.312:                              ; %Flow4666
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.313:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v52, 0x10000, v49
	v_cmp_eq_u32_sdwa vcc, v49, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v48, v52, v49, vcc
; %bb.314:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.i.i1085
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[68:69], v48, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v50
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr48
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.315:
	v_bfe_u32 v48, v50, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v48, v50, v48, s2
; %bb.316:                              ; %Flow4665
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.317:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v49, 0x10000, v50
	v_cmp_eq_u32_sdwa vcc, v50, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v48, v49, v50, vcc
; %bb.318:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.i.i1089
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[70:71], v48, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v51
	v_cmp_ne_u32_e32 vcc, s0, v48
                                        ; implicit-def: $vgpr52
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.319:
	v_bfe_u32 v48, v51, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v51, v48, s2
                                        ; implicit-def: $vgpr48_vgpr49_vgpr50_vgpr51
; %bb.320:                              ; %Flow4664
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.321:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v49, 0x10000, v51
	v_cmp_eq_u32_sdwa vcc, v51, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v52, v49, v51, vcc
; %bb.322:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3.i.i1093
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v48, 0x7f800000, v44
	v_cmp_ne_u32_e32 vcc, s0, v48
	global_store_short_d16_hi v[60:61], v52, off offset:96
                                        ; implicit-def: $vgpr50
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.323:
	v_bfe_u32 v48, v44, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v50, v44, v48, s2
; %bb.324:                              ; %Flow4663
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.325:
	v_mov_b32_e32 v48, 0
	v_or_b32_e32 v49, 0x10000, v44
	v_cmp_eq_u32_sdwa vcc, v44, v48 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v50, v49, v44, vcc
; %bb.326:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.190.i.i1097
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v45
	v_lshl_add_u64 v[48:49], v[112:113], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v44
	global_store_short_d16_hi v[48:49], v50, off
                                        ; implicit-def: $vgpr44
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.327:
	v_bfe_u32 v44, v45, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v44, v45, v44, s2
; %bb.328:                              ; %Flow4662
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.329:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v50, 0x10000, v45
	v_cmp_eq_u32_sdwa vcc, v45, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v44, v50, v45, vcc
; %bb.330:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.195.i.i1101
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[50:51], v[114:115], 1, v[64:65]
	global_store_short_d16_hi v[50:51], v44, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v46
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr44
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.331:
	v_bfe_u32 v44, v46, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v44, v46, v44, s2
; %bb.332:                              ; %Flow4661
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.333:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v45, 0x10000, v46
	v_cmp_eq_u32_sdwa vcc, v46, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v44, v45, v46, vcc
; %bb.334:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1100.i.i1105
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[52:53], v[116:117], 1, v[64:65]
	global_store_short_d16_hi v[52:53], v44, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v44, 0x7f800000, v47
	v_cmp_ne_u32_e32 vcc, s0, v44
                                        ; implicit-def: $vgpr54
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.335:
	v_bfe_u32 v44, v47, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v54, v47, v44, s2
                                        ; implicit-def: $vgpr44_vgpr45_vgpr46_vgpr47
; %bb.336:                              ; %Flow4660
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.337:
	v_mov_b32_e32 v44, 0
	v_or_b32_e32 v45, 0x10000, v47
	v_cmp_eq_u32_sdwa vcc, v47, v44 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v54, v45, v47, vcc
; %bb.338:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1103.i.i1109
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v46, 0x7f800000, v40
	v_lshl_add_u64 v[44:45], v[108:109], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v46
	global_store_short_d16_hi v[44:45], v54, off
                                        ; implicit-def: $vgpr46
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.339:
	v_bfe_u32 v46, v40, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v46, v40, v46, s2
; %bb.340:                              ; %Flow4659
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.341:
	v_mov_b32_e32 v46, 0
	v_or_b32_e32 v47, 0x10000, v40
	v_cmp_eq_u32_sdwa vcc, v40, v46 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v46, v47, v40, vcc
; %bb.342:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i1113
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v41
	v_cmp_ne_u32_e32 vcc, s0, v40
	global_store_short_d16_hi v[48:49], v46, off offset:32
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.343:
	v_bfe_u32 v40, v41, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v41, v40, s2
; %bb.344:                              ; %Flow4658
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.345:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v46, 0x10000, v41
	v_cmp_eq_u32_sdwa vcc, v41, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v46, v41, vcc
; %bb.346:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.1.i.i1117
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[50:51], v40, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v42
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.347:
	v_bfe_u32 v40, v42, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v42, v40, s2
; %bb.348:                              ; %Flow4657
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.349:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v41, 0x10000, v42
	v_cmp_eq_u32_sdwa vcc, v42, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v41, v42, vcc
; %bb.350:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.1.i.i1121
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[52:53], v40, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v43
	v_cmp_ne_u32_e32 vcc, s0, v40
                                        ; implicit-def: $vgpr46
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.351:
	v_bfe_u32 v40, v43, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v46, v43, v40, s2
                                        ; implicit-def: $vgpr40_vgpr41_vgpr42_vgpr43
; %bb.352:                              ; %Flow4656
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.353:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v41, 0x10000, v43
	v_cmp_eq_u32_sdwa vcc, v43, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v46, v41, v43, vcc
; %bb.354:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.1.i.i1125
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v40, 0x7f800000, v36
	v_cmp_ne_u32_e32 vcc, s0, v40
	global_store_short_d16_hi v[44:45], v46, off offset:32
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.355:
	v_bfe_u32 v40, v36, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v36, v40, s2
; %bb.356:                              ; %Flow4655
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.357:
	v_mov_b32_e32 v40, 0
	v_or_b32_e32 v41, 0x10000, v36
	v_cmp_eq_u32_sdwa vcc, v36, v40 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v41, v36, vcc
; %bb.358:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.1.i.i1129
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v37
	v_cmp_ne_u32_e32 vcc, s0, v36
	global_store_short_d16_hi v[48:49], v40, off offset:64
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.359:
	v_bfe_u32 v36, v37, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v37, v36, s2
; %bb.360:                              ; %Flow4654
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.361:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v40, 0x10000, v37
	v_cmp_eq_u32_sdwa vcc, v37, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v40, v37, vcc
; %bb.362:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.1.i.i1133
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[50:51], v36, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v38
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.363:
	v_bfe_u32 v36, v38, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v38, v36, s2
; %bb.364:                              ; %Flow4653
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.365:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v37, 0x10000, v38
	v_cmp_eq_u32_sdwa vcc, v38, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v37, v38, vcc
; %bb.366:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.1.i.i1137
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[52:53], v36, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v39
	v_cmp_ne_u32_e32 vcc, s0, v36
                                        ; implicit-def: $vgpr40
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.367:
	v_bfe_u32 v36, v39, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v40, v39, v36, s2
                                        ; implicit-def: $vgpr36_vgpr37_vgpr38_vgpr39
; %bb.368:                              ; %Flow4652
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.369:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v37, 0x10000, v39
	v_cmp_eq_u32_sdwa vcc, v39, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v40, v37, v39, vcc
; %bb.370:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.1.i.i1141
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v36, 0x7f800000, v32
	v_cmp_ne_u32_e32 vcc, s0, v36
	global_store_short_d16_hi v[44:45], v40, off offset:64
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.371:
	v_bfe_u32 v36, v32, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v32, v36, s2
; %bb.372:                              ; %Flow4651
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.373:
	v_mov_b32_e32 v36, 0
	v_or_b32_e32 v37, 0x10000, v32
	v_cmp_eq_u32_sdwa vcc, v32, v36 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v37, v32, vcc
; %bb.374:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.1.i.i1145
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v33
	v_cmp_ne_u32_e32 vcc, s0, v32
	global_store_short_d16_hi v[48:49], v36, off offset:96
                                        ; implicit-def: $vgpr32
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.375:
	v_bfe_u32 v32, v33, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v32, v33, v32, s2
; %bb.376:                              ; %Flow4650
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.377:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v36, 0x10000, v33
	v_cmp_eq_u32_sdwa vcc, v33, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v32, v36, v33, vcc
; %bb.378:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.1.i.i1149
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[50:51], v32, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v34
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr32
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.379:
	v_bfe_u32 v32, v34, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v32, v34, v32, s2
; %bb.380:                              ; %Flow4649
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.381:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v33, 0x10000, v34
	v_cmp_eq_u32_sdwa vcc, v34, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v32, v33, v34, vcc
; %bb.382:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.1.i.i1153
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[52:53], v32, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v35
	v_cmp_ne_u32_e32 vcc, s0, v32
                                        ; implicit-def: $vgpr36
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.383:
	v_bfe_u32 v32, v35, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v36, v35, v32, s2
                                        ; implicit-def: $vgpr32_vgpr33_vgpr34_vgpr35
; %bb.384:                              ; %Flow4648
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.385:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v33, 0x10000, v35
	v_cmp_eq_u32_sdwa vcc, v35, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v36, v33, v35, vcc
; %bb.386:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3.1.i.i1157
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v32, 0x7f800000, v28
	v_cmp_ne_u32_e32 vcc, s0, v32
	global_store_short_d16_hi v[44:45], v36, off offset:96
                                        ; implicit-def: $vgpr34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.387:
	v_bfe_u32 v32, v28, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v34, v28, v32, s2
; %bb.388:                              ; %Flow4647
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.389:
	v_mov_b32_e32 v32, 0
	v_or_b32_e32 v33, 0x10000, v28
	v_cmp_eq_u32_sdwa vcc, v28, v32 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v34, v33, v28, vcc
; %bb.390:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2110.i.i1161
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v29
	v_lshl_add_u64 v[32:33], v[96:97], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v28
	global_store_short_d16_hi v[32:33], v34, off
                                        ; implicit-def: $vgpr28
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.391:
	v_bfe_u32 v28, v29, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v28, v29, v28, s2
; %bb.392:                              ; %Flow4646
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.393:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v34, 0x10000, v29
	v_cmp_eq_u32_sdwa vcc, v29, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v28, v34, v29, vcc
; %bb.394:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2115.i.i1165
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[34:35], v[98:99], 1, v[64:65]
	global_store_short_d16_hi v[34:35], v28, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v30
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr28
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.395:
	v_bfe_u32 v28, v30, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v28, v30, v28, s2
; %bb.396:                              ; %Flow4645
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.397:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v29, 0x10000, v30
	v_cmp_eq_u32_sdwa vcc, v30, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v28, v29, v30, vcc
; %bb.398:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2120.i.i1169
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[36:37], v[100:101], 1, v[64:65]
	global_store_short_d16_hi v[36:37], v28, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v28, 0x7f800000, v31
	v_cmp_ne_u32_e32 vcc, s0, v28
                                        ; implicit-def: $vgpr38
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.399:
	v_bfe_u32 v28, v31, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v38, v31, v28, s2
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
; %bb.400:                              ; %Flow4644
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.401:
	v_mov_b32_e32 v28, 0
	v_or_b32_e32 v29, 0x10000, v31
	v_cmp_eq_u32_sdwa vcc, v31, v28 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v38, v29, v31, vcc
; %bb.402:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2123.i.i1173
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v30, 0x7f800000, v24
	v_lshl_add_u64 v[28:29], v[92:93], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v30
	global_store_short_d16_hi v[28:29], v38, off
                                        ; implicit-def: $vgpr30
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.403:
	v_bfe_u32 v30, v24, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v30, v24, v30, s2
; %bb.404:                              ; %Flow4643
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.405:
	v_mov_b32_e32 v30, 0
	v_or_b32_e32 v31, 0x10000, v24
	v_cmp_eq_u32_sdwa vcc, v24, v30 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v30, v31, v24, vcc
; %bb.406:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.2.i.i1177
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v25
	v_cmp_ne_u32_e32 vcc, s0, v24
	global_store_short_d16_hi v[32:33], v30, off offset:32
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.407:
	v_bfe_u32 v24, v25, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v25, v24, s2
; %bb.408:                              ; %Flow4642
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.409:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v30, 0x10000, v25
	v_cmp_eq_u32_sdwa vcc, v25, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v30, v25, vcc
; %bb.410:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.2.i.i1181
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[34:35], v24, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v26
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.411:
	v_bfe_u32 v24, v26, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v26, v24, s2
; %bb.412:                              ; %Flow4641
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.413:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v25, 0x10000, v26
	v_cmp_eq_u32_sdwa vcc, v26, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v25, v26, vcc
; %bb.414:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.2.i.i1185
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[36:37], v24, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v27
	v_cmp_ne_u32_e32 vcc, s0, v24
                                        ; implicit-def: $vgpr30
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.415:
	v_bfe_u32 v24, v27, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v30, v27, v24, s2
                                        ; implicit-def: $vgpr24_vgpr25_vgpr26_vgpr27
; %bb.416:                              ; %Flow4640
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.417:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v25, 0x10000, v27
	v_cmp_eq_u32_sdwa vcc, v27, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v30, v25, v27, vcc
; %bb.418:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.2.i.i1189
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v24, 0x7f800000, v20
	v_cmp_ne_u32_e32 vcc, s0, v24
	global_store_short_d16_hi v[28:29], v30, off offset:32
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.419:
	v_bfe_u32 v24, v20, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v20, v24, s2
; %bb.420:                              ; %Flow4639
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.421:
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v25, 0x10000, v20
	v_cmp_eq_u32_sdwa vcc, v20, v24 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v25, v20, vcc
; %bb.422:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.2.i.i1193
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v21
	v_cmp_ne_u32_e32 vcc, s0, v20
	global_store_short_d16_hi v[32:33], v24, off offset:64
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.423:
	v_bfe_u32 v20, v21, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v21, v20, s2
; %bb.424:                              ; %Flow4638
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.425:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v24, 0x10000, v21
	v_cmp_eq_u32_sdwa vcc, v21, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v24, v21, vcc
; %bb.426:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.2.i.i1197
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[34:35], v20, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v22
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.427:
	v_bfe_u32 v20, v22, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v22, v20, s2
; %bb.428:                              ; %Flow4637
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.429:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v21, 0x10000, v22
	v_cmp_eq_u32_sdwa vcc, v22, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v21, v22, vcc
; %bb.430:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.2.i.i1201
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[36:37], v20, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v23
	v_cmp_ne_u32_e32 vcc, s0, v20
                                        ; implicit-def: $vgpr24
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.431:
	v_bfe_u32 v20, v23, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v24, v23, v20, s2
                                        ; implicit-def: $vgpr20_vgpr21_vgpr22_vgpr23
; %bb.432:                              ; %Flow4636
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.433:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v21, 0x10000, v23
	v_cmp_eq_u32_sdwa vcc, v23, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v24, v21, v23, vcc
; %bb.434:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.2.i.i1205
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v20, 0x7f800000, v16
	v_cmp_ne_u32_e32 vcc, s0, v20
	global_store_short_d16_hi v[28:29], v24, off offset:64
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.435:
	v_bfe_u32 v20, v16, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v16, v20, s2
; %bb.436:                              ; %Flow4635
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.437:
	v_mov_b32_e32 v20, 0
	v_or_b32_e32 v21, 0x10000, v16
	v_cmp_eq_u32_sdwa vcc, v16, v20 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v21, v16, vcc
; %bb.438:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.2.i.i1209
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v17
	v_cmp_ne_u32_e32 vcc, s0, v16
	global_store_short_d16_hi v[32:33], v20, off offset:96
                                        ; implicit-def: $vgpr16
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.439:
	v_bfe_u32 v16, v17, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v16, v17, v16, s2
; %bb.440:                              ; %Flow4634
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.441:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v20, 0x10000, v17
	v_cmp_eq_u32_sdwa vcc, v17, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v16, v20, v17, vcc
; %bb.442:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.2.i.i1213
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[34:35], v16, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v18
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr16
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.443:
	v_bfe_u32 v16, v18, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v16, v18, v16, s2
; %bb.444:                              ; %Flow4633
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.445:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v17, 0x10000, v18
	v_cmp_eq_u32_sdwa vcc, v18, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v16, v17, v18, vcc
; %bb.446:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.2.i.i1217
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[36:37], v16, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v19
	v_cmp_ne_u32_e32 vcc, s0, v16
                                        ; implicit-def: $vgpr20
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.447:
	v_bfe_u32 v16, v19, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v20, v19, v16, s2
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
; %bb.448:                              ; %Flow4632
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.449:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v17, 0x10000, v19
	v_cmp_eq_u32_sdwa vcc, v19, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v20, v17, v19, vcc
; %bb.450:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3.2.i.i1221
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v16, 0x7f800000, v12
	v_cmp_ne_u32_e32 vcc, s0, v16
	global_store_short_d16_hi v[28:29], v20, off offset:96
                                        ; implicit-def: $vgpr18
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.451:
	v_bfe_u32 v16, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v12, v16, s2
; %bb.452:                              ; %Flow4631
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.453:
	v_mov_b32_e32 v16, 0
	v_or_b32_e32 v17, 0x10000, v12
	v_cmp_eq_u32_sdwa vcc, v12, v16 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v18, v17, v12, vcc
; %bb.454:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3130.i.i1225
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v13
	v_lshl_add_u64 v[16:17], v[80:81], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v12
	global_store_short_d16_hi v[16:17], v18, off
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.455:
	v_bfe_u32 v12, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v12, v13, v12, s2
; %bb.456:                              ; %Flow4630
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.457:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v18, 0x10000, v13
	v_cmp_eq_u32_sdwa vcc, v13, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v12, v18, v13, vcc
; %bb.458:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3135.i.i1229
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[18:19], v[82:83], 1, v[64:65]
	global_store_short_d16_hi v[18:19], v12, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v14
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr12
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.459:
	v_bfe_u32 v12, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v12, v14, v12, s2
; %bb.460:                              ; %Flow4629
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.461:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v13, 0x10000, v14
	v_cmp_eq_u32_sdwa vcc, v14, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v12, v13, v14, vcc
; %bb.462:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3140.i.i1233
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[20:21], v[84:85], 1, v[64:65]
	global_store_short_d16_hi v[20:21], v12, off
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v12, 0x7f800000, v15
	v_cmp_ne_u32_e32 vcc, s0, v12
                                        ; implicit-def: $vgpr22
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.463:
	v_bfe_u32 v12, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v22, v15, v12, s2
                                        ; implicit-def: $vgpr12_vgpr13_vgpr14_vgpr15
; %bb.464:                              ; %Flow4628
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.465:
	v_mov_b32_e32 v12, 0
	v_or_b32_e32 v13, 0x10000, v15
	v_cmp_eq_u32_sdwa vcc, v15, v12 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v22, v13, v15, vcc
; %bb.466:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.3143.i.i1237
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v14, 0x7f800000, v8
	v_lshl_add_u64 v[12:13], v[76:77], 1, v[64:65]
	v_cmp_ne_u32_e32 vcc, s0, v14
	global_store_short_d16_hi v[12:13], v22, off
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.467:
	v_bfe_u32 v14, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v14, v8, v14, s2
; %bb.468:                              ; %Flow4627
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.469:
	v_mov_b32_e32 v14, 0
	v_or_b32_e32 v15, 0x10000, v8
	v_cmp_eq_u32_sdwa vcc, v8, v14 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v14, v15, v8, vcc
; %bb.470:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.3.i.i1241
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v9
	v_cmp_ne_u32_e32 vcc, s0, v8
	global_store_short_d16_hi v[16:17], v14, off offset:32
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.471:
	v_bfe_u32 v8, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v8, v9, v8, s2
; %bb.472:                              ; %Flow4626
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.473:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v14, 0x10000, v9
	v_cmp_eq_u32_sdwa vcc, v9, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v8, v14, v9, vcc
; %bb.474:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.1.3.i.i1245
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[18:19], v8, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v10
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.475:
	v_bfe_u32 v8, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v8, v10, v8, s2
; %bb.476:                              ; %Flow4625
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.477:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v9, 0x10000, v10
	v_cmp_eq_u32_sdwa vcc, v10, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v8, v9, v10, vcc
; %bb.478:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.1.3.i.i1249
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[20:21], v8, off offset:32
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v11
	v_cmp_ne_u32_e32 vcc, s0, v8
                                        ; implicit-def: $vgpr14
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.479:
	v_bfe_u32 v8, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v14, v11, v8, s2
                                        ; implicit-def: $vgpr8_vgpr9_vgpr10_vgpr11
; %bb.480:                              ; %Flow4624
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.481:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v9, 0x10000, v11
	v_cmp_eq_u32_sdwa vcc, v11, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v14, v9, v11, vcc
; %bb.482:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.1.3.i.i1253
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v8, 0x7f800000, v4
	v_cmp_ne_u32_e32 vcc, s0, v8
	global_store_short_d16_hi v[12:13], v14, off offset:32
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.483:
	v_bfe_u32 v8, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v8, v4, v8, s2
; %bb.484:                              ; %Flow4623
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.485:
	v_mov_b32_e32 v8, 0
	v_or_b32_e32 v9, 0x10000, v4
	v_cmp_eq_u32_sdwa vcc, v4, v8 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v8, v9, v4, vcc
; %bb.486:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.2.3.i.i1257
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v5
	v_cmp_ne_u32_e32 vcc, s0, v4
	global_store_short_d16_hi v[16:17], v8, off offset:64
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.487:
	v_bfe_u32 v4, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v5, v4, s2
; %bb.488:                              ; %Flow4622
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.489:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v8, 0x10000, v5
	v_cmp_eq_u32_sdwa vcc, v5, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v4, v8, v5, vcc
; %bb.490:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.2.3.i.i1261
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[18:19], v4, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v6
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.491:
	v_bfe_u32 v4, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v6, v4, s2
; %bb.492:                              ; %Flow4621
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.493:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v5, 0x10000, v6
	v_cmp_eq_u32_sdwa vcc, v6, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v4, v5, v6, vcc
; %bb.494:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.2.3.i.i1265
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[20:21], v4, off offset:64
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v7
	v_cmp_ne_u32_e32 vcc, s0, v4
                                        ; implicit-def: $vgpr8
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.495:
	v_bfe_u32 v4, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v8, v7, v4, s2
                                        ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
; %bb.496:                              ; %Flow4620
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.497:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v5, 0x10000, v7
	v_cmp_eq_u32_sdwa vcc, v7, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v8, v5, v7, vcc
; %bb.498:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit77.2.3.i.i1269
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v4, 0x7f800000, v0
	v_cmp_ne_u32_e32 vcc, s0, v4
	global_store_short_d16_hi v[12:13], v8, off offset:64
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.499:
	v_bfe_u32 v4, v0, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v0, v4, s2
; %bb.500:                              ; %Flow4619
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.501:
	v_mov_b32_e32 v4, 0
	v_or_b32_e32 v5, 0x10000, v0
	v_cmp_eq_u32_sdwa vcc, v0, v4 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v4, v5, v0, vcc
; %bb.502:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.3.3.i.i1273
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v1
	v_cmp_ne_u32_e32 vcc, s0, v0
	global_store_short_d16_hi v[16:17], v4, off offset:96
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.503:
	v_bfe_u32 v0, v1, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v1, v0, s2
; %bb.504:                              ; %Flow4618
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.505:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v4, 0x10000, v1
	v_cmp_eq_u32_sdwa vcc, v1, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v4, v1, vcc
; %bb.506:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit69.3.3.i.i1277
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[18:19], v0, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v2
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
; %bb.507:
	v_bfe_u32 v0, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v0, v2, v0, s2
; %bb.508:                              ; %Flow4617
	s_andn2_saveexec_b64 s[0:1], s[0:1]
; %bb.509:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v2
	v_cmp_eq_u32_sdwa vcc, v2, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v0, v1, v2, vcc
; %bb.510:                              ; %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit73.3.3.i.i1281
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[20:21], v0, off offset:96
	s_mov_b32 s0, 0x7f800000
	v_and_b32_e32 v0, 0x7f800000, v3
	v_cmp_ne_u32_e32 vcc, s0, v0
                                        ; implicit-def: $vgpr4
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execnz .LBB0_513
; %bb.511:                              ; %Flow
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execnz .LBB0_514
.LBB0_512:                              ; %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi64ELi64ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit1350
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[12:13], v4, off offset:96
	s_endpgm
.LBB0_513:
	v_bfe_u32 v0, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v4, v3, v0, s2
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_512
.LBB0_514:
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v1, 0x10000, v3
	v_cmp_eq_u32_sdwa vcc, v3, v0 src0_sel:WORD_0 src1_sel:DWORD
	s_nop 1
	v_cndmask_b32_e32 v4, v1, v3, vcc
	s_or_b64 exec, exec, s[0:1]
	global_store_short_d16_hi v[12:13], v4, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tk13micro_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 120
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 26
		.amdhsa_accum_offset 256
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
; codeLenInByte = 20676
; NumSgprs: 32
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 120
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_d0d904fa6b62238b,@object ; @__hip_cuid_d0d904fa6b62238b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d0d904fa6b62238b
__hip_cuid_d0d904fa6b62238b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d0d904fa6b62238b, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_d0d904fa6b62238b
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
    .private_segment_fixed_size: 120
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 29
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
