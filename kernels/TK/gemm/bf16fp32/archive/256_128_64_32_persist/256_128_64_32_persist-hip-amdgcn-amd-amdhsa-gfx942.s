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
	s_cselect_b32 s3, 0, 0
	s_and_b32 s4, s3, -16
	s_mov_b32 s21, 0
	s_and_b32 s20, s3, 15
	s_add_i32 s4, s4, 16
	s_cmp_eq_u64 s[20:21], 0
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_load_dwordx2 s[18:19], s[0:1], 0x50
	s_cselect_b32 s4, s3, s4
	s_load_dwordx2 s[22:23], s[0:1], 0x20
	s_load_dwordx2 s[28:29], s[0:1], 0x60
	s_add_i32 s5, s4, 0x4000
	s_and_b32 s6, s5, -16
	s_and_b32 s20, s5, 15
	s_add_i32 s6, s6, 16
	v_lshlrev_b32_e32 v2, 3, v0
	v_lshrrev_b32_e32 v3, 3, v0
	s_cmp_eq_u64 s[20:21], 0
	v_and_b32_e32 v2, 56, v2
	v_or_b32_e32 v4, 64, v3
	s_load_dwordx2 s[24:25], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[88:89], s[8:9], v3, s22, v[2:3]
	v_mad_u64_u32 v[90:91], s[8:9], v4, s22, v[2:3]
	v_lshlrev_b32_e32 v9, 4, v0
	v_lshlrev_b32_e32 v5, 1, v2
	v_mad_u64_u32 v[92:93], s[8:9], v3, s18, v[2:3]
	v_mad_u64_u32 v[94:95], s[8:9], v4, s18, v[2:3]
	v_or_b32_e32 v3, 0xc0, v3
	s_cselect_b32 s5, s5, s6
	s_load_dwordx2 s[26:27], s[0:1], 0x80
	v_and_b32_e32 v13, 0x1f80, v9
	v_mad_u64_u32 v[96:97], s[8:9], v3, s18, v[2:3]
	v_add_u32_e32 v2, s5, v5
	v_add_u32_e32 v4, v2, v13
	v_add_u32_e32 v6, s4, v5
	v_or_b32_e32 v7, 8, v5
	v_lshrrev_b32_e32 v5, 4, v4
	v_add_u32_e32 v3, s5, v7
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v201, v5, v4
	v_add_u32_e32 v4, v3, v13
	v_lshrrev_b32_e32 v5, 4, v4
	v_or_b32_e32 v17, 0x2000, v13
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v202, v5, v4
	v_add_u32_e32 v4, v2, v17
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v203, v5, v4
	v_add_u32_e32 v4, v3, v17
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_or_b32_e32 v21, 0x4000, v13
	v_xor_b32_e32 v204, v5, v4
	v_add_u32_e32 v4, v2, v21
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v205, v5, v4
	v_add_u32_e32 v4, v3, v21
	v_lshrrev_b32_e32 v5, 4, v4
	v_or_b32_e32 v25, 0x6000, v13
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v2, v2, v25
	v_xor_b32_e32 v206, v5, v4
	v_lshrrev_b32_e32 v4, 4, v2
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v207, v4, v2
	v_add_u32_e32 v2, v3, v25
	v_lshrrev_b32_e32 v3, 4, v2
	v_bfe_u32 v1, v0, 6, 2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v208, v3, v2
	v_lshlrev_b32_e32 v209, 5, v1
	v_and_b32_e32 v2, 15, v0
	v_lshrrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_or_b32_e32 v3, v209, v2
	v_lshlrev_b32_e32 v29, 1, v1
	v_lshl_add_u32 v33, v3, 7, s5
	v_add_u32_e32 v3, v33, v29
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 0x78, v4
	v_or_b32_e32 v35, 32, v29
	v_xor_b32_e32 v210, v4, v3
	v_add_u32_e32 v3, v33, v35
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 0x78, v4
	v_or_b32_e32 v36, 16, v2
	v_xor_b32_e32 v211, v4, v3
	v_or_b32_e32 v3, v209, v36
	v_lshl_add_u32 v37, v3, 7, s5
	v_add_u32_e32 v3, v37, v29
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v212, v4, v3
	v_add_u32_e32 v3, v37, v35
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v213, v4, v3
	v_or_b32_e32 v3, 0x80, v209
	v_or_b32_e32 v4, v3, v2
	v_lshl_add_u32 v38, v4, 7, s5
	v_add_u32_e32 v4, v38, v29
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v214, v5, v4
	v_add_u32_e32 v4, v38, v35
	v_or_b32_e32 v3, v3, v36
	v_lshrrev_b32_e32 v5, 4, v4
	v_lshl_add_u32 v39, v3, 7, s5
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v3, v39, v29
	v_xor_b32_e32 v215, v5, v4
	v_lshrrev_b32_e32 v4, 4, v3
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v216, v4, v3
	v_add_u32_e32 v3, v39, v35
	v_lshrrev_b32_e32 v4, 4, v3
	v_lshrrev_b32_e32 v43, 8, v0
	v_and_b32_e32 v4, 0x78, v4
	v_xor_b32_e32 v217, v4, v3
	v_lshlrev_b32_e32 v3, 5, v43
	v_or_b32_e32 v4, v3, v2
	v_lshl_add_u32 v40, v4, 7, s4
	v_add_u32_e32 v10, v6, v13
	v_add_u32_e32 v4, v40, v29
	v_lshrrev_b32_e32 v11, 4, v10
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v11, 0x78, v11
	v_and_b32_e32 v5, 0x78, v5
	v_add_u32_e32 v8, s4, v7
	v_xor_b32_e32 v10, v11, v10
	v_xor_b32_e32 v218, v5, v4
	v_add_u32_e32 v4, v40, v35
	scratch_store_dword off, v10, off       ; 4-byte Folded Spill
	v_add_u32_e32 v10, v8, v13
	v_lshrrev_b32_e32 v5, 4, v4
	v_lshrrev_b32_e32 v11, 4, v10
	v_and_b32_e32 v5, 0x78, v5
	v_and_b32_e32 v11, 0x78, v11
	v_xor_b32_e32 v219, v5, v4
	v_or_b32_e32 v4, v36, v3
	v_xor_b32_e32 v10, v11, v10
	v_add_u32_e32 v6, v6, v17
	v_lshl_add_u32 v41, v4, 7, s4
	scratch_store_dword off, v10, off offset:4 ; 4-byte Folded Spill
	v_lshrrev_b32_e32 v10, 4, v6
	v_add_u32_e32 v4, v41, v29
	v_and_b32_e32 v10, 0x78, v10
	v_lshrrev_b32_e32 v5, 4, v4
	v_xor_b32_e32 v6, v10, v6
	v_and_b32_e32 v5, 0x78, v5
	scratch_store_dword off, v6, off offset:8 ; 4-byte Folded Spill
	v_add_u32_e32 v6, v8, v17
	v_xor_b32_e32 v220, v5, v4
	v_add_u32_e32 v4, v41, v35
	v_lshrrev_b32_e32 v8, 4, v6
	v_lshrrev_b32_e32 v5, 4, v4
	v_and_b32_e32 v8, 0x78, v8
	v_and_b32_e32 v5, 0x78, v5
	v_xor_b32_e32 v6, v8, v6
	v_xor_b32_e32 v221, v5, v4
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v4, v1, s26
	scratch_store_dword off, v6, off offset:12 ; 4-byte Folded Spill
	v_add_u32_e32 v6, s26, v4
	v_add_u32_e32 v10, s26, v6
	v_add_u32_e32 v14, s26, v10
	s_load_dword s3, s[0:1], 0x90
	v_mad_u64_u32 v[18:19], s[0:1], s26, 13, v[14:15]
	v_add_u32_e32 v22, s26, v18
	v_or_b32_e32 v42, 64, v3
	v_add_u32_e32 v26, s26, v22
	v_add_u32_e32 v30, s26, v26
	v_or_b32_e32 v1, v42, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_mov_b32_e32 v3, 0
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v19, 31, v18
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v27, 31, v26
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u32 v1, v1, 7, s4
	v_add_u32_e32 v8, v4, v2
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[2:3]
	v_add_u32_e32 v12, v6, v2
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[2:3]
	v_add_u32_e32 v16, v10, v2
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[2:3]
	v_add_u32_e32 v20, v14, v2
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[2:3]
	v_add_u32_e32 v24, v18, v2
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[2:3]
	v_add_u32_e32 v28, v22, v2
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[2:3]
	v_add_u32_e32 v32, v26, v2
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[2:3]
	v_add_u32_e32 v34, v30, v2
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[2:3]
	v_add_u32_e32 v2, v1, v29
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v222, v3, v2
	v_add_u32_e32 v2, v1, v35
	v_lshrrev_b32_e32 v3, 4, v2
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v223, v3, v2
	v_or_b32_e32 v2, v42, v36
	v_lshl_add_u32 v2, v2, 7, s4
	v_add_u32_e32 v3, v2, v29
	v_lshrrev_b32_e32 v36, 4, v3
	v_and_b32_e32 v36, 0x78, v36
	v_xor_b32_e32 v224, v36, v3
	v_add_u32_e32 v3, v2, v35
	v_lshrrev_b32_e32 v35, 4, v3
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v225, v35, v3
	v_or_b32_e32 v3, 64, v29
	v_add_u32_e32 v35, v33, v3
	v_lshrrev_b32_e32 v36, 4, v35
	v_or_b32_e32 v29, 0x60, v29
	v_and_b32_e32 v36, 0x78, v36
	v_add_u32_e32 v33, v33, v29
	v_xor_b32_e32 v226, v36, v35
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v227, v35, v33
	v_add_u32_e32 v33, v37, v3
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v228, v35, v33
	v_add_u32_e32 v33, v37, v29
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v229, v35, v33
	v_add_u32_e32 v33, v38, v3
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v230, v35, v33
	v_add_u32_e32 v33, v38, v29
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v231, v35, v33
	v_add_u32_e32 v33, v39, v3
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v232, v35, v33
	v_add_u32_e32 v33, v39, v29
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v233, v35, v33
	v_add_u32_e32 v33, v40, v3
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v234, v35, v33
	v_add_u32_e32 v33, v40, v29
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v235, v35, v33
	v_add_u32_e32 v33, v41, v3
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v236, v35, v33
	v_add_u32_e32 v33, v41, v29
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_xor_b32_e32 v237, v35, v33
	v_add_u32_e32 v33, v1, v3
	v_lshrrev_b32_e32 v35, 4, v33
	v_and_b32_e32 v35, 0x78, v35
	v_add_u32_e32 v1, v1, v29
	v_xor_b32_e32 v238, v35, v33
	v_lshrrev_b32_e32 v33, 4, v1
	v_and_b32_e32 v33, 0x78, v33
	v_xor_b32_e32 v239, v33, v1
	v_add_u32_e32 v1, v2, v3
	v_lshrrev_b32_e32 v3, 4, v1
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v240, v3, v1
	v_add_u32_e32 v1, v2, v29
	v_lshrrev_b32_e32 v2, 4, v1
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v241, v2, v1
	v_and_b32_e32 v1, 0x70, v9
	v_add_u32_e32 v2, s4, v1
	v_add_u32_e32 v3, v2, v13
	v_lshrrev_b32_e32 v9, 4, v3
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v242, v9, v3
	v_or_b32_e32 v3, 8, v1
	v_add_u32_e32 v9, s4, v3
	v_add_u32_e32 v29, v9, v13
	v_lshrrev_b32_e32 v33, 4, v29
	v_and_b32_e32 v33, 0x78, v33
	v_add_u32_e32 v2, v2, v17
	v_xor_b32_e32 v243, v33, v29
	v_lshrrev_b32_e32 v29, 4, v2
	v_and_b32_e32 v29, 0x78, v29
	v_xor_b32_e32 v244, v29, v2
	v_add_u32_e32 v2, v9, v17
	v_lshrrev_b32_e32 v9, 4, v2
	v_and_b32_e32 v9, 0x78, v9
	v_add_u32_e32 v1, s5, v1
	v_xor_b32_e32 v245, v9, v2
	v_add_u32_e32 v2, v1, v13
	v_lshrrev_b32_e32 v9, 4, v2
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v246, v9, v2
	v_add_u32_e32 v2, s5, v3
	v_add_u32_e32 v3, v2, v13
	v_lshrrev_b32_e32 v9, 4, v3
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v247, v9, v3
	v_add_u32_e32 v3, v1, v17
	v_lshrrev_b32_e32 v9, 4, v3
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v248, v9, v3
	v_add_u32_e32 v3, v2, v17
	v_lshrrev_b32_e32 v9, 4, v3
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v249, v9, v3
	v_add_u32_e32 v3, v1, v21
	v_lshrrev_b32_e32 v9, 4, v3
	v_and_b32_e32 v9, 0x78, v9
	v_xor_b32_e32 v250, v9, v3
	v_add_u32_e32 v3, v2, v21
	v_lshrrev_b32_e32 v9, 4, v3
	v_and_b32_e32 v9, 0x78, v9
	v_add_u32_e32 v1, v1, v25
	v_xor_b32_e32 v251, v9, v3
	v_lshrrev_b32_e32 v3, 4, v1
	v_and_b32_e32 v3, 0x78, v3
	v_xor_b32_e32 v252, v3, v1
	v_add_u32_e32 v1, v2, v25
	v_lshrrev_b32_e32 v2, 4, v1
	v_and_b32_e32 v2, 0x78, v2
	v_xor_b32_e32 v253, v2, v1
	s_movk_i32 s0, 0x100
	v_lshl_add_u32 v98, s18, 7, v92
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v17, 31, v16
	v_ashrrev_i32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v29, 31, v28
	v_ashrrev_i32_e32 v33, 31, v32
	v_ashrrev_i32_e32 v35, 31, v34
	v_mov_b64_e32 v[2:3], 0
	v_cmp_gt_u32_e64 s[0:1], s0, v0
	v_ashrrev_i32_e32 v89, 31, v88
	v_ashrrev_i32_e32 v91, 31, v90
	v_ashrrev_i32_e32 v93, 31, v92
	v_ashrrev_i32_e32 v95, 31, v94
	v_ashrrev_i32_e32 v99, 31, v98
	v_ashrrev_i32_e32 v97, 31, v96
	v_mov_b32_e32 v200, v43
	v_cmp_eq_u32_e64 s[6:7], 1, v43
	s_lshl_b32 s10, s22, 8
	s_lshl_b32 s14, s18, 9
	s_mov_b32 s11, 0x110000
	s_mov_b32 s20, 0xffff
	v_lshlrev_b64 v[100:101], 2, v[8:9]
	v_lshlrev_b64 v[102:103], 2, v[12:13]
	v_lshlrev_b64 v[104:105], 2, v[16:17]
	v_lshlrev_b64 v[106:107], 2, v[20:21]
	v_lshlrev_b64 v[108:109], 2, v[4:5]
	v_lshlrev_b64 v[110:111], 2, v[6:7]
	v_lshlrev_b64 v[112:113], 2, v[10:11]
	v_lshlrev_b64 v[114:115], 2, v[14:15]
	v_lshlrev_b64 v[116:117], 2, v[24:25]
	v_lshlrev_b64 v[118:119], 2, v[28:29]
	v_lshlrev_b64 v[120:121], 2, v[32:33]
	v_lshlrev_b64 v[122:123], 2, v[34:35]
	v_lshlrev_b64 v[124:125], 2, v[18:19]
	v_lshlrev_b64 v[126:127], 2, v[22:23]
	v_lshlrev_b64 v[128:129], 2, v[26:27]
	v_lshlrev_b64 v[130:131], 2, v[30:31]
	s_mov_b64 s[30:31], 0x200
	v_lshlrev_b32_e32 v254, 1, v90
	v_lshlrev_b32_e32 v255, 1, v88
	v_lshlrev_b32_e32 v196, 1, v92
	v_lshlrev_b32_e32 v197, 1, v94
	v_lshlrev_b32_e32 v198, 1, v98
	v_lshlrev_b32_e32 v199, 1, v96
	v_mov_b64_e32 v[14:15], v[2:3]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[22:23], v[2:3]
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[10:11], v[2:3]
	s_branch .LBB0_2
.LBB0_1:                                ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[4:5]
	v_lshl_or_b32 v142, v200, 5, s37
	v_lshl_or_b32 v132, s33, 8, v209
	s_mul_i32 s8, s26, s36
	v_mul_lo_u32 v133, s27, v142
	v_mad_u64_u32 v[134:135], s[4:5], s26, v142, 0
	v_add3_u32 v135, v135, s8, v133
	v_ashrrev_i32_e32 v133, 31, v132
	v_lshl_add_u64 v[134:135], v[134:135], 2, s[28:29]
	v_lshlrev_b64 v[132:133], 2, v[132:133]
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[132:133]
	v_lshl_add_u64 v[136:137], v[134:135], 0, v[100:101]
	v_lshl_add_u64 v[138:139], v[134:135], 0, v[102:103]
	global_store_dword v[136:137], v84, off
	global_store_dword v[138:139], v85, off
	v_lshl_add_u64 v[84:85], v[134:135], 0, v[104:105]
	v_lshl_add_u64 v[140:141], v[134:135], 0, v[106:107]
	global_store_dword v[84:85], v86, off
	global_store_dword v[140:141], v87, off
	v_lshl_add_u64 v[86:87], v[134:135], 0, v[108:109]
	global_store_dword v[86:87], v80, off offset:64
	v_lshl_add_u64 v[86:87], v[134:135], 0, v[110:111]
	global_store_dword v[86:87], v81, off offset:64
	v_lshl_add_u64 v[80:81], v[134:135], 0, v[112:113]
	global_store_dword v[80:81], v82, off offset:64
	v_lshl_add_u64 v[80:81], v[134:135], 0, v[114:115]
	global_store_dword v[80:81], v83, off offset:64
	v_lshl_add_u64 v[80:81], v[134:135], 0, v[116:117]
	v_lshl_add_u64 v[82:83], v[134:135], 0, v[118:119]
	global_store_dword v[80:81], v76, off
	global_store_dword v[82:83], v77, off
	v_lshl_add_u64 v[76:77], v[134:135], 0, v[120:121]
	v_lshl_add_u64 v[86:87], v[134:135], 0, v[122:123]
	global_store_dword v[76:77], v78, off
	global_store_dword v[86:87], v79, off
	v_lshl_add_u64 v[78:79], v[134:135], 0, v[124:125]
	global_store_dword v[78:79], v72, off offset:64
	v_lshl_add_u64 v[78:79], v[134:135], 0, v[126:127]
	global_store_dword v[78:79], v73, off offset:64
	v_lshl_add_u64 v[72:73], v[134:135], 0, v[128:129]
	global_store_dword v[72:73], v74, off offset:64
	v_lshl_add_u64 v[72:73], v[134:135], 0, v[130:131]
	global_store_dword v[72:73], v75, off offset:64
	v_lshl_add_u64 v[72:73], v[134:135], 0, s[30:31]
	global_store_dword v[136:137], v68, off offset:512
	global_store_dword v[138:139], v69, off offset:512
	global_store_dword v[84:85], v70, off offset:512
	global_store_dword v[140:141], v71, off offset:512
	v_lshl_add_u64 v[68:69], v[72:73], 0, v[108:109]
	global_store_dword v[68:69], v64, off offset:64
	v_lshl_add_u64 v[68:69], v[72:73], 0, v[110:111]
	global_store_dword v[68:69], v65, off offset:64
	v_lshl_add_u64 v[64:65], v[72:73], 0, v[112:113]
	global_store_dword v[64:65], v66, off offset:64
	v_lshl_add_u64 v[64:65], v[72:73], 0, v[114:115]
	global_store_dword v[64:65], v67, off offset:64
	global_store_dword v[80:81], v60, off offset:512
	global_store_dword v[82:83], v61, off offset:512
	global_store_dword v[76:77], v62, off offset:512
	global_store_dword v[86:87], v63, off offset:512
	v_lshl_add_u64 v[60:61], v[72:73], 0, v[124:125]
	global_store_dword v[60:61], v56, off offset:64
	v_lshl_add_u64 v[60:61], v[72:73], 0, v[126:127]
	global_store_dword v[60:61], v57, off offset:64
	v_lshl_add_u64 v[56:57], v[72:73], 0, v[128:129]
	global_store_dword v[56:57], v58, off offset:64
	v_lshl_add_u64 v[56:57], v[72:73], 0, v[130:131]
	global_store_dword v[56:57], v59, off offset:64
	v_or_b32_e32 v56, 64, v142
	v_mul_lo_u32 v58, s27, v56
	v_mad_u64_u32 v[56:57], s[4:5], s26, v56, 0
	v_add3_u32 v57, v57, s8, v58
	v_lshl_add_u64 v[56:57], v[56:57], 2, s[28:29]
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[132:133]
	v_lshl_add_u64 v[58:59], v[56:57], 0, v[100:101]
	v_lshl_add_u64 v[60:61], v[56:57], 0, v[102:103]
	global_store_dword v[58:59], v52, off
	global_store_dword v[60:61], v53, off
	v_lshl_add_u64 v[52:53], v[56:57], 0, v[104:105]
	v_lshl_add_u64 v[62:63], v[56:57], 0, v[106:107]
	global_store_dword v[52:53], v54, off
	global_store_dword v[62:63], v55, off
	v_lshl_add_u64 v[54:55], v[56:57], 0, v[108:109]
	global_store_dword v[54:55], v48, off offset:64
	v_lshl_add_u64 v[54:55], v[56:57], 0, v[110:111]
	global_store_dword v[54:55], v49, off offset:64
	v_lshl_add_u64 v[48:49], v[56:57], 0, v[112:113]
	global_store_dword v[48:49], v50, off offset:64
	v_lshl_add_u64 v[48:49], v[56:57], 0, v[114:115]
	global_store_dword v[48:49], v51, off offset:64
	v_lshl_add_u64 v[48:49], v[56:57], 0, v[116:117]
	v_lshl_add_u64 v[50:51], v[56:57], 0, v[118:119]
	global_store_dword v[48:49], v44, off
	global_store_dword v[50:51], v45, off
	v_lshl_add_u64 v[44:45], v[56:57], 0, v[120:121]
	v_lshl_add_u64 v[54:55], v[56:57], 0, v[122:123]
	global_store_dword v[44:45], v46, off
	global_store_dword v[54:55], v47, off
	v_lshl_add_u64 v[46:47], v[56:57], 0, v[124:125]
	global_store_dword v[46:47], v40, off offset:64
	v_lshl_add_u64 v[46:47], v[56:57], 0, v[126:127]
	global_store_dword v[46:47], v41, off offset:64
	v_lshl_add_u64 v[40:41], v[56:57], 0, v[128:129]
	global_store_dword v[40:41], v42, off offset:64
	v_lshl_add_u64 v[40:41], v[56:57], 0, v[130:131]
	global_store_dword v[40:41], v43, off offset:64
	v_lshl_add_u64 v[40:41], v[56:57], 0, s[30:31]
	global_store_dword v[58:59], v36, off offset:512
	global_store_dword v[60:61], v37, off offset:512
	global_store_dword v[52:53], v38, off offset:512
	global_store_dword v[62:63], v39, off offset:512
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[108:109]
	global_store_dword v[36:37], v32, off offset:64
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[110:111]
	global_store_dword v[36:37], v33, off offset:64
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[112:113]
	global_store_dword v[32:33], v34, off offset:64
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[114:115]
	global_store_dword v[32:33], v35, off offset:64
	global_store_dword v[48:49], v28, off offset:512
	global_store_dword v[50:51], v29, off offset:512
	global_store_dword v[44:45], v30, off offset:512
	global_store_dword v[54:55], v31, off offset:512
	v_lshl_add_u64 v[28:29], v[40:41], 0, v[124:125]
	global_store_dword v[28:29], v24, off offset:64
	v_lshl_add_u64 v[28:29], v[40:41], 0, v[126:127]
	global_store_dword v[28:29], v25, off offset:64
	v_lshl_add_u64 v[24:25], v[40:41], 0, v[128:129]
	global_store_dword v[24:25], v26, off offset:64
	v_lshl_add_u64 v[24:25], v[40:41], 0, v[130:131]
	global_store_dword v[24:25], v27, off offset:64
	s_add_i32 s21, s21, 1
.LBB0_2:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_9 Depth 2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s5, s3, s21
	s_add_i32 s5, s5, s2
	s_cmpk_gt_i32 s5, 0x7ff
	s_mov_b32 s4, -1
	s_mov_b32 s33, -1
	s_cbranch_scc1 .LBB0_4
; %bb.3:                                ;   in Loop: Header=BB0_2 Depth=1
	s_ashr_i32 s4, s5, 31
	s_lshr_b32 s4, s4, 29
	s_add_i32 s4, s5, s4
	s_ashr_i32 s4, s4, 3
	s_lshl_b32 s5, s5, 8
	s_mulk_i32 s4, 0xf801
	s_add_i32 s4, s4, s5
	s_ashr_i32 s5, s4, 31
	s_lshr_b32 s5, s5, 23
	s_add_i32 s5, s4, s5
	s_ashr_i32 s8, s5, 9
	s_lshl_b32 s8, s8, 4
	s_sub_i32 s9, 64, s8
	s_min_i32 s9, s9, 16
	s_sext_i32_i16 s12, s9
	v_cvt_f32_i32_e32 v24, s12
	s_and_b32 s5, s5, 0xfffffe00
	s_sub_i32 s13, s4, s5
	s_sext_i32_i16 s4, s13
	v_cvt_f32_i32_e32 v25, s4
	v_rcp_iflag_f32_e32 v26, v24
	s_xor_b32 s4, s4, s12
	s_ashr_i32 s4, s4, 30
	s_or_b32 s12, s4, 1
	v_mul_f32_e32 v26, v25, v26
	v_trunc_f32_e32 v26, v26
	v_fma_f32 v25, -v26, v24, v25
	v_cvt_i32_f32_e32 v26, v26
	v_cmp_ge_f32_e64 s[4:5], |v25|, |v24|
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s4, s12, 0
	v_readfirstlane_b32 s5, v26
	s_add_i32 s4, s5, s4
	s_sext_i32_i16 s33, s4
	s_mul_i32 s4, s4, s9
	s_sub_i32 s4, s13, s4
	s_sext_i32_i16 s4, s4
	s_add_i32 s4, s8, s4
.LBB0_4:                                ; %_Z12get_task_idxi.exit
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_cmp_eq_u32 s4, -1
	s_cbranch_scc1 .LBB0_17
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
	s_lshl_b32 s37, s4, 7
	s_ashr_i32 s36, s37, 31
	s_mul_i32 s4, s22, s36
	s_mul_hi_u32 s5, s22, s37
	s_add_i32 s4, s5, s4
	s_mul_i32 s5, s23, s37
	s_add_i32 s5, s4, s5
	s_mul_i32 s4, s22, s37
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s4, s24, s4
	s_addc_u32 s5, s25, s5
	s_lshl_b32 s8, s33, 8
	s_ashr_i32 s9, s33, 31
	v_lshl_add_u64 v[24:25], v[88:89], 1, s[4:5]
	s_mul_hi_u32 s12, s18, s8
	s_mul_i32 s9, s18, s9
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off

	;;#ASMEND
	v_lshl_add_u64 v[28:29], v[90:91], 1, s[4:5]
	s_add_i32 s9, s12, s9
	s_mul_i32 s12, s19, s8
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	scratch_load_dword v32, off, off        ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v32, v[24:25]

	;;#ASMEND
	s_add_i32 s9, s9, s12
	s_mul_i32 s8, s18, s8
	scratch_load_dword v24, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v24, v[26:27]

	;;#ASMEND
	s_lshl_b64 s[8:9], s[8:9], 1
	scratch_load_dword v24, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v24, v[28:29]

	;;#ASMEND
	s_add_u32 s8, s16, s8
	scratch_load_dword v24, off, off offset:12 ; 4-byte Folded Reload
	s_addc_u32 s9, s17, s9
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v24, v[30:31]

	;;#ASMEND
	v_lshl_add_u64 v[24:25], v[92:93], 1, s[8:9]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	global_load_dwordx4 v[24:27], v[24:25], off

	;;#ASMEND
	v_lshl_add_u64 v[28:29], v[94:95], 1, s[8:9]
	;;#ASMSTART
	global_load_dwordx4 v[28:31], v[28:29], off

	;;#ASMEND
	v_lshl_add_u64 v[32:33], v[98:99], 1, s[8:9]
	;;#ASMSTART
	global_load_dwordx4 v[32:35], v[32:33], off

	;;#ASMEND
	v_lshl_add_u64 v[36:37], v[96:97], 1, s[8:9]
	;;#ASMSTART
	global_load_dwordx4 v[36:39], v[36:37], off

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v201, v[24:25]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v202, v[26:27]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v203, v[28:29]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v204, v[30:31]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v205, v[32:33]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v206, v[34:35]

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	ds_write_b64 v207, v[36:37]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v208, v[38:39]

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_2 Depth=1
	s_barrier
.LBB0_7:                                ; %.preheader
                                        ;   in Loop: Header=BB0_2 Depth=1
	s_or_b64 exec, exec, s[12:13]
	s_add_u32 s38, s4, 0x80
	s_addc_u32 s39, s5, 0
	s_add_u32 s40, s8, 0x80
	v_mov_b32_e32 v24, 0
	s_addc_u32 s41, s9, 0
	s_mov_b64 s[34:35], 0
	v_mov_b32_e32 v25, v24
	v_mov_b32_e32 v26, v24
	v_mov_b32_e32 v27, v24
	v_mov_b32_e32 v28, v24
	v_mov_b32_e32 v29, v24
	v_mov_b32_e32 v30, v24
	v_mov_b32_e32 v31, v24
	v_mov_b32_e32 v32, v24
	v_mov_b32_e32 v33, v24
	v_mov_b32_e32 v34, v24
	v_mov_b32_e32 v35, v24
	v_mov_b32_e32 v36, v24
	v_mov_b32_e32 v37, v24
	v_mov_b32_e32 v38, v24
	v_mov_b32_e32 v39, v24
	v_mov_b32_e32 v40, v24
	v_mov_b32_e32 v41, v24
	v_mov_b32_e32 v42, v24
	v_mov_b32_e32 v43, v24
	v_mov_b32_e32 v44, v24
	v_mov_b32_e32 v45, v24
	v_mov_b32_e32 v46, v24
	v_mov_b32_e32 v47, v24
	v_mov_b32_e32 v48, v24
	v_mov_b32_e32 v49, v24
	v_mov_b32_e32 v50, v24
	v_mov_b32_e32 v51, v24
	v_mov_b32_e32 v52, v24
	v_mov_b32_e32 v53, v24
	v_mov_b32_e32 v54, v24
	v_mov_b32_e32 v55, v24
	v_mov_b32_e32 v56, v24
	v_mov_b32_e32 v57, v24
	v_mov_b32_e32 v58, v24
	v_mov_b32_e32 v59, v24
	v_mov_b32_e32 v60, v24
	v_mov_b32_e32 v61, v24
	v_mov_b32_e32 v62, v24
	v_mov_b32_e32 v63, v24
	v_mov_b32_e32 v64, v24
	v_mov_b32_e32 v65, v24
	v_mov_b32_e32 v66, v24
	v_mov_b32_e32 v67, v24
	v_mov_b32_e32 v68, v24
	v_mov_b32_e32 v69, v24
	v_mov_b32_e32 v70, v24
	v_mov_b32_e32 v71, v24
	v_mov_b32_e32 v72, v24
	v_mov_b32_e32 v73, v24
	v_mov_b32_e32 v74, v24
	v_mov_b32_e32 v75, v24
	v_mov_b32_e32 v76, v24
	v_mov_b32_e32 v77, v24
	v_mov_b32_e32 v78, v24
	v_mov_b32_e32 v79, v24
	v_mov_b32_e32 v80, v24
	v_mov_b32_e32 v81, v24
	v_mov_b32_e32 v82, v24
	v_mov_b32_e32 v83, v24
	v_mov_b32_e32 v84, v24
	v_mov_b32_e32 v85, v24
	v_mov_b32_e32 v86, v24
	v_mov_b32_e32 v87, v24
	s_branch .LBB0_9
.LBB0_8:                                ;   in Loop: Header=BB0_9 Depth=2
	v_bfi_b32 v178, s20, v178, v178
	v_bfi_b32 v176, s20, v176, v176
	v_bfi_b32 v174, s20, v174, v174
	v_bfi_b32 v172, s20, v172, v172
	v_bfi_b32 v170, s20, v170, v170
	v_bfi_b32 v168, s20, v168, v168
	v_bfi_b32 v166, s20, v166, v166
	v_bfi_b32 v164, s20, v164, v164
	v_bfi_b32 v194, s20, v194, v194
	v_bfi_b32 v192, s20, v192, v192
	v_bfi_b32 v190, s20, v190, v190
	v_bfi_b32 v188, s20, v188, v188
	v_bfi_b32 v186, s20, v186, v186
	v_bfi_b32 v184, s20, v184, v184
	v_bfi_b32 v182, s20, v182, v182
	v_bfi_b32 v180, s20, v180, v180
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[84:87], v[194:195], v[178:179], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[194:195], v[174:175], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[190:191], v[178:179], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[190:191], v[174:175], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[194:195], v[170:171], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[194:195], v[166:167], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[190:191], v[170:171], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[190:191], v[166:167], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[186:187], v[178:179], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[186:187], v[174:175], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[182:183], v[178:179], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[182:183], v[174:175], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[186:187], v[170:171], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[186:187], v[166:167], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[182:183], v[170:171], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[182:183], v[166:167], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[192:193], v[176:177], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[192:193], v[172:173], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[188:189], v[176:177], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[188:189], v[172:173], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[192:193], v[168:169], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[192:193], v[164:165], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[188:189], v[168:169], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[188:189], v[164:165], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[184:185], v[176:177], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[184:185], v[172:173], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[180:181], v[176:177], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[180:181], v[172:173], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[184:185], v[168:169], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[184:185], v[164:165], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[180:181], v[168:169], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[180:181], v[164:165], v[24:27]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s34, s34, 0x80
	s_addc_u32 s35, s35, 0
	s_cmpk_eq_i32 s34, 0x4000
	s_cbranch_scc1 .LBB0_15
.LBB0_9:                                ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_cmpk_lg_i32 s34, 0x3f80
	s_cselect_b64 s[12:13], -1, 0
	s_cmpk_eq_i32 s34, 0x3f80
	s_cbranch_scc1 .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_9 Depth=2
	s_add_u32 s8, s38, s34
	s_addc_u32 s9, s39, s35
	buffer_load_dwordx4 v[4:7], v255, s[8:11], 0 offen
	buffer_load_dwordx4 v[8:11], v254, s[8:11], 0 offen
.LBB0_11:                               ;   in Loop: Header=BB0_9 Depth=2
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[146:147], v210

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[144:145], v211

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[142:143], v212

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[140:141], v213

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[138:139], v214

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[136:137], v215

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[134:135], v216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[132:133], v217

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[162:163], v218

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[160:161], v219

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[158:159], v220

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[156:157], v221

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[154:155], v222

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[152:153], v223

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[150:151], v224

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[148:149], v225

	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cndmask_b32_e64 v164, 0, 1, s[12:13]
	v_cmp_ne_u32_e64 s[4:5], 1, v164
	s_andn2_b64 vcc, exec, s[12:13]
	s_cbranch_vccnz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_9 Depth=2
	s_add_u32 s12, s40, s34
	s_addc_u32 s13, s41, s35
	s_mov_b32 s15, s11
	buffer_load_dwordx4 v[0:3], v196, s[12:15], 0 offen
	buffer_load_dwordx4 v[12:15], v197, s[12:15], 0 offen
	buffer_load_dwordx4 v[16:19], v198, s[12:15], 0 offen
	buffer_load_dwordx4 v[20:23], v199, s[12:15], 0 offen
.LBB0_13:                               ;   in Loop: Header=BB0_9 Depth=2
	v_bfi_b32 v146, s20, v146, v146
	v_bfi_b32 v144, s20, v144, v144
	v_bfi_b32 v142, s20, v142, v142
	v_bfi_b32 v140, s20, v140, v140
	v_bfi_b32 v138, s20, v138, v138
	v_bfi_b32 v136, s20, v136, v136
	v_bfi_b32 v134, s20, v134, v134
	v_bfi_b32 v132, s20, v132, v132
	v_bfi_b32 v162, s20, v162, v162
	v_bfi_b32 v160, s20, v160, v160
	v_bfi_b32 v158, s20, v158, v158
	v_bfi_b32 v156, s20, v156, v156
	v_bfi_b32 v154, s20, v154, v154
	v_bfi_b32 v152, s20, v152, v152
	v_bfi_b32 v150, s20, v150, v150
	v_bfi_b32 v148, s20, v148, v148
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64 v[178:179], v226

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[176:177], v227

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[174:175], v228

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[172:173], v229

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[170:171], v230

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[168:169], v231

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[166:167], v232

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[164:165], v233

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[194:195], v234

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[192:193], v235

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[190:191], v236

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[188:189], v237

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[186:187], v238

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[184:185], v239

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[182:183], v240

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64 v[180:181], v241

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_16x16x16_bf16 v[84:87], v[162:163], v[146:147], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[162:163], v[142:143], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[158:159], v[146:147], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[158:159], v[142:143], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[162:163], v[138:139], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[162:163], v[134:135], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[158:159], v[138:139], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[158:159], v[134:135], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[154:155], v[146:147], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[154:155], v[142:143], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[150:151], v[146:147], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[150:151], v[142:143], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[154:155], v[138:139], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[154:155], v[134:135], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[150:151], v[138:139], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[150:151], v[134:135], v[24:27]
	v_mfma_f32_16x16x16_bf16 v[84:87], v[160:161], v[144:145], v[84:87]
	v_mfma_f32_16x16x16_bf16 v[80:83], v[160:161], v[140:141], v[80:83]
	v_mfma_f32_16x16x16_bf16 v[76:79], v[156:157], v[144:145], v[76:79]
	v_mfma_f32_16x16x16_bf16 v[72:75], v[156:157], v[140:141], v[72:75]
	v_mfma_f32_16x16x16_bf16 v[68:71], v[160:161], v[136:137], v[68:71]
	v_mfma_f32_16x16x16_bf16 v[64:67], v[160:161], v[132:133], v[64:67]
	v_mfma_f32_16x16x16_bf16 v[60:63], v[156:157], v[136:137], v[60:63]
	v_mfma_f32_16x16x16_bf16 v[56:59], v[156:157], v[132:133], v[56:59]
	v_mfma_f32_16x16x16_bf16 v[52:55], v[152:153], v[144:145], v[52:55]
	v_mfma_f32_16x16x16_bf16 v[48:51], v[152:153], v[140:141], v[48:51]
	v_mfma_f32_16x16x16_bf16 v[44:47], v[148:149], v[144:145], v[44:47]
	v_mfma_f32_16x16x16_bf16 v[40:43], v[148:149], v[140:141], v[40:43]
	v_mfma_f32_16x16x16_bf16 v[36:39], v[152:153], v[136:137], v[36:39]
	v_mfma_f32_16x16x16_bf16 v[32:35], v[152:153], v[132:133], v[32:35]
	v_mfma_f32_16x16x16_bf16 v[28:31], v[148:149], v[136:137], v[28:31]
	v_mfma_f32_16x16x16_bf16 v[24:27], v[148:149], v[132:133], v[24:27]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_8
; %bb.14:                               ;   in Loop: Header=BB0_9 Depth=2
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v242, v[4:5]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v243, v[6:7]

	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v244, v[8:9]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v245, v[10:11]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v246, v[0:1]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v247, v[2:3]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v248, v[12:13]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v249, v[14:15]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v250, v[16:17]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v251, v[18:19]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v252, v[20:21]

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v253, v[22:23]

	;;#ASMEND
	s_branch .LBB0_8
.LBB0_15:                               ;   in Loop: Header=BB0_2 Depth=1
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_cbranch_execz .LBB0_1
; %bb.16:                               ;   in Loop: Header=BB0_2 Depth=1
	s_barrier
	s_branch .LBB0_1
.LBB0_17:                               ;   in Loop: Header=BB0_2 Depth=1
                                        ; implicit-def: $sgpr21
                                        ; implicit-def: $vgpr10_vgpr11
                                        ; implicit-def: $vgpr6_vgpr7
                                        ; implicit-def: $vgpr22_vgpr23
                                        ; implicit-def: $vgpr18_vgpr19
                                        ; implicit-def: $vgpr14_vgpr15
                                        ; implicit-def: $vgpr2_vgpr3
	s_cbranch_execz .LBB0_2
; %bb.18:                               ; %.critedge
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tk13micro_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 20
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 42
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
; codeLenInByte = 5304
; NumSgprs: 48
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 20
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 48
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_de8221682ccd7b1c,@object ; @__hip_cuid_de8221682ccd7b1c
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_de8221682ccd7b1c
__hip_cuid_de8221682ccd7b1c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_de8221682ccd7b1c, 1

	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_de8221682ccd7b1c
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
    .private_segment_fixed_size: 20
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 4
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
