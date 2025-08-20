	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z10attend_kerILi64EEv12attn_globalsIXT_EE,"axG",@progbits,_Z10attend_kerILi64EEv12attn_globalsIXT_EE,comdat
	.protected	_Z10attend_kerILi64EEv12attn_globalsIXT_EE ; -- Begin function _Z10attend_kerILi64EEv12attn_globalsIXT_EE
	.globl	_Z10attend_kerILi64EEv12attn_globalsIXT_EE
	.p2align	8
	.type	_Z10attend_kerILi64EEv12attn_globalsIXT_EE,@function
_Z10attend_kerILi64EEv12attn_globalsIXT_EE: ; @_Z10attend_kerILi64EEv12attn_globalsIXT_EE
; %bb.0:                                ; %entry
	s_mov_b64 s[6:7], src_shared_base
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s6, 0, 0
	s_cselect_b32 s5, s7, 0
	s_and_b32 s42, s6, 15
	s_and_b32 s7, s6, -16
	s_load_dwordx2 s[38:39], s[0:1], 0x30
	s_load_dword s33, s[0:1], 0x50
	s_load_dwordx4 s[16:19], s[0:1], 0x40
	s_load_dwordx2 s[40:41], s[0:1], 0x60
	s_load_dwordx4 s[8:11], s[0:1], 0x70
	s_load_dword s44, s[0:1], 0x80
	s_load_dwordx2 s[34:35], s[0:1], 0x90
	s_add_u32 s7, s7, 16
	s_mov_b32 s43, 0
	s_waitcnt lgkmcnt(0)
	s_addc_u32 s9, s5, 0
	s_cmp_eq_u64 s[42:43], 0
	s_cselect_b32 s12, s6, s7
	s_cselect_b32 s13, s5, s9
	s_add_u32 s5, s12, 0x8000
	s_addc_u32 s6, s13, 0
	s_and_b32 s42, s5, 15
	s_and_b32 s7, s5, -16
	s_add_u32 s7, s7, 16
	s_addc_u32 s9, s6, 0
	s_cmp_eq_u64 s[42:43], 0
	s_cselect_b32 s14, s5, s7
	s_mul_i32 s5, s2, s16
	v_lshlrev_b32_e32 v1, 4, v0
	s_cselect_b32 s15, s6, s9
	s_add_i32 s42, s5, s3
	v_add_u32_e32 v4, s12, v1
	s_mul_i32 s42, s42, s18
	v_lshrrev_b32_e32 v5, 4, v4
	s_movk_i32 s5, 0x70
	s_mul_i32 s6, s42, s33
	v_bitop3_b32 v4, v5, v4, s5 bitop3:0x6c
	s_ashr_i32 s7, s6, 31
	v_subrev_u32_e32 v4, s12, v4
	s_lshl_b64 s[6:7], s[6:7], 1
	v_ashrrev_i32_e32 v5, 6, v4
	s_add_u32 s24, s38, s6
	v_and_b32_e32 v2, 0x3c00, v1
	v_mov_b32_e32 v3, 0
	v_and_b32_e32 v5, -2, v5
	v_and_b32_e32 v4, 0x7e, v4
	s_addc_u32 s25, s39, s7
	v_lshl_add_u64 v[86:87], s[12:13], 0, v[2:3]
	v_mad_u64_u32 v[88:89], s[6:7], v5, s33, v[4:5]
	v_readfirstlane_b32 s6, v86
	s_mov_b32 m0, s6
	s_mul_i32 s6, s2, s8
	s_add_i32 s45, s6, s3
	v_add_u32_e32 v1, s14, v1
	s_mul_i32 s45, s45, s10
	v_lshl_add_u64 v[90:91], s[14:15], 0, v[2:3]
	v_lshrrev_b32_e32 v2, 4, v1
	s_mul_i32 s6, s45, s44
	v_bitop3_b32 v1, v2, v1, s5 bitop3:0x6c
	s_ashr_i32 s7, s6, 31
	v_subrev_u32_e32 v1, s14, v1
	s_lshl_b32 s26, s33, 8
	s_lshl_b64 s[6:7], s[6:7], 1
	v_ashrrev_i32_e32 v2, 6, v1
	s_mov_b32 s27, 0x110000
	s_add_u32 s28, s40, s6
	v_and_b32_e32 v3, -2, v2
	v_and_b32_e32 v2, 0x7e, v1
	v_readfirstlane_b32 s5, v90
	buffer_load_dwordx4 v88, s[24:27], 0 offen lds
	s_addc_u32 s29, s41, s7
	s_lshl_b32 s30, s44, 8
	s_mov_b32 s31, s27
	v_mad_u64_u32 v[92:93], s[6:7], v3, s44, v[2:3]
	s_mov_b32 m0, s5
	v_lshrrev_b32_e32 v1, 1, v0
	buffer_load_dwordx4 v92, s[28:31], 0 offen lds
	s_load_dwordx8 s[48:55], s[0:1], 0x0
	s_load_dword s5, s[0:1], 0x20
	s_load_dwordx2 s[36:37], s[0:1], 0xb0
	s_load_dwordx4 s[20:23], s[0:1], 0xa0
	v_and_b32_e32 v1, 0x1e0, v1
	v_lshl_or_b32 v87, s4, 9, v1
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s2, s52
	s_add_i32 s0, s0, s3
	s_mul_i32 s0, s0, s54
	v_add_u32_e32 v1, s0, v87
	v_mul_lo_u32 v4, v1, s5
	s_mul_i32 s0, s50, s52
	v_mov_b32_e32 v2, s48
	v_mov_b32_e32 v3, s49
	v_ashrrev_i32_e32 v5, 31, v4
	s_mul_i32 s0, s0, s54
	v_lshl_add_u64 v[2:3], v[4:5], 1, v[2:3]
	v_and_b32_e32 v1, 31, v0
	v_lshrrev_b32_e32 v4, 2, v0
	s_mul_i32 s0, s0, s5
	v_and_b32_e32 v28, 8, v4
	s_lshl_b32 s0, s0, 1
	v_mul_lo_u32 v6, v1, s5
	s_mov_b32 s21, 1
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v5, 0x20000
	v_add_lshl_u32 v29, v6, v28, 1
	s_mov_b64 s[8:9], exec
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s6, v4
	v_readfirstlane_b32 s7, v5
	v_cmp_eq_u64_e32 vcc, s[4:5], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v29, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_1
; %bb.2:
	s_mov_b64 exec, s[8:9]
	s_mov_b64 s[8:9], exec
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s6, v4
	v_readfirstlane_b32 s7, v5
	v_cmp_eq_u64_e32 vcc, s[4:5], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[10:13], v29, s[4:7], 0 offen offset:32
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_mov_b64 exec, s[8:9]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v25, 0xffff0000, v6
	v_lshlrev_b32_e32 v24, 16, v6
	s_mov_b64 s[8:9], exec
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s6, v4
	v_readfirstlane_b32 s7, v5
	v_cmp_eq_u64_e32 vcc, s[4:5], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[14:17], v29, s[4:7], 0 offen offset:64
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_5
; %bb.6:
	s_mov_b64 exec, s[8:9]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v23, 0xffff0000, v14
	v_lshlrev_b32_e32 v22, 16, v14
	v_and_b32_e32 v27, 0xffff0000, v10
	v_lshlrev_b32_e32 v26, 16, v10
	s_mov_b64 s[8:9], exec
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s6, v4
	v_readfirstlane_b32 s7, v5
	v_cmp_eq_u64_e32 vcc, s[4:5], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[18:21], v29, s[4:7], 0 offen offset:96
                                        ; implicit-def: $vgpr2_vgpr3_vgpr4_vgpr5
                                        ; implicit-def: $vgpr29
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_mov_b64 exec, s[8:9]
	s_mov_b32 s0, 0x1000c0c
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v3, 0xffff0000, v18
	v_lshlrev_b32_e32 v2, 16, v18
	v_alignbit_b32 v4, v7, v6, 16
	v_and_b32_e32 v5, 0xffff0000, v7
	v_perm_b32 v6, v7, v8, s0
	v_and_b32_e32 v7, 0xffff0000, v8
	v_alignbit_b32 v8, v9, v8, 16
	v_alignbit_b32 v10, v11, v10, 16
	v_alignbit_b32 v14, v15, v14, 16
	v_alignbit_b32 v18, v19, v18, 16
	v_and_b32_e32 v4, 0xffff0000, v4
	v_and_b32_e32 v8, 0xffff0000, v8
	v_and_b32_e32 v9, 0xffff0000, v9
	v_and_b32_e32 v30, 0xffff0000, v10
	v_perm_b32 v10, v11, v12, s0
	v_and_b32_e32 v32, 0xffff0000, v14
	v_perm_b32 v14, v15, v16, s0
	v_and_b32_e32 v34, 0xffff0000, v18
	v_perm_b32 v18, v19, v20, s0
	s_mov_b32 s0, 0x3e38aa3b
	v_and_b32_e32 v31, 0xffff0000, v11
	v_pk_mul_f32 v[4:5], v[4:5], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[0:1] op_sel_hi:[1,0]
	s_mov_b32 s1, 0x7060302
	v_and_b32_e32 v33, 0xffff0000, v15
	v_perm_b32 v51, v5, v4, s1
	v_pk_mul_f32 v[4:5], v[30:31], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[0:1] op_sel_hi:[1,0]
	v_and_b32_e32 v11, 0xffff0000, v12
	v_alignbit_b32 v12, v13, v12, 16
	v_and_b32_e32 v35, 0xffff0000, v19
	v_perm_b32 v55, v5, v4, s1
	v_pk_mul_f32 v[4:5], v[32:33], s[0:1] op_sel_hi:[1,0]
	v_perm_b32 v62, v3, v2, s1
	v_lshlrev_b32_e32 v2, 1, v28
	v_and_b32_e32 v12, 0xffff0000, v12
	v_and_b32_e32 v13, 0xffff0000, v13
	v_and_b32_e32 v15, 0xffff0000, v16
	v_alignbit_b32 v16, v17, v16, 16
	v_perm_b32 v59, v5, v4, s1
	v_pk_mul_f32 v[4:5], v[34:35], s[0:1] op_sel_hi:[1,0]
	v_lshl_or_b32 v2, v1, 7, v2
	v_and_b32_e32 v16, 0xffff0000, v16
	v_and_b32_e32 v17, 0xffff0000, v17
	v_and_b32_e32 v19, 0xffff0000, v20
	v_alignbit_b32 v20, v21, v20, 16
	v_perm_b32 v53, v9, v8, s1
	v_perm_b32 v52, v7, v6, s1
	v_pk_mul_f32 v[6:7], v[12:13], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[26:27], s[0:1] op_sel_hi:[1,0]
	v_perm_b32 v63, v5, v4, s1
	v_add_u32_e32 v89, s12, v2
	v_lshlrev_b32_e32 v2, 5, v0
	v_lshlrev_b32_e32 v3, 1, v0
	v_and_b32_e32 v4, 3, v0
	v_and_b32_e32 v20, 0xffff0000, v20
	v_and_b32_e32 v21, 0xffff0000, v21
	v_pk_mul_f32 v[10:11], v[10:11], s[0:1] op_sel_hi:[1,0]
	v_perm_b32 v57, v7, v6, s1
	v_perm_b32 v54, v9, v8, s1
	v_pk_mul_f32 v[6:7], v[16:17], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[22:23], s[0:1] op_sel_hi:[1,0]
	v_and_b32_e32 v2, 0x580, v2
	v_and_b32_e32 v3, 32, v3
	v_lshlrev_b32_e32 v4, 3, v4
	s_mov_b32 s4, 0
	v_perm_b32 v56, v11, v10, s1
	v_pk_mul_f32 v[10:11], v[14:15], s[0:1] op_sel_hi:[1,0]
	v_perm_b32 v61, v7, v6, s1
	v_perm_b32 v58, v9, v8, s1
	v_pk_mul_f32 v[6:7], v[20:21], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[18:19], s[0:1] op_sel_hi:[1,0]
	v_or3_b32 v2, v2, v3, v4
	s_mov_b32 s5, s4
	v_perm_b32 v60, v11, v10, s1
	v_perm_b32 v65, v7, v6, s1
	v_perm_b32 v64, v9, v8, s1
	v_add_u32_e32 v91, s14, v2
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s8, s4
	s_mov_b32 s9, s4
	s_mov_b32 s10, s4
	s_mov_b32 s11, s4
	s_mov_b32 s12, s4
	s_mov_b32 s13, s4
	s_mov_b32 s14, s4
	s_mov_b32 s15, s4
	s_mov_b32 s16, s4
	s_mov_b32 s17, s4
	s_mov_b32 s18, s4
	s_mov_b32 s19, s4
	v_mov_b64_e32 v[2:3], s[4:5]
	v_mov_b64_e32 v[16:17], s[18:19]
	v_perm_b32 v50, v25, v24, s1
	v_mov_b64_e32 v[4:5], s[6:7]
	v_mov_b64_e32 v[6:7], s[8:9]
	v_mov_b64_e32 v[8:9], s[10:11]
	v_mov_b64_e32 v[10:11], s[12:13]
	v_mov_b64_e32 v[12:13], s[14:15]
	v_mov_b64_e32 v[14:15], s[16:17]
	v_mov_b64_e32 v[32:33], v[16:17]
	v_mov_b32_e32 v94, 0xff800000
	v_mov_b32_e32 v93, 0
	s_mov_b32 s27, 0x110000
	s_movk_i32 s1, 0x70
	s_mov_b32 s4, 0xc2fc0000
	s_mov_b32 s0, 0x4b000000
	s_mov_b32 s5, 0x5040100
	v_mov_b32_e32 v95, 0x42fc0000
	s_mov_b32 s6, 0
	v_mov_b64_e32 v[30:31], v[14:15]
	v_mov_b64_e32 v[28:29], v[12:13]
	v_mov_b64_e32 v[26:27], v[10:11]
	v_mov_b64_e32 v[24:25], v[8:9]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
.LBB0_9:                                ; %for.body
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_10 Depth 2
	s_add_i32 s43, s43, 1
	s_lshl_b32 s7, s43, 7
	s_add_i32 s8, s7, s42
	s_mul_i32 s8, s8, s33
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	s_add_u32 s24, s38, s8
	s_addc_u32 s25, s39, s9
	s_lshl_b32 s10, s21, 14
	v_add_u32_e32 v34, s10, v86
	s_add_i32 s7, s7, s45
	v_readfirstlane_b32 s8, v34
	s_mov_b32 m0, s8
	s_mul_i32 s8, s7, s44
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 1
	v_add_u32_e32 v34, s10, v90
	s_add_u32 s28, s40, s8
	v_readfirstlane_b32 s7, v34
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v88, s[24:27], 0 offen lds
	s_addc_u32 s29, s41, s9
	s_mov_b32 s31, s27
	s_mov_b32 m0, s7
	s_lshl_b32 s7, s6, 14
	buffer_load_dwordx4 v92, s[28:31], 0 offen lds
	v_add_u32_e32 v96, s7, v89
	v_add_u32_e32 v97, s7, v91
	s_mov_b32 s7, 0
.LBB0_10:                               ; %for.body34
                                        ;   Parent Loop BB0_9 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mov_b32_e32 v98, v94
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v38, s7, v96
	v_lshrrev_b32_e32 v34, 4, v38
	v_add_u32_e32 v39, 64, v38
	v_bitop3_b32 v34, v34, v38, s1 bitop3:0x6c
	v_lshrrev_b32_e32 v40, 4, v39
	;;#ASMSTART
	ds_read_b128 v[34:37], v34 offset:0

	;;#ASMEND
	v_bitop3_b32 v39, v40, v39, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[70:73], v39 offset:0

	;;#ASMEND
	v_add_u32_e32 v39, 32, v38
	v_lshrrev_b32_e32 v40, 4, v39
	v_bitop3_b32 v39, v40, v39, s1 bitop3:0x6c
	v_add_u32_e32 v38, 0x60, v38
	;;#ASMSTART
	ds_read_b128 v[74:77], v39 offset:0

	;;#ASMEND
	v_lshrrev_b32_e32 v39, 4, v38
	v_bitop3_b32 v38, v39, v38, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[66:69], v38 offset:0

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[34:37], v[50:53], 0
	v_mov_b32_e32 v100, v93
	v_add_u32_e32 v78, s7, v97
	v_add_u32_e32 v79, 0x200, v78
	v_lshrrev_b32_e32 v80, 4, v78
	v_add_u32_e32 v81, 64, v78
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)

	;;#ASMEND
	s_addk_i32 s7, 0x1000
	v_mfma_f32_32x32x16_bf16 v[34:49], v[74:77], v[54:57], v[34:49]
	v_add_u32_e32 v74, 0x240, v78
	v_lshrrev_b32_e32 v75, 4, v79
	v_bitop3_b32 v76, v80, v78, s1 bitop3:0x6c
	v_lshrrev_b32_e32 v77, 4, v81
	v_lshrrev_b32_e32 v78, 4, v74
	v_bitop3_b32 v75, v75, v79, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b64_tr_b16 v[82:83], v76 offset:0
ds_read_b64_tr_b16 v[84:85], v75 offset:0

	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[34:49], v[70:73], v[58:61], v[34:49]
	v_bitop3_b32 v70, v77, v81, s1 bitop3:0x6c
	v_bitop3_b32 v71, v78, v74, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b64_tr_b16 v[74:75], v76 offset:0x800
ds_read_b64_tr_b16 v[76:77], v75 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[78:79], v70 offset:0
ds_read_b64_tr_b16 v[80:81], v71 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[70:71], v70 offset:0x800
ds_read_b64_tr_b16 v[72:73], v71 offset:0x800

	;;#ASMEND
	s_cmpk_eq_i32 s7, 0x4000
	;;#ASMSTART
	s_waitcnt lgkmcnt(4)

	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[34:49], v[66:69], v[62:65], v[34:49]
	s_nop 7
	s_nop 3
	v_max_f32_e32 v66, v35, v37
	v_max3_f32 v67, v34, v36, v38
	v_max3_f32 v66, v66, v39, v41
	v_max3_f32 v67, v67, v40, v42
	v_max3_f32 v66, v66, v43, v45
	v_max3_f32 v67, v67, v44, v46
	v_max3_f32 v66, v66, v47, v49
	v_max3_f32 v66, v67, v48, v66
	v_mov_b32_e32 v67, v66
	s_nop 1
	v_permlane32_swap_b32_e64 v66, v67 bound_ctrl:1
	v_max3_f32 v94, v98, v66, v67
	v_sub_f32_e32 v66, v34, v94
	v_sub_f32_e32 v67, v35, v94
	v_sub_f32_e32 v38, v38, v94
	v_sub_f32_e32 v39, v39, v94
	v_sub_f32_e32 v36, v36, v94
	v_sub_f32_e32 v37, v37, v94
	v_sub_f32_e32 v34, v40, v94
	v_sub_f32_e32 v35, v41, v94
	v_sub_f32_e32 v68, v42, v94
	v_sub_f32_e32 v69, v43, v94
	v_sub_f32_e32 v46, v46, v94
	v_sub_f32_e32 v47, v47, v94
	v_sub_f32_e32 v44, v44, v94
	v_sub_f32_e32 v45, v45, v94
	v_sub_f32_e32 v42, v48, v94
	v_sub_f32_e32 v43, v49, v94
	v_sub_f32_e32 v48, v98, v94
	v_med3_f32 v35, v35, s4, v95
	v_med3_f32 v34, v34, s4, v95
	v_med3_f32 v37, v37, s4, v95
	v_med3_f32 v36, v36, s4, v95
	v_med3_f32 v39, v39, s4, v95
	v_med3_f32 v38, v38, s4, v95
	v_med3_f32 v41, v67, s4, v95
	v_med3_f32 v40, v66, s4, v95
	v_med3_f32 v66, v48, s4, v95
	v_med3_f32 v43, v43, s4, v95
	v_med3_f32 v42, v42, s4, v95
	v_med3_f32 v45, v45, s4, v95
	v_med3_f32 v44, v44, s4, v95
	v_med3_f32 v47, v47, s4, v95
	v_med3_f32 v46, v46, s4, v95
	v_pk_mul_f32 v[34:35], v[34:35], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[0:1] op_sel_hi:[1,0]
	v_med3_f32 v48, v68, s4, v95
	v_mul_f32_e32 v66, 0x4b000000, v66
	v_pk_mul_f32 v[42:43], v[42:43], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[0:1] op_sel_hi:[1,0]
	v_cvt_i32_f32_e32 v34, v34
	v_cvt_i32_f32_e32 v35, v35
	v_cvt_i32_f32_e32 v36, v36
	v_cvt_i32_f32_e32 v37, v37
	v_cvt_i32_f32_e32 v40, v40
	v_cvt_i32_f32_e32 v41, v41
	v_cvt_i32_f32_e32 v67, v39
	v_cvt_i32_f32_e32 v68, v38
	v_cvt_i32_f32_e32 v66, v66
	v_cvt_i32_f32_e32 v93, v42
	v_cvt_i32_f32_e32 v98, v43
	v_cvt_i32_f32_e32 v99, v44
	v_cvt_i32_f32_e32 v47, v47
	v_cvt_i32_f32_e32 v101, v46
	v_med3_f32 v49, v69, s4, v95
	v_cvt_i32_f32_e32 v69, v45
	v_add_u32_e32 v39, 1.0, v41
	v_add_u32_e32 v38, 1.0, v40
	v_add_u32_e32 v41, 1.0, v67
	v_add_u32_e32 v40, 1.0, v68
	v_add_u32_e32 v43, 1.0, v37
	v_add_u32_e32 v42, 1.0, v36
	v_add_u32_e32 v45, 1.0, v35
	v_add_u32_e32 v44, 1.0, v34
	v_add_u32_e32 v46, 1.0, v66
	v_add_u32_e32 v67, 1.0, v47
	v_add_u32_e32 v66, 1.0, v101
	v_add_u32_e32 v68, 1.0, v99
	v_add_u32_e32 v99, 1.0, v98
	v_add_u32_e32 v98, 1.0, v93
	v_lshrrev_b32_e32 v34, 16, v38
	v_lshrrev_b32_e32 v35, 16, v40
	v_lshrrev_b32_e32 v47, 16, v39
	v_lshrrev_b32_e32 v36, 16, v41
	v_lshrrev_b32_e32 v93, 16, v42
	v_lshrrev_b32_e32 v37, 16, v44
	v_lshrrev_b32_e32 v101, 16, v43
	v_lshrrev_b32_e32 v102, 16, v45
	v_pk_mul_f32 v[32:33], v[32:33], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47] op_sel_hi:[1,0]
	v_permlane32_swap_b32_e64 v34, v35 bound_ctrl:1
	v_permlane32_swap_b32_e64 v47, v36 bound_ctrl:1
	v_permlane32_swap_b32_e64 v93, v37 bound_ctrl:1
	v_permlane32_swap_b32_e64 v101, v102 bound_ctrl:1
	v_pk_mul_f32 v[48:49], v[48:49], s[0:1] op_sel_hi:[1,0]
	v_perm_b32 v37, v102, v37, s5
	v_perm_b32 v36, v36, v35, s5
	v_perm_b32 v35, v101, v93, s5
	v_perm_b32 v34, v47, v34, s5
	v_cvt_i32_f32_e32 v48, v48
	v_cvt_i32_f32_e32 v49, v49
	v_mfma_f32_32x32x16_bf16 v[18:33], v[82:85], v[34:37], v[18:33]
	v_add_u32_e32 v69, 1.0, v69
	v_add_u32_e32 v48, 1.0, v48
	v_add_u32_e32 v49, 1.0, v49
	v_lshrrev_b32_e32 v103, 16, v48
	v_lshrrev_b32_e32 v104, 16, v66
	v_lshrrev_b32_e32 v105, 16, v49
	v_lshrrev_b32_e32 v106, 16, v67
	v_mfma_f32_32x32x16_bf16 v[2:17], v[78:81], v[34:37], v[2:17]
	v_lshrrev_b32_e32 v107, 16, v68
	v_lshrrev_b32_e32 v108, 16, v98
	v_lshrrev_b32_e32 v47, 16, v69
	v_lshrrev_b32_e32 v82, 16, v99
	v_permlane32_swap_b32_e64 v103, v104 bound_ctrl:1
	v_permlane32_swap_b32_e64 v105, v106 bound_ctrl:1
	v_permlane32_swap_b32_e64 v107, v108 bound_ctrl:1
	v_permlane32_swap_b32_e64 v47, v82 bound_ctrl:1
	v_perm_b32 v37, v82, v108, s5
	v_perm_b32 v36, v106, v104, s5
	v_perm_b32 v35, v47, v107, s5
	v_perm_b32 v34, v105, v103, s5
	v_pk_add_f32 v[68:69], v[68:69], v[98:99]
	v_pk_add_f32 v[48:49], v[48:49], v[66:67]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[74:77], v[34:37], v[18:33]
	v_pk_add_f32 v[42:43], v[42:43], v[44:45]
	v_pk_add_f32 v[38:39], v[38:39], v[40:41]
	v_pk_add_f32 v[40:41], v[48:49], v[68:69]
	v_pk_add_f32 v[38:39], v[38:39], v[42:43]
	v_add_f32_e32 v40, v40, v41
	v_add_f32_e32 v38, v38, v39
	v_mfma_f32_32x32x16_bf16 v[2:17], v[70:73], v[34:37], v[2:17]
	v_add_f32_e32 v34, v40, v38
	v_mov_b32_e32 v35, v34
	s_nop 1
	v_permlane32_swap_b32_e64 v34, v35 bound_ctrl:1
	v_add_f32_e32 v93, v35, v34
	v_fmac_f32_e32 v93, v100, v46
	s_cbranch_scc0 .LBB0_10
; %bb.11:                               ; %for.cond.cleanup33
                                        ;   in Loop: Header=BB0_9 Depth=1
	s_xor_b32 s6, s6, 1
	s_xor_b32 s21, s21, 1
	s_cmp_eq_u32 s43, 31
	s_cbranch_scc0 .LBB0_9
; %bb.12:                               ; %for.cond.cleanup
	s_mov_b32 s1, 0
	s_movk_i32 s4, 0x70
	s_mov_b32 s5, 0xc2fc0000
	v_mov_b32_e32 v86, 0x42fc0000
	s_mov_b32 s0, 0x4b000000
	s_mov_b32 s6, 0x5040100
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
.LBB0_13:                               ; %for.body59
                                        ; =>This Inner Loop Header: Depth=1
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v38, s1, v89
	v_add_u32_e32 v34, 0x4000, v38
	v_lshrrev_b32_e32 v35, 4, v34
	v_add_u32_e32 v39, 0x4040, v38
	v_bitop3_b32 v34, v35, v34, s4 bitop3:0x6c
	v_lshrrev_b32_e32 v40, 4, v39
	;;#ASMSTART
	ds_read_b128 v[34:37], v34 offset:0

	;;#ASMEND
	v_bitop3_b32 v39, v40, v39, s4 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[96:99], v39 offset:0

	;;#ASMEND
	v_add_u32_e32 v39, 0x4020, v38
	v_lshrrev_b32_e32 v40, 4, v39
	v_bitop3_b32 v39, v40, v39, s4 bitop3:0x6c
	v_add_u32_e32 v38, 0x4060, v38
	;;#ASMSTART
	ds_read_b128 v[100:103], v39 offset:0

	;;#ASMEND
	v_lshrrev_b32_e32 v39, 4, v38
	v_bitop3_b32 v38, v39, v38, s4 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[82:85], v38 offset:0

	;;#ASMEND
	v_add_u32_e32 v38, s1, v91
	v_add_u32_e32 v39, 0x4000, v38
	v_lshrrev_b32_e32 v40, 4, v39
	v_bitop3_b32 v39, v40, v39, s4 bitop3:0x6c
	v_add_u32_e32 v40, 0x4200, v38
	v_lshrrev_b32_e32 v41, 4, v40
	v_bitop3_b32 v40, v41, v40, s4 bitop3:0x6c
	;;#ASMSTART
	ds_read_b64_tr_b16 v[70:71], v39 offset:0
ds_read_b64_tr_b16 v[72:73], v40 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[66:67], v39 offset:0x800
ds_read_b64_tr_b16 v[68:69], v40 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v39, 0x4040, v38
	v_lshrrev_b32_e32 v40, 4, v39
	v_add_u32_e32 v38, 0x4240, v38
	v_bitop3_b32 v39, v40, v39, s4 bitop3:0x6c
	v_lshrrev_b32_e32 v40, 4, v38
	v_bitop3_b32 v38, v40, v38, s4 bitop3:0x6c
	;;#ASMSTART
	ds_read_b64_tr_b16 v[78:79], v39 offset:0
ds_read_b64_tr_b16 v[80:81], v38 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[74:75], v39 offset:0x800
ds_read_b64_tr_b16 v[76:77], v38 offset:0x800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[34:49], v[34:37], v[50:53], 0
	s_addk_i32 s1, 0x1000
	v_mov_b32_e32 v88, v93
	s_cmpk_eq_i32 s1, 0x4000
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)

	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[34:49], v[100:103], v[54:57], v[34:49]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[96:99], v[58:61], v[34:49]
	v_mfma_f32_32x32x16_bf16 v[34:49], v[82:85], v[62:65], v[34:49]
	s_nop 7
	s_nop 3
	v_max_f32_e32 v82, v35, v37
	v_max3_f32 v83, v34, v36, v38
	v_max3_f32 v82, v82, v39, v41
	v_max3_f32 v83, v83, v40, v42
	v_max3_f32 v82, v82, v43, v45
	v_max3_f32 v83, v83, v44, v46
	v_max3_f32 v82, v82, v47, v49
	v_max3_f32 v82, v83, v48, v82
	v_mov_b32_e32 v83, v82
	s_nop 1
	v_permlane32_swap_b32_e64 v82, v83 bound_ctrl:1
	v_max3_f32 v82, v94, v82, v83
	v_sub_f32_e32 v83, v34, v82
	v_sub_f32_e32 v84, v35, v82
	v_sub_f32_e32 v38, v38, v82
	v_sub_f32_e32 v39, v39, v82
	v_sub_f32_e32 v36, v36, v82
	v_sub_f32_e32 v37, v37, v82
	v_sub_f32_e32 v34, v40, v82
	v_sub_f32_e32 v35, v41, v82
	v_sub_f32_e32 v85, v42, v82
	v_sub_f32_e32 v90, v43, v82
	v_sub_f32_e32 v46, v46, v82
	v_sub_f32_e32 v47, v47, v82
	v_sub_f32_e32 v42, v48, v82
	v_sub_f32_e32 v43, v49, v82
	v_sub_f32_e32 v48, v94, v82
	v_med3_f32 v35, v35, s5, v86
	v_med3_f32 v34, v34, s5, v86
	v_med3_f32 v37, v37, s5, v86
	v_med3_f32 v36, v36, s5, v86
	v_med3_f32 v39, v39, s5, v86
	v_med3_f32 v38, v38, s5, v86
	v_med3_f32 v41, v84, s5, v86
	v_med3_f32 v40, v83, s5, v86
	v_sub_f32_e32 v44, v44, v82
	v_sub_f32_e32 v45, v45, v82
	v_mov_b32_e32 v94, v82
	v_med3_f32 v82, v48, s5, v86
	v_med3_f32 v43, v43, s5, v86
	v_med3_f32 v42, v42, s5, v86
	v_med3_f32 v47, v47, s5, v86
	v_med3_f32 v46, v46, s5, v86
	v_pk_mul_f32 v[40:41], v[40:41], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[0:1] op_sel_hi:[1,0]
	v_med3_f32 v45, v45, s5, v86
	v_med3_f32 v44, v44, s5, v86
	v_mul_f32_e32 v82, 0x4b000000, v82
	v_pk_mul_f32 v[42:43], v[42:43], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[0:1] op_sel_hi:[1,0]
	v_cvt_i32_f32_e32 v34, v34
	v_cvt_i32_f32_e32 v35, v35
	v_cvt_i32_f32_e32 v36, v36
	v_cvt_i32_f32_e32 v37, v37
	v_cvt_i32_f32_e32 v41, v41
	v_cvt_i32_f32_e32 v40, v40
	v_cvt_i32_f32_e32 v83, v39
	v_cvt_i32_f32_e32 v84, v38
	v_med3_f32 v49, v90, s5, v86
	v_pk_mul_f32 v[44:45], v[44:45], s[0:1] op_sel_hi:[1,0]
	v_cvt_i32_f32_e32 v82, v82
	v_cvt_i32_f32_e32 v90, v42
	v_cvt_i32_f32_e32 v47, v47
	v_cvt_i32_f32_e32 v95, v46
	v_med3_f32 v48, v85, s5, v86
	v_cvt_i32_f32_e32 v85, v43
	v_cvt_i32_f32_e32 v92, v44
	v_cvt_i32_f32_e32 v93, v45
	v_add_u32_e32 v39, 1.0, v41
	v_add_u32_e32 v38, 1.0, v40
	v_add_u32_e32 v41, 1.0, v83
	v_add_u32_e32 v40, 1.0, v84
	v_add_u32_e32 v43, 1.0, v37
	v_add_u32_e32 v42, 1.0, v36
	v_add_u32_e32 v45, 1.0, v35
	v_add_u32_e32 v44, 1.0, v34
	v_add_u32_e32 v46, 1.0, v82
	v_add_u32_e32 v83, 1.0, v47
	v_add_u32_e32 v82, 1.0, v95
	v_add_u32_e32 v84, 1.0, v90
	v_lshrrev_b32_e32 v47, 16, v38
	v_lshrrev_b32_e32 v36, 16, v40
	v_lshrrev_b32_e32 v90, 16, v39
	v_lshrrev_b32_e32 v95, 16, v41
	v_lshrrev_b32_e32 v96, 16, v42
	v_lshrrev_b32_e32 v37, 16, v44
	v_lshrrev_b32_e32 v97, 16, v43
	v_lshrrev_b32_e32 v98, 16, v45
	v_add_u32_e32 v35, 1.0, v93
	v_add_u32_e32 v34, 1.0, v92
	v_add_u32_e32 v85, 1.0, v85
	v_pk_mul_f32 v[32:33], v[32:33], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[46:47] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47] op_sel_hi:[1,0]
	v_permlane32_swap_b32_e64 v47, v36 bound_ctrl:1
	v_permlane32_swap_b32_e64 v90, v95 bound_ctrl:1
	v_permlane32_swap_b32_e64 v96, v37 bound_ctrl:1
	v_permlane32_swap_b32_e64 v97, v98 bound_ctrl:1
	v_pk_mul_f32 v[48:49], v[48:49], s[0:1] op_sel_hi:[1,0]
	v_pk_add_f32 v[92:93], v[34:35], v[84:85]
	v_lshrrev_b32_e32 v101, 16, v34
	v_lshrrev_b32_e32 v102, 16, v35
	v_perm_b32 v37, v98, v37, s6
	v_perm_b32 v36, v95, v36, s6
	v_perm_b32 v35, v97, v96, s6
	v_perm_b32 v34, v90, v47, s6
	v_cvt_i32_f32_e32 v48, v48
	v_cvt_i32_f32_e32 v49, v49
	v_mfma_f32_32x32x16_bf16 v[18:33], v[70:73], v[34:37], v[18:33]
	v_lshrrev_b32_e32 v103, 16, v82
	v_add_u32_e32 v48, 1.0, v48
	v_add_u32_e32 v49, 1.0, v49
	v_lshrrev_b32_e32 v99, 16, v48
	v_lshrrev_b32_e32 v100, 16, v49
	v_lshrrev_b32_e32 v47, 16, v83
	v_lshrrev_b32_e32 v70, 16, v84
	v_mfma_f32_32x32x16_bf16 v[2:17], v[78:81], v[34:37], v[2:17]
	v_lshrrev_b32_e32 v71, 16, v85
	v_permlane32_swap_b32_e64 v99, v103 bound_ctrl:1
	v_permlane32_swap_b32_e64 v100, v47 bound_ctrl:1
	v_permlane32_swap_b32_e64 v101, v70 bound_ctrl:1
	v_permlane32_swap_b32_e64 v102, v71 bound_ctrl:1
	v_perm_b32 v37, v71, v70, s6
	v_perm_b32 v36, v47, v103, s6
	v_perm_b32 v35, v102, v101, s6
	v_perm_b32 v34, v100, v99, s6
	v_pk_add_f32 v[48:49], v[48:49], v[82:83]
	v_pk_add_f32 v[42:43], v[42:43], v[44:45]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[66:69], v[34:37], v[18:33]
	v_pk_add_f32 v[38:39], v[38:39], v[40:41]
	v_pk_add_f32 v[40:41], v[48:49], v[92:93]
	v_pk_add_f32 v[38:39], v[38:39], v[42:43]
	v_add_f32_e32 v40, v40, v41
	v_add_f32_e32 v38, v38, v39
	v_mfma_f32_32x32x16_bf16 v[2:17], v[74:77], v[34:37], v[2:17]
	v_add_f32_e32 v34, v40, v38
	v_mov_b32_e32 v35, v34
	s_nop 1
	v_permlane32_swap_b32_e64 v34, v35 bound_ctrl:1
	v_add_f32_e32 v93, v35, v34
	v_fmac_f32_e32 v93, v88, v46
	s_cbranch_scc0 .LBB0_13
; %bb.14:                               ; %for.cond.cleanup58
	s_mul_i32 s0, s2, s20
	s_add_i32 s0, s0, s3
	s_mul_i32 s0, s0, s22
	v_rcp_f32_e32 v36, v93
	v_add_u32_e32 v34, s0, v87
	v_lshrrev_b32_e32 v0, 3, v0
	v_mul_lo_u32 v34, v34, s36
	v_and_b32_e32 v0, 4, v0
	v_ashrrev_i32_e32 v35, 31, v34
	v_mad_u64_u32 v[0:1], s[0:1], v1, s36, v[0:1]
	v_lshl_add_u64 v[34:35], v[34:35], 1, s[34:35]
	v_ashrrev_i32_e32 v1, 31, v0
	v_pk_mul_f32 v[20:21], v[20:21], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[36:37] op_sel_hi:[1,0]
	s_mov_b32 s0, 0x7060302
	v_pk_mul_f32 v[4:5], v[4:5], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[34:35]
	v_perm_b32 v21, v21, v20, s0
	v_perm_b32 v20, v19, v18, s0
	v_perm_b32 v5, v5, v4, s0
	v_perm_b32 v4, v3, v2, s0
	global_store_dwordx2 v[0:1], v[20:21], off
	v_pk_mul_f32 v[18:19], v[22:23], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[24:25], v[36:37] op_sel_hi:[1,0]
	global_store_dwordx2 v[0:1], v[4:5], off offset:64
	v_pk_mul_f32 v[2:3], v[6:7], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[8:9], v[36:37] op_sel_hi:[1,0]
	v_perm_b32 v21, v21, v20, s0
	v_perm_b32 v20, v19, v18, s0
	v_perm_b32 v5, v5, v4, s0
	v_perm_b32 v4, v3, v2, s0
	global_store_dwordx2 v[0:1], v[20:21], off offset:16
	v_pk_mul_f32 v[18:19], v[26:27], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[28:29], v[36:37] op_sel_hi:[1,0]
	global_store_dwordx2 v[0:1], v[4:5], off offset:80
	v_pk_mul_f32 v[2:3], v[10:11], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[12:13], v[36:37] op_sel_hi:[1,0]
	v_perm_b32 v21, v21, v20, s0
	v_perm_b32 v20, v19, v18, s0
	v_perm_b32 v5, v5, v4, s0
	v_perm_b32 v4, v3, v2, s0
	global_store_dwordx2 v[0:1], v[20:21], off offset:32
	v_pk_mul_f32 v[18:19], v[30:31], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[32:33], v[36:37] op_sel_hi:[1,0]
	global_store_dwordx2 v[0:1], v[4:5], off offset:96
	v_pk_mul_f32 v[2:3], v[14:15], v[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[16:17], v[36:37] op_sel_hi:[1,0]
	v_perm_b32 v21, v21, v20, s0
	v_perm_b32 v20, v19, v18, s0
	v_perm_b32 v5, v5, v4, s0
	v_perm_b32 v4, v3, v2, s0
	global_store_dwordx2 v[0:1], v[20:21], off offset:48
	global_store_dwordx2 v[0:1], v[4:5], off offset:112
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10attend_kerILi64EEv12attn_globalsIXT_EE
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 192
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
		.amdhsa_next_free_vgpr 109
		.amdhsa_next_free_sgpr 56
		.amdhsa_accum_offset 112
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
	.section	.text._Z10attend_kerILi64EEv12attn_globalsIXT_EE,"axG",@progbits,_Z10attend_kerILi64EEv12attn_globalsIXT_EE,comdat
.Lfunc_end0:
	.size	_Z10attend_kerILi64EEv12attn_globalsIXT_EE, .Lfunc_end0-_Z10attend_kerILi64EEv12attn_globalsIXT_EE
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4984
; NumSgprs: 62
; NumVgprs: 109
; NumAgprs: 0
; TotalNumVgprs: 109
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 109
; AccumOffset: 112
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 27
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_1ad496948509f0c0,@object ; @__hip_cuid_1ad496948509f0c0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_1ad496948509f0c0
__hip_cuid_1ad496948509f0c0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_1ad496948509f0c0, 1

	.ident	"AMD clang version 19.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25164 2b159522a6e9b34fe13b1d7b4c4ae751ef122765)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_1ad496948509f0c0
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           192
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 192
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10attend_kerILi64EEv12attn_globalsIXT_EE
    .private_segment_fixed_size: 0
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         _Z10attend_kerILi64EEv12attn_globalsIXT_EE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     109
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
