	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z10attend_kerILi128EEv12attn_globalsIXT_EE,"axG",@progbits,_Z10attend_kerILi128EEv12attn_globalsIXT_EE,comdat
	.protected	_Z10attend_kerILi128EEv12attn_globalsIXT_EE ; -- Begin function _Z10attend_kerILi128EEv12attn_globalsIXT_EE
	.globl	_Z10attend_kerILi128EEv12attn_globalsIXT_EE
	.p2align	8
	.type	_Z10attend_kerILi128EEv12attn_globalsIXT_EE,@function
_Z10attend_kerILi128EEv12attn_globalsIXT_EE: ; @_Z10attend_kerILi128EEv12attn_globalsIXT_EE
; %bb.0:
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s5, 0, 0
	s_mov_b32 s52, 0
	s_and_b32 s16, s5, -16
	s_and_b32 s14, s5, 15
	s_mov_b32 s15, s52
	s_add_i32 s16, s16, 16
	s_cmp_eq_u64 s[14:15], 0
	s_cselect_b32 s56, s5, s16
	s_add_i32 s5, s56, 0x8000
	s_load_dwordx2 s[12:13], s[0:1], 0x30
	s_load_dwordx2 s[6:7], s[0:1], 0x50
	s_load_dwordx4 s[28:31], s[0:1], 0x40
	s_load_dwordx2 s[8:9], s[0:1], 0x80
	s_load_dwordx2 s[10:11], s[0:1], 0x60
	s_load_dwordx4 s[40:43], s[0:1], 0x70
	s_and_b32 s16, s5, -16
	s_add_i32 s18, s16, 16
	s_lshl_b32 s16, s2, 3
	s_and_b32 s16, s16, 56
	s_lshr_b32 s2, s2, 3
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s17, s28, s4
	s_bfe_i64 s[44:45], s[30:31], 0x200000
	s_add_i32 s2, s16, s2
	s_mul_hi_i32 s16, s28, s4
	s_mul_i32 s20, s17, s45
	s_mul_hi_u32 s21, s17, s30
	s_add_i32 s20, s21, s20
	s_mul_i32 s16, s16, s30
	s_and_b32 s14, s5, 15
	s_lshr_b32 s19, s2, 3
	s_add_i32 s20, s20, s16
	s_mul_i32 s17, s17, s30
	s_add_u32 s16, s17, s19
	s_addc_u32 s17, s20, 0
	s_bfe_i64 s[46:47], s[6:7], 0x200000
	s_mul_i32 s7, s16, s47
	s_mul_hi_u32 s20, s16, s6
	s_add_i32 s7, s20, s7
	s_mul_i32 s17, s17, s6
	s_add_i32 s17, s7, s17
	s_mul_i32 s16, s16, s6
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s12, s12, s16
	s_addc_u32 s13, s13, s17
	s_mul_i32 s16, s40, s4
	s_bfe_i64 s[48:49], s[42:43], 0x200000
	s_mul_hi_i32 s7, s40, s4
	s_mul_i32 s17, s16, s49
	s_mul_hi_u32 s20, s16, s42
	s_add_i32 s17, s20, s17
	s_mul_i32 s7, s7, s42
	s_add_i32 s17, s17, s7
	s_mul_i32 s16, s16, s42
	s_add_u32 s7, s16, s19
	s_addc_u32 s16, s17, 0
	s_bfe_i64 s[50:51], s[8:9], 0x200000
	s_mul_i32 s9, s7, s51
	s_mul_hi_u32 s17, s7, s8
	s_add_i32 s9, s17, s9
	s_mul_i32 s16, s16, s8
	s_add_i32 s17, s9, s16
	s_mul_i32 s16, s7, s8
	s_lshl_b64 s[16:17], s[16:17], 1
	s_mul_i32 s57, s30, s6
	s_add_u32 s10, s10, s16
	s_mov_b32 s53, s57
	s_addc_u32 s11, s11, s17
	s_lshl_b64 s[6:7], s[52:53], 17
	s_or_b64 s[6:7], s[6:7], s[12:13]
	s_mul_i32 s55, s42, s8
	s_lshl_b32 s38, s57, 12
	s_and_b32 s8, s57, 0x1fff
	s_or_b32 s7, s7, -2.0
	s_mov_b32 s54, s52
	s_cmp_eq_u32 s8, 0
	s_cselect_b32 s36, s12, s6
	s_cselect_b32 s37, s13, s7
	s_lshl_b64 s[6:7], s[54:55], 17
	s_or_b64 s[6:7], s[6:7], s[10:11]
	v_lshrrev_b32_e32 v6, 6, v0
	s_lshl_b32 s33, s55, 12
	s_and_b32 s8, s55, 0x1fff
	s_or_b32 s7, s7, -2.0
	v_lshlrev_b32_e32 v1, 10, v6
	s_cmp_eq_u32 s8, 0
	v_add_u32_e32 v2, s56, v1
	v_lshlrev_b32_e32 v5, 4, v6
	s_cselect_b32 s28, s10, s6
	s_cselect_b32 s29, s11, s7
	s_cmp_eq_u64 s[14:15], 0
	v_readfirstlane_b32 s53, v2
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v4, 32, v0
	v_and_b32_e32 v7, 16, v5
	s_cselect_b32 s43, s5, s18
	v_and_b32_e32 v3, 0x3c0, v2
	s_movk_i32 s5, 0x400
	v_bitop3_b32 v2, v7, v2, v4 bitop3:0x36
	v_and_or_b32 v3, v1, s5, v3
	v_lshrrev_b32_e32 v2, 1, v2
	v_and_b32_e32 v4, 0x60, v5
	v_lshrrev_b32_e32 v3, 6, v3
	v_and_or_b32 v2, v2, 24, v4
	v_mad_u64_u32 v[2:3], s[6:7], v3, s57, v[2:3]
	s_lshl_b32 s5, s57, 5
	v_add_lshl_u32 v189, v2, s5, 1
	s_mov_b32 s5, s52
	s_mov_b32 s6, s53
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s39, 0x110000
	v_lshlrev_b32_e32 v191, 1, v2
	s_mov_b32 s7, s6
	s_mov_b32 m0, s52
	s_addk_i32 s6, 0x2000
	;;#ASMSTART
	;;#ASMEND
	v_add_u32_e32 v4, s43, v1
	;;#ASMSTART
	s_mov_b32 m0, s7
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s5 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s51, s56, 0x4000
	;;#ASMSTART
	s_mov_b32 m0, s6
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s5 offen lds
	s_load_dwordx8 s[16:23], s[0:1], 0x0
	s_load_dwordx2 s[58:59], s[0:1], 0x20
	s_load_dwordx8 s[8:15], s[0:1], 0x90
	s_load_dwordx2 s[40:41], s[0:1], 0xb0
	s_load_dwordx2 s[6:7], s[0:1], 0xc0
	s_load_dwordx2 s[34:35], s[0:1], 0xe0
	s_load_dwordx4 s[24:27], s[0:1], 0xd0
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s13, v4
	v_add_u32_e32 v4, s51, v1
	v_bfe_u32 v7, v0, 2, 3
	v_readfirstlane_b32 s25, v4
	v_lshrrev_b32_e32 v4, 4, v0
	v_lshlrev_b32_e32 v5, 3, v0
	v_and_or_b32 v7, v4, 24, v7
	v_and_b32_e32 v4, 0x60, v0
	v_and_or_b32 v4, v5, 24, v4
	s_mov_b32 s47, s42
	s_add_i32 s42, s43, 0x4000
	v_mad_u64_u32 v[4:5], s[0:1], v7, s55, v[4:5]
	v_add_u32_e32 v1, s42, v1
	s_lshl_b32 s0, s55, 5
	v_mov_b32_e32 v2, s16
	v_mov_b32_e32 v3, s17
	s_mov_b32 s49, s30
	s_mov_b32 s45, s52
	s_mov_b32 s11, s39
	s_mov_b32 s30, s33
	s_mov_b32 s31, s39
	v_readfirstlane_b32 s5, v1
	v_lshrrev_b32_e32 v1, 2, v0
	v_lshlrev_b32_e32 v202, 1, v4
	v_add_lshl_u32 v201, v4, s0, 1
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s0, s3, 8
	v_lshl_or_b32 v192, v6, 5, s0
	v_ashrrev_i32_e32 v193, 31, v192
	v_mov_b32_e32 v4, s4
	v_mad_i64_i32 v[4:5], s[0:1], s20, v4, v[192:193]
	s_mov_b32 s3, s52
	s_bfe_i64 s[0:1], s[22:23], 0x200000
	v_mov_b64_e32 v[6:7], s[2:3]
	v_mul_lo_u32 v8, v4, s1
	v_mul_lo_u32 v9, v5, s22
	v_mad_u64_u32 v[4:5], s[0:1], v4, s22, v[6:7]
	v_add3_u32 v5, v9, v5, v8
	s_bfe_i64 s[0:1], s[58:59], 0x200000
	v_mul_lo_u32 v6, v5, s58
	v_mul_lo_u32 v7, v4, s1
	v_mad_u64_u32 v[4:5], s[0:1], v4, s58, 0
	s_mul_i32 s0, s22, s58
	s_mul_i32 s1, s18, s20
	v_and_b32_e32 v195, 31, v0
	s_mul_i32 s1, s0, s1
	v_add3_u32 v5, v5, v7, v6
	v_and_b32_e32 v6, 8, v1
	s_lshl_b32 s1, s1, 1
	v_mul_lo_u32 v7, v195, s0
	v_lshl_add_u64 v[2:3], v[4:5], 1, v[2:3]
	v_mov_b32_e32 v4, s1
	v_mov_b32_e32 v5, 0x20000
	v_add_lshl_u32 v14, v7, v6, 1
	s_mov_b64 s[20:21], exec
	s_barrier
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_1
; %bb.2:
	s_mov_b64 exec, s[20:21]
	s_mov_b64 s[20:21], exec
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[10:13], v14, s[16:19], 0 offen offset:32
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_mov_b64 exec, s[20:21]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v15, 16, v6
	v_and_b32_e32 v16, 0xffff0000, v6
	v_lshlrev_b32_e32 v17, 16, v7
	v_and_b32_e32 v18, 0xffff0000, v7
	v_lshlrev_b32_e32 v19, 16, v8
	v_and_b32_e32 v20, 0xffff0000, v8
	v_lshlrev_b32_e32 v21, 16, v9
	v_and_b32_e32 v22, 0xffff0000, v9
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v23, 16, v10
	v_and_b32_e32 v10, 0xffff0000, v10
	v_lshlrev_b32_e32 v24, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v11
	v_lshlrev_b32_e32 v25, 16, v12
	v_and_b32_e32 v12, 0xffff0000, v12
	v_lshlrev_b32_e32 v26, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v13
	s_mov_b64 s[20:21], exec
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen offset:64
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_5
; %bb.6:
	s_mov_b64 exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v27, 16, v6
	v_and_b32_e32 v28, 0xffff0000, v6
	v_lshlrev_b32_e32 v29, 16, v7
	v_and_b32_e32 v30, 0xffff0000, v7
	v_lshlrev_b32_e32 v31, 16, v8
	v_and_b32_e32 v32, 0xffff0000, v8
	v_lshlrev_b32_e32 v33, 16, v9
	v_and_b32_e32 v34, 0xffff0000, v9
	s_mov_b64 s[20:21], exec
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen offset:96
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_mov_b64 exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v35, 16, v6
	v_and_b32_e32 v36, 0xffff0000, v6
	v_lshlrev_b32_e32 v37, 16, v7
	v_and_b32_e32 v38, 0xffff0000, v7
	v_lshlrev_b32_e32 v39, 16, v8
	v_and_b32_e32 v40, 0xffff0000, v8
	v_lshlrev_b32_e32 v41, 16, v9
	v_and_b32_e32 v42, 0xffff0000, v9
	s_mov_b64 s[20:21], exec
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen offset:128
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_9
; %bb.10:
	s_mov_b64 exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v43, 16, v6
	v_and_b32_e32 v44, 0xffff0000, v6
	v_lshlrev_b32_e32 v45, 16, v7
	v_and_b32_e32 v46, 0xffff0000, v7
	v_lshlrev_b32_e32 v47, 16, v8
	v_and_b32_e32 v48, 0xffff0000, v8
	v_lshlrev_b32_e32 v49, 16, v9
	v_and_b32_e32 v50, 0xffff0000, v9
	s_mov_b64 s[20:21], exec
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen offset:160
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_11
; %bb.12:
	s_mov_b64 exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v51, 16, v6
	v_and_b32_e32 v52, 0xffff0000, v6
	v_lshlrev_b32_e32 v53, 16, v7
	v_and_b32_e32 v54, 0xffff0000, v7
	v_lshlrev_b32_e32 v55, 16, v8
	v_and_b32_e32 v56, 0xffff0000, v8
	v_lshlrev_b32_e32 v57, 16, v9
	v_and_b32_e32 v58, 0xffff0000, v9
	s_mov_b64 s[20:21], exec
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen offset:192
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_13
; %bb.14:
	s_mov_b64 exec, s[20:21]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v59, 16, v6
	v_and_b32_e32 v60, 0xffff0000, v6
	v_lshlrev_b32_e32 v61, 16, v7
	v_and_b32_e32 v62, 0xffff0000, v7
	v_lshlrev_b32_e32 v63, 16, v8
	v_and_b32_e32 v64, 0xffff0000, v8
	v_lshlrev_b32_e32 v65, 16, v9
	v_and_b32_e32 v66, 0xffff0000, v9
	s_mov_b64 s[20:21], exec
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[16:19], 0 offen offset:224
                                        ; implicit-def: $vgpr2_vgpr3_vgpr4_vgpr5
                                        ; implicit-def: $vgpr14
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_15
; %bb.16:
	s_mov_b64 exec, s[20:21]
	s_lshl_b32 s0, s57, 6
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v6
	v_and_b32_e32 v3, 0xffff0000, v6
	v_lshlrev_b32_e32 v4, 16, v7
	v_and_b32_e32 v5, 0xffff0000, v7
	v_lshlrev_b32_e32 v6, 16, v8
	v_and_b32_e32 v7, 0xffff0000, v8
	v_lshlrev_b32_e32 v8, 16, v9
	v_and_b32_e32 v9, 0xffff0000, v9
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s25
	v_mul_f32_e32 v14, 0x3e0293ee, v15
	v_mul_f32_e32 v15, 0x3e0293ee, v16
	v_mul_f32_e32 v16, 0x3e0293ee, v17
	v_mul_f32_e32 v17, 0x3e0293ee, v18
	v_mul_f32_e32 v18, 0x3e0293ee, v19
	v_mul_f32_e32 v19, 0x3e0293ee, v20
	v_mul_f32_e32 v20, 0x3e0293ee, v21
	v_mul_f32_e32 v21, 0x3e0293ee, v22
	v_mul_f32_e32 v22, 0x3e0293ee, v23
	v_mul_f32_e32 v10, 0x3e0293ee, v10
	v_mul_f32_e32 v23, 0x3e0293ee, v24
	v_mul_f32_e32 v11, 0x3e0293ee, v11
	v_mul_f32_e32 v24, 0x3e0293ee, v25
	v_mul_f32_e32 v12, 0x3e0293ee, v12
	v_mul_f32_e32 v25, 0x3e0293ee, v26
	v_mul_f32_e32 v13, 0x3e0293ee, v13
	v_mul_f32_e32 v26, 0x3e0293ee, v27
	v_mul_f32_e32 v27, 0x3e0293ee, v28
	v_mul_f32_e32 v28, 0x3e0293ee, v29
	v_mul_f32_e32 v29, 0x3e0293ee, v30
	v_mul_f32_e32 v30, 0x3e0293ee, v31
	v_mul_f32_e32 v31, 0x3e0293ee, v32
	v_mul_f32_e32 v32, 0x3e0293ee, v33
	v_mul_f32_e32 v33, 0x3e0293ee, v34
	v_mul_f32_e32 v34, 0x3e0293ee, v35
	v_mul_f32_e32 v35, 0x3e0293ee, v36
	v_mul_f32_e32 v36, 0x3e0293ee, v37
	v_mul_f32_e32 v37, 0x3e0293ee, v38
	v_mul_f32_e32 v38, 0x3e0293ee, v39
	v_mul_f32_e32 v39, 0x3e0293ee, v40
	v_mul_f32_e32 v40, 0x3e0293ee, v41
	v_mul_f32_e32 v41, 0x3e0293ee, v42
	v_mul_f32_e32 v42, 0x3e0293ee, v43
	v_mul_f32_e32 v43, 0x3e0293ee, v44
	v_mul_f32_e32 v44, 0x3e0293ee, v45
	v_mul_f32_e32 v45, 0x3e0293ee, v46
	v_mul_f32_e32 v46, 0x3e0293ee, v47
	v_mul_f32_e32 v47, 0x3e0293ee, v48
	v_mul_f32_e32 v48, 0x3e0293ee, v49
	v_mul_f32_e32 v49, 0x3e0293ee, v50
	v_mul_f32_e32 v50, 0x3e0293ee, v51
	v_mul_f32_e32 v51, 0x3e0293ee, v52
	v_mul_f32_e32 v52, 0x3e0293ee, v53
	v_mul_f32_e32 v53, 0x3e0293ee, v54
	v_mul_f32_e32 v54, 0x3e0293ee, v55
	v_mul_f32_e32 v55, 0x3e0293ee, v56
	v_mul_f32_e32 v56, 0x3e0293ee, v57
	v_mul_f32_e32 v57, 0x3e0293ee, v58
	v_mul_f32_e32 v58, 0x3e0293ee, v59
	v_mul_f32_e32 v59, 0x3e0293ee, v60
	v_mul_f32_e32 v60, 0x3e0293ee, v61
	v_mul_f32_e32 v61, 0x3e0293ee, v62
	v_mul_f32_e32 v62, 0x3e0293ee, v63
	v_mul_f32_e32 v63, 0x3e0293ee, v64
	v_mul_f32_e32 v64, 0x3e0293ee, v65
	v_mul_f32_e32 v65, 0x3e0293ee, v66
	v_mul_f32_e32 v2, 0x3e0293ee, v2
	v_mul_f32_e32 v3, 0x3e0293ee, v3
	v_mul_f32_e32 v4, 0x3e0293ee, v4
	v_mul_f32_e32 v5, 0x3e0293ee, v5
	v_mul_f32_e32 v6, 0x3e0293ee, v6
	v_mul_f32_e32 v7, 0x3e0293ee, v7
	v_mul_f32_e32 v8, 0x3e0293ee, v8
	v_mul_f32_e32 v9, 0x3e0293ee, v9
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v124, v14, v15
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v125, v16, v17
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v126, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v127, v20, v21
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v120, v22, v10
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v121, v23, v11
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v122, v24, v12
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v123, v25, v13
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v116, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v117, v28, v29
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v118, v30, v31
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v119, v32, v33
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v34, v35
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v36, v37
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v114, v38, v39
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v115, v40, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v42, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v44, v45
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v46, v47
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v48, v49
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v50, v51
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v52, v53
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v54, v55
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v56, v57
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v58, v59
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v60, v61
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v62, v63
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v2, v3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v8, v9
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s16, s1
	s_mov_b32 m0, 0
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v2, 6, v0
	;;#ASMSTART
	s_mov_b32 m0, s16
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v3, 1, v0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	s_mov_b32 s0, s52
	s_mov_b32 s1, s13
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s16, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v34, 2, v0
	;;#ASMSTART
	s_mov_b32 m0, s16
	;;#ASMEND
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	v_and_b32_e32 v35, 16, v0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	v_and_b32_e32 v2, 0x7c0, v2
	v_and_b32_e32 v3, 16, v3
	v_and_or_b32 v19, v34, 32, v35
	v_bitop3_b32 v37, v3, v19, v2 bitop3:0x36
	v_or_b32_e32 v18, v3, v2
	v_add_u32_e32 v206, s56, v37
	;;#ASMSTART
	ds_read_b128 v[2:5], v206 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v206 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v206 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v206 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v206 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v206 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v206 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v206 offset:0x3800

	;;#ASMEND
	v_bitop3_b32 v36, v18, v19, 32 bitop3:0x36
	v_add_u32_e32 v207, s56, v36
	;;#ASMSTART
	ds_read_b128 v[54:57], v207 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v207 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v207 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[66:69], v207 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v207 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v207 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v207 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[82:85], v207 offset:0x3800

	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[2:5], v[124:127], 0
	s_mov_b32 s57, 0
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[18:33], v[54:57], v[120:123], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[6:9], v[116:119], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[58:61], v[112:115], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[10:13], v[108:111], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[62:65], v[104:107], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[14:17], v[100:103], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[38:41], v[124:127], 0
	v_mfma_f32_32x32x16_bf16 v[2:17], v[70:73], v[120:123], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[42:45], v[116:119], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[74:77], v[112:115], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[46:49], v[108:111], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[78:81], v[104:107], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[50:53], v[100:103], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[66:69], v[96:99], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[82:85], v[96:99], v[2:17]
	s_nop 7
	s_nop 2
	v_max_f32_e32 v38, v19, v21
	v_max3_f32 v39, v18, v20, v22
	v_max3_f32 v38, v38, v23, v25
	v_max3_f32 v39, v39, v24, v26
	v_max3_f32 v38, v38, v27, v29
	v_max3_f32 v39, v39, v28, v30
	v_max3_f32 v38, v38, v31, v33
	v_max3_f32 v39, v39, v32, v2
	v_max3_f32 v38, v38, v3, v5
	v_max3_f32 v39, v39, v4, v6
	v_max3_f32 v38, v38, v7, v9
	v_max3_f32 v39, v39, v8, v10
	v_max3_f32 v38, v38, v11, v13
	v_max3_f32 v39, v39, v12, v14
	v_max3_f32 v38, v38, v15, v17
	v_max3_f32 v38, v39, v16, v38
	v_mov_b32_e32 v39, v38
	s_nop 1
	v_permlane32_swap_b32_e64 v38, v39 bound_ctrl:1
	v_max_f32_e32 v196, v38, v39
	v_sub_f32_e32 v18, v18, v196
	v_sub_f32_e32 v19, v19, v196
	v_sub_f32_e32 v20, v20, v196
	v_sub_f32_e32 v21, v21, v196
	v_sub_f32_e32 v22, v22, v196
	v_sub_f32_e32 v23, v23, v196
	v_sub_f32_e32 v24, v24, v196
	v_sub_f32_e32 v25, v25, v196
	v_sub_f32_e32 v26, v26, v196
	v_sub_f32_e32 v27, v27, v196
	v_sub_f32_e32 v28, v28, v196
	v_sub_f32_e32 v29, v29, v196
	v_sub_f32_e32 v30, v30, v196
	v_sub_f32_e32 v31, v31, v196
	v_sub_f32_e32 v32, v32, v196
	v_sub_f32_e32 v33, v33, v196
	; sched_barrier mask(0x00000000)
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_movk_i32 s0, 0xff
	v_cmp_lt_u32_e64 s[0:1], s0, v0
	s_and_saveexec_b64 s[16:17], s[0:1]
	s_cbranch_execz .LBB0_18
; %bb.17:
	; sched_barrier mask(0x00000000)
	s_barrier
.LBB0_18:
	s_or_b64 exec, exec, s[16:17]
	v_exp_f32_e32 v209, v18
	v_exp_f32_e32 v210, v19
	v_exp_f32_e32 v188, v20
	v_exp_f32_e32 v213, v21
	v_exp_f32_e32 v190, v22
	v_exp_f32_e32 v214, v23
	v_exp_f32_e32 v211, v24
	v_exp_f32_e32 v216, v25
	v_exp_f32_e32 v212, v26
	v_exp_f32_e32 v219, v27
	v_exp_f32_e32 v215, v28
	v_exp_f32_e32 v220, v29
	v_exp_f32_e32 v217, v30
	v_exp_f32_e32 v221, v31
	v_exp_f32_e32 v218, v32
	v_exp_f32_e32 v222, v33
	v_sub_f32_e32 v178, v2, v196
	v_sub_f32_e32 v179, v3, v196
	v_sub_f32_e32 v174, v4, v196
	v_sub_f32_e32 v175, v5, v196
	v_sub_f32_e32 v180, v6, v196
	v_sub_f32_e32 v181, v7, v196
	v_sub_f32_e32 v176, v8, v196
	v_sub_f32_e32 v177, v9, v196
	v_sub_f32_e32 v172, v10, v196
	v_sub_f32_e32 v173, v11, v196
	v_sub_f32_e32 v186, v12, v196
	v_sub_f32_e32 v187, v13, v196
	v_sub_f32_e32 v184, v14, v196
	v_sub_f32_e32 v185, v15, v196
	v_sub_f32_e32 v182, v16, v196
	v_sub_f32_e32 v183, v17, v196
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v204, s51, v37
	;;#ASMSTART
	ds_read_b128 v[64:67], v204 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v204 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v204 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v204 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v204 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v204 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v204 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v204 offset:0x3800

	;;#ASMEND
	v_add_u32_e32 v205, s51, v36
	;;#ASMSTART
	ds_read_b128 v[168:171], v205 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v205 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v205 offset:0x1000

	;;#ASMEND
	v_mov_b32_e32 v2, s52
	;;#ASMSTART
	ds_read_b128 v[156:159], v205 offset:0x1800

	;;#ASMEND
	v_alignbit_b32 v2, s49, v2, 25
	;;#ASMSTART
	ds_read_b128 v[152:155], v205 offset:0x2000

	;;#ASMEND
	v_mul_lo_u32 v2, s46, v2
	;;#ASMSTART
	ds_read_b128 v[148:151], v205 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v205 offset:0x3000

	;;#ASMEND
	s_mov_b32 s1, s53
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s0, s0, 1
	v_mov_b32_e32 v2, s45
	;;#ASMSTART
	ds_read_b128 v[128:131], v205 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_alignbit_b32 v2, s47, v2, 26
	s_mov_b32 s16, s1
	s_addk_i32 s1, 0x2000
	v_mul_lo_u32 v2, s50, v2
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, s33
	;;#ASMSTART
	s_mov_b32 m0, s16
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s31, s11
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	v_readfirstlane_b32 s0, v2
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s5
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s16, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s16
	;;#ASMEND
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshrrev_b32_e32 v0, 3, v0
	v_and_b32_e32 v197, 4, v0
	v_and_or_b32 v0, v1, 3, v197
	v_and_or_b32 v1, v34, 12, v35
	v_lshlrev_b32_e32 v1, 1, v1
	v_lshl_or_b32 v0, v0, 6, v1
	s_mul_hi_i32 s17, s50, s48
	s_mul_i32 s16, s50, s48
	s_mul_hi_i32 s19, s46, s44
	s_mul_i32 s18, s46, s44
	v_add_u32_e32 v203, s43, v0
	v_add_u32_e32 v200, s42, v0
	s_lshl_b64 s[42:43], s[16:17], 7
	v_mov_b32_e32 v199, 0
	s_mul_i32 s52, s16, 0x140
	s_lshl_b64 s[20:21], s[16:17], 8
	s_mul_i32 s54, s18, 0x140
	s_lshl_b64 s[22:23], s[18:19], 8
	s_mul_i32 s19, s18, 0x180
	s_mul_i32 s17, s16, 0xc0
	s_mul_i32 s43, s18, 0xc0
	s_mov_b32 s55, -1
	s_mov_b64 s[46:47], 0
	v_mov_b32_e32 v208, 1.0
	s_mov_b32 s56, 0x41000000
	s_mov_b64 s[44:45], 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, v199
	v_mov_b32_e32 v2, v199
	v_mov_b32_e32 v3, v199
	v_mov_b32_e32 v4, v199
	v_mov_b32_e32 v5, v199
	v_mov_b32_e32 v6, v199
	v_mov_b32_e32 v7, v199
	v_mov_b32_e32 v8, v199
	v_mov_b32_e32 v9, v199
	v_mov_b32_e32 v10, v199
	v_mov_b32_e32 v11, v199
	v_mov_b32_e32 v12, v199
	v_mov_b32_e32 v13, v199
	v_mov_b32_e32 v14, v199
	v_mov_b32_e32 v15, v199
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, v199
	v_mov_b32_e32 v18, v199
	v_mov_b32_e32 v19, v199
	v_mov_b32_e32 v20, v199
	v_mov_b32_e32 v21, v199
	v_mov_b32_e32 v22, v199
	v_mov_b32_e32 v23, v199
	v_mov_b32_e32 v24, v199
	v_mov_b32_e32 v25, v199
	v_mov_b32_e32 v26, v199
	v_mov_b32_e32 v27, v199
	v_mov_b32_e32 v28, v199
	v_mov_b32_e32 v29, v199
	v_mov_b32_e32 v30, v199
	v_mov_b32_e32 v31, v199
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, v199
	v_mov_b32_e32 v34, v199
	v_mov_b32_e32 v35, v199
	v_mov_b32_e32 v36, v199
	v_mov_b32_e32 v37, v199
	v_mov_b32_e32 v38, v199
	v_mov_b32_e32 v39, v199
	v_mov_b32_e32 v40, v199
	v_mov_b32_e32 v41, v199
	v_mov_b32_e32 v42, v199
	v_mov_b32_e32 v43, v199
	v_mov_b32_e32 v44, v199
	v_mov_b32_e32 v45, v199
	v_mov_b32_e32 v46, v199
	v_mov_b32_e32 v47, v199
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, v199
	v_mov_b32_e32 v50, v199
	v_mov_b32_e32 v51, v199
	v_mov_b32_e32 v52, v199
	v_mov_b32_e32 v53, v199
	v_mov_b32_e32 v54, v199
	v_mov_b32_e32 v55, v199
	v_mov_b32_e32 v56, v199
	v_mov_b32_e32 v57, v199
	v_mov_b32_e32 v58, v199
	v_mov_b32_e32 v59, v199
	v_mov_b32_e32 v60, v199
	v_mov_b32_e32 v61, v199
	v_mov_b32_e32 v62, v199
	v_mov_b32_e32 v63, v199
	s_barrier
.LBB0_19:                               ; =>This Inner Loop Header: Depth=1
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v179, v179
	v_exp_f32_e32 v174, v174
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v224, v209, v210
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v225, v188, v213
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v226, v190, v214
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v227, v211, v216
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[120:123], v[64:79]
	v_exp_f32_e32 v168, v175
	v_exp_f32_e32 v169, v180
	v_exp_f32_e32 v170, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[92:95], v[116:119], v[64:79]
	v_exp_f32_e32 v171, v176
	v_exp_f32_e32 v175, v177
	v_exp_f32_e32 v172, v172
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[112:115], v[64:79]
	v_exp_f32_e32 v164, v173
	v_exp_f32_e32 v165, v186
	v_exp_f32_e32 v166, v187
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[108:111], v[64:79]
	v_exp_f32_e32 v167, v184
	v_exp_f32_e32 v173, v185
	v_exp_f32_e32 v176, v182
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[160:163], v[104:107], v[64:79]
	v_exp_f32_e32 v160, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[100:103], v[64:79]
	v_add_f32_e32 v84, v210, v209
	v_add_f32_e32 v84, v84, v188
	v_add_f32_e32 v84, v84, v213
	v_add_f32_e32 v84, v84, v190
	v_add_f32_e32 v84, v84, v214
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[96:99], v[64:79]
	v_add_f32_e32 v84, v84, v211
	v_add_f32_e32 v84, v84, v216
	v_add_f32_e32 v84, v84, v212
	v_add_f32_e32 v84, v84, v219
	v_add_f32_e32 v156, v84, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[80:83], v[124:127], 0
	v_add_f32_e32 v156, v156, v220
	v_add_f32_e32 v156, v156, v217
	v_add_f32_e32 v156, v156, v221
	v_add_f32_e32 v156, v156, v218
	v_add_f32_e32 v156, v156, v222
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[152:155], v[120:123], v[80:95]
	v_add_f32_e32 v152, v156, v178
	v_add_f32_e32 v152, v152, v179
	v_add_f32_e32 v152, v152, v174
	v_add_f32_e32 v152, v152, v168
	v_add_f32_e32 v152, v152, v169
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[140:143], v[116:119], v[80:95]
	v_add_f32_e32 v140, v152, v170
	v_add_f32_e32 v140, v140, v171
	v_add_f32_e32 v140, v140, v175
	v_add_f32_e32 v140, v140, v172
	v_add_f32_e32 v140, v140, v164
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[112:115], v[80:95]
	v_add_f32_e32 v140, v140, v165
	v_add_f32_e32 v140, v140, v166
	v_add_f32_e32 v140, v140, v167
	v_add_f32_e32 v140, v140, v173
	v_add_f32_e32 v140, v140, v176
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[136:139], v[108:111], v[80:95]
	v_add_f32_e32 v209, v140, v160
	v_mov_b32_e32 v210, v209
	s_nop 1
	v_permlane32_swap_b32_e64 v209, v210 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v140, v212, v219
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v141, v215, v220
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v142, v217, v221
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[144:147], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v143, v218, v222
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v178, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v174, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v169, v170
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v171, v175
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[100:103], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v172, v164
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v165, v166
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v167, v173
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v176, v160
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s43, s46
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s25
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, s1
	s_mov_b32 m0, 0
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s30
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v203 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v203 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[228:229], v203 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v203 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v203 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v203 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v203 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v203 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v203 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v203 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v203 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v203 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v203 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v203 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v203 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v203 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[230:231], v203 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v203 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v203 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v203 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v203 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v203 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v203 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v203 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v203 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v203 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v203 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v203 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v203 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[224:227], v[48:63]
	v_max_f32_e32 v188, v65, v67
	v_max3_f32 v190, v64, v66, v68
	v_max3_f32 v188, v188, v69, v71
	v_max3_f32 v190, v190, v70, v72
	v_max3_f32 v188, v188, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[216:219], v[224:227], v[32:47]
	v_max3_f32 v190, v190, v74, v76
	v_max3_f32 v188, v188, v77, v79
	v_max3_f32 v190, v190, v78, v80
	v_max3_f32 v188, v188, v81, v83
	v_max3_f32 v190, v190, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[220:223], v[224:227], v[16:31]
	v_max3_f32 v188, v188, v85, v87
	v_max3_f32 v190, v190, v86, v88
	v_max3_f32 v188, v188, v89, v91
	v_max3_f32 v190, v190, v90, v92
	v_max3_f32 v188, v188, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[228:231], v[224:227], v[0:15]
	v_max3_f32 v188, v190, v94, v188
	v_mov_b32_e32 v190, v188
	s_nop 1
	v_permlane32_swap_b32_e64 v188, v190 bound_ctrl:1
	v_max3_f32 v213, v196, v188, v190
	v_sub_f32_e32 v188, v213, v196
	v_cmp_ge_f32_e64 s[0:1], s56, v188
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	s_nop 1
	v_cndmask_b32_e64 v188, 0, 1, s[0:1]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_cmp_ne_u32_e64 s[0:1], 0, v188
	s_cmp_eq_u64 s[0:1], exec
	s_cbranch_scc0 .LBB0_29
; %bb.20:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b32_e32 v188, 1.0
	v_mov_b32_e32 v194, v208
	v_mov_b32_e32 v213, v196
.LBB0_21:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[140:143], v[48:63]
	v_sub_f32_e32 v64, v64, v213
	v_sub_f32_e32 v65, v65, v213
	v_sub_f32_e32 v66, v66, v213
	v_sub_f32_e32 v67, v67, v213
	v_sub_f32_e32 v68, v68, v213
	v_sub_f32_e32 v211, v94, v213
	v_sub_f32_e32 v212, v95, v213
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[140:143], v[32:47]
	v_sub_f32_e32 v69, v69, v213
	v_sub_f32_e32 v70, v70, v213
	v_sub_f32_e32 v71, v71, v213
	v_sub_f32_e32 v72, v72, v213
	v_sub_f32_e32 v73, v73, v213
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[140:143], v[16:31]
	v_sub_f32_e32 v74, v74, v213
	v_sub_f32_e32 v75, v75, v213
	v_sub_f32_e32 v76, v76, v213
	v_sub_f32_e32 v77, v77, v213
	v_sub_f32_e32 v78, v78, v213
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[140:143], v[0:15]
	v_sub_f32_e32 v79, v79, v213
	v_sub_f32_e32 v172, v80, v213
	v_sub_f32_e32 v173, v81, v213
	v_sub_f32_e32 v174, v82, v213
	v_sub_f32_e32 v175, v83, v213
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[136:139], v[48:63]
	v_sub_f32_e32 v176, v84, v213
	v_sub_f32_e32 v177, v85, v213
	v_sub_f32_e32 v178, v86, v213
	v_sub_f32_e32 v179, v87, v213
	v_sub_f32_e32 v180, v88, v213
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[136:139], v[32:47]
	v_sub_f32_e32 v181, v89, v213
	v_sub_f32_e32 v182, v90, v213
	v_sub_f32_e32 v183, v91, v213
	v_sub_f32_e32 v184, v92, v213
	v_sub_f32_e32 v185, v93, v213
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[160:163], v[136:139], v[16:31]
	v_exp_f32_e32 v186, v64
	v_exp_f32_e32 v187, v65
	v_exp_f32_e32 v190, v66
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[156:159], v[136:139], v[0:15]
	v_exp_f32_e32 v196, v67
	v_exp_f32_e32 v198, v68
	v_exp_f32_e32 v216, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[152:155], v[132:135], v[48:63]
	v_exp_f32_e32 v217, v70
	v_exp_f32_e32 v218, v71
	v_exp_f32_e32 v219, v72
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[132:135], v[32:47]
	v_exp_f32_e32 v220, v73
	v_exp_f32_e32 v221, v74
	v_exp_f32_e32 v222, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[144:147], v[132:135], v[16:31]
	v_exp_f32_e32 v223, v76
	v_exp_f32_e32 v224, v77
	v_exp_f32_e32 v225, v78
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[132:135], v[0:15]
	v_exp_f32_e32 v226, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s42, s44
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s13
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, s1
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s31, s11
	;;#ASMSTART
	s_mov_b32 m0, s30
	;;#ASMEND
	s_mov_b32 s30, s33
	s_addk_i32 s1, 0x2000
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v206 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v206 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v206 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v206 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v206 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[128:131], v206 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v206 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v206 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v207 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v207 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v207 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v207 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v207 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v207 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v207 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v207 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v173, v173
	v_exp_f32_e32 v174, v174
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v214, v186, v187
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v215, v190, v196
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[136:139], v[120:123], v[64:79]
	v_exp_f32_e32 v175, v175
	v_exp_f32_e32 v176, v176
	v_exp_f32_e32 v177, v177
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[116:119], v[64:79]
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v179, v179
	v_exp_f32_e32 v180, v180
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[112:115], v[64:79]
	v_exp_f32_e32 v144, v181
	v_exp_f32_e32 v145, v182
	v_exp_f32_e32 v146, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[108:111], v[64:79]
	v_exp_f32_e32 v147, v184
	v_exp_f32_e32 v181, v185
	v_exp_f32_e32 v182, v211
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[104:107], v[64:79]
	v_exp_f32_e32 v148, v212
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[100:103], v[64:79]
	v_add_f32_e32 v80, v187, v186
	v_add_f32_e32 v80, v80, v190
	v_add_f32_e32 v80, v80, v196
	v_add_f32_e32 v80, v80, v198
	v_add_f32_e32 v80, v80, v216
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v216, v198, v216
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[96:99], v[64:79]
	v_add_f32_e32 v80, v80, v217
	v_add_f32_e32 v80, v80, v218
	v_add_f32_e32 v80, v80, v219
	v_add_f32_e32 v80, v80, v220
	v_add_f32_e32 v136, v80, v221
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v217, v217, v218
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[92:95], v[124:127], 0
	v_add_f32_e32 v136, v136, v222
	v_add_f32_e32 v136, v136, v223
	v_add_f32_e32 v136, v136, v224
	v_add_f32_e32 v136, v136, v225
	v_add_f32_e32 v136, v136, v226
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[120:123], v[80:95]
	v_add_f32_e32 v136, v136, v172
	v_add_f32_e32 v136, v136, v173
	v_add_f32_e32 v136, v136, v174
	v_add_f32_e32 v136, v136, v175
	v_add_f32_e32 v136, v136, v176
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[116:119], v[80:95]
	v_add_f32_e32 v128, v136, v177
	v_add_f32_e32 v128, v128, v178
	v_add_f32_e32 v128, v128, v179
	v_add_f32_e32 v128, v128, v180
	v_add_f32_e32 v128, v128, v144
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v219, v220
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v221, v222
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[112:115], v[80:95]
	v_add_f32_e32 v128, v128, v145
	v_add_f32_e32 v128, v128, v146
	v_add_f32_e32 v128, v128, v147
	v_add_f32_e32 v128, v128, v181
	v_add_f32_e32 v128, v128, v182
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v223, v224
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v225, v226
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[108:111], v[80:95]
	v_add_f32_e32 v211, v128, v148
	v_mov_b32_e32 v212, v211
	s_nop 1
	v_permlane32_swap_b32_e64 v211, v212 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v172, v173
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v174, v175
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v176, v177
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v178, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v128, v180, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v129, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v130, v147, v181
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v131, v182, v148
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[140:143], v[100:103], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[168:171], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s22, s46
	s_addc_u32 s49, s23, s47
	s_lshl_b32 s0, s48, 1
	s_mov_b32 s1, s53
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s47, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s47
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v200 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[226:227], v200 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[230:231], v200 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v200 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v200 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v200 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v200 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v200 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v200 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v200 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v200 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v200 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v200 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v200 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[224:225], v200 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[228:229], v200 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[232:233], v200 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v200 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v200 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v200 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v200 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v200 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v200 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v200 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v200 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v200 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v200 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v200 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[218:221], v[214:217], v[48:63]
	v_max_f32_e32 v190, v65, v67
	v_max3_f32 v196, v64, v66, v68
	v_max3_f32 v190, v190, v69, v71
	v_max3_f32 v196, v196, v70, v72
	v_max3_f32 v190, v190, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[222:225], v[214:217], v[32:47]
	v_max3_f32 v196, v196, v74, v76
	v_max3_f32 v190, v190, v77, v79
	v_max3_f32 v196, v196, v78, v80
	v_max3_f32 v190, v190, v81, v83
	v_max3_f32 v196, v196, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[226:229], v[214:217], v[16:31]
	v_max3_f32 v190, v190, v85, v87
	v_max3_f32 v196, v196, v86, v88
	v_max3_f32 v190, v190, v89, v91
	v_max3_f32 v196, v196, v90, v92
	v_max3_f32 v190, v190, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[230:233], v[214:217], v[0:15]
	v_max3_f32 v190, v196, v94, v190
	v_mov_b32_e32 v196, v190
	s_nop 1
	v_permlane32_swap_b32_e64 v190, v196 bound_ctrl:1
	v_max3_f32 v196, v213, v190, v196
	v_sub_f32_e32 v190, v196, v213
	v_cmp_ge_f32_e64 s[0:1], s56, v190
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	s_nop 1
	v_cndmask_b32_e64 v190, 0, 1, s[0:1]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_cmp_ne_u32_e64 s[0:1], 0, v190
	s_cmp_eq_u64 s[0:1], exec
	v_mov_b32_e32 v190, 1.0
	s_cbranch_scc0 .LBB0_30
; %bb.22:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b32_e32 v198, 1.0
	v_mov_b32_e32 v196, v213
.LBB0_23:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[136:139], v[48:63]
	v_sub_f32_e32 v64, v64, v196
	v_sub_f32_e32 v65, v65, v196
	v_sub_f32_e32 v66, v66, v196
	v_sub_f32_e32 v67, v67, v196
	v_sub_f32_e32 v68, v68, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[136:139], v[32:47]
	v_sub_f32_e32 v69, v69, v196
	v_sub_f32_e32 v70, v70, v196
	v_sub_f32_e32 v71, v71, v196
	v_sub_f32_e32 v72, v72, v196
	v_sub_f32_e32 v73, v73, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[136:139], v[16:31]
	v_sub_f32_e32 v74, v74, v196
	v_sub_f32_e32 v75, v75, v196
	v_sub_f32_e32 v76, v76, v196
	v_sub_f32_e32 v77, v77, v196
	v_sub_f32_e32 v78, v78, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[136:139], v[0:15]
	v_sub_f32_e32 v79, v79, v196
	v_add_f32_e64 v172, v80, -v196
	v_add_f32_e64 v173, v81, -v196
	v_add_f32_e64 v174, v82, -v196
	v_add_f32_e64 v175, v83, -v196
	v_pk_add_f32 v[176:177], v[84:85], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[178:179], v[86:87], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[132:135], v[48:63]
	v_add_f32_e64 v180, v88, -v196
	v_add_f32_e64 v181, v89, -v196
	v_add_f32_e64 v182, v90, -v196
	v_add_f32_e64 v183, v91, -v196
	v_add_f32_e64 v184, v92, -v196
	v_add_f32_e64 v185, v93, -v196
	v_pk_add_f32 v[186:187], v[94:95], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[132:135], v[32:47]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[160:163], v[132:135], v[16:31]
	v_exp_f32_e32 v213, v64
	v_exp_f32_e32 v214, v65
	v_exp_f32_e32 v215, v66
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[156:159], v[132:135], v[0:15]
	v_exp_f32_e32 v217, v67
	v_exp_f32_e32 v218, v68
	v_exp_f32_e32 v219, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[152:155], v[128:131], v[48:63]
	v_exp_f32_e32 v220, v70
	v_exp_f32_e32 v221, v71
	v_exp_f32_e32 v222, v72
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[128:131], v[32:47]
	v_exp_f32_e32 v223, v73
	v_exp_f32_e32 v224, v74
	v_exp_f32_e32 v225, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[144:147], v[128:131], v[16:31]
	v_exp_f32_e32 v226, v76
	v_exp_f32_e32 v227, v77
	v_exp_f32_e32 v228, v78
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[140:143], v[128:131], v[0:15]
	v_exp_f32_e32 v229, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s17, s44
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s5
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s47, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s47
	;;#ASMEND
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v204 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v204 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v204 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v204 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v204 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[128:131], v204 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v204 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v204 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v205 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v205 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v205 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v205 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v205 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v205 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v205 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v205 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v173, v173
	v_exp_f32_e32 v174, v174
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v216, v213, v214
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[136:139], v[120:123], v[64:79]
	v_exp_f32_e32 v175, v175
	v_exp_f32_e32 v176, v176
	v_exp_f32_e32 v177, v177
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[116:119], v[64:79]
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v179, v179
	v_exp_f32_e32 v180, v180
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[112:115], v[64:79]
	v_exp_f32_e32 v144, v181
	v_exp_f32_e32 v145, v182
	v_exp_f32_e32 v146, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[108:111], v[64:79]
	v_exp_f32_e32 v147, v184
	v_exp_f32_e32 v181, v185
	v_exp_f32_e32 v182, v186
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[104:107], v[64:79]
	v_exp_f32_e32 v148, v187
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[100:103], v[64:79]
	v_add_f32_e32 v80, v214, v213
	v_add_f32_e32 v80, v80, v215
	v_add_f32_e32 v80, v80, v217
	v_add_f32_e32 v80, v80, v218
	v_add_f32_e32 v80, v80, v219
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v217, v215, v217
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v218, v218, v219
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[96:99], v[64:79]
	v_add_f32_e32 v80, v80, v220
	v_add_f32_e32 v80, v80, v221
	v_add_f32_e32 v80, v80, v222
	v_add_f32_e32 v80, v80, v223
	v_add_f32_e32 v136, v80, v224
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v219, v220, v221
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[92:95], v[124:127], 0
	v_add_f32_e32 v136, v136, v225
	v_add_f32_e32 v136, v136, v226
	v_add_f32_e32 v136, v136, v227
	v_add_f32_e32 v136, v136, v228
	v_add_f32_e32 v136, v136, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[120:123], v[80:95]
	v_add_f32_e32 v136, v136, v172
	v_add_f32_e32 v136, v136, v173
	v_add_f32_e32 v136, v136, v174
	v_add_f32_e32 v136, v136, v175
	v_add_f32_e32 v136, v136, v176
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[116:119], v[80:95]
	v_add_f32_e32 v128, v136, v177
	v_add_f32_e32 v128, v128, v178
	v_add_f32_e32 v128, v128, v179
	v_add_f32_e32 v128, v128, v180
	v_add_f32_e32 v128, v128, v144
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v222, v223
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v224, v225
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[112:115], v[80:95]
	v_add_f32_e32 v128, v128, v145
	v_add_f32_e32 v128, v128, v146
	v_add_f32_e32 v128, v128, v147
	v_add_f32_e32 v128, v128, v181
	v_add_f32_e32 v128, v128, v182
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v226, v227
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v228, v229
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[108:111], v[80:95]
	v_add_f32_e32 v213, v128, v148
	v_mov_b32_e32 v214, v213
	s_nop 1
	v_permlane32_swap_b32_e64 v213, v214 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v172, v173
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v174, v175
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v176, v177
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v178, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v128, v180, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v129, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v130, v147, v181
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v131, v182, v148
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[140:143], v[100:103], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[168:171], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s54, s46
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s25
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s30
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[224:225], v203 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[228:229], v203 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[232:233], v203 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v203 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v203 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v203 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v203 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v203 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v203 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v203 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v203 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v203 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v203 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v203 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[226:227], v203 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[230:231], v203 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[234:235], v203 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v203 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v203 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v203 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v203 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v203 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v203 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v203 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v203 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v203 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v203 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v203 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v203 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[216:219], v[48:63]
	v_max_f32_e32 v215, v65, v67
	v_max3_f32 v220, v64, v66, v68
	v_max3_f32 v215, v215, v69, v71
	v_max3_f32 v220, v220, v70, v72
	v_max3_f32 v215, v215, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[224:227], v[216:219], v[32:47]
	v_max3_f32 v220, v220, v74, v76
	v_max3_f32 v215, v215, v77, v79
	v_max3_f32 v220, v220, v78, v80
	v_max3_f32 v215, v215, v81, v83
	v_max3_f32 v220, v220, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[228:231], v[216:219], v[16:31]
	v_max3_f32 v215, v215, v85, v87
	v_max3_f32 v220, v220, v86, v88
	v_max3_f32 v215, v215, v89, v91
	v_max3_f32 v220, v220, v90, v92
	v_max3_f32 v215, v215, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[232:235], v[216:219], v[0:15]
	v_max3_f32 v215, v220, v94, v215
	v_mov_b32_e32 v216, v215
	s_nop 1
	v_permlane32_swap_b32_e64 v215, v216 bound_ctrl:1
	v_max3_f32 v215, v196, v215, v216
	v_sub_f32_e32 v216, v215, v196
	v_cmp_ge_f32_e64 s[0:1], s56, v216
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	s_nop 1
	v_cndmask_b32_e64 v216, 0, 1, s[0:1]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_cmp_ne_u32_e64 s[0:1], 0, v216
	s_cmp_eq_u64 s[0:1], exec
	s_cbranch_scc0 .LBB0_31
; %bb.24:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b32_e32 v215, v196
.LBB0_25:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[136:139], v[48:63]
	v_sub_f32_e32 v64, v64, v215
	v_sub_f32_e32 v65, v65, v215
	v_sub_f32_e32 v66, v66, v215
	v_sub_f32_e32 v67, v67, v215
	v_sub_f32_e32 v68, v68, v215
	v_sub_f32_e32 v216, v94, v215
	v_sub_f32_e32 v217, v95, v215
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[136:139], v[32:47]
	v_sub_f32_e32 v69, v69, v215
	v_sub_f32_e32 v70, v70, v215
	v_sub_f32_e32 v71, v71, v215
	v_sub_f32_e32 v72, v72, v215
	v_sub_f32_e32 v73, v73, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[136:139], v[16:31]
	v_sub_f32_e32 v74, v74, v215
	v_sub_f32_e32 v75, v75, v215
	v_sub_f32_e32 v76, v76, v215
	v_sub_f32_e32 v77, v77, v215
	v_sub_f32_e32 v78, v78, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[136:139], v[0:15]
	v_sub_f32_e32 v79, v79, v215
	v_sub_f32_e32 v172, v80, v215
	v_sub_f32_e32 v173, v81, v215
	v_sub_f32_e32 v174, v82, v215
	v_sub_f32_e32 v175, v83, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[132:135], v[48:63]
	v_sub_f32_e32 v176, v84, v215
	v_sub_f32_e32 v177, v85, v215
	v_sub_f32_e32 v178, v86, v215
	v_sub_f32_e32 v179, v87, v215
	v_sub_f32_e32 v180, v88, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[132:135], v[32:47]
	v_sub_f32_e32 v181, v89, v215
	v_sub_f32_e32 v182, v90, v215
	v_sub_f32_e32 v183, v91, v215
	v_sub_f32_e32 v184, v92, v215
	v_sub_f32_e32 v185, v93, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[160:163], v[132:135], v[16:31]
	v_exp_f32_e32 v186, v64
	v_exp_f32_e32 v187, v65
	v_exp_f32_e32 v196, v66
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[156:159], v[132:135], v[0:15]
	v_exp_f32_e32 v219, v67
	v_exp_f32_e32 v220, v68
	v_exp_f32_e32 v221, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[152:155], v[128:131], v[48:63]
	v_exp_f32_e32 v222, v70
	v_exp_f32_e32 v223, v71
	v_exp_f32_e32 v224, v72
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[128:131], v[32:47]
	v_exp_f32_e32 v225, v73
	v_exp_f32_e32 v226, v74
	v_exp_f32_e32 v227, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[144:147], v[128:131], v[16:31]
	v_exp_f32_e32 v228, v76
	v_exp_f32_e32 v229, v77
	v_exp_f32_e32 v230, v78
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[140:143], v[128:131], v[0:15]
	v_exp_f32_e32 v231, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s50, s20, s44
	s_addc_u32 s51, s21, s45
	s_lshl_b32 s0, s50, 1
	s_mov_b32 s1, s13
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, s1
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s31, s11
	;;#ASMSTART
	s_mov_b32 m0, s30
	;;#ASMEND
	s_mov_b32 s30, s33
	s_addk_i32 s1, 0x2000
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s58, 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v206 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v206 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v206 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v206 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v206 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[128:131], v206 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v206 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v206 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v207 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v207 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v207 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v207 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v207 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v207 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v207 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v207 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v172, v172
	v_exp_f32_e32 v173, v173
	v_exp_f32_e32 v174, v174
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v218, v186, v187
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[136:139], v[120:123], v[64:79]
	v_exp_f32_e32 v175, v175
	v_exp_f32_e32 v176, v176
	v_exp_f32_e32 v177, v177
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[116:119], v[64:79]
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v179, v179
	v_exp_f32_e32 v180, v180
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[112:115], v[64:79]
	v_exp_f32_e32 v144, v181
	v_exp_f32_e32 v145, v182
	v_exp_f32_e32 v146, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[108:111], v[64:79]
	v_exp_f32_e32 v147, v184
	v_exp_f32_e32 v181, v185
	v_exp_f32_e32 v182, v216
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[104:107], v[64:79]
	v_exp_f32_e32 v148, v217
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[100:103], v[64:79]
	v_add_f32_e32 v80, v187, v186
	v_add_f32_e32 v80, v80, v196
	v_add_f32_e32 v80, v80, v219
	v_add_f32_e32 v80, v80, v220
	v_add_f32_e32 v80, v80, v221
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v219, v196, v219
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v220, v220, v221
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[96:99], v[64:79]
	v_add_f32_e32 v80, v80, v222
	v_add_f32_e32 v80, v80, v223
	v_add_f32_e32 v80, v80, v224
	v_add_f32_e32 v80, v80, v225
	v_add_f32_e32 v136, v80, v226
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v221, v222, v223
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[92:95], v[124:127], 0
	v_add_f32_e32 v136, v136, v227
	v_add_f32_e32 v136, v136, v228
	v_add_f32_e32 v136, v136, v229
	v_add_f32_e32 v136, v136, v230
	v_add_f32_e32 v136, v136, v231
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[120:123], v[80:95]
	v_add_f32_e32 v136, v136, v172
	v_add_f32_e32 v136, v136, v173
	v_add_f32_e32 v136, v136, v174
	v_add_f32_e32 v136, v136, v175
	v_add_f32_e32 v136, v136, v176
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[116:119], v[80:95]
	v_add_f32_e32 v128, v136, v177
	v_add_f32_e32 v128, v128, v178
	v_add_f32_e32 v128, v128, v179
	v_add_f32_e32 v128, v128, v180
	v_add_f32_e32 v128, v128, v144
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v224, v225
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v226, v227
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[112:115], v[80:95]
	v_add_f32_e32 v128, v128, v145
	v_add_f32_e32 v128, v128, v146
	v_add_f32_e32 v128, v128, v147
	v_add_f32_e32 v128, v128, v181
	v_add_f32_e32 v128, v128, v182
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v228, v229
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v230, v231
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[108:111], v[80:95]
	v_add_f32_e32 v216, v128, v148
	v_mov_b32_e32 v217, v216
	s_nop 1
	v_permlane32_swap_b32_e64 v216, v217 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v172, v173
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v174, v175
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v176, v177
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v178, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v128, v180, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v129, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v130, v147, v181
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v131, v182, v148
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[140:143], v[100:103], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[168:171], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s19, s46
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s53
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s45, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s45
	;;#ASMEND
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[226:227], v200 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[230:231], v200 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[234:235], v200 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v200 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v200 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v200 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v200 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v200 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v200 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v200 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v200 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v200 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v200 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v200 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[224:225], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[228:229], v200 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[232:233], v200 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[236:237], v200 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v200 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v200 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v200 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v200 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v200 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v200 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v200 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v200 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v200 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v200 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v200 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[222:225], v[218:221], v[48:63]
	v_max_f32_e32 v196, v65, v67
	v_max3_f32 v222, v64, v66, v68
	v_max3_f32 v196, v196, v69, v71
	v_max3_f32 v222, v222, v70, v72
	v_max3_f32 v196, v196, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[226:229], v[218:221], v[32:47]
	v_max3_f32 v222, v222, v74, v76
	v_max3_f32 v196, v196, v77, v79
	v_max3_f32 v222, v222, v78, v80
	v_max3_f32 v196, v196, v81, v83
	v_max3_f32 v222, v222, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[230:233], v[218:221], v[16:31]
	v_max3_f32 v196, v196, v85, v87
	v_max3_f32 v222, v222, v86, v88
	v_max3_f32 v196, v196, v89, v91
	v_max3_f32 v222, v222, v90, v92
	v_max3_f32 v196, v196, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[234:237], v[218:221], v[0:15]
	v_max3_f32 v196, v222, v94, v196
	v_mov_b32_e32 v218, v196
	s_nop 1
	v_permlane32_swap_b32_e64 v196, v218 bound_ctrl:1
	v_max3_f32 v196, v215, v196, v218
	v_sub_f32_e32 v218, v196, v215
	v_cmp_ge_f32_e64 s[0:1], s56, v218
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	s_nop 1
	v_cndmask_b32_e64 v218, 0, 1, s[0:1]
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_cmp_ne_u32_e64 s[0:1], 0, v218
	s_cmp_eq_u64 s[0:1], exec
	s_cbranch_scc0 .LBB0_32
; %bb.26:                               ;   in Loop: Header=BB0_19 Depth=1
	v_mov_b32_e32 v196, v215
.LBB0_27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_lg_u32 s57, 0
	s_cselect_b64 s[0:1], -1, 0
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[136:139], v[48:63]
	v_cndmask_b32_e64 v184, 1.0, v208, s[0:1]
	v_fmac_f32_e32 v209, v199, v184
	v_add_f32_e32 v184, v209, v210
	v_fmac_f32_e32 v211, v184, v188
	v_add_f32_e32 v184, v211, v212
	v_pk_add_f32 v[186:187], v[90:91], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[136:139], v[32:47]
	v_fmac_f32_e32 v213, v184, v198
	v_add_f32_e32 v180, v213, v214
	v_fmac_f32_e32 v216, v180, v190
	v_add_f32_e32 v199, v216, v217
	v_sub_f32_e32 v64, v64, v196
	v_pk_add_f32 v[184:185], v[92:93], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[182:183], v[94:95], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[136:139], v[16:31]
	v_sub_f32_e32 v65, v65, v196
	v_sub_f32_e32 v66, v66, v196
	v_sub_f32_e32 v67, v67, v196
	v_sub_f32_e32 v68, v68, v196
	v_sub_f32_e32 v69, v69, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[136:139], v[0:15]
	v_sub_f32_e32 v70, v70, v196
	v_sub_f32_e32 v71, v71, v196
	v_sub_f32_e32 v72, v72, v196
	v_sub_f32_e32 v73, v73, v196
	v_sub_f32_e32 v74, v74, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[132:135], v[48:63]
	v_sub_f32_e32 v75, v75, v196
	v_sub_f32_e32 v76, v76, v196
	v_sub_f32_e32 v77, v77, v196
	v_sub_f32_e32 v78, v78, v196
	v_sub_f32_e32 v79, v79, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[132:135], v[32:47]
	v_add_f32_e64 v178, v80, -v196
	v_add_f32_e64 v179, v81, -v196
	v_add_f32_e64 v174, v82, -v196
	v_add_f32_e64 v175, v83, -v196
	v_add_f32_e64 v180, v84, -v196
	v_add_f32_e64 v181, v85, -v196
	v_pk_add_f32 v[176:177], v[86:87], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	v_pk_add_f32 v[172:173], v[88:89], v[196:197] op_sel_hi:[1,0] neg_lo:[0,1] neg_hi:[0,1]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[160:163], v[132:135], v[16:31]
	v_exp_f32_e32 v209, v64
	v_exp_f32_e32 v210, v65
	v_exp_f32_e32 v188, v66
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[156:159], v[132:135], v[0:15]
	v_exp_f32_e32 v213, v67
	v_exp_f32_e32 v190, v68
	v_exp_f32_e32 v214, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[152:155], v[128:131], v[48:63]
	v_exp_f32_e32 v211, v70
	v_exp_f32_e32 v216, v71
	v_exp_f32_e32 v212, v72
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[128:131], v[32:47]
	v_exp_f32_e32 v219, v73
	v_exp_f32_e32 v215, v74
	v_exp_f32_e32 v220, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[144:147], v[128:131], v[16:31]
	v_exp_f32_e32 v217, v76
	v_exp_f32_e32 v221, v77
	v_exp_f32_e32 v218, v78
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[140:143], v[128:131], v[0:15]
	v_exp_f32_e32 v222, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s52, s44
	s_lshl_b32 s0, s0, 1
	s_mov_b32 s1, s5
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s44, s1
	s_addk_i32 s1, 0x2000
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s44
	;;#ASMEND
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v204 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v204 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v204 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v204 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v204 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v204 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v204 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v204 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v205 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v205 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v205 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v205 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v205 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v205 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v205 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[128:131], v205 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s55, s55, 4
	s_cmp_gt_u32 s55, 26
	s_cbranch_scc1 .LBB0_33
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	s_mov_b64 s[46:47], s[48:49]
	s_mov_b64 s[44:45], s[50:51]
	s_mov_b32 s57, s58
	v_mov_b32_e32 v208, v194
	s_branch .LBB0_19
.LBB0_29:                               ;   in Loop: Header=BB0_19 Depth=1
	v_sub_f32_e32 v188, v196, v213
	v_exp_f32_e32 v188, v188
	s_nop 0
	v_pk_mul_f32 v[62:63], v[188:189], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[188:189], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[188:189], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[188:189], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[188:189], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[188:189], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[188:189], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[188:189], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[188:189], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[188:189], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[188:189], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[188:189], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[188:189], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[188:189], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[188:189], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[188:189], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[188:189], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[188:189], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[188:189], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[188:189], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[188:189], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[188:189], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[188:189], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[188:189], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[188:189], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[188:189], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[188:189], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[188:189], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[188:189], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[188:189], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[188:189], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[188:189], v[0:1] op_sel_hi:[0,1]
	v_mov_b32_e32 v194, v188
	s_branch .LBB0_21
.LBB0_30:                               ;   in Loop: Header=BB0_19 Depth=1
	v_sub_f32_e32 v194, v213, v196
	v_exp_f32_e32 v198, v194
	s_nop 0
	v_pk_mul_f32 v[62:63], v[198:199], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[198:199], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[198:199], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[198:199], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[198:199], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[198:199], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[198:199], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[198:199], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[198:199], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[198:199], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[198:199], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[198:199], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[198:199], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[198:199], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[198:199], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[198:199], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[198:199], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[198:199], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[198:199], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[198:199], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[198:199], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[198:199], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[198:199], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[198:199], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[198:199], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[198:199], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[198:199], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[198:199], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[198:199], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[198:199], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[198:199], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[198:199], v[0:1] op_sel_hi:[0,1]
	v_mov_b32_e32 v194, v198
	s_branch .LBB0_23
.LBB0_31:                               ;   in Loop: Header=BB0_19 Depth=1
	v_sub_f32_e32 v190, v196, v215
	v_exp_f32_e32 v190, v190
	s_nop 0
	v_pk_mul_f32 v[62:63], v[190:191], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[190:191], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[190:191], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[190:191], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[190:191], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[190:191], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[190:191], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[190:191], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[190:191], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[190:191], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[190:191], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[190:191], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[190:191], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[190:191], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[190:191], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[190:191], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[190:191], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[190:191], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[190:191], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[190:191], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[190:191], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[190:191], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[190:191], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[190:191], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[190:191], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[190:191], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[190:191], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[190:191], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[190:191], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[190:191], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[190:191], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[190:191], v[0:1] op_sel_hi:[0,1]
	v_mov_b32_e32 v194, v190
	s_branch .LBB0_25
.LBB0_32:                               ;   in Loop: Header=BB0_19 Depth=1
	v_sub_f32_e32 v194, v215, v196
	v_exp_f32_e32 v194, v194
	s_mov_b32 s58, 1
	v_pk_mul_f32 v[62:63], v[194:195], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[194:195], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[194:195], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[194:195], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[194:195], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[194:195], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[194:195], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[194:195], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[194:195], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[194:195], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[194:195], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[194:195], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[194:195], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[194:195], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[194:195], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[194:195], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[194:195], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[194:195], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[194:195], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[194:195], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[194:195], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[194:195], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[194:195], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[194:195], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[194:195], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[194:195], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[194:195], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[194:195], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[194:195], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[194:195], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[194:195], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[194:195], v[0:1] op_sel_hi:[0,1]
	s_branch .LBB0_27
.LBB0_33:
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v178, v178
	v_exp_f32_e32 v179, v179
	v_exp_f32_e32 v174, v174
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[120:123], v[64:79]
	v_exp_f32_e32 v168, v175
	v_exp_f32_e32 v169, v180
	v_exp_f32_e32 v170, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[92:95], v[116:119], v[64:79]
	v_exp_f32_e32 v171, v176
	v_exp_f32_e32 v175, v177
	v_exp_f32_e32 v172, v172
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[112:115], v[64:79]
	v_exp_f32_e32 v164, v173
	v_exp_f32_e32 v165, v186
	v_exp_f32_e32 v166, v187
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[108:111], v[64:79]
	v_exp_f32_e32 v167, v184
	v_exp_f32_e32 v173, v185
	v_exp_f32_e32 v176, v182
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[160:163], v[104:107], v[64:79]
	v_exp_f32_e32 v160, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[100:103], v[64:79]
	v_add_f32_e32 v84, v210, v209
	v_add_f32_e32 v84, v84, v188
	v_add_f32_e32 v84, v84, v213
	v_add_f32_e32 v84, v84, v190
	v_add_f32_e32 v84, v84, v214
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[96:99], v[64:79]
	v_add_f32_e32 v84, v84, v211
	v_add_f32_e32 v84, v84, v216
	v_add_f32_e32 v84, v84, v212
	v_add_f32_e32 v84, v84, v219
	v_add_f32_e32 v156, v84, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[80:83], v[124:127], 0
	v_add_f32_e32 v156, v156, v220
	v_add_f32_e32 v156, v156, v217
	v_add_f32_e32 v156, v156, v221
	v_add_f32_e32 v156, v156, v218
	v_add_f32_e32 v156, v156, v222
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[152:155], v[120:123], v[80:95]
	v_add_f32_e32 v152, v156, v178
	v_add_f32_e32 v152, v152, v179
	v_add_f32_e32 v152, v152, v174
	v_add_f32_e32 v152, v152, v168
	v_add_f32_e32 v152, v152, v169
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[140:143], v[116:119], v[80:95]
	v_add_f32_e32 v140, v152, v170
	v_add_f32_e32 v140, v140, v171
	v_add_f32_e32 v140, v140, v175
	v_add_f32_e32 v140, v140, v172
	v_add_f32_e32 v140, v140, v164
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[112:115], v[80:95]
	v_add_f32_e32 v140, v140, v165
	v_add_f32_e32 v140, v140, v166
	v_add_f32_e32 v140, v140, v167
	v_add_f32_e32 v140, v140, v173
	v_add_f32_e32 v140, v140, v176
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[136:139], v[108:111], v[80:95]
	v_add_f32_e32 v198, v140, v160
	v_mov_b32_e32 v208, v198
	s_nop 1
	v_permlane32_swap_b32_e64 v198, v208 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v209, v210
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v188, v213
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v190, v214
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[144:147], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v211, v216
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v140, v212, v219
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v141, v215, v220
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v142, v217, v221
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v143, v218, v222
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v144, v178, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v145, v174, v168
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[100:103], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v146, v169, v170
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v147, v171, v175
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v172, v164
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v165, v166
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v167, v173
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v176, v160
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mul_i32 s0, s18, 0xf80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s1, s25
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	s_add_i32 s1, s25, 0x2000
	buffer_load_dwordx4 v191, s[36:39], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v189, s[36:39], s0 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[210:211], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v203 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v203 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v203 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v203 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v203 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v203 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v203 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v203 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v203 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v203 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v203 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v203 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v203 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v203 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v203 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v203 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v203 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v203 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v203 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v203 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v203 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v203 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v203 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v203 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v203 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[224:225], v203 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v203 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v203 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v203 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[210:213], v[136:139], v[48:63]
	v_max_f32_e32 v209, v65, v67
	v_max3_f32 v210, v64, v66, v68
	v_max3_f32 v209, v209, v69, v71
	v_max3_f32 v210, v210, v70, v72
	v_max3_f32 v209, v209, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[214:217], v[140:143], v[48:63]
	v_max3_f32 v210, v210, v74, v76
	v_max3_f32 v209, v209, v77, v79
	v_max3_f32 v210, v210, v78, v80
	v_max3_f32 v209, v209, v81, v83
	v_max3_f32 v210, v210, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[218:221], v[144:147], v[48:63]
	v_max3_f32 v209, v209, v85, v87
	v_max3_f32 v210, v210, v86, v88
	v_max3_f32 v209, v209, v89, v91
	v_max3_f32 v210, v210, v90, v92
	v_max3_f32 v209, v209, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[222:225], v[132:135], v[48:63]
	v_max3_f32 v209, v210, v94, v209
	v_mov_b32_e32 v210, v209
	s_nop 1
	v_permlane32_swap_b32_e64 v209, v210 bound_ctrl:1
	v_max3_f32 v209, v196, v209, v210
	v_sub_f32_e32 v196, v196, v209
	v_sub_f32_e32 v221, v95, v209
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[136:139], v[32:47]
	v_sub_f32_e32 v64, v64, v209
	v_sub_f32_e32 v65, v65, v209
	v_sub_f32_e32 v66, v66, v209
	v_sub_f32_e32 v67, v67, v209
	v_sub_f32_e32 v68, v68, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[140:143], v[32:47]
	v_sub_f32_e32 v69, v69, v209
	v_sub_f32_e32 v70, v70, v209
	v_sub_f32_e32 v71, v71, v209
	v_sub_f32_e32 v72, v72, v209
	v_sub_f32_e32 v73, v73, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[184:187], v[144:147], v[32:47]
	v_sub_f32_e32 v74, v74, v209
	v_sub_f32_e32 v75, v75, v209
	v_sub_f32_e32 v76, v76, v209
	v_sub_f32_e32 v77, v77, v209
	v_sub_f32_e32 v78, v78, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[188:191], v[132:135], v[32:47]
	v_sub_f32_e32 v79, v79, v209
	v_sub_f32_e32 v176, v80, v209
	v_sub_f32_e32 v177, v81, v209
	v_sub_f32_e32 v178, v82, v209
	v_sub_f32_e32 v179, v83, v209
	v_sub_f32_e32 v190, v94, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[156:159], v[136:139], v[16:31]
	v_sub_f32_e32 v180, v84, v209
	v_sub_f32_e32 v181, v85, v209
	v_sub_f32_e32 v182, v86, v209
	v_sub_f32_e32 v183, v87, v209
	v_sub_f32_e32 v184, v88, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[164:167], v[140:143], v[16:31]
	v_sub_f32_e32 v185, v89, v209
	v_sub_f32_e32 v186, v90, v209
	v_sub_f32_e32 v187, v91, v209
	v_sub_f32_e32 v188, v92, v209
	v_sub_f32_e32 v189, v93, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[168:171], v[144:147], v[16:31]
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v191, v64
	v_exp_f32_e32 v210, v65
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[172:175], v[132:135], v[16:31]
	v_exp_f32_e32 v172, v66
	v_exp_f32_e32 v173, v67
	v_exp_f32_e32 v174, v68
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[136:139], v[0:15]
	v_exp_f32_e32 v175, v69
	v_exp_f32_e32 v211, v70
	v_exp_f32_e32 v212, v71
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[148:151], v[140:143], v[0:15]
	v_exp_f32_e32 v213, v72
	v_exp_f32_e32 v214, v73
	v_exp_f32_e32 v215, v74
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[152:155], v[144:147], v[0:15]
	v_exp_f32_e32 v216, v75
	v_exp_f32_e32 v217, v76
	v_exp_f32_e32 v218, v77
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[132:135], v[0:15]
	v_exp_f32_e32 v219, v78
	v_exp_f32_e32 v220, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[62:63], v[196:197], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[196:197], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[196:197], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[196:197], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[196:197], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[196:197], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[196:197], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[196:197], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[196:197], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[196:197], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[196:197], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[196:197], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[196:197], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[196:197], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[196:197], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[196:197], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[196:197], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[196:197], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[196:197], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[196:197], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[196:197], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[196:197], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[196:197], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[196:197], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[196:197], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[196:197], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[196:197], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[196:197], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[196:197], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[196:197], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[196:197], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[196:197], v[0:1] op_sel_hi:[0,1]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mul_i32 s0, s16, 0xf00
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s1, s13
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s30, s33
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	s_mov_b32 s31, s11
	s_add_i32 s1, s13, 0x2000
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v206 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v206 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v206 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v206 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v206 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[128:131], v206 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v206 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v206 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v207 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v207 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v207 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v207 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v207 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v207 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v207 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v207 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v176, v176
	v_exp_f32_e32 v177, v177
	v_exp_f32_e32 v178, v178
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[140:143], v[120:123], v[64:79]
	v_exp_f32_e32 v141, v179
	v_exp_f32_e32 v142, v180
	v_exp_f32_e32 v143, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[116:119], v[64:79]
	v_exp_f32_e32 v179, v182
	v_exp_f32_e32 v180, v183
	v_exp_f32_e32 v181, v184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[112:115], v[64:79]
	v_exp_f32_e32 v144, v185
	v_exp_f32_e32 v145, v186
	v_exp_f32_e32 v146, v187
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[108:111], v[64:79]
	v_exp_f32_e32 v147, v188
	v_exp_f32_e32 v182, v189
	v_exp_f32_e32 v183, v190
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[104:107], v[64:79]
	v_exp_f32_e32 v148, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[100:103], v[64:79]
	v_add_f32_e32 v80, v210, v191
	v_add_f32_e32 v80, v80, v172
	v_add_f32_e32 v80, v80, v173
	v_add_f32_e32 v80, v80, v174
	v_add_f32_e32 v80, v80, v175
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[96:99], v[64:79]
	v_add_f32_e32 v80, v80, v211
	v_add_f32_e32 v80, v80, v212
	v_add_f32_e32 v80, v80, v213
	v_add_f32_e32 v80, v80, v214
	v_add_f32_e32 v140, v80, v215
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[92:95], v[124:127], 0
	v_add_f32_e32 v140, v140, v216
	v_add_f32_e32 v140, v140, v217
	v_add_f32_e32 v140, v140, v218
	v_add_f32_e32 v140, v140, v219
	v_add_f32_e32 v140, v140, v220
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[120:123], v[80:95]
	v_add_f32_e32 v140, v140, v176
	v_add_f32_e32 v140, v140, v177
	v_add_f32_e32 v140, v140, v178
	v_add_f32_e32 v140, v140, v141
	v_add_f32_e32 v140, v140, v142
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[116:119], v[80:95]
	v_add_f32_e32 v128, v140, v143
	v_add_f32_e32 v128, v128, v179
	v_add_f32_e32 v128, v128, v180
	v_add_f32_e32 v128, v128, v181
	v_add_f32_e32 v128, v128, v144
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[112:115], v[80:95]
	v_add_f32_e32 v128, v128, v145
	v_add_f32_e32 v128, v128, v146
	v_add_f32_e32 v128, v128, v147
	v_add_f32_e32 v128, v128, v182
	v_add_f32_e32 v128, v128, v183
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[108:111], v[80:95]
	v_add_f32_e32 v206, v128, v148
	v_mov_b32_e32 v207, v206
	s_nop 1
	v_permlane32_swap_b32_e64 v206, v207 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v128, v191, v210
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v129, v172, v173
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v130, v174, v175
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v131, v211, v212
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v213, v214
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v215, v216
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v217, v218
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v219, v220
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v140, v176, v177
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v141, v178, v141
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[136:139], v[100:103], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v142, v142, v143
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v143, v179, v180
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v181, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v147, v182
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v183, v148
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[168:171], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[210:211], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v200 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v200 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v200 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v200 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v200 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v200 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v200 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v200 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v200 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v200 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v200 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v200 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v200 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v200 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v200 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v200 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v200 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v200 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v200 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v200 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v200 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v200 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v200 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v200 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[224:225], v200 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v200 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v200 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v200 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[210:213], v[128:131], v[48:63]
	v_max_f32_e32 v210, v65, v67
	v_max3_f32 v211, v64, v66, v68
	v_max3_f32 v210, v210, v69, v71
	v_max3_f32 v211, v211, v70, v72
	v_max3_f32 v210, v210, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[214:217], v[132:135], v[48:63]
	v_max3_f32 v211, v211, v74, v76
	v_max3_f32 v210, v210, v77, v79
	v_max3_f32 v211, v211, v78, v80
	v_max3_f32 v210, v210, v81, v83
	v_max3_f32 v211, v211, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[218:221], v[140:143], v[48:63]
	v_max3_f32 v210, v210, v85, v87
	v_max3_f32 v211, v211, v86, v88
	v_max3_f32 v210, v210, v89, v91
	v_max3_f32 v211, v211, v90, v92
	v_max3_f32 v210, v210, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[222:225], v[136:139], v[48:63]
	v_max3_f32 v210, v211, v94, v210
	v_mov_b32_e32 v211, v210
	s_nop 1
	v_permlane32_swap_b32_e64 v210, v211 bound_ctrl:1
	v_max3_f32 v210, v209, v210, v211
	v_sub_f32_e32 v209, v209, v210
	v_sub_f32_e32 v222, v95, v210
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[128:131], v[32:47]
	v_sub_f32_e32 v64, v64, v210
	v_sub_f32_e32 v65, v65, v210
	v_sub_f32_e32 v66, v66, v210
	v_sub_f32_e32 v67, v67, v210
	v_sub_f32_e32 v68, v68, v210
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[132:135], v[32:47]
	v_sub_f32_e32 v69, v69, v210
	v_sub_f32_e32 v70, v70, v210
	v_sub_f32_e32 v71, v71, v210
	v_sub_f32_e32 v72, v72, v210
	v_sub_f32_e32 v73, v73, v210
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[184:187], v[140:143], v[32:47]
	v_sub_f32_e32 v74, v74, v210
	v_sub_f32_e32 v75, v75, v210
	v_sub_f32_e32 v76, v76, v210
	v_sub_f32_e32 v77, v77, v210
	v_sub_f32_e32 v78, v78, v210
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[188:191], v[136:139], v[32:47]
	v_sub_f32_e32 v79, v79, v210
	v_sub_f32_e32 v176, v80, v210
	v_sub_f32_e32 v177, v81, v210
	v_sub_f32_e32 v178, v82, v210
	v_sub_f32_e32 v179, v83, v210
	v_sub_f32_e32 v189, v94, v210
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[156:159], v[128:131], v[16:31]
	v_sub_f32_e32 v180, v84, v210
	v_sub_f32_e32 v181, v85, v210
	v_sub_f32_e32 v182, v86, v210
	v_sub_f32_e32 v183, v87, v210
	v_sub_f32_e32 v184, v88, v210
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[164:167], v[132:135], v[16:31]
	v_sub_f32_e32 v185, v89, v210
	v_sub_f32_e32 v165, v90, v210
	v_sub_f32_e32 v186, v91, v210
	v_sub_f32_e32 v187, v92, v210
	v_sub_f32_e32 v188, v93, v210
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[168:171], v[140:143], v[16:31]
	v_exp_f32_e32 v164, v209
	v_exp_f32_e32 v190, v64
	v_exp_f32_e32 v191, v65
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[172:175], v[136:139], v[16:31]
	v_exp_f32_e32 v174, v66
	v_exp_f32_e32 v175, v67
	v_exp_f32_e32 v209, v68
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[128:131], v[0:15]
	v_exp_f32_e32 v211, v69
	v_exp_f32_e32 v212, v70
	v_exp_f32_e32 v213, v71
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[148:151], v[132:135], v[0:15]
	v_exp_f32_e32 v214, v72
	v_exp_f32_e32 v215, v73
	v_exp_f32_e32 v216, v74
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[152:155], v[140:143], v[0:15]
	v_exp_f32_e32 v217, v75
	v_exp_f32_e32 v218, v76
	v_exp_f32_e32 v219, v77
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[136:139], v[0:15]
	v_exp_f32_e32 v220, v78
	v_exp_f32_e32 v221, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[62:63], v[164:165], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[164:165], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[164:165], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[164:165], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[164:165], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[164:165], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[164:165], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[164:165], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[164:165], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[164:165], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[164:165], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[164:165], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[164:165], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[164:165], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[164:165], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[164:165], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[164:165], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[164:165], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[164:165], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[164:165], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[164:165], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[164:165], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[164:165], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[164:165], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[164:165], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[164:165], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[164:165], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[164:165], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[164:165], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[164:165], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[164:165], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[164:165], v[0:1] op_sel_hi:[0,1]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mul_i32 s0, s16, 0xf80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s1, s5
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	s_add_i32 s1, s5, 0x2000
	buffer_load_dwordx4 v202, s[28:31], s0 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	s_mov_b32 m0, s1
	;;#ASMEND
	buffer_load_dwordx4 v201, s[28:31], s0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v204 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v204 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v204 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v204 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v204 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[128:131], v204 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[132:135], v204 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[136:139], v204 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[140:143], v205 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v205 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v205 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v205 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v205 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v205 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v205 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v205 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[124:127], 0
	v_exp_f32_e32 v176, v176
	v_exp_f32_e32 v177, v177
	v_exp_f32_e32 v178, v178
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[140:143], v[120:123], v[64:79]
	v_exp_f32_e32 v140, v179
	v_exp_f32_e32 v141, v180
	v_exp_f32_e32 v142, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[116:119], v[64:79]
	v_exp_f32_e32 v143, v182
	v_exp_f32_e32 v179, v183
	v_exp_f32_e32 v180, v184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[112:115], v[64:79]
	v_exp_f32_e32 v144, v185
	v_exp_f32_e32 v145, v165
	v_exp_f32_e32 v146, v186
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[108:111], v[64:79]
	v_exp_f32_e32 v147, v187
	v_exp_f32_e32 v165, v188
	v_exp_f32_e32 v181, v189
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[104:107], v[64:79]
	v_exp_f32_e32 v148, v222
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[100:103], v[64:79]
	v_add_f32_e32 v80, v191, v190
	v_add_f32_e32 v80, v80, v174
	v_add_f32_e32 v80, v80, v175
	v_add_f32_e32 v80, v80, v209
	v_add_f32_e32 v80, v80, v211
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[96:99], v[64:79]
	v_add_f32_e32 v80, v80, v212
	v_add_f32_e32 v80, v80, v213
	v_add_f32_e32 v80, v80, v214
	v_add_f32_e32 v80, v80, v215
	v_add_f32_e32 v149, v80, v216
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[92:95], v[124:127], 0
	v_add_f32_e32 v124, v149, v217
	v_add_f32_e32 v124, v124, v218
	v_add_f32_e32 v124, v124, v219
	v_add_f32_e32 v124, v124, v220
	v_add_f32_e32 v124, v124, v221
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[120:123], v[80:95]
	v_add_f32_e32 v120, v124, v176
	v_add_f32_e32 v120, v120, v177
	v_add_f32_e32 v120, v120, v178
	v_add_f32_e32 v120, v120, v140
	v_add_f32_e32 v120, v120, v141
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[128:131], v[116:119], v[80:95]
	v_add_f32_e32 v116, v120, v142
	v_add_f32_e32 v116, v116, v143
	v_add_f32_e32 v116, v116, v179
	v_add_f32_e32 v116, v116, v180
	v_add_f32_e32 v116, v116, v144
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[112:115], v[80:95]
	v_add_f32_e32 v112, v116, v145
	v_add_f32_e32 v112, v112, v146
	v_add_f32_e32 v112, v112, v147
	v_add_f32_e32 v112, v112, v165
	v_add_f32_e32 v112, v112, v181
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[132:135], v[108:111], v[80:95]
	v_add_f32_e32 v161, v112, v148
	v_mov_b32_e32 v162, v161
	s_nop 1
	v_permlane32_swap_b32_e64 v161, v162 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v190, v191
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v174, v175
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v209, v211
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[166:169], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v212, v213
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v214, v215
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v216, v217
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v218, v219
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v220, v221
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v176, v177
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v178, v140
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[136:139], v[100:103], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v114, v141, v142
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v115, v143, v179
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v180, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v147, v165
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v181, v148
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[170:173], v[96:99], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v203 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v203 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v203 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v203 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v203 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v203 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v203 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v203 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v203 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v203 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v203 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v203 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v203 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v203 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v203 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v203 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v203 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v203 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v203 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v203 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v203 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v203 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v203 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v203 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v203 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v203 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v203 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v203 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v203 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[166:169], v[108:111], v[48:63]
	v_max_f32_e32 v160, v65, v67
	v_max3_f32 v163, v64, v66, v68
	v_max3_f32 v160, v160, v69, v71
	v_max3_f32 v163, v163, v70, v72
	v_max3_f32 v160, v160, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[170:173], v[104:107], v[48:63]
	v_max3_f32 v163, v163, v74, v76
	v_max3_f32 v160, v160, v77, v79
	s_nop 0
	v_max3_f32 v163, v163, v78, v80
	v_max3_f32 v160, v160, v81, v83
	v_max3_f32 v163, v163, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[174:177], v[112:115], v[48:63]
	v_max3_f32 v160, v160, v85, v87
	v_max3_f32 v163, v163, v86, v88
	v_max3_f32 v160, v160, v89, v91
	v_max3_f32 v163, v163, v90, v92
	v_max3_f32 v160, v160, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[178:181], v[100:103], v[48:63]
	v_max3_f32 v160, v163, v94, v160
	v_mov_b32_e32 v163, v160
	s_nop 1
	v_permlane32_swap_b32_e64 v160, v163 bound_ctrl:1
	v_max3_f32 v160, v210, v160, v163
	v_sub_f32_e32 v163, v210, v160
	v_sub_f32_e32 v94, v94, v160
	v_mfma_f32_32x32x16_bf16 v[32:47], v[144:147], v[108:111], v[32:47]
	v_sub_f32_e32 v64, v64, v160
	v_sub_f32_e32 v65, v65, v160
	v_sub_f32_e32 v66, v66, v160
	v_sub_f32_e32 v67, v67, v160
	v_sub_f32_e32 v68, v68, v160
	v_sub_f32_e32 v95, v95, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[104:107], v[32:47]
	v_sub_f32_e32 v69, v69, v160
	v_sub_f32_e32 v70, v70, v160
	v_sub_f32_e32 v71, v71, v160
	v_sub_f32_e32 v72, v72, v160
	v_sub_f32_e32 v73, v73, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[112:115], v[32:47]
	v_sub_f32_e32 v74, v74, v160
	v_sub_f32_e32 v75, v75, v160
	v_sub_f32_e32 v76, v76, v160
	v_sub_f32_e32 v77, v77, v160
	v_sub_f32_e32 v78, v78, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[156:159], v[100:103], v[32:47]
	v_sub_f32_e32 v79, v79, v160
	v_sub_f32_e32 v144, v80, v160
	v_sub_f32_e32 v81, v81, v160
	v_sub_f32_e32 v82, v82, v160
	v_sub_f32_e32 v83, v83, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[124:127], v[108:111], v[16:31]
	v_sub_f32_e32 v84, v84, v160
	v_sub_f32_e32 v85, v85, v160
	v_sub_f32_e32 v86, v86, v160
	v_sub_f32_e32 v87, v87, v160
	v_sub_f32_e32 v88, v88, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[132:135], v[104:107], v[16:31]
	v_sub_f32_e32 v89, v89, v160
	v_sub_f32_e32 v90, v90, v160
	v_sub_f32_e32 v91, v91, v160
	v_sub_f32_e32 v92, v92, v160
	v_sub_f32_e32 v93, v93, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[136:139], v[112:115], v[16:31]
	v_exp_f32_e32 v80, v163
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[140:143], v[100:103], v[16:31]
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[108:111], v[0:15]
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[116:119], v[104:107], v[0:15]
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v74, v74
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[120:123], v[112:115], v[0:15]
	v_exp_f32_e32 v75, v75
	v_exp_f32_e32 v76, v76
	v_exp_f32_e32 v77, v77
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[100:103], v[0:15]
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	; sched_barrier mask(0x00000000)
	v_exp_f32_e32 v97, v81
	v_add_f32_e32 v81, v65, v64
	v_add_f32_e32 v81, v81, v66
	v_add_f32_e32 v81, v81, v67
	v_add_f32_e32 v81, v81, v68
	v_add_f32_e32 v81, v81, v69
	v_add_f32_e32 v81, v81, v70
	v_add_f32_e32 v81, v81, v71
	v_add_f32_e32 v81, v81, v72
	v_add_f32_e32 v81, v81, v73
	v_add_f32_e32 v81, v81, v74
	v_add_f32_e32 v81, v81, v75
	v_exp_f32_e32 v96, v144
	v_add_f32_e32 v81, v81, v76
	v_add_f32_e32 v81, v81, v77
	v_exp_f32_e32 v98, v82
	v_add_f32_e32 v81, v81, v78
	v_exp_f32_e32 v83, v83
	v_add_f32_e32 v81, v81, v79
	v_exp_f32_e32 v84, v84
	v_add_f32_e32 v81, v81, v96
	v_exp_f32_e32 v85, v85
	v_add_f32_e32 v81, v81, v97
	v_exp_f32_e32 v86, v86
	v_add_f32_e32 v81, v81, v98
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v81, v81, v83
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v81, v81, v84
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v81, v81, v85
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v81, v81, v86
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v81, v81, v87
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v81, v81, v88
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v81, v81, v89
	v_exp_f32_e32 v94, v94
	v_add_f32_e32 v81, v81, v90
	v_exp_f32_e32 v95, v95
	v_add_f32_e32 v81, v81, v91
	v_add_f32_e32 v81, v81, v92
	v_add_f32_e32 v81, v81, v93
	v_add_f32_e32 v81, v81, v94
	v_add_f32_e32 v81, v81, v95
	v_mov_b32_e32 v82, v81
	s_nop 1
	v_permlane32_swap_b32_e64 v81, v82 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v64, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v65, v66, v67
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v68, v69
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v72, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v74, v75
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v76, v77
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v78, v79
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v96, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v98, v83
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v84, v85
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v86, v87
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v88, v89
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v90, v91
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v92, v93
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v94, v95
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[62:63], v[80:81], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[80:81], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[80:81], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[80:81], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[80:81], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[80:81], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[80:81], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[80:81], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[80:81], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[80:81], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[80:81], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[80:81], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[80:81], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[80:81], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[80:81], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[80:81], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[80:81], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[80:81], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[80:81], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[80:81], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[80:81], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[80:81], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[80:81], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[80:81], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[80:81], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[80:81], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[80:81], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[80:81], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[80:81], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[80:81], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[80:81], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[80:81], v[0:1] op_sel_hi:[0,1]
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[84:85], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[88:89], v200 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[92:93], v200 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v200 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[100:101], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[104:105], v200 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[108:109], v200 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v200 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v200 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v200 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v200 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v200 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v200 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v200 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v200 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v200 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[86:87], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[90:91], v200 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v200 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v200 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[102:103], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[106:107], v200 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[110:111], v200 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v200 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v200 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v200 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v200 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v200 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v200 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v200 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v200 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v200 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[84:87], v[64:67], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[88:91], v[64:67], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[92:95], v[64:67], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[64:67], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[100:103], v[68:71], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[104:107], v[68:71], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[108:111], v[68:71], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[112:115], v[68:71], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[116:119], v[72:75], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[120:123], v[72:75], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[124:127], v[72:75], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[72:75], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[132:135], v[76:79], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[136:139], v[76:79], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[140:143], v[76:79], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[76:79], v[0:15]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_35
; %bb.34:
	s_barrier
.LBB0_35:
	s_or_b64 exec, exec, s[0:1]
	v_fmac_f32_e32 v198, v199, v194
	v_add_f32_e32 v64, v198, v208
	v_fmac_f32_e32 v206, v196, v64
	v_add_f32_e32 v64, v206, v207
	v_fmac_f32_e32 v161, v164, v64
	v_add_f32_e32 v64, v161, v162
	v_fmac_f32_e32 v81, v80, v64
	v_add_f32_e32 v64, v81, v82
	v_rcp_f32_e32 v82, v64
	v_mov_b32_e32 v80, s8
	v_mov_b32_e32 v81, s9
	v_mul_f32_e32 v69, v0, v82
	v_mov_b32_e32 v0, s4
	v_mul_f32_e32 v70, v1, v82
	v_mad_i64_i32 v[0:1], s[0:1], s12, v0, v[192:193]
	v_mul_f32_e32 v71, v2, v82
	v_mul_f32_e32 v72, v3, v82
	s_bfe_i64 s[0:1], s[14:15], 0x200000
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mul_f32_e32 v65, v4, v82
	v_mul_f32_e32 v66, v5, v82
	v_mul_f32_e32 v4, v48, v82
	v_mul_f32_e32 v5, v49, v82
	v_mul_f32_e32 v48, v52, v82
	v_mul_f32_e32 v49, v53, v82
	v_mul_lo_u32 v52, v1, s14
	v_mul_lo_u32 v53, v0, s1
	v_mad_u64_u32 v[0:1], s[0:1], v0, s14, v[2:3]
	v_add3_u32 v1, v52, v1, v53
	s_bfe_i64 s[0:1], s[40:41], 0x200000
	v_mul_lo_u32 v2, v1, s40
	v_mul_lo_u32 v3, v0, s1
	v_mad_u64_u32 v[0:1], s[0:1], v0, s40, 0
	s_mul_i32 s0, s14, s40
	s_mul_i32 s1, s10, s12
	s_mul_i32 s1, s0, s1
	v_mul_f32_e32 v67, v6, v82
	v_mul_f32_e32 v6, v12, v82
	v_mul_f32_e32 v12, v50, v82
	v_add3_u32 v1, v1, v3, v2
	s_lshl_b32 s1, s1, 1
	v_mul_lo_u32 v52, v195, s0
	v_mul_f32_e32 v15, v15, v82
	v_mul_f32_e32 v14, v14, v82
	v_mul_f32_e32 v68, v7, v82
	v_mul_f32_e32 v8, v8, v82
	v_mul_f32_e32 v9, v9, v82
	v_mul_f32_e32 v10, v10, v82
	v_mul_f32_e32 v11, v11, v82
	v_mul_f32_e32 v7, v13, v82
	v_mul_f32_e32 v13, v31, v82
	v_mul_f32_e32 v30, v30, v82
	v_mul_f32_e32 v74, v16, v82
	v_mul_f32_e32 v75, v17, v82
	v_mul_f32_e32 v76, v18, v82
	v_mul_f32_e32 v77, v19, v82
	v_mul_f32_e32 v31, v20, v82
	v_mul_f32_e32 v73, v21, v82
	v_mul_f32_e32 v22, v22, v82
	v_mul_f32_e32 v23, v23, v82
	v_mul_f32_e32 v18, v24, v82
	v_mul_f32_e32 v19, v25, v82
	v_mul_f32_e32 v20, v26, v82
	v_mul_f32_e32 v21, v27, v82
	v_mul_f32_e32 v16, v28, v82
	v_mul_f32_e32 v17, v29, v82
	v_mul_f32_e32 v24, v47, v82
	v_mul_f32_e32 v25, v46, v82
	v_mul_f32_e32 v46, v32, v82
	v_mul_f32_e32 v47, v33, v82
	v_mul_f32_e32 v78, v34, v82
	v_mul_f32_e32 v79, v35, v82
	v_mul_f32_e32 v34, v36, v82
	v_mul_f32_e32 v35, v37, v82
	v_mul_f32_e32 v36, v38, v82
	v_mul_f32_e32 v37, v39, v82
	v_mul_f32_e32 v28, v40, v82
	v_mul_f32_e32 v29, v41, v82
	v_mul_f32_e32 v32, v42, v82
	v_mul_f32_e32 v33, v43, v82
	v_mul_f32_e32 v26, v44, v82
	v_mul_f32_e32 v27, v45, v82
	v_mul_f32_e32 v38, v63, v82
	v_mul_f32_e32 v39, v62, v82
	v_mul_f32_e32 v62, v51, v82
	v_mul_f32_e32 v50, v54, v82
	v_mul_f32_e32 v51, v55, v82
	v_mul_f32_e32 v42, v56, v82
	v_mul_f32_e32 v43, v57, v82
	v_mul_f32_e32 v44, v58, v82
	v_mul_f32_e32 v45, v59, v82
	v_mul_f32_e32 v40, v60, v82
	v_mul_f32_e32 v41, v61, v82
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[80:81]
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v3, 0x20000
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v12, v62
	;;#ASMEND
	v_add_lshl_u32 v12, v52, v197, 1
	s_mov_b64 s[12:13], exec
.LBB0_36:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_36
; %bb.37:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v48, v49
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v50, v51
	;;#ASMEND
.LBB0_38:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:16
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_38
; %bb.39:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v42, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v44, v45
	;;#ASMEND
.LBB0_40:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:32
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_40
; %bb.41:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v40, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v39, v38
	;;#ASMEND
.LBB0_42:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:48
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_42
; %bb.43:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v46, v47
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v78, v79
	;;#ASMEND
.LBB0_44:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:64
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_44
; %bb.45:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v34, v35
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v36, v37
	;;#ASMEND
.LBB0_46:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:80
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_46
; %bb.47:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v28, v29
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v32, v33
	;;#ASMEND
.LBB0_48:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:96
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_48
; %bb.49:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v25, v24
	;;#ASMEND
.LBB0_50:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:112
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_50
; %bb.51:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v74, v75
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v76, v77
	;;#ASMEND
.LBB0_52:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:128
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_52
; %bb.53:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v31, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v22, v23
	;;#ASMEND
.LBB0_54:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:144
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_54
; %bb.55:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v20, v21
	;;#ASMEND
.LBB0_56:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:160
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_56
; %bb.57:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v16, v17
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v30, v13
	;;#ASMEND
.LBB0_58:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:176
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_58
; %bb.59:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v69, v70
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v71, v72
	;;#ASMEND
.LBB0_60:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:192
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_60
; %bb.61:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v65, v66
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v67, v68
	;;#ASMEND
.LBB0_62:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:208
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_62
; %bb.63:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v10, v11
	;;#ASMEND
.LBB0_64:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:224
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_64
; %bb.65:
	s_mov_b64 exec, s[12:13]
	s_mov_b64 s[12:13], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v14, v15
	;;#ASMEND
.LBB0_66:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:240
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5
                                        ; implicit-def: $vgpr12
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_66
; %bb.67:
	s_mov_b64 exec, s[12:13]
	s_mov_b32 s0, 0x800000
	s_mul_i32 s1, s24, s4
	v_cmp_gt_f32_e32 vcc, s0, v64
	s_mul_hi_i32 s0, s24, s4
	s_add_u32 s2, s1, s2
	s_addc_u32 s3, s0, 0
	s_bfe_i64 s[0:1], s[26:27], 0x200000
	s_mul_i32 s0, s2, s1
	s_mul_hi_u32 s1, s2, s26
	s_add_i32 s0, s1, s0
	s_mul_i32 s3, s3, s26
	s_add_i32 s3, s0, s3
	s_mul_i32 s2, s2, s26
	s_bfe_i64 s[0:1], s[34:35], 0x200000
	v_cndmask_b32_e64 v0, 0, 32, vcc
	s_mul_i32 s0, s2, s1
	s_mul_hi_u32 s1, s2, s34
	v_ldexp_f32 v0, v64, v0
	s_add_i32 s0, s1, s0
	s_mul_i32 s3, s3, s34
	v_log_f32_e32 v0, v0
	s_add_i32 s1, s0, s3
	s_mul_i32 s0, s2, s34
	s_lshl_b64 s[0:1], s[0:1], 2
	v_mov_b32_e32 v1, 0xc1b17218
	s_add_u32 s0, s6, s0
	v_cndmask_b32_e32 v4, 0, v1, vcc
	s_addc_u32 s1, s7, s1
	v_fmac_f32_e32 v4, 0x3f317218, v0
	v_lshl_add_u64 v[0:1], v[192:193], 2, s[0:1]
	v_lshlrev_b32_e32 v2, 2, v195
	v_mov_b32_e32 v3, 0
	v_fmac_f32_e32 v4, 0x3f317218, v160
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	global_store_dword v[0:1], v4, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10attend_kerILi128EEv12attn_globalsIXT_EE
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 248
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
		.amdhsa_next_free_vgpr 238
		.amdhsa_next_free_sgpr 60
		.amdhsa_accum_offset 240
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
	.section	.text._Z10attend_kerILi128EEv12attn_globalsIXT_EE,"axG",@progbits,_Z10attend_kerILi128EEv12attn_globalsIXT_EE,comdat
.Lfunc_end0:
	.size	_Z10attend_kerILi128EEv12attn_globalsIXT_EE, .Lfunc_end0-_Z10attend_kerILi128EEv12attn_globalsIXT_EE
                                        ; -- End function
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.num_vgpr, 238
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.num_agpr, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.numbered_sgpr, 60
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.private_seg_size, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.uses_vcc, 1
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.uses_flat_scratch, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.has_dyn_sized_stack, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.has_recursion, 0
	.set _Z10attend_kerILi128EEv12attn_globalsIXT_EE.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 24320
; TotalNumSgprs: 66
; NumVgprs: 238
; NumAgprs: 0
; TotalNumVgprs: 238
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 8
; VGPRBlocks: 29
; NumSGPRsForWavesPerEU: 66
; NumVGPRsForWavesPerEU: 238
; AccumOffset: 240
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 59
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_baef1f1c0b8c08c7,@object ; @__hip_cuid_baef1f1c0b8c08c7
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_baef1f1c0b8c08c7
__hip_cuid_baef1f1c0b8c08c7:
	.byte	0                               ; 0x0
	.size	__hip_cuid_baef1f1c0b8c08c7, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_baef1f1c0b8c08c7
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           248
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 248
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z10attend_kerILi128EEv12attn_globalsIXT_EE
    .private_segment_fixed_size: 0
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         _Z10attend_kerILi128EEv12attn_globalsIXT_EE.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     238
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
