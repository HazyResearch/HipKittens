	.text
	.file	"unit_tests.cu"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	xorl	%eax, %eax
	cmpl	$2, %edi
	setge	%al
	movl	%eax, should_write_outputs(%rip)
	xorps	%xmm0, %xmm0
	movaps	%xmm0, 16(%rsp)
	movq	$0, 32(%rsp)
.Ltmp0:
	leaq	16(%rsp), %rdi
	callq	_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE
.Ltmp1:
# %bb.1:
.Ltmp2:
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$80, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp3:
# %bb.2:                                # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cout+240(%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB0_3
# %bb.5:                                # %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit.i.i
	cmpb	$0, 56(%rbx)
	je	.LBB0_7
# %bb.6:
	movzbl	67(%rbx), %eax
	jmp	.LBB0_9
.LBB0_7:
.Ltmp4:
	movq	%rbx, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
.Ltmp5:
# %bb.8:                                # %.noexc28
	movq	(%rbx), %rax
.Ltmp6:
	movq	%rbx, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp7:
.LBB0_9:                                # %_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc.exit.i
.Ltmp8:
	movsbl	%al, %esi
	movl	$_ZSt4cout, %edi
	callq	_ZNSo3putEc
.Ltmp9:
# %bb.10:                               # %.noexc30
.Ltmp10:
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
.Ltmp11:
# %bb.11:                               # %_ZNSolsEPFRSoS_E.exit
.Ltmp12:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.1, %esi
	movl	$14, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp13:
# %bb.12:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit20
	movq	16(%rsp), %r13
	movq	24(%rsp), %rax
	xorl	%r14d, %r14d
	movl	$0, %ebp
	cmpq	%rax, %r13
	je	.LBB0_28
# %bb.13:                               # %.lr.ph.preheader
	xorl	%r14d, %r14d
	movl	$0, 8(%rsp)                     # 4-byte Folded Spill
	xorl	%ebp, %ebp
	jmp	.LBB0_14
	.p2align	4, 0x90
.LBB0_25:                               # %.noexc40._ZNSolsEPFRSoS_E.exit21_crit_edge
                                        #   in Loop: Header=BB0_14 Depth=1
	incl	8(%rsp)                         # 4-byte Folded Spill
	movq	24(%rsp), %rax
.LBB0_26:                               # %_ZNSolsEPFRSoS_E.exit21
                                        #   in Loop: Header=BB0_14 Depth=1
	movb	%bpl, %bl
	addl	%ebx, %r14d
	addq	$40, %r13
	cmpq	%rax, %r13
	movl	12(%rsp), %ebp                  # 4-byte Reload
	je	.LBB0_27
.LBB0_14:                               # %.lr.ph
                                        # =>This Inner Loop Header: Depth=1
	movl	32(%r13), %ecx
	cmpl	$1, %ecx
	adcl	$0, %ebp
	movl	%ebp, 12(%rsp)                  # 4-byte Spill
	xorl	%ebx, %ebx
	cmpl	$2, %ecx
	sete	%bpl
	cmpl	$1, %ecx
	jne	.LBB0_26
# %bb.15:                               #   in Loop: Header=BB0_14 Depth=1
	movq	(%r13), %rsi
	movq	8(%r13), %rdx
.Ltmp14:
	movl	$_ZSt4cout, %edi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp15:
# %bb.16:                               # %_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE.exit
                                        #   in Loop: Header=BB0_14 Depth=1
	movq	%rax, %r15
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r15,%rax), %r12
	testq	%r12, %r12
	je	.LBB0_17
# %bb.19:                               # %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit.i.i33
                                        #   in Loop: Header=BB0_14 Depth=1
	cmpb	$0, 56(%r12)
	je	.LBB0_21
# %bb.20:                               #   in Loop: Header=BB0_14 Depth=1
	movzbl	67(%r12), %eax
	jmp	.LBB0_23
.LBB0_21:                               #   in Loop: Header=BB0_14 Depth=1
.Ltmp16:
	movq	%r12, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
.Ltmp17:
# %bb.22:                               # %.noexc38
                                        #   in Loop: Header=BB0_14 Depth=1
	movq	(%r12), %rax
.Ltmp18:
	movq	%r12, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp19:
.LBB0_23:                               # %_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc.exit.i35
                                        #   in Loop: Header=BB0_14 Depth=1
.Ltmp20:
	movsbl	%al, %esi
	movq	%r15, %rdi
	callq	_ZNSo3putEc
.Ltmp21:
# %bb.24:                               # %.noexc40
                                        #   in Loop: Header=BB0_14 Depth=1
.Ltmp22:
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
.Ltmp23:
	jmp	.LBB0_25
.LBB0_27:                               # %._crit_edge
	movl	8(%rsp), %ebx                   # 4-byte Reload
	testl	%ebx, %ebx
	jne	.LBB0_29
.LBB0_28:                               # %._crit_edge.thread
.Ltmp28:
	movl	$_ZSt4cout, %edi
	movl	$.L.str.2, %esi
	movl	$18, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp29:
	xorl	%ebx, %ebx
.LBB0_29:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit22
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cout+240(%rax), %r15
	testq	%r15, %r15
	je	.LBB0_30
# %bb.34:                               # %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit.i.i44
	cmpb	$0, 56(%r15)
	je	.LBB0_36
# %bb.35:
	movzbl	67(%r15), %eax
	jmp	.LBB0_38
.LBB0_36:
.Ltmp30:
	movq	%r15, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
.Ltmp31:
# %bb.37:                               # %.noexc49
	movq	(%r15), %rax
.Ltmp32:
	movq	%r15, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.Ltmp33:
.LBB0_38:                               # %_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc.exit.i46
.Ltmp34:
	movsbl	%al, %esi
	movl	$_ZSt4cout, %edi
	callq	_ZNSo3putEc
.Ltmp35:
# %bb.39:                               # %.noexc51
.Ltmp36:
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
.Ltmp37:
# %bb.40:                               # %_ZNSolsEPFRSoS_E.exit23
.Ltmp38:
	movl	$_ZSt4cout, %edi
	movl	%r14d, %esi
	callq	_ZNSolsEi
.Ltmp39:
# %bb.41:
.Ltmp40:
	movl	$.L.str.3, %esi
	movl	$113, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp41:
# %bb.42:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit24
.Ltmp42:
	movl	$_ZSt4cout, %edi
	movl	%ebp, %esi
	callq	_ZNSolsEi
.Ltmp43:
# %bb.43:
.Ltmp44:
	movl	$.L.str.4, %esi
	movl	$14, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp45:
# %bb.44:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit25
.Ltmp46:
	movl	$_ZSt4cout, %edi
	movl	%ebx, %esi
	callq	_ZNSolsEi
.Ltmp47:
# %bb.45:
.Ltmp48:
	movl	$.L.str.5, %esi
	movl	$14, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp49:
# %bb.46:                               # %_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc.exit26
	movq	16(%rsp), %rbx
	movq	24(%rsp), %r14
	cmpq	%r14, %rbx
	je	.LBB0_52
# %bb.47:                               # %.lr.ph.i.i.i.i.preheader
	addq	$16, %rbx
	jmp	.LBB0_48
	.p2align	4, 0x90
.LBB0_50:                               # %_ZSt8_DestroyI9test_infoEvPT_.exit.i.i.i.i
                                        #   in Loop: Header=BB0_48 Depth=1
	leaq	40(%rbx), %rax
	addq	$24, %rbx
	cmpq	%r14, %rbx
	movq	%rax, %rbx
	je	.LBB0_51
.LBB0_48:                               # %.lr.ph.i.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movq	-16(%rbx), %rdi
	cmpq	%rdi, %rbx
	je	.LBB0_50
# %bb.49:                               # %_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv.exit.i.i.i.i.i.i.i.i.i
                                        #   in Loop: Header=BB0_48 Depth=1
	movq	(%rbx), %rsi
	incq	%rsi
	callq	_ZdlPvm
	jmp	.LBB0_50
.LBB0_51:                               # %_ZSt8_DestroyIP9test_infoS0_EvT_S2_RSaIT0_E.exitthread-pre-split.i
	movq	16(%rsp), %rbx
.LBB0_52:                               # %_ZSt8_DestroyIP9test_infoS0_EvT_S2_RSaIT0_E.exit.i
	testq	%rbx, %rbx
	je	.LBB0_54
# %bb.53:
	movq	32(%rsp), %rsi
	subq	%rbx, %rsi
	movq	%rbx, %rdi
	callq	_ZdlPvm
.LBB0_54:                               # %_ZNSt6vectorI9test_infoSaIS0_EED2Ev.exit
	xorl	%eax, %eax
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_17:
	.cfi_def_cfa_offset 96
.Ltmp25:
	callq	_ZSt16__throw_bad_castv
.Ltmp26:
# %bb.18:                               # %.noexc37
.LBB0_3:
.Ltmp53:
	callq	_ZSt16__throw_bad_castv
.Ltmp54:
# %bb.4:                                # %.noexc
.LBB0_30:
.Ltmp50:
	callq	_ZSt16__throw_bad_castv
.Ltmp51:
# %bb.33:                               # %.noexc48
.LBB0_55:
.Ltmp55:
	jmp	.LBB0_56
.LBB0_32:                               # %.loopexit.split-lp
.Ltmp27:
	jmp	.LBB0_56
.LBB0_57:
.Ltmp52:
	jmp	.LBB0_56
.LBB0_31:                               # %.loopexit
.Ltmp24:
.LBB0_56:
	movq	%rax, %rbx
	leaq	16(%rsp), %rdi
	callq	_ZNSt6vectorI9test_infoSaIS0_EED2Ev
	movq	%rbx, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 1 <<
	.uleb128 .Ltmp13-.Ltmp0                 #   Call between .Ltmp0 and .Ltmp13
	.uleb128 .Ltmp55-.Lfunc_begin0          #     jumps to .Ltmp55
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp14-.Lfunc_begin0          # >> Call Site 2 <<
	.uleb128 .Ltmp23-.Ltmp14                #   Call between .Ltmp14 and .Ltmp23
	.uleb128 .Ltmp24-.Lfunc_begin0          #     jumps to .Ltmp24
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp28-.Lfunc_begin0          # >> Call Site 3 <<
	.uleb128 .Ltmp49-.Ltmp28                #   Call between .Ltmp28 and .Ltmp49
	.uleb128 .Ltmp52-.Lfunc_begin0          #     jumps to .Ltmp52
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp25-.Lfunc_begin0          # >> Call Site 4 <<
	.uleb128 .Ltmp26-.Ltmp25                #   Call between .Ltmp25 and .Ltmp26
	.uleb128 .Ltmp27-.Lfunc_begin0          #     jumps to .Ltmp27
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp53-.Lfunc_begin0          # >> Call Site 5 <<
	.uleb128 .Ltmp54-.Ltmp53                #   Call between .Ltmp53 and .Ltmp54
	.uleb128 .Ltmp55-.Lfunc_begin0          #     jumps to .Ltmp55
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp50-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp51-.Ltmp50                #   Call between .Ltmp50 and .Ltmp51
	.uleb128 .Ltmp52-.Lfunc_begin0          #     jumps to .Ltmp52
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp51-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Lfunc_end0-.Ltmp51            #   Call between .Ltmp51 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._ZNSt6vectorI9test_infoSaIS0_EED2Ev,"axG",@progbits,_ZNSt6vectorI9test_infoSaIS0_EED2Ev,comdat
	.weak	_ZNSt6vectorI9test_infoSaIS0_EED2Ev # -- Begin function _ZNSt6vectorI9test_infoSaIS0_EED2Ev
	.p2align	4, 0x90
	.type	_ZNSt6vectorI9test_infoSaIS0_EED2Ev,@function
_ZNSt6vectorI9test_infoSaIS0_EED2Ev:    # @_ZNSt6vectorI9test_infoSaIS0_EED2Ev
	.cfi_startproc
# %bb.0:
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %rbx
	movq	(%rdi), %r14
	movq	8(%rdi), %r15
	cmpq	%r15, %r14
	je	.LBB1_6
# %bb.1:                                # %.lr.ph.i.i.i.preheader
	addq	$16, %r14
	jmp	.LBB1_2
	.p2align	4, 0x90
.LBB1_4:                                # %_ZSt8_DestroyI9test_infoEvPT_.exit.i.i.i
                                        #   in Loop: Header=BB1_2 Depth=1
	leaq	40(%r14), %rax
	addq	$24, %r14
	cmpq	%r15, %r14
	movq	%rax, %r14
	je	.LBB1_5
.LBB1_2:                                # %.lr.ph.i.i.i
                                        # =>This Inner Loop Header: Depth=1
	movq	-16(%r14), %rdi
	cmpq	%rdi, %r14
	je	.LBB1_4
# %bb.3:                                # %_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv.exit.i.i.i.i.i.i.i.i
                                        #   in Loop: Header=BB1_2 Depth=1
	movq	(%r14), %rsi
	incq	%rsi
	callq	_ZdlPvm
	jmp	.LBB1_4
.LBB1_5:                                # %_ZSt8_DestroyIP9test_infoS0_EvT_S2_RSaIT0_E.exitthread-pre-split
	movq	(%rbx), %r14
.LBB1_6:                                # %_ZSt8_DestroyIP9test_infoS0_EvT_S2_RSaIT0_E.exit
	testq	%r14, %r14
	je	.LBB1_7
# %bb.8:
	movq	16(%rbx), %rsi
	subq	%r14, %rsi
	movq	%r14, %rdi
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	jmp	_ZdlPvm                         # TAILCALL
.LBB1_7:                                # %_ZNSt12_Vector_baseI9test_infoSaIS0_EED2Ev.exit
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	_ZNSt6vectorI9test_infoSaIS0_EED2Ev, .Lfunc_end1-_ZNSt6vectorI9test_infoSaIS0_EED2Ev
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_unit_tests.cu
	.type	_GLOBAL__sub_I_unit_tests.cu,@function
_GLOBAL__sub_I_unit_tests.cu:           # @_GLOBAL__sub_I_unit_tests.cu
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit                    # TAILCALL
.Lfunc_end2:
	.size	_GLOBAL__sub_I_unit_tests.cu, .Lfunc_end2-_GLOBAL__sub_I_unit_tests.cu
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"\n ------------------------------     Summary     ------------------------------\n"
	.size	.L.str, 81

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"Failed tests:\n"
	.size	.L.str.1, 15

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"ALL TESTS PASSED!\n"
	.size	.L.str.2, 19

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	" tests skipped (this is normal, and refers to tests that cannot be compiled due to invalid template parameters.)\n"
	.size	.L.str.3, 114

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	" tests passed\n"
	.size	.L.str.4, 15

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	" tests failed\n"
	.size	.L.str.5, 15

	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_unit_tests.cu
	.type	__hip_cuid_5225991860f36afa,@object # @__hip_cuid_5225991860f36afa
	.bss
	.globl	__hip_cuid_5225991860f36afa
__hip_cuid_5225991860f36afa:
	.byte	0                               # 0x0
	.size	__hip_cuid_5225991860f36afa, 1

	.section	".linker-options","e",@llvm_linker_options
	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _GLOBAL__sub_I_unit_tests.cu
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
	.addrsig_sym __hip_cuid_5225991860f36afa
