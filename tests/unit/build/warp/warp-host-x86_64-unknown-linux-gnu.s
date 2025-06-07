	.text
	.file	"warp.cu"
	.globl	_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE # -- Begin function _ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE
	.p2align	4, 0x90
	.type	_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE,@function
_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE: # @_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE
	.cfi_startproc
# %bb.0:
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rdi, %rbx
	movl	$_ZSt4cout, %edi
	movl	$.L.str, %esi
	movl	$97, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	_ZSt4cout(%rip), %rax
	movq	-24(%rax), %rax
	movq	_ZSt4cout+240(%rax), %r14
	testq	%r14, %r14
	je	.LBB0_5
# %bb.1:                                # %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit.i.i
	cmpb	$0, 56(%r14)
	je	.LBB0_3
# %bb.2:
	movzbl	67(%r14), %eax
	jmp	.LBB0_4
.LBB0_3:
	movq	%r14, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r14), %rax
	movq	%r14, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_4:                                # %_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.exit
	movsbl	%al, %esi
	movl	$_ZSt4cout, %edi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	movq	%rbx, %rdi
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	jmp	_ZN4warp3reg5testsERSt6vectorI9test_infoSaIS2_EE # TAILCALL
.LBB0_5:
	.cfi_def_cfa_offset 32
	callq	_ZSt16__throw_bad_castv
.Lfunc_end0:
	.size	_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE, .Lfunc_end0-_ZN4warp5testsERSt6vectorI9test_infoSaIS1_EE
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_warp.cu
	.type	_GLOBAL__sub_I_warp.cu,@function
_GLOBAL__sub_I_warp.cu:                 # @_GLOBAL__sub_I_warp.cu
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
.Lfunc_end1:
	.size	_GLOBAL__sub_I_warp.cu, .Lfunc_end1-_GLOBAL__sub_I_warp.cu
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"\n ------------------------------     Starting ops/warp tests!     ------------------------------\n"
	.size	.L.str, 98

	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_warp.cu
	.type	__hip_cuid_92588f2ae2cf3fad,@object # @__hip_cuid_92588f2ae2cf3fad
	.bss
	.globl	__hip_cuid_92588f2ae2cf3fad
__hip_cuid_92588f2ae2cf3fad:
	.byte	0                               # 0x0
	.size	__hip_cuid_92588f2ae2cf3fad, 1

	.section	".linker-options","e",@llvm_linker_options
	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _GLOBAL__sub_I_warp.cu
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
	.addrsig_sym __hip_cuid_92588f2ae2cf3fad
