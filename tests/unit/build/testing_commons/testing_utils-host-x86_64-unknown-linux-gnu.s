	.text
	.file	"testing_utils.cu"
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function _GLOBAL__sub_I_testing_utils.cu
	.type	_GLOBAL__sub_I_testing_utils.cu,@function
_GLOBAL__sub_I_testing_utils.cu:        # @_GLOBAL__sub_I_testing_utils.cu
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
.Lfunc_end0:
	.size	_GLOBAL__sub_I_testing_utils.cu, .Lfunc_end0-_GLOBAL__sub_I_testing_utils.cu
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object          # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	should_write_outputs,@object    # @should_write_outputs
	.bss
	.globl	should_write_outputs
	.p2align	2, 0x0
should_write_outputs:
	.long	0                               # 0x0
	.size	should_write_outputs, 4

	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	_GLOBAL__sub_I_testing_utils.cu
	.type	__hip_cuid_c07f308c1d3872f1,@object # @__hip_cuid_c07f308c1d3872f1
	.bss
	.globl	__hip_cuid_c07f308c1d3872f1
__hip_cuid_c07f308c1d3872f1:
	.byte	0                               # 0x0
	.size	__hip_cuid_c07f308c1d3872f1, 1

	.section	".linker-options","e",@llvm_linker_options
	.ident	"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _GLOBAL__sub_I_testing_utils.cu
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym __hip_cuid_c07f308c1d3872f1
