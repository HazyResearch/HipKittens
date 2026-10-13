// RUN: hk-opt %s -split-input-file --pass-pipeline='builtin.module(func.func(hk-check-budgets))' | FileCheck %s

// A col-layout bf16 operand read out of LDS does not vectorise. The kernel is
// right, only slow, so the pass records a warning on it rather than failing.

// CHECK-LABEL: func.func @slow_layout
// CHECK-SAME: hk.warnings = ["[slow-smem-layout] loading rt<bf16,16x16,col> from LDS: a bf16 operand in 'col' layout does not vectorise.
func.func @slow_layout() attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  %t = hk.load_shared %s {lds_loads = 0 : i64, wait = true} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, col>
  return
}

// -----

// The row-layout read of the same tile is the fast path: nothing to say.

// CHECK-LABEL: func.func @fast_layout
// CHECK-SAME: hk.warnings = []
func.func @fast_layout() attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  %t = hk.load_shared %s {lds_loads = 2 : i64, wait = true} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  return
}
