// RUN: hk-opt %s -split-input-file --pass-pipeline='builtin.module(func.func(hk-lds-pipeline))' -verify-diagnostics | FileCheck %s

// Two async reads in flight; retiring the first leaves the second's two
// ds_read_b128 outstanding, so the wait is lgkmcnt(2). Retiring the second
// then waits for nothing behind it.

// CHECK-LABEL: func.func @counts
func.func @counts() attributes {hk.kernel} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  %a = hk.load_shared %s {lds_loads = 2 : i64, wait = false} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  %b = hk.load_shared %s {lds_loads = 2 : i64, wait = false} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  // CHECK: hk.lds_wait_for {{.*}} {n = 2 : i64}
  hk.lds_wait_for %a {} : !hk.rt<bf16, 16x16, row>
  // CHECK: hk.lds_wait_for {{.*}} {n = 0 : i64}
  hk.lds_wait_for %b {} : !hk.rt<bf16, 16x16, row>
  return
}

// -----

func.func @unbound_read() attributes {hk.kernel} {
  %s = hk.alloc_shared : !hk.st<bf16, 16x16>
  %a = hk.load_shared %s {lds_loads = 2 : i64, wait = false} : !hk.st<bf16, 16x16> -> !hk.rt<bf16, 16x16, row>
  // expected-error@+1 {{exp reads , whose LDS reads are still in flight}}
  %e = hk.exp %a : !hk.rt<bf16, 16x16, row> -> !hk.rt<bf16, 16x16, row>
  hk.lds_wait_for %a : !hk.rt<bf16, 16x16, row>
  return
}
