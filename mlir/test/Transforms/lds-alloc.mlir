// RUN: hk-opt %s --pass-pipeline='builtin.module(func.func(hk-lds-alloc))' | FileCheck %s

// Declaration order, each aligned to its own size when that is a power of
// two: the 16 KB pair lands at 0, the 256 B vector stack right after it, and
// the 8 KB tile is pushed past the vector to the next 8 KB boundary.

// CHECK-LABEL: func.func @alloc
// CHECK-SAME: hk.lds_bytes = 32768
// CHECK-SAME: hk.lds_static = 0
func.func @alloc() attributes {hk.kernel} {
  // CHECK: hk.alloc_shared {nbytes = 16384 : i64, offset = 0 : i64}
  %As = hk.alloc_shared : !hk.st<bf16, 64x64, 2>
  // CHECK: hk.alloc_shared_vec {nbytes = 256 : i64, offset = 16384 : i64}
  %sv = hk.alloc_shared_vec : !hk.sv<f32, 16, 4>
  // CHECK: hk.alloc_shared {nbytes = 8192 : i64, offset = 24576 : i64}
  %Bs = hk.alloc_shared : !hk.st<bf16, 64x64>
  return
}
