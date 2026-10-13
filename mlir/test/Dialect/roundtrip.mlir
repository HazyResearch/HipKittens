// RUN: hk-opt %s | hk-opt | FileCheck %s

// Every type, and one op of each shape: a fresh tile (a result) and a write
// into one that exists (`into`).

// CHECK-LABEL: func.func @types
func.func @types(%g: !hk.gl<bf16>) attributes {hk.kernel} {
  // CHECK: hk.block_idx y
  %by = hk.block_idx y
  %w = hk.warp_id
  %c0 = arith.constant 0 : i32
  // CHECK: hk.coord {{.*}} : !hk.coord<element>
  %idx = hk.coord %c0, %c0, %by, %w : !hk.coord<element>
  // CHECK: hk.load {{.*}} -> !hk.rt<f32, 16x64, row>
  %t = hk.load %g[%idx] : !hk.gl<bf16>, !hk.coord<element> -> !hk.rt<f32, 16x64, row>
  // CHECK: hk.row_sum {{.*}} -> !hk.rv<f32, 16x64, row, col>
  %v = hk.row_sum %t : !hk.rt<f32, 16x64, row> -> !hk.rv<f32, 16x64, row, col>
  // CHECK: hk.row_max {{.*}} into {{.*}} {accumulate}
  hk.row_max %t into %v {accumulate} : !hk.rt<f32, 16x64, row> into !hk.rv<f32, 16x64, row, col>
  // CHECK: hk.mul_row {{.*}} into
  hk.mul_row %t, %v into %t : !hk.rt<f32, 16x64, row>, !hk.rv<f32, 16x64, row, col> into !hk.rt<f32, 16x64, row>
  // CHECK: hk.add {{.*}} {scalar = 1.000000e+00 : f64}
  hk.add %t into %t {scalar = 1.0 : f64} : !hk.rt<f32, 16x64, row> into !hk.rt<f32, 16x64, row>
  // CHECK: hk.mul {{.*}}, %{{.*}} : !hk.rt<f32, 16x64, row>, i32
  %s = hk.mul %t, %by : !hk.rt<f32, 16x64, row>, i32 -> !hk.rt<f32, 16x64, row>
  %b = hk.cast %s : !hk.rt<f32, 16x64, row> -> !hk.rt<bf16, 16x64, row>
  hk.store %b, %g[%idx] : !hk.rt<bf16, 16x64, row>, !hk.gl<bf16>, !hk.coord<element>
  // CHECK: hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %As = hk.alloc_shared : !hk.st<bf16, 128x64, 2>
  %a0 = hk.shared_at %As[%c0] : !hk.st<bf16, 128x64, 2> -> !hk.st<bf16, 128x64>
  // CHECK: hk.stage_buffer {{.*}} -> !hk.stage<bf16, 128x64, 256>
  %buf = hk.stage_buffer %As : !hk.st<bf16, 128x64, 2> -> !hk.stage<bf16, 128x64, 256>
  %sv = hk.alloc_shared_vec : !hk.sv<f32, 16, 4>
  // CHECK: hk.granules {{.*}} -> !hk.granules<bf16, 128x64>
  %gr = hk.granules %a0 : !hk.st<bf16, 128x64> -> !hk.granules<bf16, 128x64>
  %u = hk.retype_vec %v : !hk.rv<f32, 16x64, row, col> -> !hk.rv<f32, 16x16, row, col, uniform>
  // CHECK: hk.barrier {drain}
  hk.barrier {drain}
  hk.vm_wait 0
  hk.setprio 1
  return
}
