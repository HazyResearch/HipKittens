// RUN: hk-opt %s -split-input-file -verify-diagnostics

func.func @both(%t: !hk.rt<f32, 16x64, row>) {
  // expected-error@+1 {{has both a destination and a result}}
  %r = hk.exp %t into %t : !hk.rt<f32, 16x64, row> into !hk.rt<f32, 16x64, row> -> !hk.rt<f32, 16x64, row>
  return
}

// -----

func.func @gelu_bf16(%t: !hk.rt<bf16, 16x64, row>) {
  // expected-error@+1 {{has no bf16 implementation in include/rdna3}}
  %r = hk.gelu %t : !hk.rt<bf16, 16x64, row> -> !hk.rt<bf16, 16x64, row>
  return
}

// -----

func.func @mma_layout(%a: !hk.rt<bf16, 16x16, col>, %b: !hk.rt<bf16, 16x16, row>, %c: !hk.rt<f32, 16x16, col>) {
  // expected-error@+1 {{operand layouts do not match this variant}}
  hk.mma_ABt %a, %b, %c into %c : !hk.rt<bf16, 16x16, col>, !hk.rt<bf16, 16x16, row>, !hk.rt<f32, 16x16, col> into !hk.rt<f32, 16x16, col>
  return
}

// -----

func.func @reduce_kind(%t: !hk.rt<f32, 16x64, row>) {
  // expected-error@+1 {{reducing}}
  %v = hk.row_sum %t : !hk.rt<f32, 16x64, row> -> !hk.rv<f32, 16x64, row, row>
  return
}

// -----

// expected-error@+1 {{register tile rows=8 must be a positive multiple of 16}}
func.func @bad_tile(%t: !hk.rt<f32, 8x64, row>) {
  return
}
