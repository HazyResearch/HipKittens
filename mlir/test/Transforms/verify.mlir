// RUN: hk-opt %s -split-input-file --hk-verify -verify-diagnostics

// A workgroup operation every warp has to reach the same number of times
// cannot sit in a loop whose test may differ between warps.

func.func @barrier_in_while(%x: !hk.gl<bf16> {hk.name = "x"})
    attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 2 : i64, hk.consts = {}} {
  %c16 = arith.constant 16 : i32
  %c0 = arith.constant 0 : i32
  %k = scf.while (%i = %c0) : (i32) -> i32 {
    %cols = hk.extent %x cols : !hk.gl<bf16>
    %lt = arith.cmpi slt, %i, %cols : i32
    scf.condition(%lt) %i : i32
  } do {
  ^bb0(%i: i32):
    // expected-error@+1 {{is inside a runtime while loop}}
    hk.barrier
    %n = arith.addi %i, %c16 : i32
    scf.yield %n : i32
  }
  return
}

// -----

func.func @barrier_in_if(%x: !hk.gl<bf16> {hk.name = "x"})
    attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 2 : i64, hk.consts = {}} {
  %w = hk.warp_id
  %c0 = arith.constant 0 : i32
  %is0 = arith.cmpi eq, %w, %c0 : i32
  scf.if %is0 {
    // expected-error@+1 {{is inside an hk.if_}}
    hk.barrier
  }
  return
}
