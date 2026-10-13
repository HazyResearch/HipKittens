// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// A runtime while: the carried index is a local declared in front of the
// loop, the test is the loop's first statement, and the step is written back
// at the end of the body. The loop's result is the same local.

module {
  func.func @wsum(%x: !hk.gl<bf16> {hk.name = "x"}, %o: !hk.gl<f32> {hk.name = "o"})
      attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64, hk.consts = {}} {
    %c16 = arith.constant 16 : i32 loc("c")
    %c0 = arith.constant 0 : i32 loc("c")
    %z = hk.zero -> !hk.rt<f32, 16x16, row> loc("z")
    %k = scf.while (%i = %c0) : (i32) -> i32 {
      %cols = hk.extent %x cols : !hk.gl<bf16> loc("cols")
      %lt = arith.cmpi slt, %i, %cols : i32 loc("lt")
      scf.condition(%lt) %i : i32
    } do {
    ^bb0(%i: i32):
      %idx = hk.coord %c0, %c0, %c0, %i : !hk.coord<element> loc("idx")
      %ld = hk.load %x[%idx] : !hk.gl<bf16>, !hk.coord<element> -> !hk.rt<bf16, 16x16, row> loc("ld")
      %cvt = hk.cast %ld : !hk.rt<bf16, 16x16, row> -> !hk.rt<f32, 16x16, row> loc("cvt")
      hk.add %z, %cvt into %z : !hk.rt<f32, 16x16, row>, !hk.rt<f32, 16x16, row> into !hk.rt<f32, 16x16, row>
      %n = arith.addi %i, %c16 : i32 loc("add")
      scf.yield %n : i32
    } attributes {hk.carried_names = ["k"]}
    %last = arith.subi %k, %c16 : i32 loc("sub")
    %idx2 = hk.coord %c0, %c0, %c0, %last : !hk.coord<element> loc("idx")
    hk.store %z, %o[%idx2] : !hk.rt<f32, 16x16, row>, !hk.gl<f32>, !hk.coord<element>
    return
  }
  func.func @wsum__grid(%x: !hk.gl<bf16> {hk.name = "x"}, %o: !hk.gl<f32> {hk.name = "o"})
      -> (i32, i32, i32) attributes {hk.grid_of = @wsum} {
    %c1 = arith.constant 1 : i32
    return %c1, %c1, %c1 : i32, i32, i32
  }
}

// CHECK:      int k = c1;
// CHECK-NEXT: while (true) {
// CHECK-NEXT:     const int cols = g.x.cols();
// CHECK-NEXT:     const int lt = k < cols;
// CHECK-NEXT:     if (!(lt)) break;
// CHECK:          kittens::add(z, z, cvt);
// CHECK-NEXT:     const int add = k + c;
// CHECK-NEXT:     k = add;
// CHECK-NEXT: }
// CHECK-NEXT: const int sub = k - c;
