// RUN: hk-opt %s --hk-kernel-pipeline --mlir-print-debuginfo | hk-translate --hk-to-cpp | FileCheck %s

// The elementwise add from hk's docstring, written by hand: one tile of each
// operand, one store, a grid over the output.

module {
  func.func @add(%a: !hk.gl<bf16> {hk.name = "a"}, %b: !hk.gl<bf16> {hk.name = "b"},
                 %o: !hk.gl<bf16> {hk.name = "o"})
      attributes {hk.kernel, hk.arch = "gfx1100", hk.warps = 1 : i64,
                  hk.consts = {COLS = 64 : i64, ROWS = 16 : i64}} {
    %bx = hk.block_idx x loc("bid_x")
    %by = hk.block_idx y loc("bid_y")
    %bz = hk.block_idx z loc("bid_z")
    %c0 = arith.constant 0 : i32 loc("c")
    %idx = hk.coord %bz, %c0, %by, %bx : !hk.coord<tile> loc("idx")
    %ld = hk.load %a[%idx] : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x64, row> loc("ld")
    %ld1 = hk.load %b[%idx] : !hk.gl<bf16>, !hk.coord<tile> -> !hk.rt<bf16, 16x64, row> loc("ld")
    %s = hk.add %ld, %ld1 : !hk.rt<bf16, 16x64, row>, !hk.rt<bf16, 16x64, row> -> !hk.rt<bf16, 16x64, row> loc("add")
    hk.store %s, %o[%idx] : !hk.rt<bf16, 16x64, row>, !hk.gl<bf16>, !hk.coord<tile>
    return
  }
  func.func @add__grid(%a: !hk.gl<bf16> {hk.name = "a"}, %b: !hk.gl<bf16> {hk.name = "b"},
                       %o: !hk.gl<bf16> {hk.name = "o"}) -> (i32, i32, i32)
      attributes {hk.grid_of = @add} {
    %cols = hk.extent %o cols : !hk.gl<bf16>
    %c64 = arith.constant 64 : i32
    %c1 = arith.constant 1 : i32
    %s = arith.addi %cols, %c64 : i32
    %t = arith.subi %s, %c1 : i32
    %x = arith.divsi %t, %c64 : i32
    %rows = hk.extent %o rows : !hk.gl<bf16>
    %c16 = arith.constant 16 : i32
    %s2 = arith.addi %rows, %c16 : i32
    %t2 = arith.subi %s2, %c1 : i32
    %y = arith.divsi %t2, %c16 : i32
    %z = hk.extent %o batch : !hk.gl<bf16>
    return %x, %y, %z : i32, i32, i32
  }
}

// CHECK: // kernel : add
// CHECK: // const  : COLS=64, ROWS=16
// CHECK: return dim3((((g.o.cols() + 64) - 1) / 64), (((g.o.rows() + 16) - 1) / 16), g.o.batch());
// CHECK: using rt_bf16_16x64_row = kittens::rt<bf16, 16, 64, kittens::ducks::rt_layout::row>;
// CHECK: const int4 idx = make_int4(bid_z, c, bid_y, bid_x);
// CHECK: rt_bf16_16x64_row ld;
// CHECK-NEXT: kittens::load(ld, g.a, kittens::coord<rt_bf16_16x64_row>(idx));
// CHECK: rt_bf16_16x64_row ld1;
// CHECK: kittens::add(add, ld, ld1);
// CHECK: kittens::store(g.o, add, kittens::coord<rt_bf16_16x64_row>(idx));
