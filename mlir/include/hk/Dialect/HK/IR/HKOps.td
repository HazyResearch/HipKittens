#ifndef HK_OPS_TD
#define HK_OPS_TD

include "hk/Dialect/HK/IR/HKBase.td"
include "hk/Dialect/HK/IR/HKTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def HK_Axis : I32EnumAttr<"Axis", "grid axis", [
    I32EnumAttrCase<"x", 0>, I32EnumAttrCase<"y", 1>, I32EnumAttrCase<"z", 2>]> {
  let cppNamespace = "::mlir::hk";
}

def HK_Extent : I32EnumAttr<"Extent", "global tensor extent", [
    I32EnumAttrCase<"batch", 0>, I32EnumAttrCase<"depth", 1>,
    I32EnumAttrCase<"rows", 2>, I32EnumAttrCase<"cols", 3>]> {
  let cppNamespace = "::mlir::hk";
}

//===----------------------------------------------------------------------===//
// Indices and extents
//===----------------------------------------------------------------------===//

def HK_BlockIdxOp : HK_Op<"block_idx", [Pure]> {
  let summary = "blockIdx.{x,y,z}";
  let arguments = (ins HK_Axis:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$axis attr-dict";
}

def HK_WarpIdOp : HK_Op<"warp_id", [Pure]> {
  let summary = "this warp's index in the workgroup (kittens::warpid_uniform)";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def HK_LaneIdOp : HK_Op<"lane_id", [Pure]> {
  let summary = "kittens::laneid()";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def HK_CoordOp : HK_Op<"coord", [Pure]> {
  let summary = "a (b, d, r, c) index";
  let arguments = (ins I32:$b, I32:$d, I32:$r, I32:$c);
  let results = (outs HK_CoordType:$result);
  let assemblyFormat = "$b `,` $d `,` $r `,` $c attr-dict `:` qualified(type($result))";
}

def HK_ExtentOp : HK_Op<"extent", [Pure]> {
  let summary = "one extent of a global, at runtime (g.x.rows())";
  let arguments = (ins HK_GlobalType:$tensor, HK_Extent:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$tensor $axis attr-dict `:` qualified(type($tensor))";
}

def HK_AnyNeOp : HK_Op<"any_ne"> {
  let summary = "wave-level: did any entry of lhs differ from rhs";
  let arguments = (ins HK_RegVecType:$lhs, HK_RegVecType:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Global memory
//===----------------------------------------------------------------------===//

def HK_LoadOp : HK_Op<"load"> {
  let summary = "global -> register tile or vector";
  let description = [{
    `bounded` predicates the tile on the global's rows: rows past the last
    read as zero and their memory is never touched.
  }];
  let arguments = (ins Optional<HK_Elementwise>:$dst, HK_GlobalType:$src,
                       HK_CoordType:$idx, UnitAttr:$bounded);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    $src `[` $idx `]` (`into` $dst^)? attr-dict `:` qualified(type($src)) `,` qualified(type($idx))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreOp : HK_Op<"store"> {
  let summary = "register tile or vector -> global";
  let arguments = (ins HK_GlobalType:$dst, HK_Elementwise:$value,
                       HK_CoordType:$idx, UnitAttr:$bounded);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` attr-dict `:` qualified(type($value)) `,` qualified(type($dst)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_StoreScalarOp : HK_Op<"store_scalar"> {
  let summary = "one entry of a uniform register vector -> one global element";
  let arguments = (ins HK_GlobalType:$dst, HK_RegVecType:$value,
                       HK_CoordType:$idx);
  let assemblyFormat = [{
    $value `,` $dst `[` $idx `]` attr-dict `:` qualified(type($value)) `,` qualified(type($dst)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

def HK_LoadScalarOp : HK_Op<"load_scalar"> {
  let summary = "one int32 element of a global -> a wave-uniform scalar";
  let arguments = (ins HK_GlobalType:$src, HK_CoordType:$idx);
  let results = (outs I32:$result);
  let assemblyFormat = "$src `[` $idx `]` attr-dict `:` qualified(type($src)) `,` qualified(type($idx))";
  let hasVerifier = 1;
}

def HK_WgArriveOp : HK_Op<"wg_arrive"> {
  let summary = "count this workgroup's arrival; how many arrived before it";
  let arguments = (ins HK_GlobalType:$counter, HK_CoordType:$idx,
                       I32:$expected);
  let results = (outs I32:$result);
  let assemblyFormat = [{
    $counter `[` $idx `]` `,` $expected attr-dict `:` qualified(type($counter)) `,`
    qualified(type($idx))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LDS
//===----------------------------------------------------------------------===//

def HK_AllocSharedOp : HK_Op<"alloc_shared"> {
  let summary = "reserve an LDS tile, or `count` of them contiguously";
  let arguments = (ins OptionalAttr<I64Attr>:$offset,
                       OptionalAttr<I64Attr>:$nbytes);
  let results = (outs HK_SharedTileType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def HK_SharedAtOp : HK_Op<"shared_at", [Pure]> {
  let summary = "one buffer of a multi-buffer LDS allocation";
  let arguments = (ins HK_SharedTileType:$buf, I32:$index);
  let results = (outs HK_SharedTileType:$result);
  let assemblyFormat = "$buf `[` $index `]` attr-dict `:` qualified(type($buf)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_SubtileOp : HK_Op<"subtile", [Pure]> {
  let summary = "a window of an LDS tile, indexed in units of itself";
  let arguments = (ins HK_SharedTileType:$src, I32:$row, I32:$col);
  let results = (outs HK_SharedTileType:$result);
  let assemblyFormat = [{
    $src `[` $row `,` $col `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def HK_LoadSharedOp : HK_Op<"load_shared"> {
  let summary = "LDS -> register tile; `wait = false` leaves the reads in flight";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_SharedTileType:$src,
                       DefaultValuedAttr<BoolAttr, "true">:$wait,
                       I64Attr:$lds_loads);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreSharedOp : HK_Op<"store_shared"> {
  let summary = "register tile -> LDS";
  let arguments = (ins HK_SharedTileType:$dst, HK_RegTileType:$value);
  let assemblyFormat = "$value `,` $dst attr-dict `:` qualified(type($value)) `,` qualified(type($dst))";
}

def HK_GranulesOp : HK_Op<"granules"> {
  let summary = "factor one LDS tile's addressing into swizzle_bytes/16 registers";
  let arguments = (ins HK_SharedTileType:$src);
  let results = (outs HK_GranulesType:$result);
  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_LoadFragOp : HK_Op<"load_frag"> {
  let summary = "one 16x16 operand fragment out of a granule set; never waits";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_GranulesType:$granules,
                       I64Attr:$row, I64Attr:$col, I64Attr:$lds_loads,
                       DefaultValuedAttr<BoolAttr, "false">:$wait);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $granules (`into` $dst^)? attr-dict `:` qualified(type($granules))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_StoreFragOp : HK_Op<"store_frag"> {
  let summary = "one 16x16 fragment into a granule set's tile";
  let arguments = (ins HK_GranulesType:$granules, HK_RegTileType:$value,
                       Optional<I32>:$row_index, OptionalAttr<I64Attr>:$row,
                       I64Attr:$col);
  let assemblyFormat = [{
    $value `,` $granules (`,` $row_index^)? attr-dict `:` qualified(type($value)) `,`
    qualified(type($granules))
  }];
  let hasVerifier = 1;
}

def HK_LdsWaitForOp : HK_Op<"lds_wait_for"> {
  let summary = "s_waitcnt lgkmcnt(n), then re-bind each retired tile";
  let description = [{
    `n` is derived by the hk-lds-pipeline pass from the issue queue, never
    written by hand.
  }];
  let arguments = (ins Variadic<HK_RegTileType>:$tiles,
                       OptionalAttr<I64Attr>:$n);
  let assemblyFormat = "$tiles attr-dict `:` qualified(type($tiles))";
  let hasVerifier = 1;
}

def HK_AllocSharedVecOp : HK_Op<"alloc_shared_vec"> {
  let summary = "reserve `count` LDS vectors, contiguous";
  let arguments = (ins OptionalAttr<I64Attr>:$offset,
                       OptionalAttr<I64Attr>:$nbytes);
  let results = (outs HK_SharedVecType:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def HK_StoreSharedVecOp : HK_Op<"store_shared_vec"> {
  let summary = "register vector -> one entry of an LDS vector stack";
  let arguments = (ins HK_SharedVecType:$dst, HK_RegVecType:$value,
                       I32:$index);
  let assemblyFormat = [{
    $value `,` $dst `[` $index `]` attr-dict `:` qualified(type($value)) `,` qualified(type($dst))
  }];
}

def HK_LoadSharedVecOp : HK_Op<"load_shared_vec"> {
  let summary = "one entry of an LDS vector stack -> a register vector";
  let arguments = (ins HK_SharedVecType:$src, I32:$index);
  let results = (outs HK_RegVecType:$result);
  let assemblyFormat = "$src `[` $index `]` attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// Staging: global -> VGPRs -> LDS
//===----------------------------------------------------------------------===//

def HK_StageBufferOp : HK_Op<"stage_buffer"> {
  let summary = "declare the register buffer for a group copy into an LDS tile";
  let arguments = (ins HK_SharedTileType:$dst);
  let results = (outs HK_StageType:$result);
  let assemblyFormat = "$dst attr-dict `:` qualified(type($dst)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_StageLoadOp : HK_Op<"stage_load"> {
  let summary = "issue the global loads for one tile into a stage buffer";
  let arguments = (ins HK_StageType:$buf, HK_GlobalType:$src,
                       HK_CoordType:$idx, I64Attr:$slot);
  let assemblyFormat = [{
    $src `[` $idx `]` `into` $buf attr-dict `:` qualified(type($src)) `,` qualified(type($idx))
    `into` qualified(type($buf))
  }];
  let hasVerifier = 1;
}

def HK_StageCommitOp : HK_Op<"stage_commit"> {
  let summary = "write a staged buffer to LDS";
  let arguments = (ins HK_SharedTileType:$dst, HK_StageType:$buf,
                       I64Attr:$slot, DefaultValuedAttr<BoolAttr, "false">:$wait);
  let assemblyFormat = "$buf `into` $dst attr-dict `:` qualified(type($buf)) `into` qualified(type($dst))";
  let hasVerifier = 1;
}

def HK_VmWaitOp : HK_Op<"vm_wait"> {
  let summary = "s_waitcnt vmcnt(keep)";
  let arguments = (ins I64Attr:$keep);
  let assemblyFormat = "$keep attr-dict";
}

def HK_GroupLoadOp : HK_Op<"group_load"> {
  let summary = "global -> LDS by the whole workgroup, issued and waited";
  let arguments = (ins HK_SharedTileType:$dst, HK_GlobalType:$src,
                       HK_CoordType:$idx, I64Attr:$threads);
  let assemblyFormat = [{
    $src `[` $idx `]` `into` $dst attr-dict `:` qualified(type($src)) `,` qualified(type($idx))
    `into` qualified(type($dst))
  }];
}

def HK_GroupStoreOp : HK_Op<"group_store"> {
  let summary = "LDS -> global by the whole workgroup";
  let arguments = (ins HK_GlobalType:$dst, HK_SharedTileType:$src,
                       HK_CoordType:$idx, I64Attr:$threads);
  let assemblyFormat = [{
    $src `into` $dst `[` $idx `]` attr-dict `:` qualified(type($src)) `into` qualified(type($dst))
    `,` qualified(type($idx))
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization and scheduling
//===----------------------------------------------------------------------===//

def HK_BarrierOp : HK_Op<"barrier"> {
  let summary = "s_barrier; `drain` puts an lgkmcnt(0) in front of it";
  let arguments = (ins UnitAttr:$drain);
  let assemblyFormat = "attr-dict";
}

def HK_SchedBarrierOp : HK_Op<"sched_barrier"> {
  let summary = "__builtin_amdgcn_sched_barrier(0): nothing crosses this line";
  let assemblyFormat = "attr-dict";
}

def HK_SetPrioOp : HK_Op<"setprio"> {
  let summary = "s_setprio";
  let arguments = (ins I64Attr:$level);
  let assemblyFormat = "$level attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tile creation
//===----------------------------------------------------------------------===//

class HK_CreateOp<string mnemonic, dag attrs = (ins)>
    : HK_Op<mnemonic> {
  let arguments = !con((ins Optional<HK_Elementwise>:$dst), attrs);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    (`into` $dst^)? attr-dict (`:` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_ZeroOp : HK_CreateOp<"zero">;
def HK_FullOp : HK_CreateOp<"full", (ins F64Attr:$value)>;
def HK_NegInftyOp : HK_CreateOp<"neg_infty">;

//===----------------------------------------------------------------------===//
// Elementwise maps
//===----------------------------------------------------------------------===//

class HK_BinaryMapOp<string mnemonic>
    : HK_Op<mnemonic, [AttrSizedOperandSegments]> {
  let summary = "elementwise " # mnemonic # " on a tile or a vector";
  let description = [{
    The right operand is a tile or vector of the same type, a runtime i32
    scalar (cast to the element type), or absent with a literal `scalar`.
  }];
  let arguments = (ins Optional<HK_Elementwise>:$dst, HK_Elementwise:$lhs,
                       Optional<AnyTypeOf<[HK_RegTileType, HK_RegVecType, I32]>>:$rhs,
                       OptionalAttr<F64Attr>:$scalar);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    $lhs (`,` $rhs^)? (`into` $dst^)? attr-dict `:` qualified(type($lhs))
    (`,` qualified(type($rhs))^)? (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_AddOp : HK_BinaryMapOp<"add">;
def HK_SubOp : HK_BinaryMapOp<"sub">;
def HK_MulOp : HK_BinaryMapOp<"mul">;
def HK_DivOp : HK_BinaryMapOp<"div">;
def HK_MaxOp : HK_BinaryMapOp<"max">;
def HK_MinOp : HK_BinaryMapOp<"min">;

class HK_UnaryMapOp<string mnemonic, dag attrs = (ins)> : HK_Op<mnemonic> {
  let summary = "elementwise " # mnemonic;
  let arguments = !con((ins Optional<HK_Elementwise>:$dst, HK_Elementwise:$src),
                       attrs);
  let results = (outs Optional<HK_Elementwise>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_ExpOp : HK_UnaryMapOp<"exp">;
def HK_Exp2Op : HK_UnaryMapOp<"exp2">;
def HK_LogOp : HK_UnaryMapOp<"log">;
def HK_Log2Op : HK_UnaryMapOp<"log2">;
def HK_AbsOp : HK_UnaryMapOp<"abs">;
def HK_ReluOp : HK_UnaryMapOp<"relu">;
def HK_CopyOp : HK_UnaryMapOp<"copy">;
def HK_GeluOp : HK_UnaryMapOp<"gelu">;
def HK_DgeluOp : HK_UnaryMapOp<"dgelu">;
def HK_SqrtOp : HK_UnaryMapOp<"sqrt">;
def HK_RsqrtOp : HK_UnaryMapOp<"rsqrt">;
def HK_SiluOp : HK_UnaryMapOp<"silu">;

//===----------------------------------------------------------------------===//
// Reductions and broadcasts
//===----------------------------------------------------------------------===//

class HK_ReduceOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": reduce a tile to one value per row or column";
  let description = [{
    `row_*` reduces along each row and gives one value per row -- a column
    vector. `accumulate` folds into `dst` instead of overwriting it.
  }];
  let arguments = (ins Optional<HK_RegVecType>:$dst, HK_RegTileType:$src,
                       UnitAttr:$accumulate);
  let results = (outs Optional<HK_RegVecType>:$result);
  let assemblyFormat = [{
    $src (`into` $dst^)? attr-dict `:` qualified(type($src)) (`into` qualified(type($dst))^)?
    (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_RowMaxOp : HK_ReduceOp<"row_max">;
def HK_RowMinOp : HK_ReduceOp<"row_min">;
def HK_RowSumOp : HK_ReduceOp<"row_sum">;
def HK_RowProdOp : HK_ReduceOp<"row_prod">;
def HK_ColMaxOp : HK_ReduceOp<"col_max">;
def HK_ColMinOp : HK_ReduceOp<"col_min">;
def HK_ColSumOp : HK_ReduceOp<"col_sum">;
def HK_ColProdOp : HK_ReduceOp<"col_prod">;

class HK_BroadcastOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": apply vec[i] to every element of row/col i";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$src,
                       HK_RegVecType:$vec);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `,` $vec (`into` $dst^)? attr-dict `:` qualified(type($src)) `,` qualified(type($vec))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_AddRowOp : HK_BroadcastOp<"add_row">;
def HK_SubRowOp : HK_BroadcastOp<"sub_row">;
def HK_MulRowOp : HK_BroadcastOp<"mul_row">;
def HK_DivRowOp : HK_BroadcastOp<"div_row">;
def HK_AddColOp : HK_BroadcastOp<"add_col">;
def HK_SubColOp : HK_BroadcastOp<"sub_col">;
def HK_MulColOp : HK_BroadcastOp<"mul_col">;
def HK_DivColOp : HK_BroadcastOp<"div_col">;

class HK_BroadcastFillOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": a tile whose every row/col is the vector";
  let arguments = (ins HK_RegVecType:$vec);
  let results = (outs HK_RegTileType:$result);
  let assemblyFormat = "$vec attr-dict `:` qualified(type($vec)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def HK_BroadcastRowOp : HK_BroadcastFillOp<"broadcast_row">;
def HK_BroadcastColOp : HK_BroadcastFillOp<"broadcast_col">;

def HK_RetypeVecOp : HK_Op<"retype_vec"> {
  let summary = "re-attach a register vector to a tile of another shape";
  let arguments = (ins HK_RegVecType:$src);
  let results = (outs HK_RegVecType:$result);
  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Masks
//===----------------------------------------------------------------------===//

class HK_FillOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": overwrite the masked part of a tile with `value`";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$src,
                       I32:$idx, F64Attr:$value);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $src `,` $idx (`into` $dst^)? attr-dict `:` qualified(type($src))
    (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_LeftFillOp : HK_FillOp<"left_fill">;
def HK_RightFillOp : HK_FillOp<"right_fill">;
def HK_UpperFillOp : HK_FillOp<"upper_fill">;
def HK_LowerFillOp : HK_FillOp<"lower_fill">;
def HK_TriuOp : HK_FillOp<"triu">;
def HK_TrilOp : HK_FillOp<"tril">;

def HK_MakeCausalOp : HK_UnaryMapOp<"make_causal", (ins F64Attr:$value)>;
def HK_MakeCausalTOp : HK_UnaryMapOp<"make_causal_t", (ins F64Attr:$value)>;

//===----------------------------------------------------------------------===//
// Conversions and WMMA
//===----------------------------------------------------------------------===//

def HK_CastOp : HK_UnaryMapOp<"cast">;
def HK_TransposeOp : HK_UnaryMapOp<"transpose">;
def HK_TransposeSepOp : HK_UnaryMapOp<"transpose_sep">;

class HK_MmaOp<string mnemonic> : HK_Op<mnemonic> {
  let summary = mnemonic # ": acc + A x B on the WMMA unit";
  let arguments = (ins Optional<HK_RegTileType>:$dst, HK_RegTileType:$a,
                       HK_RegTileType:$b, HK_RegTileType:$c);
  let results = (outs Optional<HK_RegTileType>:$result);
  let assemblyFormat = [{
    $a `,` $b `,` $c (`into` $dst^)? attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `,`
    qualified(type($c)) (`into` qualified(type($dst))^)? (`->` qualified(type($result))^)?
  }];
  let hasVerifier = 1;
}

def HK_MmaABOp : HK_MmaOp<"mma_AB">;
def HK_MmaABtOp : HK_MmaOp<"mma_ABt">;
def HK_MmaAtBOp : HK_MmaOp<"mma_AtB">;
def HK_MmaAtBtOp : HK_MmaOp<"mma_AtBt">;

#endif // HK_OPS_TD
