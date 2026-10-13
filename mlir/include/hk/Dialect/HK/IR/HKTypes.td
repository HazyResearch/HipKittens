#ifndef HK_TYPES_TD
#define HK_TYPES_TD

include "hk/Dialect/HK/IR/HKBase.td"

def HK_RegTileType : HK_Type<"RegTile", "rt"> {
  let summary = "a register tile: kittens::rt<T, rows, cols, layout>";
  let description = [{
    `!hk.rt<f32, 16x64, row>`. The layout is not cosmetic: it decides which
    mma variant applies and whether an LDS load is a ds_read_b128 or sixteen
    scalar reads. Extents are in units of the 16x16 WMMA fragment.
  }];
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols, "bool":$colLayout);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    ::llvm::StringRef getLayout() const { return getColLayout() ? "col" : "row"; }
    unsigned getBaseTiles() const { return (getRows() / 16) * (getCols() / 16); }
    RegTileType transposed() const {
      return RegTileType::get(getContext(), getElementType(), getCols(),
                              getRows(), !getColLayout());
    }
  }];
}

def HK_SharedTileType : HK_Type<"SharedTile", "st"> {
  let summary = "an LDS tile: kittens::st<T, rows, cols>, or `count` of them";
  let description = [{
    `!hk.st<bf16, 128x64>`, or `!hk.st<bf16, 128x64, 2>` for a contiguous,
    size-aligned double buffer. Shared tiles carry no layout: the swizzle is a
    property of the C++ type.
  }];
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols, "unsigned":$count);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    SharedTileType element() const {
      return SharedTileType::get(getContext(), getElementType(), getRows(),
                                 getCols(), 1);
    }
    uint64_t getNumBytes() const;
  }];
}

def HK_RegVecType : HK_Type<"RegVec", "rv"> {
  let summary = "a register vector: one value per row (col) or column (row) of a tile";
  let description = [{
    `!hk.rv<f32, 16x64, row, col>`: the col_vec of an rt<f32,16x64,row>. It
    carries the tile it came off, because an rv's lane layout is derived from
    that tile and a vector spelled independently of it reduces the wrong
    lanes. `uniform` records that every entry holds the same value.
  }];
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$tileRows,
                        "unsigned":$tileCols, "bool":$tileColLayout,
                        "bool":$colKind, "bool":$uniform);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ::llvm::StringRef getKind() const { return getColKind() ? "col" : "row"; }
    unsigned getLength() const { return getColKind() ? getTileRows() : getTileCols(); }
    RegTileType getTile() const {
      return RegTileType::get(getContext(), getElementType(), getTileRows(),
                              getTileCols(), getTileColLayout());
    }
    RegVecType plain() const {
      return RegVecType::get(getContext(), getElementType(), getTileRows(),
                             getTileCols(), getTileColLayout(), getColKind(),
                             false);
    }
    /// Same C++ type: rt<T,R,C,L>::col_vec does not depend on C (rt.cuh:78).
    bool sameCpp(RegVecType other) const {
      return getElementType() == other.getElementType() &&
             getLength() == other.getLength() &&
             getColKind() == other.getColKind() &&
             getTileColLayout() == other.getTileColLayout();
    }
  }];
}

def HK_SharedVecType : HK_Type<"SharedVec", "sv"> {
  let summary = "an LDS vector: kittens::sv<T, length>, or `count` of them";
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$length,
                        "unsigned":$count);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    uint64_t getNumBytes() const;
  }];
}

def HK_GlobalType : HK_Type<"Global", "gl"> {
  let summary = "a global tensor, 4D (b, d, r, c): kittens::gl<T, -1, -1, -1, -1>";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def HK_CoordType : HK_Type<"Coord", "coord"> {
  let summary = "a (b, d, r, c) index, in tiles or in elements";
  let description = [{
    `!hk.coord<tile>` indexes in whole tiles (C++ `coord<RT>`),
    `!hk.coord<element>` in elements (`coord<>`). Mixing them up reads the
    wrong memory without a diagnostic, which is why the unit is in the type.
  }];
  let parameters = (ins "bool":$element);
  let hasCustomAssemblyFormat = 1;
}

def HK_GranulesType : HK_Type<"Granules", "granules"> {
  let summary = "the per-lane LDS addresses of one shared tile, factored";
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    unsigned getSwizzleBytes() const;
    unsigned getNumAddresses() const { return getSwizzleBytes() / 16; }
    SharedTileType getTile() const {
      return SharedTileType::get(getContext(), getElementType(), getRows(),
                                 getCols(), 1);
    }
  }];
}

def HK_StageType : HK_Type<"Stage", "stage"> {
  let summary = "float4 buf[depth][calls]: the VGPR half of a global->LDS copy";
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols, "unsigned":$threads,
                        "unsigned":$depth);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    unsigned getCalls() const;
    unsigned getFloats() const { return getCalls() * getDepth(); }
    SharedTileType getTile() const {
      return SharedTileType::get(getContext(), getElementType(), getRows(),
                                 getCols(), 1);
    }
  }];
}

def HK_AnyTile : AnyTypeOf<[HK_RegTileType]>;
def HK_Elementwise : AnyTypeOf<[HK_RegTileType, HK_RegVecType],
                               "register tile or vector">;

#endif // HK_TYPES_TD
