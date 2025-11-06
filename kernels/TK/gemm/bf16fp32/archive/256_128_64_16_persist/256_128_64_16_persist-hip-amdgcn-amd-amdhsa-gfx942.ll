; ModuleID = '256_128_64_16_persist-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_128_64_16_persist.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%"struct.kittens::alignment_dummy" = type { i32, [12 x i8] }
%struct.micro_globals = type { %"struct.kittens::gl", %"struct.kittens::gl", %"struct.kittens::gl.0" }
%"struct.kittens::gl" = type <{ ptr, %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::detail::descriptor_dict", [7 x i8] }>
%"struct.kittens::ducks::gl::runtime_dim" = type { i64 }
%"struct.kittens::detail::descriptor_dict" = type { i8 }
%"struct.kittens::gl.0" = type <{ ptr, %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::detail::descriptor_dict", [7 x i8] }>
%struct.__hip_bfloat16 = type { i16 }

@__shm = external hidden addrspace(3) global [0 x %"struct.kittens::alignment_dummy"], align 16
@__hip_cuid_acd2e23c385ef365 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_acd2e23c385ef365 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.01072.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.71076.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.71076.0.copyload = load i64, ptr addrspace(4) %.sroa.71076.0..sroa_idx, align 8
  %.sroa.91077.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.91077.0.copyload = load ptr, ptr addrspace(4) %.sroa.91077.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.171081.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.171081.0.copyload = load ptr, ptr addrspace(4) %.sroa.171081.0..sroa_idx, align 8
  %.sroa.20.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 128
  %.sroa.20.0.copyload = load i64, ptr addrspace(4) %.sroa.20.0..sroa_idx, align 8
  %2 = and i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @__shm to ptr) to i64), 15, !dbg !12
  %.not.i.i = icmp eq i64 %2, 0, !dbg !18
  %reass.sub.i.i = and i64 ptrtoint (ptr addrspacecast (ptr addrspace(3) @__shm to ptr) to i64), -16, !dbg !19
  %3 = add i64 %reass.sub.i.i, 16, !dbg !19
  %4 = inttoptr i64 %3 to ptr, !dbg !19
  %5 = select i1 %.not.i.i, ptr addrspacecast (ptr addrspace(3) @__shm to ptr), ptr %4, !dbg !19
  %6 = getelementptr inbounds i8, ptr %5, i64 16384, !dbg !20
  %7 = ptrtoint ptr %6 to i64, !dbg !21
  %8 = and i64 %7, 15, !dbg !25
  %.not.i.i170 = icmp eq i64 %8, 0, !dbg !26
  %reass.sub.i.i171 = and i64 %7, -16, !dbg !27
  %9 = add i64 %reass.sub.i.i171, 16, !dbg !27
  %10 = inttoptr i64 %9 to ptr, !dbg !27
  %11 = select i1 %.not.i.i170, ptr %6, ptr %10, !dbg !27
  %12 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !28
  %13 = lshr i32 %12, 6, !dbg !36
  %14 = lshr i32 %12, 8, !dbg !37
  %15 = and i32 %13, 3, !dbg !38
  %16 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %17 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %18 = load i32, ptr addrspace(4) %17, align 4, !tbaa !39
  %19 = trunc i64 %.sroa.71076.0.copyload to i32
  %20 = ptrtoint ptr %5 to i64
  %21 = shl nuw nsw i32 %12, 3
  %22 = and i32 %21, 56
  %23 = lshr i32 %12, 3
  %24 = and i32 %23, 63
  %25 = mul i32 %24, %19
  %26 = add i32 %25, %22
  %27 = sext i32 %26 to i64
  %28 = or i32 %23, 64
  %29 = mul i32 %28, %19
  %30 = add i32 %29, %22
  %31 = sext i32 %30 to i64
  %32 = trunc i64 %20 to i32
  %33 = shl nuw nsw i32 %12, 4
  %.masked.i.i = and i32 %33, 8064
  %34 = shl nuw nsw i32 %22, 1
  %35 = add i32 %34, %32
  %36 = or disjoint i32 %34, 8
  %37 = add i32 %36, %32
  %38 = add i32 %35, %.masked.i.i
  %39 = lshr exact i32 %38, 4
  %40 = and i32 %39, 120
  %41 = xor i32 %40, %38
  %42 = add i32 %37, %.masked.i.i
  %43 = lshr i32 %42, 4
  %44 = and i32 %43, 120
  %45 = xor i32 %44, %42
  %46 = or disjoint i32 %.masked.i.i, 8192
  %47 = add i32 %35, %46
  %48 = lshr exact i32 %47, 4
  %49 = and i32 %48, 120
  %50 = xor i32 %49, %47
  %51 = add i32 %37, %46
  %52 = lshr i32 %51, 4
  %53 = and i32 %52, 120
  %54 = xor i32 %53, %51
  %55 = trunc i64 %.sroa.15.0.copyload to i32
  %56 = ptrtoint ptr %11 to i64
  %57 = mul i32 %24, %55
  %58 = add i32 %57, %22
  %59 = sext i32 %58 to i64
  %60 = mul i32 %28, %55
  %61 = add i32 %60, %22
  %62 = sext i32 %61 to i64
  %63 = or disjoint i32 %24, 128
  %64 = mul i32 %63, %55
  %65 = add i32 %64, %22
  %66 = sext i32 %65 to i64
  %67 = or i32 %23, 192
  %68 = mul i32 %67, %55
  %69 = add i32 %68, %22
  %70 = sext i32 %69 to i64
  %71 = trunc i64 %56 to i32
  %72 = add i32 %34, %71
  %73 = add i32 %36, %71
  %74 = add i32 %72, %.masked.i.i
  %75 = lshr exact i32 %74, 4
  %76 = and i32 %75, 120
  %77 = xor i32 %76, %74
  %78 = add i32 %73, %.masked.i.i
  %79 = lshr i32 %78, 4
  %80 = and i32 %79, 120
  %81 = xor i32 %80, %78
  %82 = add i32 %72, %46
  %83 = lshr exact i32 %82, 4
  %84 = and i32 %83, 120
  %85 = xor i32 %84, %82
  %86 = add i32 %73, %46
  %87 = lshr i32 %86, 4
  %88 = and i32 %87, 120
  %89 = xor i32 %88, %86
  %90 = or disjoint i32 %.masked.i.i, 16384
  %91 = add i32 %72, %90
  %92 = lshr exact i32 %91, 4
  %93 = and i32 %92, 120
  %94 = xor i32 %93, %91
  %95 = add i32 %73, %90
  %96 = lshr i32 %95, 4
  %97 = and i32 %96, 120
  %98 = xor i32 %97, %95
  %99 = or disjoint i32 %.masked.i.i, 24576
  %100 = add i32 %72, %99
  %101 = lshr exact i32 %100, 4
  %102 = and i32 %101, 120
  %103 = xor i32 %102, %100
  %104 = add i32 %73, %99
  %105 = lshr i32 %104, 4
  %106 = and i32 %105, 120
  %107 = xor i32 %106, %104
  %108 = icmp eq i32 %14, 1
  %109 = shl nuw nsw i32 %15, 5
  %110 = and i32 %12, 15
  %111 = lshr i32 %12, 2
  %112 = and i32 %111, 12
  %113 = or disjoint i32 %109, %110
  %114 = shl nuw nsw i32 %113, 7
  %115 = shl nuw nsw i32 %112, 1
  %116 = add i32 %114, %71
  %117 = add i32 %116, %115
  %118 = lshr i32 %117, 4
  %119 = and i32 %118, 120
  %120 = xor i32 %119, %117
  %121 = or disjoint i32 %110, 16
  %122 = or disjoint i32 %109, %121
  %123 = shl nuw nsw i32 %122, 7
  %124 = add i32 %123, %71
  %125 = add i32 %124, %115
  %126 = lshr i32 %125, 4
  %127 = and i32 %126, 120
  %128 = xor i32 %127, %125
  %129 = or disjoint i32 %109, 128
  %130 = or disjoint i32 %129, %110
  %131 = shl nuw nsw i32 %130, 7
  %132 = add i32 %131, %71
  %133 = add i32 %132, %115
  %134 = lshr i32 %133, 4
  %135 = and i32 %134, 120
  %136 = xor i32 %135, %133
  %137 = or disjoint i32 %129, %121
  %138 = shl nuw nsw i32 %137, 7
  %139 = add i32 %138, %71
  %140 = add i32 %139, %115
  %141 = lshr i32 %140, 4
  %142 = and i32 %141, 120
  %143 = xor i32 %142, %140
  %144 = shl nuw nsw i32 %14, 5
  %145 = or disjoint i32 %144, %110
  %146 = shl nuw nsw i32 %145, 7
  %147 = add i32 %146, %32
  %148 = add i32 %147, %115
  %149 = lshr i32 %148, 4
  %150 = and i32 %149, 120
  %151 = xor i32 %150, %148
  %152 = or disjoint i32 %121, %144
  %153 = shl nuw nsw i32 %152, 7
  %154 = add i32 %153, %32
  %155 = add i32 %154, %115
  %156 = lshr i32 %155, 4
  %157 = and i32 %156, 120
  %158 = xor i32 %157, %155
  %159 = add nuw nsw i32 %144, 64
  %160 = or disjoint i32 %159, %110
  %161 = shl nuw nsw i32 %160, 7
  %162 = add i32 %161, %32
  %163 = add i32 %162, %115
  %164 = lshr i32 %163, 4
  %165 = and i32 %164, 120
  %166 = xor i32 %165, %163
  %167 = or disjoint i32 %159, %121
  %168 = shl nuw nsw i32 %167, 7
  %169 = add i32 %168, %32
  %170 = add i32 %169, %115
  %171 = lshr i32 %170, 4
  %172 = and i32 %171, 120
  %173 = xor i32 %172, %170
  %174 = shl i32 %30, 1
  %175 = shl i32 %26, 1
  %176 = shl i32 %19, 8
  %177 = or disjoint i32 %115, 32
  %178 = add i32 %116, %177
  %179 = lshr i32 %178, 4
  %180 = and i32 %179, 120
  %181 = xor i32 %180, %178
  %182 = add i32 %124, %177
  %183 = lshr i32 %182, 4
  %184 = and i32 %183, 120
  %185 = xor i32 %184, %182
  %186 = add i32 %132, %177
  %187 = lshr i32 %186, 4
  %188 = and i32 %187, 120
  %189 = xor i32 %188, %186
  %190 = add i32 %139, %177
  %191 = lshr i32 %190, 4
  %192 = and i32 %191, 120
  %193 = xor i32 %192, %190
  %194 = add i32 %147, %177
  %195 = lshr i32 %194, 4
  %196 = and i32 %195, 120
  %197 = xor i32 %196, %194
  %198 = add i32 %154, %177
  %199 = lshr i32 %198, 4
  %200 = and i32 %199, 120
  %201 = xor i32 %200, %198
  %202 = add i32 %162, %177
  %203 = lshr i32 %202, 4
  %204 = and i32 %203, 120
  %205 = xor i32 %204, %202
  %206 = add i32 %169, %177
  %207 = lshr i32 %206, 4
  %208 = and i32 %207, 120
  %209 = xor i32 %208, %206
  %210 = shl i32 %55, 9
  %211 = shl i32 %58, 1
  %212 = shl i32 %61, 1
  %213 = shl i32 %65, 1
  %214 = shl i32 %69, 1
  %215 = or disjoint i32 %115, 64
  %216 = add i32 %116, %215
  %217 = lshr i32 %216, 4
  %218 = and i32 %217, 120
  %219 = xor i32 %218, %216
  %220 = add i32 %124, %215
  %221 = lshr i32 %220, 4
  %222 = and i32 %221, 120
  %223 = xor i32 %222, %220
  %224 = add i32 %132, %215
  %225 = lshr i32 %224, 4
  %226 = and i32 %225, 120
  %227 = xor i32 %226, %224
  %228 = add i32 %139, %215
  %229 = lshr i32 %228, 4
  %230 = and i32 %229, 120
  %231 = xor i32 %230, %228
  %232 = add i32 %147, %215
  %233 = lshr i32 %232, 4
  %234 = and i32 %233, 120
  %235 = xor i32 %234, %232
  %236 = add i32 %154, %215
  %237 = lshr i32 %236, 4
  %238 = and i32 %237, 120
  %239 = xor i32 %238, %236
  %240 = add i32 %162, %215
  %241 = lshr i32 %240, 4
  %242 = and i32 %241, 120
  %243 = xor i32 %242, %240
  %244 = add i32 %169, %215
  %245 = lshr i32 %244, 4
  %246 = and i32 %245, 120
  %247 = xor i32 %246, %244
  %248 = or disjoint i32 %115, 96
  %249 = add i32 %116, %248
  %250 = lshr i32 %249, 4
  %251 = and i32 %250, 120
  %252 = xor i32 %251, %249
  %253 = add i32 %124, %248
  %254 = lshr i32 %253, 4
  %255 = and i32 %254, 120
  %256 = xor i32 %255, %253
  %257 = add i32 %132, %248
  %258 = lshr i32 %257, 4
  %259 = and i32 %258, 120
  %260 = xor i32 %259, %257
  %261 = add i32 %139, %248
  %262 = lshr i32 %261, 4
  %263 = and i32 %262, 120
  %264 = xor i32 %263, %261
  %265 = add i32 %147, %248
  %266 = lshr i32 %265, 4
  %267 = and i32 %266, 120
  %268 = xor i32 %267, %265
  %269 = add i32 %154, %248
  %270 = lshr i32 %269, 4
  %271 = and i32 %270, 120
  %272 = xor i32 %271, %269
  %273 = add i32 %162, %248
  %274 = lshr i32 %273, 4
  %275 = and i32 %274, 120
  %276 = xor i32 %275, %273
  %277 = add i32 %169, %248
  %278 = lshr i32 %277, 4
  %279 = and i32 %278, 120
  %280 = xor i32 %279, %277
  %281 = and i32 %33, 112
  %282 = add i32 %281, %32
  %283 = or disjoint i32 %281, 8
  %284 = add i32 %283, %32
  %285 = add i32 %282, %.masked.i.i
  %286 = lshr exact i32 %285, 4
  %287 = and i32 %286, 120
  %288 = xor i32 %287, %285
  %289 = add i32 %284, %.masked.i.i
  %290 = lshr i32 %289, 4
  %291 = and i32 %290, 120
  %292 = xor i32 %291, %289
  %293 = add i32 %282, %46
  %294 = lshr exact i32 %293, 4
  %295 = and i32 %294, 120
  %296 = xor i32 %295, %293
  %297 = add i32 %284, %46
  %298 = lshr i32 %297, 4
  %299 = and i32 %298, 120
  %300 = xor i32 %299, %297
  %301 = add i32 %281, %71
  %302 = add i32 %283, %71
  %303 = add i32 %301, %.masked.i.i
  %304 = lshr exact i32 %303, 4
  %305 = and i32 %304, 120
  %306 = xor i32 %305, %303
  %307 = add i32 %302, %.masked.i.i
  %308 = lshr i32 %307, 4
  %309 = and i32 %308, 120
  %310 = xor i32 %309, %307
  %311 = add i32 %301, %46
  %312 = lshr exact i32 %311, 4
  %313 = and i32 %312, 120
  %314 = xor i32 %313, %311
  %315 = add i32 %302, %46
  %316 = lshr i32 %315, 4
  %317 = and i32 %316, 120
  %318 = xor i32 %317, %315
  %319 = add i32 %301, %90
  %320 = lshr exact i32 %319, 4
  %321 = and i32 %320, 120
  %322 = xor i32 %321, %319
  %323 = add i32 %302, %90
  %324 = lshr i32 %323, 4
  %325 = and i32 %324, 120
  %326 = xor i32 %325, %323
  %327 = add i32 %301, %99
  %328 = lshr exact i32 %327, 4
  %329 = and i32 %328, 120
  %330 = xor i32 %329, %327
  %331 = add i32 %302, %99
  %332 = lshr i32 %331, 4
  %333 = and i32 %332, 120
  %334 = xor i32 %333, %331
  %335 = icmp ult i32 %12, 256
  %336 = trunc i64 %.sroa.20.0.copyload to i32
  %337 = mul nsw i32 %112, %336
  %338 = or disjoint i32 %112, 1
  %339 = mul nsw i32 %338, %336
  %340 = or disjoint i32 %112, 2
  %341 = mul nsw i32 %340, %336
  %342 = or disjoint i32 %112, 3
  %343 = mul nsw i32 %342, %336
  %344 = add nsw i32 %337, %110
  %345 = sext i32 %344 to i64
  %346 = add nsw i32 %339, %110
  %347 = sext i32 %346 to i64
  %348 = add nsw i32 %341, %110
  %349 = sext i32 %348 to i64
  %350 = add nsw i32 %343, %110
  %351 = sext i32 %350 to i64
  %352 = add nsw i32 %337, %121
  %353 = sext i32 %352 to i64
  %354 = add nsw i32 %339, %121
  %355 = sext i32 %354 to i64
  %356 = add nsw i32 %341, %121
  %357 = sext i32 %356 to i64
  %358 = add nsw i32 %343, %121
  %359 = sext i32 %358 to i64
  %360 = or disjoint i32 %112, 16
  %361 = mul nsw i32 %360, %336
  %362 = or disjoint i32 %112, 17
  %363 = mul nsw i32 %362, %336
  %364 = or disjoint i32 %112, 18
  %365 = mul nsw i32 %364, %336
  %366 = or disjoint i32 %112, 19
  %367 = mul nsw i32 %366, %336
  %368 = add nsw i32 %361, %110
  %369 = sext i32 %368 to i64
  %370 = add nsw i32 %363, %110
  %371 = sext i32 %370 to i64
  %372 = add nsw i32 %365, %110
  %373 = sext i32 %372 to i64
  %374 = add nsw i32 %367, %110
  %375 = sext i32 %374 to i64
  %376 = add nsw i32 %361, %121
  %377 = sext i32 %376 to i64
  %378 = add nsw i32 %363, %121
  %379 = sext i32 %378 to i64
  %380 = add nsw i32 %365, %121
  %381 = sext i32 %380 to i64
  %382 = add nsw i32 %367, %121
  %383 = sext i32 %382 to i64
  br label %384, !dbg !43

384:                                              ; preds = %718, %1
  %.sroa.0796.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.0796.2.off0, %718 ]
  %.sroa.0796.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.0796.2.off64, %718 ]
  %.sroa.5.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.5.2.off0, %718 ]
  %.sroa.5.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.5.2.off64, %718 ]
  %.sroa.8797.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.8797.2.off0, %718 ]
  %.sroa.8797.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.8797.2.off64, %718 ]
  %.sroa.11.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.11.2.off0, %718 ]
  %.sroa.11.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.11.2.off64, %718 ]
  %.sroa.0798.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.0798.2.off0, %718 ]
  %.sroa.0798.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.0798.2.off64, %718 ]
  %.sroa.5799.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.5799.2.off0, %718 ]
  %.sroa.5799.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.5799.2.off64, %718 ]
  %.0 = phi i32 [ 0, %1 ], [ %806, %718 ], !dbg !44
  %385 = mul i32 %18, %.0, !dbg !45
  %386 = add i32 %385, %16, !dbg !46
  %387 = icmp sgt i32 %386, 2047, !dbg !47
  br i1 %387, label %_Z12get_task_idxi.exit, label %388, !dbg !50

388:                                              ; preds = %384
  %389 = sdiv i32 %386, 8, !dbg !51
  %390 = shl i32 %386, 8, !dbg !52
  %391 = mul i32 %389, -2047, !dbg !52
  %392 = add i32 %391, %390, !dbg !52
  %.frozen = freeze i32 %392, !dbg !53
  %393 = sdiv i32 %.frozen, 512, !dbg !53
  %394 = shl nsw i32 %393, 4, !dbg !54
  %395 = sub nsw i32 64, %394, !dbg !55
  %396 = tail call noundef range(i32 -16, 8388722) i32 @llvm.smin.i32(i32 %395, i32 16), !dbg !56
  %397 = mul i32 %393, 512, !dbg !60
  %.decomposed = sub i32 %.frozen, %397, !dbg !60
  %.lhs.trunc.i = trunc nsw i32 %.decomposed to i16, !dbg !61
  %.rhs.trunc.i = trunc i32 %396 to i16, !dbg !61
  %398 = sdiv i16 %.lhs.trunc.i, %.rhs.trunc.i, !dbg !62
  %399 = mul i16 %398, %.rhs.trunc.i, !dbg !61
  %.decomposed1404 = sub i16 %.lhs.trunc.i, %399, !dbg !61
  %.sext.i = sext i16 %.decomposed1404 to i32, !dbg !61
  %400 = add nsw i32 %394, %.sext.i, !dbg !63
  %.sext23.i = sext i16 %398 to i32, !dbg !62
  %401 = insertelement <2 x i32> poison, i32 %400, i64 0, !dbg !64
  %402 = insertelement <2 x i32> %401, i32 %.sext23.i, i64 1, !dbg !64
  br label %_Z12get_task_idxi.exit

_Z12get_task_idxi.exit:                           ; preds = %384, %388
  %.sroa.0.0.i = phi <2 x i32> [ %402, %388 ], [ <i32 -1, i32 -1>, %384 ], !dbg !70
  %.sroa.0.0.vec.extract.i = extractelement <2 x i32> %.sroa.0.0.i, i64 0, !dbg !73
  %.sroa.0.4.vec.extract.i = extractelement <2 x i32> %.sroa.0.0.i, i64 1, !dbg !73
  %.not169 = icmp eq i32 %.sroa.0.0.vec.extract.i, -1, !dbg !74
  br i1 %.not169, label %.critedge, label %403, !dbg !75

403:                                              ; preds = %_Z12get_task_idxi.exit
  %404 = shl nsw i32 %.sroa.0.0.vec.extract.i, 7, !dbg !76
  %405 = sext i32 %404 to i64, !dbg !86
  %406 = mul i64 %.sroa.71076.0.copyload, %405, !dbg !90
  %407 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.01072.0.copyload, i64 %406, !dbg !91
  %408 = getelementptr inbounds %struct.__hip_bfloat16, ptr %407, i64 %27, !dbg !92
  %409 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %408) #7, !dbg !93, !srcloc !97
  %410 = getelementptr inbounds %struct.__hip_bfloat16, ptr %407, i64 %31, !dbg !92
  %411 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %410) #7, !dbg !93, !srcloc !97
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !98, !srcloc !99
  %.sroa.0.0.extract.trunc.i.i = trunc i128 %409 to i64, !dbg !100
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %41, i64 %.sroa.0.0.extract.trunc.i.i) #7, !dbg !101, !srcloc !104
  %.sroa.0.8.extract.shift.i.i = lshr i128 %409, 64, !dbg !105
  %.sroa.0.8.extract.trunc.i.i = trunc nuw i128 %.sroa.0.8.extract.shift.i.i to i64, !dbg !105
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %45, i64 %.sroa.0.8.extract.trunc.i.i) #7, !dbg !106, !srcloc !104
  %.sroa.5.16.extract.trunc.i.i = trunc i128 %411 to i64, !dbg !100
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %50, i64 %.sroa.5.16.extract.trunc.i.i) #7, !dbg !101, !srcloc !104
  %.sroa.5.24.extract.shift.i.i = lshr i128 %411, 64, !dbg !105
  %.sroa.5.24.extract.trunc.i.i = trunc nuw i128 %.sroa.5.24.extract.shift.i.i to i64, !dbg !105
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %54, i64 %.sroa.5.24.extract.trunc.i.i) #7, !dbg !106, !srcloc !104
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !108, !srcloc !109
  %412 = shl nsw i32 %.sroa.0.4.vec.extract.i, 8, !dbg !110
  %413 = sext i32 %412 to i64, !dbg !117
  %414 = mul i64 %.sroa.15.0.copyload, %413, !dbg !119
  %415 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.91077.0.copyload, i64 %414, !dbg !120
  %416 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %59, !dbg !121
  %417 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %416) #7, !dbg !122, !srcloc !97
  %418 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %62, !dbg !121
  %419 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %418) #7, !dbg !122, !srcloc !97
  %420 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %66, !dbg !121
  %421 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %420) #7, !dbg !122, !srcloc !97
  %422 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %70, !dbg !121
  %423 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %422) #7, !dbg !122, !srcloc !97
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !124, !srcloc !99
  %.sroa.0.0.extract.trunc.i.i176 = trunc i128 %417 to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %77, i64 %.sroa.0.0.extract.trunc.i.i176) #7, !dbg !126, !srcloc !104
  %.sroa.0.8.extract.shift.i.i177 = lshr i128 %417, 64, !dbg !128
  %.sroa.0.8.extract.trunc.i.i178 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i177 to i64, !dbg !128
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %81, i64 %.sroa.0.8.extract.trunc.i.i178) #7, !dbg !129, !srcloc !104
  %.sroa.5.16.extract.trunc.i.i179 = trunc i128 %419 to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %85, i64 %.sroa.5.16.extract.trunc.i.i179) #7, !dbg !126, !srcloc !104
  %.sroa.5.24.extract.shift.i.i180 = lshr i128 %419, 64, !dbg !128
  %.sroa.5.24.extract.trunc.i.i181 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i180 to i64, !dbg !128
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %89, i64 %.sroa.5.24.extract.trunc.i.i181) #7, !dbg !129, !srcloc !104
  %.sroa.8.32.extract.trunc.i.i = trunc i128 %421 to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %94, i64 %.sroa.8.32.extract.trunc.i.i) #7, !dbg !126, !srcloc !104
  %.sroa.8.40.extract.shift.i.i = lshr i128 %421, 64, !dbg !128
  %.sroa.8.40.extract.trunc.i.i = trunc nuw i128 %.sroa.8.40.extract.shift.i.i to i64, !dbg !128
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %98, i64 %.sroa.8.40.extract.trunc.i.i) #7, !dbg !129, !srcloc !104
  %.sroa.11.48.extract.trunc.i.i = trunc i128 %423 to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %103, i64 %.sroa.11.48.extract.trunc.i.i) #7, !dbg !126, !srcloc !104
  %.sroa.11.56.extract.shift.i.i = lshr i128 %423, 64, !dbg !128
  %.sroa.11.56.extract.trunc.i.i = trunc nuw i128 %.sroa.11.56.extract.shift.i.i to i64, !dbg !128
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %107, i64 %.sroa.11.56.extract.trunc.i.i) #7, !dbg !129, !srcloc !104
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !131, !srcloc !109
  tail call void @llvm.amdgcn.s.barrier(), !dbg !132
  br i1 %108, label %424, label %.preheader, !dbg !133

.preheader:                                       ; preds = %424, %403
  br label %426, !dbg !134

424:                                              ; preds = %403
  tail call void @llvm.amdgcn.s.barrier(), !dbg !135
  br label %.preheader, !dbg !136

425:                                              ; preds = %700
  br i1 %335, label %717, label %718, !dbg !137

426:                                              ; preds = %.preheader, %700
  %427 = phi <4 x float> [ %716, %700 ], [ zeroinitializer, %.preheader ]
  %428 = phi <4 x float> [ %715, %700 ], [ zeroinitializer, %.preheader ]
  %429 = phi <4 x float> [ %714, %700 ], [ zeroinitializer, %.preheader ]
  %.val30.i2721207 = phi <4 x float> [ %713, %700 ], [ zeroinitializer, %.preheader ]
  %430 = phi <4 x float> [ %712, %700 ], [ zeroinitializer, %.preheader ]
  %431 = phi <4 x float> [ %711, %700 ], [ zeroinitializer, %.preheader ]
  %432 = phi <4 x float> [ %710, %700 ], [ zeroinitializer, %.preheader ]
  %.val30.i2601198 = phi <4 x float> [ %709, %700 ], [ zeroinitializer, %.preheader ]
  %433 = phi <4 x float> [ %708, %700 ], [ zeroinitializer, %.preheader ]
  %434 = phi <4 x float> [ %707, %700 ], [ zeroinitializer, %.preheader ]
  %435 = phi <4 x float> [ %706, %700 ], [ zeroinitializer, %.preheader ]
  %.val30.i2481189 = phi <4 x float> [ %705, %700 ], [ zeroinitializer, %.preheader ]
  %.val30.1.1.i6461186 = phi <4 x float> [ %704, %700 ], [ zeroinitializer, %.preheader ]
  %.val30.135.i6431183 = phi <4 x float> [ %703, %700 ], [ zeroinitializer, %.preheader ]
  %.val30.1.i6401180 = phi <4 x float> [ %702, %700 ], [ zeroinitializer, %.preheader ]
  %.01671177 = phi i32 [ %436, %700 ], [ 0, %.preheader ]
  %.sroa.5799.1.off641176 = phi i64 [ %.sroa.5799.2.off64, %700 ], [ %.sroa.5799.0.off64, %.preheader ]
  %.sroa.5799.1.off01175 = phi i64 [ %.sroa.5799.2.off0, %700 ], [ %.sroa.5799.0.off0, %.preheader ]
  %.sroa.0798.1.off641174 = phi i64 [ %.sroa.0798.2.off64, %700 ], [ %.sroa.0798.0.off64, %.preheader ]
  %.sroa.0798.1.off01173 = phi i64 [ %.sroa.0798.2.off0, %700 ], [ %.sroa.0798.0.off0, %.preheader ]
  %.sroa.11.1.off641172 = phi i64 [ %.sroa.11.2.off64, %700 ], [ %.sroa.11.0.off64, %.preheader ]
  %.sroa.11.1.off01171 = phi i64 [ %.sroa.11.2.off0, %700 ], [ %.sroa.11.0.off0, %.preheader ]
  %.sroa.8797.1.off641170 = phi i64 [ %.sroa.8797.2.off64, %700 ], [ %.sroa.8797.0.off64, %.preheader ]
  %.sroa.8797.1.off01169 = phi i64 [ %.sroa.8797.2.off0, %700 ], [ %.sroa.8797.0.off0, %.preheader ]
  %.sroa.5.1.off641168 = phi i64 [ %.sroa.5.2.off64, %700 ], [ %.sroa.5.0.off64, %.preheader ]
  %.sroa.5.1.off01167 = phi i64 [ %.sroa.5.2.off0, %700 ], [ %.sroa.5.0.off0, %.preheader ]
  %.sroa.0796.1.off641166 = phi i64 [ %.sroa.0796.2.off64, %700 ], [ %.sroa.0796.0.off64, %.preheader ]
  %.sroa.0796.1.off01165 = phi i64 [ %.sroa.0796.2.off0, %700 ], [ %.sroa.0796.0.off0, %.preheader ]
  %.val30.i11631164 = phi <4 x float> [ %701, %700 ], [ zeroinitializer, %.preheader ]
  %436 = add nuw nsw i32 %.01671177, 1, !dbg !138
  %.not = icmp eq i32 %.01671177, 127, !dbg !139
  %437 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %120) #7, !dbg !140, !srcloc !146
  %.sroa.036.2.extract.shift37.i = lshr i64 %437, 16, !dbg !147
  %438 = insertelement <2 x i64> poison, i64 %437, i64 0, !dbg !147
  %439 = insertelement <2 x i64> %438, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !147
  %440 = shufflevector <2 x i64> %438, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !140
  %441 = lshr <2 x i64> %440, <i64 32, i64 48>, !dbg !140
  %442 = shufflevector <2 x i64> %439, <2 x i64> %441, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !148
  %.sroa.0866.6.vec.insert8821373 = trunc <4 x i64> %442 to <4 x i16>, !dbg !148
  %443 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %128) #7, !dbg !140, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %443, 16, !dbg !147
  %444 = insertelement <2 x i64> poison, i64 %443, i64 0, !dbg !147
  %445 = insertelement <2 x i64> %444, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !147
  %446 = shufflevector <2 x i64> %444, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !140
  %447 = lshr <2 x i64> %446, <i64 32, i64 48>, !dbg !140
  %448 = shufflevector <2 x i64> %445, <2 x i64> %447, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !148
  %.sroa.18883.14.vec.insert8991372 = trunc <4 x i64> %448 to <4 x i16>, !dbg !148
  %449 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %136) #7, !dbg !152, !srcloc !146
  %.sroa.036.2.extract.shift37.i189 = lshr i64 %449, 16, !dbg !155
  %450 = insertelement <2 x i64> poison, i64 %449, i64 0, !dbg !155
  %451 = insertelement <2 x i64> %450, i64 %.sroa.036.2.extract.shift37.i189, i64 1, !dbg !155
  %452 = shufflevector <2 x i64> %450, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !152
  %453 = lshr <2 x i64> %452, <i64 32, i64 48>, !dbg !152
  %454 = shufflevector <2 x i64> %451, <2 x i64> %453, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !156
  %.sroa.0801.6.vec.insert8171371 = trunc <4 x i64> %454 to <4 x i16>, !dbg !156
  %455 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %143) #7, !dbg !152, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i198 = lshr i64 %455, 16, !dbg !155
  %456 = insertelement <2 x i64> poison, i64 %455, i64 0, !dbg !155
  %457 = insertelement <2 x i64> %456, i64 %.sroa.036.2.extract.shift37.1.i198, i64 1, !dbg !155
  %458 = shufflevector <2 x i64> %456, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !152
  %459 = lshr <2 x i64> %458, <i64 32, i64 48>, !dbg !152
  %460 = shufflevector <2 x i64> %457, <2 x i64> %459, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !156
  %.sroa.18.14.vec.insert8331370 = trunc <4 x i64> %460 to <4 x i16>, !dbg !156
  %461 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %151) #7, !dbg !158, !srcloc !146
  %.sroa.036.2.extract.shift37.i208 = lshr i64 %461, 16, !dbg !162
  %462 = insertelement <2 x i64> poison, i64 %461, i64 0, !dbg !162
  %463 = insertelement <2 x i64> %462, i64 %.sroa.036.2.extract.shift37.i208, i64 1, !dbg !162
  %464 = shufflevector <2 x i64> %462, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !158
  %465 = lshr <2 x i64> %464, <i64 32, i64 48>, !dbg !158
  %466 = shufflevector <2 x i64> %463, <2 x i64> %465, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !163
  %.sroa.01002.6.vec.insert10181369 = trunc <4 x i64> %466 to <4 x i16>, !dbg !163
  %467 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %158) #7, !dbg !158, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i215 = lshr i64 %467, 16, !dbg !162
  %468 = insertelement <2 x i64> poison, i64 %467, i64 0, !dbg !162
  %469 = insertelement <2 x i64> %468, i64 %.sroa.036.2.extract.shift37.1.i215, i64 1, !dbg !162
  %470 = shufflevector <2 x i64> %468, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !158
  %471 = lshr <2 x i64> %470, <i64 32, i64 48>, !dbg !158
  %472 = shufflevector <2 x i64> %469, <2 x i64> %471, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !163
  %.sroa.181019.14.vec.insert10351368 = trunc <4 x i64> %472 to <4 x i16>, !dbg !163
  %473 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %166) #7, !dbg !165, !srcloc !146
  %.sroa.036.2.extract.shift37.i227 = lshr i64 %473, 16, !dbg !168
  %474 = insertelement <2 x i64> poison, i64 %473, i64 0, !dbg !168
  %475 = insertelement <2 x i64> %474, i64 %.sroa.036.2.extract.shift37.i227, i64 1, !dbg !168
  %476 = shufflevector <2 x i64> %474, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !165
  %477 = lshr <2 x i64> %476, <i64 32, i64 48>, !dbg !165
  %478 = shufflevector <2 x i64> %475, <2 x i64> %477, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !169
  %.sroa.0934.6.vec.insert9501367 = trunc <4 x i64> %478 to <4 x i16>, !dbg !169
  %479 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %173) #7, !dbg !165, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i236 = lshr i64 %479, 16, !dbg !168
  %480 = insertelement <2 x i64> poison, i64 %479, i64 0, !dbg !168
  %481 = insertelement <2 x i64> %480, i64 %.sroa.036.2.extract.shift37.1.i236, i64 1, !dbg !168
  %482 = shufflevector <2 x i64> %480, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !165
  %483 = lshr <2 x i64> %482, <i64 32, i64 48>, !dbg !165
  %484 = shufflevector <2 x i64> %481, <2 x i64> %483, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !169
  %.sroa.18951.14.vec.insert9671366 = trunc <4 x i64> %484 to <4 x i16>, !dbg !169
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !171
  br i1 %.not, label %494, label %485, !dbg !172

485:                                              ; preds = %426
  %486 = shl nuw nsw i32 %436, 6
  %487 = zext nneg i32 %486 to i64
  %488 = getelementptr %struct.__hip_bfloat16, ptr %407, i64 %487
  %489 = ptrtoint ptr %488 to i64
  %490 = bitcast i64 %489 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %490, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %491 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %491, i32 %176, i64 2
  %492 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %175, i32 noundef 0, i32 noundef 0) #7, !dbg !173
  %493 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %174, i32 noundef 0, i32 noundef 0) #7, !dbg !173
  %extract.t1136 = trunc i128 %492 to i64, !dbg !178
  %extract1138 = lshr i128 %492, 64, !dbg !178
  %extract.t1139 = trunc nuw i128 %extract1138 to i64, !dbg !178
  %extract.t1149 = trunc i128 %493 to i64, !dbg !178
  %extract1151 = lshr i128 %493, 64, !dbg !178
  %extract.t1152 = trunc nuw i128 %extract1151 to i64, !dbg !178
  br label %494, !dbg !178

494:                                              ; preds = %485, %426
  %.sroa.0798.2.off0 = phi i64 [ %.sroa.0798.1.off01173, %426 ], [ %extract.t1136, %485 ]
  %.sroa.0798.2.off64 = phi i64 [ %.sroa.0798.1.off641174, %426 ], [ %extract.t1139, %485 ]
  %.sroa.5799.2.off0 = phi i64 [ %.sroa.5799.1.off01175, %426 ], [ %extract.t1149, %485 ]
  %.sroa.5799.2.off64 = phi i64 [ %.sroa.5799.1.off641176, %426 ], [ %extract.t1152, %485 ]
  tail call void @llvm.amdgcn.s.barrier(), !dbg !179
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !180
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !181, !srcloc !182
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !183
  %495 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert10181369, <4 x i16> %.sroa.0866.6.vec.insert8821373, <4 x float> %.val30.i11631164, i32 0, i32 0, i32 0), !dbg !184
  %496 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert10181369, <4 x i16> %.sroa.18883.14.vec.insert8991372, <4 x float> %.val30.1.i6401180, i32 0, i32 0, i32 0), !dbg !184
  %497 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert10351368, <4 x i16> %.sroa.0866.6.vec.insert8821373, <4 x float> %.val30.135.i6431183, i32 0, i32 0, i32 0), !dbg !184
  %498 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert10351368, <4 x i16> %.sroa.18883.14.vec.insert8991372, <4 x float> %.val30.1.1.i6461186, i32 0, i32 0, i32 0), !dbg !184
  %499 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert10181369, <4 x i16> %.sroa.0801.6.vec.insert8171371, <4 x float> %.val30.i2481189, i32 0, i32 0, i32 0), !dbg !192
  %500 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert10181369, <4 x i16> %.sroa.18.14.vec.insert8331370, <4 x float> %435, i32 0, i32 0, i32 0), !dbg !192
  %501 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert10351368, <4 x i16> %.sroa.0801.6.vec.insert8171371, <4 x float> %434, i32 0, i32 0, i32 0), !dbg !192
  %502 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert10351368, <4 x i16> %.sroa.18.14.vec.insert8331370, <4 x float> %433, i32 0, i32 0, i32 0), !dbg !192
  %503 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert9501367, <4 x i16> %.sroa.0866.6.vec.insert8821373, <4 x float> %.val30.i2601198, i32 0, i32 0, i32 0), !dbg !196
  %504 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert9501367, <4 x i16> %.sroa.18883.14.vec.insert8991372, <4 x float> %432, i32 0, i32 0, i32 0), !dbg !196
  %505 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert9671366, <4 x i16> %.sroa.0866.6.vec.insert8821373, <4 x float> %431, i32 0, i32 0, i32 0), !dbg !196
  %506 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert9671366, <4 x i16> %.sroa.18883.14.vec.insert8991372, <4 x float> %430, i32 0, i32 0, i32 0), !dbg !196
  %507 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert9501367, <4 x i16> %.sroa.0801.6.vec.insert8171371, <4 x float> %.val30.i2721207, i32 0, i32 0, i32 0), !dbg !200
  %508 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert9501367, <4 x i16> %.sroa.18.14.vec.insert8331370, <4 x float> %429, i32 0, i32 0, i32 0), !dbg !200
  %509 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert9671366, <4 x i16> %.sroa.0801.6.vec.insert8171371, <4 x float> %428, i32 0, i32 0, i32 0), !dbg !200
  %510 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert9671366, <4 x i16> %.sroa.18.14.vec.insert8331370, <4 x float> %427, i32 0, i32 0, i32 0), !dbg !200
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !204
  tail call void @llvm.amdgcn.s.barrier(), !dbg !205
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !206
  %511 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %181) #7, !dbg !207, !srcloc !146
  %.sroa.036.2.extract.shift37.i289 = lshr i64 %511, 16, !dbg !210
  %512 = insertelement <2 x i64> poison, i64 %511, i64 0, !dbg !210
  %513 = insertelement <2 x i64> %512, i64 %.sroa.036.2.extract.shift37.i289, i64 1, !dbg !210
  %514 = shufflevector <2 x i64> %512, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %515 = lshr <2 x i64> %514, <i64 32, i64 48>, !dbg !207
  %516 = shufflevector <2 x i64> %513, <2 x i64> %515, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.34900.22.vec.insert9161365 = trunc <4 x i64> %516 to <4 x i16>, !dbg !211
  %517 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %185) #7, !dbg !207, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i298 = lshr i64 %517, 16, !dbg !210
  %518 = insertelement <2 x i64> poison, i64 %517, i64 0, !dbg !210
  %519 = insertelement <2 x i64> %518, i64 %.sroa.036.2.extract.shift37.1.i298, i64 1, !dbg !210
  %520 = shufflevector <2 x i64> %518, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %521 = lshr <2 x i64> %520, <i64 32, i64 48>, !dbg !207
  %522 = shufflevector <2 x i64> %519, <2 x i64> %521, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.50917.30.vec.insert9331364 = trunc <4 x i64> %522 to <4 x i16>, !dbg !211
  %523 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %189) #7, !dbg !213, !srcloc !146
  %.sroa.036.2.extract.shift37.i310 = lshr i64 %523, 16, !dbg !216
  %524 = insertelement <2 x i64> poison, i64 %523, i64 0, !dbg !216
  %525 = insertelement <2 x i64> %524, i64 %.sroa.036.2.extract.shift37.i310, i64 1, !dbg !216
  %526 = shufflevector <2 x i64> %524, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %527 = lshr <2 x i64> %526, <i64 32, i64 48>, !dbg !213
  %528 = shufflevector <2 x i64> %525, <2 x i64> %527, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.34.22.vec.insert8491363 = trunc <4 x i64> %528 to <4 x i16>, !dbg !217
  %529 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %193) #7, !dbg !213, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i319 = lshr i64 %529, 16, !dbg !216
  %530 = insertelement <2 x i64> poison, i64 %529, i64 0, !dbg !216
  %531 = insertelement <2 x i64> %530, i64 %.sroa.036.2.extract.shift37.1.i319, i64 1, !dbg !216
  %532 = shufflevector <2 x i64> %530, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %533 = lshr <2 x i64> %532, <i64 32, i64 48>, !dbg !213
  %534 = shufflevector <2 x i64> %531, <2 x i64> %533, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.50.30.vec.insert8651362 = trunc <4 x i64> %534 to <4 x i16>, !dbg !217
  %535 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %197) #7, !dbg !219, !srcloc !146
  %.sroa.036.2.extract.shift37.i331 = lshr i64 %535, 16, !dbg !222
  %536 = insertelement <2 x i64> poison, i64 %535, i64 0, !dbg !222
  %537 = insertelement <2 x i64> %536, i64 %.sroa.036.2.extract.shift37.i331, i64 1, !dbg !222
  %538 = shufflevector <2 x i64> %536, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !219
  %539 = lshr <2 x i64> %538, <i64 32, i64 48>, !dbg !219
  %540 = shufflevector <2 x i64> %537, <2 x i64> %539, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !223
  %.sroa.341036.22.vec.insert10521361 = trunc <4 x i64> %540 to <4 x i16>, !dbg !223
  %541 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %201) #7, !dbg !219, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i340 = lshr i64 %541, 16, !dbg !222
  %542 = insertelement <2 x i64> poison, i64 %541, i64 0, !dbg !222
  %543 = insertelement <2 x i64> %542, i64 %.sroa.036.2.extract.shift37.1.i340, i64 1, !dbg !222
  %544 = shufflevector <2 x i64> %542, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !219
  %545 = lshr <2 x i64> %544, <i64 32, i64 48>, !dbg !219
  %546 = shufflevector <2 x i64> %543, <2 x i64> %545, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !223
  %.sroa.501053.30.vec.insert10691360 = trunc <4 x i64> %546 to <4 x i16>, !dbg !223
  %547 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %205) #7, !dbg !225, !srcloc !146
  %.sroa.036.2.extract.shift37.i352 = lshr i64 %547, 16, !dbg !228
  %548 = insertelement <2 x i64> poison, i64 %547, i64 0, !dbg !228
  %549 = insertelement <2 x i64> %548, i64 %.sroa.036.2.extract.shift37.i352, i64 1, !dbg !228
  %550 = shufflevector <2 x i64> %548, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !225
  %551 = lshr <2 x i64> %550, <i64 32, i64 48>, !dbg !225
  %552 = shufflevector <2 x i64> %549, <2 x i64> %551, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !229
  %.sroa.34968.22.vec.insert9841359 = trunc <4 x i64> %552 to <4 x i16>, !dbg !229
  %553 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %209) #7, !dbg !225, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i361 = lshr i64 %553, 16, !dbg !228
  %554 = insertelement <2 x i64> poison, i64 %553, i64 0, !dbg !228
  %555 = insertelement <2 x i64> %554, i64 %.sroa.036.2.extract.shift37.1.i361, i64 1, !dbg !228
  %556 = shufflevector <2 x i64> %554, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !225
  %557 = lshr <2 x i64> %556, <i64 32, i64 48>, !dbg !225
  %558 = shufflevector <2 x i64> %555, <2 x i64> %557, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !229
  %.sroa.50985.30.vec.insert10011358 = trunc <4 x i64> %558 to <4 x i16>, !dbg !229
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !231
  br i1 %.not, label %570, label %559, !dbg !232

559:                                              ; preds = %494
  %560 = shl nuw nsw i32 %436, 6, !dbg !233
  %561 = zext nneg i32 %560 to i64, !dbg !237
  %562 = getelementptr %struct.__hip_bfloat16, ptr %415, i64 %561, !dbg !239
  %563 = ptrtoint ptr %562 to i64, !dbg !240
  %564 = bitcast i64 %563 to <2 x i32>, !dbg !240
  %.sroa.0.0.vec.expand.i.i369 = shufflevector <2 x i32> %564, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>, !dbg !240
  %565 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i369, i32 1114112, i64 3, !dbg !240
  %.sroa.0.12.vec.insert.i.i370 = insertelement <4 x i32> %565, i32 %210, i64 2, !dbg !240
  %566 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i370, i32 noundef %211, i32 noundef 0, i32 noundef 0) #7, !dbg !243
  %567 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i370, i32 noundef %212, i32 noundef 0, i32 noundef 0) #7, !dbg !243
  %568 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i370, i32 noundef %213, i32 noundef 0, i32 noundef 0) #7, !dbg !243
  %569 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i370, i32 noundef %214, i32 noundef 0, i32 noundef 0) #7, !dbg !243
  %extract.t1084 = trunc i128 %566 to i64, !dbg !245
  %extract1086 = lshr i128 %566, 64, !dbg !245
  %extract.t1087 = trunc nuw i128 %extract1086 to i64, !dbg !245
  %extract.t1097 = trunc i128 %567 to i64, !dbg !245
  %extract1099 = lshr i128 %567, 64, !dbg !245
  %extract.t1100 = trunc nuw i128 %extract1099 to i64, !dbg !245
  %extract.t1110 = trunc i128 %568 to i64, !dbg !245
  %extract1112 = lshr i128 %568, 64, !dbg !245
  %extract.t1113 = trunc nuw i128 %extract1112 to i64, !dbg !245
  %extract.t1123 = trunc i128 %569 to i64, !dbg !245
  %extract1125 = lshr i128 %569, 64, !dbg !245
  %extract.t1126 = trunc nuw i128 %extract1125 to i64, !dbg !245
  br label %570, !dbg !245

570:                                              ; preds = %559, %494
  %.sroa.0796.2.off0 = phi i64 [ %.sroa.0796.1.off01165, %494 ], [ %extract.t1084, %559 ]
  %.sroa.0796.2.off64 = phi i64 [ %.sroa.0796.1.off641166, %494 ], [ %extract.t1087, %559 ]
  %.sroa.5.2.off0 = phi i64 [ %.sroa.5.1.off01167, %494 ], [ %extract.t1097, %559 ]
  %.sroa.5.2.off64 = phi i64 [ %.sroa.5.1.off641168, %494 ], [ %extract.t1100, %559 ]
  %.sroa.8797.2.off0 = phi i64 [ %.sroa.8797.1.off01169, %494 ], [ %extract.t1110, %559 ]
  %.sroa.8797.2.off64 = phi i64 [ %.sroa.8797.1.off641170, %494 ], [ %extract.t1113, %559 ]
  %.sroa.11.2.off0 = phi i64 [ %.sroa.11.1.off01171, %494 ], [ %extract.t1123, %559 ]
  %.sroa.11.2.off64 = phi i64 [ %.sroa.11.1.off641172, %494 ], [ %extract.t1126, %559 ]
  tail call void @llvm.amdgcn.s.barrier(), !dbg !246
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !247
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !248, !srcloc !249
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !250
  %571 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert10521361, <4 x i16> %.sroa.34900.22.vec.insert9161365, <4 x float> %495, i32 0, i32 0, i32 0), !dbg !251
  %572 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert10521361, <4 x i16> %.sroa.50917.30.vec.insert9331364, <4 x float> %496, i32 0, i32 0, i32 0), !dbg !251
  %573 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert10691360, <4 x i16> %.sroa.34900.22.vec.insert9161365, <4 x float> %497, i32 0, i32 0, i32 0), !dbg !251
  %574 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert10691360, <4 x i16> %.sroa.50917.30.vec.insert9331364, <4 x float> %498, i32 0, i32 0, i32 0), !dbg !251
  %575 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert10521361, <4 x i16> %.sroa.34.22.vec.insert8491363, <4 x float> %499, i32 0, i32 0, i32 0), !dbg !255
  %576 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert10521361, <4 x i16> %.sroa.50.30.vec.insert8651362, <4 x float> %500, i32 0, i32 0, i32 0), !dbg !255
  %577 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert10691360, <4 x i16> %.sroa.34.22.vec.insert8491363, <4 x float> %501, i32 0, i32 0, i32 0), !dbg !255
  %578 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert10691360, <4 x i16> %.sroa.50.30.vec.insert8651362, <4 x float> %502, i32 0, i32 0, i32 0), !dbg !255
  %579 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert9841359, <4 x i16> %.sroa.34900.22.vec.insert9161365, <4 x float> %503, i32 0, i32 0, i32 0), !dbg !259
  %580 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert9841359, <4 x i16> %.sroa.50917.30.vec.insert9331364, <4 x float> %504, i32 0, i32 0, i32 0), !dbg !259
  %581 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert10011358, <4 x i16> %.sroa.34900.22.vec.insert9161365, <4 x float> %505, i32 0, i32 0, i32 0), !dbg !259
  %582 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert10011358, <4 x i16> %.sroa.50917.30.vec.insert9331364, <4 x float> %506, i32 0, i32 0, i32 0), !dbg !259
  %583 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert9841359, <4 x i16> %.sroa.34.22.vec.insert8491363, <4 x float> %507, i32 0, i32 0, i32 0), !dbg !263
  %584 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert9841359, <4 x i16> %.sroa.50.30.vec.insert8651362, <4 x float> %508, i32 0, i32 0, i32 0), !dbg !263
  %585 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert10011358, <4 x i16> %.sroa.34.22.vec.insert8491363, <4 x float> %509, i32 0, i32 0, i32 0), !dbg !263
  %586 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert10011358, <4 x i16> %.sroa.50.30.vec.insert8651362, <4 x float> %510, i32 0, i32 0, i32 0), !dbg !263
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !267
  tail call void @llvm.amdgcn.s.barrier(), !dbg !268
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !269
  %587 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %219) #7, !dbg !270, !srcloc !146
  %.sroa.036.2.extract.shift37.i426 = lshr i64 %587, 16, !dbg !273
  %588 = insertelement <2 x i64> poison, i64 %587, i64 0, !dbg !273
  %589 = insertelement <2 x i64> %588, i64 %.sroa.036.2.extract.shift37.i426, i64 1, !dbg !273
  %590 = shufflevector <2 x i64> %588, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !270
  %591 = lshr <2 x i64> %590, <i64 32, i64 48>, !dbg !270
  %592 = shufflevector <2 x i64> %589, <2 x i64> %591, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !274
  %.sroa.0866.6.vec.insert1357 = trunc <4 x i64> %592 to <4 x i16>, !dbg !274
  %593 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %223) #7, !dbg !270, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i435 = lshr i64 %593, 16, !dbg !273
  %594 = insertelement <2 x i64> poison, i64 %593, i64 0, !dbg !273
  %595 = insertelement <2 x i64> %594, i64 %.sroa.036.2.extract.shift37.1.i435, i64 1, !dbg !273
  %596 = shufflevector <2 x i64> %594, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !270
  %597 = lshr <2 x i64> %596, <i64 32, i64 48>, !dbg !270
  %598 = shufflevector <2 x i64> %595, <2 x i64> %597, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !274
  %.sroa.18883.14.vec.insert1356 = trunc <4 x i64> %598 to <4 x i16>, !dbg !274
  %599 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %227) #7, !dbg !276, !srcloc !146
  %.sroa.036.2.extract.shift37.i447 = lshr i64 %599, 16, !dbg !279
  %600 = insertelement <2 x i64> poison, i64 %599, i64 0, !dbg !279
  %601 = insertelement <2 x i64> %600, i64 %.sroa.036.2.extract.shift37.i447, i64 1, !dbg !279
  %602 = shufflevector <2 x i64> %600, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !276
  %603 = lshr <2 x i64> %602, <i64 32, i64 48>, !dbg !276
  %604 = shufflevector <2 x i64> %601, <2 x i64> %603, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !280
  %.sroa.0801.6.vec.insert1355 = trunc <4 x i64> %604 to <4 x i16>, !dbg !280
  %605 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %231) #7, !dbg !276, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i456 = lshr i64 %605, 16, !dbg !279
  %606 = insertelement <2 x i64> poison, i64 %605, i64 0, !dbg !279
  %607 = insertelement <2 x i64> %606, i64 %.sroa.036.2.extract.shift37.1.i456, i64 1, !dbg !279
  %608 = shufflevector <2 x i64> %606, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !276
  %609 = lshr <2 x i64> %608, <i64 32, i64 48>, !dbg !276
  %610 = shufflevector <2 x i64> %607, <2 x i64> %609, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !280
  %.sroa.18.14.vec.insert1354 = trunc <4 x i64> %610 to <4 x i16>, !dbg !280
  %611 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %235) #7, !dbg !282, !srcloc !146
  %.sroa.036.2.extract.shift37.i468 = lshr i64 %611, 16, !dbg !285
  %612 = insertelement <2 x i64> poison, i64 %611, i64 0, !dbg !285
  %613 = insertelement <2 x i64> %612, i64 %.sroa.036.2.extract.shift37.i468, i64 1, !dbg !285
  %614 = shufflevector <2 x i64> %612, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !282
  %615 = lshr <2 x i64> %614, <i64 32, i64 48>, !dbg !282
  %616 = shufflevector <2 x i64> %613, <2 x i64> %615, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !286
  %.sroa.01002.6.vec.insert1353 = trunc <4 x i64> %616 to <4 x i16>, !dbg !286
  %617 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %239) #7, !dbg !282, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i477 = lshr i64 %617, 16, !dbg !285
  %618 = insertelement <2 x i64> poison, i64 %617, i64 0, !dbg !285
  %619 = insertelement <2 x i64> %618, i64 %.sroa.036.2.extract.shift37.1.i477, i64 1, !dbg !285
  %620 = shufflevector <2 x i64> %618, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !282
  %621 = lshr <2 x i64> %620, <i64 32, i64 48>, !dbg !282
  %622 = shufflevector <2 x i64> %619, <2 x i64> %621, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !286
  %.sroa.181019.14.vec.insert1352 = trunc <4 x i64> %622 to <4 x i16>, !dbg !286
  %623 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %243) #7, !dbg !288, !srcloc !146
  %.sroa.036.2.extract.shift37.i489 = lshr i64 %623, 16, !dbg !291
  %624 = insertelement <2 x i64> poison, i64 %623, i64 0, !dbg !291
  %625 = insertelement <2 x i64> %624, i64 %.sroa.036.2.extract.shift37.i489, i64 1, !dbg !291
  %626 = shufflevector <2 x i64> %624, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !288
  %627 = lshr <2 x i64> %626, <i64 32, i64 48>, !dbg !288
  %628 = shufflevector <2 x i64> %625, <2 x i64> %627, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !292
  %.sroa.0934.6.vec.insert1351 = trunc <4 x i64> %628 to <4 x i16>, !dbg !292
  %629 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %247) #7, !dbg !288, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i498 = lshr i64 %629, 16, !dbg !291
  %630 = insertelement <2 x i64> poison, i64 %629, i64 0, !dbg !291
  %631 = insertelement <2 x i64> %630, i64 %.sroa.036.2.extract.shift37.1.i498, i64 1, !dbg !291
  %632 = shufflevector <2 x i64> %630, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !288
  %633 = lshr <2 x i64> %632, <i64 32, i64 48>, !dbg !288
  %634 = shufflevector <2 x i64> %631, <2 x i64> %633, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !292
  %.sroa.18951.14.vec.insert1350 = trunc <4 x i64> %634 to <4 x i16>, !dbg !292
  %635 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %252) #7, !dbg !294, !srcloc !146
  %.sroa.036.2.extract.shift37.i510 = lshr i64 %635, 16, !dbg !297
  %636 = insertelement <2 x i64> poison, i64 %635, i64 0, !dbg !297
  %637 = insertelement <2 x i64> %636, i64 %.sroa.036.2.extract.shift37.i510, i64 1, !dbg !297
  %638 = shufflevector <2 x i64> %636, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !294
  %639 = lshr <2 x i64> %638, <i64 32, i64 48>, !dbg !294
  %640 = shufflevector <2 x i64> %637, <2 x i64> %639, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !298
  %.sroa.34900.22.vec.insert1349 = trunc <4 x i64> %640 to <4 x i16>, !dbg !298
  %641 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %256) #7, !dbg !294, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i519 = lshr i64 %641, 16, !dbg !297
  %642 = insertelement <2 x i64> poison, i64 %641, i64 0, !dbg !297
  %643 = insertelement <2 x i64> %642, i64 %.sroa.036.2.extract.shift37.1.i519, i64 1, !dbg !297
  %644 = shufflevector <2 x i64> %642, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !294
  %645 = lshr <2 x i64> %644, <i64 32, i64 48>, !dbg !294
  %646 = shufflevector <2 x i64> %643, <2 x i64> %645, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !298
  %.sroa.50917.30.vec.insert1348 = trunc <4 x i64> %646 to <4 x i16>, !dbg !298
  %647 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %260) #7, !dbg !300, !srcloc !146
  %.sroa.036.2.extract.shift37.i531 = lshr i64 %647, 16, !dbg !303
  %648 = insertelement <2 x i64> poison, i64 %647, i64 0, !dbg !303
  %649 = insertelement <2 x i64> %648, i64 %.sroa.036.2.extract.shift37.i531, i64 1, !dbg !303
  %650 = shufflevector <2 x i64> %648, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !300
  %651 = lshr <2 x i64> %650, <i64 32, i64 48>, !dbg !300
  %652 = shufflevector <2 x i64> %649, <2 x i64> %651, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !304
  %.sroa.34.22.vec.insert1347 = trunc <4 x i64> %652 to <4 x i16>, !dbg !304
  %653 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %264) #7, !dbg !300, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i540 = lshr i64 %653, 16, !dbg !303
  %654 = insertelement <2 x i64> poison, i64 %653, i64 0, !dbg !303
  %655 = insertelement <2 x i64> %654, i64 %.sroa.036.2.extract.shift37.1.i540, i64 1, !dbg !303
  %656 = shufflevector <2 x i64> %654, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !300
  %657 = lshr <2 x i64> %656, <i64 32, i64 48>, !dbg !300
  %658 = shufflevector <2 x i64> %655, <2 x i64> %657, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !304
  %.sroa.50.30.vec.insert1346 = trunc <4 x i64> %658 to <4 x i16>, !dbg !304
  %659 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %268) #7, !dbg !306, !srcloc !146
  %.sroa.036.2.extract.shift37.i552 = lshr i64 %659, 16, !dbg !309
  %660 = insertelement <2 x i64> poison, i64 %659, i64 0, !dbg !309
  %661 = insertelement <2 x i64> %660, i64 %.sroa.036.2.extract.shift37.i552, i64 1, !dbg !309
  %662 = shufflevector <2 x i64> %660, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !306
  %663 = lshr <2 x i64> %662, <i64 32, i64 48>, !dbg !306
  %664 = shufflevector <2 x i64> %661, <2 x i64> %663, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !310
  %.sroa.341036.22.vec.insert1345 = trunc <4 x i64> %664 to <4 x i16>, !dbg !310
  %665 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %272) #7, !dbg !306, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i561 = lshr i64 %665, 16, !dbg !309
  %666 = insertelement <2 x i64> poison, i64 %665, i64 0, !dbg !309
  %667 = insertelement <2 x i64> %666, i64 %.sroa.036.2.extract.shift37.1.i561, i64 1, !dbg !309
  %668 = shufflevector <2 x i64> %666, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !306
  %669 = lshr <2 x i64> %668, <i64 32, i64 48>, !dbg !306
  %670 = shufflevector <2 x i64> %667, <2 x i64> %669, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !310
  %.sroa.501053.30.vec.insert1344 = trunc <4 x i64> %670 to <4 x i16>, !dbg !310
  %671 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %276) #7, !dbg !312, !srcloc !146
  %.sroa.036.2.extract.shift37.i573 = lshr i64 %671, 16, !dbg !315
  %672 = insertelement <2 x i64> poison, i64 %671, i64 0, !dbg !315
  %673 = insertelement <2 x i64> %672, i64 %.sroa.036.2.extract.shift37.i573, i64 1, !dbg !315
  %674 = shufflevector <2 x i64> %672, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !312
  %675 = lshr <2 x i64> %674, <i64 32, i64 48>, !dbg !312
  %676 = shufflevector <2 x i64> %673, <2 x i64> %675, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !316
  %.sroa.34968.22.vec.insert1343 = trunc <4 x i64> %676 to <4 x i16>, !dbg !316
  %677 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %280) #7, !dbg !312, !srcloc !146
  %.sroa.036.2.extract.shift37.1.i582 = lshr i64 %677, 16, !dbg !315
  %678 = insertelement <2 x i64> poison, i64 %677, i64 0, !dbg !315
  %679 = insertelement <2 x i64> %678, i64 %.sroa.036.2.extract.shift37.1.i582, i64 1, !dbg !315
  %680 = shufflevector <2 x i64> %678, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !312
  %681 = lshr <2 x i64> %680, <i64 32, i64 48>, !dbg !312
  %682 = shufflevector <2 x i64> %679, <2 x i64> %681, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !316
  %.sroa.50985.30.vec.insert1342 = trunc <4 x i64> %682 to <4 x i16>, !dbg !316
  tail call void @llvm.amdgcn.s.barrier(), !dbg !318
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !319
  tail call void asm sideeffect "s_waitcnt lgkmcnt(8)", ""() #7, !dbg !320, !srcloc !321
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !322
  %683 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert1353, <4 x i16> %.sroa.0866.6.vec.insert1357, <4 x float> %571, i32 0, i32 0, i32 0), !dbg !323
  %684 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert1353, <4 x i16> %.sroa.18883.14.vec.insert1356, <4 x float> %572, i32 0, i32 0, i32 0), !dbg !323
  %685 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert1352, <4 x i16> %.sroa.0866.6.vec.insert1357, <4 x float> %573, i32 0, i32 0, i32 0), !dbg !323
  %686 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert1352, <4 x i16> %.sroa.18883.14.vec.insert1356, <4 x float> %574, i32 0, i32 0, i32 0), !dbg !323
  %687 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert1353, <4 x i16> %.sroa.0801.6.vec.insert1355, <4 x float> %575, i32 0, i32 0, i32 0), !dbg !327
  %688 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01002.6.vec.insert1353, <4 x i16> %.sroa.18.14.vec.insert1354, <4 x float> %576, i32 0, i32 0, i32 0), !dbg !327
  %689 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert1352, <4 x i16> %.sroa.0801.6.vec.insert1355, <4 x float> %577, i32 0, i32 0, i32 0), !dbg !327
  %690 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181019.14.vec.insert1352, <4 x i16> %.sroa.18.14.vec.insert1354, <4 x float> %578, i32 0, i32 0, i32 0), !dbg !327
  %691 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert1351, <4 x i16> %.sroa.0866.6.vec.insert1357, <4 x float> %579, i32 0, i32 0, i32 0), !dbg !331
  %692 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert1351, <4 x i16> %.sroa.18883.14.vec.insert1356, <4 x float> %580, i32 0, i32 0, i32 0), !dbg !331
  %693 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert1350, <4 x i16> %.sroa.0866.6.vec.insert1357, <4 x float> %581, i32 0, i32 0, i32 0), !dbg !331
  %694 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert1350, <4 x i16> %.sroa.18883.14.vec.insert1356, <4 x float> %582, i32 0, i32 0, i32 0), !dbg !331
  %695 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert1351, <4 x i16> %.sroa.0801.6.vec.insert1355, <4 x float> %583, i32 0, i32 0, i32 0), !dbg !335
  %696 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0934.6.vec.insert1351, <4 x i16> %.sroa.18.14.vec.insert1354, <4 x float> %584, i32 0, i32 0, i32 0), !dbg !335
  %697 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert1350, <4 x i16> %.sroa.0801.6.vec.insert1355, <4 x float> %585, i32 0, i32 0, i32 0), !dbg !335
  %698 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18951.14.vec.insert1350, <4 x i16> %.sroa.18.14.vec.insert1354, <4 x float> %586, i32 0, i32 0, i32 0), !dbg !335
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !339
  tail call void @llvm.amdgcn.s.barrier(), !dbg !340
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !341
  br i1 %.not, label %700, label %699, !dbg !342

699:                                              ; preds = %570
  tail call void asm sideeffect "s_waitcnt vmcnt(4)", ""() #7, !dbg !343, !srcloc !344
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %288, i64 %.sroa.0798.2.off0) #7, !dbg !345, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %292, i64 %.sroa.0798.2.off64) #7, !dbg !351, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %296, i64 %.sroa.5799.2.off0) #7, !dbg !345, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %300, i64 %.sroa.5799.2.off64) #7, !dbg !351, !srcloc !350
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !353, !srcloc !354
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %306, i64 %.sroa.0796.2.off0) #7, !dbg !355, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %310, i64 %.sroa.0796.2.off64) #7, !dbg !359, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %314, i64 %.sroa.5.2.off0) #7, !dbg !355, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %318, i64 %.sroa.5.2.off64) #7, !dbg !359, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %322, i64 %.sroa.8797.2.off0) #7, !dbg !355, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %326, i64 %.sroa.8797.2.off64) #7, !dbg !359, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %330, i64 %.sroa.11.2.off0) #7, !dbg !355, !srcloc !350
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %334, i64 %.sroa.11.2.off64) #7, !dbg !359, !srcloc !350
  br label %700, !dbg !361

700:                                              ; preds = %699, %570
  tail call void @llvm.amdgcn.s.barrier(), !dbg !362
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !363
  tail call void asm sideeffect "s_waitcnt lgkmcnt(12)", ""() #7, !dbg !364, !srcloc !365
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !366
  %701 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert1345, <4 x i16> %.sroa.34900.22.vec.insert1349, <4 x float> %683, i32 0, i32 0, i32 0), !dbg !367
  %702 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert1345, <4 x i16> %.sroa.50917.30.vec.insert1348, <4 x float> %684, i32 0, i32 0, i32 0), !dbg !367
  %703 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert1344, <4 x i16> %.sroa.34900.22.vec.insert1349, <4 x float> %685, i32 0, i32 0, i32 0), !dbg !367
  %704 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert1344, <4 x i16> %.sroa.50917.30.vec.insert1348, <4 x float> %686, i32 0, i32 0, i32 0), !dbg !367
  %705 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert1345, <4 x i16> %.sroa.34.22.vec.insert1347, <4 x float> %687, i32 0, i32 0, i32 0), !dbg !371
  %706 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341036.22.vec.insert1345, <4 x i16> %.sroa.50.30.vec.insert1346, <4 x float> %688, i32 0, i32 0, i32 0), !dbg !371
  %707 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert1344, <4 x i16> %.sroa.34.22.vec.insert1347, <4 x float> %689, i32 0, i32 0, i32 0), !dbg !371
  %708 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501053.30.vec.insert1344, <4 x i16> %.sroa.50.30.vec.insert1346, <4 x float> %690, i32 0, i32 0, i32 0), !dbg !371
  %709 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert1343, <4 x i16> %.sroa.34900.22.vec.insert1349, <4 x float> %691, i32 0, i32 0, i32 0), !dbg !375
  %710 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert1343, <4 x i16> %.sroa.50917.30.vec.insert1348, <4 x float> %692, i32 0, i32 0, i32 0), !dbg !375
  %711 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert1342, <4 x i16> %.sroa.34900.22.vec.insert1349, <4 x float> %693, i32 0, i32 0, i32 0), !dbg !375
  %712 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert1342, <4 x i16> %.sroa.50917.30.vec.insert1348, <4 x float> %694, i32 0, i32 0, i32 0), !dbg !375
  %713 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert1343, <4 x i16> %.sroa.34.22.vec.insert1347, <4 x float> %695, i32 0, i32 0, i32 0), !dbg !379
  %714 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34968.22.vec.insert1343, <4 x i16> %.sroa.50.30.vec.insert1346, <4 x float> %696, i32 0, i32 0, i32 0), !dbg !379
  %715 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert1342, <4 x i16> %.sroa.34.22.vec.insert1347, <4 x float> %697, i32 0, i32 0, i32 0), !dbg !379
  %716 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50985.30.vec.insert1342, <4 x i16> %.sroa.50.30.vec.insert1346, <4 x float> %698, i32 0, i32 0, i32 0), !dbg !379
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !383
  tail call void @llvm.amdgcn.s.barrier(), !dbg !384
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !385
  %exitcond.not = icmp eq i32 %436, 128, !dbg !386
  br i1 %exitcond.not, label %425, label %426, !dbg !134, !llvm.loop !387

717:                                              ; preds = %425
  tail call void @llvm.amdgcn.s.barrier(), !dbg !390
  br label %718, !dbg !391

718:                                              ; preds = %717, %425
  %719 = shl nsw i32 %.sroa.0.0.vec.extract.i, 2
  %720 = or disjoint i32 %719, %14
  %721 = shl nsw i32 %.sroa.0.4.vec.extract.i, 3
  %722 = or disjoint i32 %721, %15, !dbg !392
  %723 = shl nsw i32 %720, 5, !dbg !393
  %724 = shl nsw i32 %722, 5, !dbg !401
  %725 = sext i32 %723 to i64, !dbg !402
  %726 = mul i64 %.sroa.20.0.copyload, %725, !dbg !405
  %727 = sext i32 %724 to i64, !dbg !406
  %728 = getelementptr float, ptr %.sroa.171081.0.copyload, i64 %726, !dbg !407
  %729 = getelementptr float, ptr %728, i64 %727, !dbg !407
  %.sroa.0.0.vec.extract = extractelement <4 x float> %701, i64 0, !dbg !408
  %730 = getelementptr inbounds float, ptr %729, i64 %345, !dbg !409
  store float %.sroa.0.0.vec.extract, ptr %730, align 4, !dbg !410, !tbaa !411
  %.sroa.0.4.vec.extract = extractelement <4 x float> %701, i64 1, !dbg !415
  %731 = getelementptr inbounds float, ptr %729, i64 %347, !dbg !416
  store float %.sroa.0.4.vec.extract, ptr %731, align 4, !dbg !417, !tbaa !411
  %.sroa.0.8.vec.extract = extractelement <4 x float> %701, i64 2, !dbg !418
  %732 = getelementptr inbounds float, ptr %729, i64 %349, !dbg !419
  store float %.sroa.0.8.vec.extract, ptr %732, align 4, !dbg !420, !tbaa !411
  %.sroa.0.12.vec.extract = extractelement <4 x float> %701, i64 3, !dbg !421
  %733 = getelementptr inbounds float, ptr %729, i64 %351, !dbg !422
  store float %.sroa.0.12.vec.extract, ptr %733, align 4, !dbg !423, !tbaa !411
  %.sroa.12.16.vec.extract = extractelement <4 x float> %702, i64 0, !dbg !408
  %734 = getelementptr inbounds float, ptr %729, i64 %353, !dbg !409
  store float %.sroa.12.16.vec.extract, ptr %734, align 4, !dbg !410, !tbaa !411
  %.sroa.12.20.vec.extract = extractelement <4 x float> %702, i64 1, !dbg !415
  %735 = getelementptr inbounds float, ptr %729, i64 %355, !dbg !416
  store float %.sroa.12.20.vec.extract, ptr %735, align 4, !dbg !417, !tbaa !411
  %.sroa.12.24.vec.extract = extractelement <4 x float> %702, i64 2, !dbg !418
  %736 = getelementptr inbounds float, ptr %729, i64 %357, !dbg !419
  store float %.sroa.12.24.vec.extract, ptr %736, align 4, !dbg !420, !tbaa !411
  %.sroa.12.28.vec.extract = extractelement <4 x float> %702, i64 3, !dbg !421
  %737 = getelementptr inbounds float, ptr %729, i64 %359, !dbg !422
  store float %.sroa.12.28.vec.extract, ptr %737, align 4, !dbg !423, !tbaa !411
  %.sroa.22.32.vec.extract = extractelement <4 x float> %703, i64 0, !dbg !408
  %738 = getelementptr inbounds float, ptr %729, i64 %369, !dbg !409
  store float %.sroa.22.32.vec.extract, ptr %738, align 4, !dbg !410, !tbaa !411
  %.sroa.22.36.vec.extract = extractelement <4 x float> %703, i64 1, !dbg !415
  %739 = getelementptr inbounds float, ptr %729, i64 %371, !dbg !416
  store float %.sroa.22.36.vec.extract, ptr %739, align 4, !dbg !417, !tbaa !411
  %.sroa.22.40.vec.extract = extractelement <4 x float> %703, i64 2, !dbg !418
  %740 = getelementptr inbounds float, ptr %729, i64 %373, !dbg !419
  store float %.sroa.22.40.vec.extract, ptr %740, align 4, !dbg !420, !tbaa !411
  %.sroa.22.44.vec.extract = extractelement <4 x float> %703, i64 3, !dbg !421
  %741 = getelementptr inbounds float, ptr %729, i64 %375, !dbg !422
  store float %.sroa.22.44.vec.extract, ptr %741, align 4, !dbg !423, !tbaa !411
  %.sroa.32.48.vec.extract = extractelement <4 x float> %704, i64 0, !dbg !408
  %742 = getelementptr inbounds float, ptr %729, i64 %377, !dbg !409
  store float %.sroa.32.48.vec.extract, ptr %742, align 4, !dbg !410, !tbaa !411
  %.sroa.32.52.vec.extract = extractelement <4 x float> %704, i64 1, !dbg !415
  %743 = getelementptr inbounds float, ptr %729, i64 %379, !dbg !416
  store float %.sroa.32.52.vec.extract, ptr %743, align 4, !dbg !417, !tbaa !411
  %.sroa.32.56.vec.extract = extractelement <4 x float> %704, i64 2, !dbg !418
  %744 = getelementptr inbounds float, ptr %729, i64 %381, !dbg !419
  store float %.sroa.32.56.vec.extract, ptr %744, align 4, !dbg !420, !tbaa !411
  %.sroa.32.60.vec.extract = extractelement <4 x float> %704, i64 3, !dbg !421
  %745 = getelementptr inbounds float, ptr %729, i64 %383, !dbg !422
  store float %.sroa.32.60.vec.extract, ptr %745, align 4, !dbg !423, !tbaa !411
  %746 = or disjoint i32 %721, %15, !dbg !392
  %747 = shl i32 %746, 5, !dbg !401
  %748 = or disjoint i32 %747, 128, !dbg !401
  %749 = sext i32 %748 to i64, !dbg !406
  %750 = getelementptr float, ptr %728, i64 %749, !dbg !407
  %.sroa.42.64.vec.extract = extractelement <4 x float> %705, i64 0, !dbg !408
  %751 = getelementptr inbounds float, ptr %750, i64 %345, !dbg !409
  store float %.sroa.42.64.vec.extract, ptr %751, align 4, !dbg !410, !tbaa !411
  %.sroa.42.68.vec.extract = extractelement <4 x float> %705, i64 1, !dbg !415
  %752 = getelementptr inbounds float, ptr %750, i64 %347, !dbg !416
  store float %.sroa.42.68.vec.extract, ptr %752, align 4, !dbg !417, !tbaa !411
  %.sroa.42.72.vec.extract = extractelement <4 x float> %705, i64 2, !dbg !418
  %753 = getelementptr inbounds float, ptr %750, i64 %349, !dbg !419
  store float %.sroa.42.72.vec.extract, ptr %753, align 4, !dbg !420, !tbaa !411
  %.sroa.42.76.vec.extract = extractelement <4 x float> %705, i64 3, !dbg !421
  %754 = getelementptr inbounds float, ptr %750, i64 %351, !dbg !422
  store float %.sroa.42.76.vec.extract, ptr %754, align 4, !dbg !423, !tbaa !411
  %.sroa.52.80.vec.extract = extractelement <4 x float> %706, i64 0, !dbg !408
  %755 = getelementptr inbounds float, ptr %750, i64 %353, !dbg !409
  store float %.sroa.52.80.vec.extract, ptr %755, align 4, !dbg !410, !tbaa !411
  %.sroa.52.84.vec.extract = extractelement <4 x float> %706, i64 1, !dbg !415
  %756 = getelementptr inbounds float, ptr %750, i64 %355, !dbg !416
  store float %.sroa.52.84.vec.extract, ptr %756, align 4, !dbg !417, !tbaa !411
  %.sroa.52.88.vec.extract = extractelement <4 x float> %706, i64 2, !dbg !418
  %757 = getelementptr inbounds float, ptr %750, i64 %357, !dbg !419
  store float %.sroa.52.88.vec.extract, ptr %757, align 4, !dbg !420, !tbaa !411
  %.sroa.52.92.vec.extract = extractelement <4 x float> %706, i64 3, !dbg !421
  %758 = getelementptr inbounds float, ptr %750, i64 %359, !dbg !422
  store float %.sroa.52.92.vec.extract, ptr %758, align 4, !dbg !423, !tbaa !411
  %.sroa.62.96.vec.extract = extractelement <4 x float> %707, i64 0, !dbg !408
  %759 = getelementptr inbounds float, ptr %750, i64 %369, !dbg !409
  store float %.sroa.62.96.vec.extract, ptr %759, align 4, !dbg !410, !tbaa !411
  %.sroa.62.100.vec.extract = extractelement <4 x float> %707, i64 1, !dbg !415
  %760 = getelementptr inbounds float, ptr %750, i64 %371, !dbg !416
  store float %.sroa.62.100.vec.extract, ptr %760, align 4, !dbg !417, !tbaa !411
  %.sroa.62.104.vec.extract = extractelement <4 x float> %707, i64 2, !dbg !418
  %761 = getelementptr inbounds float, ptr %750, i64 %373, !dbg !419
  store float %.sroa.62.104.vec.extract, ptr %761, align 4, !dbg !420, !tbaa !411
  %.sroa.62.108.vec.extract = extractelement <4 x float> %707, i64 3, !dbg !421
  %762 = getelementptr inbounds float, ptr %750, i64 %375, !dbg !422
  store float %.sroa.62.108.vec.extract, ptr %762, align 4, !dbg !423, !tbaa !411
  %.sroa.72.112.vec.extract = extractelement <4 x float> %708, i64 0, !dbg !408
  %763 = getelementptr inbounds float, ptr %750, i64 %377, !dbg !409
  store float %.sroa.72.112.vec.extract, ptr %763, align 4, !dbg !410, !tbaa !411
  %.sroa.72.116.vec.extract = extractelement <4 x float> %708, i64 1, !dbg !415
  %764 = getelementptr inbounds float, ptr %750, i64 %379, !dbg !416
  store float %.sroa.72.116.vec.extract, ptr %764, align 4, !dbg !417, !tbaa !411
  %.sroa.72.120.vec.extract = extractelement <4 x float> %708, i64 2, !dbg !418
  %765 = getelementptr inbounds float, ptr %750, i64 %381, !dbg !419
  store float %.sroa.72.120.vec.extract, ptr %765, align 4, !dbg !420, !tbaa !411
  %.sroa.72.124.vec.extract = extractelement <4 x float> %708, i64 3, !dbg !421
  %766 = getelementptr inbounds float, ptr %750, i64 %383, !dbg !422
  store float %.sroa.72.124.vec.extract, ptr %766, align 4, !dbg !423, !tbaa !411
  %767 = shl i32 %720, 5, !dbg !393
  %768 = add i32 %767, 64, !dbg !393
  %769 = sext i32 %768 to i64, !dbg !402
  %770 = mul i64 %.sroa.20.0.copyload, %769, !dbg !405
  %771 = getelementptr float, ptr %.sroa.171081.0.copyload, i64 %770, !dbg !407
  %772 = getelementptr float, ptr %771, i64 %727, !dbg !407
  %.sroa.82.128.vec.extract = extractelement <4 x float> %709, i64 0, !dbg !408
  %773 = getelementptr inbounds float, ptr %772, i64 %345, !dbg !409
  store float %.sroa.82.128.vec.extract, ptr %773, align 4, !dbg !410, !tbaa !411
  %.sroa.82.132.vec.extract = extractelement <4 x float> %709, i64 1, !dbg !415
  %774 = getelementptr inbounds float, ptr %772, i64 %347, !dbg !416
  store float %.sroa.82.132.vec.extract, ptr %774, align 4, !dbg !417, !tbaa !411
  %.sroa.82.136.vec.extract = extractelement <4 x float> %709, i64 2, !dbg !418
  %775 = getelementptr inbounds float, ptr %772, i64 %349, !dbg !419
  store float %.sroa.82.136.vec.extract, ptr %775, align 4, !dbg !420, !tbaa !411
  %.sroa.82.140.vec.extract = extractelement <4 x float> %709, i64 3, !dbg !421
  %776 = getelementptr inbounds float, ptr %772, i64 %351, !dbg !422
  store float %.sroa.82.140.vec.extract, ptr %776, align 4, !dbg !423, !tbaa !411
  %.sroa.92.144.vec.extract = extractelement <4 x float> %710, i64 0, !dbg !408
  %777 = getelementptr inbounds float, ptr %772, i64 %353, !dbg !409
  store float %.sroa.92.144.vec.extract, ptr %777, align 4, !dbg !410, !tbaa !411
  %.sroa.92.148.vec.extract = extractelement <4 x float> %710, i64 1, !dbg !415
  %778 = getelementptr inbounds float, ptr %772, i64 %355, !dbg !416
  store float %.sroa.92.148.vec.extract, ptr %778, align 4, !dbg !417, !tbaa !411
  %.sroa.92.152.vec.extract = extractelement <4 x float> %710, i64 2, !dbg !418
  %779 = getelementptr inbounds float, ptr %772, i64 %357, !dbg !419
  store float %.sroa.92.152.vec.extract, ptr %779, align 4, !dbg !420, !tbaa !411
  %.sroa.92.156.vec.extract = extractelement <4 x float> %710, i64 3, !dbg !421
  %780 = getelementptr inbounds float, ptr %772, i64 %359, !dbg !422
  store float %.sroa.92.156.vec.extract, ptr %780, align 4, !dbg !423, !tbaa !411
  %.sroa.102.160.vec.extract = extractelement <4 x float> %711, i64 0, !dbg !408
  %781 = getelementptr inbounds float, ptr %772, i64 %369, !dbg !409
  store float %.sroa.102.160.vec.extract, ptr %781, align 4, !dbg !410, !tbaa !411
  %.sroa.102.164.vec.extract = extractelement <4 x float> %711, i64 1, !dbg !415
  %782 = getelementptr inbounds float, ptr %772, i64 %371, !dbg !416
  store float %.sroa.102.164.vec.extract, ptr %782, align 4, !dbg !417, !tbaa !411
  %.sroa.102.168.vec.extract = extractelement <4 x float> %711, i64 2, !dbg !418
  %783 = getelementptr inbounds float, ptr %772, i64 %373, !dbg !419
  store float %.sroa.102.168.vec.extract, ptr %783, align 4, !dbg !420, !tbaa !411
  %.sroa.102.172.vec.extract = extractelement <4 x float> %711, i64 3, !dbg !421
  %784 = getelementptr inbounds float, ptr %772, i64 %375, !dbg !422
  store float %.sroa.102.172.vec.extract, ptr %784, align 4, !dbg !423, !tbaa !411
  %.sroa.112.176.vec.extract = extractelement <4 x float> %712, i64 0, !dbg !408
  %785 = getelementptr inbounds float, ptr %772, i64 %377, !dbg !409
  store float %.sroa.112.176.vec.extract, ptr %785, align 4, !dbg !410, !tbaa !411
  %.sroa.112.180.vec.extract = extractelement <4 x float> %712, i64 1, !dbg !415
  %786 = getelementptr inbounds float, ptr %772, i64 %379, !dbg !416
  store float %.sroa.112.180.vec.extract, ptr %786, align 4, !dbg !417, !tbaa !411
  %.sroa.112.184.vec.extract = extractelement <4 x float> %712, i64 2, !dbg !418
  %787 = getelementptr inbounds float, ptr %772, i64 %381, !dbg !419
  store float %.sroa.112.184.vec.extract, ptr %787, align 4, !dbg !420, !tbaa !411
  %.sroa.112.188.vec.extract = extractelement <4 x float> %712, i64 3, !dbg !421
  %788 = getelementptr inbounds float, ptr %772, i64 %383, !dbg !422
  store float %.sroa.112.188.vec.extract, ptr %788, align 4, !dbg !423, !tbaa !411
  %789 = getelementptr float, ptr %771, i64 %749, !dbg !407
  %.sroa.122.192.vec.extract = extractelement <4 x float> %713, i64 0, !dbg !408
  %790 = getelementptr inbounds float, ptr %789, i64 %345, !dbg !409
  store float %.sroa.122.192.vec.extract, ptr %790, align 4, !dbg !410, !tbaa !411
  %.sroa.122.196.vec.extract = extractelement <4 x float> %713, i64 1, !dbg !415
  %791 = getelementptr inbounds float, ptr %789, i64 %347, !dbg !416
  store float %.sroa.122.196.vec.extract, ptr %791, align 4, !dbg !417, !tbaa !411
  %.sroa.122.200.vec.extract = extractelement <4 x float> %713, i64 2, !dbg !418
  %792 = getelementptr inbounds float, ptr %789, i64 %349, !dbg !419
  store float %.sroa.122.200.vec.extract, ptr %792, align 4, !dbg !420, !tbaa !411
  %.sroa.122.204.vec.extract = extractelement <4 x float> %713, i64 3, !dbg !421
  %793 = getelementptr inbounds float, ptr %789, i64 %351, !dbg !422
  store float %.sroa.122.204.vec.extract, ptr %793, align 4, !dbg !423, !tbaa !411
  %.sroa.132.208.vec.extract = extractelement <4 x float> %714, i64 0, !dbg !408
  %794 = getelementptr inbounds float, ptr %789, i64 %353, !dbg !409
  store float %.sroa.132.208.vec.extract, ptr %794, align 4, !dbg !410, !tbaa !411
  %.sroa.132.212.vec.extract = extractelement <4 x float> %714, i64 1, !dbg !415
  %795 = getelementptr inbounds float, ptr %789, i64 %355, !dbg !416
  store float %.sroa.132.212.vec.extract, ptr %795, align 4, !dbg !417, !tbaa !411
  %.sroa.132.216.vec.extract = extractelement <4 x float> %714, i64 2, !dbg !418
  %796 = getelementptr inbounds float, ptr %789, i64 %357, !dbg !419
  store float %.sroa.132.216.vec.extract, ptr %796, align 4, !dbg !420, !tbaa !411
  %.sroa.132.220.vec.extract = extractelement <4 x float> %714, i64 3, !dbg !421
  %797 = getelementptr inbounds float, ptr %789, i64 %359, !dbg !422
  store float %.sroa.132.220.vec.extract, ptr %797, align 4, !dbg !423, !tbaa !411
  %.sroa.142.224.vec.extract = extractelement <4 x float> %715, i64 0, !dbg !408
  %798 = getelementptr inbounds float, ptr %789, i64 %369, !dbg !409
  store float %.sroa.142.224.vec.extract, ptr %798, align 4, !dbg !410, !tbaa !411
  %.sroa.142.228.vec.extract = extractelement <4 x float> %715, i64 1, !dbg !415
  %799 = getelementptr inbounds float, ptr %789, i64 %371, !dbg !416
  store float %.sroa.142.228.vec.extract, ptr %799, align 4, !dbg !417, !tbaa !411
  %.sroa.142.232.vec.extract = extractelement <4 x float> %715, i64 2, !dbg !418
  %800 = getelementptr inbounds float, ptr %789, i64 %373, !dbg !419
  store float %.sroa.142.232.vec.extract, ptr %800, align 4, !dbg !420, !tbaa !411
  %.sroa.142.236.vec.extract = extractelement <4 x float> %715, i64 3, !dbg !421
  %801 = getelementptr inbounds float, ptr %789, i64 %375, !dbg !422
  store float %.sroa.142.236.vec.extract, ptr %801, align 4, !dbg !423, !tbaa !411
  %.sroa.152.240.vec.extract = extractelement <4 x float> %716, i64 0, !dbg !408
  %802 = getelementptr inbounds float, ptr %789, i64 %377, !dbg !409
  store float %.sroa.152.240.vec.extract, ptr %802, align 4, !dbg !410, !tbaa !411
  %.sroa.152.244.vec.extract = extractelement <4 x float> %716, i64 1, !dbg !415
  %803 = getelementptr inbounds float, ptr %789, i64 %379, !dbg !416
  store float %.sroa.152.244.vec.extract, ptr %803, align 4, !dbg !417, !tbaa !411
  %.sroa.152.248.vec.extract = extractelement <4 x float> %716, i64 2, !dbg !418
  %804 = getelementptr inbounds float, ptr %789, i64 %381, !dbg !419
  store float %.sroa.152.248.vec.extract, ptr %804, align 4, !dbg !420, !tbaa !411
  %.sroa.152.252.vec.extract = extractelement <4 x float> %716, i64 3, !dbg !421
  %805 = getelementptr inbounds float, ptr %789, i64 %383, !dbg !422
  store float %.sroa.152.252.vec.extract, ptr %805, align 4, !dbg !423, !tbaa !411
  %806 = add nuw nsw i32 %.0, 1, !dbg !424
  br label %384, !dbg !425, !llvm.loop !426

.critedge:                                        ; preds = %_Z12get_task_idxi.exit
  ret void, !dbg !428
}

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setprio(i16 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(read)
declare i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32>, i32, i32, i32 immarg) #3

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #6

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,512" "amdgpu-waves-per-eu"="2" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(read) }
attributes #4 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!opencl.ocl.version = !{!7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "256_128_64_16_persist.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_16_persist")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"}
!9 = distinct !DISubprogram(name: "micro_tk", scope: !1, file: !1, line: 75, type: !10, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !15)
!13 = distinct !DISubprogram(name: "align_ptr<16>", scope: !14, file: !14, line: 228, type: !10, scopeLine: 228, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DIFile(filename: "ThunderKittens-HIP/include/common/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!15 = distinct !DILocation(line: 252, column: 13, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 128, 64> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DILocation(line: 78, column: 44, scope: !9)
!18 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !15)
!19 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !15)
!20 = !DILocation(line: 255, column: 17, scope: !16, inlinedAt: !17)
!21 = !DILocation(line: 230, column: 30, scope: !13, inlinedAt: !22)
!22 = distinct !DILocation(line: 252, column: 13, scope: !23, inlinedAt: !24)
!23 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 256, 64> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = distinct !DILocation(line: 79, column: 44, scope: !9)
!25 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !22)
!26 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !22)
!27 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !22)
!28 = !DILocation(line: 265, column: 117, scope: !29, inlinedAt: !31)
!29 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !30, file: !30, line: 265, type: !10, scopeLine: 265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!30 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h", directory: "")
!31 = distinct !DILocation(line: 291, column: 160, scope: !32, inlinedAt: !33)
!32 = distinct !DISubprogram(name: "__get_x", scope: !30, file: !30, line: 291, type: !10, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!33 = distinct !DILocation(line: 57, column: 85, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "warpid", scope: !14, file: !14, line: 57, type: !10, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!35 = distinct !DILocation(line: 81, column: 25, scope: !9)
!36 = !DILocation(line: 57, column: 97, scope: !34, inlinedAt: !35)
!37 = !DILocation(line: 82, column: 34, scope: !9)
!38 = !DILocation(line: 83, column: 34, scope: !9)
!39 = !{!40, !40, i64 0}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C/C++ TBAA"}
!43 = !DILocation(line: 87, column: 9, scope: !9)
!44 = !DILocation(line: 87, scope: !9)
!45 = !DILocation(line: 89, column: 46, scope: !9)
!46 = !DILocation(line: 89, column: 34, scope: !9)
!47 = !DILocation(line: 44, column: 17, scope: !48, inlinedAt: !49)
!48 = distinct !DISubprogram(name: "get_task_idx", scope: !1, file: !1, line: 40, type: !10, scopeLine: 40, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!49 = distinct !DILocation(line: 90, column: 23, scope: !9)
!50 = !DILocation(line: 44, column: 9, scope: !48, inlinedAt: !49)
!51 = !DILocation(line: 52, column: 61, scope: !48, inlinedAt: !49)
!52 = !DILocation(line: 52, column: 53, scope: !48, inlinedAt: !49)
!53 = !DILocation(line: 60, column: 25, scope: !48, inlinedAt: !49)
!54 = !DILocation(line: 61, column: 32, scope: !48, inlinedAt: !49)
!55 = !DILocation(line: 62, column: 38, scope: !48, inlinedAt: !49)
!56 = !DILocation(line: 1321, column: 10, scope: !57, inlinedAt: !59)
!57 = distinct !DISubprogram(name: "min", scope: !58, file: !58, line: 1320, type: !10, scopeLine: 1320, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!58 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/lib/clang/19/include/__clang_hip_math.h", directory: "")
!59 = distinct !DILocation(line: 62, column: 24, scope: !48, inlinedAt: !49)
!60 = !DILocation(line: 63, column: 38, scope: !48, inlinedAt: !49)
!61 = !DILocation(line: 63, column: 59, scope: !48, inlinedAt: !49)
!62 = !DILocation(line: 64, column: 44, scope: !48, inlinedAt: !49)
!63 = !DILocation(line: 63, column: 29, scope: !48, inlinedAt: !49)
!64 = !DILocation(line: 204, column: 52, scope: !65, inlinedAt: !67)
!65 = distinct !DISubprogram(name: "HIP_vector_base", scope: !66, file: !66, line: 204, type: !10, scopeLine: 204, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!66 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_vector_types.h", directory: "")
!67 = distinct !DILocation(line: 467, column: 15, scope: !68, inlinedAt: !69)
!68 = distinct !DISubprogram(name: "HIP_vector_type<int, int, nullptr>", scope: !66, file: !66, line: 466, type: !10, scopeLine: 468, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!69 = distinct !DILocation(line: 70, column: 12, scope: !48, inlinedAt: !49)
!70 = !DILocation(line: 204, column: 48, scope: !65, inlinedAt: !71)
!71 = distinct !DILocation(line: 467, column: 15, scope: !68, inlinedAt: !72)
!72 = distinct !DILocation(line: 0, scope: !48, inlinedAt: !49)
!73 = !DILocation(line: 71, column: 1, scope: !48, inlinedAt: !49)
!74 = !DILocation(line: 91, column: 21, scope: !9)
!75 = !DILocation(line: 91, column: 12, scope: !9)
!76 = !DILocation(line: 60, column: 34, scope: !77, inlinedAt: !79)
!77 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !78, file: !78, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!78 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!79 = distinct !DILocation(line: 37, column: 39, scope: !80, inlinedAt: !82)
!80 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !81, file: !81, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!81 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!82 = distinct !DILocation(line: 12, column: 5, scope: !83, inlinedAt: !85)
!83 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> > >", scope: !84, file: !84, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!84 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!85 = distinct !DILocation(line: 106, column: 9, scope: !9)
!86 = !DILocation(line: 63, column: 54, scope: !87, inlinedAt: !89)
!87 = distinct !DISubprogram(name: "operator[]", scope: !88, file: !88, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!88 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/gl.cuh", directory: "/workdir/AMD-benchmarking-harness")
!89 = distinct !DILocation(line: 38, column: 57, scope: !80, inlinedAt: !82)
!90 = !DILocation(line: 63, column: 60, scope: !87, inlinedAt: !89)
!91 = !DILocation(line: 63, column: 16, scope: !87, inlinedAt: !89)
!92 = !DILocation(line: 58, column: 68, scope: !80, inlinedAt: !82)
!93 = !DILocation(line: 92, column: 5, scope: !94, inlinedAt: !96)
!94 = distinct !DISubprogram(name: "load_global_vec4_async", scope: !95, file: !95, line: 89, type: !10, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!95 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/util/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!96 = distinct !DILocation(line: 58, column: 26, scope: !80, inlinedAt: !82)
!97 = !{i64 3065636}
!98 = !DILocation(line: 62, column: 9, scope: !80, inlinedAt: !82)
!99 = !{i64 3083819}
!100 = !DILocation(line: 71, column: 72, scope: !80, inlinedAt: !82)
!101 = !DILocation(line: 27, column: 5, scope: !102, inlinedAt: !103)
!102 = distinct !DISubprogram(name: "store_shared_vec", scope: !95, file: !95, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!103 = distinct !DILocation(line: 71, column: 17, scope: !80, inlinedAt: !82)
!104 = !{i64 3063971}
!105 = !DILocation(line: 72, column: 95, scope: !80, inlinedAt: !82)
!106 = !DILocation(line: 27, column: 5, scope: !102, inlinedAt: !107)
!107 = distinct !DILocation(line: 72, column: 17, scope: !80, inlinedAt: !82)
!108 = !DILocation(line: 75, column: 9, scope: !80, inlinedAt: !82)
!109 = !{i64 3084360}
!110 = !DILocation(line: 60, column: 34, scope: !111, inlinedAt: !112)
!111 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !78, file: !78, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!112 = distinct !DILocation(line: 37, column: 39, scope: !113, inlinedAt: !114)
!113 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !81, file: !81, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!114 = distinct !DILocation(line: 12, column: 5, scope: !115, inlinedAt: !116)
!115 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> > >", scope: !84, file: !84, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!116 = distinct !DILocation(line: 107, column: 9, scope: !9)
!117 = !DILocation(line: 63, column: 54, scope: !87, inlinedAt: !118)
!118 = distinct !DILocation(line: 38, column: 57, scope: !113, inlinedAt: !114)
!119 = !DILocation(line: 63, column: 60, scope: !87, inlinedAt: !118)
!120 = !DILocation(line: 63, column: 16, scope: !87, inlinedAt: !118)
!121 = !DILocation(line: 58, column: 68, scope: !113, inlinedAt: !114)
!122 = !DILocation(line: 92, column: 5, scope: !94, inlinedAt: !123)
!123 = distinct !DILocation(line: 58, column: 26, scope: !113, inlinedAt: !114)
!124 = !DILocation(line: 62, column: 9, scope: !113, inlinedAt: !114)
!125 = !DILocation(line: 71, column: 72, scope: !113, inlinedAt: !114)
!126 = !DILocation(line: 27, column: 5, scope: !102, inlinedAt: !127)
!127 = distinct !DILocation(line: 71, column: 17, scope: !113, inlinedAt: !114)
!128 = !DILocation(line: 72, column: 95, scope: !113, inlinedAt: !114)
!129 = !DILocation(line: 27, column: 5, scope: !102, inlinedAt: !130)
!130 = distinct !DILocation(line: 72, column: 17, scope: !113, inlinedAt: !114)
!131 = !DILocation(line: 75, column: 9, scope: !113, inlinedAt: !114)
!132 = !DILocation(line: 108, column: 9, scope: !9)
!133 = !DILocation(line: 110, column: 13, scope: !9)
!134 = !DILocation(line: 114, column: 9, scope: !9)
!135 = !DILocation(line: 111, column: 13, scope: !9)
!136 = !DILocation(line: 112, column: 9, scope: !9)
!137 = !DILocation(line: 216, column: 13, scope: !9)
!138 = !DILocation(line: 115, column: 39, scope: !9)
!139 = !DILocation(line: 115, column: 43, scope: !9)
!140 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !143)
!141 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !142, file: !142, line: 227, type: !10, scopeLine: 227, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!142 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_16_persist")
!143 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !145)
!144 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 64>, 32, 16> >", scope: !142, file: !142, line: 247, type: !10, scopeLine: 247, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!145 = distinct !DILocation(line: 118, column: 13, scope: !9)
!146 = !{i64 4294578}
!147 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !145)
!148 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !151)
!149 = distinct !DISubprogram(name: "operator=", scope: !150, file: !150, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!150 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!151 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !145)
!152 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !153)
!153 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !154)
!154 = distinct !DILocation(line: 119, column: 13, scope: !9)
!155 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !154)
!156 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !157)
!157 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !154)
!158 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !159)
!159 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !161)
!160 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 32, 16> >", scope: !142, file: !142, line: 247, type: !10, scopeLine: 247, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!161 = distinct !DILocation(line: 120, column: 13, scope: !9)
!162 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !161)
!163 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !164)
!164 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !161)
!165 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !166)
!166 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !167)
!167 = distinct !DILocation(line: 121, column: 13, scope: !9)
!168 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !167)
!169 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !170)
!170 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !167)
!171 = !DILocation(line: 122, column: 13, scope: !9)
!172 = !DILocation(line: 124, column: 17, scope: !9)
!173 = !DILocation(line: 26, column: 23, scope: !174, inlinedAt: !175)
!174 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !142, file: !142, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!175 = distinct !DILocation(line: 177, column: 39, scope: !176, inlinedAt: !177)
!176 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !142, file: !142, line: 139, type: !10, scopeLine: 142, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!177 = distinct !DILocation(line: 125, column: 17, scope: !9)
!178 = !DILocation(line: 127, column: 13, scope: !9)
!179 = !DILocation(line: 128, column: 13, scope: !9)
!180 = !DILocation(line: 129, column: 13, scope: !9)
!181 = !DILocation(line: 132, column: 13, scope: !9)
!182 = !{i64 4303822}
!183 = !DILocation(line: 133, column: 13, scope: !9)
!184 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !187)
!185 = distinct !DISubprogram(name: "mfma161616", scope: !186, file: !186, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!186 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!187 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !189)
!188 = distinct !DISubprogram(name: "mma_ABt_base", scope: !186, file: !186, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!189 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !191)
!190 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !186, file: !186, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!191 = distinct !DILocation(line: 134, column: 13, scope: !9)
!192 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !193)
!193 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !194)
!194 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !195)
!195 = distinct !DILocation(line: 135, column: 13, scope: !9)
!196 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !197)
!197 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !198)
!198 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !199)
!199 = distinct !DILocation(line: 136, column: 13, scope: !9)
!200 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !201)
!201 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !202)
!202 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !203)
!203 = distinct !DILocation(line: 137, column: 13, scope: !9)
!204 = !DILocation(line: 138, column: 13, scope: !9)
!205 = !DILocation(line: 139, column: 13, scope: !9)
!206 = !DILocation(line: 140, column: 13, scope: !9)
!207 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !208)
!208 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !209)
!209 = distinct !DILocation(line: 143, column: 13, scope: !9)
!210 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !209)
!211 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !212)
!212 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !209)
!213 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !214)
!214 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !215)
!215 = distinct !DILocation(line: 144, column: 13, scope: !9)
!216 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !215)
!217 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !218)
!218 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !215)
!219 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !220)
!220 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !221)
!221 = distinct !DILocation(line: 145, column: 13, scope: !9)
!222 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !221)
!223 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !224)
!224 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !221)
!225 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !226)
!226 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !227)
!227 = distinct !DILocation(line: 146, column: 13, scope: !9)
!228 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !227)
!229 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !230)
!230 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !227)
!231 = !DILocation(line: 147, column: 13, scope: !9)
!232 = !DILocation(line: 149, column: 17, scope: !9)
!233 = !DILocation(line: 61, column: 18, scope: !111, inlinedAt: !234)
!234 = distinct !DILocation(line: 153, column: 39, scope: !235, inlinedAt: !236)
!235 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !142, file: !142, line: 139, type: !10, scopeLine: 142, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!236 = distinct !DILocation(line: 150, column: 17, scope: !9)
!237 = !DILocation(line: 63, column: 68, scope: !87, inlinedAt: !238)
!238 = distinct !DILocation(line: 154, column: 24, scope: !235, inlinedAt: !236)
!239 = !DILocation(line: 63, column: 16, scope: !87, inlinedAt: !238)
!240 = !DILocation(line: 32, column: 28, scope: !241, inlinedAt: !242)
!241 = distinct !DISubprogram(name: "make_srsrc", scope: !142, file: !142, line: 31, type: !10, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!242 = distinct !DILocation(line: 160, column: 19, scope: !235, inlinedAt: !236)
!243 = !DILocation(line: 26, column: 23, scope: !174, inlinedAt: !244)
!244 = distinct !DILocation(line: 177, column: 39, scope: !235, inlinedAt: !236)
!245 = !DILocation(line: 152, column: 13, scope: !9)
!246 = !DILocation(line: 153, column: 13, scope: !9)
!247 = !DILocation(line: 154, column: 13, scope: !9)
!248 = !DILocation(line: 157, column: 13, scope: !9)
!249 = !{i64 4305199}
!250 = !DILocation(line: 158, column: 13, scope: !9)
!251 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !252)
!252 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !253)
!253 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !254)
!254 = distinct !DILocation(line: 159, column: 13, scope: !9)
!255 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !256)
!256 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !257)
!257 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !258)
!258 = distinct !DILocation(line: 160, column: 13, scope: !9)
!259 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !260)
!260 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !261)
!261 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !262)
!262 = distinct !DILocation(line: 161, column: 13, scope: !9)
!263 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !264)
!264 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !265)
!265 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !266)
!266 = distinct !DILocation(line: 162, column: 13, scope: !9)
!267 = !DILocation(line: 163, column: 13, scope: !9)
!268 = !DILocation(line: 164, column: 13, scope: !9)
!269 = !DILocation(line: 165, column: 13, scope: !9)
!270 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !271)
!271 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !272)
!272 = distinct !DILocation(line: 169, column: 13, scope: !9)
!273 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !272)
!274 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !275)
!275 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !272)
!276 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !277)
!277 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !278)
!278 = distinct !DILocation(line: 170, column: 13, scope: !9)
!279 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !278)
!280 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !281)
!281 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !278)
!282 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !283)
!283 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !284)
!284 = distinct !DILocation(line: 171, column: 13, scope: !9)
!285 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !284)
!286 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !287)
!287 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !284)
!288 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !289)
!289 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !290)
!290 = distinct !DILocation(line: 172, column: 13, scope: !9)
!291 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !290)
!292 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !293)
!293 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !290)
!294 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !295)
!295 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !296)
!296 = distinct !DILocation(line: 173, column: 13, scope: !9)
!297 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !296)
!298 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !299)
!299 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !296)
!300 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !301)
!301 = distinct !DILocation(line: 285, column: 37, scope: !144, inlinedAt: !302)
!302 = distinct !DILocation(line: 174, column: 13, scope: !9)
!303 = !DILocation(line: 287, column: 47, scope: !144, inlinedAt: !302)
!304 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !305)
!305 = distinct !DILocation(line: 288, column: 45, scope: !144, inlinedAt: !302)
!306 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !307)
!307 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !308)
!308 = distinct !DILocation(line: 175, column: 13, scope: !9)
!309 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !308)
!310 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !311)
!311 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !308)
!312 = !DILocation(line: 229, column: 5, scope: !141, inlinedAt: !313)
!313 = distinct !DILocation(line: 285, column: 37, scope: !160, inlinedAt: !314)
!314 = distinct !DILocation(line: 176, column: 13, scope: !9)
!315 = !DILocation(line: 287, column: 47, scope: !160, inlinedAt: !314)
!316 = !DILocation(line: 494, column: 7, scope: !149, inlinedAt: !317)
!317 = distinct !DILocation(line: 288, column: 45, scope: !160, inlinedAt: !314)
!318 = !DILocation(line: 178, column: 13, scope: !9)
!319 = !DILocation(line: 179, column: 13, scope: !9)
!320 = !DILocation(line: 182, column: 13, scope: !9)
!321 = !{i64 4306715}
!322 = !DILocation(line: 183, column: 13, scope: !9)
!323 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !324)
!324 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !325)
!325 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !326)
!326 = distinct !DILocation(line: 184, column: 13, scope: !9)
!327 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !328)
!328 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !329)
!329 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !330)
!330 = distinct !DILocation(line: 185, column: 13, scope: !9)
!331 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !332)
!332 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !333)
!333 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !334)
!334 = distinct !DILocation(line: 186, column: 13, scope: !9)
!335 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !336)
!336 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !337)
!337 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !338)
!338 = distinct !DILocation(line: 187, column: 13, scope: !9)
!339 = !DILocation(line: 188, column: 13, scope: !9)
!340 = !DILocation(line: 189, column: 13, scope: !9)
!341 = !DILocation(line: 190, column: 13, scope: !9)
!342 = !DILocation(line: 193, column: 17, scope: !9)
!343 = !DILocation(line: 194, column: 17, scope: !9)
!344 = !{i64 4307249}
!345 = !DILocation(line: 17, column: 5, scope: !346, inlinedAt: !347)
!346 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !142, file: !142, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!347 = distinct !DILocation(line: 219, column: 17, scope: !348, inlinedAt: !349)
!348 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !142, file: !142, line: 188, type: !10, scopeLine: 190, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!349 = distinct !DILocation(line: 195, column: 17, scope: !9)
!350 = !{i64 4290525}
!351 = !DILocation(line: 17, column: 5, scope: !346, inlinedAt: !352)
!352 = distinct !DILocation(line: 220, column: 17, scope: !348, inlinedAt: !349)
!353 = !DILocation(line: 197, column: 17, scope: !9)
!354 = !{i64 4307442}
!355 = !DILocation(line: 17, column: 5, scope: !346, inlinedAt: !356)
!356 = distinct !DILocation(line: 219, column: 17, scope: !357, inlinedAt: !358)
!357 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 64>, 512>", scope: !142, file: !142, line: 188, type: !10, scopeLine: 190, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!358 = distinct !DILocation(line: 198, column: 17, scope: !9)
!359 = !DILocation(line: 17, column: 5, scope: !346, inlinedAt: !360)
!360 = distinct !DILocation(line: 220, column: 17, scope: !357, inlinedAt: !358)
!361 = !DILocation(line: 200, column: 13, scope: !9)
!362 = !DILocation(line: 201, column: 13, scope: !9)
!363 = !DILocation(line: 202, column: 13, scope: !9)
!364 = !DILocation(line: 205, column: 13, scope: !9)
!365 = !{i64 4307736}
!366 = !DILocation(line: 206, column: 13, scope: !9)
!367 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !368)
!368 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !369)
!369 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !370)
!370 = distinct !DILocation(line: 207, column: 13, scope: !9)
!371 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !372)
!372 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !373)
!373 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !374)
!374 = distinct !DILocation(line: 208, column: 13, scope: !9)
!375 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !376)
!376 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !377)
!377 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !378)
!378 = distinct !DILocation(line: 209, column: 13, scope: !9)
!379 = !DILocation(line: 29, column: 27, scope: !185, inlinedAt: !380)
!380 = distinct !DILocation(line: 81, column: 5, scope: !188, inlinedAt: !381)
!381 = distinct !DILocation(line: 216, column: 13, scope: !190, inlinedAt: !382)
!382 = distinct !DILocation(line: 210, column: 13, scope: !9)
!383 = !DILocation(line: 211, column: 13, scope: !9)
!384 = !DILocation(line: 212, column: 13, scope: !9)
!385 = !DILocation(line: 213, column: 13, scope: !9)
!386 = !DILocation(line: 114, column: 33, scope: !9)
!387 = distinct !{!387, !134, !388, !389}
!388 = !DILocation(line: 214, column: 9, scope: !9)
!389 = !{!"llvm.loop.mustprogress"}
!390 = !DILocation(line: 217, column: 13, scope: !9)
!391 = !DILocation(line: 218, column: 9, scope: !9)
!392 = !DILocation(line: 227, column: 36, scope: !9)
!393 = !DILocation(line: 60, column: 34, scope: !394, inlinedAt: !395)
!394 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !78, file: !78, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!395 = distinct !DILocation(line: 192, column: 41, scope: !396, inlinedAt: !398)
!396 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !397, file: !397, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!397 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!398 = distinct !DILocation(line: 212, column: 5, scope: !399, inlinedAt: !400)
!399 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !397, file: !397, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!400 = distinct !DILocation(line: 224, column: 13, scope: !9)
!401 = !DILocation(line: 61, column: 18, scope: !394, inlinedAt: !395)
!402 = !DILocation(line: 63, column: 54, scope: !403, inlinedAt: !404)
!403 = distinct !DISubprogram(name: "operator[]", scope: !88, file: !88, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!404 = distinct !DILocation(line: 192, column: 23, scope: !396, inlinedAt: !398)
!405 = !DILocation(line: 63, column: 60, scope: !403, inlinedAt: !404)
!406 = !DILocation(line: 63, column: 68, scope: !403, inlinedAt: !404)
!407 = !DILocation(line: 63, column: 16, scope: !403, inlinedAt: !404)
!408 = !DILocation(line: 203, column: 49, scope: !396, inlinedAt: !398)
!409 = !DILocation(line: 203, column: 13, scope: !396, inlinedAt: !398)
!410 = !DILocation(line: 203, column: 47, scope: !396, inlinedAt: !398)
!411 = !{!412, !412, i64 0}
!412 = !{!"float", !413, i64 0}
!413 = !{!"omnipotent char", !414, i64 0}
!414 = !{!"Simple C++ TBAA"}
!415 = !DILocation(line: 204, column: 49, scope: !396, inlinedAt: !398)
!416 = !DILocation(line: 204, column: 13, scope: !396, inlinedAt: !398)
!417 = !DILocation(line: 204, column: 47, scope: !396, inlinedAt: !398)
!418 = !DILocation(line: 205, column: 49, scope: !396, inlinedAt: !398)
!419 = !DILocation(line: 205, column: 13, scope: !396, inlinedAt: !398)
!420 = !DILocation(line: 205, column: 47, scope: !396, inlinedAt: !398)
!421 = !DILocation(line: 206, column: 49, scope: !396, inlinedAt: !398)
!422 = !DILocation(line: 206, column: 13, scope: !396, inlinedAt: !398)
!423 = !DILocation(line: 206, column: 47, scope: !396, inlinedAt: !398)
!424 = !DILocation(line: 87, column: 43, scope: !9)
!425 = !DILocation(line: 87, column: 5, scope: !9)
!426 = distinct !{!426, !425, !427, !389}
!427 = !DILocation(line: 231, column: 5, scope: !9)
!428 = !DILocation(line: 232, column: 1, scope: !9)
