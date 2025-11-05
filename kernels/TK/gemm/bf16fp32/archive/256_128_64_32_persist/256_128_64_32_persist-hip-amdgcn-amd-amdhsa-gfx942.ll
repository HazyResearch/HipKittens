; ModuleID = '256_128_64_32_persist-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_128_64_32_persist.cpp"
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
@__hip_cuid_d98d5abd5998475c = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_d98d5abd5998475c to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.0798.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.7.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.7.0.copyload = load i64, ptr addrspace(4) %.sroa.7.0..sroa_idx, align 8
  %.sroa.9803.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.9803.0.copyload = load ptr, ptr addrspace(4) %.sroa.9803.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.17807.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.17807.0.copyload = load ptr, ptr addrspace(4) %.sroa.17807.0..sroa_idx, align 8
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
  %.not.i.i106 = icmp eq i64 %8, 0, !dbg !26
  %reass.sub.i.i107 = and i64 %7, -16, !dbg !27
  %9 = add i64 %reass.sub.i.i107, 16, !dbg !27
  %10 = inttoptr i64 %9 to ptr, !dbg !27
  %11 = select i1 %.not.i.i106, ptr %6, ptr %10, !dbg !27
  %12 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !28
  %13 = lshr i32 %12, 6, !dbg !36
  %14 = lshr i32 %12, 8, !dbg !37
  %15 = and i32 %13, 3, !dbg !38
  %16 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %17 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %18 = load i32, ptr addrspace(4) %17, align 4, !tbaa !39
  %19 = trunc i64 %.sroa.7.0.copyload to i32
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
  %121 = shl i32 %30, 1
  %122 = shl i32 %26, 1
  %123 = shl i32 %19, 8
  %124 = or disjoint i32 %115, 32
  %125 = add i32 %116, %124
  %126 = lshr i32 %125, 4
  %127 = and i32 %126, 120
  %128 = xor i32 %127, %125
  %129 = or disjoint i32 %110, 16
  %130 = or disjoint i32 %109, %129
  %131 = shl nuw nsw i32 %130, 7
  %132 = add i32 %131, %71
  %133 = add i32 %132, %115
  %134 = lshr i32 %133, 4
  %135 = and i32 %134, 120
  %136 = xor i32 %135, %133
  %137 = add i32 %132, %124
  %138 = lshr i32 %137, 4
  %139 = and i32 %138, 120
  %140 = xor i32 %139, %137
  %141 = or disjoint i32 %109, 128
  %142 = or disjoint i32 %141, %110
  %143 = shl nuw nsw i32 %142, 7
  %144 = add i32 %143, %71
  %145 = add i32 %144, %115
  %146 = lshr i32 %145, 4
  %147 = and i32 %146, 120
  %148 = xor i32 %147, %145
  %149 = add i32 %144, %124
  %150 = lshr i32 %149, 4
  %151 = and i32 %150, 120
  %152 = xor i32 %151, %149
  %153 = or disjoint i32 %141, %129
  %154 = shl nuw nsw i32 %153, 7
  %155 = add i32 %154, %71
  %156 = add i32 %155, %115
  %157 = lshr i32 %156, 4
  %158 = and i32 %157, 120
  %159 = xor i32 %158, %156
  %160 = add i32 %155, %124
  %161 = lshr i32 %160, 4
  %162 = and i32 %161, 120
  %163 = xor i32 %162, %160
  %164 = shl nuw nsw i32 %14, 5
  %165 = or disjoint i32 %164, %110
  %166 = shl nuw nsw i32 %165, 7
  %167 = add i32 %166, %32
  %168 = add i32 %167, %115
  %169 = lshr i32 %168, 4
  %170 = and i32 %169, 120
  %171 = xor i32 %170, %168
  %172 = add i32 %167, %124
  %173 = lshr i32 %172, 4
  %174 = and i32 %173, 120
  %175 = xor i32 %174, %172
  %176 = or disjoint i32 %129, %164
  %177 = shl nuw nsw i32 %176, 7
  %178 = add i32 %177, %32
  %179 = add i32 %178, %115
  %180 = lshr i32 %179, 4
  %181 = and i32 %180, 120
  %182 = xor i32 %181, %179
  %183 = add i32 %178, %124
  %184 = lshr i32 %183, 4
  %185 = and i32 %184, 120
  %186 = xor i32 %185, %183
  %187 = add nuw nsw i32 %164, 64
  %188 = or disjoint i32 %187, %110
  %189 = shl nuw nsw i32 %188, 7
  %190 = add i32 %189, %32
  %191 = add i32 %190, %115
  %192 = lshr i32 %191, 4
  %193 = and i32 %192, 120
  %194 = xor i32 %193, %191
  %195 = add i32 %190, %124
  %196 = lshr i32 %195, 4
  %197 = and i32 %196, 120
  %198 = xor i32 %197, %195
  %199 = or disjoint i32 %187, %129
  %200 = shl nuw nsw i32 %199, 7
  %201 = add i32 %200, %32
  %202 = add i32 %201, %115
  %203 = lshr i32 %202, 4
  %204 = and i32 %203, 120
  %205 = xor i32 %204, %202
  %206 = add i32 %201, %124
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
  %220 = or disjoint i32 %115, 96
  %221 = add i32 %116, %220
  %222 = lshr i32 %221, 4
  %223 = and i32 %222, 120
  %224 = xor i32 %223, %221
  %225 = add i32 %132, %215
  %226 = lshr i32 %225, 4
  %227 = and i32 %226, 120
  %228 = xor i32 %227, %225
  %229 = add i32 %132, %220
  %230 = lshr i32 %229, 4
  %231 = and i32 %230, 120
  %232 = xor i32 %231, %229
  %233 = add i32 %144, %215
  %234 = lshr i32 %233, 4
  %235 = and i32 %234, 120
  %236 = xor i32 %235, %233
  %237 = add i32 %144, %220
  %238 = lshr i32 %237, 4
  %239 = and i32 %238, 120
  %240 = xor i32 %239, %237
  %241 = add i32 %155, %215
  %242 = lshr i32 %241, 4
  %243 = and i32 %242, 120
  %244 = xor i32 %243, %241
  %245 = add i32 %155, %220
  %246 = lshr i32 %245, 4
  %247 = and i32 %246, 120
  %248 = xor i32 %247, %245
  %249 = add i32 %167, %215
  %250 = lshr i32 %249, 4
  %251 = and i32 %250, 120
  %252 = xor i32 %251, %249
  %253 = add i32 %167, %220
  %254 = lshr i32 %253, 4
  %255 = and i32 %254, 120
  %256 = xor i32 %255, %253
  %257 = add i32 %178, %215
  %258 = lshr i32 %257, 4
  %259 = and i32 %258, 120
  %260 = xor i32 %259, %257
  %261 = add i32 %178, %220
  %262 = lshr i32 %261, 4
  %263 = and i32 %262, 120
  %264 = xor i32 %263, %261
  %265 = add i32 %190, %215
  %266 = lshr i32 %265, 4
  %267 = and i32 %266, 120
  %268 = xor i32 %267, %265
  %269 = add i32 %190, %220
  %270 = lshr i32 %269, 4
  %271 = and i32 %270, 120
  %272 = xor i32 %271, %269
  %273 = add i32 %201, %215
  %274 = lshr i32 %273, 4
  %275 = and i32 %274, 120
  %276 = xor i32 %275, %273
  %277 = add i32 %201, %220
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
  %352 = add nsw i32 %337, %129
  %353 = sext i32 %352 to i64
  %354 = add nsw i32 %339, %129
  %355 = sext i32 %354 to i64
  %356 = add nsw i32 %341, %129
  %357 = sext i32 %356 to i64
  %358 = add nsw i32 %343, %129
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
  %376 = add nsw i32 %361, %129
  %377 = sext i32 %376 to i64
  %378 = add nsw i32 %363, %129
  %379 = sext i32 %378 to i64
  %380 = add nsw i32 %365, %129
  %381 = sext i32 %380 to i64
  %382 = add nsw i32 %367, %129
  %383 = sext i32 %382 to i64
  br label %384, !dbg !43

384:                                              ; preds = %718, %1
  %.sroa.0638.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.0638.2.off0, %718 ]
  %.sroa.0638.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.0638.2.off64, %718 ]
  %.sroa.5.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.5.2.off0, %718 ]
  %.sroa.5.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.5.2.off64, %718 ]
  %.sroa.8639.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.8639.2.off0, %718 ]
  %.sroa.8639.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.8639.2.off64, %718 ]
  %.sroa.11.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.11.2.off0, %718 ]
  %.sroa.11.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.11.2.off64, %718 ]
  %.sroa.0640.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.0640.2.off0, %718 ]
  %.sroa.0640.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.0640.2.off64, %718 ]
  %.sroa.5641.0.off0 = phi i64 [ undef, %1 ], [ %.sroa.5641.2.off0, %718 ]
  %.sroa.5641.0.off64 = phi i64 [ 0, %1 ], [ %.sroa.5641.2.off64, %718 ]
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
  %.decomposed1130 = sub i16 %.lhs.trunc.i, %399, !dbg !61
  %.sext.i = sext i16 %.decomposed1130 to i32, !dbg !61
  %400 = add nsw i32 %394, %.sext.i, !dbg !63
  %.sext23.i = sext i16 %398 to i32, !dbg !62
  %401 = insertelement <2 x i32> poison, i32 %400, i64 0, !dbg !64
  %402 = insertelement <2 x i32> %401, i32 %.sext23.i, i64 1, !dbg !64
  br label %_Z12get_task_idxi.exit

_Z12get_task_idxi.exit:                           ; preds = %384, %388
  %.sroa.0.0.i = phi <2 x i32> [ %402, %388 ], [ <i32 -1, i32 -1>, %384 ], !dbg !70
  %.sroa.0.0.vec.extract.i = extractelement <2 x i32> %.sroa.0.0.i, i64 0, !dbg !73
  %.sroa.0.4.vec.extract.i = extractelement <2 x i32> %.sroa.0.0.i, i64 1, !dbg !73
  %.not105 = icmp eq i32 %.sroa.0.0.vec.extract.i, -1, !dbg !74
  br i1 %.not105, label %.critedge, label %403, !dbg !75

403:                                              ; preds = %_Z12get_task_idxi.exit
  %404 = shl nsw i32 %.sroa.0.0.vec.extract.i, 7, !dbg !76
  %405 = sext i32 %404 to i64, !dbg !86
  %406 = mul i64 %.sroa.7.0.copyload, %405, !dbg !90
  %407 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.0798.0.copyload, i64 %406, !dbg !91
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
  %415 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.9803.0.copyload, i64 %414, !dbg !120
  %416 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %59, !dbg !121
  %417 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %416) #7, !dbg !122, !srcloc !97
  %418 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %62, !dbg !121
  %419 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %418) #7, !dbg !122, !srcloc !97
  %420 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %66, !dbg !121
  %421 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %420) #7, !dbg !122, !srcloc !97
  %422 = getelementptr inbounds %struct.__hip_bfloat16, ptr %415, i64 %70, !dbg !121
  %423 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %422) #7, !dbg !122, !srcloc !97
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !124, !srcloc !99
  %.sroa.0.0.extract.trunc.i.i112 = trunc i128 %417 to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %77, i64 %.sroa.0.0.extract.trunc.i.i112) #7, !dbg !126, !srcloc !104
  %.sroa.0.8.extract.shift.i.i113 = lshr i128 %417, 64, !dbg !128
  %.sroa.0.8.extract.trunc.i.i114 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i113 to i64, !dbg !128
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %81, i64 %.sroa.0.8.extract.trunc.i.i114) #7, !dbg !129, !srcloc !104
  %.sroa.5.16.extract.trunc.i.i115 = trunc i128 %419 to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %85, i64 %.sroa.5.16.extract.trunc.i.i115) #7, !dbg !126, !srcloc !104
  %.sroa.5.24.extract.shift.i.i116 = lshr i128 %419, 64, !dbg !128
  %.sroa.5.24.extract.trunc.i.i117 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i116 to i64, !dbg !128
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %89, i64 %.sroa.5.24.extract.trunc.i.i117) #7, !dbg !129, !srcloc !104
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

425:                                              ; preds = %684
  br i1 %335, label %717, label %718, !dbg !137

426:                                              ; preds = %.preheader, %684
  %427 = phi <4 x float> [ %716, %684 ], [ zeroinitializer, %.preheader ]
  %428 = phi <4 x float> [ %714, %684 ], [ zeroinitializer, %.preheader ]
  %429 = phi <4 x float> [ %712, %684 ], [ zeroinitializer, %.preheader ]
  %.val33.i443933 = phi <4 x float> [ %710, %684 ], [ zeroinitializer, %.preheader ]
  %430 = phi <4 x float> [ %708, %684 ], [ zeroinitializer, %.preheader ]
  %431 = phi <4 x float> [ %706, %684 ], [ zeroinitializer, %.preheader ]
  %432 = phi <4 x float> [ %704, %684 ], [ zeroinitializer, %.preheader ]
  %.val33.i423924 = phi <4 x float> [ %702, %684 ], [ zeroinitializer, %.preheader ]
  %433 = phi <4 x float> [ %700, %684 ], [ zeroinitializer, %.preheader ]
  %434 = phi <4 x float> [ %698, %684 ], [ zeroinitializer, %.preheader ]
  %435 = phi <4 x float> [ %696, %684 ], [ zeroinitializer, %.preheader ]
  %.val33.i403915 = phi <4 x float> [ %694, %684 ], [ zeroinitializer, %.preheader ]
  %.val33.1.1.i478912 = phi <4 x float> [ %692, %684 ], [ zeroinitializer, %.preheader ]
  %.val33.138.i473909 = phi <4 x float> [ %690, %684 ], [ zeroinitializer, %.preheader ]
  %.val33.1.i468906 = phi <4 x float> [ %688, %684 ], [ zeroinitializer, %.preheader ]
  %.0103903 = phi i32 [ %436, %684 ], [ 0, %.preheader ]
  %.sroa.5641.1.off64902 = phi i64 [ %.sroa.5641.2.off64, %684 ], [ %.sroa.5641.0.off64, %.preheader ]
  %.sroa.5641.1.off0901 = phi i64 [ %.sroa.5641.2.off0, %684 ], [ %.sroa.5641.0.off0, %.preheader ]
  %.sroa.0640.1.off64900 = phi i64 [ %.sroa.0640.2.off64, %684 ], [ %.sroa.0640.0.off64, %.preheader ]
  %.sroa.0640.1.off0899 = phi i64 [ %.sroa.0640.2.off0, %684 ], [ %.sroa.0640.0.off0, %.preheader ]
  %.sroa.11.1.off64898 = phi i64 [ %.sroa.11.2.off64, %684 ], [ %.sroa.11.0.off64, %.preheader ]
  %.sroa.11.1.off0897 = phi i64 [ %.sroa.11.2.off0, %684 ], [ %.sroa.11.0.off0, %.preheader ]
  %.sroa.8639.1.off64896 = phi i64 [ %.sroa.8639.2.off64, %684 ], [ %.sroa.8639.0.off64, %.preheader ]
  %.sroa.8639.1.off0895 = phi i64 [ %.sroa.8639.2.off0, %684 ], [ %.sroa.8639.0.off0, %.preheader ]
  %.sroa.5.1.off64894 = phi i64 [ %.sroa.5.2.off64, %684 ], [ %.sroa.5.0.off64, %.preheader ]
  %.sroa.5.1.off0893 = phi i64 [ %.sroa.5.2.off0, %684 ], [ %.sroa.5.0.off0, %.preheader ]
  %.sroa.0638.1.off64892 = phi i64 [ %.sroa.0638.2.off64, %684 ], [ %.sroa.0638.0.off64, %.preheader ]
  %.sroa.0638.1.off0891 = phi i64 [ %.sroa.0638.2.off0, %684 ], [ %.sroa.0638.0.off0, %.preheader ]
  %.val33.i889890 = phi <4 x float> [ %686, %684 ], [ zeroinitializer, %.preheader ]
  %436 = add nuw nsw i32 %.0103903, 1, !dbg !138
  %.not = icmp eq i32 %.0103903, 127, !dbg !139
  br i1 %.not, label %446, label %437, !dbg !140

437:                                              ; preds = %426
  %438 = shl nuw nsw i32 %436, 6
  %439 = zext nneg i32 %438 to i64
  %440 = getelementptr %struct.__hip_bfloat16, ptr %407, i64 %439
  %441 = ptrtoint ptr %440 to i64
  %442 = bitcast i64 %441 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %442, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %443 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %443, i32 %123, i64 2
  %444 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %122, i32 noundef 0, i32 noundef 0) #7, !dbg !141
  %445 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %121, i32 noundef 0, i32 noundef 0) #7, !dbg !141
  %extract.t862 = trunc i128 %444 to i64, !dbg !147
  %extract864 = lshr i128 %444, 64, !dbg !147
  %extract.t865 = trunc nuw i128 %extract864 to i64, !dbg !147
  %extract.t875 = trunc i128 %445 to i64, !dbg !147
  %extract877 = lshr i128 %445, 64, !dbg !147
  %extract.t878 = trunc nuw i128 %extract877 to i64, !dbg !147
  br label %446, !dbg !147

446:                                              ; preds = %437, %426
  %.sroa.0640.2.off0 = phi i64 [ %.sroa.0640.1.off0899, %426 ], [ %extract.t862, %437 ]
  %.sroa.0640.2.off64 = phi i64 [ %.sroa.0640.1.off64900, %426 ], [ %extract.t865, %437 ]
  %.sroa.5641.2.off0 = phi i64 [ %.sroa.5641.1.off0901, %426 ], [ %extract.t875, %437 ]
  %.sroa.5641.2.off64 = phi i64 [ %.sroa.5641.1.off64902, %426 ], [ %extract.t878, %437 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !148
  %447 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %120) #7, !dbg !149, !srcloc !154
  %.sroa.036.2.extract.shift37.i = lshr i64 %447, 16, !dbg !155
  %448 = insertelement <2 x i64> poison, i64 %447, i64 0, !dbg !155
  %449 = insertelement <2 x i64> %448, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !155
  %450 = shufflevector <2 x i64> %448, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !149
  %451 = lshr <2 x i64> %450, <i64 32, i64 48>, !dbg !149
  %452 = shufflevector <2 x i64> %449, <2 x i64> %451, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !156
  %.sroa.0676.6.vec.insert1099 = trunc <4 x i64> %452 to <4 x i16>, !dbg !156
  %453 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %128) #7, !dbg !149, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %453, 16, !dbg !155
  %454 = insertelement <2 x i64> poison, i64 %453, i64 0, !dbg !155
  %455 = insertelement <2 x i64> %454, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !155
  %456 = shufflevector <2 x i64> %454, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !149
  %457 = lshr <2 x i64> %456, <i64 32, i64 48>, !dbg !149
  %458 = shufflevector <2 x i64> %455, <2 x i64> %457, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !156
  %.sroa.10681.14.vec.insert1098 = trunc <4 x i64> %458 to <4 x i16>, !dbg !156
  %459 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %136) #7, !dbg !149, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i = lshr i64 %459, 16, !dbg !155
  %460 = insertelement <2 x i64> poison, i64 %459, i64 0, !dbg !155
  %461 = insertelement <2 x i64> %460, i64 %.sroa.036.2.extract.shift37.145.i, i64 1, !dbg !155
  %462 = shufflevector <2 x i64> %460, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !149
  %463 = lshr <2 x i64> %462, <i64 32, i64 48>, !dbg !149
  %464 = shufflevector <2 x i64> %461, <2 x i64> %463, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !156
  %.sroa.18686.22.vec.insert1097 = trunc <4 x i64> %464 to <4 x i16>, !dbg !156
  %465 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %140) #7, !dbg !149, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i = lshr i64 %465, 16, !dbg !155
  %466 = insertelement <2 x i64> poison, i64 %465, i64 0, !dbg !155
  %467 = insertelement <2 x i64> %466, i64 %.sroa.036.2.extract.shift37.1.1.i, i64 1, !dbg !155
  %468 = shufflevector <2 x i64> %466, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !149
  %469 = lshr <2 x i64> %468, <i64 32, i64 48>, !dbg !149
  %470 = shufflevector <2 x i64> %467, <2 x i64> %469, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !156
  %.sroa.26691.30.vec.insert1096 = trunc <4 x i64> %470 to <4 x i16>, !dbg !156
  %471 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %148) #7, !dbg !160, !srcloc !154
  %.sroa.036.2.extract.shift37.i126 = lshr i64 %471, 16, !dbg !163
  %472 = insertelement <2 x i64> poison, i64 %471, i64 0, !dbg !163
  %473 = insertelement <2 x i64> %472, i64 %.sroa.036.2.extract.shift37.i126, i64 1, !dbg !163
  %474 = shufflevector <2 x i64> %472, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !160
  %475 = lshr <2 x i64> %474, <i64 32, i64 48>, !dbg !160
  %476 = shufflevector <2 x i64> %473, <2 x i64> %475, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !164
  %.sroa.0643.6.vec.insert1095 = trunc <4 x i64> %476 to <4 x i16>, !dbg !164
  %477 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %152) #7, !dbg !160, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i135 = lshr i64 %477, 16, !dbg !163
  %478 = insertelement <2 x i64> poison, i64 %477, i64 0, !dbg !163
  %479 = insertelement <2 x i64> %478, i64 %.sroa.036.2.extract.shift37.1.i135, i64 1, !dbg !163
  %480 = shufflevector <2 x i64> %478, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !160
  %481 = lshr <2 x i64> %480, <i64 32, i64 48>, !dbg !160
  %482 = shufflevector <2 x i64> %479, <2 x i64> %481, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !164
  %.sroa.10.14.vec.insert1094 = trunc <4 x i64> %482 to <4 x i16>, !dbg !164
  %483 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %159) #7, !dbg !160, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i144 = lshr i64 %483, 16, !dbg !163
  %484 = insertelement <2 x i64> poison, i64 %483, i64 0, !dbg !163
  %485 = insertelement <2 x i64> %484, i64 %.sroa.036.2.extract.shift37.145.i144, i64 1, !dbg !163
  %486 = shufflevector <2 x i64> %484, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !160
  %487 = lshr <2 x i64> %486, <i64 32, i64 48>, !dbg !160
  %488 = shufflevector <2 x i64> %485, <2 x i64> %487, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !164
  %.sroa.18.22.vec.insert1093 = trunc <4 x i64> %488 to <4 x i16>, !dbg !164
  %489 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %163) #7, !dbg !160, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i153 = lshr i64 %489, 16, !dbg !163
  %490 = insertelement <2 x i64> poison, i64 %489, i64 0, !dbg !163
  %491 = insertelement <2 x i64> %490, i64 %.sroa.036.2.extract.shift37.1.1.i153, i64 1, !dbg !163
  %492 = shufflevector <2 x i64> %490, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !160
  %493 = lshr <2 x i64> %492, <i64 32, i64 48>, !dbg !160
  %494 = shufflevector <2 x i64> %491, <2 x i64> %493, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !164
  %.sroa.26.30.vec.insert1092 = trunc <4 x i64> %494 to <4 x i16>, !dbg !164
  %495 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %171) #7, !dbg !166, !srcloc !154
  %.sroa.036.2.extract.shift37.i163 = lshr i64 %495, 16, !dbg !170
  %496 = insertelement <2 x i64> poison, i64 %495, i64 0, !dbg !170
  %497 = insertelement <2 x i64> %496, i64 %.sroa.036.2.extract.shift37.i163, i64 1, !dbg !170
  %498 = shufflevector <2 x i64> %496, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %499 = lshr <2 x i64> %498, <i64 32, i64 48>, !dbg !166
  %500 = shufflevector <2 x i64> %497, <2 x i64> %499, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !171
  %.sroa.0756.6.vec.insert1091 = trunc <4 x i64> %500 to <4 x i16>, !dbg !171
  %501 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %175) #7, !dbg !166, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i170 = lshr i64 %501, 16, !dbg !170
  %502 = insertelement <2 x i64> poison, i64 %501, i64 0, !dbg !170
  %503 = insertelement <2 x i64> %502, i64 %.sroa.036.2.extract.shift37.1.i170, i64 1, !dbg !170
  %504 = shufflevector <2 x i64> %502, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %505 = lshr <2 x i64> %504, <i64 32, i64 48>, !dbg !166
  %506 = shufflevector <2 x i64> %503, <2 x i64> %505, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !171
  %.sroa.10761.14.vec.insert1090 = trunc <4 x i64> %506 to <4 x i16>, !dbg !171
  %507 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %182) #7, !dbg !166, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i177 = lshr i64 %507, 16, !dbg !170
  %508 = insertelement <2 x i64> poison, i64 %507, i64 0, !dbg !170
  %509 = insertelement <2 x i64> %508, i64 %.sroa.036.2.extract.shift37.145.i177, i64 1, !dbg !170
  %510 = shufflevector <2 x i64> %508, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %511 = lshr <2 x i64> %510, <i64 32, i64 48>, !dbg !166
  %512 = shufflevector <2 x i64> %509, <2 x i64> %511, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !171
  %.sroa.18766.22.vec.insert1089 = trunc <4 x i64> %512 to <4 x i16>, !dbg !171
  %513 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %186) #7, !dbg !166, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i184 = lshr i64 %513, 16, !dbg !170
  %514 = insertelement <2 x i64> poison, i64 %513, i64 0, !dbg !170
  %515 = insertelement <2 x i64> %514, i64 %.sroa.036.2.extract.shift37.1.1.i184, i64 1, !dbg !170
  %516 = shufflevector <2 x i64> %514, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %517 = lshr <2 x i64> %516, <i64 32, i64 48>, !dbg !166
  %518 = shufflevector <2 x i64> %515, <2 x i64> %517, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !171
  %.sroa.26771.30.vec.insert1088 = trunc <4 x i64> %518 to <4 x i16>, !dbg !171
  %519 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %194) #7, !dbg !173, !srcloc !154
  %.sroa.036.2.extract.shift37.i196 = lshr i64 %519, 16, !dbg !176
  %520 = insertelement <2 x i64> poison, i64 %519, i64 0, !dbg !176
  %521 = insertelement <2 x i64> %520, i64 %.sroa.036.2.extract.shift37.i196, i64 1, !dbg !176
  %522 = shufflevector <2 x i64> %520, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !173
  %523 = lshr <2 x i64> %522, <i64 32, i64 48>, !dbg !173
  %524 = shufflevector <2 x i64> %521, <2 x i64> %523, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.0716.6.vec.insert1087 = trunc <4 x i64> %524 to <4 x i16>, !dbg !177
  %525 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %198) #7, !dbg !173, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i205 = lshr i64 %525, 16, !dbg !176
  %526 = insertelement <2 x i64> poison, i64 %525, i64 0, !dbg !176
  %527 = insertelement <2 x i64> %526, i64 %.sroa.036.2.extract.shift37.1.i205, i64 1, !dbg !176
  %528 = shufflevector <2 x i64> %526, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !173
  %529 = lshr <2 x i64> %528, <i64 32, i64 48>, !dbg !173
  %530 = shufflevector <2 x i64> %527, <2 x i64> %529, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.10721.14.vec.insert1086 = trunc <4 x i64> %530 to <4 x i16>, !dbg !177
  %531 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %205) #7, !dbg !173, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i214 = lshr i64 %531, 16, !dbg !176
  %532 = insertelement <2 x i64> poison, i64 %531, i64 0, !dbg !176
  %533 = insertelement <2 x i64> %532, i64 %.sroa.036.2.extract.shift37.145.i214, i64 1, !dbg !176
  %534 = shufflevector <2 x i64> %532, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !173
  %535 = lshr <2 x i64> %534, <i64 32, i64 48>, !dbg !173
  %536 = shufflevector <2 x i64> %533, <2 x i64> %535, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.18726.22.vec.insert1085 = trunc <4 x i64> %536 to <4 x i16>, !dbg !177
  %537 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %209) #7, !dbg !173, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i223 = lshr i64 %537, 16, !dbg !176
  %538 = insertelement <2 x i64> poison, i64 %537, i64 0, !dbg !176
  %539 = insertelement <2 x i64> %538, i64 %.sroa.036.2.extract.shift37.1.1.i223, i64 1, !dbg !176
  %540 = shufflevector <2 x i64> %538, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !173
  %541 = lshr <2 x i64> %540, <i64 32, i64 48>, !dbg !173
  %542 = shufflevector <2 x i64> %539, <2 x i64> %541, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.26731.30.vec.insert1084 = trunc <4 x i64> %542 to <4 x i16>, !dbg !177
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !179
  br i1 %.not, label %554, label %543, !dbg !180

543:                                              ; preds = %446
  %544 = shl nuw nsw i32 %436, 6, !dbg !181
  %545 = zext nneg i32 %544 to i64, !dbg !185
  %546 = getelementptr %struct.__hip_bfloat16, ptr %415, i64 %545, !dbg !187
  %547 = ptrtoint ptr %546 to i64, !dbg !188
  %548 = bitcast i64 %547 to <2 x i32>, !dbg !188
  %.sroa.0.0.vec.expand.i.i231 = shufflevector <2 x i32> %548, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>, !dbg !188
  %549 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i231, i32 1114112, i64 3, !dbg !188
  %.sroa.0.12.vec.insert.i.i232 = insertelement <4 x i32> %549, i32 %210, i64 2, !dbg !188
  %550 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i232, i32 noundef %211, i32 noundef 0, i32 noundef 0) #7, !dbg !191
  %551 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i232, i32 noundef %212, i32 noundef 0, i32 noundef 0) #7, !dbg !191
  %552 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i232, i32 noundef %213, i32 noundef 0, i32 noundef 0) #7, !dbg !191
  %553 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i232, i32 noundef %214, i32 noundef 0, i32 noundef 0) #7, !dbg !191
  %extract.t810 = trunc i128 %550 to i64, !dbg !193
  %extract812 = lshr i128 %550, 64, !dbg !193
  %extract.t813 = trunc nuw i128 %extract812 to i64, !dbg !193
  %extract.t823 = trunc i128 %551 to i64, !dbg !193
  %extract825 = lshr i128 %551, 64, !dbg !193
  %extract.t826 = trunc nuw i128 %extract825 to i64, !dbg !193
  %extract.t836 = trunc i128 %552 to i64, !dbg !193
  %extract838 = lshr i128 %552, 64, !dbg !193
  %extract.t839 = trunc nuw i128 %extract838 to i64, !dbg !193
  %extract.t849 = trunc i128 %553 to i64, !dbg !193
  %extract851 = lshr i128 %553, 64, !dbg !193
  %extract.t852 = trunc nuw i128 %extract851 to i64, !dbg !193
  br label %554, !dbg !193

554:                                              ; preds = %543, %446
  %.sroa.0638.2.off0 = phi i64 [ %.sroa.0638.1.off0891, %446 ], [ %extract.t810, %543 ]
  %.sroa.0638.2.off64 = phi i64 [ %.sroa.0638.1.off64892, %446 ], [ %extract.t813, %543 ]
  %.sroa.5.2.off0 = phi i64 [ %.sroa.5.1.off0893, %446 ], [ %extract.t823, %543 ]
  %.sroa.5.2.off64 = phi i64 [ %.sroa.5.1.off64894, %446 ], [ %extract.t826, %543 ]
  %.sroa.8639.2.off0 = phi i64 [ %.sroa.8639.1.off0895, %446 ], [ %extract.t836, %543 ]
  %.sroa.8639.2.off64 = phi i64 [ %.sroa.8639.1.off64896, %446 ], [ %extract.t839, %543 ]
  %.sroa.11.2.off0 = phi i64 [ %.sroa.11.1.off0897, %446 ], [ %extract.t849, %543 ]
  %.sroa.11.2.off64 = phi i64 [ %.sroa.11.1.off64898, %446 ], [ %extract.t852, %543 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !194
  %555 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %219) #7, !dbg !195, !srcloc !154
  %.sroa.036.2.extract.shift37.i240 = lshr i64 %555, 16, !dbg !198
  %556 = insertelement <2 x i64> poison, i64 %555, i64 0, !dbg !198
  %557 = insertelement <2 x i64> %556, i64 %.sroa.036.2.extract.shift37.i240, i64 1, !dbg !198
  %558 = shufflevector <2 x i64> %556, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !195
  %559 = lshr <2 x i64> %558, <i64 32, i64 48>, !dbg !195
  %560 = shufflevector <2 x i64> %557, <2 x i64> %559, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !199
  %.sroa.34696.38.vec.insert1083 = trunc <4 x i64> %560 to <4 x i16>, !dbg !199
  %561 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %224) #7, !dbg !195, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i249 = lshr i64 %561, 16, !dbg !198
  %562 = insertelement <2 x i64> poison, i64 %561, i64 0, !dbg !198
  %563 = insertelement <2 x i64> %562, i64 %.sroa.036.2.extract.shift37.1.i249, i64 1, !dbg !198
  %564 = shufflevector <2 x i64> %562, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !195
  %565 = lshr <2 x i64> %564, <i64 32, i64 48>, !dbg !195
  %566 = shufflevector <2 x i64> %563, <2 x i64> %565, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !199
  %.sroa.42701.46.vec.insert1082 = trunc <4 x i64> %566 to <4 x i16>, !dbg !199
  %567 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %228) #7, !dbg !195, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i258 = lshr i64 %567, 16, !dbg !198
  %568 = insertelement <2 x i64> poison, i64 %567, i64 0, !dbg !198
  %569 = insertelement <2 x i64> %568, i64 %.sroa.036.2.extract.shift37.145.i258, i64 1, !dbg !198
  %570 = shufflevector <2 x i64> %568, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !195
  %571 = lshr <2 x i64> %570, <i64 32, i64 48>, !dbg !195
  %572 = shufflevector <2 x i64> %569, <2 x i64> %571, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !199
  %.sroa.50706.54.vec.insert1081 = trunc <4 x i64> %572 to <4 x i16>, !dbg !199
  %573 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %232) #7, !dbg !195, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i267 = lshr i64 %573, 16, !dbg !198
  %574 = insertelement <2 x i64> poison, i64 %573, i64 0, !dbg !198
  %575 = insertelement <2 x i64> %574, i64 %.sroa.036.2.extract.shift37.1.1.i267, i64 1, !dbg !198
  %576 = shufflevector <2 x i64> %574, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !195
  %577 = lshr <2 x i64> %576, <i64 32, i64 48>, !dbg !195
  %578 = shufflevector <2 x i64> %575, <2 x i64> %577, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !199
  %.sroa.58711.62.vec.insert1080 = trunc <4 x i64> %578 to <4 x i16>, !dbg !199
  %579 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %236) #7, !dbg !201, !srcloc !154
  %.sroa.036.2.extract.shift37.i279 = lshr i64 %579, 16, !dbg !204
  %580 = insertelement <2 x i64> poison, i64 %579, i64 0, !dbg !204
  %581 = insertelement <2 x i64> %580, i64 %.sroa.036.2.extract.shift37.i279, i64 1, !dbg !204
  %582 = shufflevector <2 x i64> %580, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %583 = lshr <2 x i64> %582, <i64 32, i64 48>, !dbg !201
  %584 = shufflevector <2 x i64> %581, <2 x i64> %583, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.34.38.vec.insert1079 = trunc <4 x i64> %584 to <4 x i16>, !dbg !205
  %585 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %240) #7, !dbg !201, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i288 = lshr i64 %585, 16, !dbg !204
  %586 = insertelement <2 x i64> poison, i64 %585, i64 0, !dbg !204
  %587 = insertelement <2 x i64> %586, i64 %.sroa.036.2.extract.shift37.1.i288, i64 1, !dbg !204
  %588 = shufflevector <2 x i64> %586, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %589 = lshr <2 x i64> %588, <i64 32, i64 48>, !dbg !201
  %590 = shufflevector <2 x i64> %587, <2 x i64> %589, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.42.46.vec.insert1078 = trunc <4 x i64> %590 to <4 x i16>, !dbg !205
  %591 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %244) #7, !dbg !201, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i297 = lshr i64 %591, 16, !dbg !204
  %592 = insertelement <2 x i64> poison, i64 %591, i64 0, !dbg !204
  %593 = insertelement <2 x i64> %592, i64 %.sroa.036.2.extract.shift37.145.i297, i64 1, !dbg !204
  %594 = shufflevector <2 x i64> %592, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %595 = lshr <2 x i64> %594, <i64 32, i64 48>, !dbg !201
  %596 = shufflevector <2 x i64> %593, <2 x i64> %595, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.50.54.vec.insert1077 = trunc <4 x i64> %596 to <4 x i16>, !dbg !205
  %597 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %248) #7, !dbg !201, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i306 = lshr i64 %597, 16, !dbg !204
  %598 = insertelement <2 x i64> poison, i64 %597, i64 0, !dbg !204
  %599 = insertelement <2 x i64> %598, i64 %.sroa.036.2.extract.shift37.1.1.i306, i64 1, !dbg !204
  %600 = shufflevector <2 x i64> %598, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %601 = lshr <2 x i64> %600, <i64 32, i64 48>, !dbg !201
  %602 = shufflevector <2 x i64> %599, <2 x i64> %601, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.58.62.vec.insert1076 = trunc <4 x i64> %602 to <4 x i16>, !dbg !205
  %603 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %252) #7, !dbg !207, !srcloc !154
  %.sroa.036.2.extract.shift37.i318 = lshr i64 %603, 16, !dbg !210
  %604 = insertelement <2 x i64> poison, i64 %603, i64 0, !dbg !210
  %605 = insertelement <2 x i64> %604, i64 %.sroa.036.2.extract.shift37.i318, i64 1, !dbg !210
  %606 = shufflevector <2 x i64> %604, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %607 = lshr <2 x i64> %606, <i64 32, i64 48>, !dbg !207
  %608 = shufflevector <2 x i64> %605, <2 x i64> %607, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.34776.38.vec.insert1075 = trunc <4 x i64> %608 to <4 x i16>, !dbg !211
  %609 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %256) #7, !dbg !207, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i327 = lshr i64 %609, 16, !dbg !210
  %610 = insertelement <2 x i64> poison, i64 %609, i64 0, !dbg !210
  %611 = insertelement <2 x i64> %610, i64 %.sroa.036.2.extract.shift37.1.i327, i64 1, !dbg !210
  %612 = shufflevector <2 x i64> %610, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %613 = lshr <2 x i64> %612, <i64 32, i64 48>, !dbg !207
  %614 = shufflevector <2 x i64> %611, <2 x i64> %613, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.42781.46.vec.insert1074 = trunc <4 x i64> %614 to <4 x i16>, !dbg !211
  %615 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %260) #7, !dbg !207, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i336 = lshr i64 %615, 16, !dbg !210
  %616 = insertelement <2 x i64> poison, i64 %615, i64 0, !dbg !210
  %617 = insertelement <2 x i64> %616, i64 %.sroa.036.2.extract.shift37.145.i336, i64 1, !dbg !210
  %618 = shufflevector <2 x i64> %616, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %619 = lshr <2 x i64> %618, <i64 32, i64 48>, !dbg !207
  %620 = shufflevector <2 x i64> %617, <2 x i64> %619, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.50786.54.vec.insert1073 = trunc <4 x i64> %620 to <4 x i16>, !dbg !211
  %621 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %264) #7, !dbg !207, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i345 = lshr i64 %621, 16, !dbg !210
  %622 = insertelement <2 x i64> poison, i64 %621, i64 0, !dbg !210
  %623 = insertelement <2 x i64> %622, i64 %.sroa.036.2.extract.shift37.1.1.i345, i64 1, !dbg !210
  %624 = shufflevector <2 x i64> %622, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %625 = lshr <2 x i64> %624, <i64 32, i64 48>, !dbg !207
  %626 = shufflevector <2 x i64> %623, <2 x i64> %625, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.58791.62.vec.insert1072 = trunc <4 x i64> %626 to <4 x i16>, !dbg !211
  %627 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %268) #7, !dbg !213, !srcloc !154
  %.sroa.036.2.extract.shift37.i357 = lshr i64 %627, 16, !dbg !216
  %628 = insertelement <2 x i64> poison, i64 %627, i64 0, !dbg !216
  %629 = insertelement <2 x i64> %628, i64 %.sroa.036.2.extract.shift37.i357, i64 1, !dbg !216
  %630 = shufflevector <2 x i64> %628, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %631 = lshr <2 x i64> %630, <i64 32, i64 48>, !dbg !213
  %632 = shufflevector <2 x i64> %629, <2 x i64> %631, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.34736.38.vec.insert1071 = trunc <4 x i64> %632 to <4 x i16>, !dbg !217
  %633 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %272) #7, !dbg !213, !srcloc !154
  %.sroa.036.2.extract.shift37.1.i366 = lshr i64 %633, 16, !dbg !216
  %634 = insertelement <2 x i64> poison, i64 %633, i64 0, !dbg !216
  %635 = insertelement <2 x i64> %634, i64 %.sroa.036.2.extract.shift37.1.i366, i64 1, !dbg !216
  %636 = shufflevector <2 x i64> %634, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %637 = lshr <2 x i64> %636, <i64 32, i64 48>, !dbg !213
  %638 = shufflevector <2 x i64> %635, <2 x i64> %637, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.42741.46.vec.insert1070 = trunc <4 x i64> %638 to <4 x i16>, !dbg !217
  %639 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %276) #7, !dbg !213, !srcloc !154
  %.sroa.036.2.extract.shift37.145.i375 = lshr i64 %639, 16, !dbg !216
  %640 = insertelement <2 x i64> poison, i64 %639, i64 0, !dbg !216
  %641 = insertelement <2 x i64> %640, i64 %.sroa.036.2.extract.shift37.145.i375, i64 1, !dbg !216
  %642 = shufflevector <2 x i64> %640, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %643 = lshr <2 x i64> %642, <i64 32, i64 48>, !dbg !213
  %644 = shufflevector <2 x i64> %641, <2 x i64> %643, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.50746.54.vec.insert1069 = trunc <4 x i64> %644 to <4 x i16>, !dbg !217
  %645 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %280) #7, !dbg !213, !srcloc !154
  %.sroa.036.2.extract.shift37.1.1.i384 = lshr i64 %645, 16, !dbg !216
  %646 = insertelement <2 x i64> poison, i64 %645, i64 0, !dbg !216
  %647 = insertelement <2 x i64> %646, i64 %.sroa.036.2.extract.shift37.1.1.i384, i64 1, !dbg !216
  %648 = shufflevector <2 x i64> %646, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %649 = lshr <2 x i64> %648, <i64 32, i64 48>, !dbg !213
  %650 = shufflevector <2 x i64> %647, <2 x i64> %649, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.58751.62.vec.insert1068 = trunc <4 x i64> %650 to <4 x i16>, !dbg !217
  tail call void @llvm.amdgcn.s.barrier(), !dbg !219
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !220
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !221
  %651 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0756.6.vec.insert1091, <4 x i16> %.sroa.0676.6.vec.insert1099, <4 x float> %.val33.i889890, i32 0, i32 0, i32 0), !dbg !222
  %652 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10761.14.vec.insert1090, <4 x i16> %.sroa.10681.14.vec.insert1098, <4 x float> %651, i32 0, i32 0, i32 0), !dbg !230
  %653 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0756.6.vec.insert1091, <4 x i16> %.sroa.18686.22.vec.insert1097, <4 x float> %.val33.1.i468906, i32 0, i32 0, i32 0), !dbg !222
  %654 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10761.14.vec.insert1090, <4 x i16> %.sroa.26691.30.vec.insert1096, <4 x float> %653, i32 0, i32 0, i32 0), !dbg !230
  %655 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18766.22.vec.insert1089, <4 x i16> %.sroa.0676.6.vec.insert1099, <4 x float> %.val33.138.i473909, i32 0, i32 0, i32 0), !dbg !222
  %656 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26771.30.vec.insert1088, <4 x i16> %.sroa.10681.14.vec.insert1098, <4 x float> %655, i32 0, i32 0, i32 0), !dbg !230
  %657 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18766.22.vec.insert1089, <4 x i16> %.sroa.18686.22.vec.insert1097, <4 x float> %.val33.1.1.i478912, i32 0, i32 0, i32 0), !dbg !222
  %658 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26771.30.vec.insert1088, <4 x i16> %.sroa.26691.30.vec.insert1096, <4 x float> %657, i32 0, i32 0, i32 0), !dbg !230
  %659 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0756.6.vec.insert1091, <4 x i16> %.sroa.0643.6.vec.insert1095, <4 x float> %.val33.i403915, i32 0, i32 0, i32 0), !dbg !233
  %660 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10761.14.vec.insert1090, <4 x i16> %.sroa.10.14.vec.insert1094, <4 x float> %659, i32 0, i32 0, i32 0), !dbg !237
  %661 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0756.6.vec.insert1091, <4 x i16> %.sroa.18.22.vec.insert1093, <4 x float> %435, i32 0, i32 0, i32 0), !dbg !233
  %662 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10761.14.vec.insert1090, <4 x i16> %.sroa.26.30.vec.insert1092, <4 x float> %661, i32 0, i32 0, i32 0), !dbg !237
  %663 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18766.22.vec.insert1089, <4 x i16> %.sroa.0643.6.vec.insert1095, <4 x float> %434, i32 0, i32 0, i32 0), !dbg !233
  %664 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26771.30.vec.insert1088, <4 x i16> %.sroa.10.14.vec.insert1094, <4 x float> %663, i32 0, i32 0, i32 0), !dbg !237
  %665 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18766.22.vec.insert1089, <4 x i16> %.sroa.18.22.vec.insert1093, <4 x float> %433, i32 0, i32 0, i32 0), !dbg !233
  %666 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26771.30.vec.insert1088, <4 x i16> %.sroa.26.30.vec.insert1092, <4 x float> %665, i32 0, i32 0, i32 0), !dbg !237
  %667 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0716.6.vec.insert1087, <4 x i16> %.sroa.0676.6.vec.insert1099, <4 x float> %.val33.i423924, i32 0, i32 0, i32 0), !dbg !240
  %668 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10721.14.vec.insert1086, <4 x i16> %.sroa.10681.14.vec.insert1098, <4 x float> %667, i32 0, i32 0, i32 0), !dbg !244
  %669 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0716.6.vec.insert1087, <4 x i16> %.sroa.18686.22.vec.insert1097, <4 x float> %432, i32 0, i32 0, i32 0), !dbg !240
  %670 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10721.14.vec.insert1086, <4 x i16> %.sroa.26691.30.vec.insert1096, <4 x float> %669, i32 0, i32 0, i32 0), !dbg !244
  %671 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18726.22.vec.insert1085, <4 x i16> %.sroa.0676.6.vec.insert1099, <4 x float> %431, i32 0, i32 0, i32 0), !dbg !240
  %672 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26731.30.vec.insert1084, <4 x i16> %.sroa.10681.14.vec.insert1098, <4 x float> %671, i32 0, i32 0, i32 0), !dbg !244
  %673 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18726.22.vec.insert1085, <4 x i16> %.sroa.18686.22.vec.insert1097, <4 x float> %430, i32 0, i32 0, i32 0), !dbg !240
  %674 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26731.30.vec.insert1084, <4 x i16> %.sroa.26691.30.vec.insert1096, <4 x float> %673, i32 0, i32 0, i32 0), !dbg !244
  %675 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0716.6.vec.insert1087, <4 x i16> %.sroa.0643.6.vec.insert1095, <4 x float> %.val33.i443933, i32 0, i32 0, i32 0), !dbg !247
  %676 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10721.14.vec.insert1086, <4 x i16> %.sroa.10.14.vec.insert1094, <4 x float> %675, i32 0, i32 0, i32 0), !dbg !251
  %677 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0716.6.vec.insert1087, <4 x i16> %.sroa.18.22.vec.insert1093, <4 x float> %429, i32 0, i32 0, i32 0), !dbg !247
  %678 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10721.14.vec.insert1086, <4 x i16> %.sroa.26.30.vec.insert1092, <4 x float> %677, i32 0, i32 0, i32 0), !dbg !251
  %679 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18726.22.vec.insert1085, <4 x i16> %.sroa.0643.6.vec.insert1095, <4 x float> %428, i32 0, i32 0, i32 0), !dbg !247
  %680 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26731.30.vec.insert1084, <4 x i16> %.sroa.10.14.vec.insert1094, <4 x float> %679, i32 0, i32 0, i32 0), !dbg !251
  %681 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18726.22.vec.insert1085, <4 x i16> %.sroa.18.22.vec.insert1093, <4 x float> %427, i32 0, i32 0, i32 0), !dbg !247
  %682 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26731.30.vec.insert1084, <4 x i16> %.sroa.26.30.vec.insert1092, <4 x float> %681, i32 0, i32 0, i32 0), !dbg !251
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !254
  tail call void @llvm.amdgcn.s.barrier(), !dbg !255
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !256
  br i1 %.not, label %684, label %683, !dbg !257

683:                                              ; preds = %554
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !258, !srcloc !259
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %288, i64 %.sroa.0640.2.off0) #7, !dbg !260, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %292, i64 %.sroa.0640.2.off64) #7, !dbg !266, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %296, i64 %.sroa.5641.2.off0) #7, !dbg !260, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %300, i64 %.sroa.5641.2.off64) #7, !dbg !266, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %306, i64 %.sroa.0638.2.off0) #7, !dbg !268, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %310, i64 %.sroa.0638.2.off64) #7, !dbg !272, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %314, i64 %.sroa.5.2.off0) #7, !dbg !268, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %318, i64 %.sroa.5.2.off64) #7, !dbg !272, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %322, i64 %.sroa.8639.2.off0) #7, !dbg !268, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %326, i64 %.sroa.8639.2.off64) #7, !dbg !272, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %330, i64 %.sroa.11.2.off0) #7, !dbg !268, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %334, i64 %.sroa.11.2.off64) #7, !dbg !272, !srcloc !265
  br label %684, !dbg !274

684:                                              ; preds = %683, %554
  tail call void @llvm.amdgcn.s.barrier(), !dbg !275
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !276
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !277
  %685 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34776.38.vec.insert1075, <4 x i16> %.sroa.34696.38.vec.insert1083, <4 x float> %652, i32 0, i32 0, i32 0), !dbg !278
  %686 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42781.46.vec.insert1074, <4 x i16> %.sroa.42701.46.vec.insert1082, <4 x float> %685, i32 0, i32 0, i32 0), !dbg !282
  %687 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34776.38.vec.insert1075, <4 x i16> %.sroa.50706.54.vec.insert1081, <4 x float> %654, i32 0, i32 0, i32 0), !dbg !278
  %688 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42781.46.vec.insert1074, <4 x i16> %.sroa.58711.62.vec.insert1080, <4 x float> %687, i32 0, i32 0, i32 0), !dbg !282
  %689 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50786.54.vec.insert1073, <4 x i16> %.sroa.34696.38.vec.insert1083, <4 x float> %656, i32 0, i32 0, i32 0), !dbg !278
  %690 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58791.62.vec.insert1072, <4 x i16> %.sroa.42701.46.vec.insert1082, <4 x float> %689, i32 0, i32 0, i32 0), !dbg !282
  %691 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50786.54.vec.insert1073, <4 x i16> %.sroa.50706.54.vec.insert1081, <4 x float> %658, i32 0, i32 0, i32 0), !dbg !278
  %692 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58791.62.vec.insert1072, <4 x i16> %.sroa.58711.62.vec.insert1080, <4 x float> %691, i32 0, i32 0, i32 0), !dbg !282
  %693 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34776.38.vec.insert1075, <4 x i16> %.sroa.34.38.vec.insert1079, <4 x float> %660, i32 0, i32 0, i32 0), !dbg !285
  %694 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42781.46.vec.insert1074, <4 x i16> %.sroa.42.46.vec.insert1078, <4 x float> %693, i32 0, i32 0, i32 0), !dbg !289
  %695 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34776.38.vec.insert1075, <4 x i16> %.sroa.50.54.vec.insert1077, <4 x float> %662, i32 0, i32 0, i32 0), !dbg !285
  %696 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42781.46.vec.insert1074, <4 x i16> %.sroa.58.62.vec.insert1076, <4 x float> %695, i32 0, i32 0, i32 0), !dbg !289
  %697 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50786.54.vec.insert1073, <4 x i16> %.sroa.34.38.vec.insert1079, <4 x float> %664, i32 0, i32 0, i32 0), !dbg !285
  %698 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58791.62.vec.insert1072, <4 x i16> %.sroa.42.46.vec.insert1078, <4 x float> %697, i32 0, i32 0, i32 0), !dbg !289
  %699 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50786.54.vec.insert1073, <4 x i16> %.sroa.50.54.vec.insert1077, <4 x float> %666, i32 0, i32 0, i32 0), !dbg !285
  %700 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58791.62.vec.insert1072, <4 x i16> %.sroa.58.62.vec.insert1076, <4 x float> %699, i32 0, i32 0, i32 0), !dbg !289
  %701 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34736.38.vec.insert1071, <4 x i16> %.sroa.34696.38.vec.insert1083, <4 x float> %668, i32 0, i32 0, i32 0), !dbg !292
  %702 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42741.46.vec.insert1070, <4 x i16> %.sroa.42701.46.vec.insert1082, <4 x float> %701, i32 0, i32 0, i32 0), !dbg !296
  %703 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34736.38.vec.insert1071, <4 x i16> %.sroa.50706.54.vec.insert1081, <4 x float> %670, i32 0, i32 0, i32 0), !dbg !292
  %704 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42741.46.vec.insert1070, <4 x i16> %.sroa.58711.62.vec.insert1080, <4 x float> %703, i32 0, i32 0, i32 0), !dbg !296
  %705 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50746.54.vec.insert1069, <4 x i16> %.sroa.34696.38.vec.insert1083, <4 x float> %672, i32 0, i32 0, i32 0), !dbg !292
  %706 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58751.62.vec.insert1068, <4 x i16> %.sroa.42701.46.vec.insert1082, <4 x float> %705, i32 0, i32 0, i32 0), !dbg !296
  %707 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50746.54.vec.insert1069, <4 x i16> %.sroa.50706.54.vec.insert1081, <4 x float> %674, i32 0, i32 0, i32 0), !dbg !292
  %708 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58751.62.vec.insert1068, <4 x i16> %.sroa.58711.62.vec.insert1080, <4 x float> %707, i32 0, i32 0, i32 0), !dbg !296
  %709 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34736.38.vec.insert1071, <4 x i16> %.sroa.34.38.vec.insert1079, <4 x float> %676, i32 0, i32 0, i32 0), !dbg !299
  %710 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42741.46.vec.insert1070, <4 x i16> %.sroa.42.46.vec.insert1078, <4 x float> %709, i32 0, i32 0, i32 0), !dbg !303
  %711 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34736.38.vec.insert1071, <4 x i16> %.sroa.50.54.vec.insert1077, <4 x float> %678, i32 0, i32 0, i32 0), !dbg !299
  %712 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42741.46.vec.insert1070, <4 x i16> %.sroa.58.62.vec.insert1076, <4 x float> %711, i32 0, i32 0, i32 0), !dbg !303
  %713 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50746.54.vec.insert1069, <4 x i16> %.sroa.34.38.vec.insert1079, <4 x float> %680, i32 0, i32 0, i32 0), !dbg !299
  %714 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58751.62.vec.insert1068, <4 x i16> %.sroa.42.46.vec.insert1078, <4 x float> %713, i32 0, i32 0, i32 0), !dbg !303
  %715 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50746.54.vec.insert1069, <4 x i16> %.sroa.50.54.vec.insert1077, <4 x float> %682, i32 0, i32 0, i32 0), !dbg !299
  %716 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58751.62.vec.insert1068, <4 x i16> %.sroa.58.62.vec.insert1076, <4 x float> %715, i32 0, i32 0, i32 0), !dbg !303
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !306
  tail call void @llvm.amdgcn.s.barrier(), !dbg !307
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !308
  %exitcond.not = icmp eq i32 %436, 128, !dbg !309
  br i1 %exitcond.not, label %425, label %426, !dbg !134, !llvm.loop !310

717:                                              ; preds = %425
  tail call void @llvm.amdgcn.s.barrier(), !dbg !313
  br label %718, !dbg !314

718:                                              ; preds = %717, %425
  %719 = shl nsw i32 %.sroa.0.0.vec.extract.i, 2
  %720 = or disjoint i32 %719, %14
  %721 = shl nsw i32 %.sroa.0.4.vec.extract.i, 3
  %722 = or disjoint i32 %721, %15, !dbg !315
  %723 = shl nsw i32 %720, 5, !dbg !316
  %724 = shl nsw i32 %722, 5, !dbg !324
  %725 = sext i32 %723 to i64, !dbg !325
  %726 = mul i64 %.sroa.20.0.copyload, %725, !dbg !328
  %727 = sext i32 %724 to i64, !dbg !329
  %728 = getelementptr float, ptr %.sroa.17807.0.copyload, i64 %726, !dbg !330
  %729 = getelementptr float, ptr %728, i64 %727, !dbg !330
  %.sroa.0.0.vec.extract = extractelement <4 x float> %686, i64 0, !dbg !331
  %730 = getelementptr inbounds float, ptr %729, i64 %345, !dbg !332
  store float %.sroa.0.0.vec.extract, ptr %730, align 4, !dbg !333, !tbaa !334
  %.sroa.0.4.vec.extract = extractelement <4 x float> %686, i64 1, !dbg !338
  %731 = getelementptr inbounds float, ptr %729, i64 %347, !dbg !339
  store float %.sroa.0.4.vec.extract, ptr %731, align 4, !dbg !340, !tbaa !334
  %.sroa.0.8.vec.extract = extractelement <4 x float> %686, i64 2, !dbg !341
  %732 = getelementptr inbounds float, ptr %729, i64 %349, !dbg !342
  store float %.sroa.0.8.vec.extract, ptr %732, align 4, !dbg !343, !tbaa !334
  %.sroa.0.12.vec.extract = extractelement <4 x float> %686, i64 3, !dbg !344
  %733 = getelementptr inbounds float, ptr %729, i64 %351, !dbg !345
  store float %.sroa.0.12.vec.extract, ptr %733, align 4, !dbg !346, !tbaa !334
  %.sroa.12.16.vec.extract = extractelement <4 x float> %688, i64 0, !dbg !331
  %734 = getelementptr inbounds float, ptr %729, i64 %353, !dbg !332
  store float %.sroa.12.16.vec.extract, ptr %734, align 4, !dbg !333, !tbaa !334
  %.sroa.12.20.vec.extract = extractelement <4 x float> %688, i64 1, !dbg !338
  %735 = getelementptr inbounds float, ptr %729, i64 %355, !dbg !339
  store float %.sroa.12.20.vec.extract, ptr %735, align 4, !dbg !340, !tbaa !334
  %.sroa.12.24.vec.extract = extractelement <4 x float> %688, i64 2, !dbg !341
  %736 = getelementptr inbounds float, ptr %729, i64 %357, !dbg !342
  store float %.sroa.12.24.vec.extract, ptr %736, align 4, !dbg !343, !tbaa !334
  %.sroa.12.28.vec.extract = extractelement <4 x float> %688, i64 3, !dbg !344
  %737 = getelementptr inbounds float, ptr %729, i64 %359, !dbg !345
  store float %.sroa.12.28.vec.extract, ptr %737, align 4, !dbg !346, !tbaa !334
  %.sroa.22.32.vec.extract = extractelement <4 x float> %690, i64 0, !dbg !331
  %738 = getelementptr inbounds float, ptr %729, i64 %369, !dbg !332
  store float %.sroa.22.32.vec.extract, ptr %738, align 4, !dbg !333, !tbaa !334
  %.sroa.22.36.vec.extract = extractelement <4 x float> %690, i64 1, !dbg !338
  %739 = getelementptr inbounds float, ptr %729, i64 %371, !dbg !339
  store float %.sroa.22.36.vec.extract, ptr %739, align 4, !dbg !340, !tbaa !334
  %.sroa.22.40.vec.extract = extractelement <4 x float> %690, i64 2, !dbg !341
  %740 = getelementptr inbounds float, ptr %729, i64 %373, !dbg !342
  store float %.sroa.22.40.vec.extract, ptr %740, align 4, !dbg !343, !tbaa !334
  %.sroa.22.44.vec.extract = extractelement <4 x float> %690, i64 3, !dbg !344
  %741 = getelementptr inbounds float, ptr %729, i64 %375, !dbg !345
  store float %.sroa.22.44.vec.extract, ptr %741, align 4, !dbg !346, !tbaa !334
  %.sroa.32.48.vec.extract = extractelement <4 x float> %692, i64 0, !dbg !331
  %742 = getelementptr inbounds float, ptr %729, i64 %377, !dbg !332
  store float %.sroa.32.48.vec.extract, ptr %742, align 4, !dbg !333, !tbaa !334
  %.sroa.32.52.vec.extract = extractelement <4 x float> %692, i64 1, !dbg !338
  %743 = getelementptr inbounds float, ptr %729, i64 %379, !dbg !339
  store float %.sroa.32.52.vec.extract, ptr %743, align 4, !dbg !340, !tbaa !334
  %.sroa.32.56.vec.extract = extractelement <4 x float> %692, i64 2, !dbg !341
  %744 = getelementptr inbounds float, ptr %729, i64 %381, !dbg !342
  store float %.sroa.32.56.vec.extract, ptr %744, align 4, !dbg !343, !tbaa !334
  %.sroa.32.60.vec.extract = extractelement <4 x float> %692, i64 3, !dbg !344
  %745 = getelementptr inbounds float, ptr %729, i64 %383, !dbg !345
  store float %.sroa.32.60.vec.extract, ptr %745, align 4, !dbg !346, !tbaa !334
  %746 = or disjoint i32 %721, %15, !dbg !315
  %747 = shl i32 %746, 5, !dbg !324
  %748 = or disjoint i32 %747, 128, !dbg !324
  %749 = sext i32 %748 to i64, !dbg !329
  %750 = getelementptr float, ptr %728, i64 %749, !dbg !330
  %.sroa.42.64.vec.extract = extractelement <4 x float> %694, i64 0, !dbg !331
  %751 = getelementptr inbounds float, ptr %750, i64 %345, !dbg !332
  store float %.sroa.42.64.vec.extract, ptr %751, align 4, !dbg !333, !tbaa !334
  %.sroa.42.68.vec.extract = extractelement <4 x float> %694, i64 1, !dbg !338
  %752 = getelementptr inbounds float, ptr %750, i64 %347, !dbg !339
  store float %.sroa.42.68.vec.extract, ptr %752, align 4, !dbg !340, !tbaa !334
  %.sroa.42.72.vec.extract = extractelement <4 x float> %694, i64 2, !dbg !341
  %753 = getelementptr inbounds float, ptr %750, i64 %349, !dbg !342
  store float %.sroa.42.72.vec.extract, ptr %753, align 4, !dbg !343, !tbaa !334
  %.sroa.42.76.vec.extract = extractelement <4 x float> %694, i64 3, !dbg !344
  %754 = getelementptr inbounds float, ptr %750, i64 %351, !dbg !345
  store float %.sroa.42.76.vec.extract, ptr %754, align 4, !dbg !346, !tbaa !334
  %.sroa.52.80.vec.extract = extractelement <4 x float> %696, i64 0, !dbg !331
  %755 = getelementptr inbounds float, ptr %750, i64 %353, !dbg !332
  store float %.sroa.52.80.vec.extract, ptr %755, align 4, !dbg !333, !tbaa !334
  %.sroa.52.84.vec.extract = extractelement <4 x float> %696, i64 1, !dbg !338
  %756 = getelementptr inbounds float, ptr %750, i64 %355, !dbg !339
  store float %.sroa.52.84.vec.extract, ptr %756, align 4, !dbg !340, !tbaa !334
  %.sroa.52.88.vec.extract = extractelement <4 x float> %696, i64 2, !dbg !341
  %757 = getelementptr inbounds float, ptr %750, i64 %357, !dbg !342
  store float %.sroa.52.88.vec.extract, ptr %757, align 4, !dbg !343, !tbaa !334
  %.sroa.52.92.vec.extract = extractelement <4 x float> %696, i64 3, !dbg !344
  %758 = getelementptr inbounds float, ptr %750, i64 %359, !dbg !345
  store float %.sroa.52.92.vec.extract, ptr %758, align 4, !dbg !346, !tbaa !334
  %.sroa.62.96.vec.extract = extractelement <4 x float> %698, i64 0, !dbg !331
  %759 = getelementptr inbounds float, ptr %750, i64 %369, !dbg !332
  store float %.sroa.62.96.vec.extract, ptr %759, align 4, !dbg !333, !tbaa !334
  %.sroa.62.100.vec.extract = extractelement <4 x float> %698, i64 1, !dbg !338
  %760 = getelementptr inbounds float, ptr %750, i64 %371, !dbg !339
  store float %.sroa.62.100.vec.extract, ptr %760, align 4, !dbg !340, !tbaa !334
  %.sroa.62.104.vec.extract = extractelement <4 x float> %698, i64 2, !dbg !341
  %761 = getelementptr inbounds float, ptr %750, i64 %373, !dbg !342
  store float %.sroa.62.104.vec.extract, ptr %761, align 4, !dbg !343, !tbaa !334
  %.sroa.62.108.vec.extract = extractelement <4 x float> %698, i64 3, !dbg !344
  %762 = getelementptr inbounds float, ptr %750, i64 %375, !dbg !345
  store float %.sroa.62.108.vec.extract, ptr %762, align 4, !dbg !346, !tbaa !334
  %.sroa.72.112.vec.extract = extractelement <4 x float> %700, i64 0, !dbg !331
  %763 = getelementptr inbounds float, ptr %750, i64 %377, !dbg !332
  store float %.sroa.72.112.vec.extract, ptr %763, align 4, !dbg !333, !tbaa !334
  %.sroa.72.116.vec.extract = extractelement <4 x float> %700, i64 1, !dbg !338
  %764 = getelementptr inbounds float, ptr %750, i64 %379, !dbg !339
  store float %.sroa.72.116.vec.extract, ptr %764, align 4, !dbg !340, !tbaa !334
  %.sroa.72.120.vec.extract = extractelement <4 x float> %700, i64 2, !dbg !341
  %765 = getelementptr inbounds float, ptr %750, i64 %381, !dbg !342
  store float %.sroa.72.120.vec.extract, ptr %765, align 4, !dbg !343, !tbaa !334
  %.sroa.72.124.vec.extract = extractelement <4 x float> %700, i64 3, !dbg !344
  %766 = getelementptr inbounds float, ptr %750, i64 %383, !dbg !345
  store float %.sroa.72.124.vec.extract, ptr %766, align 4, !dbg !346, !tbaa !334
  %767 = shl i32 %720, 5, !dbg !316
  %768 = add i32 %767, 64, !dbg !316
  %769 = sext i32 %768 to i64, !dbg !325
  %770 = mul i64 %.sroa.20.0.copyload, %769, !dbg !328
  %771 = getelementptr float, ptr %.sroa.17807.0.copyload, i64 %770, !dbg !330
  %772 = getelementptr float, ptr %771, i64 %727, !dbg !330
  %.sroa.82.128.vec.extract = extractelement <4 x float> %702, i64 0, !dbg !331
  %773 = getelementptr inbounds float, ptr %772, i64 %345, !dbg !332
  store float %.sroa.82.128.vec.extract, ptr %773, align 4, !dbg !333, !tbaa !334
  %.sroa.82.132.vec.extract = extractelement <4 x float> %702, i64 1, !dbg !338
  %774 = getelementptr inbounds float, ptr %772, i64 %347, !dbg !339
  store float %.sroa.82.132.vec.extract, ptr %774, align 4, !dbg !340, !tbaa !334
  %.sroa.82.136.vec.extract = extractelement <4 x float> %702, i64 2, !dbg !341
  %775 = getelementptr inbounds float, ptr %772, i64 %349, !dbg !342
  store float %.sroa.82.136.vec.extract, ptr %775, align 4, !dbg !343, !tbaa !334
  %.sroa.82.140.vec.extract = extractelement <4 x float> %702, i64 3, !dbg !344
  %776 = getelementptr inbounds float, ptr %772, i64 %351, !dbg !345
  store float %.sroa.82.140.vec.extract, ptr %776, align 4, !dbg !346, !tbaa !334
  %.sroa.92.144.vec.extract = extractelement <4 x float> %704, i64 0, !dbg !331
  %777 = getelementptr inbounds float, ptr %772, i64 %353, !dbg !332
  store float %.sroa.92.144.vec.extract, ptr %777, align 4, !dbg !333, !tbaa !334
  %.sroa.92.148.vec.extract = extractelement <4 x float> %704, i64 1, !dbg !338
  %778 = getelementptr inbounds float, ptr %772, i64 %355, !dbg !339
  store float %.sroa.92.148.vec.extract, ptr %778, align 4, !dbg !340, !tbaa !334
  %.sroa.92.152.vec.extract = extractelement <4 x float> %704, i64 2, !dbg !341
  %779 = getelementptr inbounds float, ptr %772, i64 %357, !dbg !342
  store float %.sroa.92.152.vec.extract, ptr %779, align 4, !dbg !343, !tbaa !334
  %.sroa.92.156.vec.extract = extractelement <4 x float> %704, i64 3, !dbg !344
  %780 = getelementptr inbounds float, ptr %772, i64 %359, !dbg !345
  store float %.sroa.92.156.vec.extract, ptr %780, align 4, !dbg !346, !tbaa !334
  %.sroa.102.160.vec.extract = extractelement <4 x float> %706, i64 0, !dbg !331
  %781 = getelementptr inbounds float, ptr %772, i64 %369, !dbg !332
  store float %.sroa.102.160.vec.extract, ptr %781, align 4, !dbg !333, !tbaa !334
  %.sroa.102.164.vec.extract = extractelement <4 x float> %706, i64 1, !dbg !338
  %782 = getelementptr inbounds float, ptr %772, i64 %371, !dbg !339
  store float %.sroa.102.164.vec.extract, ptr %782, align 4, !dbg !340, !tbaa !334
  %.sroa.102.168.vec.extract = extractelement <4 x float> %706, i64 2, !dbg !341
  %783 = getelementptr inbounds float, ptr %772, i64 %373, !dbg !342
  store float %.sroa.102.168.vec.extract, ptr %783, align 4, !dbg !343, !tbaa !334
  %.sroa.102.172.vec.extract = extractelement <4 x float> %706, i64 3, !dbg !344
  %784 = getelementptr inbounds float, ptr %772, i64 %375, !dbg !345
  store float %.sroa.102.172.vec.extract, ptr %784, align 4, !dbg !346, !tbaa !334
  %.sroa.112.176.vec.extract = extractelement <4 x float> %708, i64 0, !dbg !331
  %785 = getelementptr inbounds float, ptr %772, i64 %377, !dbg !332
  store float %.sroa.112.176.vec.extract, ptr %785, align 4, !dbg !333, !tbaa !334
  %.sroa.112.180.vec.extract = extractelement <4 x float> %708, i64 1, !dbg !338
  %786 = getelementptr inbounds float, ptr %772, i64 %379, !dbg !339
  store float %.sroa.112.180.vec.extract, ptr %786, align 4, !dbg !340, !tbaa !334
  %.sroa.112.184.vec.extract = extractelement <4 x float> %708, i64 2, !dbg !341
  %787 = getelementptr inbounds float, ptr %772, i64 %381, !dbg !342
  store float %.sroa.112.184.vec.extract, ptr %787, align 4, !dbg !343, !tbaa !334
  %.sroa.112.188.vec.extract = extractelement <4 x float> %708, i64 3, !dbg !344
  %788 = getelementptr inbounds float, ptr %772, i64 %383, !dbg !345
  store float %.sroa.112.188.vec.extract, ptr %788, align 4, !dbg !346, !tbaa !334
  %789 = getelementptr float, ptr %771, i64 %749, !dbg !330
  %.sroa.122.192.vec.extract = extractelement <4 x float> %710, i64 0, !dbg !331
  %790 = getelementptr inbounds float, ptr %789, i64 %345, !dbg !332
  store float %.sroa.122.192.vec.extract, ptr %790, align 4, !dbg !333, !tbaa !334
  %.sroa.122.196.vec.extract = extractelement <4 x float> %710, i64 1, !dbg !338
  %791 = getelementptr inbounds float, ptr %789, i64 %347, !dbg !339
  store float %.sroa.122.196.vec.extract, ptr %791, align 4, !dbg !340, !tbaa !334
  %.sroa.122.200.vec.extract = extractelement <4 x float> %710, i64 2, !dbg !341
  %792 = getelementptr inbounds float, ptr %789, i64 %349, !dbg !342
  store float %.sroa.122.200.vec.extract, ptr %792, align 4, !dbg !343, !tbaa !334
  %.sroa.122.204.vec.extract = extractelement <4 x float> %710, i64 3, !dbg !344
  %793 = getelementptr inbounds float, ptr %789, i64 %351, !dbg !345
  store float %.sroa.122.204.vec.extract, ptr %793, align 4, !dbg !346, !tbaa !334
  %.sroa.132.208.vec.extract = extractelement <4 x float> %712, i64 0, !dbg !331
  %794 = getelementptr inbounds float, ptr %789, i64 %353, !dbg !332
  store float %.sroa.132.208.vec.extract, ptr %794, align 4, !dbg !333, !tbaa !334
  %.sroa.132.212.vec.extract = extractelement <4 x float> %712, i64 1, !dbg !338
  %795 = getelementptr inbounds float, ptr %789, i64 %355, !dbg !339
  store float %.sroa.132.212.vec.extract, ptr %795, align 4, !dbg !340, !tbaa !334
  %.sroa.132.216.vec.extract = extractelement <4 x float> %712, i64 2, !dbg !341
  %796 = getelementptr inbounds float, ptr %789, i64 %357, !dbg !342
  store float %.sroa.132.216.vec.extract, ptr %796, align 4, !dbg !343, !tbaa !334
  %.sroa.132.220.vec.extract = extractelement <4 x float> %712, i64 3, !dbg !344
  %797 = getelementptr inbounds float, ptr %789, i64 %359, !dbg !345
  store float %.sroa.132.220.vec.extract, ptr %797, align 4, !dbg !346, !tbaa !334
  %.sroa.142.224.vec.extract = extractelement <4 x float> %714, i64 0, !dbg !331
  %798 = getelementptr inbounds float, ptr %789, i64 %369, !dbg !332
  store float %.sroa.142.224.vec.extract, ptr %798, align 4, !dbg !333, !tbaa !334
  %.sroa.142.228.vec.extract = extractelement <4 x float> %714, i64 1, !dbg !338
  %799 = getelementptr inbounds float, ptr %789, i64 %371, !dbg !339
  store float %.sroa.142.228.vec.extract, ptr %799, align 4, !dbg !340, !tbaa !334
  %.sroa.142.232.vec.extract = extractelement <4 x float> %714, i64 2, !dbg !341
  %800 = getelementptr inbounds float, ptr %789, i64 %373, !dbg !342
  store float %.sroa.142.232.vec.extract, ptr %800, align 4, !dbg !343, !tbaa !334
  %.sroa.142.236.vec.extract = extractelement <4 x float> %714, i64 3, !dbg !344
  %801 = getelementptr inbounds float, ptr %789, i64 %375, !dbg !345
  store float %.sroa.142.236.vec.extract, ptr %801, align 4, !dbg !346, !tbaa !334
  %.sroa.152.240.vec.extract = extractelement <4 x float> %716, i64 0, !dbg !331
  %802 = getelementptr inbounds float, ptr %789, i64 %377, !dbg !332
  store float %.sroa.152.240.vec.extract, ptr %802, align 4, !dbg !333, !tbaa !334
  %.sroa.152.244.vec.extract = extractelement <4 x float> %716, i64 1, !dbg !338
  %803 = getelementptr inbounds float, ptr %789, i64 %379, !dbg !339
  store float %.sroa.152.244.vec.extract, ptr %803, align 4, !dbg !340, !tbaa !334
  %.sroa.152.248.vec.extract = extractelement <4 x float> %716, i64 2, !dbg !341
  %804 = getelementptr inbounds float, ptr %789, i64 %381, !dbg !342
  store float %.sroa.152.248.vec.extract, ptr %804, align 4, !dbg !343, !tbaa !334
  %.sroa.152.252.vec.extract = extractelement <4 x float> %716, i64 3, !dbg !344
  %805 = getelementptr inbounds float, ptr %789, i64 %383, !dbg !345
  store float %.sroa.152.252.vec.extract, ptr %805, align 4, !dbg !346, !tbaa !334
  %806 = add nuw nsw i32 %.0, 1, !dbg !347
  br label %384, !dbg !348, !llvm.loop !349

.critedge:                                        ; preds = %_Z12get_task_idxi.exit
  ret void, !dbg !351
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
!1 = !DIFile(filename: "256_128_64_32_persist.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_persist")
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
!137 = !DILocation(line: 173, column: 13, scope: !9)
!138 = !DILocation(line: 116, column: 39, scope: !9)
!139 = !DILocation(line: 116, column: 43, scope: !9)
!140 = !DILocation(line: 118, column: 17, scope: !9)
!141 = !DILocation(line: 26, column: 23, scope: !142, inlinedAt: !144)
!142 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !143, file: !143, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!143 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_persist")
!144 = distinct !DILocation(line: 177, column: 39, scope: !145, inlinedAt: !146)
!145 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !143, file: !143, line: 139, type: !10, scopeLine: 142, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!146 = distinct !DILocation(line: 119, column: 17, scope: !9)
!147 = !DILocation(line: 121, column: 13, scope: !9)
!148 = !DILocation(line: 122, column: 13, scope: !9)
!149 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !151)
!150 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !143, file: !143, line: 227, type: !10, scopeLine: 227, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!151 = distinct !DILocation(line: 285, column: 37, scope: !152, inlinedAt: !153)
!152 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 64>, 32, 32> >", scope: !143, file: !143, line: 247, type: !10, scopeLine: 247, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!153 = distinct !DILocation(line: 124, column: 13, scope: !9)
!154 = !{i64 4294578}
!155 = !DILocation(line: 287, column: 47, scope: !152, inlinedAt: !153)
!156 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !159)
!157 = distinct !DISubprogram(name: "operator=", scope: !158, file: !158, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!158 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!159 = distinct !DILocation(line: 288, column: 45, scope: !152, inlinedAt: !153)
!160 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !161)
!161 = distinct !DILocation(line: 285, column: 37, scope: !152, inlinedAt: !162)
!162 = distinct !DILocation(line: 125, column: 13, scope: !9)
!163 = !DILocation(line: 287, column: 47, scope: !152, inlinedAt: !162)
!164 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !165)
!165 = distinct !DILocation(line: 288, column: 45, scope: !152, inlinedAt: !162)
!166 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !167)
!167 = distinct !DILocation(line: 285, column: 37, scope: !168, inlinedAt: !169)
!168 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 32, 32> >", scope: !143, file: !143, line: 247, type: !10, scopeLine: 247, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!169 = distinct !DILocation(line: 126, column: 13, scope: !9)
!170 = !DILocation(line: 287, column: 47, scope: !168, inlinedAt: !169)
!171 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !172)
!172 = distinct !DILocation(line: 288, column: 45, scope: !168, inlinedAt: !169)
!173 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !174)
!174 = distinct !DILocation(line: 285, column: 37, scope: !168, inlinedAt: !175)
!175 = distinct !DILocation(line: 127, column: 13, scope: !9)
!176 = !DILocation(line: 287, column: 47, scope: !168, inlinedAt: !175)
!177 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !178)
!178 = distinct !DILocation(line: 288, column: 45, scope: !168, inlinedAt: !175)
!179 = !DILocation(line: 128, column: 13, scope: !9)
!180 = !DILocation(line: 130, column: 17, scope: !9)
!181 = !DILocation(line: 61, column: 18, scope: !111, inlinedAt: !182)
!182 = distinct !DILocation(line: 153, column: 39, scope: !183, inlinedAt: !184)
!183 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !143, file: !143, line: 139, type: !10, scopeLine: 142, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!184 = distinct !DILocation(line: 131, column: 17, scope: !9)
!185 = !DILocation(line: 63, column: 68, scope: !87, inlinedAt: !186)
!186 = distinct !DILocation(line: 154, column: 24, scope: !183, inlinedAt: !184)
!187 = !DILocation(line: 63, column: 16, scope: !87, inlinedAt: !186)
!188 = !DILocation(line: 32, column: 28, scope: !189, inlinedAt: !190)
!189 = distinct !DISubprogram(name: "make_srsrc", scope: !143, file: !143, line: 31, type: !10, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!190 = distinct !DILocation(line: 160, column: 19, scope: !183, inlinedAt: !184)
!191 = !DILocation(line: 26, column: 23, scope: !142, inlinedAt: !192)
!192 = distinct !DILocation(line: 177, column: 39, scope: !183, inlinedAt: !184)
!193 = !DILocation(line: 133, column: 13, scope: !9)
!194 = !DILocation(line: 134, column: 13, scope: !9)
!195 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !196)
!196 = distinct !DILocation(line: 285, column: 37, scope: !152, inlinedAt: !197)
!197 = distinct !DILocation(line: 136, column: 13, scope: !9)
!198 = !DILocation(line: 287, column: 47, scope: !152, inlinedAt: !197)
!199 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !200)
!200 = distinct !DILocation(line: 288, column: 45, scope: !152, inlinedAt: !197)
!201 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !202)
!202 = distinct !DILocation(line: 285, column: 37, scope: !152, inlinedAt: !203)
!203 = distinct !DILocation(line: 137, column: 13, scope: !9)
!204 = !DILocation(line: 287, column: 47, scope: !152, inlinedAt: !203)
!205 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !206)
!206 = distinct !DILocation(line: 288, column: 45, scope: !152, inlinedAt: !203)
!207 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !208)
!208 = distinct !DILocation(line: 285, column: 37, scope: !168, inlinedAt: !209)
!209 = distinct !DILocation(line: 138, column: 13, scope: !9)
!210 = !DILocation(line: 287, column: 47, scope: !168, inlinedAt: !209)
!211 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !212)
!212 = distinct !DILocation(line: 288, column: 45, scope: !168, inlinedAt: !209)
!213 = !DILocation(line: 229, column: 5, scope: !150, inlinedAt: !214)
!214 = distinct !DILocation(line: 285, column: 37, scope: !168, inlinedAt: !215)
!215 = distinct !DILocation(line: 139, column: 13, scope: !9)
!216 = !DILocation(line: 287, column: 47, scope: !168, inlinedAt: !215)
!217 = !DILocation(line: 494, column: 7, scope: !157, inlinedAt: !218)
!218 = distinct !DILocation(line: 288, column: 45, scope: !168, inlinedAt: !215)
!219 = !DILocation(line: 140, column: 13, scope: !9)
!220 = !DILocation(line: 141, column: 13, scope: !9)
!221 = !DILocation(line: 143, column: 13, scope: !9)
!222 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !225)
!223 = distinct !DISubprogram(name: "mfma161616", scope: !224, file: !224, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!224 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!225 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !227)
!226 = distinct !DISubprogram(name: "mma_ABt_base", scope: !224, file: !224, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!227 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !229)
!228 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !224, file: !224, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!229 = distinct !DILocation(line: 144, column: 13, scope: !9)
!230 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !231)
!231 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !232)
!232 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !229)
!233 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !234)
!234 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !235)
!235 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !236)
!236 = distinct !DILocation(line: 145, column: 13, scope: !9)
!237 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !238)
!238 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !239)
!239 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !236)
!240 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !241)
!241 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !242)
!242 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !243)
!243 = distinct !DILocation(line: 146, column: 13, scope: !9)
!244 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !245)
!245 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !246)
!246 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !243)
!247 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !248)
!248 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !249)
!249 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !250)
!250 = distinct !DILocation(line: 147, column: 13, scope: !9)
!251 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !252)
!252 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !253)
!253 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !250)
!254 = !DILocation(line: 148, column: 13, scope: !9)
!255 = !DILocation(line: 149, column: 13, scope: !9)
!256 = !DILocation(line: 150, column: 13, scope: !9)
!257 = !DILocation(line: 153, column: 17, scope: !9)
!258 = !DILocation(line: 154, column: 17, scope: !9)
!259 = !{i64 4305136}
!260 = !DILocation(line: 17, column: 5, scope: !261, inlinedAt: !262)
!261 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !143, file: !143, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!262 = distinct !DILocation(line: 219, column: 17, scope: !263, inlinedAt: !264)
!263 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !143, file: !143, line: 188, type: !10, scopeLine: 190, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!264 = distinct !DILocation(line: 155, column: 17, scope: !9)
!265 = !{i64 4290525}
!266 = !DILocation(line: 17, column: 5, scope: !261, inlinedAt: !267)
!267 = distinct !DILocation(line: 220, column: 17, scope: !263, inlinedAt: !264)
!268 = !DILocation(line: 17, column: 5, scope: !261, inlinedAt: !269)
!269 = distinct !DILocation(line: 219, column: 17, scope: !270, inlinedAt: !271)
!270 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 64>, 512>", scope: !143, file: !143, line: 188, type: !10, scopeLine: 190, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!271 = distinct !DILocation(line: 157, column: 17, scope: !9)
!272 = !DILocation(line: 17, column: 5, scope: !261, inlinedAt: !273)
!273 = distinct !DILocation(line: 220, column: 17, scope: !270, inlinedAt: !271)
!274 = !DILocation(line: 159, column: 13, scope: !9)
!275 = !DILocation(line: 160, column: 13, scope: !9)
!276 = !DILocation(line: 161, column: 13, scope: !9)
!277 = !DILocation(line: 163, column: 13, scope: !9)
!278 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !279)
!279 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !280)
!280 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !281)
!281 = distinct !DILocation(line: 164, column: 13, scope: !9)
!282 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !283)
!283 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !284)
!284 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !281)
!285 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !286)
!286 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !287)
!287 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !288)
!288 = distinct !DILocation(line: 165, column: 13, scope: !9)
!289 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !290)
!290 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !291)
!291 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !288)
!292 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !293)
!293 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !294)
!294 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !295)
!295 = distinct !DILocation(line: 166, column: 13, scope: !9)
!296 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !297)
!297 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !298)
!298 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !295)
!299 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !300)
!300 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !301)
!301 = distinct !DILocation(line: 216, column: 13, scope: !228, inlinedAt: !302)
!302 = distinct !DILocation(line: 167, column: 13, scope: !9)
!303 = !DILocation(line: 29, column: 27, scope: !223, inlinedAt: !304)
!304 = distinct !DILocation(line: 81, column: 5, scope: !226, inlinedAt: !305)
!305 = distinct !DILocation(line: 224, column: 17, scope: !228, inlinedAt: !302)
!306 = !DILocation(line: 168, column: 13, scope: !9)
!307 = !DILocation(line: 169, column: 13, scope: !9)
!308 = !DILocation(line: 170, column: 13, scope: !9)
!309 = !DILocation(line: 114, column: 33, scope: !9)
!310 = distinct !{!310, !134, !311, !312}
!311 = !DILocation(line: 171, column: 9, scope: !9)
!312 = !{!"llvm.loop.mustprogress"}
!313 = !DILocation(line: 174, column: 13, scope: !9)
!314 = !DILocation(line: 175, column: 9, scope: !9)
!315 = !DILocation(line: 184, column: 36, scope: !9)
!316 = !DILocation(line: 60, column: 34, scope: !317, inlinedAt: !318)
!317 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !78, file: !78, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!318 = distinct !DILocation(line: 192, column: 41, scope: !319, inlinedAt: !321)
!319 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !320, file: !320, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!320 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!321 = distinct !DILocation(line: 212, column: 5, scope: !322, inlinedAt: !323)
!322 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !320, file: !320, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!323 = distinct !DILocation(line: 181, column: 13, scope: !9)
!324 = !DILocation(line: 61, column: 18, scope: !317, inlinedAt: !318)
!325 = !DILocation(line: 63, column: 54, scope: !326, inlinedAt: !327)
!326 = distinct !DISubprogram(name: "operator[]", scope: !88, file: !88, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!327 = distinct !DILocation(line: 192, column: 23, scope: !319, inlinedAt: !321)
!328 = !DILocation(line: 63, column: 60, scope: !326, inlinedAt: !327)
!329 = !DILocation(line: 63, column: 68, scope: !326, inlinedAt: !327)
!330 = !DILocation(line: 63, column: 16, scope: !326, inlinedAt: !327)
!331 = !DILocation(line: 203, column: 49, scope: !319, inlinedAt: !321)
!332 = !DILocation(line: 203, column: 13, scope: !319, inlinedAt: !321)
!333 = !DILocation(line: 203, column: 47, scope: !319, inlinedAt: !321)
!334 = !{!335, !335, i64 0}
!335 = !{!"float", !336, i64 0}
!336 = !{!"omnipotent char", !337, i64 0}
!337 = !{!"Simple C++ TBAA"}
!338 = !DILocation(line: 204, column: 49, scope: !319, inlinedAt: !321)
!339 = !DILocation(line: 204, column: 13, scope: !319, inlinedAt: !321)
!340 = !DILocation(line: 204, column: 47, scope: !319, inlinedAt: !321)
!341 = !DILocation(line: 205, column: 49, scope: !319, inlinedAt: !321)
!342 = !DILocation(line: 205, column: 13, scope: !319, inlinedAt: !321)
!343 = !DILocation(line: 205, column: 47, scope: !319, inlinedAt: !321)
!344 = !DILocation(line: 206, column: 49, scope: !319, inlinedAt: !321)
!345 = !DILocation(line: 206, column: 13, scope: !319, inlinedAt: !321)
!346 = !DILocation(line: 206, column: 47, scope: !319, inlinedAt: !321)
!347 = !DILocation(line: 87, column: 43, scope: !9)
!348 = !DILocation(line: 87, column: 5, scope: !9)
!349 = distinct !{!349, !348, !350, !312}
!350 = !DILocation(line: 188, column: 5, scope: !9)
!351 = !DILocation(line: 189, column: 1, scope: !9)
