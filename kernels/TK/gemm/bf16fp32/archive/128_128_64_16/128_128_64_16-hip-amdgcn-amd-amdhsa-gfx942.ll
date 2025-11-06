; ModuleID = '128_128_64_16-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "128_128_64_16.cpp"
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
@__hip_cuid_910013c81390fca1 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_910013c81390fca1 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.0926.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.7.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.7.0.copyload = load i64, ptr addrspace(4) %.sroa.7.0..sroa_idx, align 8
  %.sroa.9930.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.9930.0.copyload = load ptr, ptr addrspace(4) %.sroa.9930.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.17933.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.17933.0.copyload = load ptr, ptr addrspace(4) %.sroa.17933.0..sroa_idx, align 8
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
  %8 = and i64 %7, 15, !dbg !24
  %reass.sub.i.i342 = and i64 %7, -16, !dbg !25
  %9 = add i64 %reass.sub.i.i342, 16, !dbg !25
  %10 = inttoptr i64 %9 to ptr, !dbg !25
  %.not.i.i341 = icmp eq i64 %8, 0, !dbg !26
  %11 = select i1 %.not.i.i341, ptr %6, ptr %10, !dbg !25
  %12 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y(), !dbg !27
  %13 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr(), !dbg !33
  %14 = load i32, ptr addrspace(4) %13, align 4, !dbg !33, !tbaa !38
  %15 = mul i32 %14, %12, !dbg !42
  %16 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x(), !dbg !43
  %17 = add i32 %15, %16, !dbg !48
  %18 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 4, !dbg !49
  %19 = load i32, ptr addrspace(4) %18, align 4, !dbg !49, !tbaa !38
  %20 = mul i32 %19, %14, !dbg !54
  %.frozen = freeze i32 %17, !dbg !55
  %21 = sdiv i32 %.frozen, 8, !dbg !55
  %22 = mul i32 %21, 8, !dbg !56
  %.decomposed = sub i32 %.frozen, %22, !dbg !56
  %23 = sdiv i32 %20, 8, !dbg !57
  %24 = mul nsw i32 %.decomposed, %23, !dbg !58
  %25 = add nsw i32 %24, %21, !dbg !59
  %.frozen1097 = freeze i32 %25, !dbg !60
  %26 = sdiv i32 %.frozen1097, 512, !dbg !60
  %27 = shl nsw i32 %26, 3, !dbg !61
  %28 = sub nsw i32 64, %27, !dbg !62
  %29 = tail call noundef range(i32 -33554360, 33554497) i32 @llvm.smin.i32(i32 %28, i32 8), !dbg !63
  %30 = mul i32 %26, 512, !dbg !67
  %.decomposed1098 = sub i32 %.frozen1097, %30, !dbg !67
  %31 = sdiv i32 %.decomposed1098, %29, !dbg !68
  %32 = mul i32 %31, %29, !dbg !69
  %.decomposed1099 = sub i32 %.decomposed1098, %32, !dbg !69
  %33 = add nsw i32 %27, %.decomposed1099, !dbg !70
  %34 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !71
  %35 = lshr i32 %34, 6, !dbg !78
  %36 = lshr i32 %34, 8, !dbg !79
  %37 = and i32 %35, 3, !dbg !80
  %38 = trunc i64 %.sroa.7.0.copyload to i32, !dbg !81
  %39 = shl nsw i32 %33, 7, !dbg !88
  %40 = sext i32 %39 to i64, !dbg !92
  %41 = mul i64 %.sroa.7.0.copyload, %40, !dbg !96
  %42 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.0926.0.copyload, i64 %41, !dbg !97
  %43 = ptrtoint ptr %5 to i64, !dbg !98
  %44 = shl nuw nsw i32 %34, 3
  %45 = and i32 %44, 56
  %46 = lshr i32 %34, 3, !dbg !99
  %47 = and i32 %46, 63, !dbg !99
  %48 = mul i32 %47, %38, !dbg !100
  %49 = add i32 %48, %45, !dbg !101
  %50 = sext i32 %49 to i64, !dbg !102
  %51 = getelementptr inbounds %struct.__hip_bfloat16, ptr %42, i64 %50, !dbg !102
  %52 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %51) #7, !dbg !103, !srcloc !107
  %53 = or i32 %46, 64, !dbg !99
  %54 = mul i32 %53, %38, !dbg !100
  %55 = add i32 %54, %45, !dbg !101
  %56 = sext i32 %55 to i64, !dbg !102
  %57 = getelementptr inbounds %struct.__hip_bfloat16, ptr %42, i64 %56, !dbg !102
  %58 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %57) #7, !dbg !103, !srcloc !107
  %59 = trunc i64 %43 to i32, !dbg !98
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !108, !srcloc !109
  %60 = shl nuw nsw i32 %34, 4
  %.masked.i.i = and i32 %60, 8064
  %61 = shl nuw nsw i32 %45, 1
  %62 = add i32 %61, %59
  %63 = or disjoint i32 %61, 8
  %64 = add i32 %63, %59
  %65 = add i32 %62, %.masked.i.i, !dbg !110
  %66 = lshr exact i32 %65, 4, !dbg !114
  %67 = and i32 %66, 120, !dbg !114
  %68 = xor i32 %67, %65, !dbg !115
  %.sroa.0.0.extract.trunc.i.i = trunc i128 %52 to i64, !dbg !116
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %68, i64 %.sroa.0.0.extract.trunc.i.i) #7, !dbg !117, !srcloc !120
  %69 = add i32 %64, %.masked.i.i, !dbg !121
  %70 = lshr i32 %69, 4, !dbg !123
  %71 = and i32 %70, 120, !dbg !123
  %72 = xor i32 %71, %69, !dbg !124
  %.sroa.0.8.extract.shift.i.i = lshr i128 %52, 64, !dbg !125
  %.sroa.0.8.extract.trunc.i.i = trunc nuw i128 %.sroa.0.8.extract.shift.i.i to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %72, i64 %.sroa.0.8.extract.trunc.i.i) #7, !dbg !126, !srcloc !120
  %73 = or disjoint i32 %.masked.i.i, 8192, !dbg !128
  %74 = add i32 %62, %73, !dbg !110
  %75 = lshr exact i32 %74, 4, !dbg !114
  %76 = and i32 %75, 120, !dbg !114
  %77 = xor i32 %76, %74, !dbg !115
  %.sroa.5.16.extract.trunc.i.i = trunc i128 %58 to i64, !dbg !116
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %77, i64 %.sroa.5.16.extract.trunc.i.i) #7, !dbg !117, !srcloc !120
  %78 = add i32 %64, %73, !dbg !121
  %79 = lshr i32 %78, 4, !dbg !123
  %80 = and i32 %79, 120, !dbg !123
  %81 = xor i32 %80, %78, !dbg !124
  %.sroa.5.24.extract.shift.i.i = lshr i128 %58, 64, !dbg !125
  %.sroa.5.24.extract.trunc.i.i = trunc nuw i128 %.sroa.5.24.extract.shift.i.i to i64, !dbg !125
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %81, i64 %.sroa.5.24.extract.trunc.i.i) #7, !dbg !126, !srcloc !120
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !129, !srcloc !130
  %82 = trunc i64 %.sroa.15.0.copyload to i32, !dbg !131
  %83 = shl nsw i32 %31, 7, !dbg !134
  %84 = sext i32 %83 to i64, !dbg !136
  %85 = mul i64 %.sroa.15.0.copyload, %84, !dbg !138
  %86 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.9930.0.copyload, i64 %85, !dbg !139
  %87 = ptrtoint ptr %11 to i64, !dbg !140
  %88 = mul i32 %47, %82, !dbg !141
  %89 = add i32 %88, %45, !dbg !142
  %90 = sext i32 %89 to i64, !dbg !143
  %91 = getelementptr inbounds %struct.__hip_bfloat16, ptr %86, i64 %90, !dbg !143
  %92 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %91) #7, !dbg !144, !srcloc !107
  %93 = mul i32 %53, %82, !dbg !141
  %94 = add i32 %93, %45, !dbg !142
  %95 = sext i32 %94 to i64, !dbg !143
  %96 = getelementptr inbounds %struct.__hip_bfloat16, ptr %86, i64 %95, !dbg !143
  %97 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %96) #7, !dbg !144, !srcloc !107
  %98 = trunc i64 %87 to i32, !dbg !140
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !146, !srcloc !109
  %99 = add i32 %61, %98
  %100 = add i32 %63, %98
  %101 = add i32 %99, %.masked.i.i, !dbg !147
  %102 = lshr exact i32 %101, 4, !dbg !149
  %103 = and i32 %102, 120, !dbg !149
  %104 = xor i32 %103, %101, !dbg !150
  %.sroa.0.0.extract.trunc.i.i347 = trunc i128 %92 to i64, !dbg !151
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %104, i64 %.sroa.0.0.extract.trunc.i.i347) #7, !dbg !152, !srcloc !120
  %105 = add i32 %100, %.masked.i.i, !dbg !154
  %106 = lshr i32 %105, 4, !dbg !156
  %107 = and i32 %106, 120, !dbg !156
  %108 = xor i32 %107, %105, !dbg !157
  %.sroa.0.8.extract.shift.i.i348 = lshr i128 %92, 64, !dbg !158
  %.sroa.0.8.extract.trunc.i.i349 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i348 to i64, !dbg !158
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %108, i64 %.sroa.0.8.extract.trunc.i.i349) #7, !dbg !159, !srcloc !120
  %109 = add i32 %99, %73, !dbg !147
  %110 = lshr exact i32 %109, 4, !dbg !149
  %111 = and i32 %110, 120, !dbg !149
  %112 = xor i32 %111, %109, !dbg !150
  %.sroa.5.16.extract.trunc.i.i350 = trunc i128 %97 to i64, !dbg !151
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %112, i64 %.sroa.5.16.extract.trunc.i.i350) #7, !dbg !152, !srcloc !120
  %113 = add i32 %100, %73, !dbg !154
  %114 = lshr i32 %113, 4, !dbg !156
  %115 = and i32 %114, 120, !dbg !156
  %116 = xor i32 %115, %113, !dbg !157
  %.sroa.5.24.extract.shift.i.i351 = lshr i128 %97, 64, !dbg !158
  %.sroa.5.24.extract.trunc.i.i352 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i351 to i64, !dbg !158
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %116, i64 %.sroa.5.24.extract.trunc.i.i352) #7, !dbg !159, !srcloc !120
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !161, !srcloc !130
  tail call void @llvm.amdgcn.s.barrier(), !dbg !162
  %117 = icmp eq i32 %36, 1, !dbg !163
  br i1 %117, label %118, label %119, !dbg !164

118:                                              ; preds = %1
  tail call void @llvm.amdgcn.s.barrier(), !dbg !165
  br label %119, !dbg !166

119:                                              ; preds = %118, %1
  %120 = and i32 %34, 15
  %121 = lshr i32 %34, 2
  %122 = and i32 %121, 12
  %123 = shl nuw nsw i32 %37, 11
  %124 = shl nuw nsw i32 %120, 7
  %125 = or disjoint i32 %123, %124
  %126 = shl nuw nsw i32 %122, 1
  %127 = add i32 %125, %98
  %128 = add i32 %127, %126
  %129 = lshr i32 %128, 4
  %130 = and i32 %129, 120
  %131 = xor i32 %130, %128
  %132 = shl i32 %55, 1
  %133 = shl i32 %49, 1
  %134 = shl i32 %38, 8
  %135 = or disjoint i32 %125, 8192
  %136 = add i32 %135, %98
  %137 = add i32 %136, %126
  %138 = lshr i32 %137, 4
  %139 = and i32 %138, 120
  %140 = xor i32 %139, %137
  %141 = shl nuw nsw i32 %36, 11
  %142 = or disjoint i32 %141, %124
  %143 = add i32 %142, %59
  %144 = add i32 %143, %126
  %145 = lshr i32 %144, 4
  %146 = and i32 %145, 120
  %147 = xor i32 %146, %144
  %148 = shl nuw nsw i32 %36, 4
  %149 = add nuw nsw i32 %148, 32
  %150 = or disjoint i32 %149, %120
  %151 = shl nuw nsw i32 %150, 7
  %152 = add i32 %151, %59
  %153 = add i32 %152, %126
  %154 = lshr i32 %153, 4
  %155 = and i32 %154, 120
  %156 = xor i32 %155, %153
  %157 = or disjoint i32 %142, 8192
  %158 = add i32 %157, %59
  %159 = add i32 %158, %126
  %160 = lshr i32 %159, 4
  %161 = and i32 %160, 120
  %162 = xor i32 %161, %159
  %163 = add nuw nsw i32 %148, 96
  %164 = or disjoint i32 %163, %120
  %165 = shl nuw nsw i32 %164, 7
  %166 = add i32 %165, %59
  %167 = add i32 %166, %126
  %168 = lshr i32 %167, 4
  %169 = and i32 %168, 120
  %170 = xor i32 %169, %167
  %171 = shl i32 %94, 1
  %172 = shl i32 %89, 1
  %173 = shl i32 %82, 8
  %174 = or disjoint i32 %126, 32
  %175 = add i32 %127, %174
  %176 = lshr i32 %175, 4
  %177 = and i32 %176, 120
  %178 = xor i32 %177, %175
  %179 = add i32 %136, %174
  %180 = lshr i32 %179, 4
  %181 = and i32 %180, 120
  %182 = xor i32 %181, %179
  %183 = add i32 %143, %174
  %184 = lshr i32 %183, 4
  %185 = and i32 %184, 120
  %186 = xor i32 %185, %183
  %187 = add i32 %152, %174
  %188 = lshr i32 %187, 4
  %189 = and i32 %188, 120
  %190 = xor i32 %189, %187
  %191 = add i32 %158, %174
  %192 = lshr i32 %191, 4
  %193 = and i32 %192, 120
  %194 = xor i32 %193, %191
  %195 = add i32 %166, %174
  %196 = lshr i32 %195, 4
  %197 = and i32 %196, 120
  %198 = xor i32 %197, %195
  %199 = or disjoint i32 %126, 64
  %200 = add i32 %127, %199
  %201 = lshr i32 %200, 4
  %202 = and i32 %201, 120
  %203 = xor i32 %202, %200
  %204 = add i32 %136, %199
  %205 = lshr i32 %204, 4
  %206 = and i32 %205, 120
  %207 = xor i32 %206, %204
  %208 = add i32 %143, %199
  %209 = lshr i32 %208, 4
  %210 = and i32 %209, 120
  %211 = xor i32 %210, %208
  %212 = add i32 %152, %199
  %213 = lshr i32 %212, 4
  %214 = and i32 %213, 120
  %215 = xor i32 %214, %212
  %216 = add i32 %158, %199
  %217 = lshr i32 %216, 4
  %218 = and i32 %217, 120
  %219 = xor i32 %218, %216
  %220 = add i32 %166, %199
  %221 = lshr i32 %220, 4
  %222 = and i32 %221, 120
  %223 = xor i32 %222, %220
  %224 = or disjoint i32 %126, 96
  %225 = add i32 %127, %224
  %226 = lshr i32 %225, 4
  %227 = and i32 %226, 120
  %228 = xor i32 %227, %225
  %229 = add i32 %136, %224
  %230 = lshr i32 %229, 4
  %231 = and i32 %230, 120
  %232 = xor i32 %231, %229
  %233 = add i32 %143, %224
  %234 = lshr i32 %233, 4
  %235 = and i32 %234, 120
  %236 = xor i32 %235, %233
  %237 = add i32 %152, %224
  %238 = lshr i32 %237, 4
  %239 = and i32 %238, 120
  %240 = xor i32 %239, %237
  %241 = add i32 %158, %224
  %242 = lshr i32 %241, 4
  %243 = and i32 %242, 120
  %244 = xor i32 %243, %241
  %245 = add i32 %166, %224
  %246 = lshr i32 %245, 4
  %247 = and i32 %246, 120
  %248 = xor i32 %247, %245
  %249 = and i32 %60, 112
  %250 = add i32 %249, %59
  %251 = or disjoint i32 %249, 8
  %252 = add i32 %251, %59
  %253 = add i32 %250, %.masked.i.i
  %254 = lshr exact i32 %253, 4
  %255 = and i32 %254, 120
  %256 = xor i32 %255, %253
  %257 = add i32 %252, %.masked.i.i
  %258 = lshr i32 %257, 4
  %259 = and i32 %258, 120
  %260 = xor i32 %259, %257
  %261 = add i32 %250, %73
  %262 = lshr exact i32 %261, 4
  %263 = and i32 %262, 120
  %264 = xor i32 %263, %261
  %265 = add i32 %252, %73
  %266 = lshr i32 %265, 4
  %267 = and i32 %266, 120
  %268 = xor i32 %267, %265
  %269 = add i32 %249, %98
  %270 = add i32 %251, %98
  %271 = add i32 %269, %.masked.i.i
  %272 = lshr exact i32 %271, 4
  %273 = and i32 %272, 120
  %274 = xor i32 %273, %271
  %275 = add i32 %270, %.masked.i.i
  %276 = lshr i32 %275, 4
  %277 = and i32 %276, 120
  %278 = xor i32 %277, %275
  %279 = add i32 %269, %73
  %280 = lshr exact i32 %279, 4
  %281 = and i32 %280, 120
  %282 = xor i32 %281, %279
  %283 = add i32 %270, %73
  %284 = lshr i32 %283, 4
  %285 = and i32 %284, 120
  %286 = xor i32 %285, %283
  br label %289, !dbg !167

287:                                              ; preds = %480
  %288 = icmp ult i32 %34, 256, !dbg !168
  br i1 %288, label %489, label %490, !dbg !169

289:                                              ; preds = %119, %480
  %.val315994 = phi <4 x float> [ zeroinitializer, %119 ], [ %488, %480 ]
  %.val318991 = phi <4 x float> [ zeroinitializer, %119 ], [ %487, %480 ]
  %.val321988 = phi <4 x float> [ zeroinitializer, %119 ], [ %486, %480 ]
  %.val324985 = phi <4 x float> [ zeroinitializer, %119 ], [ %485, %480 ]
  %.val327982 = phi <4 x float> [ zeroinitializer, %119 ], [ %484, %480 ]
  %.val330979 = phi <4 x float> [ zeroinitializer, %119 ], [ %483, %480 ]
  %.val333977 = phi <4 x float> [ zeroinitializer, %119 ], [ %482, %480 ]
  %.0239975 = phi i32 [ 0, %119 ], [ %290, %480 ]
  %.sroa.5752.0.off64974 = phi i64 [ 0, %119 ], [ %.sroa.5752.1.off64, %480 ]
  %.sroa.5752.0.off0973 = phi i64 [ undef, %119 ], [ %.sroa.5752.1.off0, %480 ]
  %.sroa.0751.0.off64972 = phi i64 [ 0, %119 ], [ %.sroa.0751.1.off64, %480 ]
  %.sroa.0751.0.off0971 = phi i64 [ undef, %119 ], [ %.sroa.0751.1.off0, %480 ]
  %.sroa.5.0.off64970 = phi i64 [ 0, %119 ], [ %.sroa.5.1.off64, %480 ]
  %.sroa.5.0.off0969 = phi i64 [ undef, %119 ], [ %.sroa.5.1.off0, %480 ]
  %.sroa.0749.0.off64968 = phi i64 [ 0, %119 ], [ %.sroa.0749.1.off64, %480 ]
  %.sroa.0749.0.off0967 = phi i64 [ undef, %119 ], [ %.sroa.0749.1.off0, %480 ]
  %.val336965966 = phi <4 x float> [ zeroinitializer, %119 ], [ %481, %480 ]
  %290 = add nuw nsw i32 %.0239975, 1, !dbg !170
  %.not = icmp eq i32 %.0239975, 127, !dbg !171
  br i1 %.not, label %300, label %291, !dbg !172

291:                                              ; preds = %289
  %292 = shl nuw nsw i32 %290, 6
  %293 = zext nneg i32 %292 to i64
  %294 = getelementptr %struct.__hip_bfloat16, ptr %42, i64 %293
  %295 = ptrtoint ptr %294 to i64
  %296 = bitcast i64 %295 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %296, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %297 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %297, i32 %134, i64 2
  %298 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %133, i32 noundef 0, i32 noundef 0) #7, !dbg !173
  %299 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %132, i32 noundef 0, i32 noundef 0) #7, !dbg !173
  %extract.t950 = trunc i128 %298 to i64, !dbg !179
  %extract952 = lshr i128 %298, 64, !dbg !179
  %extract.t953 = trunc nuw i128 %extract952 to i64, !dbg !179
  %extract.t957 = trunc i128 %299 to i64, !dbg !179
  %extract959 = lshr i128 %299, 64, !dbg !179
  %extract.t960 = trunc nuw i128 %extract959 to i64, !dbg !179
  br label %300, !dbg !179

300:                                              ; preds = %291, %289
  %.sroa.0751.1.off0 = phi i64 [ %.sroa.0751.0.off0971, %289 ], [ %extract.t950, %291 ]
  %.sroa.0751.1.off64 = phi i64 [ %.sroa.0751.0.off64972, %289 ], [ %extract.t953, %291 ]
  %.sroa.5752.1.off0 = phi i64 [ %.sroa.5752.0.off0973, %289 ], [ %extract.t957, %291 ]
  %.sroa.5752.1.off64 = phi i64 [ %.sroa.5752.0.off64974, %289 ], [ %extract.t960, %291 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !180
  %301 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %131) #7, !dbg !181, !srcloc !186
  %.sroa.038.2.extract.shift39.i = lshr i64 %301, 16, !dbg !187
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !187
  %303 = insertelement <2 x i64> %302, i64 %.sroa.038.2.extract.shift39.i, i64 1, !dbg !187
  %304 = shufflevector <2 x i64> %302, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !181
  %305 = lshr <2 x i64> %304, <i64 32, i64 48>, !dbg !181
  %306 = shufflevector <2 x i64> %303, <2 x i64> %305, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !188
  %.sroa.0787.6.vec.insert8031082 = trunc <4 x i64> %306 to <4 x i16>, !dbg !188
  %307 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %140) #7, !dbg !192, !srcloc !186
  %.sroa.038.2.extract.shift39.i361 = lshr i64 %307, 16, !dbg !195
  %308 = insertelement <2 x i64> poison, i64 %307, i64 0, !dbg !195
  %309 = insertelement <2 x i64> %308, i64 %.sroa.038.2.extract.shift39.i361, i64 1, !dbg !195
  %310 = shufflevector <2 x i64> %308, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !192
  %311 = lshr <2 x i64> %310, <i64 32, i64 48>, !dbg !192
  %312 = shufflevector <2 x i64> %309, <2 x i64> %311, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !196
  %.sroa.0754.6.vec.insert7701081 = trunc <4 x i64> %312 to <4 x i16>, !dbg !196
  %313 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %147) #7, !dbg !198, !srcloc !186
  %.sroa.038.2.extract.shift39.i373 = lshr i64 %313, 16, !dbg !201
  %314 = insertelement <2 x i64> poison, i64 %313, i64 0, !dbg !201
  %315 = insertelement <2 x i64> %314, i64 %.sroa.038.2.extract.shift39.i373, i64 1, !dbg !201
  %316 = shufflevector <2 x i64> %314, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !198
  %317 = lshr <2 x i64> %316, <i64 32, i64 48>, !dbg !198
  %318 = shufflevector <2 x i64> %315, <2 x i64> %317, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !202
  %.sroa.0898.6.vec.insert9101080 = trunc <4 x i64> %318 to <4 x i16>, !dbg !202
  %319 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %156) #7, !dbg !204, !srcloc !186
  %.sroa.038.2.extract.shift39.i385 = lshr i64 %319, 16, !dbg !207
  %320 = insertelement <2 x i64> poison, i64 %319, i64 0, !dbg !207
  %321 = insertelement <2 x i64> %320, i64 %.sroa.038.2.extract.shift39.i385, i64 1, !dbg !207
  %322 = shufflevector <2 x i64> %320, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !204
  %323 = lshr <2 x i64> %322, <i64 32, i64 48>, !dbg !204
  %324 = shufflevector <2 x i64> %321, <2 x i64> %323, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !208
  %.sroa.0872.6.vec.insert8841079 = trunc <4 x i64> %324 to <4 x i16>, !dbg !208
  %325 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %162) #7, !dbg !210, !srcloc !186
  %.sroa.038.2.extract.shift39.i397 = lshr i64 %325, 16, !dbg !213
  %326 = insertelement <2 x i64> poison, i64 %325, i64 0, !dbg !213
  %327 = insertelement <2 x i64> %326, i64 %.sroa.038.2.extract.shift39.i397, i64 1, !dbg !213
  %328 = shufflevector <2 x i64> %326, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !210
  %329 = lshr <2 x i64> %328, <i64 32, i64 48>, !dbg !210
  %330 = shufflevector <2 x i64> %327, <2 x i64> %329, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !214
  %.sroa.0846.6.vec.insert8581078 = trunc <4 x i64> %330 to <4 x i16>, !dbg !214
  %331 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0As_waitcnt lgkmcnt(0)\0A", "=v,v,~{memory}"(i32 %170) #7, !dbg !216, !srcloc !222
  %.sroa.038.2.extract.shift39.i409 = lshr i64 %331, 16, !dbg !223
  %332 = insertelement <2 x i64> poison, i64 %331, i64 0, !dbg !223
  %333 = insertelement <2 x i64> %332, i64 %.sroa.038.2.extract.shift39.i409, i64 1, !dbg !223
  %334 = shufflevector <2 x i64> %332, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !216
  %335 = lshr <2 x i64> %334, <i64 32, i64 48>, !dbg !216
  %336 = shufflevector <2 x i64> %333, <2 x i64> %335, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !224
  %.sroa.0821.6.vec.insert8331077 = trunc <4 x i64> %336 to <4 x i16>, !dbg !224
  tail call void @llvm.amdgcn.s.barrier(), !dbg !226
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !227
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !228
  %337 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0898.6.vec.insert9101080, <4 x i16> %.sroa.0787.6.vec.insert8031082, <4 x float> %.val336965966, i32 0, i32 0, i32 0), !dbg !229
  %338 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0898.6.vec.insert9101080, <4 x i16> %.sroa.0754.6.vec.insert7701081, <4 x float> %.val333977, i32 0, i32 0, i32 0), !dbg !237
  %339 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0872.6.vec.insert8841079, <4 x i16> %.sroa.0787.6.vec.insert8031082, <4 x float> %.val330979, i32 0, i32 0, i32 0), !dbg !241
  %340 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0872.6.vec.insert8841079, <4 x i16> %.sroa.0754.6.vec.insert7701081, <4 x float> %.val327982, i32 0, i32 0, i32 0), !dbg !245
  %341 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0846.6.vec.insert8581078, <4 x i16> %.sroa.0787.6.vec.insert8031082, <4 x float> %.val324985, i32 0, i32 0, i32 0), !dbg !249
  %342 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0846.6.vec.insert8581078, <4 x i16> %.sroa.0754.6.vec.insert7701081, <4 x float> %.val321988, i32 0, i32 0, i32 0), !dbg !253
  %343 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0821.6.vec.insert8331077, <4 x i16> %.sroa.0787.6.vec.insert8031082, <4 x float> %.val318991, i32 0, i32 0, i32 0), !dbg !257
  %344 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0821.6.vec.insert8331077, <4 x i16> %.sroa.0754.6.vec.insert7701081, <4 x float> %.val315994, i32 0, i32 0, i32 0), !dbg !261
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !265
  tail call void @llvm.amdgcn.s.barrier(), !dbg !266
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !267
  br i1 %.not, label %354, label %345, !dbg !268

345:                                              ; preds = %300
  %346 = shl nuw nsw i32 %290, 6
  %347 = zext nneg i32 %346 to i64
  %348 = getelementptr %struct.__hip_bfloat16, ptr %86, i64 %347
  %349 = ptrtoint ptr %348 to i64
  %350 = bitcast i64 %349 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i417 = shufflevector <2 x i32> %350, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %351 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i417, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i418 = insertelement <4 x i32> %351, i32 %173, i64 2
  %352 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i418, i32 noundef %172, i32 noundef 0, i32 noundef 0) #7, !dbg !269
  %353 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i418, i32 noundef %171, i32 noundef 0, i32 noundef 0) #7, !dbg !269
  %extract.t936 = trunc i128 %352 to i64, !dbg !272
  %extract938 = lshr i128 %352, 64, !dbg !272
  %extract.t939 = trunc nuw i128 %extract938 to i64, !dbg !272
  %extract.t943 = trunc i128 %353 to i64, !dbg !272
  %extract945 = lshr i128 %353, 64, !dbg !272
  %extract.t946 = trunc nuw i128 %extract945 to i64, !dbg !272
  br label %354, !dbg !272

354:                                              ; preds = %345, %300
  %.sroa.0749.1.off0 = phi i64 [ %.sroa.0749.0.off0967, %300 ], [ %extract.t936, %345 ]
  %.sroa.0749.1.off64 = phi i64 [ %.sroa.0749.0.off64968, %300 ], [ %extract.t939, %345 ]
  %.sroa.5.1.off0 = phi i64 [ %.sroa.5.0.off0969, %300 ], [ %extract.t943, %345 ]
  %.sroa.5.1.off64 = phi i64 [ %.sroa.5.0.off64970, %300 ], [ %extract.t946, %345 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !273
  %355 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %178) #7, !dbg !274, !srcloc !186
  %.sroa.038.2.extract.shift39.i426 = lshr i64 %355, 16, !dbg !277
  %356 = insertelement <2 x i64> poison, i64 %355, i64 0, !dbg !277
  %357 = insertelement <2 x i64> %356, i64 %.sroa.038.2.extract.shift39.i426, i64 1, !dbg !277
  %358 = shufflevector <2 x i64> %356, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !274
  %359 = lshr <2 x i64> %358, <i64 32, i64 48>, !dbg !274
  %360 = shufflevector <2 x i64> %357, <2 x i64> %359, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !278
  %.sroa.18804.14.vec.insert8201064 = trunc <4 x i64> %360 to <4 x i16>, !dbg !278
  %361 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %182) #7, !dbg !280, !srcloc !186
  %.sroa.038.2.extract.shift39.i438 = lshr i64 %361, 16, !dbg !283
  %362 = insertelement <2 x i64> poison, i64 %361, i64 0, !dbg !283
  %363 = insertelement <2 x i64> %362, i64 %.sroa.038.2.extract.shift39.i438, i64 1, !dbg !283
  %364 = shufflevector <2 x i64> %362, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !280
  %365 = lshr <2 x i64> %364, <i64 32, i64 48>, !dbg !280
  %366 = shufflevector <2 x i64> %363, <2 x i64> %365, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !284
  %.sroa.18.14.vec.insert7861063 = trunc <4 x i64> %366 to <4 x i16>, !dbg !284
  %367 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %186) #7, !dbg !286, !srcloc !186
  %.sroa.038.2.extract.shift39.i450 = lshr i64 %367, 16, !dbg !289
  %368 = insertelement <2 x i64> poison, i64 %367, i64 0, !dbg !289
  %369 = insertelement <2 x i64> %368, i64 %.sroa.038.2.extract.shift39.i450, i64 1, !dbg !289
  %370 = shufflevector <2 x i64> %368, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !286
  %371 = lshr <2 x i64> %370, <i64 32, i64 48>, !dbg !286
  %372 = shufflevector <2 x i64> %369, <2 x i64> %371, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !290
  %.sroa.14911.14.vec.insert9231062 = trunc <4 x i64> %372 to <4 x i16>, !dbg !290
  %373 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %190) #7, !dbg !292, !srcloc !186
  %.sroa.038.2.extract.shift39.i462 = lshr i64 %373, 16, !dbg !295
  %374 = insertelement <2 x i64> poison, i64 %373, i64 0, !dbg !295
  %375 = insertelement <2 x i64> %374, i64 %.sroa.038.2.extract.shift39.i462, i64 1, !dbg !295
  %376 = shufflevector <2 x i64> %374, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !292
  %377 = lshr <2 x i64> %376, <i64 32, i64 48>, !dbg !292
  %378 = shufflevector <2 x i64> %375, <2 x i64> %377, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !296
  %.sroa.14885.14.vec.insert8971061 = trunc <4 x i64> %378 to <4 x i16>, !dbg !296
  %379 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %194) #7, !dbg !298, !srcloc !186
  %.sroa.038.2.extract.shift39.i474 = lshr i64 %379, 16, !dbg !301
  %380 = insertelement <2 x i64> poison, i64 %379, i64 0, !dbg !301
  %381 = insertelement <2 x i64> %380, i64 %.sroa.038.2.extract.shift39.i474, i64 1, !dbg !301
  %382 = shufflevector <2 x i64> %380, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !298
  %383 = lshr <2 x i64> %382, <i64 32, i64 48>, !dbg !298
  %384 = shufflevector <2 x i64> %381, <2 x i64> %383, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !302
  %.sroa.14859.14.vec.insert8711060 = trunc <4 x i64> %384 to <4 x i16>, !dbg !302
  %385 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0As_waitcnt lgkmcnt(0)\0A", "=v,v,~{memory}"(i32 %198) #7, !dbg !304, !srcloc !222
  %.sroa.038.2.extract.shift39.i486 = lshr i64 %385, 16, !dbg !307
  %386 = insertelement <2 x i64> poison, i64 %385, i64 0, !dbg !307
  %387 = insertelement <2 x i64> %386, i64 %.sroa.038.2.extract.shift39.i486, i64 1, !dbg !307
  %388 = shufflevector <2 x i64> %386, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !304
  %389 = lshr <2 x i64> %388, <i64 32, i64 48>, !dbg !304
  %390 = shufflevector <2 x i64> %387, <2 x i64> %389, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !308
  %.sroa.14.14.vec.insert8451059 = trunc <4 x i64> %390 to <4 x i16>, !dbg !308
  tail call void @llvm.amdgcn.s.barrier(), !dbg !310
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !311
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !312
  %391 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14911.14.vec.insert9231062, <4 x i16> %.sroa.18804.14.vec.insert8201064, <4 x float> %337, i32 0, i32 0, i32 0), !dbg !313
  %392 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14911.14.vec.insert9231062, <4 x i16> %.sroa.18.14.vec.insert7861063, <4 x float> %338, i32 0, i32 0, i32 0), !dbg !317
  %393 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14885.14.vec.insert8971061, <4 x i16> %.sroa.18804.14.vec.insert8201064, <4 x float> %339, i32 0, i32 0, i32 0), !dbg !321
  %394 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14885.14.vec.insert8971061, <4 x i16> %.sroa.18.14.vec.insert7861063, <4 x float> %340, i32 0, i32 0, i32 0), !dbg !325
  %395 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14859.14.vec.insert8711060, <4 x i16> %.sroa.18804.14.vec.insert8201064, <4 x float> %341, i32 0, i32 0, i32 0), !dbg !329
  %396 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14859.14.vec.insert8711060, <4 x i16> %.sroa.18.14.vec.insert7861063, <4 x float> %342, i32 0, i32 0, i32 0), !dbg !333
  %397 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14.14.vec.insert8451059, <4 x i16> %.sroa.18804.14.vec.insert8201064, <4 x float> %343, i32 0, i32 0, i32 0), !dbg !337
  %398 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14.14.vec.insert8451059, <4 x i16> %.sroa.18.14.vec.insert7861063, <4 x float> %344, i32 0, i32 0, i32 0), !dbg !341
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !345
  tail call void @llvm.amdgcn.s.barrier(), !dbg !346
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !347
  %399 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %203) #7, !dbg !348, !srcloc !186
  %.sroa.038.2.extract.shift39.i498 = lshr i64 %399, 16, !dbg !351
  %400 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !351
  %401 = insertelement <2 x i64> %400, i64 %.sroa.038.2.extract.shift39.i498, i64 1, !dbg !351
  %402 = shufflevector <2 x i64> %400, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !348
  %403 = lshr <2 x i64> %402, <i64 32, i64 48>, !dbg !348
  %404 = shufflevector <2 x i64> %401, <2 x i64> %403, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !352
  %.sroa.0787.6.vec.insert1076 = trunc <4 x i64> %404 to <4 x i16>, !dbg !352
  %405 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %207) #7, !dbg !354, !srcloc !186
  %.sroa.038.2.extract.shift39.i510 = lshr i64 %405, 16, !dbg !357
  %406 = insertelement <2 x i64> poison, i64 %405, i64 0, !dbg !357
  %407 = insertelement <2 x i64> %406, i64 %.sroa.038.2.extract.shift39.i510, i64 1, !dbg !357
  %408 = shufflevector <2 x i64> %406, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !354
  %409 = lshr <2 x i64> %408, <i64 32, i64 48>, !dbg !354
  %410 = shufflevector <2 x i64> %407, <2 x i64> %409, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !358
  %.sroa.0754.6.vec.insert1075 = trunc <4 x i64> %410 to <4 x i16>, !dbg !358
  %411 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %211) #7, !dbg !360, !srcloc !186
  %.sroa.038.2.extract.shift39.i522 = lshr i64 %411, 16, !dbg !363
  %412 = insertelement <2 x i64> poison, i64 %411, i64 0, !dbg !363
  %413 = insertelement <2 x i64> %412, i64 %.sroa.038.2.extract.shift39.i522, i64 1, !dbg !363
  %414 = shufflevector <2 x i64> %412, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !360
  %415 = lshr <2 x i64> %414, <i64 32, i64 48>, !dbg !360
  %416 = shufflevector <2 x i64> %413, <2 x i64> %415, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !364
  %.sroa.0898.6.vec.insert1074 = trunc <4 x i64> %416 to <4 x i16>, !dbg !364
  %417 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %215) #7, !dbg !366, !srcloc !186
  %.sroa.038.2.extract.shift39.i534 = lshr i64 %417, 16, !dbg !369
  %418 = insertelement <2 x i64> poison, i64 %417, i64 0, !dbg !369
  %419 = insertelement <2 x i64> %418, i64 %.sroa.038.2.extract.shift39.i534, i64 1, !dbg !369
  %420 = shufflevector <2 x i64> %418, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !366
  %421 = lshr <2 x i64> %420, <i64 32, i64 48>, !dbg !366
  %422 = shufflevector <2 x i64> %419, <2 x i64> %421, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !370
  %.sroa.0872.6.vec.insert1073 = trunc <4 x i64> %422 to <4 x i16>, !dbg !370
  %423 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %219) #7, !dbg !372, !srcloc !186
  %.sroa.038.2.extract.shift39.i546 = lshr i64 %423, 16, !dbg !375
  %424 = insertelement <2 x i64> poison, i64 %423, i64 0, !dbg !375
  %425 = insertelement <2 x i64> %424, i64 %.sroa.038.2.extract.shift39.i546, i64 1, !dbg !375
  %426 = shufflevector <2 x i64> %424, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !372
  %427 = lshr <2 x i64> %426, <i64 32, i64 48>, !dbg !372
  %428 = shufflevector <2 x i64> %425, <2 x i64> %427, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !376
  %.sroa.0846.6.vec.insert1072 = trunc <4 x i64> %428 to <4 x i16>, !dbg !376
  %429 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %223) #7, !dbg !378, !srcloc !186
  %.sroa.038.2.extract.shift39.i558 = lshr i64 %429, 16, !dbg !381
  %430 = insertelement <2 x i64> poison, i64 %429, i64 0, !dbg !381
  %431 = insertelement <2 x i64> %430, i64 %.sroa.038.2.extract.shift39.i558, i64 1, !dbg !381
  %432 = shufflevector <2 x i64> %430, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !378
  %433 = lshr <2 x i64> %432, <i64 32, i64 48>, !dbg !378
  %434 = shufflevector <2 x i64> %431, <2 x i64> %433, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !382
  %.sroa.0821.6.vec.insert1071 = trunc <4 x i64> %434 to <4 x i16>, !dbg !382
  %435 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %228) #7, !dbg !384, !srcloc !186
  %.sroa.038.2.extract.shift39.i570 = lshr i64 %435, 16, !dbg !387
  %436 = insertelement <2 x i64> poison, i64 %435, i64 0, !dbg !387
  %437 = insertelement <2 x i64> %436, i64 %.sroa.038.2.extract.shift39.i570, i64 1, !dbg !387
  %438 = shufflevector <2 x i64> %436, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !384
  %439 = lshr <2 x i64> %438, <i64 32, i64 48>, !dbg !384
  %440 = shufflevector <2 x i64> %437, <2 x i64> %439, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !388
  %.sroa.18804.14.vec.insert1070 = trunc <4 x i64> %440 to <4 x i16>, !dbg !388
  %441 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %232) #7, !dbg !390, !srcloc !186
  %.sroa.038.2.extract.shift39.i582 = lshr i64 %441, 16, !dbg !393
  %442 = insertelement <2 x i64> poison, i64 %441, i64 0, !dbg !393
  %443 = insertelement <2 x i64> %442, i64 %.sroa.038.2.extract.shift39.i582, i64 1, !dbg !393
  %444 = shufflevector <2 x i64> %442, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !390
  %445 = lshr <2 x i64> %444, <i64 32, i64 48>, !dbg !390
  %446 = shufflevector <2 x i64> %443, <2 x i64> %445, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !394
  %.sroa.18.14.vec.insert1069 = trunc <4 x i64> %446 to <4 x i16>, !dbg !394
  %447 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %236) #7, !dbg !396, !srcloc !186
  %.sroa.038.2.extract.shift39.i594 = lshr i64 %447, 16, !dbg !399
  %448 = insertelement <2 x i64> poison, i64 %447, i64 0, !dbg !399
  %449 = insertelement <2 x i64> %448, i64 %.sroa.038.2.extract.shift39.i594, i64 1, !dbg !399
  %450 = shufflevector <2 x i64> %448, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !396
  %451 = lshr <2 x i64> %450, <i64 32, i64 48>, !dbg !396
  %452 = shufflevector <2 x i64> %449, <2 x i64> %451, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !400
  %.sroa.14911.14.vec.insert1068 = trunc <4 x i64> %452 to <4 x i16>, !dbg !400
  %453 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %240) #7, !dbg !402, !srcloc !186
  %.sroa.038.2.extract.shift39.i606 = lshr i64 %453, 16, !dbg !405
  %454 = insertelement <2 x i64> poison, i64 %453, i64 0, !dbg !405
  %455 = insertelement <2 x i64> %454, i64 %.sroa.038.2.extract.shift39.i606, i64 1, !dbg !405
  %456 = shufflevector <2 x i64> %454, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !402
  %457 = lshr <2 x i64> %456, <i64 32, i64 48>, !dbg !402
  %458 = shufflevector <2 x i64> %455, <2 x i64> %457, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !406
  %.sroa.14885.14.vec.insert1067 = trunc <4 x i64> %458 to <4 x i16>, !dbg !406
  %459 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %244) #7, !dbg !408, !srcloc !186
  %.sroa.038.2.extract.shift39.i618 = lshr i64 %459, 16, !dbg !411
  %460 = insertelement <2 x i64> poison, i64 %459, i64 0, !dbg !411
  %461 = insertelement <2 x i64> %460, i64 %.sroa.038.2.extract.shift39.i618, i64 1, !dbg !411
  %462 = shufflevector <2 x i64> %460, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !408
  %463 = lshr <2 x i64> %462, <i64 32, i64 48>, !dbg !408
  %464 = shufflevector <2 x i64> %461, <2 x i64> %463, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !412
  %.sroa.14859.14.vec.insert1066 = trunc <4 x i64> %464 to <4 x i16>, !dbg !412
  %465 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0As_waitcnt lgkmcnt(0)\0A", "=v,v,~{memory}"(i32 %248) #7, !dbg !414, !srcloc !222
  %.sroa.038.2.extract.shift39.i630 = lshr i64 %465, 16, !dbg !417
  %466 = insertelement <2 x i64> poison, i64 %465, i64 0, !dbg !417
  %467 = insertelement <2 x i64> %466, i64 %.sroa.038.2.extract.shift39.i630, i64 1, !dbg !417
  %468 = shufflevector <2 x i64> %466, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !414
  %469 = lshr <2 x i64> %468, <i64 32, i64 48>, !dbg !414
  %470 = shufflevector <2 x i64> %467, <2 x i64> %469, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !418
  %.sroa.14.14.vec.insert1065 = trunc <4 x i64> %470 to <4 x i16>, !dbg !418
  tail call void @llvm.amdgcn.s.barrier(), !dbg !420
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !421
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !422
  %471 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0898.6.vec.insert1074, <4 x i16> %.sroa.0787.6.vec.insert1076, <4 x float> %391, i32 0, i32 0, i32 0), !dbg !423
  %472 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0898.6.vec.insert1074, <4 x i16> %.sroa.0754.6.vec.insert1075, <4 x float> %392, i32 0, i32 0, i32 0), !dbg !427
  %473 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0872.6.vec.insert1073, <4 x i16> %.sroa.0787.6.vec.insert1076, <4 x float> %393, i32 0, i32 0, i32 0), !dbg !431
  %474 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0872.6.vec.insert1073, <4 x i16> %.sroa.0754.6.vec.insert1075, <4 x float> %394, i32 0, i32 0, i32 0), !dbg !435
  %475 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0846.6.vec.insert1072, <4 x i16> %.sroa.0787.6.vec.insert1076, <4 x float> %395, i32 0, i32 0, i32 0), !dbg !439
  %476 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0846.6.vec.insert1072, <4 x i16> %.sroa.0754.6.vec.insert1075, <4 x float> %396, i32 0, i32 0, i32 0), !dbg !443
  %477 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0821.6.vec.insert1071, <4 x i16> %.sroa.0787.6.vec.insert1076, <4 x float> %397, i32 0, i32 0, i32 0), !dbg !447
  %478 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0821.6.vec.insert1071, <4 x i16> %.sroa.0754.6.vec.insert1075, <4 x float> %398, i32 0, i32 0, i32 0), !dbg !451
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !455
  tail call void @llvm.amdgcn.s.barrier(), !dbg !456
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !457
  br i1 %.not, label %480, label %479, !dbg !458

479:                                              ; preds = %354
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !459, !srcloc !460
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %256, i64 %.sroa.0751.1.off0) #7, !dbg !461, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %260, i64 %.sroa.0751.1.off64) #7, !dbg !467, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %264, i64 %.sroa.5752.1.off0) #7, !dbg !461, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %268, i64 %.sroa.5752.1.off64) #7, !dbg !467, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %274, i64 %.sroa.0749.1.off0) #7, !dbg !469, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %278, i64 %.sroa.0749.1.off64) #7, !dbg !472, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %282, i64 %.sroa.5.1.off0) #7, !dbg !469, !srcloc !466
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %286, i64 %.sroa.5.1.off64) #7, !dbg !472, !srcloc !466
  br label %480, !dbg !474

480:                                              ; preds = %479, %354
  tail call void @llvm.amdgcn.s.barrier(), !dbg !475
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !476
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !477
  %481 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14911.14.vec.insert1068, <4 x i16> %.sroa.18804.14.vec.insert1070, <4 x float> %471, i32 0, i32 0, i32 0), !dbg !478
  %482 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14911.14.vec.insert1068, <4 x i16> %.sroa.18.14.vec.insert1069, <4 x float> %472, i32 0, i32 0, i32 0), !dbg !482
  %483 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14885.14.vec.insert1067, <4 x i16> %.sroa.18804.14.vec.insert1070, <4 x float> %473, i32 0, i32 0, i32 0), !dbg !486
  %484 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14885.14.vec.insert1067, <4 x i16> %.sroa.18.14.vec.insert1069, <4 x float> %474, i32 0, i32 0, i32 0), !dbg !490
  %485 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14859.14.vec.insert1066, <4 x i16> %.sroa.18804.14.vec.insert1070, <4 x float> %475, i32 0, i32 0, i32 0), !dbg !494
  %486 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14859.14.vec.insert1066, <4 x i16> %.sroa.18.14.vec.insert1069, <4 x float> %476, i32 0, i32 0, i32 0), !dbg !498
  %487 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14.14.vec.insert1065, <4 x i16> %.sroa.18804.14.vec.insert1070, <4 x float> %477, i32 0, i32 0, i32 0), !dbg !502
  %488 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.14.14.vec.insert1065, <4 x i16> %.sroa.18.14.vec.insert1069, <4 x float> %478, i32 0, i32 0, i32 0), !dbg !506
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !510
  tail call void @llvm.amdgcn.s.barrier(), !dbg !511
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !512
  %exitcond.not = icmp eq i32 %290, 128, !dbg !513
  br i1 %exitcond.not, label %287, label %289, !dbg !167, !llvm.loop !514

489:                                              ; preds = %287
  tail call void @llvm.amdgcn.s.barrier(), !dbg !517
  br label %490, !dbg !518

490:                                              ; preds = %489, %287
  %491 = shl nsw i32 %33, 3
  %492 = or disjoint i32 %491, %36
  %493 = shl nsw i32 %31, 3
  %494 = trunc i64 %.sroa.20.0.copyload to i32
  %495 = mul nsw i32 %122, %494
  %496 = add nsw i32 %495, %120
  %497 = sext i32 %496 to i64
  %498 = or disjoint i32 %122, 1
  %499 = mul nsw i32 %498, %494
  %500 = add nsw i32 %499, %120
  %501 = sext i32 %500 to i64
  %502 = or disjoint i32 %122, 2
  %503 = mul nsw i32 %502, %494
  %504 = add nsw i32 %503, %120
  %505 = sext i32 %504 to i64
  %506 = or disjoint i32 %122, 3
  %507 = mul nsw i32 %506, %494
  %508 = add nsw i32 %507, %120
  %509 = sext i32 %508 to i64
  %510 = or disjoint i32 %493, %37, !dbg !519
  %511 = shl nsw i32 %492, 4, !dbg !520
  %512 = shl nsw i32 %510, 4, !dbg !528
  %513 = sext i32 %511 to i64, !dbg !529
  %514 = mul i64 %.sroa.20.0.copyload, %513, !dbg !532
  %515 = sext i32 %512 to i64, !dbg !533
  %516 = getelementptr float, ptr %.sroa.17933.0.copyload, i64 %514, !dbg !534
  %517 = getelementptr float, ptr %516, i64 %515, !dbg !534
  %.sroa.0.0.vec.extract = extractelement <4 x float> %481, i64 0, !dbg !535
  %518 = getelementptr inbounds float, ptr %517, i64 %497, !dbg !536
  store float %.sroa.0.0.vec.extract, ptr %518, align 4, !dbg !537, !tbaa !538
  %.sroa.0.4.vec.extract = extractelement <4 x float> %481, i64 1, !dbg !542
  %519 = getelementptr inbounds float, ptr %517, i64 %501, !dbg !543
  store float %.sroa.0.4.vec.extract, ptr %519, align 4, !dbg !544, !tbaa !538
  %.sroa.0.8.vec.extract = extractelement <4 x float> %481, i64 2, !dbg !545
  %520 = getelementptr inbounds float, ptr %517, i64 %505, !dbg !546
  store float %.sroa.0.8.vec.extract, ptr %520, align 4, !dbg !547, !tbaa !538
  %.sroa.0.12.vec.extract = extractelement <4 x float> %481, i64 3, !dbg !548
  %521 = getelementptr inbounds float, ptr %517, i64 %509, !dbg !549
  store float %.sroa.0.12.vec.extract, ptr %521, align 4, !dbg !550, !tbaa !538
  %522 = or disjoint i32 %493, %37, !dbg !519
  %523 = shl i32 %522, 4, !dbg !528
  %524 = or disjoint i32 %523, 64, !dbg !528
  %525 = sext i32 %524 to i64, !dbg !533
  %526 = getelementptr float, ptr %516, i64 %525, !dbg !534
  %.sroa.12.16.vec.extract = extractelement <4 x float> %482, i64 0, !dbg !535
  %527 = getelementptr inbounds float, ptr %526, i64 %497, !dbg !536
  store float %.sroa.12.16.vec.extract, ptr %527, align 4, !dbg !537, !tbaa !538
  %.sroa.12.20.vec.extract = extractelement <4 x float> %482, i64 1, !dbg !542
  %528 = getelementptr inbounds float, ptr %526, i64 %501, !dbg !543
  store float %.sroa.12.20.vec.extract, ptr %528, align 4, !dbg !544, !tbaa !538
  %.sroa.12.24.vec.extract = extractelement <4 x float> %482, i64 2, !dbg !545
  %529 = getelementptr inbounds float, ptr %526, i64 %505, !dbg !546
  store float %.sroa.12.24.vec.extract, ptr %529, align 4, !dbg !547, !tbaa !538
  %.sroa.12.28.vec.extract = extractelement <4 x float> %482, i64 3, !dbg !548
  %530 = getelementptr inbounds float, ptr %526, i64 %509, !dbg !549
  store float %.sroa.12.28.vec.extract, ptr %530, align 4, !dbg !550, !tbaa !538
  %531 = shl i32 %492, 4, !dbg !520
  %532 = add nuw nsw i32 %531, 32, !dbg !520
  %533 = sext i32 %532 to i64, !dbg !529
  %534 = mul i64 %.sroa.20.0.copyload, %533, !dbg !532
  %535 = getelementptr float, ptr %.sroa.17933.0.copyload, i64 %534, !dbg !534
  %536 = getelementptr float, ptr %535, i64 %515, !dbg !534
  %.sroa.22.32.vec.extract = extractelement <4 x float> %483, i64 0, !dbg !535
  %537 = getelementptr inbounds float, ptr %536, i64 %497, !dbg !536
  store float %.sroa.22.32.vec.extract, ptr %537, align 4, !dbg !537, !tbaa !538
  %.sroa.22.36.vec.extract = extractelement <4 x float> %483, i64 1, !dbg !542
  %538 = getelementptr inbounds float, ptr %536, i64 %501, !dbg !543
  store float %.sroa.22.36.vec.extract, ptr %538, align 4, !dbg !544, !tbaa !538
  %.sroa.22.40.vec.extract = extractelement <4 x float> %483, i64 2, !dbg !545
  %539 = getelementptr inbounds float, ptr %536, i64 %505, !dbg !546
  store float %.sroa.22.40.vec.extract, ptr %539, align 4, !dbg !547, !tbaa !538
  %.sroa.22.44.vec.extract = extractelement <4 x float> %483, i64 3, !dbg !548
  %540 = getelementptr inbounds float, ptr %536, i64 %509, !dbg !549
  store float %.sroa.22.44.vec.extract, ptr %540, align 4, !dbg !550, !tbaa !538
  %541 = getelementptr float, ptr %535, i64 %525, !dbg !534
  %.sroa.32.48.vec.extract = extractelement <4 x float> %484, i64 0, !dbg !535
  %542 = getelementptr inbounds float, ptr %541, i64 %497, !dbg !536
  store float %.sroa.32.48.vec.extract, ptr %542, align 4, !dbg !537, !tbaa !538
  %.sroa.32.52.vec.extract = extractelement <4 x float> %484, i64 1, !dbg !542
  %543 = getelementptr inbounds float, ptr %541, i64 %501, !dbg !543
  store float %.sroa.32.52.vec.extract, ptr %543, align 4, !dbg !544, !tbaa !538
  %.sroa.32.56.vec.extract = extractelement <4 x float> %484, i64 2, !dbg !545
  %544 = getelementptr inbounds float, ptr %541, i64 %505, !dbg !546
  store float %.sroa.32.56.vec.extract, ptr %544, align 4, !dbg !547, !tbaa !538
  %.sroa.32.60.vec.extract = extractelement <4 x float> %484, i64 3, !dbg !548
  %545 = getelementptr inbounds float, ptr %541, i64 %509, !dbg !549
  store float %.sroa.32.60.vec.extract, ptr %545, align 4, !dbg !550, !tbaa !538
  %546 = shl i32 %492, 4, !dbg !520
  %547 = or disjoint i32 %546, 64, !dbg !520
  %548 = sext i32 %547 to i64, !dbg !529
  %549 = mul i64 %.sroa.20.0.copyload, %548, !dbg !532
  %550 = getelementptr float, ptr %.sroa.17933.0.copyload, i64 %549, !dbg !534
  %551 = getelementptr float, ptr %550, i64 %515, !dbg !534
  %.sroa.42.64.vec.extract = extractelement <4 x float> %485, i64 0, !dbg !535
  %552 = getelementptr inbounds float, ptr %551, i64 %497, !dbg !536
  store float %.sroa.42.64.vec.extract, ptr %552, align 4, !dbg !537, !tbaa !538
  %.sroa.42.68.vec.extract = extractelement <4 x float> %485, i64 1, !dbg !542
  %553 = getelementptr inbounds float, ptr %551, i64 %501, !dbg !543
  store float %.sroa.42.68.vec.extract, ptr %553, align 4, !dbg !544, !tbaa !538
  %.sroa.42.72.vec.extract = extractelement <4 x float> %485, i64 2, !dbg !545
  %554 = getelementptr inbounds float, ptr %551, i64 %505, !dbg !546
  store float %.sroa.42.72.vec.extract, ptr %554, align 4, !dbg !547, !tbaa !538
  %.sroa.42.76.vec.extract = extractelement <4 x float> %485, i64 3, !dbg !548
  %555 = getelementptr inbounds float, ptr %551, i64 %509, !dbg !549
  store float %.sroa.42.76.vec.extract, ptr %555, align 4, !dbg !550, !tbaa !538
  %556 = getelementptr float, ptr %550, i64 %525, !dbg !534
  %.sroa.52.80.vec.extract = extractelement <4 x float> %486, i64 0, !dbg !535
  %557 = getelementptr inbounds float, ptr %556, i64 %497, !dbg !536
  store float %.sroa.52.80.vec.extract, ptr %557, align 4, !dbg !537, !tbaa !538
  %.sroa.52.84.vec.extract = extractelement <4 x float> %486, i64 1, !dbg !542
  %558 = getelementptr inbounds float, ptr %556, i64 %501, !dbg !543
  store float %.sroa.52.84.vec.extract, ptr %558, align 4, !dbg !544, !tbaa !538
  %.sroa.52.88.vec.extract = extractelement <4 x float> %486, i64 2, !dbg !545
  %559 = getelementptr inbounds float, ptr %556, i64 %505, !dbg !546
  store float %.sroa.52.88.vec.extract, ptr %559, align 4, !dbg !547, !tbaa !538
  %.sroa.52.92.vec.extract = extractelement <4 x float> %486, i64 3, !dbg !548
  %560 = getelementptr inbounds float, ptr %556, i64 %509, !dbg !549
  store float %.sroa.52.92.vec.extract, ptr %560, align 4, !dbg !550, !tbaa !538
  %561 = shl i32 %492, 4, !dbg !520
  %562 = add i32 %561, 96, !dbg !520
  %563 = sext i32 %562 to i64, !dbg !529
  %564 = mul i64 %.sroa.20.0.copyload, %563, !dbg !532
  %565 = getelementptr float, ptr %.sroa.17933.0.copyload, i64 %564, !dbg !534
  %566 = getelementptr float, ptr %565, i64 %515, !dbg !534
  %.sroa.62.96.vec.extract = extractelement <4 x float> %487, i64 0, !dbg !535
  %567 = getelementptr inbounds float, ptr %566, i64 %497, !dbg !536
  store float %.sroa.62.96.vec.extract, ptr %567, align 4, !dbg !537, !tbaa !538
  %.sroa.62.100.vec.extract = extractelement <4 x float> %487, i64 1, !dbg !542
  %568 = getelementptr inbounds float, ptr %566, i64 %501, !dbg !543
  store float %.sroa.62.100.vec.extract, ptr %568, align 4, !dbg !544, !tbaa !538
  %.sroa.62.104.vec.extract = extractelement <4 x float> %487, i64 2, !dbg !545
  %569 = getelementptr inbounds float, ptr %566, i64 %505, !dbg !546
  store float %.sroa.62.104.vec.extract, ptr %569, align 4, !dbg !547, !tbaa !538
  %.sroa.62.108.vec.extract = extractelement <4 x float> %487, i64 3, !dbg !548
  %570 = getelementptr inbounds float, ptr %566, i64 %509, !dbg !549
  store float %.sroa.62.108.vec.extract, ptr %570, align 4, !dbg !550, !tbaa !538
  %571 = getelementptr float, ptr %565, i64 %525, !dbg !534
  %.sroa.72.112.vec.extract = extractelement <4 x float> %488, i64 0, !dbg !535
  %572 = getelementptr inbounds float, ptr %571, i64 %497, !dbg !536
  store float %.sroa.72.112.vec.extract, ptr %572, align 4, !dbg !537, !tbaa !538
  %.sroa.72.116.vec.extract = extractelement <4 x float> %488, i64 1, !dbg !542
  %573 = getelementptr inbounds float, ptr %571, i64 %501, !dbg !543
  store float %.sroa.72.116.vec.extract, ptr %573, align 4, !dbg !544, !tbaa !538
  %.sroa.72.120.vec.extract = extractelement <4 x float> %488, i64 2, !dbg !545
  %574 = getelementptr inbounds float, ptr %571, i64 %505, !dbg !546
  store float %.sroa.72.120.vec.extract, ptr %574, align 4, !dbg !547, !tbaa !538
  %.sroa.72.124.vec.extract = extractelement <4 x float> %488, i64 3, !dbg !548
  %575 = getelementptr inbounds float, ptr %571, i64 %509, !dbg !549
  store float %.sroa.72.124.vec.extract, ptr %575, align 4, !dbg !550, !tbaa !538
  ret void, !dbg !551
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

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #5

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
!1 = !DIFile(filename: "128_128_64_16.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"}
!9 = distinct !DISubprogram(name: "micro_tk", scope: !1, file: !1, line: 39, type: !10, scopeLine: 39, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !15)
!13 = distinct !DISubprogram(name: "align_ptr<16>", scope: !14, file: !14, line: 228, type: !10, scopeLine: 228, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DIFile(filename: "ThunderKittens-HIP/include/common/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!15 = distinct !DILocation(line: 252, column: 13, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 128, 64> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DILocation(line: 42, column: 44, scope: !9)
!18 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !15)
!19 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !15)
!20 = !DILocation(line: 255, column: 17, scope: !16, inlinedAt: !17)
!21 = !DILocation(line: 230, column: 30, scope: !13, inlinedAt: !22)
!22 = distinct !DILocation(line: 252, column: 13, scope: !16, inlinedAt: !23)
!23 = distinct !DILocation(line: 43, column: 44, scope: !9)
!24 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !22)
!25 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !22)
!26 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !22)
!27 = !DILocation(line: 271, column: 116, scope: !28, inlinedAt: !30)
!28 = distinct !DISubprogram(name: "__hip_get_block_idx_y", scope: !29, file: !29, line: 271, type: !10, scopeLine: 271, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!29 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h", directory: "")
!30 = distinct !DILocation(line: 301, column: 160, scope: !31, inlinedAt: !32)
!31 = distinct !DISubprogram(name: "__get_y", scope: !29, file: !29, line: 301, type: !10, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!32 = distinct !DILocation(line: 56, column: 17, scope: !9)
!33 = !DILocation(line: 280, column: 115, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "__hip_get_grid_dim_x", scope: !29, file: !29, line: 280, type: !10, scopeLine: 280, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!35 = distinct !DILocation(line: 318, column: 160, scope: !36, inlinedAt: !37)
!36 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 318, type: !10, scopeLine: 318, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!37 = distinct !DILocation(line: 56, column: 30, scope: !9)
!38 = !{!39, !39, i64 0}
!39 = !{!"int", !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C/C++ TBAA"}
!42 = !DILocation(line: 56, column: 28, scope: !9)
!43 = !DILocation(line: 270, column: 116, scope: !44, inlinedAt: !45)
!44 = distinct !DISubprogram(name: "__hip_get_block_idx_x", scope: !29, file: !29, line: 270, type: !10, scopeLine: 270, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!45 = distinct !DILocation(line: 300, column: 160, scope: !46, inlinedAt: !47)
!46 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 300, type: !10, scopeLine: 300, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!47 = distinct !DILocation(line: 56, column: 43, scope: !9)
!48 = !DILocation(line: 56, column: 41, scope: !9)
!49 = !DILocation(line: 281, column: 115, scope: !50, inlinedAt: !51)
!50 = distinct !DISubprogram(name: "__hip_get_grid_dim_y", scope: !29, file: !29, line: 281, type: !10, scopeLine: 281, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = distinct !DILocation(line: 319, column: 160, scope: !52, inlinedAt: !53)
!52 = distinct !DISubprogram(name: "__get_y", scope: !29, file: !29, line: 319, type: !10, scopeLine: 319, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!53 = distinct !DILocation(line: 58, column: 37, scope: !9)
!54 = !DILocation(line: 58, column: 35, scope: !9)
!55 = !DILocation(line: 64, column: 61, scope: !9)
!56 = !DILocation(line: 64, column: 18, scope: !9)
!57 = !DILocation(line: 64, column: 41, scope: !9)
!58 = !DILocation(line: 64, column: 30, scope: !9)
!59 = !DILocation(line: 64, column: 53, scope: !9)
!60 = !DILocation(line: 72, column: 25, scope: !9)
!61 = !DILocation(line: 73, column: 32, scope: !9)
!62 = !DILocation(line: 74, column: 38, scope: !9)
!63 = !DILocation(line: 1321, column: 10, scope: !64, inlinedAt: !66)
!64 = distinct !DISubprogram(name: "min", scope: !65, file: !65, line: 1320, type: !10, scopeLine: 1320, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!65 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/lib/clang/19/include/__clang_hip_math.h", directory: "")
!66 = distinct !DILocation(line: 74, column: 24, scope: !9)
!67 = !DILocation(line: 75, column: 38, scope: !9)
!68 = !DILocation(line: 76, column: 44, scope: !9)
!69 = !DILocation(line: 75, column: 59, scope: !9)
!70 = !DILocation(line: 75, column: 29, scope: !9)
!71 = !DILocation(line: 265, column: 117, scope: !72, inlinedAt: !73)
!72 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !29, file: !29, line: 265, type: !10, scopeLine: 265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!73 = distinct !DILocation(line: 291, column: 160, scope: !74, inlinedAt: !75)
!74 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 291, type: !10, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!75 = distinct !DILocation(line: 57, column: 85, scope: !76, inlinedAt: !77)
!76 = distinct !DISubprogram(name: "warpid", scope: !14, file: !14, line: 57, type: !10, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!77 = distinct !DILocation(line: 82, column: 25, scope: !9)
!78 = !DILocation(line: 57, column: 97, scope: !76, inlinedAt: !77)
!79 = !DILocation(line: 83, column: 34, scope: !9)
!80 = !DILocation(line: 84, column: 34, scope: !9)
!81 = !DILocation(line: 30, column: 28, scope: !82, inlinedAt: !84)
!82 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !83, file: !83, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!83 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!84 = distinct !DILocation(line: 12, column: 5, scope: !85, inlinedAt: !87)
!85 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> > >", scope: !86, file: !86, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!86 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!87 = distinct !DILocation(line: 89, column: 5, scope: !9)
!88 = !DILocation(line: 60, column: 34, scope: !89, inlinedAt: !91)
!89 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !90, file: !90, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!90 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!91 = distinct !DILocation(line: 37, column: 39, scope: !82, inlinedAt: !84)
!92 = !DILocation(line: 63, column: 54, scope: !93, inlinedAt: !95)
!93 = distinct !DISubprogram(name: "operator[]", scope: !94, file: !94, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!94 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/gl.cuh", directory: "/workdir/AMD-benchmarking-harness")
!95 = distinct !DILocation(line: 38, column: 57, scope: !82, inlinedAt: !84)
!96 = !DILocation(line: 63, column: 60, scope: !93, inlinedAt: !95)
!97 = !DILocation(line: 63, column: 16, scope: !93, inlinedAt: !95)
!98 = !DILocation(line: 40, column: 24, scope: !82, inlinedAt: !84)
!99 = !DILocation(line: 54, column: 32, scope: !82, inlinedAt: !84)
!100 = !DILocation(line: 58, column: 75, scope: !82, inlinedAt: !84)
!101 = !DILocation(line: 58, column: 88, scope: !82, inlinedAt: !84)
!102 = !DILocation(line: 58, column: 68, scope: !82, inlinedAt: !84)
!103 = !DILocation(line: 92, column: 5, scope: !104, inlinedAt: !106)
!104 = distinct !DISubprogram(name: "load_global_vec4_async", scope: !105, file: !105, line: 89, type: !10, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!105 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/util/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!106 = distinct !DILocation(line: 58, column: 26, scope: !82, inlinedAt: !84)
!107 = !{i64 3065620}
!108 = !DILocation(line: 62, column: 9, scope: !82, inlinedAt: !84)
!109 = !{i64 3083803}
!110 = !DILocation(line: 103, column: 35, scope: !111, inlinedAt: !113)
!111 = distinct !DISubprogram(name: "idx", scope: !112, file: !112, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!112 = !DIFile(filename: "ThunderKittens-HIP/include/types/shared/st.cuh", directory: "/workdir/AMD-benchmarking-harness")
!113 = distinct !DILocation(line: 71, column: 34, scope: !82, inlinedAt: !84)
!114 = !DILocation(line: 104, column: 60, scope: !111, inlinedAt: !113)
!115 = !DILocation(line: 105, column: 22, scope: !111, inlinedAt: !113)
!116 = !DILocation(line: 71, column: 72, scope: !82, inlinedAt: !84)
!117 = !DILocation(line: 27, column: 5, scope: !118, inlinedAt: !119)
!118 = distinct !DISubprogram(name: "store_shared_vec", scope: !105, file: !105, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!119 = distinct !DILocation(line: 71, column: 17, scope: !82, inlinedAt: !84)
!120 = !{i64 3063955}
!121 = !DILocation(line: 103, column: 35, scope: !111, inlinedAt: !122)
!122 = distinct !DILocation(line: 72, column: 34, scope: !82, inlinedAt: !84)
!123 = !DILocation(line: 104, column: 60, scope: !111, inlinedAt: !122)
!124 = !DILocation(line: 105, column: 22, scope: !111, inlinedAt: !122)
!125 = !DILocation(line: 72, column: 95, scope: !82, inlinedAt: !84)
!126 = !DILocation(line: 27, column: 5, scope: !118, inlinedAt: !127)
!127 = distinct !DILocation(line: 72, column: 17, scope: !82, inlinedAt: !84)
!128 = !DILocation(line: 103, column: 46, scope: !111, inlinedAt: !113)
!129 = !DILocation(line: 75, column: 9, scope: !82, inlinedAt: !84)
!130 = !{i64 3084344}
!131 = !DILocation(line: 30, column: 28, scope: !82, inlinedAt: !132)
!132 = distinct !DILocation(line: 12, column: 5, scope: !85, inlinedAt: !133)
!133 = distinct !DILocation(line: 90, column: 5, scope: !9)
!134 = !DILocation(line: 60, column: 34, scope: !89, inlinedAt: !135)
!135 = distinct !DILocation(line: 37, column: 39, scope: !82, inlinedAt: !132)
!136 = !DILocation(line: 63, column: 54, scope: !93, inlinedAt: !137)
!137 = distinct !DILocation(line: 38, column: 57, scope: !82, inlinedAt: !132)
!138 = !DILocation(line: 63, column: 60, scope: !93, inlinedAt: !137)
!139 = !DILocation(line: 63, column: 16, scope: !93, inlinedAt: !137)
!140 = !DILocation(line: 40, column: 24, scope: !82, inlinedAt: !132)
!141 = !DILocation(line: 58, column: 75, scope: !82, inlinedAt: !132)
!142 = !DILocation(line: 58, column: 88, scope: !82, inlinedAt: !132)
!143 = !DILocation(line: 58, column: 68, scope: !82, inlinedAt: !132)
!144 = !DILocation(line: 92, column: 5, scope: !104, inlinedAt: !145)
!145 = distinct !DILocation(line: 58, column: 26, scope: !82, inlinedAt: !132)
!146 = !DILocation(line: 62, column: 9, scope: !82, inlinedAt: !132)
!147 = !DILocation(line: 103, column: 35, scope: !111, inlinedAt: !148)
!148 = distinct !DILocation(line: 71, column: 34, scope: !82, inlinedAt: !132)
!149 = !DILocation(line: 104, column: 60, scope: !111, inlinedAt: !148)
!150 = !DILocation(line: 105, column: 22, scope: !111, inlinedAt: !148)
!151 = !DILocation(line: 71, column: 72, scope: !82, inlinedAt: !132)
!152 = !DILocation(line: 27, column: 5, scope: !118, inlinedAt: !153)
!153 = distinct !DILocation(line: 71, column: 17, scope: !82, inlinedAt: !132)
!154 = !DILocation(line: 103, column: 35, scope: !111, inlinedAt: !155)
!155 = distinct !DILocation(line: 72, column: 34, scope: !82, inlinedAt: !132)
!156 = !DILocation(line: 104, column: 60, scope: !111, inlinedAt: !155)
!157 = !DILocation(line: 105, column: 22, scope: !111, inlinedAt: !155)
!158 = !DILocation(line: 72, column: 95, scope: !82, inlinedAt: !132)
!159 = !DILocation(line: 27, column: 5, scope: !118, inlinedAt: !160)
!160 = distinct !DILocation(line: 72, column: 17, scope: !82, inlinedAt: !132)
!161 = !DILocation(line: 75, column: 9, scope: !82, inlinedAt: !132)
!162 = !DILocation(line: 91, column: 5, scope: !9)
!163 = !DILocation(line: 93, column: 18, scope: !9)
!164 = !DILocation(line: 93, column: 9, scope: !9)
!165 = !DILocation(line: 94, column: 9, scope: !9)
!166 = !DILocation(line: 95, column: 5, scope: !9)
!167 = !DILocation(line: 97, column: 5, scope: !9)
!168 = !DILocation(line: 216, column: 18, scope: !9)
!169 = !DILocation(line: 216, column: 9, scope: !9)
!170 = !DILocation(line: 98, column: 35, scope: !9)
!171 = !DILocation(line: 98, column: 39, scope: !9)
!172 = !DILocation(line: 101, column: 13, scope: !9)
!173 = !DILocation(line: 26, column: 23, scope: !174, inlinedAt: !176)
!174 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !175, file: !175, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!175 = !DIFile(filename: "./utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x")
!176 = distinct !DILocation(line: 77, column: 39, scope: !177, inlinedAt: !178)
!177 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !175, file: !175, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!178 = distinct !DILocation(line: 102, column: 13, scope: !9)
!179 = !DILocation(line: 104, column: 9, scope: !9)
!180 = !DILocation(line: 105, column: 9, scope: !9)
!181 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !183)
!182 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !175, file: !175, line: 127, type: !10, scopeLine: 127, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!183 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !185)
!184 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 16, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 16, 16> >", scope: !175, file: !175, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!185 = distinct !DILocation(line: 107, column: 9, scope: !9)
!186 = !{i64 4294464}
!187 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !185)
!188 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !191)
!189 = distinct !DISubprogram(name: "operator=", scope: !190, file: !190, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!190 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!191 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !185)
!192 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !193)
!193 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !194)
!194 = distinct !DILocation(line: 108, column: 9, scope: !9)
!195 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !194)
!196 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !197)
!197 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !194)
!198 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !199)
!199 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !200)
!200 = distinct !DILocation(line: 109, column: 9, scope: !9)
!201 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !200)
!202 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !203)
!203 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !200)
!204 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !205)
!205 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !206)
!206 = distinct !DILocation(line: 110, column: 9, scope: !9)
!207 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !206)
!208 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !209)
!209 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !206)
!210 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !211)
!211 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !212)
!212 = distinct !DILocation(line: 111, column: 9, scope: !9)
!213 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !212)
!214 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !215)
!215 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !212)
!216 = !DILocation(line: 16, column: 5, scope: !217, inlinedAt: !218)
!217 = distinct !DISubprogram(name: "load_shared_vec", scope: !105, file: !105, line: 14, type: !10, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!218 = distinct !DILocation(line: 70, column: 37, scope: !219, inlinedAt: !221)
!219 = distinct !DISubprogram(name: "load<kittens::rt<__hip_bfloat16, 16, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 16, 16> >", scope: !220, file: !220, line: 28, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!220 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/shared_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!221 = distinct !DILocation(line: 112, column: 9, scope: !9)
!222 = !{i64 3063694, i64 3063726}
!223 = !DILocation(line: 72, column: 47, scope: !219, inlinedAt: !221)
!224 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !225)
!225 = distinct !DILocation(line: 73, column: 45, scope: !219, inlinedAt: !221)
!226 = !DILocation(line: 113, column: 9, scope: !9)
!227 = !DILocation(line: 114, column: 9, scope: !9)
!228 = !DILocation(line: 117, column: 9, scope: !9)
!229 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !232)
!230 = distinct !DISubprogram(name: "mfma161616", scope: !231, file: !231, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!231 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!232 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !234)
!233 = distinct !DISubprogram(name: "mma_ABt_base", scope: !231, file: !231, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!234 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !236)
!235 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 16, 16, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 16, 16, kittens::ducks::rt_layout::row>, kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col> >", scope: !231, file: !231, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!236 = distinct !DILocation(line: 118, column: 9, scope: !9)
!237 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !238)
!238 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !239)
!239 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !240)
!240 = distinct !DILocation(line: 119, column: 9, scope: !9)
!241 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !242)
!242 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !243)
!243 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !244)
!244 = distinct !DILocation(line: 120, column: 9, scope: !9)
!245 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !246)
!246 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !247)
!247 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !248)
!248 = distinct !DILocation(line: 121, column: 9, scope: !9)
!249 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !250)
!250 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !251)
!251 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !252)
!252 = distinct !DILocation(line: 122, column: 9, scope: !9)
!253 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !254)
!254 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !255)
!255 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !256)
!256 = distinct !DILocation(line: 123, column: 9, scope: !9)
!257 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !258)
!258 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !259)
!259 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !260)
!260 = distinct !DILocation(line: 124, column: 9, scope: !9)
!261 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !262)
!262 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !263)
!263 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !264)
!264 = distinct !DILocation(line: 125, column: 9, scope: !9)
!265 = !DILocation(line: 126, column: 9, scope: !9)
!266 = !DILocation(line: 127, column: 9, scope: !9)
!267 = !DILocation(line: 128, column: 9, scope: !9)
!268 = !DILocation(line: 131, column: 13, scope: !9)
!269 = !DILocation(line: 26, column: 23, scope: !174, inlinedAt: !270)
!270 = distinct !DILocation(line: 77, column: 39, scope: !177, inlinedAt: !271)
!271 = distinct !DILocation(line: 132, column: 13, scope: !9)
!272 = !DILocation(line: 134, column: 9, scope: !9)
!273 = !DILocation(line: 135, column: 9, scope: !9)
!274 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !275)
!275 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !276)
!276 = distinct !DILocation(line: 137, column: 9, scope: !9)
!277 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !276)
!278 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !279)
!279 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !276)
!280 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !281)
!281 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !282)
!282 = distinct !DILocation(line: 138, column: 9, scope: !9)
!283 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !282)
!284 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !285)
!285 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !282)
!286 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !287)
!287 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !288)
!288 = distinct !DILocation(line: 139, column: 9, scope: !9)
!289 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !288)
!290 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !291)
!291 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !288)
!292 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !293)
!293 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !294)
!294 = distinct !DILocation(line: 140, column: 9, scope: !9)
!295 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !294)
!296 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !297)
!297 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !294)
!298 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !299)
!299 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !300)
!300 = distinct !DILocation(line: 141, column: 9, scope: !9)
!301 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !300)
!302 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !303)
!303 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !300)
!304 = !DILocation(line: 16, column: 5, scope: !217, inlinedAt: !305)
!305 = distinct !DILocation(line: 70, column: 37, scope: !219, inlinedAt: !306)
!306 = distinct !DILocation(line: 142, column: 9, scope: !9)
!307 = !DILocation(line: 72, column: 47, scope: !219, inlinedAt: !306)
!308 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !309)
!309 = distinct !DILocation(line: 73, column: 45, scope: !219, inlinedAt: !306)
!310 = !DILocation(line: 143, column: 9, scope: !9)
!311 = !DILocation(line: 144, column: 9, scope: !9)
!312 = !DILocation(line: 147, column: 9, scope: !9)
!313 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !314)
!314 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !315)
!315 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !316)
!316 = distinct !DILocation(line: 148, column: 9, scope: !9)
!317 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !318)
!318 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !319)
!319 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !320)
!320 = distinct !DILocation(line: 149, column: 9, scope: !9)
!321 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !322)
!322 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !323)
!323 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !324)
!324 = distinct !DILocation(line: 150, column: 9, scope: !9)
!325 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !326)
!326 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !327)
!327 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !328)
!328 = distinct !DILocation(line: 151, column: 9, scope: !9)
!329 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !330)
!330 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !331)
!331 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !332)
!332 = distinct !DILocation(line: 152, column: 9, scope: !9)
!333 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !334)
!334 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !335)
!335 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !336)
!336 = distinct !DILocation(line: 153, column: 9, scope: !9)
!337 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !338)
!338 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !339)
!339 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !340)
!340 = distinct !DILocation(line: 154, column: 9, scope: !9)
!341 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !342)
!342 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !343)
!343 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !344)
!344 = distinct !DILocation(line: 155, column: 9, scope: !9)
!345 = !DILocation(line: 156, column: 9, scope: !9)
!346 = !DILocation(line: 157, column: 9, scope: !9)
!347 = !DILocation(line: 158, column: 9, scope: !9)
!348 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !349)
!349 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !350)
!350 = distinct !DILocation(line: 161, column: 9, scope: !9)
!351 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !350)
!352 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !353)
!353 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !350)
!354 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !355)
!355 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !356)
!356 = distinct !DILocation(line: 162, column: 9, scope: !9)
!357 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !356)
!358 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !359)
!359 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !356)
!360 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !361)
!361 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !362)
!362 = distinct !DILocation(line: 163, column: 9, scope: !9)
!363 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !362)
!364 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !365)
!365 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !362)
!366 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !367)
!367 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !368)
!368 = distinct !DILocation(line: 164, column: 9, scope: !9)
!369 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !368)
!370 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !371)
!371 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !368)
!372 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !373)
!373 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !374)
!374 = distinct !DILocation(line: 165, column: 9, scope: !9)
!375 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !374)
!376 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !377)
!377 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !374)
!378 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !379)
!379 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !380)
!380 = distinct !DILocation(line: 166, column: 9, scope: !9)
!381 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !380)
!382 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !383)
!383 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !380)
!384 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !385)
!385 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !386)
!386 = distinct !DILocation(line: 167, column: 9, scope: !9)
!387 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !386)
!388 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !389)
!389 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !386)
!390 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !391)
!391 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !392)
!392 = distinct !DILocation(line: 168, column: 9, scope: !9)
!393 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !392)
!394 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !395)
!395 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !392)
!396 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !397)
!397 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !398)
!398 = distinct !DILocation(line: 169, column: 9, scope: !9)
!399 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !398)
!400 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !401)
!401 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !398)
!402 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !403)
!403 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !404)
!404 = distinct !DILocation(line: 170, column: 9, scope: !9)
!405 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !404)
!406 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !407)
!407 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !404)
!408 = !DILocation(line: 129, column: 5, scope: !182, inlinedAt: !409)
!409 = distinct !DILocation(line: 185, column: 37, scope: !184, inlinedAt: !410)
!410 = distinct !DILocation(line: 171, column: 9, scope: !9)
!411 = !DILocation(line: 187, column: 47, scope: !184, inlinedAt: !410)
!412 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !413)
!413 = distinct !DILocation(line: 188, column: 45, scope: !184, inlinedAt: !410)
!414 = !DILocation(line: 16, column: 5, scope: !217, inlinedAt: !415)
!415 = distinct !DILocation(line: 70, column: 37, scope: !219, inlinedAt: !416)
!416 = distinct !DILocation(line: 172, column: 9, scope: !9)
!417 = !DILocation(line: 72, column: 47, scope: !219, inlinedAt: !416)
!418 = !DILocation(line: 494, column: 7, scope: !189, inlinedAt: !419)
!419 = distinct !DILocation(line: 73, column: 45, scope: !219, inlinedAt: !416)
!420 = !DILocation(line: 173, column: 9, scope: !9)
!421 = !DILocation(line: 174, column: 9, scope: !9)
!422 = !DILocation(line: 177, column: 9, scope: !9)
!423 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !424)
!424 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !425)
!425 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !426)
!426 = distinct !DILocation(line: 178, column: 9, scope: !9)
!427 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !428)
!428 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !429)
!429 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !430)
!430 = distinct !DILocation(line: 179, column: 9, scope: !9)
!431 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !432)
!432 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !433)
!433 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !434)
!434 = distinct !DILocation(line: 180, column: 9, scope: !9)
!435 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !436)
!436 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !437)
!437 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !438)
!438 = distinct !DILocation(line: 181, column: 9, scope: !9)
!439 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !440)
!440 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !441)
!441 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !442)
!442 = distinct !DILocation(line: 182, column: 9, scope: !9)
!443 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !444)
!444 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !445)
!445 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !446)
!446 = distinct !DILocation(line: 183, column: 9, scope: !9)
!447 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !448)
!448 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !449)
!449 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !450)
!450 = distinct !DILocation(line: 184, column: 9, scope: !9)
!451 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !452)
!452 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !453)
!453 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !454)
!454 = distinct !DILocation(line: 185, column: 9, scope: !9)
!455 = !DILocation(line: 186, column: 9, scope: !9)
!456 = !DILocation(line: 187, column: 9, scope: !9)
!457 = !DILocation(line: 188, column: 9, scope: !9)
!458 = !DILocation(line: 191, column: 13, scope: !9)
!459 = !DILocation(line: 192, column: 13, scope: !9)
!460 = !{i64 4307864}
!461 = !DILocation(line: 17, column: 5, scope: !462, inlinedAt: !463)
!462 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !175, file: !175, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!463 = distinct !DILocation(line: 119, column: 17, scope: !464, inlinedAt: !465)
!464 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !175, file: !175, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!465 = distinct !DILocation(line: 193, column: 13, scope: !9)
!466 = !{i64 4290490}
!467 = !DILocation(line: 17, column: 5, scope: !462, inlinedAt: !468)
!468 = distinct !DILocation(line: 120, column: 17, scope: !464, inlinedAt: !465)
!469 = !DILocation(line: 17, column: 5, scope: !462, inlinedAt: !470)
!470 = distinct !DILocation(line: 119, column: 17, scope: !464, inlinedAt: !471)
!471 = distinct !DILocation(line: 195, column: 13, scope: !9)
!472 = !DILocation(line: 17, column: 5, scope: !462, inlinedAt: !473)
!473 = distinct !DILocation(line: 120, column: 17, scope: !464, inlinedAt: !471)
!474 = !DILocation(line: 197, column: 9, scope: !9)
!475 = !DILocation(line: 198, column: 9, scope: !9)
!476 = !DILocation(line: 199, column: 9, scope: !9)
!477 = !DILocation(line: 202, column: 9, scope: !9)
!478 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !479)
!479 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !480)
!480 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !481)
!481 = distinct !DILocation(line: 203, column: 9, scope: !9)
!482 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !483)
!483 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !484)
!484 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !485)
!485 = distinct !DILocation(line: 204, column: 9, scope: !9)
!486 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !487)
!487 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !488)
!488 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !489)
!489 = distinct !DILocation(line: 205, column: 9, scope: !9)
!490 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !491)
!491 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !492)
!492 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !493)
!493 = distinct !DILocation(line: 206, column: 9, scope: !9)
!494 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !495)
!495 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !496)
!496 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !497)
!497 = distinct !DILocation(line: 207, column: 9, scope: !9)
!498 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !499)
!499 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !500)
!500 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !501)
!501 = distinct !DILocation(line: 208, column: 9, scope: !9)
!502 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !503)
!503 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !504)
!504 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !505)
!505 = distinct !DILocation(line: 209, column: 9, scope: !9)
!506 = !DILocation(line: 29, column: 27, scope: !230, inlinedAt: !507)
!507 = distinct !DILocation(line: 81, column: 5, scope: !233, inlinedAt: !508)
!508 = distinct !DILocation(line: 216, column: 13, scope: !235, inlinedAt: !509)
!509 = distinct !DILocation(line: 210, column: 9, scope: !9)
!510 = !DILocation(line: 211, column: 9, scope: !9)
!511 = !DILocation(line: 212, column: 9, scope: !9)
!512 = !DILocation(line: 213, column: 9, scope: !9)
!513 = !DILocation(line: 97, column: 29, scope: !9)
!514 = distinct !{!514, !167, !515, !516}
!515 = !DILocation(line: 214, column: 5, scope: !9)
!516 = !{!"llvm.loop.mustprogress"}
!517 = !DILocation(line: 217, column: 9, scope: !9)
!518 = !DILocation(line: 218, column: 5, scope: !9)
!519 = !DILocation(line: 227, column: 32, scope: !9)
!520 = !DILocation(line: 60, column: 34, scope: !521, inlinedAt: !522)
!521 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !90, file: !90, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!522 = distinct !DILocation(line: 192, column: 41, scope: !523, inlinedAt: !525)
!523 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col> > >", scope: !524, file: !524, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!524 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!525 = distinct !DILocation(line: 212, column: 5, scope: !526, inlinedAt: !527)
!526 = distinct !DISubprogram(name: "store<kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 16, 16, kittens::ducks::rt_layout::col> > >", scope: !524, file: !524, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!527 = distinct !DILocation(line: 224, column: 9, scope: !9)
!528 = !DILocation(line: 61, column: 18, scope: !521, inlinedAt: !522)
!529 = !DILocation(line: 63, column: 54, scope: !530, inlinedAt: !531)
!530 = distinct !DISubprogram(name: "operator[]", scope: !94, file: !94, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!531 = distinct !DILocation(line: 192, column: 23, scope: !523, inlinedAt: !525)
!532 = !DILocation(line: 63, column: 60, scope: !530, inlinedAt: !531)
!533 = !DILocation(line: 63, column: 68, scope: !530, inlinedAt: !531)
!534 = !DILocation(line: 63, column: 16, scope: !530, inlinedAt: !531)
!535 = !DILocation(line: 203, column: 49, scope: !523, inlinedAt: !525)
!536 = !DILocation(line: 203, column: 13, scope: !523, inlinedAt: !525)
!537 = !DILocation(line: 203, column: 47, scope: !523, inlinedAt: !525)
!538 = !{!539, !539, i64 0}
!539 = !{!"float", !540, i64 0}
!540 = !{!"omnipotent char", !541, i64 0}
!541 = !{!"Simple C++ TBAA"}
!542 = !DILocation(line: 204, column: 49, scope: !523, inlinedAt: !525)
!543 = !DILocation(line: 204, column: 13, scope: !523, inlinedAt: !525)
!544 = !DILocation(line: 204, column: 47, scope: !523, inlinedAt: !525)
!545 = !DILocation(line: 205, column: 49, scope: !523, inlinedAt: !525)
!546 = !DILocation(line: 205, column: 13, scope: !523, inlinedAt: !525)
!547 = !DILocation(line: 205, column: 47, scope: !523, inlinedAt: !525)
!548 = !DILocation(line: 206, column: 49, scope: !523, inlinedAt: !525)
!549 = !DILocation(line: 206, column: 13, scope: !523, inlinedAt: !525)
!550 = !DILocation(line: 206, column: 47, scope: !523, inlinedAt: !525)
!551 = !DILocation(line: 230, column: 1, scope: !9)
