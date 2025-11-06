; ModuleID = '256_256_32_16-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_256_32_16.cpp"
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
@__hip_cuid_97d6ca1748dc0448 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_97d6ca1748dc0448 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.0816.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.7820.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.7820.0.copyload = load i64, ptr addrspace(4) %.sroa.7820.0..sroa_idx, align 8
  %.sroa.9821.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.9821.0.copyload = load ptr, ptr addrspace(4) %.sroa.9821.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.17824.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.17824.0.copyload = load ptr, ptr addrspace(4) %.sroa.17824.0..sroa_idx, align 8
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
  %reass.sub.i.i141 = and i64 %7, -16, !dbg !25
  %9 = add i64 %reass.sub.i.i141, 16, !dbg !25
  %10 = inttoptr i64 %9 to ptr, !dbg !25
  %.not.i.i140 = icmp eq i64 %8, 0, !dbg !26
  %11 = select i1 %.not.i.i140, ptr %6, ptr %10, !dbg !25
  %12 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y(), !dbg !27
  %13 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr(), !dbg !33
  %14 = load i32, ptr addrspace(4) %13, align 4, !dbg !33, !tbaa !38
  %15 = mul i32 %14, %12, !dbg !42
  %16 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x(), !dbg !43
  %17 = add i32 %15, %16, !dbg !48
  %18 = sdiv i32 %17, 8, !dbg !49
  %19 = shl i32 %17, 7, !dbg !50
  %20 = mul i32 %18, -1023, !dbg !50
  %21 = add i32 %20, %19, !dbg !50
  %.frozen = freeze i32 %21, !dbg !51
  %22 = sdiv i32 %.frozen, 512, !dbg !51
  %23 = shl nsw i32 %22, 4, !dbg !52
  %24 = sub nsw i32 32, %23, !dbg !53
  %25 = tail call noundef range(i32 -8388592, 8388657) i32 @llvm.smin.i32(i32 %24, i32 16), !dbg !54
  %26 = mul i32 %22, 512, !dbg !58
  %.decomposed = sub i32 %.frozen, %26, !dbg !58
  %27 = sdiv i32 %.decomposed, %25, !dbg !59
  %28 = mul i32 %27, %25, !dbg !60
  %.decomposed1281 = sub i32 %.decomposed, %28, !dbg !60
  %29 = add nsw i32 %23, %.decomposed1281, !dbg !61
  %30 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !62
  %31 = lshr i32 %30, 6, !dbg !69
  %32 = lshr i32 %30, 8, !dbg !70
  %33 = and i32 %31, 3, !dbg !71
  %34 = trunc i64 %.sroa.7820.0.copyload to i32, !dbg !72
  %35 = shl nsw i32 %29, 8, !dbg !79
  %36 = sext i32 %35 to i64, !dbg !83
  %37 = mul i64 %.sroa.7820.0.copyload, %36, !dbg !87
  %38 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.0816.0.copyload, i64 %37, !dbg !88
  %39 = ptrtoint ptr %5 to i64, !dbg !89
  %40 = shl nuw nsw i32 %30, 3
  %41 = and i32 %40, 24
  %42 = lshr i32 %30, 2, !dbg !90
  %43 = and i32 %42, 127, !dbg !90
  %44 = mul i32 %43, %34, !dbg !91
  %45 = add i32 %44, %41, !dbg !92
  %46 = sext i32 %45 to i64, !dbg !93
  %47 = getelementptr inbounds %struct.__hip_bfloat16, ptr %38, i64 %46, !dbg !93
  %48 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %47) #7, !dbg !94, !srcloc !98
  %49 = or i32 %42, 128, !dbg !90
  %50 = mul i32 %49, %34, !dbg !91
  %51 = add i32 %50, %41, !dbg !92
  %52 = sext i32 %51 to i64, !dbg !93
  %53 = getelementptr inbounds %struct.__hip_bfloat16, ptr %38, i64 %52, !dbg !93
  %54 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %53) #7, !dbg !94, !srcloc !98
  %55 = trunc i64 %39 to i32, !dbg !89
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !99, !srcloc !100
  %56 = shl nuw nsw i32 %30, 4
  %.masked.i.i = and i32 %56, 8128
  %57 = shl nuw nsw i32 %41, 1
  %58 = add i32 %57, %55
  %59 = or disjoint i32 %57, 8
  %60 = add i32 %59, %55
  %61 = add i32 %58, %.masked.i.i, !dbg !101
  %62 = lshr exact i32 %61, 4, !dbg !105
  %63 = and i32 %62, 56, !dbg !105
  %64 = xor i32 %63, %61, !dbg !106
  %.sroa.0.0.extract.trunc.i.i = trunc i128 %48 to i64, !dbg !107
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %64, i64 %.sroa.0.0.extract.trunc.i.i) #7, !dbg !108, !srcloc !111
  %65 = add i32 %60, %.masked.i.i, !dbg !112
  %66 = lshr i32 %65, 4, !dbg !114
  %67 = and i32 %66, 56, !dbg !114
  %68 = xor i32 %67, %65, !dbg !115
  %.sroa.0.8.extract.shift.i.i = lshr i128 %48, 64, !dbg !116
  %.sroa.0.8.extract.trunc.i.i = trunc nuw i128 %.sroa.0.8.extract.shift.i.i to i64, !dbg !116
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %68, i64 %.sroa.0.8.extract.trunc.i.i) #7, !dbg !117, !srcloc !111
  %69 = or disjoint i32 %.masked.i.i, 8192, !dbg !119
  %70 = add i32 %58, %69, !dbg !101
  %71 = lshr exact i32 %70, 4, !dbg !105
  %72 = and i32 %71, 56, !dbg !105
  %73 = xor i32 %72, %70, !dbg !106
  %.sroa.5.16.extract.trunc.i.i = trunc i128 %54 to i64, !dbg !107
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %73, i64 %.sroa.5.16.extract.trunc.i.i) #7, !dbg !108, !srcloc !111
  %74 = add i32 %60, %69, !dbg !112
  %75 = lshr i32 %74, 4, !dbg !114
  %76 = and i32 %75, 56, !dbg !114
  %77 = xor i32 %76, %74, !dbg !115
  %.sroa.5.24.extract.shift.i.i = lshr i128 %54, 64, !dbg !116
  %.sroa.5.24.extract.trunc.i.i = trunc nuw i128 %.sroa.5.24.extract.shift.i.i to i64, !dbg !116
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %77, i64 %.sroa.5.24.extract.trunc.i.i) #7, !dbg !117, !srcloc !111
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !120, !srcloc !121
  %78 = trunc i64 %.sroa.15.0.copyload to i32, !dbg !122
  %79 = shl nsw i32 %27, 8, !dbg !125
  %80 = sext i32 %79 to i64, !dbg !127
  %81 = mul i64 %.sroa.15.0.copyload, %80, !dbg !129
  %82 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.9821.0.copyload, i64 %81, !dbg !130
  %83 = ptrtoint ptr %11 to i64, !dbg !131
  %84 = mul i32 %43, %78, !dbg !132
  %85 = add i32 %84, %41, !dbg !133
  %86 = sext i32 %85 to i64, !dbg !134
  %87 = getelementptr inbounds %struct.__hip_bfloat16, ptr %82, i64 %86, !dbg !134
  %88 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %87) #7, !dbg !135, !srcloc !98
  %89 = mul i32 %49, %78, !dbg !132
  %90 = add i32 %89, %41, !dbg !133
  %91 = sext i32 %90 to i64, !dbg !134
  %92 = getelementptr inbounds %struct.__hip_bfloat16, ptr %82, i64 %91, !dbg !134
  %93 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %92) #7, !dbg !135, !srcloc !98
  %94 = trunc i64 %83 to i32, !dbg !131
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !137, !srcloc !100
  %95 = add i32 %57, %94
  %96 = add i32 %59, %94
  %97 = add i32 %95, %.masked.i.i, !dbg !138
  %98 = lshr exact i32 %97, 4, !dbg !140
  %99 = and i32 %98, 56, !dbg !140
  %100 = xor i32 %99, %97, !dbg !141
  %.sroa.0.0.extract.trunc.i.i146 = trunc i128 %88 to i64, !dbg !142
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %100, i64 %.sroa.0.0.extract.trunc.i.i146) #7, !dbg !143, !srcloc !111
  %101 = add i32 %96, %.masked.i.i, !dbg !145
  %102 = lshr i32 %101, 4, !dbg !147
  %103 = and i32 %102, 56, !dbg !147
  %104 = xor i32 %103, %101, !dbg !148
  %.sroa.0.8.extract.shift.i.i147 = lshr i128 %88, 64, !dbg !149
  %.sroa.0.8.extract.trunc.i.i148 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i147 to i64, !dbg !149
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %104, i64 %.sroa.0.8.extract.trunc.i.i148) #7, !dbg !150, !srcloc !111
  %105 = add i32 %95, %69, !dbg !138
  %106 = lshr exact i32 %105, 4, !dbg !140
  %107 = and i32 %106, 56, !dbg !140
  %108 = xor i32 %107, %105, !dbg !141
  %.sroa.5.16.extract.trunc.i.i149 = trunc i128 %93 to i64, !dbg !142
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %108, i64 %.sroa.5.16.extract.trunc.i.i149) #7, !dbg !143, !srcloc !111
  %109 = add i32 %96, %69, !dbg !145
  %110 = lshr i32 %109, 4, !dbg !147
  %111 = and i32 %110, 56, !dbg !147
  %112 = xor i32 %111, %109, !dbg !148
  %.sroa.5.24.extract.shift.i.i150 = lshr i128 %93, 64, !dbg !149
  %.sroa.5.24.extract.trunc.i.i151 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i150 to i64, !dbg !149
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %112, i64 %.sroa.5.24.extract.trunc.i.i151) #7, !dbg !150, !srcloc !111
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !152, !srcloc !121
  tail call void @llvm.amdgcn.s.barrier(), !dbg !153
  %113 = icmp eq i32 %32, 1, !dbg !154
  br i1 %113, label %114, label %115, !dbg !155

114:                                              ; preds = %1
  tail call void @llvm.amdgcn.s.barrier(), !dbg !156
  br label %115, !dbg !157

115:                                              ; preds = %114, %1
  %116 = shl nuw nsw i32 %33, 5
  %117 = and i32 %30, 15
  %118 = and i32 %42, 12
  %119 = or disjoint i32 %116, %117
  %120 = shl nuw nsw i32 %119, 6
  %121 = shl nuw nsw i32 %118, 1
  %122 = add i32 %120, %94
  %123 = add i32 %122, %121
  %124 = lshr i32 %123, 4
  %125 = and i32 %124, 56
  %126 = xor i32 %125, %123
  %127 = shl i32 %51, 1
  %128 = shl i32 %45, 1
  %129 = shl i32 %34, 9
  %130 = or disjoint i32 %117, 16
  %131 = or disjoint i32 %116, %130
  %132 = shl nuw nsw i32 %131, 6
  %133 = add i32 %132, %94
  %134 = add i32 %133, %121
  %135 = lshr i32 %134, 4
  %136 = and i32 %135, 56
  %137 = xor i32 %136, %134
  %138 = or disjoint i32 %116, 128
  %139 = or disjoint i32 %138, %117
  %140 = shl nuw nsw i32 %139, 6
  %141 = add i32 %140, %94
  %142 = add i32 %141, %121
  %143 = lshr i32 %142, 4
  %144 = and i32 %143, 56
  %145 = xor i32 %144, %142
  %146 = or disjoint i32 %138, %130
  %147 = shl nuw nsw i32 %146, 6
  %148 = add i32 %147, %94
  %149 = add i32 %148, %121
  %150 = lshr i32 %149, 4
  %151 = and i32 %150, 56
  %152 = xor i32 %151, %149
  %153 = shl nuw nsw i32 %32, 5
  %154 = or disjoint i32 %153, %117
  %155 = shl nuw nsw i32 %154, 6
  %156 = add i32 %155, %55
  %157 = add i32 %156, %121
  %158 = lshr i32 %157, 4
  %159 = and i32 %158, 56
  %160 = xor i32 %159, %157
  %161 = or disjoint i32 %130, %153
  %162 = shl nuw nsw i32 %161, 6
  %163 = add i32 %162, %55
  %164 = add i32 %163, %121
  %165 = lshr i32 %164, 4
  %166 = and i32 %165, 56
  %167 = xor i32 %166, %164
  %168 = add nuw nsw i32 %153, 64
  %169 = or disjoint i32 %168, %117
  %170 = shl nuw nsw i32 %169, 6
  %171 = add i32 %170, %55
  %172 = add i32 %171, %121
  %173 = lshr i32 %172, 4
  %174 = and i32 %173, 56
  %175 = xor i32 %174, %172
  %176 = or disjoint i32 %168, %130
  %177 = shl nuw nsw i32 %176, 6
  %178 = add i32 %177, %55
  %179 = add i32 %178, %121
  %180 = lshr i32 %179, 4
  %181 = and i32 %180, 56
  %182 = xor i32 %181, %179
  %183 = or disjoint i32 %153, 128
  %184 = or disjoint i32 %183, %117
  %185 = shl nuw nsw i32 %184, 6
  %186 = add i32 %185, %55
  %187 = add i32 %186, %121
  %188 = lshr i32 %187, 4
  %189 = and i32 %188, 56
  %190 = xor i32 %189, %187
  %191 = or disjoint i32 %183, %130
  %192 = shl nuw nsw i32 %191, 6
  %193 = add i32 %192, %55
  %194 = add i32 %193, %121
  %195 = lshr i32 %194, 4
  %196 = and i32 %195, 56
  %197 = xor i32 %196, %194
  %198 = add nuw nsw i32 %153, 192
  %199 = or disjoint i32 %198, %117
  %200 = shl nuw nsw i32 %199, 6
  %201 = add i32 %200, %55
  %202 = add i32 %201, %121
  %203 = lshr i32 %202, 4
  %204 = and i32 %203, 56
  %205 = xor i32 %204, %202
  %206 = or disjoint i32 %198, %130
  %207 = shl nuw nsw i32 %206, 6
  %208 = add i32 %207, %55
  %209 = add i32 %208, %121
  %210 = lshr i32 %209, 4
  %211 = and i32 %210, 56
  %212 = xor i32 %211, %209
  %213 = shl i32 %90, 1
  %214 = shl i32 %85, 1
  %215 = shl i32 %78, 9
  %216 = or disjoint i32 %121, 32
  %217 = add i32 %122, %216
  %218 = lshr i32 %217, 4
  %219 = and i32 %218, 56
  %220 = xor i32 %219, %217
  %221 = add i32 %133, %216
  %222 = lshr i32 %221, 4
  %223 = and i32 %222, 56
  %224 = xor i32 %223, %221
  %225 = add i32 %141, %216
  %226 = lshr i32 %225, 4
  %227 = and i32 %226, 56
  %228 = xor i32 %227, %225
  %229 = add i32 %148, %216
  %230 = lshr i32 %229, 4
  %231 = and i32 %230, 56
  %232 = xor i32 %231, %229
  %233 = add i32 %156, %216
  %234 = lshr i32 %233, 4
  %235 = and i32 %234, 56
  %236 = xor i32 %235, %233
  %237 = add i32 %163, %216
  %238 = lshr i32 %237, 4
  %239 = and i32 %238, 56
  %240 = xor i32 %239, %237
  %241 = add i32 %171, %216
  %242 = lshr i32 %241, 4
  %243 = and i32 %242, 56
  %244 = xor i32 %243, %241
  %245 = add i32 %178, %216
  %246 = lshr i32 %245, 4
  %247 = and i32 %246, 56
  %248 = xor i32 %247, %245
  %249 = add i32 %186, %216
  %250 = lshr i32 %249, 4
  %251 = and i32 %250, 56
  %252 = xor i32 %251, %249
  %253 = add i32 %193, %216
  %254 = lshr i32 %253, 4
  %255 = and i32 %254, 56
  %256 = xor i32 %255, %253
  %257 = add i32 %201, %216
  %258 = lshr i32 %257, 4
  %259 = and i32 %258, 56
  %260 = xor i32 %259, %257
  %261 = add i32 %208, %216
  %262 = lshr i32 %261, 4
  %263 = and i32 %262, 56
  %264 = xor i32 %263, %261
  %265 = and i32 %56, 48
  %266 = add i32 %265, %55
  %267 = or disjoint i32 %265, 8
  %268 = add i32 %267, %55
  %269 = add i32 %266, %.masked.i.i
  %270 = lshr exact i32 %269, 4
  %271 = and i32 %270, 56
  %272 = xor i32 %271, %269
  %273 = add i32 %268, %.masked.i.i
  %274 = lshr i32 %273, 4
  %275 = and i32 %274, 56
  %276 = xor i32 %275, %273
  %277 = add i32 %266, %69
  %278 = lshr exact i32 %277, 4
  %279 = and i32 %278, 56
  %280 = xor i32 %279, %277
  %281 = add i32 %268, %69
  %282 = lshr i32 %281, 4
  %283 = and i32 %282, 56
  %284 = xor i32 %283, %281
  %285 = add i32 %265, %94
  %286 = add i32 %267, %94
  %287 = add i32 %285, %.masked.i.i
  %288 = lshr exact i32 %287, 4
  %289 = and i32 %288, 56
  %290 = xor i32 %289, %287
  %291 = add i32 %286, %.masked.i.i
  %292 = lshr i32 %291, 4
  %293 = and i32 %292, 56
  %294 = xor i32 %293, %291
  %295 = add i32 %285, %69
  %296 = lshr exact i32 %295, 4
  %297 = and i32 %296, 56
  %298 = xor i32 %297, %295
  %299 = add i32 %286, %69
  %300 = lshr i32 %299, 4
  %301 = and i32 %300, 56
  %302 = xor i32 %301, %299
  br label %305, !dbg !158

303:                                              ; preds = %525
  %304 = icmp ult i32 %30, 256, !dbg !159
  br i1 %304, label %558, label %559, !dbg !160

305:                                              ; preds = %115, %525
  %306 = phi <4 x float> [ zeroinitializer, %115 ], [ %557, %525 ]
  %307 = phi <4 x float> [ zeroinitializer, %115 ], [ %556, %525 ]
  %308 = phi <4 x float> [ zeroinitializer, %115 ], [ %555, %525 ]
  %.val30.i467932 = phi <4 x float> [ zeroinitializer, %115 ], [ %554, %525 ]
  %309 = phi <4 x float> [ zeroinitializer, %115 ], [ %553, %525 ]
  %310 = phi <4 x float> [ zeroinitializer, %115 ], [ %552, %525 ]
  %311 = phi <4 x float> [ zeroinitializer, %115 ], [ %551, %525 ]
  %.val30.i455923 = phi <4 x float> [ zeroinitializer, %115 ], [ %550, %525 ]
  %312 = phi <4 x float> [ zeroinitializer, %115 ], [ %549, %525 ]
  %313 = phi <4 x float> [ zeroinitializer, %115 ], [ %548, %525 ]
  %314 = phi <4 x float> [ zeroinitializer, %115 ], [ %547, %525 ]
  %.val30.i443914 = phi <4 x float> [ zeroinitializer, %115 ], [ %546, %525 ]
  %315 = phi <4 x float> [ zeroinitializer, %115 ], [ %545, %525 ]
  %316 = phi <4 x float> [ zeroinitializer, %115 ], [ %544, %525 ]
  %317 = phi <4 x float> [ zeroinitializer, %115 ], [ %543, %525 ]
  %.val30.i431905 = phi <4 x float> [ zeroinitializer, %115 ], [ %542, %525 ]
  %318 = phi <4 x float> [ zeroinitializer, %115 ], [ %541, %525 ]
  %319 = phi <4 x float> [ zeroinitializer, %115 ], [ %540, %525 ]
  %320 = phi <4 x float> [ zeroinitializer, %115 ], [ %539, %525 ]
  %.val30.i419896 = phi <4 x float> [ zeroinitializer, %115 ], [ %538, %525 ]
  %321 = phi <4 x float> [ zeroinitializer, %115 ], [ %537, %525 ]
  %322 = phi <4 x float> [ zeroinitializer, %115 ], [ %536, %525 ]
  %323 = phi <4 x float> [ zeroinitializer, %115 ], [ %535, %525 ]
  %.val30.i407887 = phi <4 x float> [ zeroinitializer, %115 ], [ %534, %525 ]
  %324 = phi <4 x float> [ zeroinitializer, %115 ], [ %533, %525 ]
  %325 = phi <4 x float> [ zeroinitializer, %115 ], [ %532, %525 ]
  %326 = phi <4 x float> [ zeroinitializer, %115 ], [ %531, %525 ]
  %.val30.i395878 = phi <4 x float> [ zeroinitializer, %115 ], [ %530, %525 ]
  %.val30.1.1.i488875 = phi <4 x float> [ zeroinitializer, %115 ], [ %529, %525 ]
  %.val30.135.i485872 = phi <4 x float> [ zeroinitializer, %115 ], [ %528, %525 ]
  %.val30.1.i482869 = phi <4 x float> [ zeroinitializer, %115 ], [ %527, %525 ]
  %.0137866 = phi i32 [ 0, %115 ], [ %327, %525 ]
  %.sroa.5665.0.off64865 = phi i64 [ 0, %115 ], [ %.sroa.5665.1.off64, %525 ]
  %.sroa.5665.0.off0864 = phi i64 [ undef, %115 ], [ %.sroa.5665.1.off0, %525 ]
  %.sroa.0664.0.off64863 = phi i64 [ 0, %115 ], [ %.sroa.0664.1.off64, %525 ]
  %.sroa.0664.0.off0862 = phi i64 [ undef, %115 ], [ %.sroa.0664.1.off0, %525 ]
  %.sroa.5.0.off64861 = phi i64 [ 0, %115 ], [ %.sroa.5.1.off64, %525 ]
  %.sroa.5.0.off0860 = phi i64 [ undef, %115 ], [ %.sroa.5.1.off0, %525 ]
  %.sroa.0662.0.off64859 = phi i64 [ 0, %115 ], [ %.sroa.0662.1.off64, %525 ]
  %.sroa.0662.0.off0858 = phi i64 [ undef, %115 ], [ %.sroa.0662.1.off0, %525 ]
  %.val30.i856857 = phi <4 x float> [ zeroinitializer, %115 ], [ %526, %525 ]
  %327 = add nuw nsw i32 %.0137866, 1, !dbg !161
  %.not = icmp eq i32 %.0137866, 255, !dbg !162
  br i1 %.not, label %337, label %328, !dbg !163

328:                                              ; preds = %305
  %329 = shl nuw nsw i32 %327, 5
  %330 = zext nneg i32 %329 to i64
  %331 = getelementptr %struct.__hip_bfloat16, ptr %38, i64 %330
  %332 = ptrtoint ptr %331 to i64
  %333 = bitcast i64 %332 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %333, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %334 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %334, i32 %129, i64 2
  %335 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %128, i32 noundef 0, i32 noundef 0) #7, !dbg !164
  %336 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %127, i32 noundef 0, i32 noundef 0) #7, !dbg !164
  %extract.t841 = trunc i128 %335 to i64, !dbg !170
  %extract843 = lshr i128 %335, 64, !dbg !170
  %extract.t844 = trunc nuw i128 %extract843 to i64, !dbg !170
  %extract.t848 = trunc i128 %336 to i64, !dbg !170
  %extract850 = lshr i128 %336, 64, !dbg !170
  %extract.t851 = trunc nuw i128 %extract850 to i64, !dbg !170
  br label %337, !dbg !170

337:                                              ; preds = %328, %305
  %.sroa.0664.1.off0 = phi i64 [ %.sroa.0664.0.off0862, %305 ], [ %extract.t841, %328 ]
  %.sroa.0664.1.off64 = phi i64 [ %.sroa.0664.0.off64863, %305 ], [ %extract.t844, %328 ]
  %.sroa.5665.1.off0 = phi i64 [ %.sroa.5665.0.off0864, %305 ], [ %extract.t848, %328 ]
  %.sroa.5665.1.off64 = phi i64 [ %.sroa.5665.0.off64865, %305 ], [ %extract.t851, %328 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !171
  %338 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %126) #7, !dbg !172, !srcloc !177
  %.sroa.036.2.extract.shift37.i = lshr i64 %338, 16, !dbg !178
  %339 = insertelement <2 x i64> poison, i64 %338, i64 0, !dbg !178
  %340 = insertelement <2 x i64> %339, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !178
  %341 = shufflevector <2 x i64> %339, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !172
  %342 = lshr <2 x i64> %341, <i64 32, i64 48>, !dbg !172
  %343 = shufflevector <2 x i64> %340, <2 x i64> %342, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !179
  %.sroa.0700.6.vec.insert1218 = trunc <4 x i64> %343 to <4 x i16>, !dbg !179
  %344 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %137) #7, !dbg !172, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %344, 16, !dbg !178
  %345 = insertelement <2 x i64> poison, i64 %344, i64 0, !dbg !178
  %346 = insertelement <2 x i64> %345, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !178
  %347 = shufflevector <2 x i64> %345, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !172
  %348 = lshr <2 x i64> %347, <i64 32, i64 48>, !dbg !172
  %349 = shufflevector <2 x i64> %346, <2 x i64> %348, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !179
  %.sroa.14709.14.vec.insert1217 = trunc <4 x i64> %349 to <4 x i16>, !dbg !179
  %350 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %145) #7, !dbg !183, !srcloc !177
  %.sroa.036.2.extract.shift37.i160 = lshr i64 %350, 16, !dbg !186
  %351 = insertelement <2 x i64> poison, i64 %350, i64 0, !dbg !186
  %352 = insertelement <2 x i64> %351, i64 %.sroa.036.2.extract.shift37.i160, i64 1, !dbg !186
  %353 = shufflevector <2 x i64> %351, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !183
  %354 = lshr <2 x i64> %353, <i64 32, i64 48>, !dbg !183
  %355 = shufflevector <2 x i64> %352, <2 x i64> %354, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !187
  %.sroa.0667.6.vec.insert1216 = trunc <4 x i64> %355 to <4 x i16>, !dbg !187
  %356 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %152) #7, !dbg !183, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i169 = lshr i64 %356, 16, !dbg !186
  %357 = insertelement <2 x i64> poison, i64 %356, i64 0, !dbg !186
  %358 = insertelement <2 x i64> %357, i64 %.sroa.036.2.extract.shift37.1.i169, i64 1, !dbg !186
  %359 = shufflevector <2 x i64> %357, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !183
  %360 = lshr <2 x i64> %359, <i64 32, i64 48>, !dbg !183
  %361 = shufflevector <2 x i64> %358, <2 x i64> %360, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !187
  %.sroa.14.14.vec.insert1215 = trunc <4 x i64> %361 to <4 x i16>, !dbg !187
  %362 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %160) #7, !dbg !189, !srcloc !177
  %.sroa.036.2.extract.shift37.i181 = lshr i64 %362, 16, !dbg !192
  %363 = insertelement <2 x i64> poison, i64 %362, i64 0, !dbg !192
  %364 = insertelement <2 x i64> %363, i64 %.sroa.036.2.extract.shift37.i181, i64 1, !dbg !192
  %365 = shufflevector <2 x i64> %363, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !189
  %366 = lshr <2 x i64> %365, <i64 32, i64 48>, !dbg !189
  %367 = shufflevector <2 x i64> %364, <2 x i64> %366, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !193
  %.sroa.0794.6.vec.insert1214 = trunc <4 x i64> %367 to <4 x i16>, !dbg !193
  %368 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %167) #7, !dbg !189, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i190 = lshr i64 %368, 16, !dbg !192
  %369 = insertelement <2 x i64> poison, i64 %368, i64 0, !dbg !192
  %370 = insertelement <2 x i64> %369, i64 %.sroa.036.2.extract.shift37.1.i190, i64 1, !dbg !192
  %371 = shufflevector <2 x i64> %369, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !189
  %372 = lshr <2 x i64> %371, <i64 32, i64 48>, !dbg !189
  %373 = shufflevector <2 x i64> %370, <2 x i64> %372, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !193
  %.sroa.10799.14.vec.insert1213 = trunc <4 x i64> %373 to <4 x i16>, !dbg !193
  %374 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %175) #7, !dbg !195, !srcloc !177
  %.sroa.036.2.extract.shift37.i202 = lshr i64 %374, 16, !dbg !198
  %375 = insertelement <2 x i64> poison, i64 %374, i64 0, !dbg !198
  %376 = insertelement <2 x i64> %375, i64 %.sroa.036.2.extract.shift37.i202, i64 1, !dbg !198
  %377 = shufflevector <2 x i64> %375, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !195
  %378 = lshr <2 x i64> %377, <i64 32, i64 48>, !dbg !195
  %379 = shufflevector <2 x i64> %376, <2 x i64> %378, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !199
  %.sroa.0774.6.vec.insert1212 = trunc <4 x i64> %379 to <4 x i16>, !dbg !199
  %380 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %182) #7, !dbg !195, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i211 = lshr i64 %380, 16, !dbg !198
  %381 = insertelement <2 x i64> poison, i64 %380, i64 0, !dbg !198
  %382 = insertelement <2 x i64> %381, i64 %.sroa.036.2.extract.shift37.1.i211, i64 1, !dbg !198
  %383 = shufflevector <2 x i64> %381, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !195
  %384 = lshr <2 x i64> %383, <i64 32, i64 48>, !dbg !195
  %385 = shufflevector <2 x i64> %382, <2 x i64> %384, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !199
  %.sroa.10779.14.vec.insert1211 = trunc <4 x i64> %385 to <4 x i16>, !dbg !199
  %386 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %190) #7, !dbg !201, !srcloc !177
  %.sroa.036.2.extract.shift37.i223 = lshr i64 %386, 16, !dbg !204
  %387 = insertelement <2 x i64> poison, i64 %386, i64 0, !dbg !204
  %388 = insertelement <2 x i64> %387, i64 %.sroa.036.2.extract.shift37.i223, i64 1, !dbg !204
  %389 = shufflevector <2 x i64> %387, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %390 = lshr <2 x i64> %389, <i64 32, i64 48>, !dbg !201
  %391 = shufflevector <2 x i64> %388, <2 x i64> %390, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.0754.6.vec.insert1210 = trunc <4 x i64> %391 to <4 x i16>, !dbg !205
  %392 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %197) #7, !dbg !201, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i232 = lshr i64 %392, 16, !dbg !204
  %393 = insertelement <2 x i64> poison, i64 %392, i64 0, !dbg !204
  %394 = insertelement <2 x i64> %393, i64 %.sroa.036.2.extract.shift37.1.i232, i64 1, !dbg !204
  %395 = shufflevector <2 x i64> %393, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %396 = lshr <2 x i64> %395, <i64 32, i64 48>, !dbg !201
  %397 = shufflevector <2 x i64> %394, <2 x i64> %396, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.10759.14.vec.insert1209 = trunc <4 x i64> %397 to <4 x i16>, !dbg !205
  %398 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %205) #7, !dbg !207, !srcloc !177
  %.sroa.036.2.extract.shift37.i244 = lshr i64 %398, 16, !dbg !210
  %399 = insertelement <2 x i64> poison, i64 %398, i64 0, !dbg !210
  %400 = insertelement <2 x i64> %399, i64 %.sroa.036.2.extract.shift37.i244, i64 1, !dbg !210
  %401 = shufflevector <2 x i64> %399, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %402 = lshr <2 x i64> %401, <i64 32, i64 48>, !dbg !207
  %403 = shufflevector <2 x i64> %400, <2 x i64> %402, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.0736.6.vec.insert1208 = trunc <4 x i64> %403 to <4 x i16>, !dbg !211
  %404 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %212) #7, !dbg !207, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i253 = lshr i64 %404, 16, !dbg !210
  %405 = insertelement <2 x i64> poison, i64 %404, i64 0, !dbg !210
  %406 = insertelement <2 x i64> %405, i64 %.sroa.036.2.extract.shift37.1.i253, i64 1, !dbg !210
  %407 = shufflevector <2 x i64> %405, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %408 = lshr <2 x i64> %407, <i64 32, i64 48>, !dbg !207
  %409 = shufflevector <2 x i64> %406, <2 x i64> %408, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.10.14.vec.insert1207 = trunc <4 x i64> %409 to <4 x i16>, !dbg !211
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !213
  br i1 %.not, label %419, label %410, !dbg !214

410:                                              ; preds = %337
  %411 = shl nuw nsw i32 %327, 5
  %412 = zext nneg i32 %411 to i64
  %413 = getelementptr %struct.__hip_bfloat16, ptr %82, i64 %412
  %414 = ptrtoint ptr %413 to i64
  %415 = bitcast i64 %414 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i261 = shufflevector <2 x i32> %415, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %416 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i261, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i262 = insertelement <4 x i32> %416, i32 %215, i64 2
  %417 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i262, i32 noundef %214, i32 noundef 0, i32 noundef 0) #7, !dbg !215
  %418 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i262, i32 noundef %213, i32 noundef 0, i32 noundef 0) #7, !dbg !215
  %extract.t827 = trunc i128 %417 to i64, !dbg !218
  %extract829 = lshr i128 %417, 64, !dbg !218
  %extract.t830 = trunc nuw i128 %extract829 to i64, !dbg !218
  %extract.t834 = trunc i128 %418 to i64, !dbg !218
  %extract836 = lshr i128 %418, 64, !dbg !218
  %extract.t837 = trunc nuw i128 %extract836 to i64, !dbg !218
  br label %419, !dbg !218

419:                                              ; preds = %410, %337
  %.sroa.0662.1.off0 = phi i64 [ %.sroa.0662.0.off0858, %337 ], [ %extract.t827, %410 ]
  %.sroa.0662.1.off64 = phi i64 [ %.sroa.0662.0.off64859, %337 ], [ %extract.t830, %410 ]
  %.sroa.5.1.off0 = phi i64 [ %.sroa.5.0.off0860, %337 ], [ %extract.t834, %410 ]
  %.sroa.5.1.off64 = phi i64 [ %.sroa.5.0.off64861, %337 ], [ %extract.t837, %410 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !219
  %420 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %220) #7, !dbg !220, !srcloc !177
  %.sroa.036.2.extract.shift37.i270 = lshr i64 %420, 16, !dbg !223
  %421 = insertelement <2 x i64> poison, i64 %420, i64 0, !dbg !223
  %422 = insertelement <2 x i64> %421, i64 %.sroa.036.2.extract.shift37.i270, i64 1, !dbg !223
  %423 = shufflevector <2 x i64> %421, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !220
  %424 = lshr <2 x i64> %423, <i64 32, i64 48>, !dbg !220
  %425 = shufflevector <2 x i64> %422, <2 x i64> %424, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !224
  %.sroa.26718.22.vec.insert1206 = trunc <4 x i64> %425 to <4 x i16>, !dbg !224
  %426 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %224) #7, !dbg !220, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i279 = lshr i64 %426, 16, !dbg !223
  %427 = insertelement <2 x i64> poison, i64 %426, i64 0, !dbg !223
  %428 = insertelement <2 x i64> %427, i64 %.sroa.036.2.extract.shift37.1.i279, i64 1, !dbg !223
  %429 = shufflevector <2 x i64> %427, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !220
  %430 = lshr <2 x i64> %429, <i64 32, i64 48>, !dbg !220
  %431 = shufflevector <2 x i64> %428, <2 x i64> %430, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !224
  %.sroa.38727.30.vec.insert1205 = trunc <4 x i64> %431 to <4 x i16>, !dbg !224
  %432 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %228) #7, !dbg !226, !srcloc !177
  %.sroa.036.2.extract.shift37.i291 = lshr i64 %432, 16, !dbg !229
  %433 = insertelement <2 x i64> poison, i64 %432, i64 0, !dbg !229
  %434 = insertelement <2 x i64> %433, i64 %.sroa.036.2.extract.shift37.i291, i64 1, !dbg !229
  %435 = shufflevector <2 x i64> %433, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !226
  %436 = lshr <2 x i64> %435, <i64 32, i64 48>, !dbg !226
  %437 = shufflevector <2 x i64> %434, <2 x i64> %436, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !230
  %.sroa.26.22.vec.insert1204 = trunc <4 x i64> %437 to <4 x i16>, !dbg !230
  %438 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %232) #7, !dbg !226, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i300 = lshr i64 %438, 16, !dbg !229
  %439 = insertelement <2 x i64> poison, i64 %438, i64 0, !dbg !229
  %440 = insertelement <2 x i64> %439, i64 %.sroa.036.2.extract.shift37.1.i300, i64 1, !dbg !229
  %441 = shufflevector <2 x i64> %439, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !226
  %442 = lshr <2 x i64> %441, <i64 32, i64 48>, !dbg !226
  %443 = shufflevector <2 x i64> %440, <2 x i64> %442, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !230
  %.sroa.38.30.vec.insert1203 = trunc <4 x i64> %443 to <4 x i16>, !dbg !230
  %444 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %236) #7, !dbg !232, !srcloc !177
  %.sroa.036.2.extract.shift37.i312 = lshr i64 %444, 16, !dbg !235
  %445 = insertelement <2 x i64> poison, i64 %444, i64 0, !dbg !235
  %446 = insertelement <2 x i64> %445, i64 %.sroa.036.2.extract.shift37.i312, i64 1, !dbg !235
  %447 = shufflevector <2 x i64> %445, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !232
  %448 = lshr <2 x i64> %447, <i64 32, i64 48>, !dbg !232
  %449 = shufflevector <2 x i64> %446, <2 x i64> %448, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !236
  %.sroa.18804.22.vec.insert1202 = trunc <4 x i64> %449 to <4 x i16>, !dbg !236
  %450 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %240) #7, !dbg !232, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i321 = lshr i64 %450, 16, !dbg !235
  %451 = insertelement <2 x i64> poison, i64 %450, i64 0, !dbg !235
  %452 = insertelement <2 x i64> %451, i64 %.sroa.036.2.extract.shift37.1.i321, i64 1, !dbg !235
  %453 = shufflevector <2 x i64> %451, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !232
  %454 = lshr <2 x i64> %453, <i64 32, i64 48>, !dbg !232
  %455 = shufflevector <2 x i64> %452, <2 x i64> %454, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !236
  %.sroa.26809.30.vec.insert1201 = trunc <4 x i64> %455 to <4 x i16>, !dbg !236
  %456 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %244) #7, !dbg !238, !srcloc !177
  %.sroa.036.2.extract.shift37.i333 = lshr i64 %456, 16, !dbg !241
  %457 = insertelement <2 x i64> poison, i64 %456, i64 0, !dbg !241
  %458 = insertelement <2 x i64> %457, i64 %.sroa.036.2.extract.shift37.i333, i64 1, !dbg !241
  %459 = shufflevector <2 x i64> %457, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !238
  %460 = lshr <2 x i64> %459, <i64 32, i64 48>, !dbg !238
  %461 = shufflevector <2 x i64> %458, <2 x i64> %460, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !242
  %.sroa.18784.22.vec.insert1200 = trunc <4 x i64> %461 to <4 x i16>, !dbg !242
  %462 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %248) #7, !dbg !238, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i342 = lshr i64 %462, 16, !dbg !241
  %463 = insertelement <2 x i64> poison, i64 %462, i64 0, !dbg !241
  %464 = insertelement <2 x i64> %463, i64 %.sroa.036.2.extract.shift37.1.i342, i64 1, !dbg !241
  %465 = shufflevector <2 x i64> %463, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !238
  %466 = lshr <2 x i64> %465, <i64 32, i64 48>, !dbg !238
  %467 = shufflevector <2 x i64> %464, <2 x i64> %466, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !242
  %.sroa.26789.30.vec.insert1199 = trunc <4 x i64> %467 to <4 x i16>, !dbg !242
  %468 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %252) #7, !dbg !244, !srcloc !177
  %.sroa.036.2.extract.shift37.i354 = lshr i64 %468, 16, !dbg !247
  %469 = insertelement <2 x i64> poison, i64 %468, i64 0, !dbg !247
  %470 = insertelement <2 x i64> %469, i64 %.sroa.036.2.extract.shift37.i354, i64 1, !dbg !247
  %471 = shufflevector <2 x i64> %469, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !244
  %472 = lshr <2 x i64> %471, <i64 32, i64 48>, !dbg !244
  %473 = shufflevector <2 x i64> %470, <2 x i64> %472, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !248
  %.sroa.18764.22.vec.insert1198 = trunc <4 x i64> %473 to <4 x i16>, !dbg !248
  %474 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %256) #7, !dbg !244, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i363 = lshr i64 %474, 16, !dbg !247
  %475 = insertelement <2 x i64> poison, i64 %474, i64 0, !dbg !247
  %476 = insertelement <2 x i64> %475, i64 %.sroa.036.2.extract.shift37.1.i363, i64 1, !dbg !247
  %477 = shufflevector <2 x i64> %475, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !244
  %478 = lshr <2 x i64> %477, <i64 32, i64 48>, !dbg !244
  %479 = shufflevector <2 x i64> %476, <2 x i64> %478, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !248
  %.sroa.26769.30.vec.insert1197 = trunc <4 x i64> %479 to <4 x i16>, !dbg !248
  %480 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %260) #7, !dbg !250, !srcloc !177
  %.sroa.036.2.extract.shift37.i375 = lshr i64 %480, 16, !dbg !253
  %481 = insertelement <2 x i64> poison, i64 %480, i64 0, !dbg !253
  %482 = insertelement <2 x i64> %481, i64 %.sroa.036.2.extract.shift37.i375, i64 1, !dbg !253
  %483 = shufflevector <2 x i64> %481, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !250
  %484 = lshr <2 x i64> %483, <i64 32, i64 48>, !dbg !250
  %485 = shufflevector <2 x i64> %482, <2 x i64> %484, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !254
  %.sroa.18.22.vec.insert1196 = trunc <4 x i64> %485 to <4 x i16>, !dbg !254
  %486 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %264) #7, !dbg !250, !srcloc !177
  %.sroa.036.2.extract.shift37.1.i384 = lshr i64 %486, 16, !dbg !253
  %487 = insertelement <2 x i64> poison, i64 %486, i64 0, !dbg !253
  %488 = insertelement <2 x i64> %487, i64 %.sroa.036.2.extract.shift37.1.i384, i64 1, !dbg !253
  %489 = shufflevector <2 x i64> %487, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !250
  %490 = lshr <2 x i64> %489, <i64 32, i64 48>, !dbg !250
  %491 = shufflevector <2 x i64> %488, <2 x i64> %490, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !254
  %.sroa.26749.30.vec.insert1195 = trunc <4 x i64> %491 to <4 x i16>, !dbg !254
  tail call void @llvm.amdgcn.s.barrier(), !dbg !256
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !257
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !258
  %492 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0794.6.vec.insert1214, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %.val30.i856857, i32 0, i32 0, i32 0), !dbg !259
  %493 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0794.6.vec.insert1214, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %.val30.1.i482869, i32 0, i32 0, i32 0), !dbg !259
  %494 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10799.14.vec.insert1213, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %.val30.135.i485872, i32 0, i32 0, i32 0), !dbg !259
  %495 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10799.14.vec.insert1213, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %.val30.1.1.i488875, i32 0, i32 0, i32 0), !dbg !259
  %496 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0794.6.vec.insert1214, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %.val30.i395878, i32 0, i32 0, i32 0), !dbg !267
  %497 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0794.6.vec.insert1214, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %326, i32 0, i32 0, i32 0), !dbg !267
  %498 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10799.14.vec.insert1213, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %325, i32 0, i32 0, i32 0), !dbg !267
  %499 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10799.14.vec.insert1213, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %324, i32 0, i32 0, i32 0), !dbg !267
  %500 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0774.6.vec.insert1212, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %.val30.i407887, i32 0, i32 0, i32 0), !dbg !271
  %501 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0774.6.vec.insert1212, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %323, i32 0, i32 0, i32 0), !dbg !271
  %502 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10779.14.vec.insert1211, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %322, i32 0, i32 0, i32 0), !dbg !271
  %503 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10779.14.vec.insert1211, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %321, i32 0, i32 0, i32 0), !dbg !271
  %504 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0774.6.vec.insert1212, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %.val30.i419896, i32 0, i32 0, i32 0), !dbg !275
  %505 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0774.6.vec.insert1212, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %320, i32 0, i32 0, i32 0), !dbg !275
  %506 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10779.14.vec.insert1211, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %319, i32 0, i32 0, i32 0), !dbg !275
  %507 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10779.14.vec.insert1211, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %318, i32 0, i32 0, i32 0), !dbg !275
  %508 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0754.6.vec.insert1210, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %.val30.i431905, i32 0, i32 0, i32 0), !dbg !279
  %509 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0754.6.vec.insert1210, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %317, i32 0, i32 0, i32 0), !dbg !279
  %510 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10759.14.vec.insert1209, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %316, i32 0, i32 0, i32 0), !dbg !279
  %511 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10759.14.vec.insert1209, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %315, i32 0, i32 0, i32 0), !dbg !279
  %512 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0754.6.vec.insert1210, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %.val30.i443914, i32 0, i32 0, i32 0), !dbg !283
  %513 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0754.6.vec.insert1210, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %314, i32 0, i32 0, i32 0), !dbg !283
  %514 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10759.14.vec.insert1209, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %313, i32 0, i32 0, i32 0), !dbg !283
  %515 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10759.14.vec.insert1209, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %312, i32 0, i32 0, i32 0), !dbg !283
  %516 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0736.6.vec.insert1208, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %.val30.i455923, i32 0, i32 0, i32 0), !dbg !287
  %517 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0736.6.vec.insert1208, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %311, i32 0, i32 0, i32 0), !dbg !287
  %518 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10.14.vec.insert1207, <4 x i16> %.sroa.0700.6.vec.insert1218, <4 x float> %310, i32 0, i32 0, i32 0), !dbg !287
  %519 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10.14.vec.insert1207, <4 x i16> %.sroa.14709.14.vec.insert1217, <4 x float> %309, i32 0, i32 0, i32 0), !dbg !287
  %520 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0736.6.vec.insert1208, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %.val30.i467932, i32 0, i32 0, i32 0), !dbg !291
  %521 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0736.6.vec.insert1208, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %308, i32 0, i32 0, i32 0), !dbg !291
  %522 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10.14.vec.insert1207, <4 x i16> %.sroa.0667.6.vec.insert1216, <4 x float> %307, i32 0, i32 0, i32 0), !dbg !291
  %523 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10.14.vec.insert1207, <4 x i16> %.sroa.14.14.vec.insert1215, <4 x float> %306, i32 0, i32 0, i32 0), !dbg !291
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !295
  tail call void @llvm.amdgcn.s.barrier(), !dbg !296
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !297
  br i1 %.not, label %525, label %524, !dbg !298

524:                                              ; preds = %419
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !299, !srcloc !300
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %272, i64 %.sroa.0664.1.off0) #7, !dbg !301, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %276, i64 %.sroa.0664.1.off64) #7, !dbg !307, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %280, i64 %.sroa.5665.1.off0) #7, !dbg !301, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %284, i64 %.sroa.5665.1.off64) #7, !dbg !307, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %290, i64 %.sroa.0662.1.off0) #7, !dbg !309, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %294, i64 %.sroa.0662.1.off64) #7, !dbg !312, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %298, i64 %.sroa.5.1.off0) #7, !dbg !309, !srcloc !306
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %302, i64 %.sroa.5.1.off64) #7, !dbg !312, !srcloc !306
  br label %525, !dbg !314

525:                                              ; preds = %524, %419
  tail call void @llvm.amdgcn.s.barrier(), !dbg !315
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !316
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !317
  %526 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18804.22.vec.insert1202, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %492, i32 0, i32 0, i32 0), !dbg !318
  %527 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18804.22.vec.insert1202, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %493, i32 0, i32 0, i32 0), !dbg !318
  %528 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26809.30.vec.insert1201, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %494, i32 0, i32 0, i32 0), !dbg !318
  %529 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26809.30.vec.insert1201, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %495, i32 0, i32 0, i32 0), !dbg !318
  %530 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18804.22.vec.insert1202, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %496, i32 0, i32 0, i32 0), !dbg !322
  %531 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18804.22.vec.insert1202, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %497, i32 0, i32 0, i32 0), !dbg !322
  %532 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26809.30.vec.insert1201, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %498, i32 0, i32 0, i32 0), !dbg !322
  %533 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26809.30.vec.insert1201, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %499, i32 0, i32 0, i32 0), !dbg !322
  %534 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18784.22.vec.insert1200, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %500, i32 0, i32 0, i32 0), !dbg !326
  %535 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18784.22.vec.insert1200, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %501, i32 0, i32 0, i32 0), !dbg !326
  %536 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26789.30.vec.insert1199, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %502, i32 0, i32 0, i32 0), !dbg !326
  %537 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26789.30.vec.insert1199, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %503, i32 0, i32 0, i32 0), !dbg !326
  %538 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18784.22.vec.insert1200, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %504, i32 0, i32 0, i32 0), !dbg !330
  %539 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18784.22.vec.insert1200, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %505, i32 0, i32 0, i32 0), !dbg !330
  %540 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26789.30.vec.insert1199, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %506, i32 0, i32 0, i32 0), !dbg !330
  %541 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26789.30.vec.insert1199, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %507, i32 0, i32 0, i32 0), !dbg !330
  %542 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18764.22.vec.insert1198, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %508, i32 0, i32 0, i32 0), !dbg !334
  %543 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18764.22.vec.insert1198, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %509, i32 0, i32 0, i32 0), !dbg !334
  %544 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26769.30.vec.insert1197, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %510, i32 0, i32 0, i32 0), !dbg !334
  %545 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26769.30.vec.insert1197, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %511, i32 0, i32 0, i32 0), !dbg !334
  %546 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18764.22.vec.insert1198, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %512, i32 0, i32 0, i32 0), !dbg !338
  %547 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18764.22.vec.insert1198, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %513, i32 0, i32 0, i32 0), !dbg !338
  %548 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26769.30.vec.insert1197, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %514, i32 0, i32 0, i32 0), !dbg !338
  %549 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26769.30.vec.insert1197, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %515, i32 0, i32 0, i32 0), !dbg !338
  %550 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18.22.vec.insert1196, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %516, i32 0, i32 0, i32 0), !dbg !342
  %551 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18.22.vec.insert1196, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %517, i32 0, i32 0, i32 0), !dbg !342
  %552 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26749.30.vec.insert1195, <4 x i16> %.sroa.26718.22.vec.insert1206, <4 x float> %518, i32 0, i32 0, i32 0), !dbg !342
  %553 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26749.30.vec.insert1195, <4 x i16> %.sroa.38727.30.vec.insert1205, <4 x float> %519, i32 0, i32 0, i32 0), !dbg !342
  %554 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18.22.vec.insert1196, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %520, i32 0, i32 0, i32 0), !dbg !346
  %555 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18.22.vec.insert1196, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %521, i32 0, i32 0, i32 0), !dbg !346
  %556 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26749.30.vec.insert1195, <4 x i16> %.sroa.26.22.vec.insert1204, <4 x float> %522, i32 0, i32 0, i32 0), !dbg !346
  %557 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26749.30.vec.insert1195, <4 x i16> %.sroa.38.30.vec.insert1203, <4 x float> %523, i32 0, i32 0, i32 0), !dbg !346
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !350
  tail call void @llvm.amdgcn.s.barrier(), !dbg !351
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !352
  %exitcond.not = icmp eq i32 %327, 256, !dbg !353
  br i1 %exitcond.not, label %303, label %305, !dbg !158, !llvm.loop !354

558:                                              ; preds = %303
  tail call void @llvm.amdgcn.s.barrier(), !dbg !357
  br label %559, !dbg !358

559:                                              ; preds = %558, %303
  %560 = shl nsw i32 %29, 3
  %561 = or disjoint i32 %560, %32
  %562 = shl nsw i32 %27, 3
  %563 = trunc i64 %.sroa.20.0.copyload to i32
  %564 = mul nsw i32 %118, %563
  %565 = or disjoint i32 %118, 1
  %566 = mul nsw i32 %565, %563
  %567 = or disjoint i32 %118, 2
  %568 = mul nsw i32 %567, %563
  %569 = or disjoint i32 %118, 3
  %570 = mul nsw i32 %569, %563
  %571 = add nsw i32 %564, %117
  %572 = sext i32 %571 to i64
  %573 = add nsw i32 %566, %117
  %574 = sext i32 %573 to i64
  %575 = add nsw i32 %568, %117
  %576 = sext i32 %575 to i64
  %577 = add nsw i32 %570, %117
  %578 = sext i32 %577 to i64
  %579 = add nsw i32 %564, %130
  %580 = sext i32 %579 to i64
  %581 = add nsw i32 %566, %130
  %582 = sext i32 %581 to i64
  %583 = add nsw i32 %568, %130
  %584 = sext i32 %583 to i64
  %585 = add nsw i32 %570, %130
  %586 = sext i32 %585 to i64
  %587 = or disjoint i32 %118, 16
  %588 = mul nsw i32 %587, %563
  %589 = or disjoint i32 %118, 17
  %590 = mul nsw i32 %589, %563
  %591 = or disjoint i32 %118, 18
  %592 = mul nsw i32 %591, %563
  %593 = or disjoint i32 %118, 19
  %594 = mul nsw i32 %593, %563
  %595 = add nsw i32 %588, %117
  %596 = sext i32 %595 to i64
  %597 = add nsw i32 %590, %117
  %598 = sext i32 %597 to i64
  %599 = add nsw i32 %592, %117
  %600 = sext i32 %599 to i64
  %601 = add nsw i32 %594, %117
  %602 = sext i32 %601 to i64
  %603 = add nsw i32 %588, %130
  %604 = sext i32 %603 to i64
  %605 = add nsw i32 %590, %130
  %606 = sext i32 %605 to i64
  %607 = add nsw i32 %592, %130
  %608 = sext i32 %607 to i64
  %609 = add nsw i32 %594, %130
  %610 = sext i32 %609 to i64
  %611 = or disjoint i32 %562, %33, !dbg !359
  %612 = shl nsw i32 %561, 5, !dbg !360
  %613 = shl nsw i32 %611, 5, !dbg !368
  %614 = sext i32 %612 to i64, !dbg !369
  %615 = mul i64 %.sroa.20.0.copyload, %614, !dbg !372
  %616 = sext i32 %613 to i64, !dbg !373
  %617 = getelementptr float, ptr %.sroa.17824.0.copyload, i64 %615, !dbg !374
  %618 = getelementptr float, ptr %617, i64 %616, !dbg !374
  %.sroa.0.0.vec.extract = extractelement <4 x float> %526, i64 0, !dbg !375
  %619 = getelementptr inbounds float, ptr %618, i64 %572, !dbg !376
  store float %.sroa.0.0.vec.extract, ptr %619, align 4, !dbg !377, !tbaa !378
  %.sroa.0.4.vec.extract = extractelement <4 x float> %526, i64 1, !dbg !382
  %620 = getelementptr inbounds float, ptr %618, i64 %574, !dbg !383
  store float %.sroa.0.4.vec.extract, ptr %620, align 4, !dbg !384, !tbaa !378
  %.sroa.0.8.vec.extract = extractelement <4 x float> %526, i64 2, !dbg !385
  %621 = getelementptr inbounds float, ptr %618, i64 %576, !dbg !386
  store float %.sroa.0.8.vec.extract, ptr %621, align 4, !dbg !387, !tbaa !378
  %.sroa.0.12.vec.extract = extractelement <4 x float> %526, i64 3, !dbg !388
  %622 = getelementptr inbounds float, ptr %618, i64 %578, !dbg !389
  store float %.sroa.0.12.vec.extract, ptr %622, align 4, !dbg !390, !tbaa !378
  %.sroa.12.16.vec.extract = extractelement <4 x float> %527, i64 0, !dbg !375
  %623 = getelementptr inbounds float, ptr %618, i64 %580, !dbg !376
  store float %.sroa.12.16.vec.extract, ptr %623, align 4, !dbg !377, !tbaa !378
  %.sroa.12.20.vec.extract = extractelement <4 x float> %527, i64 1, !dbg !382
  %624 = getelementptr inbounds float, ptr %618, i64 %582, !dbg !383
  store float %.sroa.12.20.vec.extract, ptr %624, align 4, !dbg !384, !tbaa !378
  %.sroa.12.24.vec.extract = extractelement <4 x float> %527, i64 2, !dbg !385
  %625 = getelementptr inbounds float, ptr %618, i64 %584, !dbg !386
  store float %.sroa.12.24.vec.extract, ptr %625, align 4, !dbg !387, !tbaa !378
  %.sroa.12.28.vec.extract = extractelement <4 x float> %527, i64 3, !dbg !388
  %626 = getelementptr inbounds float, ptr %618, i64 %586, !dbg !389
  store float %.sroa.12.28.vec.extract, ptr %626, align 4, !dbg !390, !tbaa !378
  %.sroa.22.32.vec.extract = extractelement <4 x float> %528, i64 0, !dbg !375
  %627 = getelementptr inbounds float, ptr %618, i64 %596, !dbg !376
  store float %.sroa.22.32.vec.extract, ptr %627, align 4, !dbg !377, !tbaa !378
  %.sroa.22.36.vec.extract = extractelement <4 x float> %528, i64 1, !dbg !382
  %628 = getelementptr inbounds float, ptr %618, i64 %598, !dbg !383
  store float %.sroa.22.36.vec.extract, ptr %628, align 4, !dbg !384, !tbaa !378
  %.sroa.22.40.vec.extract = extractelement <4 x float> %528, i64 2, !dbg !385
  %629 = getelementptr inbounds float, ptr %618, i64 %600, !dbg !386
  store float %.sroa.22.40.vec.extract, ptr %629, align 4, !dbg !387, !tbaa !378
  %.sroa.22.44.vec.extract = extractelement <4 x float> %528, i64 3, !dbg !388
  %630 = getelementptr inbounds float, ptr %618, i64 %602, !dbg !389
  store float %.sroa.22.44.vec.extract, ptr %630, align 4, !dbg !390, !tbaa !378
  %.sroa.32.48.vec.extract = extractelement <4 x float> %529, i64 0, !dbg !375
  %631 = getelementptr inbounds float, ptr %618, i64 %604, !dbg !376
  store float %.sroa.32.48.vec.extract, ptr %631, align 4, !dbg !377, !tbaa !378
  %.sroa.32.52.vec.extract = extractelement <4 x float> %529, i64 1, !dbg !382
  %632 = getelementptr inbounds float, ptr %618, i64 %606, !dbg !383
  store float %.sroa.32.52.vec.extract, ptr %632, align 4, !dbg !384, !tbaa !378
  %.sroa.32.56.vec.extract = extractelement <4 x float> %529, i64 2, !dbg !385
  %633 = getelementptr inbounds float, ptr %618, i64 %608, !dbg !386
  store float %.sroa.32.56.vec.extract, ptr %633, align 4, !dbg !387, !tbaa !378
  %.sroa.32.60.vec.extract = extractelement <4 x float> %529, i64 3, !dbg !388
  %634 = getelementptr inbounds float, ptr %618, i64 %610, !dbg !389
  store float %.sroa.32.60.vec.extract, ptr %634, align 4, !dbg !390, !tbaa !378
  %635 = or disjoint i32 %562, %33, !dbg !359
  %636 = shl i32 %635, 5, !dbg !368
  %637 = or disjoint i32 %636, 128, !dbg !368
  %638 = sext i32 %637 to i64, !dbg !373
  %639 = getelementptr float, ptr %617, i64 %638, !dbg !374
  %.sroa.42.64.vec.extract = extractelement <4 x float> %530, i64 0, !dbg !375
  %640 = getelementptr inbounds float, ptr %639, i64 %572, !dbg !376
  store float %.sroa.42.64.vec.extract, ptr %640, align 4, !dbg !377, !tbaa !378
  %.sroa.42.68.vec.extract = extractelement <4 x float> %530, i64 1, !dbg !382
  %641 = getelementptr inbounds float, ptr %639, i64 %574, !dbg !383
  store float %.sroa.42.68.vec.extract, ptr %641, align 4, !dbg !384, !tbaa !378
  %.sroa.42.72.vec.extract = extractelement <4 x float> %530, i64 2, !dbg !385
  %642 = getelementptr inbounds float, ptr %639, i64 %576, !dbg !386
  store float %.sroa.42.72.vec.extract, ptr %642, align 4, !dbg !387, !tbaa !378
  %.sroa.42.76.vec.extract = extractelement <4 x float> %530, i64 3, !dbg !388
  %643 = getelementptr inbounds float, ptr %639, i64 %578, !dbg !389
  store float %.sroa.42.76.vec.extract, ptr %643, align 4, !dbg !390, !tbaa !378
  %.sroa.52.80.vec.extract = extractelement <4 x float> %531, i64 0, !dbg !375
  %644 = getelementptr inbounds float, ptr %639, i64 %580, !dbg !376
  store float %.sroa.52.80.vec.extract, ptr %644, align 4, !dbg !377, !tbaa !378
  %.sroa.52.84.vec.extract = extractelement <4 x float> %531, i64 1, !dbg !382
  %645 = getelementptr inbounds float, ptr %639, i64 %582, !dbg !383
  store float %.sroa.52.84.vec.extract, ptr %645, align 4, !dbg !384, !tbaa !378
  %.sroa.52.88.vec.extract = extractelement <4 x float> %531, i64 2, !dbg !385
  %646 = getelementptr inbounds float, ptr %639, i64 %584, !dbg !386
  store float %.sroa.52.88.vec.extract, ptr %646, align 4, !dbg !387, !tbaa !378
  %.sroa.52.92.vec.extract = extractelement <4 x float> %531, i64 3, !dbg !388
  %647 = getelementptr inbounds float, ptr %639, i64 %586, !dbg !389
  store float %.sroa.52.92.vec.extract, ptr %647, align 4, !dbg !390, !tbaa !378
  %.sroa.62.96.vec.extract = extractelement <4 x float> %532, i64 0, !dbg !375
  %648 = getelementptr inbounds float, ptr %639, i64 %596, !dbg !376
  store float %.sroa.62.96.vec.extract, ptr %648, align 4, !dbg !377, !tbaa !378
  %.sroa.62.100.vec.extract = extractelement <4 x float> %532, i64 1, !dbg !382
  %649 = getelementptr inbounds float, ptr %639, i64 %598, !dbg !383
  store float %.sroa.62.100.vec.extract, ptr %649, align 4, !dbg !384, !tbaa !378
  %.sroa.62.104.vec.extract = extractelement <4 x float> %532, i64 2, !dbg !385
  %650 = getelementptr inbounds float, ptr %639, i64 %600, !dbg !386
  store float %.sroa.62.104.vec.extract, ptr %650, align 4, !dbg !387, !tbaa !378
  %.sroa.62.108.vec.extract = extractelement <4 x float> %532, i64 3, !dbg !388
  %651 = getelementptr inbounds float, ptr %639, i64 %602, !dbg !389
  store float %.sroa.62.108.vec.extract, ptr %651, align 4, !dbg !390, !tbaa !378
  %.sroa.72.112.vec.extract = extractelement <4 x float> %533, i64 0, !dbg !375
  %652 = getelementptr inbounds float, ptr %639, i64 %604, !dbg !376
  store float %.sroa.72.112.vec.extract, ptr %652, align 4, !dbg !377, !tbaa !378
  %.sroa.72.116.vec.extract = extractelement <4 x float> %533, i64 1, !dbg !382
  %653 = getelementptr inbounds float, ptr %639, i64 %606, !dbg !383
  store float %.sroa.72.116.vec.extract, ptr %653, align 4, !dbg !384, !tbaa !378
  %.sroa.72.120.vec.extract = extractelement <4 x float> %533, i64 2, !dbg !385
  %654 = getelementptr inbounds float, ptr %639, i64 %608, !dbg !386
  store float %.sroa.72.120.vec.extract, ptr %654, align 4, !dbg !387, !tbaa !378
  %.sroa.72.124.vec.extract = extractelement <4 x float> %533, i64 3, !dbg !388
  %655 = getelementptr inbounds float, ptr %639, i64 %610, !dbg !389
  store float %.sroa.72.124.vec.extract, ptr %655, align 4, !dbg !390, !tbaa !378
  %656 = shl i32 %561, 5, !dbg !360
  %657 = add nuw nsw i32 %656, 64, !dbg !360
  %658 = sext i32 %657 to i64, !dbg !369
  %659 = mul i64 %.sroa.20.0.copyload, %658, !dbg !372
  %660 = getelementptr float, ptr %.sroa.17824.0.copyload, i64 %659, !dbg !374
  %661 = getelementptr float, ptr %660, i64 %616, !dbg !374
  %.sroa.82.128.vec.extract = extractelement <4 x float> %534, i64 0, !dbg !375
  %662 = getelementptr inbounds float, ptr %661, i64 %572, !dbg !376
  store float %.sroa.82.128.vec.extract, ptr %662, align 4, !dbg !377, !tbaa !378
  %.sroa.82.132.vec.extract = extractelement <4 x float> %534, i64 1, !dbg !382
  %663 = getelementptr inbounds float, ptr %661, i64 %574, !dbg !383
  store float %.sroa.82.132.vec.extract, ptr %663, align 4, !dbg !384, !tbaa !378
  %.sroa.82.136.vec.extract = extractelement <4 x float> %534, i64 2, !dbg !385
  %664 = getelementptr inbounds float, ptr %661, i64 %576, !dbg !386
  store float %.sroa.82.136.vec.extract, ptr %664, align 4, !dbg !387, !tbaa !378
  %.sroa.82.140.vec.extract = extractelement <4 x float> %534, i64 3, !dbg !388
  %665 = getelementptr inbounds float, ptr %661, i64 %578, !dbg !389
  store float %.sroa.82.140.vec.extract, ptr %665, align 4, !dbg !390, !tbaa !378
  %.sroa.92.144.vec.extract = extractelement <4 x float> %535, i64 0, !dbg !375
  %666 = getelementptr inbounds float, ptr %661, i64 %580, !dbg !376
  store float %.sroa.92.144.vec.extract, ptr %666, align 4, !dbg !377, !tbaa !378
  %.sroa.92.148.vec.extract = extractelement <4 x float> %535, i64 1, !dbg !382
  %667 = getelementptr inbounds float, ptr %661, i64 %582, !dbg !383
  store float %.sroa.92.148.vec.extract, ptr %667, align 4, !dbg !384, !tbaa !378
  %.sroa.92.152.vec.extract = extractelement <4 x float> %535, i64 2, !dbg !385
  %668 = getelementptr inbounds float, ptr %661, i64 %584, !dbg !386
  store float %.sroa.92.152.vec.extract, ptr %668, align 4, !dbg !387, !tbaa !378
  %.sroa.92.156.vec.extract = extractelement <4 x float> %535, i64 3, !dbg !388
  %669 = getelementptr inbounds float, ptr %661, i64 %586, !dbg !389
  store float %.sroa.92.156.vec.extract, ptr %669, align 4, !dbg !390, !tbaa !378
  %.sroa.102.160.vec.extract = extractelement <4 x float> %536, i64 0, !dbg !375
  %670 = getelementptr inbounds float, ptr %661, i64 %596, !dbg !376
  store float %.sroa.102.160.vec.extract, ptr %670, align 4, !dbg !377, !tbaa !378
  %.sroa.102.164.vec.extract = extractelement <4 x float> %536, i64 1, !dbg !382
  %671 = getelementptr inbounds float, ptr %661, i64 %598, !dbg !383
  store float %.sroa.102.164.vec.extract, ptr %671, align 4, !dbg !384, !tbaa !378
  %.sroa.102.168.vec.extract = extractelement <4 x float> %536, i64 2, !dbg !385
  %672 = getelementptr inbounds float, ptr %661, i64 %600, !dbg !386
  store float %.sroa.102.168.vec.extract, ptr %672, align 4, !dbg !387, !tbaa !378
  %.sroa.102.172.vec.extract = extractelement <4 x float> %536, i64 3, !dbg !388
  %673 = getelementptr inbounds float, ptr %661, i64 %602, !dbg !389
  store float %.sroa.102.172.vec.extract, ptr %673, align 4, !dbg !390, !tbaa !378
  %.sroa.112.176.vec.extract = extractelement <4 x float> %537, i64 0, !dbg !375
  %674 = getelementptr inbounds float, ptr %661, i64 %604, !dbg !376
  store float %.sroa.112.176.vec.extract, ptr %674, align 4, !dbg !377, !tbaa !378
  %.sroa.112.180.vec.extract = extractelement <4 x float> %537, i64 1, !dbg !382
  %675 = getelementptr inbounds float, ptr %661, i64 %606, !dbg !383
  store float %.sroa.112.180.vec.extract, ptr %675, align 4, !dbg !384, !tbaa !378
  %.sroa.112.184.vec.extract = extractelement <4 x float> %537, i64 2, !dbg !385
  %676 = getelementptr inbounds float, ptr %661, i64 %608, !dbg !386
  store float %.sroa.112.184.vec.extract, ptr %676, align 4, !dbg !387, !tbaa !378
  %.sroa.112.188.vec.extract = extractelement <4 x float> %537, i64 3, !dbg !388
  %677 = getelementptr inbounds float, ptr %661, i64 %610, !dbg !389
  store float %.sroa.112.188.vec.extract, ptr %677, align 4, !dbg !390, !tbaa !378
  %678 = getelementptr float, ptr %660, i64 %638, !dbg !374
  %.sroa.122.192.vec.extract = extractelement <4 x float> %538, i64 0, !dbg !375
  %679 = getelementptr inbounds float, ptr %678, i64 %572, !dbg !376
  store float %.sroa.122.192.vec.extract, ptr %679, align 4, !dbg !377, !tbaa !378
  %.sroa.122.196.vec.extract = extractelement <4 x float> %538, i64 1, !dbg !382
  %680 = getelementptr inbounds float, ptr %678, i64 %574, !dbg !383
  store float %.sroa.122.196.vec.extract, ptr %680, align 4, !dbg !384, !tbaa !378
  %.sroa.122.200.vec.extract = extractelement <4 x float> %538, i64 2, !dbg !385
  %681 = getelementptr inbounds float, ptr %678, i64 %576, !dbg !386
  store float %.sroa.122.200.vec.extract, ptr %681, align 4, !dbg !387, !tbaa !378
  %.sroa.122.204.vec.extract = extractelement <4 x float> %538, i64 3, !dbg !388
  %682 = getelementptr inbounds float, ptr %678, i64 %578, !dbg !389
  store float %.sroa.122.204.vec.extract, ptr %682, align 4, !dbg !390, !tbaa !378
  %.sroa.132.208.vec.extract = extractelement <4 x float> %539, i64 0, !dbg !375
  %683 = getelementptr inbounds float, ptr %678, i64 %580, !dbg !376
  store float %.sroa.132.208.vec.extract, ptr %683, align 4, !dbg !377, !tbaa !378
  %.sroa.132.212.vec.extract = extractelement <4 x float> %539, i64 1, !dbg !382
  %684 = getelementptr inbounds float, ptr %678, i64 %582, !dbg !383
  store float %.sroa.132.212.vec.extract, ptr %684, align 4, !dbg !384, !tbaa !378
  %.sroa.132.216.vec.extract = extractelement <4 x float> %539, i64 2, !dbg !385
  %685 = getelementptr inbounds float, ptr %678, i64 %584, !dbg !386
  store float %.sroa.132.216.vec.extract, ptr %685, align 4, !dbg !387, !tbaa !378
  %.sroa.132.220.vec.extract = extractelement <4 x float> %539, i64 3, !dbg !388
  %686 = getelementptr inbounds float, ptr %678, i64 %586, !dbg !389
  store float %.sroa.132.220.vec.extract, ptr %686, align 4, !dbg !390, !tbaa !378
  %.sroa.142.224.vec.extract = extractelement <4 x float> %540, i64 0, !dbg !375
  %687 = getelementptr inbounds float, ptr %678, i64 %596, !dbg !376
  store float %.sroa.142.224.vec.extract, ptr %687, align 4, !dbg !377, !tbaa !378
  %.sroa.142.228.vec.extract = extractelement <4 x float> %540, i64 1, !dbg !382
  %688 = getelementptr inbounds float, ptr %678, i64 %598, !dbg !383
  store float %.sroa.142.228.vec.extract, ptr %688, align 4, !dbg !384, !tbaa !378
  %.sroa.142.232.vec.extract = extractelement <4 x float> %540, i64 2, !dbg !385
  %689 = getelementptr inbounds float, ptr %678, i64 %600, !dbg !386
  store float %.sroa.142.232.vec.extract, ptr %689, align 4, !dbg !387, !tbaa !378
  %.sroa.142.236.vec.extract = extractelement <4 x float> %540, i64 3, !dbg !388
  %690 = getelementptr inbounds float, ptr %678, i64 %602, !dbg !389
  store float %.sroa.142.236.vec.extract, ptr %690, align 4, !dbg !390, !tbaa !378
  %.sroa.152.240.vec.extract = extractelement <4 x float> %541, i64 0, !dbg !375
  %691 = getelementptr inbounds float, ptr %678, i64 %604, !dbg !376
  store float %.sroa.152.240.vec.extract, ptr %691, align 4, !dbg !377, !tbaa !378
  %.sroa.152.244.vec.extract = extractelement <4 x float> %541, i64 1, !dbg !382
  %692 = getelementptr inbounds float, ptr %678, i64 %606, !dbg !383
  store float %.sroa.152.244.vec.extract, ptr %692, align 4, !dbg !384, !tbaa !378
  %.sroa.152.248.vec.extract = extractelement <4 x float> %541, i64 2, !dbg !385
  %693 = getelementptr inbounds float, ptr %678, i64 %608, !dbg !386
  store float %.sroa.152.248.vec.extract, ptr %693, align 4, !dbg !387, !tbaa !378
  %.sroa.152.252.vec.extract = extractelement <4 x float> %541, i64 3, !dbg !388
  %694 = getelementptr inbounds float, ptr %678, i64 %610, !dbg !389
  store float %.sroa.152.252.vec.extract, ptr %694, align 4, !dbg !390, !tbaa !378
  %695 = shl i32 %561, 5, !dbg !360
  %696 = or disjoint i32 %695, 128, !dbg !360
  %697 = sext i32 %696 to i64, !dbg !369
  %698 = mul i64 %.sroa.20.0.copyload, %697, !dbg !372
  %699 = getelementptr float, ptr %.sroa.17824.0.copyload, i64 %698, !dbg !374
  %700 = getelementptr float, ptr %699, i64 %616, !dbg !374
  %.sroa.162.256.vec.extract = extractelement <4 x float> %542, i64 0, !dbg !375
  %701 = getelementptr inbounds float, ptr %700, i64 %572, !dbg !376
  store float %.sroa.162.256.vec.extract, ptr %701, align 4, !dbg !377, !tbaa !378
  %.sroa.162.260.vec.extract = extractelement <4 x float> %542, i64 1, !dbg !382
  %702 = getelementptr inbounds float, ptr %700, i64 %574, !dbg !383
  store float %.sroa.162.260.vec.extract, ptr %702, align 4, !dbg !384, !tbaa !378
  %.sroa.162.264.vec.extract = extractelement <4 x float> %542, i64 2, !dbg !385
  %703 = getelementptr inbounds float, ptr %700, i64 %576, !dbg !386
  store float %.sroa.162.264.vec.extract, ptr %703, align 4, !dbg !387, !tbaa !378
  %.sroa.162.268.vec.extract = extractelement <4 x float> %542, i64 3, !dbg !388
  %704 = getelementptr inbounds float, ptr %700, i64 %578, !dbg !389
  store float %.sroa.162.268.vec.extract, ptr %704, align 4, !dbg !390, !tbaa !378
  %.sroa.172.272.vec.extract = extractelement <4 x float> %543, i64 0, !dbg !375
  %705 = getelementptr inbounds float, ptr %700, i64 %580, !dbg !376
  store float %.sroa.172.272.vec.extract, ptr %705, align 4, !dbg !377, !tbaa !378
  %.sroa.172.276.vec.extract = extractelement <4 x float> %543, i64 1, !dbg !382
  %706 = getelementptr inbounds float, ptr %700, i64 %582, !dbg !383
  store float %.sroa.172.276.vec.extract, ptr %706, align 4, !dbg !384, !tbaa !378
  %.sroa.172.280.vec.extract = extractelement <4 x float> %543, i64 2, !dbg !385
  %707 = getelementptr inbounds float, ptr %700, i64 %584, !dbg !386
  store float %.sroa.172.280.vec.extract, ptr %707, align 4, !dbg !387, !tbaa !378
  %.sroa.172.284.vec.extract = extractelement <4 x float> %543, i64 3, !dbg !388
  %708 = getelementptr inbounds float, ptr %700, i64 %586, !dbg !389
  store float %.sroa.172.284.vec.extract, ptr %708, align 4, !dbg !390, !tbaa !378
  %.sroa.182.288.vec.extract = extractelement <4 x float> %544, i64 0, !dbg !375
  %709 = getelementptr inbounds float, ptr %700, i64 %596, !dbg !376
  store float %.sroa.182.288.vec.extract, ptr %709, align 4, !dbg !377, !tbaa !378
  %.sroa.182.292.vec.extract = extractelement <4 x float> %544, i64 1, !dbg !382
  %710 = getelementptr inbounds float, ptr %700, i64 %598, !dbg !383
  store float %.sroa.182.292.vec.extract, ptr %710, align 4, !dbg !384, !tbaa !378
  %.sroa.182.296.vec.extract = extractelement <4 x float> %544, i64 2, !dbg !385
  %711 = getelementptr inbounds float, ptr %700, i64 %600, !dbg !386
  store float %.sroa.182.296.vec.extract, ptr %711, align 4, !dbg !387, !tbaa !378
  %.sroa.182.300.vec.extract = extractelement <4 x float> %544, i64 3, !dbg !388
  %712 = getelementptr inbounds float, ptr %700, i64 %602, !dbg !389
  store float %.sroa.182.300.vec.extract, ptr %712, align 4, !dbg !390, !tbaa !378
  %.sroa.192.304.vec.extract = extractelement <4 x float> %545, i64 0, !dbg !375
  %713 = getelementptr inbounds float, ptr %700, i64 %604, !dbg !376
  store float %.sroa.192.304.vec.extract, ptr %713, align 4, !dbg !377, !tbaa !378
  %.sroa.192.308.vec.extract = extractelement <4 x float> %545, i64 1, !dbg !382
  %714 = getelementptr inbounds float, ptr %700, i64 %606, !dbg !383
  store float %.sroa.192.308.vec.extract, ptr %714, align 4, !dbg !384, !tbaa !378
  %.sroa.192.312.vec.extract = extractelement <4 x float> %545, i64 2, !dbg !385
  %715 = getelementptr inbounds float, ptr %700, i64 %608, !dbg !386
  store float %.sroa.192.312.vec.extract, ptr %715, align 4, !dbg !387, !tbaa !378
  %.sroa.192.316.vec.extract = extractelement <4 x float> %545, i64 3, !dbg !388
  %716 = getelementptr inbounds float, ptr %700, i64 %610, !dbg !389
  store float %.sroa.192.316.vec.extract, ptr %716, align 4, !dbg !390, !tbaa !378
  %717 = getelementptr float, ptr %699, i64 %638, !dbg !374
  %.sroa.202.320.vec.extract = extractelement <4 x float> %546, i64 0, !dbg !375
  %718 = getelementptr inbounds float, ptr %717, i64 %572, !dbg !376
  store float %.sroa.202.320.vec.extract, ptr %718, align 4, !dbg !377, !tbaa !378
  %.sroa.202.324.vec.extract = extractelement <4 x float> %546, i64 1, !dbg !382
  %719 = getelementptr inbounds float, ptr %717, i64 %574, !dbg !383
  store float %.sroa.202.324.vec.extract, ptr %719, align 4, !dbg !384, !tbaa !378
  %.sroa.202.328.vec.extract = extractelement <4 x float> %546, i64 2, !dbg !385
  %720 = getelementptr inbounds float, ptr %717, i64 %576, !dbg !386
  store float %.sroa.202.328.vec.extract, ptr %720, align 4, !dbg !387, !tbaa !378
  %.sroa.202.332.vec.extract = extractelement <4 x float> %546, i64 3, !dbg !388
  %721 = getelementptr inbounds float, ptr %717, i64 %578, !dbg !389
  store float %.sroa.202.332.vec.extract, ptr %721, align 4, !dbg !390, !tbaa !378
  %.sroa.212.336.vec.extract = extractelement <4 x float> %547, i64 0, !dbg !375
  %722 = getelementptr inbounds float, ptr %717, i64 %580, !dbg !376
  store float %.sroa.212.336.vec.extract, ptr %722, align 4, !dbg !377, !tbaa !378
  %.sroa.212.340.vec.extract = extractelement <4 x float> %547, i64 1, !dbg !382
  %723 = getelementptr inbounds float, ptr %717, i64 %582, !dbg !383
  store float %.sroa.212.340.vec.extract, ptr %723, align 4, !dbg !384, !tbaa !378
  %.sroa.212.344.vec.extract = extractelement <4 x float> %547, i64 2, !dbg !385
  %724 = getelementptr inbounds float, ptr %717, i64 %584, !dbg !386
  store float %.sroa.212.344.vec.extract, ptr %724, align 4, !dbg !387, !tbaa !378
  %.sroa.212.348.vec.extract = extractelement <4 x float> %547, i64 3, !dbg !388
  %725 = getelementptr inbounds float, ptr %717, i64 %586, !dbg !389
  store float %.sroa.212.348.vec.extract, ptr %725, align 4, !dbg !390, !tbaa !378
  %.sroa.222.352.vec.extract = extractelement <4 x float> %548, i64 0, !dbg !375
  %726 = getelementptr inbounds float, ptr %717, i64 %596, !dbg !376
  store float %.sroa.222.352.vec.extract, ptr %726, align 4, !dbg !377, !tbaa !378
  %.sroa.222.356.vec.extract = extractelement <4 x float> %548, i64 1, !dbg !382
  %727 = getelementptr inbounds float, ptr %717, i64 %598, !dbg !383
  store float %.sroa.222.356.vec.extract, ptr %727, align 4, !dbg !384, !tbaa !378
  %.sroa.222.360.vec.extract = extractelement <4 x float> %548, i64 2, !dbg !385
  %728 = getelementptr inbounds float, ptr %717, i64 %600, !dbg !386
  store float %.sroa.222.360.vec.extract, ptr %728, align 4, !dbg !387, !tbaa !378
  %.sroa.222.364.vec.extract = extractelement <4 x float> %548, i64 3, !dbg !388
  %729 = getelementptr inbounds float, ptr %717, i64 %602, !dbg !389
  store float %.sroa.222.364.vec.extract, ptr %729, align 4, !dbg !390, !tbaa !378
  %.sroa.232.368.vec.extract = extractelement <4 x float> %549, i64 0, !dbg !375
  %730 = getelementptr inbounds float, ptr %717, i64 %604, !dbg !376
  store float %.sroa.232.368.vec.extract, ptr %730, align 4, !dbg !377, !tbaa !378
  %.sroa.232.372.vec.extract = extractelement <4 x float> %549, i64 1, !dbg !382
  %731 = getelementptr inbounds float, ptr %717, i64 %606, !dbg !383
  store float %.sroa.232.372.vec.extract, ptr %731, align 4, !dbg !384, !tbaa !378
  %.sroa.232.376.vec.extract = extractelement <4 x float> %549, i64 2, !dbg !385
  %732 = getelementptr inbounds float, ptr %717, i64 %608, !dbg !386
  store float %.sroa.232.376.vec.extract, ptr %732, align 4, !dbg !387, !tbaa !378
  %.sroa.232.380.vec.extract = extractelement <4 x float> %549, i64 3, !dbg !388
  %733 = getelementptr inbounds float, ptr %717, i64 %610, !dbg !389
  store float %.sroa.232.380.vec.extract, ptr %733, align 4, !dbg !390, !tbaa !378
  %734 = shl i32 %561, 5, !dbg !360
  %735 = add i32 %734, 192, !dbg !360
  %736 = sext i32 %735 to i64, !dbg !369
  %737 = mul i64 %.sroa.20.0.copyload, %736, !dbg !372
  %738 = getelementptr float, ptr %.sroa.17824.0.copyload, i64 %737, !dbg !374
  %739 = getelementptr float, ptr %738, i64 %616, !dbg !374
  %.sroa.242.384.vec.extract = extractelement <4 x float> %550, i64 0, !dbg !375
  %740 = getelementptr inbounds float, ptr %739, i64 %572, !dbg !376
  store float %.sroa.242.384.vec.extract, ptr %740, align 4, !dbg !377, !tbaa !378
  %.sroa.242.388.vec.extract = extractelement <4 x float> %550, i64 1, !dbg !382
  %741 = getelementptr inbounds float, ptr %739, i64 %574, !dbg !383
  store float %.sroa.242.388.vec.extract, ptr %741, align 4, !dbg !384, !tbaa !378
  %.sroa.242.392.vec.extract = extractelement <4 x float> %550, i64 2, !dbg !385
  %742 = getelementptr inbounds float, ptr %739, i64 %576, !dbg !386
  store float %.sroa.242.392.vec.extract, ptr %742, align 4, !dbg !387, !tbaa !378
  %.sroa.242.396.vec.extract = extractelement <4 x float> %550, i64 3, !dbg !388
  %743 = getelementptr inbounds float, ptr %739, i64 %578, !dbg !389
  store float %.sroa.242.396.vec.extract, ptr %743, align 4, !dbg !390, !tbaa !378
  %.sroa.252.400.vec.extract = extractelement <4 x float> %551, i64 0, !dbg !375
  %744 = getelementptr inbounds float, ptr %739, i64 %580, !dbg !376
  store float %.sroa.252.400.vec.extract, ptr %744, align 4, !dbg !377, !tbaa !378
  %.sroa.252.404.vec.extract = extractelement <4 x float> %551, i64 1, !dbg !382
  %745 = getelementptr inbounds float, ptr %739, i64 %582, !dbg !383
  store float %.sroa.252.404.vec.extract, ptr %745, align 4, !dbg !384, !tbaa !378
  %.sroa.252.408.vec.extract = extractelement <4 x float> %551, i64 2, !dbg !385
  %746 = getelementptr inbounds float, ptr %739, i64 %584, !dbg !386
  store float %.sroa.252.408.vec.extract, ptr %746, align 4, !dbg !387, !tbaa !378
  %.sroa.252.412.vec.extract = extractelement <4 x float> %551, i64 3, !dbg !388
  %747 = getelementptr inbounds float, ptr %739, i64 %586, !dbg !389
  store float %.sroa.252.412.vec.extract, ptr %747, align 4, !dbg !390, !tbaa !378
  %.sroa.262.416.vec.extract = extractelement <4 x float> %552, i64 0, !dbg !375
  %748 = getelementptr inbounds float, ptr %739, i64 %596, !dbg !376
  store float %.sroa.262.416.vec.extract, ptr %748, align 4, !dbg !377, !tbaa !378
  %.sroa.262.420.vec.extract = extractelement <4 x float> %552, i64 1, !dbg !382
  %749 = getelementptr inbounds float, ptr %739, i64 %598, !dbg !383
  store float %.sroa.262.420.vec.extract, ptr %749, align 4, !dbg !384, !tbaa !378
  %.sroa.262.424.vec.extract = extractelement <4 x float> %552, i64 2, !dbg !385
  %750 = getelementptr inbounds float, ptr %739, i64 %600, !dbg !386
  store float %.sroa.262.424.vec.extract, ptr %750, align 4, !dbg !387, !tbaa !378
  %.sroa.262.428.vec.extract = extractelement <4 x float> %552, i64 3, !dbg !388
  %751 = getelementptr inbounds float, ptr %739, i64 %602, !dbg !389
  store float %.sroa.262.428.vec.extract, ptr %751, align 4, !dbg !390, !tbaa !378
  %.sroa.272.432.vec.extract = extractelement <4 x float> %553, i64 0, !dbg !375
  %752 = getelementptr inbounds float, ptr %739, i64 %604, !dbg !376
  store float %.sroa.272.432.vec.extract, ptr %752, align 4, !dbg !377, !tbaa !378
  %.sroa.272.436.vec.extract = extractelement <4 x float> %553, i64 1, !dbg !382
  %753 = getelementptr inbounds float, ptr %739, i64 %606, !dbg !383
  store float %.sroa.272.436.vec.extract, ptr %753, align 4, !dbg !384, !tbaa !378
  %.sroa.272.440.vec.extract = extractelement <4 x float> %553, i64 2, !dbg !385
  %754 = getelementptr inbounds float, ptr %739, i64 %608, !dbg !386
  store float %.sroa.272.440.vec.extract, ptr %754, align 4, !dbg !387, !tbaa !378
  %.sroa.272.444.vec.extract = extractelement <4 x float> %553, i64 3, !dbg !388
  %755 = getelementptr inbounds float, ptr %739, i64 %610, !dbg !389
  store float %.sroa.272.444.vec.extract, ptr %755, align 4, !dbg !390, !tbaa !378
  %756 = getelementptr float, ptr %738, i64 %638, !dbg !374
  %.sroa.282.448.vec.extract = extractelement <4 x float> %554, i64 0, !dbg !375
  %757 = getelementptr inbounds float, ptr %756, i64 %572, !dbg !376
  store float %.sroa.282.448.vec.extract, ptr %757, align 4, !dbg !377, !tbaa !378
  %.sroa.282.452.vec.extract = extractelement <4 x float> %554, i64 1, !dbg !382
  %758 = getelementptr inbounds float, ptr %756, i64 %574, !dbg !383
  store float %.sroa.282.452.vec.extract, ptr %758, align 4, !dbg !384, !tbaa !378
  %.sroa.282.456.vec.extract = extractelement <4 x float> %554, i64 2, !dbg !385
  %759 = getelementptr inbounds float, ptr %756, i64 %576, !dbg !386
  store float %.sroa.282.456.vec.extract, ptr %759, align 4, !dbg !387, !tbaa !378
  %.sroa.282.460.vec.extract = extractelement <4 x float> %554, i64 3, !dbg !388
  %760 = getelementptr inbounds float, ptr %756, i64 %578, !dbg !389
  store float %.sroa.282.460.vec.extract, ptr %760, align 4, !dbg !390, !tbaa !378
  %.sroa.292.464.vec.extract = extractelement <4 x float> %555, i64 0, !dbg !375
  %761 = getelementptr inbounds float, ptr %756, i64 %580, !dbg !376
  store float %.sroa.292.464.vec.extract, ptr %761, align 4, !dbg !377, !tbaa !378
  %.sroa.292.468.vec.extract = extractelement <4 x float> %555, i64 1, !dbg !382
  %762 = getelementptr inbounds float, ptr %756, i64 %582, !dbg !383
  store float %.sroa.292.468.vec.extract, ptr %762, align 4, !dbg !384, !tbaa !378
  %.sroa.292.472.vec.extract = extractelement <4 x float> %555, i64 2, !dbg !385
  %763 = getelementptr inbounds float, ptr %756, i64 %584, !dbg !386
  store float %.sroa.292.472.vec.extract, ptr %763, align 4, !dbg !387, !tbaa !378
  %.sroa.292.476.vec.extract = extractelement <4 x float> %555, i64 3, !dbg !388
  %764 = getelementptr inbounds float, ptr %756, i64 %586, !dbg !389
  store float %.sroa.292.476.vec.extract, ptr %764, align 4, !dbg !390, !tbaa !378
  %.sroa.302.480.vec.extract = extractelement <4 x float> %556, i64 0, !dbg !375
  %765 = getelementptr inbounds float, ptr %756, i64 %596, !dbg !376
  store float %.sroa.302.480.vec.extract, ptr %765, align 4, !dbg !377, !tbaa !378
  %.sroa.302.484.vec.extract = extractelement <4 x float> %556, i64 1, !dbg !382
  %766 = getelementptr inbounds float, ptr %756, i64 %598, !dbg !383
  store float %.sroa.302.484.vec.extract, ptr %766, align 4, !dbg !384, !tbaa !378
  %.sroa.302.488.vec.extract = extractelement <4 x float> %556, i64 2, !dbg !385
  %767 = getelementptr inbounds float, ptr %756, i64 %600, !dbg !386
  store float %.sroa.302.488.vec.extract, ptr %767, align 4, !dbg !387, !tbaa !378
  %.sroa.302.492.vec.extract = extractelement <4 x float> %556, i64 3, !dbg !388
  %768 = getelementptr inbounds float, ptr %756, i64 %602, !dbg !389
  store float %.sroa.302.492.vec.extract, ptr %768, align 4, !dbg !390, !tbaa !378
  %.sroa.312.496.vec.extract = extractelement <4 x float> %557, i64 0, !dbg !375
  %769 = getelementptr inbounds float, ptr %756, i64 %604, !dbg !376
  store float %.sroa.312.496.vec.extract, ptr %769, align 4, !dbg !377, !tbaa !378
  %.sroa.312.500.vec.extract = extractelement <4 x float> %557, i64 1, !dbg !382
  %770 = getelementptr inbounds float, ptr %756, i64 %606, !dbg !383
  store float %.sroa.312.500.vec.extract, ptr %770, align 4, !dbg !384, !tbaa !378
  %.sroa.312.504.vec.extract = extractelement <4 x float> %557, i64 2, !dbg !385
  %771 = getelementptr inbounds float, ptr %756, i64 %608, !dbg !386
  store float %.sroa.312.504.vec.extract, ptr %771, align 4, !dbg !387, !tbaa !378
  %.sroa.312.508.vec.extract = extractelement <4 x float> %557, i64 3, !dbg !388
  %772 = getelementptr inbounds float, ptr %756, i64 %610, !dbg !389
  store float %.sroa.312.508.vec.extract, ptr %772, align 4, !dbg !390, !tbaa !378
  ret void, !dbg !391
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
!1 = !DIFile(filename: "256_256_32_16.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_256_32_16")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"}
!9 = distinct !DISubprogram(name: "micro_tk", scope: !1, file: !1, line: 37, type: !10, scopeLine: 37, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !15)
!13 = distinct !DISubprogram(name: "align_ptr<16>", scope: !14, file: !14, line: 228, type: !10, scopeLine: 228, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DIFile(filename: "ThunderKittens-HIP/include/common/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!15 = distinct !DILocation(line: 252, column: 13, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 256, 32> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DILocation(line: 41, column: 39, scope: !9)
!18 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !15)
!19 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !15)
!20 = !DILocation(line: 255, column: 17, scope: !16, inlinedAt: !17)
!21 = !DILocation(line: 230, column: 30, scope: !13, inlinedAt: !22)
!22 = distinct !DILocation(line: 252, column: 13, scope: !16, inlinedAt: !23)
!23 = distinct !DILocation(line: 42, column: 39, scope: !9)
!24 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !22)
!25 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !22)
!26 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !22)
!27 = !DILocation(line: 271, column: 116, scope: !28, inlinedAt: !30)
!28 = distinct !DISubprogram(name: "__hip_get_block_idx_y", scope: !29, file: !29, line: 271, type: !10, scopeLine: 271, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!29 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h", directory: "")
!30 = distinct !DILocation(line: 301, column: 160, scope: !31, inlinedAt: !32)
!31 = distinct !DISubprogram(name: "__get_y", scope: !29, file: !29, line: 301, type: !10, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!32 = distinct !DILocation(line: 57, column: 15, scope: !9)
!33 = !DILocation(line: 280, column: 115, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "__hip_get_grid_dim_x", scope: !29, file: !29, line: 280, type: !10, scopeLine: 280, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!35 = distinct !DILocation(line: 318, column: 160, scope: !36, inlinedAt: !37)
!36 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 318, type: !10, scopeLine: 318, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!37 = distinct !DILocation(line: 57, column: 28, scope: !9)
!38 = !{!39, !39, i64 0}
!39 = !{!"int", !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C/C++ TBAA"}
!42 = !DILocation(line: 57, column: 26, scope: !9)
!43 = !DILocation(line: 270, column: 116, scope: !44, inlinedAt: !45)
!44 = distinct !DISubprogram(name: "__hip_get_block_idx_x", scope: !29, file: !29, line: 270, type: !10, scopeLine: 270, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!45 = distinct !DILocation(line: 300, column: 160, scope: !46, inlinedAt: !47)
!46 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 300, type: !10, scopeLine: 300, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!47 = distinct !DILocation(line: 57, column: 41, scope: !9)
!48 = !DILocation(line: 57, column: 39, scope: !9)
!49 = !DILocation(line: 65, column: 59, scope: !9)
!50 = !DILocation(line: 65, column: 51, scope: !9)
!51 = !DILocation(line: 73, column: 29, scope: !9)
!52 = !DILocation(line: 74, column: 36, scope: !9)
!53 = !DILocation(line: 75, column: 42, scope: !9)
!54 = !DILocation(line: 1321, column: 10, scope: !55, inlinedAt: !57)
!55 = distinct !DISubprogram(name: "min", scope: !56, file: !56, line: 1320, type: !10, scopeLine: 1320, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!56 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/lib/clang/19/include/__clang_hip_math.h", directory: "")
!57 = distinct !DILocation(line: 75, column: 28, scope: !9)
!58 = !DILocation(line: 76, column: 42, scope: !9)
!59 = !DILocation(line: 77, column: 48, scope: !9)
!60 = !DILocation(line: 76, column: 63, scope: !9)
!61 = !DILocation(line: 76, column: 33, scope: !9)
!62 = !DILocation(line: 265, column: 117, scope: !63, inlinedAt: !64)
!63 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !29, file: !29, line: 265, type: !10, scopeLine: 265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!64 = distinct !DILocation(line: 291, column: 160, scope: !65, inlinedAt: !66)
!65 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 291, type: !10, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!66 = distinct !DILocation(line: 57, column: 85, scope: !67, inlinedAt: !68)
!67 = distinct !DISubprogram(name: "warpid", scope: !14, file: !14, line: 57, type: !10, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!68 = distinct !DILocation(line: 87, column: 23, scope: !9)
!69 = !DILocation(line: 57, column: 97, scope: !67, inlinedAt: !68)
!70 = !DILocation(line: 88, column: 32, scope: !9)
!71 = !DILocation(line: 89, column: 32, scope: !9)
!72 = !DILocation(line: 30, column: 28, scope: !73, inlinedAt: !75)
!73 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 256, 32>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 32> >, 512>", scope: !74, file: !74, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!74 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!75 = distinct !DILocation(line: 12, column: 5, scope: !76, inlinedAt: !78)
!76 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 256, 32>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 32> > >", scope: !77, file: !77, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!77 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!78 = distinct !DILocation(line: 94, column: 3, scope: !9)
!79 = !DILocation(line: 60, column: 34, scope: !80, inlinedAt: !82)
!80 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !81, file: !81, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!81 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!82 = distinct !DILocation(line: 37, column: 39, scope: !73, inlinedAt: !75)
!83 = !DILocation(line: 63, column: 54, scope: !84, inlinedAt: !86)
!84 = distinct !DISubprogram(name: "operator[]", scope: !85, file: !85, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!85 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/gl.cuh", directory: "/workdir/AMD-benchmarking-harness")
!86 = distinct !DILocation(line: 38, column: 57, scope: !73, inlinedAt: !75)
!87 = !DILocation(line: 63, column: 60, scope: !84, inlinedAt: !86)
!88 = !DILocation(line: 63, column: 16, scope: !84, inlinedAt: !86)
!89 = !DILocation(line: 40, column: 24, scope: !73, inlinedAt: !75)
!90 = !DILocation(line: 54, column: 32, scope: !73, inlinedAt: !75)
!91 = !DILocation(line: 58, column: 75, scope: !73, inlinedAt: !75)
!92 = !DILocation(line: 58, column: 88, scope: !73, inlinedAt: !75)
!93 = !DILocation(line: 58, column: 68, scope: !73, inlinedAt: !75)
!94 = !DILocation(line: 92, column: 5, scope: !95, inlinedAt: !97)
!95 = distinct !DISubprogram(name: "load_global_vec4_async", scope: !96, file: !96, line: 89, type: !10, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!96 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/util/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!97 = distinct !DILocation(line: 58, column: 26, scope: !73, inlinedAt: !75)
!98 = !{i64 3065620}
!99 = !DILocation(line: 62, column: 9, scope: !73, inlinedAt: !75)
!100 = !{i64 3083803}
!101 = !DILocation(line: 103, column: 35, scope: !102, inlinedAt: !104)
!102 = distinct !DISubprogram(name: "idx", scope: !103, file: !103, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!103 = !DIFile(filename: "ThunderKittens-HIP/include/types/shared/st.cuh", directory: "/workdir/AMD-benchmarking-harness")
!104 = distinct !DILocation(line: 71, column: 34, scope: !73, inlinedAt: !75)
!105 = !DILocation(line: 104, column: 60, scope: !102, inlinedAt: !104)
!106 = !DILocation(line: 105, column: 22, scope: !102, inlinedAt: !104)
!107 = !DILocation(line: 71, column: 72, scope: !73, inlinedAt: !75)
!108 = !DILocation(line: 27, column: 5, scope: !109, inlinedAt: !110)
!109 = distinct !DISubprogram(name: "store_shared_vec", scope: !96, file: !96, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!110 = distinct !DILocation(line: 71, column: 17, scope: !73, inlinedAt: !75)
!111 = !{i64 3063955}
!112 = !DILocation(line: 103, column: 35, scope: !102, inlinedAt: !113)
!113 = distinct !DILocation(line: 72, column: 34, scope: !73, inlinedAt: !75)
!114 = !DILocation(line: 104, column: 60, scope: !102, inlinedAt: !113)
!115 = !DILocation(line: 105, column: 22, scope: !102, inlinedAt: !113)
!116 = !DILocation(line: 72, column: 95, scope: !73, inlinedAt: !75)
!117 = !DILocation(line: 27, column: 5, scope: !109, inlinedAt: !118)
!118 = distinct !DILocation(line: 72, column: 17, scope: !73, inlinedAt: !75)
!119 = !DILocation(line: 103, column: 46, scope: !102, inlinedAt: !104)
!120 = !DILocation(line: 75, column: 9, scope: !73, inlinedAt: !75)
!121 = !{i64 3084344}
!122 = !DILocation(line: 30, column: 28, scope: !73, inlinedAt: !123)
!123 = distinct !DILocation(line: 12, column: 5, scope: !76, inlinedAt: !124)
!124 = distinct !DILocation(line: 95, column: 3, scope: !9)
!125 = !DILocation(line: 60, column: 34, scope: !80, inlinedAt: !126)
!126 = distinct !DILocation(line: 37, column: 39, scope: !73, inlinedAt: !123)
!127 = !DILocation(line: 63, column: 54, scope: !84, inlinedAt: !128)
!128 = distinct !DILocation(line: 38, column: 57, scope: !73, inlinedAt: !123)
!129 = !DILocation(line: 63, column: 60, scope: !84, inlinedAt: !128)
!130 = !DILocation(line: 63, column: 16, scope: !84, inlinedAt: !128)
!131 = !DILocation(line: 40, column: 24, scope: !73, inlinedAt: !123)
!132 = !DILocation(line: 58, column: 75, scope: !73, inlinedAt: !123)
!133 = !DILocation(line: 58, column: 88, scope: !73, inlinedAt: !123)
!134 = !DILocation(line: 58, column: 68, scope: !73, inlinedAt: !123)
!135 = !DILocation(line: 92, column: 5, scope: !95, inlinedAt: !136)
!136 = distinct !DILocation(line: 58, column: 26, scope: !73, inlinedAt: !123)
!137 = !DILocation(line: 62, column: 9, scope: !73, inlinedAt: !123)
!138 = !DILocation(line: 103, column: 35, scope: !102, inlinedAt: !139)
!139 = distinct !DILocation(line: 71, column: 34, scope: !73, inlinedAt: !123)
!140 = !DILocation(line: 104, column: 60, scope: !102, inlinedAt: !139)
!141 = !DILocation(line: 105, column: 22, scope: !102, inlinedAt: !139)
!142 = !DILocation(line: 71, column: 72, scope: !73, inlinedAt: !123)
!143 = !DILocation(line: 27, column: 5, scope: !109, inlinedAt: !144)
!144 = distinct !DILocation(line: 71, column: 17, scope: !73, inlinedAt: !123)
!145 = !DILocation(line: 103, column: 35, scope: !102, inlinedAt: !146)
!146 = distinct !DILocation(line: 72, column: 34, scope: !73, inlinedAt: !123)
!147 = !DILocation(line: 104, column: 60, scope: !102, inlinedAt: !146)
!148 = !DILocation(line: 105, column: 22, scope: !102, inlinedAt: !146)
!149 = !DILocation(line: 72, column: 95, scope: !73, inlinedAt: !123)
!150 = !DILocation(line: 27, column: 5, scope: !109, inlinedAt: !151)
!151 = distinct !DILocation(line: 72, column: 17, scope: !73, inlinedAt: !123)
!152 = !DILocation(line: 75, column: 9, scope: !73, inlinedAt: !123)
!153 = !DILocation(line: 96, column: 3, scope: !9)
!154 = !DILocation(line: 98, column: 16, scope: !9)
!155 = !DILocation(line: 98, column: 7, scope: !9)
!156 = !DILocation(line: 99, column: 5, scope: !9)
!157 = !DILocation(line: 100, column: 3, scope: !9)
!158 = !DILocation(line: 102, column: 3, scope: !9)
!159 = !DILocation(line: 194, column: 16, scope: !9)
!160 = !DILocation(line: 194, column: 7, scope: !9)
!161 = !DILocation(line: 103, column: 31, scope: !9)
!162 = !DILocation(line: 103, column: 35, scope: !9)
!163 = !DILocation(line: 105, column: 9, scope: !9)
!164 = !DILocation(line: 26, column: 23, scope: !165, inlinedAt: !167)
!165 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !166, file: !166, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!166 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_256_32_16")
!167 = distinct !DILocation(line: 77, column: 39, scope: !168, inlinedAt: !169)
!168 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 32>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 32> >, 512>", scope: !166, file: !166, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!169 = distinct !DILocation(line: 106, column: 7, scope: !9)
!170 = !DILocation(line: 109, column: 5, scope: !9)
!171 = !DILocation(line: 110, column: 5, scope: !9)
!172 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !174)
!173 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !166, file: !166, line: 127, type: !10, scopeLine: 127, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!174 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !176)
!175 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 32>, 32, 16> >", scope: !166, file: !166, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!176 = distinct !DILocation(line: 112, column: 5, scope: !9)
!177 = !{i64 4294467}
!178 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !176)
!179 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !182)
!180 = distinct !DISubprogram(name: "operator=", scope: !181, file: !181, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!181 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!182 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !176)
!183 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !184)
!184 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !185)
!185 = distinct !DILocation(line: 114, column: 5, scope: !9)
!186 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !185)
!187 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !188)
!188 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !185)
!189 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !190)
!190 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !191)
!191 = distinct !DILocation(line: 117, column: 5, scope: !9)
!192 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !191)
!193 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !194)
!194 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !191)
!195 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !196)
!196 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !197)
!197 = distinct !DILocation(line: 119, column: 5, scope: !9)
!198 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !197)
!199 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !200)
!200 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !197)
!201 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !202)
!202 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !203)
!203 = distinct !DILocation(line: 122, column: 5, scope: !9)
!204 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !203)
!205 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !206)
!206 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !203)
!207 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !208)
!208 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !209)
!209 = distinct !DILocation(line: 125, column: 5, scope: !9)
!210 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !209)
!211 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !212)
!212 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !209)
!213 = !DILocation(line: 128, column: 5, scope: !9)
!214 = !DILocation(line: 130, column: 9, scope: !9)
!215 = !DILocation(line: 26, column: 23, scope: !165, inlinedAt: !216)
!216 = distinct !DILocation(line: 77, column: 39, scope: !168, inlinedAt: !217)
!217 = distinct !DILocation(line: 131, column: 7, scope: !9)
!218 = !DILocation(line: 134, column: 5, scope: !9)
!219 = !DILocation(line: 135, column: 5, scope: !9)
!220 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !221)
!221 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !222)
!222 = distinct !DILocation(line: 137, column: 5, scope: !9)
!223 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !222)
!224 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !225)
!225 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !222)
!226 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !227)
!227 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !228)
!228 = distinct !DILocation(line: 139, column: 5, scope: !9)
!229 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !228)
!230 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !231)
!231 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !228)
!232 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !233)
!233 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !234)
!234 = distinct !DILocation(line: 142, column: 5, scope: !9)
!235 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !234)
!236 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !237)
!237 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !234)
!238 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !239)
!239 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !240)
!240 = distinct !DILocation(line: 144, column: 5, scope: !9)
!241 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !240)
!242 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !243)
!243 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !240)
!244 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !245)
!245 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !246)
!246 = distinct !DILocation(line: 147, column: 5, scope: !9)
!247 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !246)
!248 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !249)
!249 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !246)
!250 = !DILocation(line: 129, column: 5, scope: !173, inlinedAt: !251)
!251 = distinct !DILocation(line: 185, column: 37, scope: !175, inlinedAt: !252)
!252 = distinct !DILocation(line: 150, column: 5, scope: !9)
!253 = !DILocation(line: 187, column: 47, scope: !175, inlinedAt: !252)
!254 = !DILocation(line: 494, column: 7, scope: !180, inlinedAt: !255)
!255 = distinct !DILocation(line: 188, column: 45, scope: !175, inlinedAt: !252)
!256 = !DILocation(line: 153, column: 5, scope: !9)
!257 = !DILocation(line: 154, column: 5, scope: !9)
!258 = !DILocation(line: 156, column: 5, scope: !9)
!259 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !262)
!260 = distinct !DISubprogram(name: "mfma161616", scope: !261, file: !261, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!261 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!262 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !264)
!263 = distinct !DISubprogram(name: "mma_ABt_base", scope: !261, file: !261, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!264 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !266)
!265 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !261, file: !261, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!266 = distinct !DILocation(line: 157, column: 5, scope: !9)
!267 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !268)
!268 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !269)
!269 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !270)
!270 = distinct !DILocation(line: 158, column: 5, scope: !9)
!271 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !272)
!272 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !273)
!273 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !274)
!274 = distinct !DILocation(line: 159, column: 5, scope: !9)
!275 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !276)
!276 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !277)
!277 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !278)
!278 = distinct !DILocation(line: 160, column: 5, scope: !9)
!279 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !280)
!280 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !281)
!281 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !282)
!282 = distinct !DILocation(line: 161, column: 5, scope: !9)
!283 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !284)
!284 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !285)
!285 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !286)
!286 = distinct !DILocation(line: 162, column: 5, scope: !9)
!287 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !288)
!288 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !289)
!289 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !290)
!290 = distinct !DILocation(line: 163, column: 5, scope: !9)
!291 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !292)
!292 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !293)
!293 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !294)
!294 = distinct !DILocation(line: 164, column: 5, scope: !9)
!295 = !DILocation(line: 165, column: 5, scope: !9)
!296 = !DILocation(line: 166, column: 5, scope: !9)
!297 = !DILocation(line: 167, column: 5, scope: !9)
!298 = !DILocation(line: 170, column: 9, scope: !9)
!299 = !DILocation(line: 171, column: 7, scope: !9)
!300 = !{i64 4305107}
!301 = !DILocation(line: 17, column: 5, scope: !302, inlinedAt: !303)
!302 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !166, file: !166, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!303 = distinct !DILocation(line: 119, column: 17, scope: !304, inlinedAt: !305)
!304 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 32>, 512>", scope: !166, file: !166, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!305 = distinct !DILocation(line: 172, column: 7, scope: !9)
!306 = !{i64 4290493}
!307 = !DILocation(line: 17, column: 5, scope: !302, inlinedAt: !308)
!308 = distinct !DILocation(line: 120, column: 17, scope: !304, inlinedAt: !305)
!309 = !DILocation(line: 17, column: 5, scope: !302, inlinedAt: !310)
!310 = distinct !DILocation(line: 119, column: 17, scope: !304, inlinedAt: !311)
!311 = distinct !DILocation(line: 174, column: 7, scope: !9)
!312 = !DILocation(line: 17, column: 5, scope: !302, inlinedAt: !313)
!313 = distinct !DILocation(line: 120, column: 17, scope: !304, inlinedAt: !311)
!314 = !DILocation(line: 176, column: 5, scope: !9)
!315 = !DILocation(line: 177, column: 5, scope: !9)
!316 = !DILocation(line: 178, column: 5, scope: !9)
!317 = !DILocation(line: 180, column: 5, scope: !9)
!318 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !319)
!319 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !320)
!320 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !321)
!321 = distinct !DILocation(line: 181, column: 5, scope: !9)
!322 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !323)
!323 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !324)
!324 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !325)
!325 = distinct !DILocation(line: 182, column: 5, scope: !9)
!326 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !327)
!327 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !328)
!328 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !329)
!329 = distinct !DILocation(line: 183, column: 5, scope: !9)
!330 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !331)
!331 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !332)
!332 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !333)
!333 = distinct !DILocation(line: 184, column: 5, scope: !9)
!334 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !335)
!335 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !336)
!336 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !337)
!337 = distinct !DILocation(line: 185, column: 5, scope: !9)
!338 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !339)
!339 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !340)
!340 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !341)
!341 = distinct !DILocation(line: 186, column: 5, scope: !9)
!342 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !343)
!343 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !344)
!344 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !345)
!345 = distinct !DILocation(line: 187, column: 5, scope: !9)
!346 = !DILocation(line: 29, column: 27, scope: !260, inlinedAt: !347)
!347 = distinct !DILocation(line: 81, column: 5, scope: !263, inlinedAt: !348)
!348 = distinct !DILocation(line: 216, column: 13, scope: !265, inlinedAt: !349)
!349 = distinct !DILocation(line: 188, column: 5, scope: !9)
!350 = !DILocation(line: 189, column: 5, scope: !9)
!351 = !DILocation(line: 190, column: 5, scope: !9)
!352 = !DILocation(line: 191, column: 5, scope: !9)
!353 = !DILocation(line: 102, column: 27, scope: !9)
!354 = distinct !{!354, !158, !355, !356}
!355 = !DILocation(line: 192, column: 3, scope: !9)
!356 = !{!"llvm.loop.mustprogress"}
!357 = !DILocation(line: 195, column: 5, scope: !9)
!358 = !DILocation(line: 196, column: 3, scope: !9)
!359 = !DILocation(line: 204, column: 31, scope: !9)
!360 = !DILocation(line: 60, column: 34, scope: !361, inlinedAt: !362)
!361 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !81, file: !81, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!362 = distinct !DILocation(line: 192, column: 41, scope: !363, inlinedAt: !365)
!363 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !364, file: !364, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!364 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!365 = distinct !DILocation(line: 212, column: 5, scope: !366, inlinedAt: !367)
!366 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !364, file: !364, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!367 = distinct !DILocation(line: 202, column: 5, scope: !9)
!368 = !DILocation(line: 61, column: 18, scope: !361, inlinedAt: !362)
!369 = !DILocation(line: 63, column: 54, scope: !370, inlinedAt: !371)
!370 = distinct !DISubprogram(name: "operator[]", scope: !85, file: !85, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!371 = distinct !DILocation(line: 192, column: 23, scope: !363, inlinedAt: !365)
!372 = !DILocation(line: 63, column: 60, scope: !370, inlinedAt: !371)
!373 = !DILocation(line: 63, column: 68, scope: !370, inlinedAt: !371)
!374 = !DILocation(line: 63, column: 16, scope: !370, inlinedAt: !371)
!375 = !DILocation(line: 203, column: 49, scope: !363, inlinedAt: !365)
!376 = !DILocation(line: 203, column: 13, scope: !363, inlinedAt: !365)
!377 = !DILocation(line: 203, column: 47, scope: !363, inlinedAt: !365)
!378 = !{!379, !379, i64 0}
!379 = !{!"float", !380, i64 0}
!380 = !{!"omnipotent char", !381, i64 0}
!381 = !{!"Simple C++ TBAA"}
!382 = !DILocation(line: 204, column: 49, scope: !363, inlinedAt: !365)
!383 = !DILocation(line: 204, column: 13, scope: !363, inlinedAt: !365)
!384 = !DILocation(line: 204, column: 47, scope: !363, inlinedAt: !365)
!385 = !DILocation(line: 205, column: 49, scope: !363, inlinedAt: !365)
!386 = !DILocation(line: 205, column: 13, scope: !363, inlinedAt: !365)
!387 = !DILocation(line: 205, column: 47, scope: !363, inlinedAt: !365)
!388 = !DILocation(line: 206, column: 49, scope: !363, inlinedAt: !365)
!389 = !DILocation(line: 206, column: 13, scope: !363, inlinedAt: !365)
!390 = !DILocation(line: 206, column: 47, scope: !363, inlinedAt: !365)
!391 = !DILocation(line: 206, column: 1, scope: !9)
