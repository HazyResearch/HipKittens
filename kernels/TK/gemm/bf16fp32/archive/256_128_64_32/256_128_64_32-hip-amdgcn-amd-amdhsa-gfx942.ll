; ModuleID = '256_128_64_32-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_128_64_32.cpp"
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
@__hip_cuid_750642e6c4d4e27 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_750642e6c4d4e27 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.0810.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.7.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.7.0.copyload = load i64, ptr addrspace(4) %.sroa.7.0..sroa_idx, align 8
  %.sroa.9815.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.9815.0.copyload = load ptr, ptr addrspace(4) %.sroa.9815.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.17819.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.17819.0.copyload = load ptr, ptr addrspace(4) %.sroa.17819.0..sroa_idx, align 8
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
  %reass.sub.i.i119 = and i64 %7, -16, !dbg !26
  %9 = add i64 %reass.sub.i.i119, 16, !dbg !26
  %10 = inttoptr i64 %9 to ptr, !dbg !26
  %.not.i.i118 = icmp eq i64 %8, 0, !dbg !27
  %11 = select i1 %.not.i.i118, ptr %6, ptr %10, !dbg !26
  %12 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y(), !dbg !28
  %13 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr(), !dbg !34
  %14 = load i32, ptr addrspace(4) %13, align 4, !dbg !34, !tbaa !39
  %15 = mul i32 %14, %12, !dbg !43
  %16 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x(), !dbg !44
  %17 = add i32 %15, %16, !dbg !49
  %18 = getelementptr inbounds i8, ptr addrspace(4) %13, i64 4, !dbg !50
  %19 = load i32, ptr addrspace(4) %18, align 4, !dbg !50, !tbaa !39
  %20 = mul i32 %19, %14, !dbg !55
  %.frozen = freeze i32 %17, !dbg !56
  %21 = sdiv i32 %.frozen, 8, !dbg !56
  %22 = mul i32 %21, 8, !dbg !57
  %.decomposed = sub i32 %.frozen, %22, !dbg !57
  %23 = sdiv i32 %20, 8, !dbg !58
  %24 = mul nsw i32 %.decomposed, %23, !dbg !59
  %25 = add nsw i32 %24, %21, !dbg !60
  %.frozen1106 = freeze i32 %25, !dbg !61
  %26 = sdiv i32 %.frozen1106, 512, !dbg !61
  %27 = shl nsw i32 %26, 4, !dbg !62
  %28 = sub nsw i32 64, %27, !dbg !63
  %29 = tail call noundef range(i32 -67108816, 67108930) i32 @llvm.smin.i32(i32 %28, i32 16), !dbg !64
  %30 = mul i32 %26, 512, !dbg !68
  %.decomposed1107 = sub i32 %.frozen1106, %30, !dbg !68
  %31 = sdiv i32 %.decomposed1107, %29, !dbg !69
  %32 = mul i32 %31, %29, !dbg !70
  %.decomposed1108 = sub i32 %.decomposed1107, %32, !dbg !70
  %33 = add nsw i32 %27, %.decomposed1108, !dbg !71
  %34 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !72
  %35 = lshr i32 %34, 6, !dbg !79
  %36 = lshr i32 %34, 8, !dbg !80
  %37 = and i32 %35, 3, !dbg !81
  %38 = trunc i64 %.sroa.7.0.copyload to i32, !dbg !82
  %39 = shl nsw i32 %33, 7, !dbg !89
  %40 = sext i32 %39 to i64, !dbg !93
  %41 = mul i64 %.sroa.7.0.copyload, %40, !dbg !97
  %42 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.0810.0.copyload, i64 %41, !dbg !98
  %43 = ptrtoint ptr %5 to i64, !dbg !99
  %44 = shl nuw nsw i32 %34, 3
  %45 = and i32 %44, 56
  %46 = lshr i32 %34, 3, !dbg !100
  %47 = and i32 %46, 63, !dbg !100
  %48 = mul i32 %47, %38, !dbg !101
  %49 = add i32 %48, %45, !dbg !102
  %50 = sext i32 %49 to i64, !dbg !103
  %51 = getelementptr inbounds %struct.__hip_bfloat16, ptr %42, i64 %50, !dbg !103
  %52 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %51) #7, !dbg !104, !srcloc !108
  %53 = or i32 %46, 64, !dbg !100
  %54 = mul i32 %53, %38, !dbg !101
  %55 = add i32 %54, %45, !dbg !102
  %56 = sext i32 %55 to i64, !dbg !103
  %57 = getelementptr inbounds %struct.__hip_bfloat16, ptr %42, i64 %56, !dbg !103
  %58 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %57) #7, !dbg !104, !srcloc !108
  %59 = trunc i64 %43 to i32, !dbg !99
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !109, !srcloc !110
  %60 = shl nuw nsw i32 %34, 4
  %.masked.i.i = and i32 %60, 8064
  %61 = shl nuw nsw i32 %45, 1
  %62 = add i32 %61, %59
  %63 = or disjoint i32 %61, 8
  %64 = add i32 %63, %59
  %65 = add i32 %62, %.masked.i.i, !dbg !111
  %66 = lshr exact i32 %65, 4, !dbg !115
  %67 = and i32 %66, 120, !dbg !115
  %68 = xor i32 %67, %65, !dbg !116
  %.sroa.0.0.extract.trunc.i.i = trunc i128 %52 to i64, !dbg !117
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %68, i64 %.sroa.0.0.extract.trunc.i.i) #7, !dbg !118, !srcloc !121
  %69 = add i32 %64, %.masked.i.i, !dbg !122
  %70 = lshr i32 %69, 4, !dbg !124
  %71 = and i32 %70, 120, !dbg !124
  %72 = xor i32 %71, %69, !dbg !125
  %.sroa.0.8.extract.shift.i.i = lshr i128 %52, 64, !dbg !126
  %.sroa.0.8.extract.trunc.i.i = trunc nuw i128 %.sroa.0.8.extract.shift.i.i to i64, !dbg !126
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %72, i64 %.sroa.0.8.extract.trunc.i.i) #7, !dbg !127, !srcloc !121
  %73 = or disjoint i32 %.masked.i.i, 8192, !dbg !129
  %74 = add i32 %62, %73, !dbg !111
  %75 = lshr exact i32 %74, 4, !dbg !115
  %76 = and i32 %75, 120, !dbg !115
  %77 = xor i32 %76, %74, !dbg !116
  %.sroa.5.16.extract.trunc.i.i = trunc i128 %58 to i64, !dbg !117
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %77, i64 %.sroa.5.16.extract.trunc.i.i) #7, !dbg !118, !srcloc !121
  %78 = add i32 %64, %73, !dbg !122
  %79 = lshr i32 %78, 4, !dbg !124
  %80 = and i32 %79, 120, !dbg !124
  %81 = xor i32 %80, %78, !dbg !125
  %.sroa.5.24.extract.shift.i.i = lshr i128 %58, 64, !dbg !126
  %.sroa.5.24.extract.trunc.i.i = trunc nuw i128 %.sroa.5.24.extract.shift.i.i to i64, !dbg !126
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %81, i64 %.sroa.5.24.extract.trunc.i.i) #7, !dbg !127, !srcloc !121
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !130, !srcloc !131
  %82 = trunc i64 %.sroa.15.0.copyload to i32, !dbg !132
  %83 = shl nsw i32 %31, 8, !dbg !137
  %84 = sext i32 %83 to i64, !dbg !140
  %85 = mul i64 %.sroa.15.0.copyload, %84, !dbg !142
  %86 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.9815.0.copyload, i64 %85, !dbg !143
  %87 = ptrtoint ptr %11 to i64, !dbg !144
  %88 = mul i32 %47, %82, !dbg !145
  %89 = add i32 %88, %45, !dbg !146
  %90 = sext i32 %89 to i64, !dbg !147
  %91 = getelementptr inbounds %struct.__hip_bfloat16, ptr %86, i64 %90, !dbg !147
  %92 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %91) #7, !dbg !148, !srcloc !108
  %93 = mul i32 %53, %82, !dbg !145
  %94 = add i32 %93, %45, !dbg !146
  %95 = sext i32 %94 to i64, !dbg !147
  %96 = getelementptr inbounds %struct.__hip_bfloat16, ptr %86, i64 %95, !dbg !147
  %97 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %96) #7, !dbg !148, !srcloc !108
  %98 = or disjoint i32 %47, 128, !dbg !150
  %99 = mul i32 %98, %82, !dbg !145
  %100 = add i32 %99, %45, !dbg !146
  %101 = sext i32 %100 to i64, !dbg !147
  %102 = getelementptr inbounds %struct.__hip_bfloat16, ptr %86, i64 %101, !dbg !147
  %103 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %102) #7, !dbg !148, !srcloc !108
  %104 = or i32 %46, 192, !dbg !150
  %105 = mul i32 %104, %82, !dbg !145
  %106 = add i32 %105, %45, !dbg !146
  %107 = sext i32 %106 to i64, !dbg !147
  %108 = getelementptr inbounds %struct.__hip_bfloat16, ptr %86, i64 %107, !dbg !147
  %109 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %108) #7, !dbg !148, !srcloc !108
  %110 = trunc i64 %87 to i32, !dbg !144
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !151, !srcloc !110
  %111 = add i32 %61, %110
  %112 = add i32 %63, %110
  %113 = add i32 %111, %.masked.i.i, !dbg !152
  %114 = lshr exact i32 %113, 4, !dbg !155
  %115 = and i32 %114, 120, !dbg !155
  %116 = xor i32 %115, %113, !dbg !156
  %.sroa.0.0.extract.trunc.i.i124 = trunc i128 %92 to i64, !dbg !157
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %116, i64 %.sroa.0.0.extract.trunc.i.i124) #7, !dbg !158, !srcloc !121
  %117 = add i32 %112, %.masked.i.i, !dbg !160
  %118 = lshr i32 %117, 4, !dbg !162
  %119 = and i32 %118, 120, !dbg !162
  %120 = xor i32 %119, %117, !dbg !163
  %.sroa.0.8.extract.shift.i.i125 = lshr i128 %92, 64, !dbg !164
  %.sroa.0.8.extract.trunc.i.i126 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i125 to i64, !dbg !164
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %120, i64 %.sroa.0.8.extract.trunc.i.i126) #7, !dbg !165, !srcloc !121
  %121 = add i32 %111, %73, !dbg !152
  %122 = lshr exact i32 %121, 4, !dbg !155
  %123 = and i32 %122, 120, !dbg !155
  %124 = xor i32 %123, %121, !dbg !156
  %.sroa.5.16.extract.trunc.i.i127 = trunc i128 %97 to i64, !dbg !157
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %124, i64 %.sroa.5.16.extract.trunc.i.i127) #7, !dbg !158, !srcloc !121
  %125 = add i32 %112, %73, !dbg !160
  %126 = lshr i32 %125, 4, !dbg !162
  %127 = and i32 %126, 120, !dbg !162
  %128 = xor i32 %127, %125, !dbg !163
  %.sroa.5.24.extract.shift.i.i128 = lshr i128 %97, 64, !dbg !164
  %.sroa.5.24.extract.trunc.i.i129 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i128 to i64, !dbg !164
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %128, i64 %.sroa.5.24.extract.trunc.i.i129) #7, !dbg !165, !srcloc !121
  %129 = or disjoint i32 %.masked.i.i, 16384, !dbg !167
  %130 = add i32 %111, %129, !dbg !152
  %131 = lshr exact i32 %130, 4, !dbg !155
  %132 = and i32 %131, 120, !dbg !155
  %133 = xor i32 %132, %130, !dbg !156
  %.sroa.8.32.extract.trunc.i.i = trunc i128 %103 to i64, !dbg !157
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %133, i64 %.sroa.8.32.extract.trunc.i.i) #7, !dbg !158, !srcloc !121
  %134 = add i32 %112, %129, !dbg !160
  %135 = lshr i32 %134, 4, !dbg !162
  %136 = and i32 %135, 120, !dbg !162
  %137 = xor i32 %136, %134, !dbg !163
  %.sroa.8.40.extract.shift.i.i = lshr i128 %103, 64, !dbg !164
  %.sroa.8.40.extract.trunc.i.i = trunc nuw i128 %.sroa.8.40.extract.shift.i.i to i64, !dbg !164
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %137, i64 %.sroa.8.40.extract.trunc.i.i) #7, !dbg !165, !srcloc !121
  %138 = or disjoint i32 %.masked.i.i, 24576, !dbg !167
  %139 = add i32 %111, %138, !dbg !152
  %140 = lshr exact i32 %139, 4, !dbg !155
  %141 = and i32 %140, 120, !dbg !155
  %142 = xor i32 %141, %139, !dbg !156
  %.sroa.11.48.extract.trunc.i.i = trunc i128 %109 to i64, !dbg !157
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %142, i64 %.sroa.11.48.extract.trunc.i.i) #7, !dbg !158, !srcloc !121
  %143 = add i32 %112, %138, !dbg !160
  %144 = lshr i32 %143, 4, !dbg !162
  %145 = and i32 %144, 120, !dbg !162
  %146 = xor i32 %145, %143, !dbg !163
  %.sroa.11.56.extract.shift.i.i = lshr i128 %109, 64, !dbg !164
  %.sroa.11.56.extract.trunc.i.i = trunc nuw i128 %.sroa.11.56.extract.shift.i.i to i64, !dbg !164
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %146, i64 %.sroa.11.56.extract.trunc.i.i) #7, !dbg !165, !srcloc !121
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !168, !srcloc !131
  tail call void @llvm.amdgcn.s.barrier(), !dbg !169
  %147 = icmp eq i32 %36, 1, !dbg !170
  br i1 %147, label %148, label %149, !dbg !171

148:                                              ; preds = %1
  tail call void @llvm.amdgcn.s.barrier(), !dbg !172
  br label %149, !dbg !173

149:                                              ; preds = %148, %1
  %150 = shl nuw nsw i32 %37, 5
  %151 = and i32 %34, 15
  %152 = lshr i32 %34, 2
  %153 = and i32 %152, 12
  %154 = or disjoint i32 %150, %151
  %155 = shl nuw nsw i32 %154, 7
  %156 = shl nuw nsw i32 %153, 1
  %157 = add i32 %155, %110
  %158 = add i32 %157, %156
  %159 = lshr i32 %158, 4
  %160 = and i32 %159, 120
  %161 = xor i32 %160, %158
  %162 = shl i32 %55, 1
  %163 = shl i32 %49, 1
  %164 = shl i32 %38, 8
  %165 = or disjoint i32 %156, 32
  %166 = add i32 %157, %165
  %167 = lshr i32 %166, 4
  %168 = and i32 %167, 120
  %169 = xor i32 %168, %166
  %170 = or disjoint i32 %151, 16
  %171 = or disjoint i32 %150, %170
  %172 = shl nuw nsw i32 %171, 7
  %173 = add i32 %172, %110
  %174 = add i32 %173, %156
  %175 = lshr i32 %174, 4
  %176 = and i32 %175, 120
  %177 = xor i32 %176, %174
  %178 = add i32 %173, %165
  %179 = lshr i32 %178, 4
  %180 = and i32 %179, 120
  %181 = xor i32 %180, %178
  %182 = or disjoint i32 %150, 128
  %183 = or disjoint i32 %182, %151
  %184 = shl nuw nsw i32 %183, 7
  %185 = add i32 %184, %110
  %186 = add i32 %185, %156
  %187 = lshr i32 %186, 4
  %188 = and i32 %187, 120
  %189 = xor i32 %188, %186
  %190 = add i32 %185, %165
  %191 = lshr i32 %190, 4
  %192 = and i32 %191, 120
  %193 = xor i32 %192, %190
  %194 = or disjoint i32 %182, %170
  %195 = shl nuw nsw i32 %194, 7
  %196 = add i32 %195, %110
  %197 = add i32 %196, %156
  %198 = lshr i32 %197, 4
  %199 = and i32 %198, 120
  %200 = xor i32 %199, %197
  %201 = add i32 %196, %165
  %202 = lshr i32 %201, 4
  %203 = and i32 %202, 120
  %204 = xor i32 %203, %201
  %205 = shl nuw nsw i32 %36, 5
  %206 = or disjoint i32 %205, %151
  %207 = shl nuw nsw i32 %206, 7
  %208 = add i32 %207, %59
  %209 = add i32 %208, %156
  %210 = lshr i32 %209, 4
  %211 = and i32 %210, 120
  %212 = xor i32 %211, %209
  %213 = add i32 %208, %165
  %214 = lshr i32 %213, 4
  %215 = and i32 %214, 120
  %216 = xor i32 %215, %213
  %217 = or disjoint i32 %170, %205
  %218 = shl nuw nsw i32 %217, 7
  %219 = add i32 %218, %59
  %220 = add i32 %219, %156
  %221 = lshr i32 %220, 4
  %222 = and i32 %221, 120
  %223 = xor i32 %222, %220
  %224 = add i32 %219, %165
  %225 = lshr i32 %224, 4
  %226 = and i32 %225, 120
  %227 = xor i32 %226, %224
  %228 = add nuw nsw i32 %205, 64
  %229 = or disjoint i32 %228, %151
  %230 = shl nuw nsw i32 %229, 7
  %231 = add i32 %230, %59
  %232 = add i32 %231, %156
  %233 = lshr i32 %232, 4
  %234 = and i32 %233, 120
  %235 = xor i32 %234, %232
  %236 = add i32 %231, %165
  %237 = lshr i32 %236, 4
  %238 = and i32 %237, 120
  %239 = xor i32 %238, %236
  %240 = or disjoint i32 %228, %170
  %241 = shl nuw nsw i32 %240, 7
  %242 = add i32 %241, %59
  %243 = add i32 %242, %156
  %244 = lshr i32 %243, 4
  %245 = and i32 %244, 120
  %246 = xor i32 %245, %243
  %247 = add i32 %242, %165
  %248 = lshr i32 %247, 4
  %249 = and i32 %248, 120
  %250 = xor i32 %249, %247
  %251 = shl i32 %82, 9
  %252 = shl i32 %89, 1
  %253 = shl i32 %94, 1
  %254 = shl i32 %100, 1
  %255 = shl i32 %106, 1
  %256 = or disjoint i32 %156, 64
  %257 = add i32 %157, %256
  %258 = lshr i32 %257, 4
  %259 = and i32 %258, 120
  %260 = xor i32 %259, %257
  %261 = or disjoint i32 %156, 96
  %262 = add i32 %157, %261
  %263 = lshr i32 %262, 4
  %264 = and i32 %263, 120
  %265 = xor i32 %264, %262
  %266 = add i32 %173, %256
  %267 = lshr i32 %266, 4
  %268 = and i32 %267, 120
  %269 = xor i32 %268, %266
  %270 = add i32 %173, %261
  %271 = lshr i32 %270, 4
  %272 = and i32 %271, 120
  %273 = xor i32 %272, %270
  %274 = add i32 %185, %256
  %275 = lshr i32 %274, 4
  %276 = and i32 %275, 120
  %277 = xor i32 %276, %274
  %278 = add i32 %185, %261
  %279 = lshr i32 %278, 4
  %280 = and i32 %279, 120
  %281 = xor i32 %280, %278
  %282 = add i32 %196, %256
  %283 = lshr i32 %282, 4
  %284 = and i32 %283, 120
  %285 = xor i32 %284, %282
  %286 = add i32 %196, %261
  %287 = lshr i32 %286, 4
  %288 = and i32 %287, 120
  %289 = xor i32 %288, %286
  %290 = add i32 %208, %256
  %291 = lshr i32 %290, 4
  %292 = and i32 %291, 120
  %293 = xor i32 %292, %290
  %294 = add i32 %208, %261
  %295 = lshr i32 %294, 4
  %296 = and i32 %295, 120
  %297 = xor i32 %296, %294
  %298 = add i32 %219, %256
  %299 = lshr i32 %298, 4
  %300 = and i32 %299, 120
  %301 = xor i32 %300, %298
  %302 = add i32 %219, %261
  %303 = lshr i32 %302, 4
  %304 = and i32 %303, 120
  %305 = xor i32 %304, %302
  %306 = add i32 %231, %256
  %307 = lshr i32 %306, 4
  %308 = and i32 %307, 120
  %309 = xor i32 %308, %306
  %310 = add i32 %231, %261
  %311 = lshr i32 %310, 4
  %312 = and i32 %311, 120
  %313 = xor i32 %312, %310
  %314 = add i32 %242, %256
  %315 = lshr i32 %314, 4
  %316 = and i32 %315, 120
  %317 = xor i32 %316, %314
  %318 = add i32 %242, %261
  %319 = lshr i32 %318, 4
  %320 = and i32 %319, 120
  %321 = xor i32 %320, %318
  %322 = and i32 %60, 112
  %323 = add i32 %322, %59
  %324 = or disjoint i32 %322, 8
  %325 = add i32 %324, %59
  %326 = add i32 %323, %.masked.i.i
  %327 = lshr exact i32 %326, 4
  %328 = and i32 %327, 120
  %329 = xor i32 %328, %326
  %330 = add i32 %325, %.masked.i.i
  %331 = lshr i32 %330, 4
  %332 = and i32 %331, 120
  %333 = xor i32 %332, %330
  %334 = add i32 %323, %73
  %335 = lshr exact i32 %334, 4
  %336 = and i32 %335, 120
  %337 = xor i32 %336, %334
  %338 = add i32 %325, %73
  %339 = lshr i32 %338, 4
  %340 = and i32 %339, 120
  %341 = xor i32 %340, %338
  %342 = add i32 %322, %110
  %343 = add i32 %324, %110
  %344 = add i32 %342, %.masked.i.i
  %345 = lshr exact i32 %344, 4
  %346 = and i32 %345, 120
  %347 = xor i32 %346, %344
  %348 = add i32 %343, %.masked.i.i
  %349 = lshr i32 %348, 4
  %350 = and i32 %349, 120
  %351 = xor i32 %350, %348
  %352 = add i32 %342, %73
  %353 = lshr exact i32 %352, 4
  %354 = and i32 %353, 120
  %355 = xor i32 %354, %352
  %356 = add i32 %343, %73
  %357 = lshr i32 %356, 4
  %358 = and i32 %357, 120
  %359 = xor i32 %358, %356
  %360 = add i32 %342, %129
  %361 = lshr exact i32 %360, 4
  %362 = and i32 %361, 120
  %363 = xor i32 %362, %360
  %364 = add i32 %343, %129
  %365 = lshr i32 %364, 4
  %366 = and i32 %365, 120
  %367 = xor i32 %366, %364
  %368 = add i32 %342, %138
  %369 = lshr exact i32 %368, 4
  %370 = and i32 %369, 120
  %371 = xor i32 %370, %368
  %372 = add i32 %343, %138
  %373 = lshr i32 %372, 4
  %374 = and i32 %373, 120
  %375 = xor i32 %374, %372
  br label %378, !dbg !174

376:                                              ; preds = %636
  %377 = icmp ult i32 %34, 256, !dbg !175
  br i1 %377, label %669, label %670, !dbg !176

378:                                              ; preds = %149, %636
  %379 = phi <4 x float> [ zeroinitializer, %149 ], [ %668, %636 ]
  %380 = phi <4 x float> [ zeroinitializer, %149 ], [ %666, %636 ]
  %381 = phi <4 x float> [ zeroinitializer, %149 ], [ %664, %636 ]
  %.val33.i455909 = phi <4 x float> [ zeroinitializer, %149 ], [ %662, %636 ]
  %382 = phi <4 x float> [ zeroinitializer, %149 ], [ %660, %636 ]
  %383 = phi <4 x float> [ zeroinitializer, %149 ], [ %658, %636 ]
  %384 = phi <4 x float> [ zeroinitializer, %149 ], [ %656, %636 ]
  %.val33.i435900 = phi <4 x float> [ zeroinitializer, %149 ], [ %654, %636 ]
  %385 = phi <4 x float> [ zeroinitializer, %149 ], [ %652, %636 ]
  %386 = phi <4 x float> [ zeroinitializer, %149 ], [ %650, %636 ]
  %387 = phi <4 x float> [ zeroinitializer, %149 ], [ %648, %636 ]
  %.val33.i415891 = phi <4 x float> [ zeroinitializer, %149 ], [ %646, %636 ]
  %.val33.1.1.i490888 = phi <4 x float> [ zeroinitializer, %149 ], [ %644, %636 ]
  %.val33.138.i485885 = phi <4 x float> [ zeroinitializer, %149 ], [ %642, %636 ]
  %.val33.1.i480882 = phi <4 x float> [ zeroinitializer, %149 ], [ %640, %636 ]
  %.0111879 = phi i32 [ 0, %149 ], [ %388, %636 ]
  %.sroa.5653.0.off64878 = phi i64 [ 0, %149 ], [ %.sroa.5653.1.off64, %636 ]
  %.sroa.5653.0.off0877 = phi i64 [ undef, %149 ], [ %.sroa.5653.1.off0, %636 ]
  %.sroa.0652.0.off64876 = phi i64 [ 0, %149 ], [ %.sroa.0652.1.off64, %636 ]
  %.sroa.0652.0.off0875 = phi i64 [ undef, %149 ], [ %.sroa.0652.1.off0, %636 ]
  %.sroa.11.0.off64874 = phi i64 [ 0, %149 ], [ %.sroa.11.1.off64, %636 ]
  %.sroa.11.0.off0873 = phi i64 [ undef, %149 ], [ %.sroa.11.1.off0, %636 ]
  %.sroa.8651.0.off64872 = phi i64 [ 0, %149 ], [ %.sroa.8651.1.off64, %636 ]
  %.sroa.8651.0.off0871 = phi i64 [ undef, %149 ], [ %.sroa.8651.1.off0, %636 ]
  %.sroa.5.0.off64870 = phi i64 [ 0, %149 ], [ %.sroa.5.1.off64, %636 ]
  %.sroa.5.0.off0869 = phi i64 [ undef, %149 ], [ %.sroa.5.1.off0, %636 ]
  %.sroa.0650.0.off64868 = phi i64 [ 0, %149 ], [ %.sroa.0650.1.off64, %636 ]
  %.sroa.0650.0.off0867 = phi i64 [ undef, %149 ], [ %.sroa.0650.1.off0, %636 ]
  %.val33.i865866 = phi <4 x float> [ zeroinitializer, %149 ], [ %638, %636 ]
  %388 = add nuw nsw i32 %.0111879, 1, !dbg !177
  %.not = icmp eq i32 %.0111879, 127, !dbg !178
  br i1 %.not, label %398, label %389, !dbg !179

389:                                              ; preds = %378
  %390 = shl nuw nsw i32 %388, 6
  %391 = zext nneg i32 %390 to i64
  %392 = getelementptr %struct.__hip_bfloat16, ptr %42, i64 %391
  %393 = ptrtoint ptr %392 to i64
  %394 = bitcast i64 %393 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %394, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %395 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %395, i32 %164, i64 2
  %396 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %163, i32 noundef 0, i32 noundef 0) #7, !dbg !180
  %397 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %162, i32 noundef 0, i32 noundef 0) #7, !dbg !180
  %extract.t850 = trunc i128 %396 to i64, !dbg !186
  %extract852 = lshr i128 %396, 64, !dbg !186
  %extract.t853 = trunc nuw i128 %extract852 to i64, !dbg !186
  %extract.t857 = trunc i128 %397 to i64, !dbg !186
  %extract859 = lshr i128 %397, 64, !dbg !186
  %extract.t860 = trunc nuw i128 %extract859 to i64, !dbg !186
  br label %398, !dbg !186

398:                                              ; preds = %389, %378
  %.sroa.0652.1.off0 = phi i64 [ %.sroa.0652.0.off0875, %378 ], [ %extract.t850, %389 ]
  %.sroa.0652.1.off64 = phi i64 [ %.sroa.0652.0.off64876, %378 ], [ %extract.t853, %389 ]
  %.sroa.5653.1.off0 = phi i64 [ %.sroa.5653.0.off0877, %378 ], [ %extract.t857, %389 ]
  %.sroa.5653.1.off64 = phi i64 [ %.sroa.5653.0.off64878, %378 ], [ %extract.t860, %389 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !187
  %399 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %161) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.i = lshr i64 %399, 16, !dbg !194
  %400 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !194
  %401 = insertelement <2 x i64> %400, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !194
  %402 = shufflevector <2 x i64> %400, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %403 = lshr <2 x i64> %402, <i64 32, i64 48>, !dbg !188
  %404 = shufflevector <2 x i64> %401, <2 x i64> %403, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.0688.6.vec.insert1075 = trunc <4 x i64> %404 to <4 x i16>, !dbg !195
  %405 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %169) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %405, 16, !dbg !194
  %406 = insertelement <2 x i64> poison, i64 %405, i64 0, !dbg !194
  %407 = insertelement <2 x i64> %406, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !194
  %408 = shufflevector <2 x i64> %406, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %409 = lshr <2 x i64> %408, <i64 32, i64 48>, !dbg !188
  %410 = shufflevector <2 x i64> %407, <2 x i64> %409, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.10693.14.vec.insert1074 = trunc <4 x i64> %410 to <4 x i16>, !dbg !195
  %411 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %177) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i = lshr i64 %411, 16, !dbg !194
  %412 = insertelement <2 x i64> poison, i64 %411, i64 0, !dbg !194
  %413 = insertelement <2 x i64> %412, i64 %.sroa.036.2.extract.shift37.145.i, i64 1, !dbg !194
  %414 = shufflevector <2 x i64> %412, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %415 = lshr <2 x i64> %414, <i64 32, i64 48>, !dbg !188
  %416 = shufflevector <2 x i64> %413, <2 x i64> %415, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.18698.22.vec.insert1073 = trunc <4 x i64> %416 to <4 x i16>, !dbg !195
  %417 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %181) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i = lshr i64 %417, 16, !dbg !194
  %418 = insertelement <2 x i64> poison, i64 %417, i64 0, !dbg !194
  %419 = insertelement <2 x i64> %418, i64 %.sroa.036.2.extract.shift37.1.1.i, i64 1, !dbg !194
  %420 = shufflevector <2 x i64> %418, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %421 = lshr <2 x i64> %420, <i64 32, i64 48>, !dbg !188
  %422 = shufflevector <2 x i64> %419, <2 x i64> %421, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.26703.30.vec.insert1072 = trunc <4 x i64> %422 to <4 x i16>, !dbg !195
  %423 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %189) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.i138 = lshr i64 %423, 16, !dbg !202
  %424 = insertelement <2 x i64> poison, i64 %423, i64 0, !dbg !202
  %425 = insertelement <2 x i64> %424, i64 %.sroa.036.2.extract.shift37.i138, i64 1, !dbg !202
  %426 = shufflevector <2 x i64> %424, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %427 = lshr <2 x i64> %426, <i64 32, i64 48>, !dbg !199
  %428 = shufflevector <2 x i64> %425, <2 x i64> %427, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.0655.6.vec.insert1071 = trunc <4 x i64> %428 to <4 x i16>, !dbg !203
  %429 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %193) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i147 = lshr i64 %429, 16, !dbg !202
  %430 = insertelement <2 x i64> poison, i64 %429, i64 0, !dbg !202
  %431 = insertelement <2 x i64> %430, i64 %.sroa.036.2.extract.shift37.1.i147, i64 1, !dbg !202
  %432 = shufflevector <2 x i64> %430, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %433 = lshr <2 x i64> %432, <i64 32, i64 48>, !dbg !199
  %434 = shufflevector <2 x i64> %431, <2 x i64> %433, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.10.14.vec.insert1070 = trunc <4 x i64> %434 to <4 x i16>, !dbg !203
  %435 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %200) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i156 = lshr i64 %435, 16, !dbg !202
  %436 = insertelement <2 x i64> poison, i64 %435, i64 0, !dbg !202
  %437 = insertelement <2 x i64> %436, i64 %.sroa.036.2.extract.shift37.145.i156, i64 1, !dbg !202
  %438 = shufflevector <2 x i64> %436, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %439 = lshr <2 x i64> %438, <i64 32, i64 48>, !dbg !199
  %440 = shufflevector <2 x i64> %437, <2 x i64> %439, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.18.22.vec.insert1069 = trunc <4 x i64> %440 to <4 x i16>, !dbg !203
  %441 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %204) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i165 = lshr i64 %441, 16, !dbg !202
  %442 = insertelement <2 x i64> poison, i64 %441, i64 0, !dbg !202
  %443 = insertelement <2 x i64> %442, i64 %.sroa.036.2.extract.shift37.1.1.i165, i64 1, !dbg !202
  %444 = shufflevector <2 x i64> %442, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %445 = lshr <2 x i64> %444, <i64 32, i64 48>, !dbg !199
  %446 = shufflevector <2 x i64> %443, <2 x i64> %445, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.26.30.vec.insert1068 = trunc <4 x i64> %446 to <4 x i16>, !dbg !203
  %447 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %212) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.i175 = lshr i64 %447, 16, !dbg !209
  %448 = insertelement <2 x i64> poison, i64 %447, i64 0, !dbg !209
  %449 = insertelement <2 x i64> %448, i64 %.sroa.036.2.extract.shift37.i175, i64 1, !dbg !209
  %450 = shufflevector <2 x i64> %448, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %451 = lshr <2 x i64> %450, <i64 32, i64 48>, !dbg !205
  %452 = shufflevector <2 x i64> %449, <2 x i64> %451, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.0768.6.vec.insert1067 = trunc <4 x i64> %452 to <4 x i16>, !dbg !210
  %453 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %216) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i182 = lshr i64 %453, 16, !dbg !209
  %454 = insertelement <2 x i64> poison, i64 %453, i64 0, !dbg !209
  %455 = insertelement <2 x i64> %454, i64 %.sroa.036.2.extract.shift37.1.i182, i64 1, !dbg !209
  %456 = shufflevector <2 x i64> %454, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %457 = lshr <2 x i64> %456, <i64 32, i64 48>, !dbg !205
  %458 = shufflevector <2 x i64> %455, <2 x i64> %457, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.10773.14.vec.insert1066 = trunc <4 x i64> %458 to <4 x i16>, !dbg !210
  %459 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %223) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i189 = lshr i64 %459, 16, !dbg !209
  %460 = insertelement <2 x i64> poison, i64 %459, i64 0, !dbg !209
  %461 = insertelement <2 x i64> %460, i64 %.sroa.036.2.extract.shift37.145.i189, i64 1, !dbg !209
  %462 = shufflevector <2 x i64> %460, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %463 = lshr <2 x i64> %462, <i64 32, i64 48>, !dbg !205
  %464 = shufflevector <2 x i64> %461, <2 x i64> %463, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.18778.22.vec.insert1065 = trunc <4 x i64> %464 to <4 x i16>, !dbg !210
  %465 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %227) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i196 = lshr i64 %465, 16, !dbg !209
  %466 = insertelement <2 x i64> poison, i64 %465, i64 0, !dbg !209
  %467 = insertelement <2 x i64> %466, i64 %.sroa.036.2.extract.shift37.1.1.i196, i64 1, !dbg !209
  %468 = shufflevector <2 x i64> %466, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %469 = lshr <2 x i64> %468, <i64 32, i64 48>, !dbg !205
  %470 = shufflevector <2 x i64> %467, <2 x i64> %469, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.26783.30.vec.insert1064 = trunc <4 x i64> %470 to <4 x i16>, !dbg !210
  %471 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %235) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.i208 = lshr i64 %471, 16, !dbg !215
  %472 = insertelement <2 x i64> poison, i64 %471, i64 0, !dbg !215
  %473 = insertelement <2 x i64> %472, i64 %.sroa.036.2.extract.shift37.i208, i64 1, !dbg !215
  %474 = shufflevector <2 x i64> %472, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %475 = lshr <2 x i64> %474, <i64 32, i64 48>, !dbg !212
  %476 = shufflevector <2 x i64> %473, <2 x i64> %475, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.0728.6.vec.insert1063 = trunc <4 x i64> %476 to <4 x i16>, !dbg !216
  %477 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %239) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i217 = lshr i64 %477, 16, !dbg !215
  %478 = insertelement <2 x i64> poison, i64 %477, i64 0, !dbg !215
  %479 = insertelement <2 x i64> %478, i64 %.sroa.036.2.extract.shift37.1.i217, i64 1, !dbg !215
  %480 = shufflevector <2 x i64> %478, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %481 = lshr <2 x i64> %480, <i64 32, i64 48>, !dbg !212
  %482 = shufflevector <2 x i64> %479, <2 x i64> %481, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.10733.14.vec.insert1062 = trunc <4 x i64> %482 to <4 x i16>, !dbg !216
  %483 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %246) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i226 = lshr i64 %483, 16, !dbg !215
  %484 = insertelement <2 x i64> poison, i64 %483, i64 0, !dbg !215
  %485 = insertelement <2 x i64> %484, i64 %.sroa.036.2.extract.shift37.145.i226, i64 1, !dbg !215
  %486 = shufflevector <2 x i64> %484, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %487 = lshr <2 x i64> %486, <i64 32, i64 48>, !dbg !212
  %488 = shufflevector <2 x i64> %485, <2 x i64> %487, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.18738.22.vec.insert1061 = trunc <4 x i64> %488 to <4 x i16>, !dbg !216
  %489 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %250) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i235 = lshr i64 %489, 16, !dbg !215
  %490 = insertelement <2 x i64> poison, i64 %489, i64 0, !dbg !215
  %491 = insertelement <2 x i64> %490, i64 %.sroa.036.2.extract.shift37.1.1.i235, i64 1, !dbg !215
  %492 = shufflevector <2 x i64> %490, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %493 = lshr <2 x i64> %492, <i64 32, i64 48>, !dbg !212
  %494 = shufflevector <2 x i64> %491, <2 x i64> %493, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.26743.30.vec.insert1060 = trunc <4 x i64> %494 to <4 x i16>, !dbg !216
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !218
  br i1 %.not, label %506, label %495, !dbg !219

495:                                              ; preds = %398
  %496 = shl nuw nsw i32 %388, 6, !dbg !220
  %497 = zext nneg i32 %496 to i64, !dbg !224
  %498 = getelementptr %struct.__hip_bfloat16, ptr %86, i64 %497, !dbg !226
  %499 = ptrtoint ptr %498 to i64, !dbg !227
  %500 = bitcast i64 %499 to <2 x i32>, !dbg !227
  %.sroa.0.0.vec.expand.i.i243 = shufflevector <2 x i32> %500, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>, !dbg !227
  %501 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i243, i32 1114112, i64 3, !dbg !227
  %.sroa.0.12.vec.insert.i.i244 = insertelement <4 x i32> %501, i32 %251, i64 2, !dbg !227
  %502 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %252, i32 noundef 0, i32 noundef 0) #7, !dbg !230
  %503 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %253, i32 noundef 0, i32 noundef 0) #7, !dbg !230
  %504 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %254, i32 noundef 0, i32 noundef 0) #7, !dbg !230
  %505 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %255, i32 noundef 0, i32 noundef 0) #7, !dbg !230
  %extract.t822 = trunc i128 %502 to i64, !dbg !232
  %extract824 = lshr i128 %502, 64, !dbg !232
  %extract.t825 = trunc nuw i128 %extract824 to i64, !dbg !232
  %extract.t829 = trunc i128 %503 to i64, !dbg !232
  %extract831 = lshr i128 %503, 64, !dbg !232
  %extract.t832 = trunc nuw i128 %extract831 to i64, !dbg !232
  %extract.t836 = trunc i128 %504 to i64, !dbg !232
  %extract838 = lshr i128 %504, 64, !dbg !232
  %extract.t839 = trunc nuw i128 %extract838 to i64, !dbg !232
  %extract.t843 = trunc i128 %505 to i64, !dbg !232
  %extract845 = lshr i128 %505, 64, !dbg !232
  %extract.t846 = trunc nuw i128 %extract845 to i64, !dbg !232
  br label %506, !dbg !232

506:                                              ; preds = %495, %398
  %.sroa.0650.1.off0 = phi i64 [ %.sroa.0650.0.off0867, %398 ], [ %extract.t822, %495 ]
  %.sroa.0650.1.off64 = phi i64 [ %.sroa.0650.0.off64868, %398 ], [ %extract.t825, %495 ]
  %.sroa.5.1.off0 = phi i64 [ %.sroa.5.0.off0869, %398 ], [ %extract.t829, %495 ]
  %.sroa.5.1.off64 = phi i64 [ %.sroa.5.0.off64870, %398 ], [ %extract.t832, %495 ]
  %.sroa.8651.1.off0 = phi i64 [ %.sroa.8651.0.off0871, %398 ], [ %extract.t836, %495 ]
  %.sroa.8651.1.off64 = phi i64 [ %.sroa.8651.0.off64872, %398 ], [ %extract.t839, %495 ]
  %.sroa.11.1.off0 = phi i64 [ %.sroa.11.0.off0873, %398 ], [ %extract.t843, %495 ]
  %.sroa.11.1.off64 = phi i64 [ %.sroa.11.0.off64874, %398 ], [ %extract.t846, %495 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !233
  %507 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %260) #7, !dbg !234, !srcloc !193
  %.sroa.036.2.extract.shift37.i252 = lshr i64 %507, 16, !dbg !237
  %508 = insertelement <2 x i64> poison, i64 %507, i64 0, !dbg !237
  %509 = insertelement <2 x i64> %508, i64 %.sroa.036.2.extract.shift37.i252, i64 1, !dbg !237
  %510 = shufflevector <2 x i64> %508, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !234
  %511 = lshr <2 x i64> %510, <i64 32, i64 48>, !dbg !234
  %512 = shufflevector <2 x i64> %509, <2 x i64> %511, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !238
  %.sroa.34708.38.vec.insert1059 = trunc <4 x i64> %512 to <4 x i16>, !dbg !238
  %513 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %265) #7, !dbg !234, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i261 = lshr i64 %513, 16, !dbg !237
  %514 = insertelement <2 x i64> poison, i64 %513, i64 0, !dbg !237
  %515 = insertelement <2 x i64> %514, i64 %.sroa.036.2.extract.shift37.1.i261, i64 1, !dbg !237
  %516 = shufflevector <2 x i64> %514, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !234
  %517 = lshr <2 x i64> %516, <i64 32, i64 48>, !dbg !234
  %518 = shufflevector <2 x i64> %515, <2 x i64> %517, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !238
  %.sroa.42713.46.vec.insert1058 = trunc <4 x i64> %518 to <4 x i16>, !dbg !238
  %519 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %269) #7, !dbg !234, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i270 = lshr i64 %519, 16, !dbg !237
  %520 = insertelement <2 x i64> poison, i64 %519, i64 0, !dbg !237
  %521 = insertelement <2 x i64> %520, i64 %.sroa.036.2.extract.shift37.145.i270, i64 1, !dbg !237
  %522 = shufflevector <2 x i64> %520, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !234
  %523 = lshr <2 x i64> %522, <i64 32, i64 48>, !dbg !234
  %524 = shufflevector <2 x i64> %521, <2 x i64> %523, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !238
  %.sroa.50718.54.vec.insert1057 = trunc <4 x i64> %524 to <4 x i16>, !dbg !238
  %525 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %273) #7, !dbg !234, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i279 = lshr i64 %525, 16, !dbg !237
  %526 = insertelement <2 x i64> poison, i64 %525, i64 0, !dbg !237
  %527 = insertelement <2 x i64> %526, i64 %.sroa.036.2.extract.shift37.1.1.i279, i64 1, !dbg !237
  %528 = shufflevector <2 x i64> %526, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !234
  %529 = lshr <2 x i64> %528, <i64 32, i64 48>, !dbg !234
  %530 = shufflevector <2 x i64> %527, <2 x i64> %529, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !238
  %.sroa.58723.62.vec.insert1056 = trunc <4 x i64> %530 to <4 x i16>, !dbg !238
  %531 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %277) #7, !dbg !240, !srcloc !193
  %.sroa.036.2.extract.shift37.i291 = lshr i64 %531, 16, !dbg !243
  %532 = insertelement <2 x i64> poison, i64 %531, i64 0, !dbg !243
  %533 = insertelement <2 x i64> %532, i64 %.sroa.036.2.extract.shift37.i291, i64 1, !dbg !243
  %534 = shufflevector <2 x i64> %532, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !240
  %535 = lshr <2 x i64> %534, <i64 32, i64 48>, !dbg !240
  %536 = shufflevector <2 x i64> %533, <2 x i64> %535, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !244
  %.sroa.34.38.vec.insert1055 = trunc <4 x i64> %536 to <4 x i16>, !dbg !244
  %537 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %281) #7, !dbg !240, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i300 = lshr i64 %537, 16, !dbg !243
  %538 = insertelement <2 x i64> poison, i64 %537, i64 0, !dbg !243
  %539 = insertelement <2 x i64> %538, i64 %.sroa.036.2.extract.shift37.1.i300, i64 1, !dbg !243
  %540 = shufflevector <2 x i64> %538, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !240
  %541 = lshr <2 x i64> %540, <i64 32, i64 48>, !dbg !240
  %542 = shufflevector <2 x i64> %539, <2 x i64> %541, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !244
  %.sroa.42.46.vec.insert1054 = trunc <4 x i64> %542 to <4 x i16>, !dbg !244
  %543 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %285) #7, !dbg !240, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i309 = lshr i64 %543, 16, !dbg !243
  %544 = insertelement <2 x i64> poison, i64 %543, i64 0, !dbg !243
  %545 = insertelement <2 x i64> %544, i64 %.sroa.036.2.extract.shift37.145.i309, i64 1, !dbg !243
  %546 = shufflevector <2 x i64> %544, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !240
  %547 = lshr <2 x i64> %546, <i64 32, i64 48>, !dbg !240
  %548 = shufflevector <2 x i64> %545, <2 x i64> %547, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !244
  %.sroa.50.54.vec.insert1053 = trunc <4 x i64> %548 to <4 x i16>, !dbg !244
  %549 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %289) #7, !dbg !240, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i318 = lshr i64 %549, 16, !dbg !243
  %550 = insertelement <2 x i64> poison, i64 %549, i64 0, !dbg !243
  %551 = insertelement <2 x i64> %550, i64 %.sroa.036.2.extract.shift37.1.1.i318, i64 1, !dbg !243
  %552 = shufflevector <2 x i64> %550, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !240
  %553 = lshr <2 x i64> %552, <i64 32, i64 48>, !dbg !240
  %554 = shufflevector <2 x i64> %551, <2 x i64> %553, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !244
  %.sroa.58.62.vec.insert1052 = trunc <4 x i64> %554 to <4 x i16>, !dbg !244
  %555 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %293) #7, !dbg !246, !srcloc !193
  %.sroa.036.2.extract.shift37.i330 = lshr i64 %555, 16, !dbg !249
  %556 = insertelement <2 x i64> poison, i64 %555, i64 0, !dbg !249
  %557 = insertelement <2 x i64> %556, i64 %.sroa.036.2.extract.shift37.i330, i64 1, !dbg !249
  %558 = shufflevector <2 x i64> %556, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !246
  %559 = lshr <2 x i64> %558, <i64 32, i64 48>, !dbg !246
  %560 = shufflevector <2 x i64> %557, <2 x i64> %559, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !250
  %.sroa.34788.38.vec.insert1051 = trunc <4 x i64> %560 to <4 x i16>, !dbg !250
  %561 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %297) #7, !dbg !246, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i339 = lshr i64 %561, 16, !dbg !249
  %562 = insertelement <2 x i64> poison, i64 %561, i64 0, !dbg !249
  %563 = insertelement <2 x i64> %562, i64 %.sroa.036.2.extract.shift37.1.i339, i64 1, !dbg !249
  %564 = shufflevector <2 x i64> %562, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !246
  %565 = lshr <2 x i64> %564, <i64 32, i64 48>, !dbg !246
  %566 = shufflevector <2 x i64> %563, <2 x i64> %565, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !250
  %.sroa.42793.46.vec.insert1050 = trunc <4 x i64> %566 to <4 x i16>, !dbg !250
  %567 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %301) #7, !dbg !246, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i348 = lshr i64 %567, 16, !dbg !249
  %568 = insertelement <2 x i64> poison, i64 %567, i64 0, !dbg !249
  %569 = insertelement <2 x i64> %568, i64 %.sroa.036.2.extract.shift37.145.i348, i64 1, !dbg !249
  %570 = shufflevector <2 x i64> %568, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !246
  %571 = lshr <2 x i64> %570, <i64 32, i64 48>, !dbg !246
  %572 = shufflevector <2 x i64> %569, <2 x i64> %571, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !250
  %.sroa.50798.54.vec.insert1049 = trunc <4 x i64> %572 to <4 x i16>, !dbg !250
  %573 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %305) #7, !dbg !246, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i357 = lshr i64 %573, 16, !dbg !249
  %574 = insertelement <2 x i64> poison, i64 %573, i64 0, !dbg !249
  %575 = insertelement <2 x i64> %574, i64 %.sroa.036.2.extract.shift37.1.1.i357, i64 1, !dbg !249
  %576 = shufflevector <2 x i64> %574, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !246
  %577 = lshr <2 x i64> %576, <i64 32, i64 48>, !dbg !246
  %578 = shufflevector <2 x i64> %575, <2 x i64> %577, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !250
  %.sroa.58803.62.vec.insert1048 = trunc <4 x i64> %578 to <4 x i16>, !dbg !250
  %579 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %309) #7, !dbg !252, !srcloc !193
  %.sroa.036.2.extract.shift37.i369 = lshr i64 %579, 16, !dbg !255
  %580 = insertelement <2 x i64> poison, i64 %579, i64 0, !dbg !255
  %581 = insertelement <2 x i64> %580, i64 %.sroa.036.2.extract.shift37.i369, i64 1, !dbg !255
  %582 = shufflevector <2 x i64> %580, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !252
  %583 = lshr <2 x i64> %582, <i64 32, i64 48>, !dbg !252
  %584 = shufflevector <2 x i64> %581, <2 x i64> %583, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !256
  %.sroa.34748.38.vec.insert1047 = trunc <4 x i64> %584 to <4 x i16>, !dbg !256
  %585 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %313) #7, !dbg !252, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i378 = lshr i64 %585, 16, !dbg !255
  %586 = insertelement <2 x i64> poison, i64 %585, i64 0, !dbg !255
  %587 = insertelement <2 x i64> %586, i64 %.sroa.036.2.extract.shift37.1.i378, i64 1, !dbg !255
  %588 = shufflevector <2 x i64> %586, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !252
  %589 = lshr <2 x i64> %588, <i64 32, i64 48>, !dbg !252
  %590 = shufflevector <2 x i64> %587, <2 x i64> %589, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !256
  %.sroa.42753.46.vec.insert1046 = trunc <4 x i64> %590 to <4 x i16>, !dbg !256
  %591 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %317) #7, !dbg !252, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i387 = lshr i64 %591, 16, !dbg !255
  %592 = insertelement <2 x i64> poison, i64 %591, i64 0, !dbg !255
  %593 = insertelement <2 x i64> %592, i64 %.sroa.036.2.extract.shift37.145.i387, i64 1, !dbg !255
  %594 = shufflevector <2 x i64> %592, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !252
  %595 = lshr <2 x i64> %594, <i64 32, i64 48>, !dbg !252
  %596 = shufflevector <2 x i64> %593, <2 x i64> %595, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !256
  %.sroa.50758.54.vec.insert1045 = trunc <4 x i64> %596 to <4 x i16>, !dbg !256
  %597 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %321) #7, !dbg !252, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i396 = lshr i64 %597, 16, !dbg !255
  %598 = insertelement <2 x i64> poison, i64 %597, i64 0, !dbg !255
  %599 = insertelement <2 x i64> %598, i64 %.sroa.036.2.extract.shift37.1.1.i396, i64 1, !dbg !255
  %600 = shufflevector <2 x i64> %598, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !252
  %601 = lshr <2 x i64> %600, <i64 32, i64 48>, !dbg !252
  %602 = shufflevector <2 x i64> %599, <2 x i64> %601, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !256
  %.sroa.58763.62.vec.insert1044 = trunc <4 x i64> %602 to <4 x i16>, !dbg !256
  tail call void @llvm.amdgcn.s.barrier(), !dbg !258
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !259
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !260
  %603 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1067, <4 x i16> %.sroa.0688.6.vec.insert1075, <4 x float> %.val33.i865866, i32 0, i32 0, i32 0), !dbg !261
  %604 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1066, <4 x i16> %.sroa.10693.14.vec.insert1074, <4 x float> %603, i32 0, i32 0, i32 0), !dbg !269
  %605 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1067, <4 x i16> %.sroa.18698.22.vec.insert1073, <4 x float> %.val33.1.i480882, i32 0, i32 0, i32 0), !dbg !261
  %606 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1066, <4 x i16> %.sroa.26703.30.vec.insert1072, <4 x float> %605, i32 0, i32 0, i32 0), !dbg !269
  %607 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1065, <4 x i16> %.sroa.0688.6.vec.insert1075, <4 x float> %.val33.138.i485885, i32 0, i32 0, i32 0), !dbg !261
  %608 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1064, <4 x i16> %.sroa.10693.14.vec.insert1074, <4 x float> %607, i32 0, i32 0, i32 0), !dbg !269
  %609 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1065, <4 x i16> %.sroa.18698.22.vec.insert1073, <4 x float> %.val33.1.1.i490888, i32 0, i32 0, i32 0), !dbg !261
  %610 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1064, <4 x i16> %.sroa.26703.30.vec.insert1072, <4 x float> %609, i32 0, i32 0, i32 0), !dbg !269
  %611 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1067, <4 x i16> %.sroa.0655.6.vec.insert1071, <4 x float> %.val33.i415891, i32 0, i32 0, i32 0), !dbg !272
  %612 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1066, <4 x i16> %.sroa.10.14.vec.insert1070, <4 x float> %611, i32 0, i32 0, i32 0), !dbg !276
  %613 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1067, <4 x i16> %.sroa.18.22.vec.insert1069, <4 x float> %387, i32 0, i32 0, i32 0), !dbg !272
  %614 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1066, <4 x i16> %.sroa.26.30.vec.insert1068, <4 x float> %613, i32 0, i32 0, i32 0), !dbg !276
  %615 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1065, <4 x i16> %.sroa.0655.6.vec.insert1071, <4 x float> %386, i32 0, i32 0, i32 0), !dbg !272
  %616 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1064, <4 x i16> %.sroa.10.14.vec.insert1070, <4 x float> %615, i32 0, i32 0, i32 0), !dbg !276
  %617 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1065, <4 x i16> %.sroa.18.22.vec.insert1069, <4 x float> %385, i32 0, i32 0, i32 0), !dbg !272
  %618 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1064, <4 x i16> %.sroa.26.30.vec.insert1068, <4 x float> %617, i32 0, i32 0, i32 0), !dbg !276
  %619 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1063, <4 x i16> %.sroa.0688.6.vec.insert1075, <4 x float> %.val33.i435900, i32 0, i32 0, i32 0), !dbg !279
  %620 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1062, <4 x i16> %.sroa.10693.14.vec.insert1074, <4 x float> %619, i32 0, i32 0, i32 0), !dbg !283
  %621 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1063, <4 x i16> %.sroa.18698.22.vec.insert1073, <4 x float> %384, i32 0, i32 0, i32 0), !dbg !279
  %622 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1062, <4 x i16> %.sroa.26703.30.vec.insert1072, <4 x float> %621, i32 0, i32 0, i32 0), !dbg !283
  %623 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1061, <4 x i16> %.sroa.0688.6.vec.insert1075, <4 x float> %383, i32 0, i32 0, i32 0), !dbg !279
  %624 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1060, <4 x i16> %.sroa.10693.14.vec.insert1074, <4 x float> %623, i32 0, i32 0, i32 0), !dbg !283
  %625 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1061, <4 x i16> %.sroa.18698.22.vec.insert1073, <4 x float> %382, i32 0, i32 0, i32 0), !dbg !279
  %626 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1060, <4 x i16> %.sroa.26703.30.vec.insert1072, <4 x float> %625, i32 0, i32 0, i32 0), !dbg !283
  %627 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1063, <4 x i16> %.sroa.0655.6.vec.insert1071, <4 x float> %.val33.i455909, i32 0, i32 0, i32 0), !dbg !286
  %628 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1062, <4 x i16> %.sroa.10.14.vec.insert1070, <4 x float> %627, i32 0, i32 0, i32 0), !dbg !290
  %629 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1063, <4 x i16> %.sroa.18.22.vec.insert1069, <4 x float> %381, i32 0, i32 0, i32 0), !dbg !286
  %630 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1062, <4 x i16> %.sroa.26.30.vec.insert1068, <4 x float> %629, i32 0, i32 0, i32 0), !dbg !290
  %631 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1061, <4 x i16> %.sroa.0655.6.vec.insert1071, <4 x float> %380, i32 0, i32 0, i32 0), !dbg !286
  %632 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1060, <4 x i16> %.sroa.10.14.vec.insert1070, <4 x float> %631, i32 0, i32 0, i32 0), !dbg !290
  %633 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1061, <4 x i16> %.sroa.18.22.vec.insert1069, <4 x float> %379, i32 0, i32 0, i32 0), !dbg !286
  %634 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1060, <4 x i16> %.sroa.26.30.vec.insert1068, <4 x float> %633, i32 0, i32 0, i32 0), !dbg !290
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !293
  tail call void @llvm.amdgcn.s.barrier(), !dbg !294
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !295
  br i1 %.not, label %636, label %635, !dbg !296

635:                                              ; preds = %506
  tail call void asm sideeffect "s_waitcnt vmcnt(2)", ""() #7, !dbg !297, !srcloc !298
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %329, i64 %.sroa.0652.1.off0) #7, !dbg !299, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %333, i64 %.sroa.0652.1.off64) #7, !dbg !305, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %337, i64 %.sroa.5653.1.off0) #7, !dbg !299, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %341, i64 %.sroa.5653.1.off64) #7, !dbg !305, !srcloc !304
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !307, !srcloc !308
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %347, i64 %.sroa.0650.1.off0) #7, !dbg !309, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %351, i64 %.sroa.0650.1.off64) #7, !dbg !313, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %355, i64 %.sroa.5.1.off0) #7, !dbg !309, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %359, i64 %.sroa.5.1.off64) #7, !dbg !313, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %363, i64 %.sroa.8651.1.off0) #7, !dbg !309, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %367, i64 %.sroa.8651.1.off64) #7, !dbg !313, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %371, i64 %.sroa.11.1.off0) #7, !dbg !309, !srcloc !304
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %375, i64 %.sroa.11.1.off64) #7, !dbg !313, !srcloc !304
  br label %636, !dbg !315

636:                                              ; preds = %635, %506
  tail call void @llvm.amdgcn.s.barrier(), !dbg !316
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !317
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !318
  %637 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1051, <4 x i16> %.sroa.34708.38.vec.insert1059, <4 x float> %604, i32 0, i32 0, i32 0), !dbg !319
  %638 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1050, <4 x i16> %.sroa.42713.46.vec.insert1058, <4 x float> %637, i32 0, i32 0, i32 0), !dbg !323
  %639 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1051, <4 x i16> %.sroa.50718.54.vec.insert1057, <4 x float> %606, i32 0, i32 0, i32 0), !dbg !319
  %640 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1050, <4 x i16> %.sroa.58723.62.vec.insert1056, <4 x float> %639, i32 0, i32 0, i32 0), !dbg !323
  %641 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1049, <4 x i16> %.sroa.34708.38.vec.insert1059, <4 x float> %608, i32 0, i32 0, i32 0), !dbg !319
  %642 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1048, <4 x i16> %.sroa.42713.46.vec.insert1058, <4 x float> %641, i32 0, i32 0, i32 0), !dbg !323
  %643 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1049, <4 x i16> %.sroa.50718.54.vec.insert1057, <4 x float> %610, i32 0, i32 0, i32 0), !dbg !319
  %644 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1048, <4 x i16> %.sroa.58723.62.vec.insert1056, <4 x float> %643, i32 0, i32 0, i32 0), !dbg !323
  %645 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1051, <4 x i16> %.sroa.34.38.vec.insert1055, <4 x float> %612, i32 0, i32 0, i32 0), !dbg !326
  %646 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1050, <4 x i16> %.sroa.42.46.vec.insert1054, <4 x float> %645, i32 0, i32 0, i32 0), !dbg !330
  %647 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1051, <4 x i16> %.sroa.50.54.vec.insert1053, <4 x float> %614, i32 0, i32 0, i32 0), !dbg !326
  %648 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1050, <4 x i16> %.sroa.58.62.vec.insert1052, <4 x float> %647, i32 0, i32 0, i32 0), !dbg !330
  %649 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1049, <4 x i16> %.sroa.34.38.vec.insert1055, <4 x float> %616, i32 0, i32 0, i32 0), !dbg !326
  %650 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1048, <4 x i16> %.sroa.42.46.vec.insert1054, <4 x float> %649, i32 0, i32 0, i32 0), !dbg !330
  %651 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1049, <4 x i16> %.sroa.50.54.vec.insert1053, <4 x float> %618, i32 0, i32 0, i32 0), !dbg !326
  %652 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1048, <4 x i16> %.sroa.58.62.vec.insert1052, <4 x float> %651, i32 0, i32 0, i32 0), !dbg !330
  %653 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1047, <4 x i16> %.sroa.34708.38.vec.insert1059, <4 x float> %620, i32 0, i32 0, i32 0), !dbg !333
  %654 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1046, <4 x i16> %.sroa.42713.46.vec.insert1058, <4 x float> %653, i32 0, i32 0, i32 0), !dbg !337
  %655 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1047, <4 x i16> %.sroa.50718.54.vec.insert1057, <4 x float> %622, i32 0, i32 0, i32 0), !dbg !333
  %656 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1046, <4 x i16> %.sroa.58723.62.vec.insert1056, <4 x float> %655, i32 0, i32 0, i32 0), !dbg !337
  %657 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1045, <4 x i16> %.sroa.34708.38.vec.insert1059, <4 x float> %624, i32 0, i32 0, i32 0), !dbg !333
  %658 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1044, <4 x i16> %.sroa.42713.46.vec.insert1058, <4 x float> %657, i32 0, i32 0, i32 0), !dbg !337
  %659 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1045, <4 x i16> %.sroa.50718.54.vec.insert1057, <4 x float> %626, i32 0, i32 0, i32 0), !dbg !333
  %660 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1044, <4 x i16> %.sroa.58723.62.vec.insert1056, <4 x float> %659, i32 0, i32 0, i32 0), !dbg !337
  %661 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1047, <4 x i16> %.sroa.34.38.vec.insert1055, <4 x float> %628, i32 0, i32 0, i32 0), !dbg !340
  %662 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1046, <4 x i16> %.sroa.42.46.vec.insert1054, <4 x float> %661, i32 0, i32 0, i32 0), !dbg !344
  %663 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1047, <4 x i16> %.sroa.50.54.vec.insert1053, <4 x float> %630, i32 0, i32 0, i32 0), !dbg !340
  %664 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1046, <4 x i16> %.sroa.58.62.vec.insert1052, <4 x float> %663, i32 0, i32 0, i32 0), !dbg !344
  %665 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1045, <4 x i16> %.sroa.34.38.vec.insert1055, <4 x float> %632, i32 0, i32 0, i32 0), !dbg !340
  %666 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1044, <4 x i16> %.sroa.42.46.vec.insert1054, <4 x float> %665, i32 0, i32 0, i32 0), !dbg !344
  %667 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1045, <4 x i16> %.sroa.50.54.vec.insert1053, <4 x float> %634, i32 0, i32 0, i32 0), !dbg !340
  %668 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1044, <4 x i16> %.sroa.58.62.vec.insert1052, <4 x float> %667, i32 0, i32 0, i32 0), !dbg !344
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !347
  tail call void @llvm.amdgcn.s.barrier(), !dbg !348
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !349
  %exitcond.not = icmp eq i32 %388, 128, !dbg !350
  br i1 %exitcond.not, label %376, label %378, !dbg !174, !llvm.loop !351

669:                                              ; preds = %376
  tail call void @llvm.amdgcn.s.barrier(), !dbg !354
  br label %670, !dbg !355

670:                                              ; preds = %669, %376
  %671 = shl nsw i32 %33, 2
  %672 = or disjoint i32 %671, %36
  %673 = shl nsw i32 %31, 3
  %674 = trunc i64 %.sroa.20.0.copyload to i32
  %675 = mul nsw i32 %153, %674
  %676 = or disjoint i32 %153, 1
  %677 = mul nsw i32 %676, %674
  %678 = or disjoint i32 %153, 2
  %679 = mul nsw i32 %678, %674
  %680 = or disjoint i32 %153, 3
  %681 = mul nsw i32 %680, %674
  %682 = add nsw i32 %675, %151
  %683 = sext i32 %682 to i64
  %684 = add nsw i32 %677, %151
  %685 = sext i32 %684 to i64
  %686 = add nsw i32 %679, %151
  %687 = sext i32 %686 to i64
  %688 = add nsw i32 %681, %151
  %689 = sext i32 %688 to i64
  %690 = add nsw i32 %675, %170
  %691 = sext i32 %690 to i64
  %692 = add nsw i32 %677, %170
  %693 = sext i32 %692 to i64
  %694 = add nsw i32 %679, %170
  %695 = sext i32 %694 to i64
  %696 = add nsw i32 %681, %170
  %697 = sext i32 %696 to i64
  %698 = or disjoint i32 %153, 16
  %699 = mul nsw i32 %698, %674
  %700 = or disjoint i32 %153, 17
  %701 = mul nsw i32 %700, %674
  %702 = or disjoint i32 %153, 18
  %703 = mul nsw i32 %702, %674
  %704 = or disjoint i32 %153, 19
  %705 = mul nsw i32 %704, %674
  %706 = add nsw i32 %699, %151
  %707 = sext i32 %706 to i64
  %708 = add nsw i32 %701, %151
  %709 = sext i32 %708 to i64
  %710 = add nsw i32 %703, %151
  %711 = sext i32 %710 to i64
  %712 = add nsw i32 %705, %151
  %713 = sext i32 %712 to i64
  %714 = add nsw i32 %699, %170
  %715 = sext i32 %714 to i64
  %716 = add nsw i32 %701, %170
  %717 = sext i32 %716 to i64
  %718 = add nsw i32 %703, %170
  %719 = sext i32 %718 to i64
  %720 = add nsw i32 %705, %170
  %721 = sext i32 %720 to i64
  %722 = or disjoint i32 %673, %37, !dbg !356
  %723 = shl nsw i32 %672, 5, !dbg !357
  %724 = shl nsw i32 %722, 5, !dbg !365
  %725 = sext i32 %723 to i64, !dbg !366
  %726 = mul i64 %.sroa.20.0.copyload, %725, !dbg !369
  %727 = sext i32 %724 to i64, !dbg !370
  %728 = getelementptr float, ptr %.sroa.17819.0.copyload, i64 %726, !dbg !371
  %729 = getelementptr float, ptr %728, i64 %727, !dbg !371
  %.sroa.0.0.vec.extract = extractelement <4 x float> %638, i64 0, !dbg !372
  %730 = getelementptr inbounds float, ptr %729, i64 %683, !dbg !373
  store float %.sroa.0.0.vec.extract, ptr %730, align 4, !dbg !374, !tbaa !375
  %.sroa.0.4.vec.extract = extractelement <4 x float> %638, i64 1, !dbg !379
  %731 = getelementptr inbounds float, ptr %729, i64 %685, !dbg !380
  store float %.sroa.0.4.vec.extract, ptr %731, align 4, !dbg !381, !tbaa !375
  %.sroa.0.8.vec.extract = extractelement <4 x float> %638, i64 2, !dbg !382
  %732 = getelementptr inbounds float, ptr %729, i64 %687, !dbg !383
  store float %.sroa.0.8.vec.extract, ptr %732, align 4, !dbg !384, !tbaa !375
  %.sroa.0.12.vec.extract = extractelement <4 x float> %638, i64 3, !dbg !385
  %733 = getelementptr inbounds float, ptr %729, i64 %689, !dbg !386
  store float %.sroa.0.12.vec.extract, ptr %733, align 4, !dbg !387, !tbaa !375
  %.sroa.12.16.vec.extract = extractelement <4 x float> %640, i64 0, !dbg !372
  %734 = getelementptr inbounds float, ptr %729, i64 %691, !dbg !373
  store float %.sroa.12.16.vec.extract, ptr %734, align 4, !dbg !374, !tbaa !375
  %.sroa.12.20.vec.extract = extractelement <4 x float> %640, i64 1, !dbg !379
  %735 = getelementptr inbounds float, ptr %729, i64 %693, !dbg !380
  store float %.sroa.12.20.vec.extract, ptr %735, align 4, !dbg !381, !tbaa !375
  %.sroa.12.24.vec.extract = extractelement <4 x float> %640, i64 2, !dbg !382
  %736 = getelementptr inbounds float, ptr %729, i64 %695, !dbg !383
  store float %.sroa.12.24.vec.extract, ptr %736, align 4, !dbg !384, !tbaa !375
  %.sroa.12.28.vec.extract = extractelement <4 x float> %640, i64 3, !dbg !385
  %737 = getelementptr inbounds float, ptr %729, i64 %697, !dbg !386
  store float %.sroa.12.28.vec.extract, ptr %737, align 4, !dbg !387, !tbaa !375
  %.sroa.22.32.vec.extract = extractelement <4 x float> %642, i64 0, !dbg !372
  %738 = getelementptr inbounds float, ptr %729, i64 %707, !dbg !373
  store float %.sroa.22.32.vec.extract, ptr %738, align 4, !dbg !374, !tbaa !375
  %.sroa.22.36.vec.extract = extractelement <4 x float> %642, i64 1, !dbg !379
  %739 = getelementptr inbounds float, ptr %729, i64 %709, !dbg !380
  store float %.sroa.22.36.vec.extract, ptr %739, align 4, !dbg !381, !tbaa !375
  %.sroa.22.40.vec.extract = extractelement <4 x float> %642, i64 2, !dbg !382
  %740 = getelementptr inbounds float, ptr %729, i64 %711, !dbg !383
  store float %.sroa.22.40.vec.extract, ptr %740, align 4, !dbg !384, !tbaa !375
  %.sroa.22.44.vec.extract = extractelement <4 x float> %642, i64 3, !dbg !385
  %741 = getelementptr inbounds float, ptr %729, i64 %713, !dbg !386
  store float %.sroa.22.44.vec.extract, ptr %741, align 4, !dbg !387, !tbaa !375
  %.sroa.32.48.vec.extract = extractelement <4 x float> %644, i64 0, !dbg !372
  %742 = getelementptr inbounds float, ptr %729, i64 %715, !dbg !373
  store float %.sroa.32.48.vec.extract, ptr %742, align 4, !dbg !374, !tbaa !375
  %.sroa.32.52.vec.extract = extractelement <4 x float> %644, i64 1, !dbg !379
  %743 = getelementptr inbounds float, ptr %729, i64 %717, !dbg !380
  store float %.sroa.32.52.vec.extract, ptr %743, align 4, !dbg !381, !tbaa !375
  %.sroa.32.56.vec.extract = extractelement <4 x float> %644, i64 2, !dbg !382
  %744 = getelementptr inbounds float, ptr %729, i64 %719, !dbg !383
  store float %.sroa.32.56.vec.extract, ptr %744, align 4, !dbg !384, !tbaa !375
  %.sroa.32.60.vec.extract = extractelement <4 x float> %644, i64 3, !dbg !385
  %745 = getelementptr inbounds float, ptr %729, i64 %721, !dbg !386
  store float %.sroa.32.60.vec.extract, ptr %745, align 4, !dbg !387, !tbaa !375
  %746 = or disjoint i32 %673, %37, !dbg !356
  %747 = shl i32 %746, 5, !dbg !365
  %748 = or disjoint i32 %747, 128, !dbg !365
  %749 = sext i32 %748 to i64, !dbg !370
  %750 = getelementptr float, ptr %728, i64 %749, !dbg !371
  %.sroa.42.64.vec.extract = extractelement <4 x float> %646, i64 0, !dbg !372
  %751 = getelementptr inbounds float, ptr %750, i64 %683, !dbg !373
  store float %.sroa.42.64.vec.extract, ptr %751, align 4, !dbg !374, !tbaa !375
  %.sroa.42.68.vec.extract = extractelement <4 x float> %646, i64 1, !dbg !379
  %752 = getelementptr inbounds float, ptr %750, i64 %685, !dbg !380
  store float %.sroa.42.68.vec.extract, ptr %752, align 4, !dbg !381, !tbaa !375
  %.sroa.42.72.vec.extract = extractelement <4 x float> %646, i64 2, !dbg !382
  %753 = getelementptr inbounds float, ptr %750, i64 %687, !dbg !383
  store float %.sroa.42.72.vec.extract, ptr %753, align 4, !dbg !384, !tbaa !375
  %.sroa.42.76.vec.extract = extractelement <4 x float> %646, i64 3, !dbg !385
  %754 = getelementptr inbounds float, ptr %750, i64 %689, !dbg !386
  store float %.sroa.42.76.vec.extract, ptr %754, align 4, !dbg !387, !tbaa !375
  %.sroa.52.80.vec.extract = extractelement <4 x float> %648, i64 0, !dbg !372
  %755 = getelementptr inbounds float, ptr %750, i64 %691, !dbg !373
  store float %.sroa.52.80.vec.extract, ptr %755, align 4, !dbg !374, !tbaa !375
  %.sroa.52.84.vec.extract = extractelement <4 x float> %648, i64 1, !dbg !379
  %756 = getelementptr inbounds float, ptr %750, i64 %693, !dbg !380
  store float %.sroa.52.84.vec.extract, ptr %756, align 4, !dbg !381, !tbaa !375
  %.sroa.52.88.vec.extract = extractelement <4 x float> %648, i64 2, !dbg !382
  %757 = getelementptr inbounds float, ptr %750, i64 %695, !dbg !383
  store float %.sroa.52.88.vec.extract, ptr %757, align 4, !dbg !384, !tbaa !375
  %.sroa.52.92.vec.extract = extractelement <4 x float> %648, i64 3, !dbg !385
  %758 = getelementptr inbounds float, ptr %750, i64 %697, !dbg !386
  store float %.sroa.52.92.vec.extract, ptr %758, align 4, !dbg !387, !tbaa !375
  %.sroa.62.96.vec.extract = extractelement <4 x float> %650, i64 0, !dbg !372
  %759 = getelementptr inbounds float, ptr %750, i64 %707, !dbg !373
  store float %.sroa.62.96.vec.extract, ptr %759, align 4, !dbg !374, !tbaa !375
  %.sroa.62.100.vec.extract = extractelement <4 x float> %650, i64 1, !dbg !379
  %760 = getelementptr inbounds float, ptr %750, i64 %709, !dbg !380
  store float %.sroa.62.100.vec.extract, ptr %760, align 4, !dbg !381, !tbaa !375
  %.sroa.62.104.vec.extract = extractelement <4 x float> %650, i64 2, !dbg !382
  %761 = getelementptr inbounds float, ptr %750, i64 %711, !dbg !383
  store float %.sroa.62.104.vec.extract, ptr %761, align 4, !dbg !384, !tbaa !375
  %.sroa.62.108.vec.extract = extractelement <4 x float> %650, i64 3, !dbg !385
  %762 = getelementptr inbounds float, ptr %750, i64 %713, !dbg !386
  store float %.sroa.62.108.vec.extract, ptr %762, align 4, !dbg !387, !tbaa !375
  %.sroa.72.112.vec.extract = extractelement <4 x float> %652, i64 0, !dbg !372
  %763 = getelementptr inbounds float, ptr %750, i64 %715, !dbg !373
  store float %.sroa.72.112.vec.extract, ptr %763, align 4, !dbg !374, !tbaa !375
  %.sroa.72.116.vec.extract = extractelement <4 x float> %652, i64 1, !dbg !379
  %764 = getelementptr inbounds float, ptr %750, i64 %717, !dbg !380
  store float %.sroa.72.116.vec.extract, ptr %764, align 4, !dbg !381, !tbaa !375
  %.sroa.72.120.vec.extract = extractelement <4 x float> %652, i64 2, !dbg !382
  %765 = getelementptr inbounds float, ptr %750, i64 %719, !dbg !383
  store float %.sroa.72.120.vec.extract, ptr %765, align 4, !dbg !384, !tbaa !375
  %.sroa.72.124.vec.extract = extractelement <4 x float> %652, i64 3, !dbg !385
  %766 = getelementptr inbounds float, ptr %750, i64 %721, !dbg !386
  store float %.sroa.72.124.vec.extract, ptr %766, align 4, !dbg !387, !tbaa !375
  %767 = shl i32 %672, 5, !dbg !357
  %768 = add i32 %767, 64, !dbg !357
  %769 = sext i32 %768 to i64, !dbg !366
  %770 = mul i64 %.sroa.20.0.copyload, %769, !dbg !369
  %771 = getelementptr float, ptr %.sroa.17819.0.copyload, i64 %770, !dbg !371
  %772 = getelementptr float, ptr %771, i64 %727, !dbg !371
  %.sroa.82.128.vec.extract = extractelement <4 x float> %654, i64 0, !dbg !372
  %773 = getelementptr inbounds float, ptr %772, i64 %683, !dbg !373
  store float %.sroa.82.128.vec.extract, ptr %773, align 4, !dbg !374, !tbaa !375
  %.sroa.82.132.vec.extract = extractelement <4 x float> %654, i64 1, !dbg !379
  %774 = getelementptr inbounds float, ptr %772, i64 %685, !dbg !380
  store float %.sroa.82.132.vec.extract, ptr %774, align 4, !dbg !381, !tbaa !375
  %.sroa.82.136.vec.extract = extractelement <4 x float> %654, i64 2, !dbg !382
  %775 = getelementptr inbounds float, ptr %772, i64 %687, !dbg !383
  store float %.sroa.82.136.vec.extract, ptr %775, align 4, !dbg !384, !tbaa !375
  %.sroa.82.140.vec.extract = extractelement <4 x float> %654, i64 3, !dbg !385
  %776 = getelementptr inbounds float, ptr %772, i64 %689, !dbg !386
  store float %.sroa.82.140.vec.extract, ptr %776, align 4, !dbg !387, !tbaa !375
  %.sroa.92.144.vec.extract = extractelement <4 x float> %656, i64 0, !dbg !372
  %777 = getelementptr inbounds float, ptr %772, i64 %691, !dbg !373
  store float %.sroa.92.144.vec.extract, ptr %777, align 4, !dbg !374, !tbaa !375
  %.sroa.92.148.vec.extract = extractelement <4 x float> %656, i64 1, !dbg !379
  %778 = getelementptr inbounds float, ptr %772, i64 %693, !dbg !380
  store float %.sroa.92.148.vec.extract, ptr %778, align 4, !dbg !381, !tbaa !375
  %.sroa.92.152.vec.extract = extractelement <4 x float> %656, i64 2, !dbg !382
  %779 = getelementptr inbounds float, ptr %772, i64 %695, !dbg !383
  store float %.sroa.92.152.vec.extract, ptr %779, align 4, !dbg !384, !tbaa !375
  %.sroa.92.156.vec.extract = extractelement <4 x float> %656, i64 3, !dbg !385
  %780 = getelementptr inbounds float, ptr %772, i64 %697, !dbg !386
  store float %.sroa.92.156.vec.extract, ptr %780, align 4, !dbg !387, !tbaa !375
  %.sroa.102.160.vec.extract = extractelement <4 x float> %658, i64 0, !dbg !372
  %781 = getelementptr inbounds float, ptr %772, i64 %707, !dbg !373
  store float %.sroa.102.160.vec.extract, ptr %781, align 4, !dbg !374, !tbaa !375
  %.sroa.102.164.vec.extract = extractelement <4 x float> %658, i64 1, !dbg !379
  %782 = getelementptr inbounds float, ptr %772, i64 %709, !dbg !380
  store float %.sroa.102.164.vec.extract, ptr %782, align 4, !dbg !381, !tbaa !375
  %.sroa.102.168.vec.extract = extractelement <4 x float> %658, i64 2, !dbg !382
  %783 = getelementptr inbounds float, ptr %772, i64 %711, !dbg !383
  store float %.sroa.102.168.vec.extract, ptr %783, align 4, !dbg !384, !tbaa !375
  %.sroa.102.172.vec.extract = extractelement <4 x float> %658, i64 3, !dbg !385
  %784 = getelementptr inbounds float, ptr %772, i64 %713, !dbg !386
  store float %.sroa.102.172.vec.extract, ptr %784, align 4, !dbg !387, !tbaa !375
  %.sroa.112.176.vec.extract = extractelement <4 x float> %660, i64 0, !dbg !372
  %785 = getelementptr inbounds float, ptr %772, i64 %715, !dbg !373
  store float %.sroa.112.176.vec.extract, ptr %785, align 4, !dbg !374, !tbaa !375
  %.sroa.112.180.vec.extract = extractelement <4 x float> %660, i64 1, !dbg !379
  %786 = getelementptr inbounds float, ptr %772, i64 %717, !dbg !380
  store float %.sroa.112.180.vec.extract, ptr %786, align 4, !dbg !381, !tbaa !375
  %.sroa.112.184.vec.extract = extractelement <4 x float> %660, i64 2, !dbg !382
  %787 = getelementptr inbounds float, ptr %772, i64 %719, !dbg !383
  store float %.sroa.112.184.vec.extract, ptr %787, align 4, !dbg !384, !tbaa !375
  %.sroa.112.188.vec.extract = extractelement <4 x float> %660, i64 3, !dbg !385
  %788 = getelementptr inbounds float, ptr %772, i64 %721, !dbg !386
  store float %.sroa.112.188.vec.extract, ptr %788, align 4, !dbg !387, !tbaa !375
  %789 = getelementptr float, ptr %771, i64 %749, !dbg !371
  %.sroa.122.192.vec.extract = extractelement <4 x float> %662, i64 0, !dbg !372
  %790 = getelementptr inbounds float, ptr %789, i64 %683, !dbg !373
  store float %.sroa.122.192.vec.extract, ptr %790, align 4, !dbg !374, !tbaa !375
  %.sroa.122.196.vec.extract = extractelement <4 x float> %662, i64 1, !dbg !379
  %791 = getelementptr inbounds float, ptr %789, i64 %685, !dbg !380
  store float %.sroa.122.196.vec.extract, ptr %791, align 4, !dbg !381, !tbaa !375
  %.sroa.122.200.vec.extract = extractelement <4 x float> %662, i64 2, !dbg !382
  %792 = getelementptr inbounds float, ptr %789, i64 %687, !dbg !383
  store float %.sroa.122.200.vec.extract, ptr %792, align 4, !dbg !384, !tbaa !375
  %.sroa.122.204.vec.extract = extractelement <4 x float> %662, i64 3, !dbg !385
  %793 = getelementptr inbounds float, ptr %789, i64 %689, !dbg !386
  store float %.sroa.122.204.vec.extract, ptr %793, align 4, !dbg !387, !tbaa !375
  %.sroa.132.208.vec.extract = extractelement <4 x float> %664, i64 0, !dbg !372
  %794 = getelementptr inbounds float, ptr %789, i64 %691, !dbg !373
  store float %.sroa.132.208.vec.extract, ptr %794, align 4, !dbg !374, !tbaa !375
  %.sroa.132.212.vec.extract = extractelement <4 x float> %664, i64 1, !dbg !379
  %795 = getelementptr inbounds float, ptr %789, i64 %693, !dbg !380
  store float %.sroa.132.212.vec.extract, ptr %795, align 4, !dbg !381, !tbaa !375
  %.sroa.132.216.vec.extract = extractelement <4 x float> %664, i64 2, !dbg !382
  %796 = getelementptr inbounds float, ptr %789, i64 %695, !dbg !383
  store float %.sroa.132.216.vec.extract, ptr %796, align 4, !dbg !384, !tbaa !375
  %.sroa.132.220.vec.extract = extractelement <4 x float> %664, i64 3, !dbg !385
  %797 = getelementptr inbounds float, ptr %789, i64 %697, !dbg !386
  store float %.sroa.132.220.vec.extract, ptr %797, align 4, !dbg !387, !tbaa !375
  %.sroa.142.224.vec.extract = extractelement <4 x float> %666, i64 0, !dbg !372
  %798 = getelementptr inbounds float, ptr %789, i64 %707, !dbg !373
  store float %.sroa.142.224.vec.extract, ptr %798, align 4, !dbg !374, !tbaa !375
  %.sroa.142.228.vec.extract = extractelement <4 x float> %666, i64 1, !dbg !379
  %799 = getelementptr inbounds float, ptr %789, i64 %709, !dbg !380
  store float %.sroa.142.228.vec.extract, ptr %799, align 4, !dbg !381, !tbaa !375
  %.sroa.142.232.vec.extract = extractelement <4 x float> %666, i64 2, !dbg !382
  %800 = getelementptr inbounds float, ptr %789, i64 %711, !dbg !383
  store float %.sroa.142.232.vec.extract, ptr %800, align 4, !dbg !384, !tbaa !375
  %.sroa.142.236.vec.extract = extractelement <4 x float> %666, i64 3, !dbg !385
  %801 = getelementptr inbounds float, ptr %789, i64 %713, !dbg !386
  store float %.sroa.142.236.vec.extract, ptr %801, align 4, !dbg !387, !tbaa !375
  %.sroa.152.240.vec.extract = extractelement <4 x float> %668, i64 0, !dbg !372
  %802 = getelementptr inbounds float, ptr %789, i64 %715, !dbg !373
  store float %.sroa.152.240.vec.extract, ptr %802, align 4, !dbg !374, !tbaa !375
  %.sroa.152.244.vec.extract = extractelement <4 x float> %668, i64 1, !dbg !379
  %803 = getelementptr inbounds float, ptr %789, i64 %717, !dbg !380
  store float %.sroa.152.244.vec.extract, ptr %803, align 4, !dbg !381, !tbaa !375
  %.sroa.152.248.vec.extract = extractelement <4 x float> %668, i64 2, !dbg !382
  %804 = getelementptr inbounds float, ptr %789, i64 %719, !dbg !383
  store float %.sroa.152.248.vec.extract, ptr %804, align 4, !dbg !384, !tbaa !375
  %.sroa.152.252.vec.extract = extractelement <4 x float> %668, i64 3, !dbg !385
  %805 = getelementptr inbounds float, ptr %789, i64 %721, !dbg !386
  store float %.sroa.152.252.vec.extract, ptr %805, align 4, !dbg !387, !tbaa !375
  ret void, !dbg !388
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
!1 = !DIFile(filename: "256_128_64_32.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32")
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
!22 = distinct !DILocation(line: 252, column: 13, scope: !23, inlinedAt: !24)
!23 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 256, 64> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = distinct !DILocation(line: 43, column: 44, scope: !9)
!25 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !22)
!26 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !22)
!27 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !22)
!28 = !DILocation(line: 271, column: 116, scope: !29, inlinedAt: !31)
!29 = distinct !DISubprogram(name: "__hip_get_block_idx_y", scope: !30, file: !30, line: 271, type: !10, scopeLine: 271, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!30 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h", directory: "")
!31 = distinct !DILocation(line: 301, column: 160, scope: !32, inlinedAt: !33)
!32 = distinct !DISubprogram(name: "__get_y", scope: !30, file: !30, line: 301, type: !10, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!33 = distinct !DILocation(line: 56, column: 17, scope: !9)
!34 = !DILocation(line: 280, column: 115, scope: !35, inlinedAt: !36)
!35 = distinct !DISubprogram(name: "__hip_get_grid_dim_x", scope: !30, file: !30, line: 280, type: !10, scopeLine: 280, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!36 = distinct !DILocation(line: 318, column: 160, scope: !37, inlinedAt: !38)
!37 = distinct !DISubprogram(name: "__get_x", scope: !30, file: !30, line: 318, type: !10, scopeLine: 318, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!38 = distinct !DILocation(line: 56, column: 30, scope: !9)
!39 = !{!40, !40, i64 0}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C/C++ TBAA"}
!43 = !DILocation(line: 56, column: 28, scope: !9)
!44 = !DILocation(line: 270, column: 116, scope: !45, inlinedAt: !46)
!45 = distinct !DISubprogram(name: "__hip_get_block_idx_x", scope: !30, file: !30, line: 270, type: !10, scopeLine: 270, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!46 = distinct !DILocation(line: 300, column: 160, scope: !47, inlinedAt: !48)
!47 = distinct !DISubprogram(name: "__get_x", scope: !30, file: !30, line: 300, type: !10, scopeLine: 300, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!48 = distinct !DILocation(line: 56, column: 43, scope: !9)
!49 = !DILocation(line: 56, column: 41, scope: !9)
!50 = !DILocation(line: 281, column: 115, scope: !51, inlinedAt: !52)
!51 = distinct !DISubprogram(name: "__hip_get_grid_dim_y", scope: !30, file: !30, line: 281, type: !10, scopeLine: 281, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!52 = distinct !DILocation(line: 319, column: 160, scope: !53, inlinedAt: !54)
!53 = distinct !DISubprogram(name: "__get_y", scope: !30, file: !30, line: 319, type: !10, scopeLine: 319, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!54 = distinct !DILocation(line: 58, column: 37, scope: !9)
!55 = !DILocation(line: 58, column: 35, scope: !9)
!56 = !DILocation(line: 64, column: 61, scope: !9)
!57 = !DILocation(line: 64, column: 18, scope: !9)
!58 = !DILocation(line: 64, column: 41, scope: !9)
!59 = !DILocation(line: 64, column: 30, scope: !9)
!60 = !DILocation(line: 64, column: 53, scope: !9)
!61 = !DILocation(line: 72, column: 25, scope: !9)
!62 = !DILocation(line: 73, column: 32, scope: !9)
!63 = !DILocation(line: 74, column: 38, scope: !9)
!64 = !DILocation(line: 1321, column: 10, scope: !65, inlinedAt: !67)
!65 = distinct !DISubprogram(name: "min", scope: !66, file: !66, line: 1320, type: !10, scopeLine: 1320, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!66 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/lib/clang/19/include/__clang_hip_math.h", directory: "")
!67 = distinct !DILocation(line: 74, column: 24, scope: !9)
!68 = !DILocation(line: 75, column: 38, scope: !9)
!69 = !DILocation(line: 76, column: 44, scope: !9)
!70 = !DILocation(line: 75, column: 59, scope: !9)
!71 = !DILocation(line: 75, column: 29, scope: !9)
!72 = !DILocation(line: 265, column: 117, scope: !73, inlinedAt: !74)
!73 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !30, file: !30, line: 265, type: !10, scopeLine: 265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!74 = distinct !DILocation(line: 291, column: 160, scope: !75, inlinedAt: !76)
!75 = distinct !DISubprogram(name: "__get_x", scope: !30, file: !30, line: 291, type: !10, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!76 = distinct !DILocation(line: 57, column: 85, scope: !77, inlinedAt: !78)
!77 = distinct !DISubprogram(name: "warpid", scope: !14, file: !14, line: 57, type: !10, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!78 = distinct !DILocation(line: 82, column: 25, scope: !9)
!79 = !DILocation(line: 57, column: 97, scope: !77, inlinedAt: !78)
!80 = !DILocation(line: 83, column: 34, scope: !9)
!81 = !DILocation(line: 84, column: 34, scope: !9)
!82 = !DILocation(line: 30, column: 28, scope: !83, inlinedAt: !85)
!83 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !84, file: !84, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!84 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!85 = distinct !DILocation(line: 12, column: 5, scope: !86, inlinedAt: !88)
!86 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> > >", scope: !87, file: !87, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!87 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!88 = distinct !DILocation(line: 89, column: 5, scope: !9)
!89 = !DILocation(line: 60, column: 34, scope: !90, inlinedAt: !92)
!90 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!91 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!92 = distinct !DILocation(line: 37, column: 39, scope: !83, inlinedAt: !85)
!93 = !DILocation(line: 63, column: 54, scope: !94, inlinedAt: !96)
!94 = distinct !DISubprogram(name: "operator[]", scope: !95, file: !95, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!95 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/gl.cuh", directory: "/workdir/AMD-benchmarking-harness")
!96 = distinct !DILocation(line: 38, column: 57, scope: !83, inlinedAt: !85)
!97 = !DILocation(line: 63, column: 60, scope: !94, inlinedAt: !96)
!98 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !96)
!99 = !DILocation(line: 40, column: 24, scope: !83, inlinedAt: !85)
!100 = !DILocation(line: 54, column: 32, scope: !83, inlinedAt: !85)
!101 = !DILocation(line: 58, column: 75, scope: !83, inlinedAt: !85)
!102 = !DILocation(line: 58, column: 88, scope: !83, inlinedAt: !85)
!103 = !DILocation(line: 58, column: 68, scope: !83, inlinedAt: !85)
!104 = !DILocation(line: 92, column: 5, scope: !105, inlinedAt: !107)
!105 = distinct !DISubprogram(name: "load_global_vec4_async", scope: !106, file: !106, line: 89, type: !10, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!106 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/util/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!107 = distinct !DILocation(line: 58, column: 26, scope: !83, inlinedAt: !85)
!108 = !{i64 3065620}
!109 = !DILocation(line: 62, column: 9, scope: !83, inlinedAt: !85)
!110 = !{i64 3083803}
!111 = !DILocation(line: 103, column: 35, scope: !112, inlinedAt: !114)
!112 = distinct !DISubprogram(name: "idx", scope: !113, file: !113, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!113 = !DIFile(filename: "ThunderKittens-HIP/include/types/shared/st.cuh", directory: "/workdir/AMD-benchmarking-harness")
!114 = distinct !DILocation(line: 71, column: 34, scope: !83, inlinedAt: !85)
!115 = !DILocation(line: 104, column: 60, scope: !112, inlinedAt: !114)
!116 = !DILocation(line: 105, column: 22, scope: !112, inlinedAt: !114)
!117 = !DILocation(line: 71, column: 72, scope: !83, inlinedAt: !85)
!118 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !120)
!119 = distinct !DISubprogram(name: "store_shared_vec", scope: !106, file: !106, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!120 = distinct !DILocation(line: 71, column: 17, scope: !83, inlinedAt: !85)
!121 = !{i64 3063955}
!122 = !DILocation(line: 103, column: 35, scope: !112, inlinedAt: !123)
!123 = distinct !DILocation(line: 72, column: 34, scope: !83, inlinedAt: !85)
!124 = !DILocation(line: 104, column: 60, scope: !112, inlinedAt: !123)
!125 = !DILocation(line: 105, column: 22, scope: !112, inlinedAt: !123)
!126 = !DILocation(line: 72, column: 95, scope: !83, inlinedAt: !85)
!127 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !128)
!128 = distinct !DILocation(line: 72, column: 17, scope: !83, inlinedAt: !85)
!129 = !DILocation(line: 103, column: 46, scope: !112, inlinedAt: !114)
!130 = !DILocation(line: 75, column: 9, scope: !83, inlinedAt: !85)
!131 = !{i64 3084344}
!132 = !DILocation(line: 30, column: 28, scope: !133, inlinedAt: !134)
!133 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !84, file: !84, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!134 = distinct !DILocation(line: 12, column: 5, scope: !135, inlinedAt: !136)
!135 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> > >", scope: !87, file: !87, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!136 = distinct !DILocation(line: 90, column: 5, scope: !9)
!137 = !DILocation(line: 60, column: 34, scope: !138, inlinedAt: !139)
!138 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!139 = distinct !DILocation(line: 37, column: 39, scope: !133, inlinedAt: !134)
!140 = !DILocation(line: 63, column: 54, scope: !94, inlinedAt: !141)
!141 = distinct !DILocation(line: 38, column: 57, scope: !133, inlinedAt: !134)
!142 = !DILocation(line: 63, column: 60, scope: !94, inlinedAt: !141)
!143 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !141)
!144 = !DILocation(line: 40, column: 24, scope: !133, inlinedAt: !134)
!145 = !DILocation(line: 58, column: 75, scope: !133, inlinedAt: !134)
!146 = !DILocation(line: 58, column: 88, scope: !133, inlinedAt: !134)
!147 = !DILocation(line: 58, column: 68, scope: !133, inlinedAt: !134)
!148 = !DILocation(line: 92, column: 5, scope: !105, inlinedAt: !149)
!149 = distinct !DILocation(line: 58, column: 26, scope: !133, inlinedAt: !134)
!150 = !DILocation(line: 54, column: 32, scope: !133, inlinedAt: !134)
!151 = !DILocation(line: 62, column: 9, scope: !133, inlinedAt: !134)
!152 = !DILocation(line: 103, column: 35, scope: !153, inlinedAt: !154)
!153 = distinct !DISubprogram(name: "idx", scope: !113, file: !113, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!154 = distinct !DILocation(line: 71, column: 34, scope: !133, inlinedAt: !134)
!155 = !DILocation(line: 104, column: 60, scope: !153, inlinedAt: !154)
!156 = !DILocation(line: 105, column: 22, scope: !153, inlinedAt: !154)
!157 = !DILocation(line: 71, column: 72, scope: !133, inlinedAt: !134)
!158 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !159)
!159 = distinct !DILocation(line: 71, column: 17, scope: !133, inlinedAt: !134)
!160 = !DILocation(line: 103, column: 35, scope: !153, inlinedAt: !161)
!161 = distinct !DILocation(line: 72, column: 34, scope: !133, inlinedAt: !134)
!162 = !DILocation(line: 104, column: 60, scope: !153, inlinedAt: !161)
!163 = !DILocation(line: 105, column: 22, scope: !153, inlinedAt: !161)
!164 = !DILocation(line: 72, column: 95, scope: !133, inlinedAt: !134)
!165 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !166)
!166 = distinct !DILocation(line: 72, column: 17, scope: !133, inlinedAt: !134)
!167 = !DILocation(line: 103, column: 46, scope: !153, inlinedAt: !154)
!168 = !DILocation(line: 75, column: 9, scope: !133, inlinedAt: !134)
!169 = !DILocation(line: 91, column: 5, scope: !9)
!170 = !DILocation(line: 93, column: 18, scope: !9)
!171 = !DILocation(line: 93, column: 9, scope: !9)
!172 = !DILocation(line: 94, column: 9, scope: !9)
!173 = !DILocation(line: 95, column: 5, scope: !9)
!174 = !DILocation(line: 97, column: 5, scope: !9)
!175 = !DILocation(line: 156, column: 18, scope: !9)
!176 = !DILocation(line: 156, column: 9, scope: !9)
!177 = !DILocation(line: 98, column: 35, scope: !9)
!178 = !DILocation(line: 98, column: 39, scope: !9)
!179 = !DILocation(line: 100, column: 13, scope: !9)
!180 = !DILocation(line: 26, column: 23, scope: !181, inlinedAt: !183)
!181 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !182, file: !182, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!182 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32")
!183 = distinct !DILocation(line: 77, column: 39, scope: !184, inlinedAt: !185)
!184 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !182, file: !182, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!185 = distinct !DILocation(line: 101, column: 13, scope: !9)
!186 = !DILocation(line: 103, column: 9, scope: !9)
!187 = !DILocation(line: 104, column: 9, scope: !9)
!188 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !190)
!189 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !182, file: !182, line: 127, type: !10, scopeLine: 127, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!190 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !192)
!191 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 64>, 32, 32> >", scope: !182, file: !182, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!192 = distinct !DILocation(line: 106, column: 9, scope: !9)
!193 = !{i64 4294467}
!194 = !DILocation(line: 187, column: 47, scope: !191, inlinedAt: !192)
!195 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !198)
!196 = distinct !DISubprogram(name: "operator=", scope: !197, file: !197, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!197 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!198 = distinct !DILocation(line: 188, column: 45, scope: !191, inlinedAt: !192)
!199 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !200)
!200 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !201)
!201 = distinct !DILocation(line: 107, column: 9, scope: !9)
!202 = !DILocation(line: 187, column: 47, scope: !191, inlinedAt: !201)
!203 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !204)
!204 = distinct !DILocation(line: 188, column: 45, scope: !191, inlinedAt: !201)
!205 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !206)
!206 = distinct !DILocation(line: 185, column: 37, scope: !207, inlinedAt: !208)
!207 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 32, 32> >", scope: !182, file: !182, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!208 = distinct !DILocation(line: 108, column: 9, scope: !9)
!209 = !DILocation(line: 187, column: 47, scope: !207, inlinedAt: !208)
!210 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !211)
!211 = distinct !DILocation(line: 188, column: 45, scope: !207, inlinedAt: !208)
!212 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !213)
!213 = distinct !DILocation(line: 185, column: 37, scope: !207, inlinedAt: !214)
!214 = distinct !DILocation(line: 109, column: 9, scope: !9)
!215 = !DILocation(line: 187, column: 47, scope: !207, inlinedAt: !214)
!216 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !217)
!217 = distinct !DILocation(line: 188, column: 45, scope: !207, inlinedAt: !214)
!218 = !DILocation(line: 110, column: 9, scope: !9)
!219 = !DILocation(line: 112, column: 13, scope: !9)
!220 = !DILocation(line: 61, column: 18, scope: !138, inlinedAt: !221)
!221 = distinct !DILocation(line: 54, column: 39, scope: !222, inlinedAt: !223)
!222 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !182, file: !182, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!223 = distinct !DILocation(line: 113, column: 13, scope: !9)
!224 = !DILocation(line: 63, column: 68, scope: !94, inlinedAt: !225)
!225 = distinct !DILocation(line: 55, column: 24, scope: !222, inlinedAt: !223)
!226 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !225)
!227 = !DILocation(line: 32, column: 28, scope: !228, inlinedAt: !229)
!228 = distinct !DISubprogram(name: "make_srsrc", scope: !182, file: !182, line: 31, type: !10, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!229 = distinct !DILocation(line: 60, column: 19, scope: !222, inlinedAt: !223)
!230 = !DILocation(line: 26, column: 23, scope: !181, inlinedAt: !231)
!231 = distinct !DILocation(line: 77, column: 39, scope: !222, inlinedAt: !223)
!232 = !DILocation(line: 115, column: 9, scope: !9)
!233 = !DILocation(line: 116, column: 9, scope: !9)
!234 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !235)
!235 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !236)
!236 = distinct !DILocation(line: 118, column: 9, scope: !9)
!237 = !DILocation(line: 187, column: 47, scope: !191, inlinedAt: !236)
!238 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !239)
!239 = distinct !DILocation(line: 188, column: 45, scope: !191, inlinedAt: !236)
!240 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !241)
!241 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !242)
!242 = distinct !DILocation(line: 119, column: 9, scope: !9)
!243 = !DILocation(line: 187, column: 47, scope: !191, inlinedAt: !242)
!244 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !245)
!245 = distinct !DILocation(line: 188, column: 45, scope: !191, inlinedAt: !242)
!246 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !247)
!247 = distinct !DILocation(line: 185, column: 37, scope: !207, inlinedAt: !248)
!248 = distinct !DILocation(line: 120, column: 9, scope: !9)
!249 = !DILocation(line: 187, column: 47, scope: !207, inlinedAt: !248)
!250 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !251)
!251 = distinct !DILocation(line: 188, column: 45, scope: !207, inlinedAt: !248)
!252 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !253)
!253 = distinct !DILocation(line: 185, column: 37, scope: !207, inlinedAt: !254)
!254 = distinct !DILocation(line: 121, column: 9, scope: !9)
!255 = !DILocation(line: 187, column: 47, scope: !207, inlinedAt: !254)
!256 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !257)
!257 = distinct !DILocation(line: 188, column: 45, scope: !207, inlinedAt: !254)
!258 = !DILocation(line: 122, column: 9, scope: !9)
!259 = !DILocation(line: 123, column: 9, scope: !9)
!260 = !DILocation(line: 125, column: 9, scope: !9)
!261 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !264)
!262 = distinct !DISubprogram(name: "mfma161616", scope: !263, file: !263, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!263 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!264 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !266)
!265 = distinct !DISubprogram(name: "mma_ABt_base", scope: !263, file: !263, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!266 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !268)
!267 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !263, file: !263, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!268 = distinct !DILocation(line: 126, column: 9, scope: !9)
!269 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !270)
!270 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !271)
!271 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !268)
!272 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !273)
!273 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !274)
!274 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !275)
!275 = distinct !DILocation(line: 127, column: 9, scope: !9)
!276 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !277)
!277 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !278)
!278 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !275)
!279 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !280)
!280 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !281)
!281 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !282)
!282 = distinct !DILocation(line: 128, column: 9, scope: !9)
!283 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !284)
!284 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !285)
!285 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !282)
!286 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !287)
!287 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !288)
!288 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !289)
!289 = distinct !DILocation(line: 129, column: 9, scope: !9)
!290 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !291)
!291 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !292)
!292 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !289)
!293 = !DILocation(line: 130, column: 9, scope: !9)
!294 = !DILocation(line: 131, column: 9, scope: !9)
!295 = !DILocation(line: 132, column: 9, scope: !9)
!296 = !DILocation(line: 135, column: 13, scope: !9)
!297 = !DILocation(line: 136, column: 13, scope: !9)
!298 = !{i64 4304387}
!299 = !DILocation(line: 17, column: 5, scope: !300, inlinedAt: !301)
!300 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !182, file: !182, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!301 = distinct !DILocation(line: 119, column: 17, scope: !302, inlinedAt: !303)
!302 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !182, file: !182, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!303 = distinct !DILocation(line: 137, column: 13, scope: !9)
!304 = !{i64 4290493}
!305 = !DILocation(line: 17, column: 5, scope: !300, inlinedAt: !306)
!306 = distinct !DILocation(line: 120, column: 17, scope: !302, inlinedAt: !303)
!307 = !DILocation(line: 139, column: 13, scope: !9)
!308 = !{i64 4304568}
!309 = !DILocation(line: 17, column: 5, scope: !300, inlinedAt: !310)
!310 = distinct !DILocation(line: 119, column: 17, scope: !311, inlinedAt: !312)
!311 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 64>, 512>", scope: !182, file: !182, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!312 = distinct !DILocation(line: 140, column: 13, scope: !9)
!313 = !DILocation(line: 17, column: 5, scope: !300, inlinedAt: !314)
!314 = distinct !DILocation(line: 120, column: 17, scope: !311, inlinedAt: !312)
!315 = !DILocation(line: 142, column: 9, scope: !9)
!316 = !DILocation(line: 143, column: 9, scope: !9)
!317 = !DILocation(line: 144, column: 9, scope: !9)
!318 = !DILocation(line: 146, column: 9, scope: !9)
!319 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !320)
!320 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !321)
!321 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !322)
!322 = distinct !DILocation(line: 147, column: 9, scope: !9)
!323 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !324)
!324 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !325)
!325 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !322)
!326 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !327)
!327 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !328)
!328 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !329)
!329 = distinct !DILocation(line: 148, column: 9, scope: !9)
!330 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !331)
!331 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !332)
!332 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !329)
!333 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !334)
!334 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !335)
!335 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !336)
!336 = distinct !DILocation(line: 149, column: 9, scope: !9)
!337 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !338)
!338 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !339)
!339 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !336)
!340 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !341)
!341 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !342)
!342 = distinct !DILocation(line: 216, column: 13, scope: !267, inlinedAt: !343)
!343 = distinct !DILocation(line: 150, column: 9, scope: !9)
!344 = !DILocation(line: 29, column: 27, scope: !262, inlinedAt: !345)
!345 = distinct !DILocation(line: 81, column: 5, scope: !265, inlinedAt: !346)
!346 = distinct !DILocation(line: 224, column: 17, scope: !267, inlinedAt: !343)
!347 = !DILocation(line: 151, column: 9, scope: !9)
!348 = !DILocation(line: 152, column: 9, scope: !9)
!349 = !DILocation(line: 153, column: 9, scope: !9)
!350 = !DILocation(line: 97, column: 29, scope: !9)
!351 = distinct !{!351, !174, !352, !353}
!352 = !DILocation(line: 154, column: 5, scope: !9)
!353 = !{!"llvm.loop.mustprogress"}
!354 = !DILocation(line: 157, column: 9, scope: !9)
!355 = !DILocation(line: 158, column: 5, scope: !9)
!356 = !DILocation(line: 167, column: 32, scope: !9)
!357 = !DILocation(line: 60, column: 34, scope: !358, inlinedAt: !359)
!358 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!359 = distinct !DILocation(line: 192, column: 41, scope: !360, inlinedAt: !362)
!360 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !361, file: !361, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!361 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!362 = distinct !DILocation(line: 212, column: 5, scope: !363, inlinedAt: !364)
!363 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !361, file: !361, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!364 = distinct !DILocation(line: 164, column: 9, scope: !9)
!365 = !DILocation(line: 61, column: 18, scope: !358, inlinedAt: !359)
!366 = !DILocation(line: 63, column: 54, scope: !367, inlinedAt: !368)
!367 = distinct !DISubprogram(name: "operator[]", scope: !95, file: !95, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!368 = distinct !DILocation(line: 192, column: 23, scope: !360, inlinedAt: !362)
!369 = !DILocation(line: 63, column: 60, scope: !367, inlinedAt: !368)
!370 = !DILocation(line: 63, column: 68, scope: !367, inlinedAt: !368)
!371 = !DILocation(line: 63, column: 16, scope: !367, inlinedAt: !368)
!372 = !DILocation(line: 203, column: 49, scope: !360, inlinedAt: !362)
!373 = !DILocation(line: 203, column: 13, scope: !360, inlinedAt: !362)
!374 = !DILocation(line: 203, column: 47, scope: !360, inlinedAt: !362)
!375 = !{!376, !376, i64 0}
!376 = !{!"float", !377, i64 0}
!377 = !{!"omnipotent char", !378, i64 0}
!378 = !{!"Simple C++ TBAA"}
!379 = !DILocation(line: 204, column: 49, scope: !360, inlinedAt: !362)
!380 = !DILocation(line: 204, column: 13, scope: !360, inlinedAt: !362)
!381 = !DILocation(line: 204, column: 47, scope: !360, inlinedAt: !362)
!382 = !DILocation(line: 205, column: 49, scope: !360, inlinedAt: !362)
!383 = !DILocation(line: 205, column: 13, scope: !360, inlinedAt: !362)
!384 = !DILocation(line: 205, column: 47, scope: !360, inlinedAt: !362)
!385 = !DILocation(line: 206, column: 49, scope: !360, inlinedAt: !362)
!386 = !DILocation(line: 206, column: 13, scope: !360, inlinedAt: !362)
!387 = !DILocation(line: 206, column: 47, scope: !360, inlinedAt: !362)
!388 = !DILocation(line: 170, column: 1, scope: !9)
