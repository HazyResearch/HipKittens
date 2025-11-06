; ModuleID = '256_128_64_32-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_128_64_32.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%"struct.kittens::alignment_dummy" = type { i32, [12 x i8] }
%struct.micro_globals = type { %"struct.kittens::gl", %"struct.kittens::gl", %"struct.kittens::gl" }
%"struct.kittens::gl" = type <{ ptr, %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::ducks::gl::runtime_dim", %"struct.kittens::detail::descriptor_dict", [7 x i8] }>
%"struct.kittens::ducks::gl::runtime_dim" = type { i64 }
%"struct.kittens::detail::descriptor_dict" = type { i8 }
%struct.__hip_bfloat16 = type { i16 }

@__shm = external hidden addrspace(3) global [0 x %"struct.kittens::alignment_dummy"], align 16
@__hip_cuid_bc261f4d2c3f67d1 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_bc261f4d2c3f67d1 to ptr)], section "llvm.metadata"

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
  %11 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !27
  %12 = lshr i32 %11, 8, !dbg !35
  %13 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y(), !dbg !36
  %14 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr(), !dbg !41
  %15 = load i32, ptr addrspace(4) %14, align 4, !dbg !41, !tbaa !46
  %16 = mul i32 %15, %13, !dbg !50
  %17 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x(), !dbg !51
  %18 = add i32 %16, %17, !dbg !56
  %19 = getelementptr inbounds i8, ptr addrspace(4) %14, i64 4, !dbg !57
  %20 = load i32, ptr addrspace(4) %19, align 4, !dbg !57, !tbaa !46
  %21 = mul i32 %20, %15, !dbg !62
  %.frozen = freeze i32 %18, !dbg !63
  %22 = sdiv i32 %.frozen, 8, !dbg !63
  %23 = mul i32 %22, 8, !dbg !64
  %.decomposed = sub i32 %.frozen, %23, !dbg !64
  %24 = sdiv i32 %21, 8, !dbg !65
  %25 = mul nsw i32 %.decomposed, %24, !dbg !66
  %26 = add nsw i32 %25, %22, !dbg !67
  %.frozen1169 = freeze i32 %26, !dbg !68
  %27 = sdiv i32 %.frozen1169, 512, !dbg !68
  %28 = shl nsw i32 %27, 4, !dbg !69
  %29 = sub nsw i32 64, %28, !dbg !70
  %30 = tail call noundef range(i32 -67108816, 67108930) i32 @llvm.smin.i32(i32 %29, i32 16), !dbg !71
  %31 = mul i32 %27, 512, !dbg !75
  %.decomposed1170 = sub i32 %.frozen1169, %31, !dbg !75
  %32 = sdiv i32 %.decomposed1170, %30, !dbg !76
  %33 = mul i32 %32, %30, !dbg !77
  %.decomposed1171 = sub i32 %.decomposed1170, %33, !dbg !77
  %34 = add nsw i32 %28, %.decomposed1171, !dbg !78
  %.not.i.i118 = icmp eq i64 %8, 0, !dbg !79
  %35 = select i1 %.not.i.i118, ptr %6, ptr %10, !dbg !26
  %36 = lshr i32 %11, 6, !dbg !80
  %37 = and i32 %36, 3, !dbg !81
  %38 = trunc i64 %.sroa.7.0.copyload to i32, !dbg !82
  %39 = shl nsw i32 %34, 7, !dbg !89
  %40 = sext i32 %39 to i64, !dbg !93
  %41 = mul i64 %.sroa.7.0.copyload, %40, !dbg !97
  %42 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.0810.0.copyload, i64 %41, !dbg !98
  %43 = ptrtoint ptr %5 to i64, !dbg !99
  %44 = shl nuw nsw i32 %11, 3
  %45 = and i32 %44, 56
  %46 = lshr i32 %11, 3, !dbg !100
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
  %60 = shl nuw nsw i32 %11, 4
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
  %83 = shl nsw i32 %32, 8, !dbg !137
  %84 = sext i32 %83 to i64, !dbg !140
  %85 = mul i64 %.sroa.15.0.copyload, %84, !dbg !142
  %86 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.9815.0.copyload, i64 %85, !dbg !143
  %87 = ptrtoint ptr %35 to i64, !dbg !144
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
  %147 = icmp eq i32 %12, 1, !dbg !170
  br i1 %147, label %148, label %149, !dbg !171

148:                                              ; preds = %1
  tail call void @llvm.amdgcn.s.barrier(), !dbg !172
  br label %149, !dbg !173

149:                                              ; preds = %148, %1
  %150 = shl nuw nsw i32 %37, 5
  %151 = and i32 %11, 15
  %152 = lshr i32 %11, 2
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
  %205 = shl nuw nsw i32 %12, 5
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
  %377 = icmp ult i32 %11, 256, !dbg !175
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
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !186
  %extract.t850 = trunc i128 %396 to i64, !dbg !187
  %extract852 = lshr i128 %396, 64, !dbg !187
  %extract.t853 = trunc nuw i128 %extract852 to i64, !dbg !187
  %extract.t857 = trunc i128 %397 to i64, !dbg !187
  %extract859 = lshr i128 %397, 64, !dbg !187
  %extract.t860 = trunc nuw i128 %extract859 to i64, !dbg !187
  br label %398, !dbg !187

398:                                              ; preds = %389, %378
  %.sroa.0652.1.off0 = phi i64 [ %.sroa.0652.0.off0875, %378 ], [ %extract.t850, %389 ]
  %.sroa.0652.1.off64 = phi i64 [ %.sroa.0652.0.off64876, %378 ], [ %extract.t853, %389 ]
  %.sroa.5653.1.off0 = phi i64 [ %.sroa.5653.0.off0877, %378 ], [ %extract.t857, %389 ]
  %.sroa.5653.1.off64 = phi i64 [ %.sroa.5653.0.off64878, %378 ], [ %extract.t860, %389 ]
  %399 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %161) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.i = lshr i64 %399, 16, !dbg !194
  %400 = insertelement <2 x i64> poison, i64 %399, i64 0, !dbg !194
  %401 = insertelement <2 x i64> %400, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !194
  %402 = shufflevector <2 x i64> %400, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %403 = lshr <2 x i64> %402, <i64 32, i64 48>, !dbg !188
  %404 = shufflevector <2 x i64> %401, <2 x i64> %403, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.0688.6.vec.insert1138 = trunc <4 x i64> %404 to <4 x i16>, !dbg !195
  %405 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %169) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %405, 16, !dbg !194
  %406 = insertelement <2 x i64> poison, i64 %405, i64 0, !dbg !194
  %407 = insertelement <2 x i64> %406, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !194
  %408 = shufflevector <2 x i64> %406, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %409 = lshr <2 x i64> %408, <i64 32, i64 48>, !dbg !188
  %410 = shufflevector <2 x i64> %407, <2 x i64> %409, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.10693.14.vec.insert1137 = trunc <4 x i64> %410 to <4 x i16>, !dbg !195
  %411 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %177) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i = lshr i64 %411, 16, !dbg !194
  %412 = insertelement <2 x i64> poison, i64 %411, i64 0, !dbg !194
  %413 = insertelement <2 x i64> %412, i64 %.sroa.036.2.extract.shift37.145.i, i64 1, !dbg !194
  %414 = shufflevector <2 x i64> %412, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %415 = lshr <2 x i64> %414, <i64 32, i64 48>, !dbg !188
  %416 = shufflevector <2 x i64> %413, <2 x i64> %415, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.18698.22.vec.insert1136 = trunc <4 x i64> %416 to <4 x i16>, !dbg !195
  %417 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %181) #7, !dbg !188, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i = lshr i64 %417, 16, !dbg !194
  %418 = insertelement <2 x i64> poison, i64 %417, i64 0, !dbg !194
  %419 = insertelement <2 x i64> %418, i64 %.sroa.036.2.extract.shift37.1.1.i, i64 1, !dbg !194
  %420 = shufflevector <2 x i64> %418, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !188
  %421 = lshr <2 x i64> %420, <i64 32, i64 48>, !dbg !188
  %422 = shufflevector <2 x i64> %419, <2 x i64> %421, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.26703.30.vec.insert1135 = trunc <4 x i64> %422 to <4 x i16>, !dbg !195
  %423 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %189) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.i138 = lshr i64 %423, 16, !dbg !202
  %424 = insertelement <2 x i64> poison, i64 %423, i64 0, !dbg !202
  %425 = insertelement <2 x i64> %424, i64 %.sroa.036.2.extract.shift37.i138, i64 1, !dbg !202
  %426 = shufflevector <2 x i64> %424, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %427 = lshr <2 x i64> %426, <i64 32, i64 48>, !dbg !199
  %428 = shufflevector <2 x i64> %425, <2 x i64> %427, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.0655.6.vec.insert1134 = trunc <4 x i64> %428 to <4 x i16>, !dbg !203
  %429 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %193) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i147 = lshr i64 %429, 16, !dbg !202
  %430 = insertelement <2 x i64> poison, i64 %429, i64 0, !dbg !202
  %431 = insertelement <2 x i64> %430, i64 %.sroa.036.2.extract.shift37.1.i147, i64 1, !dbg !202
  %432 = shufflevector <2 x i64> %430, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %433 = lshr <2 x i64> %432, <i64 32, i64 48>, !dbg !199
  %434 = shufflevector <2 x i64> %431, <2 x i64> %433, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.10.14.vec.insert1133 = trunc <4 x i64> %434 to <4 x i16>, !dbg !203
  %435 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %200) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i156 = lshr i64 %435, 16, !dbg !202
  %436 = insertelement <2 x i64> poison, i64 %435, i64 0, !dbg !202
  %437 = insertelement <2 x i64> %436, i64 %.sroa.036.2.extract.shift37.145.i156, i64 1, !dbg !202
  %438 = shufflevector <2 x i64> %436, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %439 = lshr <2 x i64> %438, <i64 32, i64 48>, !dbg !199
  %440 = shufflevector <2 x i64> %437, <2 x i64> %439, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.18.22.vec.insert1132 = trunc <4 x i64> %440 to <4 x i16>, !dbg !203
  %441 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %204) #7, !dbg !199, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i165 = lshr i64 %441, 16, !dbg !202
  %442 = insertelement <2 x i64> poison, i64 %441, i64 0, !dbg !202
  %443 = insertelement <2 x i64> %442, i64 %.sroa.036.2.extract.shift37.1.1.i165, i64 1, !dbg !202
  %444 = shufflevector <2 x i64> %442, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !199
  %445 = lshr <2 x i64> %444, <i64 32, i64 48>, !dbg !199
  %446 = shufflevector <2 x i64> %443, <2 x i64> %445, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !203
  %.sroa.26.30.vec.insert1131 = trunc <4 x i64> %446 to <4 x i16>, !dbg !203
  %447 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %212) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.i175 = lshr i64 %447, 16, !dbg !209
  %448 = insertelement <2 x i64> poison, i64 %447, i64 0, !dbg !209
  %449 = insertelement <2 x i64> %448, i64 %.sroa.036.2.extract.shift37.i175, i64 1, !dbg !209
  %450 = shufflevector <2 x i64> %448, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %451 = lshr <2 x i64> %450, <i64 32, i64 48>, !dbg !205
  %452 = shufflevector <2 x i64> %449, <2 x i64> %451, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.0768.6.vec.insert1130 = trunc <4 x i64> %452 to <4 x i16>, !dbg !210
  %453 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %216) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i182 = lshr i64 %453, 16, !dbg !209
  %454 = insertelement <2 x i64> poison, i64 %453, i64 0, !dbg !209
  %455 = insertelement <2 x i64> %454, i64 %.sroa.036.2.extract.shift37.1.i182, i64 1, !dbg !209
  %456 = shufflevector <2 x i64> %454, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %457 = lshr <2 x i64> %456, <i64 32, i64 48>, !dbg !205
  %458 = shufflevector <2 x i64> %455, <2 x i64> %457, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.10773.14.vec.insert1129 = trunc <4 x i64> %458 to <4 x i16>, !dbg !210
  %459 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %223) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i189 = lshr i64 %459, 16, !dbg !209
  %460 = insertelement <2 x i64> poison, i64 %459, i64 0, !dbg !209
  %461 = insertelement <2 x i64> %460, i64 %.sroa.036.2.extract.shift37.145.i189, i64 1, !dbg !209
  %462 = shufflevector <2 x i64> %460, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %463 = lshr <2 x i64> %462, <i64 32, i64 48>, !dbg !205
  %464 = shufflevector <2 x i64> %461, <2 x i64> %463, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.18778.22.vec.insert1128 = trunc <4 x i64> %464 to <4 x i16>, !dbg !210
  %465 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %227) #7, !dbg !205, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i196 = lshr i64 %465, 16, !dbg !209
  %466 = insertelement <2 x i64> poison, i64 %465, i64 0, !dbg !209
  %467 = insertelement <2 x i64> %466, i64 %.sroa.036.2.extract.shift37.1.1.i196, i64 1, !dbg !209
  %468 = shufflevector <2 x i64> %466, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !205
  %469 = lshr <2 x i64> %468, <i64 32, i64 48>, !dbg !205
  %470 = shufflevector <2 x i64> %467, <2 x i64> %469, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !210
  %.sroa.26783.30.vec.insert1127 = trunc <4 x i64> %470 to <4 x i16>, !dbg !210
  %471 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %235) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.i208 = lshr i64 %471, 16, !dbg !215
  %472 = insertelement <2 x i64> poison, i64 %471, i64 0, !dbg !215
  %473 = insertelement <2 x i64> %472, i64 %.sroa.036.2.extract.shift37.i208, i64 1, !dbg !215
  %474 = shufflevector <2 x i64> %472, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %475 = lshr <2 x i64> %474, <i64 32, i64 48>, !dbg !212
  %476 = shufflevector <2 x i64> %473, <2 x i64> %475, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.0728.6.vec.insert1126 = trunc <4 x i64> %476 to <4 x i16>, !dbg !216
  %477 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %239) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i217 = lshr i64 %477, 16, !dbg !215
  %478 = insertelement <2 x i64> poison, i64 %477, i64 0, !dbg !215
  %479 = insertelement <2 x i64> %478, i64 %.sroa.036.2.extract.shift37.1.i217, i64 1, !dbg !215
  %480 = shufflevector <2 x i64> %478, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %481 = lshr <2 x i64> %480, <i64 32, i64 48>, !dbg !212
  %482 = shufflevector <2 x i64> %479, <2 x i64> %481, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.10733.14.vec.insert1125 = trunc <4 x i64> %482 to <4 x i16>, !dbg !216
  %483 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %246) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i226 = lshr i64 %483, 16, !dbg !215
  %484 = insertelement <2 x i64> poison, i64 %483, i64 0, !dbg !215
  %485 = insertelement <2 x i64> %484, i64 %.sroa.036.2.extract.shift37.145.i226, i64 1, !dbg !215
  %486 = shufflevector <2 x i64> %484, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %487 = lshr <2 x i64> %486, <i64 32, i64 48>, !dbg !212
  %488 = shufflevector <2 x i64> %485, <2 x i64> %487, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.18738.22.vec.insert1124 = trunc <4 x i64> %488 to <4 x i16>, !dbg !216
  %489 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %250) #7, !dbg !212, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i235 = lshr i64 %489, 16, !dbg !215
  %490 = insertelement <2 x i64> poison, i64 %489, i64 0, !dbg !215
  %491 = insertelement <2 x i64> %490, i64 %.sroa.036.2.extract.shift37.1.1.i235, i64 1, !dbg !215
  %492 = shufflevector <2 x i64> %490, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !212
  %493 = lshr <2 x i64> %492, <i64 32, i64 48>, !dbg !212
  %494 = shufflevector <2 x i64> %491, <2 x i64> %493, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !216
  %.sroa.26743.30.vec.insert1123 = trunc <4 x i64> %494 to <4 x i16>, !dbg !216
  br i1 %.not, label %506, label %495, !dbg !218

495:                                              ; preds = %398
  %496 = shl nuw nsw i32 %388, 6, !dbg !219
  %497 = zext nneg i32 %496 to i64, !dbg !223
  %498 = getelementptr %struct.__hip_bfloat16, ptr %86, i64 %497, !dbg !225
  %499 = ptrtoint ptr %498 to i64, !dbg !226
  %500 = bitcast i64 %499 to <2 x i32>, !dbg !226
  %.sroa.0.0.vec.expand.i.i243 = shufflevector <2 x i32> %500, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>, !dbg !226
  %501 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i243, i32 1114112, i64 3, !dbg !226
  %.sroa.0.12.vec.insert.i.i244 = insertelement <4 x i32> %501, i32 %251, i64 2, !dbg !226
  %502 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %252, i32 noundef 0, i32 noundef 0) #7, !dbg !229
  %503 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %253, i32 noundef 0, i32 noundef 0) #7, !dbg !229
  %504 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %254, i32 noundef 0, i32 noundef 0) #7, !dbg !229
  %505 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i244, i32 noundef %255, i32 noundef 0, i32 noundef 0) #7, !dbg !229
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !231
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
  %507 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %260) #7, !dbg !233, !srcloc !193
  %.sroa.036.2.extract.shift37.i252 = lshr i64 %507, 16, !dbg !236
  %508 = insertelement <2 x i64> poison, i64 %507, i64 0, !dbg !236
  %509 = insertelement <2 x i64> %508, i64 %.sroa.036.2.extract.shift37.i252, i64 1, !dbg !236
  %510 = shufflevector <2 x i64> %508, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !233
  %511 = lshr <2 x i64> %510, <i64 32, i64 48>, !dbg !233
  %512 = shufflevector <2 x i64> %509, <2 x i64> %511, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !237
  %.sroa.34708.38.vec.insert1122 = trunc <4 x i64> %512 to <4 x i16>, !dbg !237
  %513 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %265) #7, !dbg !233, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i261 = lshr i64 %513, 16, !dbg !236
  %514 = insertelement <2 x i64> poison, i64 %513, i64 0, !dbg !236
  %515 = insertelement <2 x i64> %514, i64 %.sroa.036.2.extract.shift37.1.i261, i64 1, !dbg !236
  %516 = shufflevector <2 x i64> %514, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !233
  %517 = lshr <2 x i64> %516, <i64 32, i64 48>, !dbg !233
  %518 = shufflevector <2 x i64> %515, <2 x i64> %517, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !237
  %.sroa.42713.46.vec.insert1121 = trunc <4 x i64> %518 to <4 x i16>, !dbg !237
  %519 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %269) #7, !dbg !233, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i270 = lshr i64 %519, 16, !dbg !236
  %520 = insertelement <2 x i64> poison, i64 %519, i64 0, !dbg !236
  %521 = insertelement <2 x i64> %520, i64 %.sroa.036.2.extract.shift37.145.i270, i64 1, !dbg !236
  %522 = shufflevector <2 x i64> %520, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !233
  %523 = lshr <2 x i64> %522, <i64 32, i64 48>, !dbg !233
  %524 = shufflevector <2 x i64> %521, <2 x i64> %523, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !237
  %.sroa.50718.54.vec.insert1120 = trunc <4 x i64> %524 to <4 x i16>, !dbg !237
  %525 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %273) #7, !dbg !233, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i279 = lshr i64 %525, 16, !dbg !236
  %526 = insertelement <2 x i64> poison, i64 %525, i64 0, !dbg !236
  %527 = insertelement <2 x i64> %526, i64 %.sroa.036.2.extract.shift37.1.1.i279, i64 1, !dbg !236
  %528 = shufflevector <2 x i64> %526, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !233
  %529 = lshr <2 x i64> %528, <i64 32, i64 48>, !dbg !233
  %530 = shufflevector <2 x i64> %527, <2 x i64> %529, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !237
  %.sroa.58723.62.vec.insert1119 = trunc <4 x i64> %530 to <4 x i16>, !dbg !237
  %531 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %277) #7, !dbg !239, !srcloc !193
  %.sroa.036.2.extract.shift37.i291 = lshr i64 %531, 16, !dbg !242
  %532 = insertelement <2 x i64> poison, i64 %531, i64 0, !dbg !242
  %533 = insertelement <2 x i64> %532, i64 %.sroa.036.2.extract.shift37.i291, i64 1, !dbg !242
  %534 = shufflevector <2 x i64> %532, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !239
  %535 = lshr <2 x i64> %534, <i64 32, i64 48>, !dbg !239
  %536 = shufflevector <2 x i64> %533, <2 x i64> %535, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !243
  %.sroa.34.38.vec.insert1118 = trunc <4 x i64> %536 to <4 x i16>, !dbg !243
  %537 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %281) #7, !dbg !239, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i300 = lshr i64 %537, 16, !dbg !242
  %538 = insertelement <2 x i64> poison, i64 %537, i64 0, !dbg !242
  %539 = insertelement <2 x i64> %538, i64 %.sroa.036.2.extract.shift37.1.i300, i64 1, !dbg !242
  %540 = shufflevector <2 x i64> %538, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !239
  %541 = lshr <2 x i64> %540, <i64 32, i64 48>, !dbg !239
  %542 = shufflevector <2 x i64> %539, <2 x i64> %541, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !243
  %.sroa.42.46.vec.insert1117 = trunc <4 x i64> %542 to <4 x i16>, !dbg !243
  %543 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %285) #7, !dbg !239, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i309 = lshr i64 %543, 16, !dbg !242
  %544 = insertelement <2 x i64> poison, i64 %543, i64 0, !dbg !242
  %545 = insertelement <2 x i64> %544, i64 %.sroa.036.2.extract.shift37.145.i309, i64 1, !dbg !242
  %546 = shufflevector <2 x i64> %544, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !239
  %547 = lshr <2 x i64> %546, <i64 32, i64 48>, !dbg !239
  %548 = shufflevector <2 x i64> %545, <2 x i64> %547, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !243
  %.sroa.50.54.vec.insert1116 = trunc <4 x i64> %548 to <4 x i16>, !dbg !243
  %549 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %289) #7, !dbg !239, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i318 = lshr i64 %549, 16, !dbg !242
  %550 = insertelement <2 x i64> poison, i64 %549, i64 0, !dbg !242
  %551 = insertelement <2 x i64> %550, i64 %.sroa.036.2.extract.shift37.1.1.i318, i64 1, !dbg !242
  %552 = shufflevector <2 x i64> %550, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !239
  %553 = lshr <2 x i64> %552, <i64 32, i64 48>, !dbg !239
  %554 = shufflevector <2 x i64> %551, <2 x i64> %553, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !243
  %.sroa.58.62.vec.insert1115 = trunc <4 x i64> %554 to <4 x i16>, !dbg !243
  %555 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %293) #7, !dbg !245, !srcloc !193
  %.sroa.036.2.extract.shift37.i330 = lshr i64 %555, 16, !dbg !248
  %556 = insertelement <2 x i64> poison, i64 %555, i64 0, !dbg !248
  %557 = insertelement <2 x i64> %556, i64 %.sroa.036.2.extract.shift37.i330, i64 1, !dbg !248
  %558 = shufflevector <2 x i64> %556, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !245
  %559 = lshr <2 x i64> %558, <i64 32, i64 48>, !dbg !245
  %560 = shufflevector <2 x i64> %557, <2 x i64> %559, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !249
  %.sroa.34788.38.vec.insert1114 = trunc <4 x i64> %560 to <4 x i16>, !dbg !249
  %561 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %297) #7, !dbg !245, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i339 = lshr i64 %561, 16, !dbg !248
  %562 = insertelement <2 x i64> poison, i64 %561, i64 0, !dbg !248
  %563 = insertelement <2 x i64> %562, i64 %.sroa.036.2.extract.shift37.1.i339, i64 1, !dbg !248
  %564 = shufflevector <2 x i64> %562, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !245
  %565 = lshr <2 x i64> %564, <i64 32, i64 48>, !dbg !245
  %566 = shufflevector <2 x i64> %563, <2 x i64> %565, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !249
  %.sroa.42793.46.vec.insert1113 = trunc <4 x i64> %566 to <4 x i16>, !dbg !249
  %567 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %301) #7, !dbg !245, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i348 = lshr i64 %567, 16, !dbg !248
  %568 = insertelement <2 x i64> poison, i64 %567, i64 0, !dbg !248
  %569 = insertelement <2 x i64> %568, i64 %.sroa.036.2.extract.shift37.145.i348, i64 1, !dbg !248
  %570 = shufflevector <2 x i64> %568, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !245
  %571 = lshr <2 x i64> %570, <i64 32, i64 48>, !dbg !245
  %572 = shufflevector <2 x i64> %569, <2 x i64> %571, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !249
  %.sroa.50798.54.vec.insert1112 = trunc <4 x i64> %572 to <4 x i16>, !dbg !249
  %573 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %305) #7, !dbg !245, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i357 = lshr i64 %573, 16, !dbg !248
  %574 = insertelement <2 x i64> poison, i64 %573, i64 0, !dbg !248
  %575 = insertelement <2 x i64> %574, i64 %.sroa.036.2.extract.shift37.1.1.i357, i64 1, !dbg !248
  %576 = shufflevector <2 x i64> %574, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !245
  %577 = lshr <2 x i64> %576, <i64 32, i64 48>, !dbg !245
  %578 = shufflevector <2 x i64> %575, <2 x i64> %577, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !249
  %.sroa.58803.62.vec.insert1111 = trunc <4 x i64> %578 to <4 x i16>, !dbg !249
  %579 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %309) #7, !dbg !251, !srcloc !193
  %.sroa.036.2.extract.shift37.i369 = lshr i64 %579, 16, !dbg !254
  %580 = insertelement <2 x i64> poison, i64 %579, i64 0, !dbg !254
  %581 = insertelement <2 x i64> %580, i64 %.sroa.036.2.extract.shift37.i369, i64 1, !dbg !254
  %582 = shufflevector <2 x i64> %580, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !251
  %583 = lshr <2 x i64> %582, <i64 32, i64 48>, !dbg !251
  %584 = shufflevector <2 x i64> %581, <2 x i64> %583, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !255
  %.sroa.34748.38.vec.insert1110 = trunc <4 x i64> %584 to <4 x i16>, !dbg !255
  %585 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %313) #7, !dbg !251, !srcloc !193
  %.sroa.036.2.extract.shift37.1.i378 = lshr i64 %585, 16, !dbg !254
  %586 = insertelement <2 x i64> poison, i64 %585, i64 0, !dbg !254
  %587 = insertelement <2 x i64> %586, i64 %.sroa.036.2.extract.shift37.1.i378, i64 1, !dbg !254
  %588 = shufflevector <2 x i64> %586, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !251
  %589 = lshr <2 x i64> %588, <i64 32, i64 48>, !dbg !251
  %590 = shufflevector <2 x i64> %587, <2 x i64> %589, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !255
  %.sroa.42753.46.vec.insert1109 = trunc <4 x i64> %590 to <4 x i16>, !dbg !255
  %591 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %317) #7, !dbg !251, !srcloc !193
  %.sroa.036.2.extract.shift37.145.i387 = lshr i64 %591, 16, !dbg !254
  %592 = insertelement <2 x i64> poison, i64 %591, i64 0, !dbg !254
  %593 = insertelement <2 x i64> %592, i64 %.sroa.036.2.extract.shift37.145.i387, i64 1, !dbg !254
  %594 = shufflevector <2 x i64> %592, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !251
  %595 = lshr <2 x i64> %594, <i64 32, i64 48>, !dbg !251
  %596 = shufflevector <2 x i64> %593, <2 x i64> %595, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !255
  %.sroa.50758.54.vec.insert1108 = trunc <4 x i64> %596 to <4 x i16>, !dbg !255
  %597 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %321) #7, !dbg !251, !srcloc !193
  %.sroa.036.2.extract.shift37.1.1.i396 = lshr i64 %597, 16, !dbg !254
  %598 = insertelement <2 x i64> poison, i64 %597, i64 0, !dbg !254
  %599 = insertelement <2 x i64> %598, i64 %.sroa.036.2.extract.shift37.1.1.i396, i64 1, !dbg !254
  %600 = shufflevector <2 x i64> %598, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !251
  %601 = lshr <2 x i64> %600, <i64 32, i64 48>, !dbg !251
  %602 = shufflevector <2 x i64> %599, <2 x i64> %601, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !255
  %.sroa.58763.62.vec.insert1107 = trunc <4 x i64> %602 to <4 x i16>, !dbg !255
  tail call void @llvm.amdgcn.s.barrier(), !dbg !257
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !258
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !259
  %603 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1130, <4 x i16> %.sroa.0688.6.vec.insert1138, <4 x float> %.val33.i865866, i32 0, i32 0, i32 0), !dbg !260
  %604 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1129, <4 x i16> %.sroa.10693.14.vec.insert1137, <4 x float> %603, i32 0, i32 0, i32 0), !dbg !268
  %605 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1130, <4 x i16> %.sroa.18698.22.vec.insert1136, <4 x float> %.val33.1.i480882, i32 0, i32 0, i32 0), !dbg !260
  %606 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1129, <4 x i16> %.sroa.26703.30.vec.insert1135, <4 x float> %605, i32 0, i32 0, i32 0), !dbg !268
  %607 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1128, <4 x i16> %.sroa.0688.6.vec.insert1138, <4 x float> %.val33.138.i485885, i32 0, i32 0, i32 0), !dbg !260
  %608 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1127, <4 x i16> %.sroa.10693.14.vec.insert1137, <4 x float> %607, i32 0, i32 0, i32 0), !dbg !268
  %609 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1128, <4 x i16> %.sroa.18698.22.vec.insert1136, <4 x float> %.val33.1.1.i490888, i32 0, i32 0, i32 0), !dbg !260
  %610 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1127, <4 x i16> %.sroa.26703.30.vec.insert1135, <4 x float> %609, i32 0, i32 0, i32 0), !dbg !268
  %611 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1130, <4 x i16> %.sroa.0655.6.vec.insert1134, <4 x float> %.val33.i415891, i32 0, i32 0, i32 0), !dbg !271
  %612 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1129, <4 x i16> %.sroa.10.14.vec.insert1133, <4 x float> %611, i32 0, i32 0, i32 0), !dbg !275
  %613 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0768.6.vec.insert1130, <4 x i16> %.sroa.18.22.vec.insert1132, <4 x float> %387, i32 0, i32 0, i32 0), !dbg !271
  %614 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10773.14.vec.insert1129, <4 x i16> %.sroa.26.30.vec.insert1131, <4 x float> %613, i32 0, i32 0, i32 0), !dbg !275
  %615 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1128, <4 x i16> %.sroa.0655.6.vec.insert1134, <4 x float> %386, i32 0, i32 0, i32 0), !dbg !271
  %616 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1127, <4 x i16> %.sroa.10.14.vec.insert1133, <4 x float> %615, i32 0, i32 0, i32 0), !dbg !275
  %617 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18778.22.vec.insert1128, <4 x i16> %.sroa.18.22.vec.insert1132, <4 x float> %385, i32 0, i32 0, i32 0), !dbg !271
  %618 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26783.30.vec.insert1127, <4 x i16> %.sroa.26.30.vec.insert1131, <4 x float> %617, i32 0, i32 0, i32 0), !dbg !275
  %619 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1126, <4 x i16> %.sroa.0688.6.vec.insert1138, <4 x float> %.val33.i435900, i32 0, i32 0, i32 0), !dbg !278
  %620 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1125, <4 x i16> %.sroa.10693.14.vec.insert1137, <4 x float> %619, i32 0, i32 0, i32 0), !dbg !282
  %621 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1126, <4 x i16> %.sroa.18698.22.vec.insert1136, <4 x float> %384, i32 0, i32 0, i32 0), !dbg !278
  %622 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1125, <4 x i16> %.sroa.26703.30.vec.insert1135, <4 x float> %621, i32 0, i32 0, i32 0), !dbg !282
  %623 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1124, <4 x i16> %.sroa.0688.6.vec.insert1138, <4 x float> %383, i32 0, i32 0, i32 0), !dbg !278
  %624 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1123, <4 x i16> %.sroa.10693.14.vec.insert1137, <4 x float> %623, i32 0, i32 0, i32 0), !dbg !282
  %625 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1124, <4 x i16> %.sroa.18698.22.vec.insert1136, <4 x float> %382, i32 0, i32 0, i32 0), !dbg !278
  %626 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1123, <4 x i16> %.sroa.26703.30.vec.insert1135, <4 x float> %625, i32 0, i32 0, i32 0), !dbg !282
  %627 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1126, <4 x i16> %.sroa.0655.6.vec.insert1134, <4 x float> %.val33.i455909, i32 0, i32 0, i32 0), !dbg !285
  %628 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1125, <4 x i16> %.sroa.10.14.vec.insert1133, <4 x float> %627, i32 0, i32 0, i32 0), !dbg !289
  %629 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0728.6.vec.insert1126, <4 x i16> %.sroa.18.22.vec.insert1132, <4 x float> %381, i32 0, i32 0, i32 0), !dbg !285
  %630 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10733.14.vec.insert1125, <4 x i16> %.sroa.26.30.vec.insert1131, <4 x float> %629, i32 0, i32 0, i32 0), !dbg !289
  %631 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1124, <4 x i16> %.sroa.0655.6.vec.insert1134, <4 x float> %380, i32 0, i32 0, i32 0), !dbg !285
  %632 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1123, <4 x i16> %.sroa.10.14.vec.insert1133, <4 x float> %631, i32 0, i32 0, i32 0), !dbg !289
  %633 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18738.22.vec.insert1124, <4 x i16> %.sroa.18.22.vec.insert1132, <4 x float> %379, i32 0, i32 0, i32 0), !dbg !285
  %634 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26743.30.vec.insert1123, <4 x i16> %.sroa.26.30.vec.insert1131, <4 x float> %633, i32 0, i32 0, i32 0), !dbg !289
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !292
  tail call void @llvm.amdgcn.s.barrier(), !dbg !293
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !294
  br i1 %.not, label %636, label %635, !dbg !295

635:                                              ; preds = %506
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !296, !srcloc !297
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %329, i64 %.sroa.0652.1.off0) #7, !dbg !298, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %333, i64 %.sroa.0652.1.off64) #7, !dbg !304, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %337, i64 %.sroa.5653.1.off0) #7, !dbg !298, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %341, i64 %.sroa.5653.1.off64) #7, !dbg !304, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %347, i64 %.sroa.0650.1.off0) #7, !dbg !306, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %351, i64 %.sroa.0650.1.off64) #7, !dbg !310, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %355, i64 %.sroa.5.1.off0) #7, !dbg !306, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %359, i64 %.sroa.5.1.off64) #7, !dbg !310, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %363, i64 %.sroa.8651.1.off0) #7, !dbg !306, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %367, i64 %.sroa.8651.1.off64) #7, !dbg !310, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %371, i64 %.sroa.11.1.off0) #7, !dbg !306, !srcloc !303
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %375, i64 %.sroa.11.1.off64) #7, !dbg !310, !srcloc !303
  br label %636, !dbg !312

636:                                              ; preds = %635, %506
  tail call void @llvm.amdgcn.s.barrier(), !dbg !313
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !314
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !315
  %637 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1114, <4 x i16> %.sroa.34708.38.vec.insert1122, <4 x float> %604, i32 0, i32 0, i32 0), !dbg !316
  %638 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1113, <4 x i16> %.sroa.42713.46.vec.insert1121, <4 x float> %637, i32 0, i32 0, i32 0), !dbg !320
  %639 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1114, <4 x i16> %.sroa.50718.54.vec.insert1120, <4 x float> %606, i32 0, i32 0, i32 0), !dbg !316
  %640 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1113, <4 x i16> %.sroa.58723.62.vec.insert1119, <4 x float> %639, i32 0, i32 0, i32 0), !dbg !320
  %641 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1112, <4 x i16> %.sroa.34708.38.vec.insert1122, <4 x float> %608, i32 0, i32 0, i32 0), !dbg !316
  %642 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1111, <4 x i16> %.sroa.42713.46.vec.insert1121, <4 x float> %641, i32 0, i32 0, i32 0), !dbg !320
  %643 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1112, <4 x i16> %.sroa.50718.54.vec.insert1120, <4 x float> %610, i32 0, i32 0, i32 0), !dbg !316
  %644 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1111, <4 x i16> %.sroa.58723.62.vec.insert1119, <4 x float> %643, i32 0, i32 0, i32 0), !dbg !320
  %645 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1114, <4 x i16> %.sroa.34.38.vec.insert1118, <4 x float> %612, i32 0, i32 0, i32 0), !dbg !323
  %646 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1113, <4 x i16> %.sroa.42.46.vec.insert1117, <4 x float> %645, i32 0, i32 0, i32 0), !dbg !327
  %647 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34788.38.vec.insert1114, <4 x i16> %.sroa.50.54.vec.insert1116, <4 x float> %614, i32 0, i32 0, i32 0), !dbg !323
  %648 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42793.46.vec.insert1113, <4 x i16> %.sroa.58.62.vec.insert1115, <4 x float> %647, i32 0, i32 0, i32 0), !dbg !327
  %649 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1112, <4 x i16> %.sroa.34.38.vec.insert1118, <4 x float> %616, i32 0, i32 0, i32 0), !dbg !323
  %650 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1111, <4 x i16> %.sroa.42.46.vec.insert1117, <4 x float> %649, i32 0, i32 0, i32 0), !dbg !327
  %651 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50798.54.vec.insert1112, <4 x i16> %.sroa.50.54.vec.insert1116, <4 x float> %618, i32 0, i32 0, i32 0), !dbg !323
  %652 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58803.62.vec.insert1111, <4 x i16> %.sroa.58.62.vec.insert1115, <4 x float> %651, i32 0, i32 0, i32 0), !dbg !327
  %653 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1110, <4 x i16> %.sroa.34708.38.vec.insert1122, <4 x float> %620, i32 0, i32 0, i32 0), !dbg !330
  %654 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1109, <4 x i16> %.sroa.42713.46.vec.insert1121, <4 x float> %653, i32 0, i32 0, i32 0), !dbg !334
  %655 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1110, <4 x i16> %.sroa.50718.54.vec.insert1120, <4 x float> %622, i32 0, i32 0, i32 0), !dbg !330
  %656 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1109, <4 x i16> %.sroa.58723.62.vec.insert1119, <4 x float> %655, i32 0, i32 0, i32 0), !dbg !334
  %657 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1108, <4 x i16> %.sroa.34708.38.vec.insert1122, <4 x float> %624, i32 0, i32 0, i32 0), !dbg !330
  %658 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1107, <4 x i16> %.sroa.42713.46.vec.insert1121, <4 x float> %657, i32 0, i32 0, i32 0), !dbg !334
  %659 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1108, <4 x i16> %.sroa.50718.54.vec.insert1120, <4 x float> %626, i32 0, i32 0, i32 0), !dbg !330
  %660 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1107, <4 x i16> %.sroa.58723.62.vec.insert1119, <4 x float> %659, i32 0, i32 0, i32 0), !dbg !334
  %661 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1110, <4 x i16> %.sroa.34.38.vec.insert1118, <4 x float> %628, i32 0, i32 0, i32 0), !dbg !337
  %662 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1109, <4 x i16> %.sroa.42.46.vec.insert1117, <4 x float> %661, i32 0, i32 0, i32 0), !dbg !341
  %663 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34748.38.vec.insert1110, <4 x i16> %.sroa.50.54.vec.insert1116, <4 x float> %630, i32 0, i32 0, i32 0), !dbg !337
  %664 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42753.46.vec.insert1109, <4 x i16> %.sroa.58.62.vec.insert1115, <4 x float> %663, i32 0, i32 0, i32 0), !dbg !341
  %665 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1108, <4 x i16> %.sroa.34.38.vec.insert1118, <4 x float> %632, i32 0, i32 0, i32 0), !dbg !337
  %666 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1107, <4 x i16> %.sroa.42.46.vec.insert1117, <4 x float> %665, i32 0, i32 0, i32 0), !dbg !341
  %667 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50758.54.vec.insert1108, <4 x i16> %.sroa.50.54.vec.insert1116, <4 x float> %634, i32 0, i32 0, i32 0), !dbg !337
  %668 = tail call fast <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58763.62.vec.insert1107, <4 x i16> %.sroa.58.62.vec.insert1115, <4 x float> %667, i32 0, i32 0, i32 0), !dbg !341
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !344
  tail call void @llvm.amdgcn.s.barrier(), !dbg !345
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !346
  %exitcond.not = icmp eq i32 %388, 128, !dbg !347
  br i1 %exitcond.not, label %376, label %378, !dbg !174, !llvm.loop !348

669:                                              ; preds = %376
  tail call void @llvm.amdgcn.s.barrier(), !dbg !351
  br label %670, !dbg !352

670:                                              ; preds = %669, %376
  %671 = shl nsw i32 %34, 2
  %672 = or disjoint i32 %671, %12
  %673 = shl nsw i32 %32, 3
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
  %722 = or disjoint i32 %673, %37, !dbg !353
  %723 = shl nsw i32 %672, 5, !dbg !354
  %724 = shl nsw i32 %722, 5, !dbg !362
  %725 = sext i32 %723 to i64, !dbg !363
  %726 = mul i64 %.sroa.20.0.copyload, %725, !dbg !365
  %727 = sext i32 %724 to i64, !dbg !366
  %728 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.17819.0.copyload, i64 %726, !dbg !367
  %729 = getelementptr %struct.__hip_bfloat16, ptr %728, i64 %727, !dbg !367
  %bc = bitcast <4 x float> %638 to <4 x i32>, !dbg !368
  %730 = extractelement <4 x i32> %bc, i64 0, !dbg !368
  %731 = and i32 %730, 2139095040, !dbg !369
  %.not.i.i.i.i.i.i = icmp eq i32 %731, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.i.i, label %737, label %732, !dbg !379

732:                                              ; preds = %670
  %733 = lshr i32 %730, 16, !dbg !380
  %734 = and i32 %733, 1, !dbg !381
  %735 = add nsw i32 %730, 32767, !dbg !382
  %736 = add nsw i32 %735, %734, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i, !dbg !384

737:                                              ; preds = %670
  %738 = and i32 %730, 65535, !dbg !385
  %.not6.i.i.i.i.i.i = icmp eq i32 %738, 0, !dbg !386
  %739 = or i32 %730, 65536
  %740 = select i1 %.not6.i.i.i.i.i.i, i32 %730, i32 %739, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i: ; preds = %737, %732
  %.sroa.0.0.i.i.i.i.i.i = phi i32 [ %736, %732 ], [ %740, %737 ], !dbg !388
  %741 = lshr i32 %.sroa.0.0.i.i.i.i.i.i, 16, !dbg !389
  %742 = trunc nuw i32 %741 to i16, !dbg !390
  %743 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %683, !dbg !391
  store i16 %742, ptr %743, align 2, !dbg !392, !tbaa !393
  %bc1014 = bitcast <4 x float> %638 to <4 x i32>, !dbg !397
  %744 = extractelement <4 x i32> %bc1014, i64 1, !dbg !397
  %745 = and i32 %744, 2139095040, !dbg !398
  %.not.i.i.i.i52.i.i = icmp eq i32 %745, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.i.i, label %751, label %746, !dbg !403

746:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i
  %747 = lshr i32 %744, 16, !dbg !404
  %748 = and i32 %747, 1, !dbg !405
  %749 = add nsw i32 %744, 32767, !dbg !406
  %750 = add nsw i32 %749, %748, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i, !dbg !408

751:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i
  %752 = and i32 %744, 65535, !dbg !409
  %.not6.i.i.i.i54.i.i = icmp eq i32 %752, 0, !dbg !410
  %753 = or i32 %744, 65536
  %754 = select i1 %.not6.i.i.i.i54.i.i, i32 %744, i32 %753, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i: ; preds = %751, %746
  %.sroa.0.0.i.i.i.i53.i.i = phi i32 [ %750, %746 ], [ %754, %751 ], !dbg !412
  %755 = lshr i32 %.sroa.0.0.i.i.i.i53.i.i, 16, !dbg !413
  %756 = trunc nuw i32 %755 to i16, !dbg !414
  %757 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %685, !dbg !415
  store i16 %756, ptr %757, align 2, !dbg !416, !tbaa !393
  %bc1015 = bitcast <4 x float> %638 to <4 x i32>, !dbg !417
  %758 = extractelement <4 x i32> %bc1015, i64 2, !dbg !417
  %759 = and i32 %758, 2139095040, !dbg !418
  %.not.i.i.i.i56.i.i = icmp eq i32 %759, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.i.i, label %765, label %760, !dbg !423

760:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i
  %761 = lshr i32 %758, 16, !dbg !424
  %762 = and i32 %761, 1, !dbg !425
  %763 = add nsw i32 %758, 32767, !dbg !426
  %764 = add nsw i32 %763, %762, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i, !dbg !428

765:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i
  %766 = and i32 %758, 65535, !dbg !429
  %.not6.i.i.i.i58.i.i = icmp eq i32 %766, 0, !dbg !430
  %767 = or i32 %758, 65536
  %768 = select i1 %.not6.i.i.i.i58.i.i, i32 %758, i32 %767, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i: ; preds = %765, %760
  %.sroa.0.0.i.i.i.i57.i.i = phi i32 [ %764, %760 ], [ %768, %765 ], !dbg !432
  %769 = lshr i32 %.sroa.0.0.i.i.i.i57.i.i, 16, !dbg !433
  %770 = trunc nuw i32 %769 to i16, !dbg !434
  %771 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %687, !dbg !435
  store i16 %770, ptr %771, align 2, !dbg !436, !tbaa !393
  %bc1016 = bitcast <4 x float> %638 to <4 x i32>, !dbg !437
  %772 = extractelement <4 x i32> %bc1016, i64 3, !dbg !437
  %773 = and i32 %772, 2139095040, !dbg !438
  %.not.i.i.i.i60.i.i = icmp eq i32 %773, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.i.i, label %779, label %774, !dbg !443

774:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i
  %775 = lshr i32 %772, 16, !dbg !444
  %776 = and i32 %775, 1, !dbg !445
  %777 = add nsw i32 %772, 32767, !dbg !446
  %778 = add nsw i32 %777, %776, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i, !dbg !448

779:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i
  %780 = and i32 %772, 65535, !dbg !449
  %.not6.i.i.i.i62.i.i = icmp eq i32 %780, 0, !dbg !450
  %781 = or i32 %772, 65536
  %782 = select i1 %.not6.i.i.i.i62.i.i, i32 %772, i32 %781, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i: ; preds = %779, %774
  %.sroa.0.0.i.i.i.i61.i.i = phi i32 [ %778, %774 ], [ %782, %779 ], !dbg !452
  %783 = lshr i32 %.sroa.0.0.i.i.i.i61.i.i, 16, !dbg !453
  %784 = trunc nuw i32 %783 to i16, !dbg !454
  %785 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %689, !dbg !455
  store i16 %784, ptr %785, align 2, !dbg !456, !tbaa !393
  %bc1017 = bitcast <4 x float> %640 to <4 x i32>, !dbg !368
  %786 = extractelement <4 x i32> %bc1017, i64 0, !dbg !368
  %787 = and i32 %786, 2139095040, !dbg !369
  %.not.i.i.i.i.1.i.i = icmp eq i32 %787, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.i.i, label %793, label %788, !dbg !379

788:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i
  %789 = lshr i32 %786, 16, !dbg !380
  %790 = and i32 %789, 1, !dbg !381
  %791 = add nsw i32 %786, 32767, !dbg !382
  %792 = add nsw i32 %791, %790, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i, !dbg !384

793:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i
  %794 = and i32 %786, 65535, !dbg !385
  %.not6.i.i.i.i.1.i.i = icmp eq i32 %794, 0, !dbg !386
  %795 = or i32 %786, 65536
  %796 = select i1 %.not6.i.i.i.i.1.i.i, i32 %786, i32 %795, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i: ; preds = %793, %788
  %.sroa.0.0.i.i.i.i.1.i.i = phi i32 [ %792, %788 ], [ %796, %793 ], !dbg !388
  %797 = lshr i32 %.sroa.0.0.i.i.i.i.1.i.i, 16, !dbg !389
  %798 = trunc nuw i32 %797 to i16, !dbg !390
  %799 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %691, !dbg !391
  store i16 %798, ptr %799, align 2, !dbg !392, !tbaa !393
  %bc1018 = bitcast <4 x float> %640 to <4 x i32>, !dbg !397
  %800 = extractelement <4 x i32> %bc1018, i64 1, !dbg !397
  %801 = and i32 %800, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.i.i = icmp eq i32 %801, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.i.i, label %807, label %802, !dbg !403

802:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i
  %803 = lshr i32 %800, 16, !dbg !404
  %804 = and i32 %803, 1, !dbg !405
  %805 = add nsw i32 %800, 32767, !dbg !406
  %806 = add nsw i32 %805, %804, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i, !dbg !408

807:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i
  %808 = and i32 %800, 65535, !dbg !409
  %.not6.i.i.i.i54.1.i.i = icmp eq i32 %808, 0, !dbg !410
  %809 = or i32 %800, 65536
  %810 = select i1 %.not6.i.i.i.i54.1.i.i, i32 %800, i32 %809, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i: ; preds = %807, %802
  %.sroa.0.0.i.i.i.i53.1.i.i = phi i32 [ %806, %802 ], [ %810, %807 ], !dbg !412
  %811 = lshr i32 %.sroa.0.0.i.i.i.i53.1.i.i, 16, !dbg !413
  %812 = trunc nuw i32 %811 to i16, !dbg !414
  %813 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %693, !dbg !415
  store i16 %812, ptr %813, align 2, !dbg !416, !tbaa !393
  %bc1019 = bitcast <4 x float> %640 to <4 x i32>, !dbg !417
  %814 = extractelement <4 x i32> %bc1019, i64 2, !dbg !417
  %815 = and i32 %814, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.i.i = icmp eq i32 %815, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.i.i, label %821, label %816, !dbg !423

816:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i
  %817 = lshr i32 %814, 16, !dbg !424
  %818 = and i32 %817, 1, !dbg !425
  %819 = add nsw i32 %814, 32767, !dbg !426
  %820 = add nsw i32 %819, %818, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i, !dbg !428

821:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i
  %822 = and i32 %814, 65535, !dbg !429
  %.not6.i.i.i.i58.1.i.i = icmp eq i32 %822, 0, !dbg !430
  %823 = or i32 %814, 65536
  %824 = select i1 %.not6.i.i.i.i58.1.i.i, i32 %814, i32 %823, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i: ; preds = %821, %816
  %.sroa.0.0.i.i.i.i57.1.i.i = phi i32 [ %820, %816 ], [ %824, %821 ], !dbg !432
  %825 = lshr i32 %.sroa.0.0.i.i.i.i57.1.i.i, 16, !dbg !433
  %826 = trunc nuw i32 %825 to i16, !dbg !434
  %827 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %695, !dbg !435
  store i16 %826, ptr %827, align 2, !dbg !436, !tbaa !393
  %bc1020 = bitcast <4 x float> %640 to <4 x i32>, !dbg !437
  %828 = extractelement <4 x i32> %bc1020, i64 3, !dbg !437
  %829 = and i32 %828, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.i.i = icmp eq i32 %829, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.i.i, label %835, label %830, !dbg !443

830:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i
  %831 = lshr i32 %828, 16, !dbg !444
  %832 = and i32 %831, 1, !dbg !445
  %833 = add nsw i32 %828, 32767, !dbg !446
  %834 = add nsw i32 %833, %832, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i, !dbg !448

835:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i
  %836 = and i32 %828, 65535, !dbg !449
  %.not6.i.i.i.i62.1.i.i = icmp eq i32 %836, 0, !dbg !450
  %837 = or i32 %828, 65536
  %838 = select i1 %.not6.i.i.i.i62.1.i.i, i32 %828, i32 %837, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i: ; preds = %835, %830
  %.sroa.0.0.i.i.i.i61.1.i.i = phi i32 [ %834, %830 ], [ %838, %835 ], !dbg !452
  %839 = lshr i32 %.sroa.0.0.i.i.i.i61.1.i.i, 16, !dbg !453
  %840 = trunc nuw i32 %839 to i16, !dbg !454
  %841 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %697, !dbg !455
  store i16 %840, ptr %841, align 2, !dbg !456, !tbaa !393
  %bc1021 = bitcast <4 x float> %642 to <4 x i32>, !dbg !368
  %842 = extractelement <4 x i32> %bc1021, i64 0, !dbg !368
  %843 = and i32 %842, 2139095040, !dbg !369
  %.not.i.i.i.i.171.i.i = icmp eq i32 %843, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.171.i.i, label %849, label %844, !dbg !379

844:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i
  %845 = lshr i32 %842, 16, !dbg !380
  %846 = and i32 %845, 1, !dbg !381
  %847 = add nsw i32 %842, 32767, !dbg !382
  %848 = add nsw i32 %847, %846, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i, !dbg !384

849:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i
  %850 = and i32 %842, 65535, !dbg !385
  %.not6.i.i.i.i.172.i.i = icmp eq i32 %850, 0, !dbg !386
  %851 = or i32 %842, 65536
  %852 = select i1 %.not6.i.i.i.i.172.i.i, i32 %842, i32 %851, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i: ; preds = %849, %844
  %.sroa.0.0.i.i.i.i.173.i.i = phi i32 [ %848, %844 ], [ %852, %849 ], !dbg !388
  %853 = lshr i32 %.sroa.0.0.i.i.i.i.173.i.i, 16, !dbg !389
  %854 = trunc nuw i32 %853 to i16, !dbg !390
  %855 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %707, !dbg !391
  store i16 %854, ptr %855, align 2, !dbg !392, !tbaa !393
  %bc1022 = bitcast <4 x float> %642 to <4 x i32>, !dbg !397
  %856 = extractelement <4 x i32> %bc1022, i64 1, !dbg !397
  %857 = and i32 %856, 2139095040, !dbg !398
  %.not.i.i.i.i52.175.i.i = icmp eq i32 %857, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.175.i.i, label %863, label %858, !dbg !403

858:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i
  %859 = lshr i32 %856, 16, !dbg !404
  %860 = and i32 %859, 1, !dbg !405
  %861 = add nsw i32 %856, 32767, !dbg !406
  %862 = add nsw i32 %861, %860, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i, !dbg !408

863:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i
  %864 = and i32 %856, 65535, !dbg !409
  %.not6.i.i.i.i54.177.i.i = icmp eq i32 %864, 0, !dbg !410
  %865 = or i32 %856, 65536
  %866 = select i1 %.not6.i.i.i.i54.177.i.i, i32 %856, i32 %865, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i: ; preds = %863, %858
  %.sroa.0.0.i.i.i.i53.178.i.i = phi i32 [ %862, %858 ], [ %866, %863 ], !dbg !412
  %867 = lshr i32 %.sroa.0.0.i.i.i.i53.178.i.i, 16, !dbg !413
  %868 = trunc nuw i32 %867 to i16, !dbg !414
  %869 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %709, !dbg !415
  store i16 %868, ptr %869, align 2, !dbg !416, !tbaa !393
  %bc1023 = bitcast <4 x float> %642 to <4 x i32>, !dbg !417
  %870 = extractelement <4 x i32> %bc1023, i64 2, !dbg !417
  %871 = and i32 %870, 2139095040, !dbg !418
  %.not.i.i.i.i56.180.i.i = icmp eq i32 %871, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.180.i.i, label %877, label %872, !dbg !423

872:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i
  %873 = lshr i32 %870, 16, !dbg !424
  %874 = and i32 %873, 1, !dbg !425
  %875 = add nsw i32 %870, 32767, !dbg !426
  %876 = add nsw i32 %875, %874, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i, !dbg !428

877:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i
  %878 = and i32 %870, 65535, !dbg !429
  %.not6.i.i.i.i58.182.i.i = icmp eq i32 %878, 0, !dbg !430
  %879 = or i32 %870, 65536
  %880 = select i1 %.not6.i.i.i.i58.182.i.i, i32 %870, i32 %879, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i: ; preds = %877, %872
  %.sroa.0.0.i.i.i.i57.183.i.i = phi i32 [ %876, %872 ], [ %880, %877 ], !dbg !432
  %881 = lshr i32 %.sroa.0.0.i.i.i.i57.183.i.i, 16, !dbg !433
  %882 = trunc nuw i32 %881 to i16, !dbg !434
  %883 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %711, !dbg !435
  store i16 %882, ptr %883, align 2, !dbg !436, !tbaa !393
  %bc1024 = bitcast <4 x float> %642 to <4 x i32>, !dbg !437
  %884 = extractelement <4 x i32> %bc1024, i64 3, !dbg !437
  %885 = and i32 %884, 2139095040, !dbg !438
  %.not.i.i.i.i60.185.i.i = icmp eq i32 %885, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.185.i.i, label %891, label %886, !dbg !443

886:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i
  %887 = lshr i32 %884, 16, !dbg !444
  %888 = and i32 %887, 1, !dbg !445
  %889 = add nsw i32 %884, 32767, !dbg !446
  %890 = add nsw i32 %889, %888, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i, !dbg !448

891:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i
  %892 = and i32 %884, 65535, !dbg !449
  %.not6.i.i.i.i62.187.i.i = icmp eq i32 %892, 0, !dbg !450
  %893 = or i32 %884, 65536
  %894 = select i1 %.not6.i.i.i.i62.187.i.i, i32 %884, i32 %893, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i: ; preds = %891, %886
  %.sroa.0.0.i.i.i.i61.188.i.i = phi i32 [ %890, %886 ], [ %894, %891 ], !dbg !452
  %895 = lshr i32 %.sroa.0.0.i.i.i.i61.188.i.i, 16, !dbg !453
  %896 = trunc nuw i32 %895 to i16, !dbg !454
  %897 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %713, !dbg !455
  store i16 %896, ptr %897, align 2, !dbg !456, !tbaa !393
  %bc1025 = bitcast <4 x float> %644 to <4 x i32>, !dbg !368
  %898 = extractelement <4 x i32> %bc1025, i64 0, !dbg !368
  %899 = and i32 %898, 2139095040, !dbg !369
  %.not.i.i.i.i.1.1.i.i = icmp eq i32 %899, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.1.i.i, label %905, label %900, !dbg !379

900:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i
  %901 = lshr i32 %898, 16, !dbg !380
  %902 = and i32 %901, 1, !dbg !381
  %903 = add nsw i32 %898, 32767, !dbg !382
  %904 = add nsw i32 %903, %902, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i, !dbg !384

905:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i
  %906 = and i32 %898, 65535, !dbg !385
  %.not6.i.i.i.i.1.1.i.i = icmp eq i32 %906, 0, !dbg !386
  %907 = or i32 %898, 65536
  %908 = select i1 %.not6.i.i.i.i.1.1.i.i, i32 %898, i32 %907, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i: ; preds = %905, %900
  %.sroa.0.0.i.i.i.i.1.1.i.i = phi i32 [ %904, %900 ], [ %908, %905 ], !dbg !388
  %909 = lshr i32 %.sroa.0.0.i.i.i.i.1.1.i.i, 16, !dbg !389
  %910 = trunc nuw i32 %909 to i16, !dbg !390
  %911 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %715, !dbg !391
  store i16 %910, ptr %911, align 2, !dbg !392, !tbaa !393
  %bc1026 = bitcast <4 x float> %644 to <4 x i32>, !dbg !397
  %912 = extractelement <4 x i32> %bc1026, i64 1, !dbg !397
  %913 = and i32 %912, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.1.i.i = icmp eq i32 %913, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.1.i.i, label %919, label %914, !dbg !403

914:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i
  %915 = lshr i32 %912, 16, !dbg !404
  %916 = and i32 %915, 1, !dbg !405
  %917 = add nsw i32 %912, 32767, !dbg !406
  %918 = add nsw i32 %917, %916, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i, !dbg !408

919:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i
  %920 = and i32 %912, 65535, !dbg !409
  %.not6.i.i.i.i54.1.1.i.i = icmp eq i32 %920, 0, !dbg !410
  %921 = or i32 %912, 65536
  %922 = select i1 %.not6.i.i.i.i54.1.1.i.i, i32 %912, i32 %921, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i: ; preds = %919, %914
  %.sroa.0.0.i.i.i.i53.1.1.i.i = phi i32 [ %918, %914 ], [ %922, %919 ], !dbg !412
  %923 = lshr i32 %.sroa.0.0.i.i.i.i53.1.1.i.i, 16, !dbg !413
  %924 = trunc nuw i32 %923 to i16, !dbg !414
  %925 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %717, !dbg !415
  store i16 %924, ptr %925, align 2, !dbg !416, !tbaa !393
  %bc1027 = bitcast <4 x float> %644 to <4 x i32>, !dbg !417
  %926 = extractelement <4 x i32> %bc1027, i64 2, !dbg !417
  %927 = and i32 %926, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.1.i.i = icmp eq i32 %927, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.1.i.i, label %933, label %928, !dbg !423

928:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i
  %929 = lshr i32 %926, 16, !dbg !424
  %930 = and i32 %929, 1, !dbg !425
  %931 = add nsw i32 %926, 32767, !dbg !426
  %932 = add nsw i32 %931, %930, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i, !dbg !428

933:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i
  %934 = and i32 %926, 65535, !dbg !429
  %.not6.i.i.i.i58.1.1.i.i = icmp eq i32 %934, 0, !dbg !430
  %935 = or i32 %926, 65536
  %936 = select i1 %.not6.i.i.i.i58.1.1.i.i, i32 %926, i32 %935, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i: ; preds = %933, %928
  %.sroa.0.0.i.i.i.i57.1.1.i.i = phi i32 [ %932, %928 ], [ %936, %933 ], !dbg !432
  %937 = lshr i32 %.sroa.0.0.i.i.i.i57.1.1.i.i, 16, !dbg !433
  %938 = trunc nuw i32 %937 to i16, !dbg !434
  %939 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %719, !dbg !435
  store i16 %938, ptr %939, align 2, !dbg !436, !tbaa !393
  %bc1028 = bitcast <4 x float> %644 to <4 x i32>, !dbg !437
  %940 = extractelement <4 x i32> %bc1028, i64 3, !dbg !437
  %941 = and i32 %940, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.1.i.i = icmp eq i32 %941, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.1.i.i, label %947, label %942, !dbg !443

942:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i
  %943 = lshr i32 %940, 16, !dbg !444
  %944 = and i32 %943, 1, !dbg !445
  %945 = add nsw i32 %940, 32767, !dbg !446
  %946 = add nsw i32 %945, %944, !dbg !447
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit, !dbg !448

947:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i
  %948 = and i32 %940, 65535, !dbg !449
  %.not6.i.i.i.i62.1.1.i.i = icmp eq i32 %948, 0, !dbg !450
  %949 = or i32 %940, 65536
  %950 = select i1 %.not6.i.i.i.i62.1.1.i.i, i32 %940, i32 %949, !dbg !451
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit, !dbg !450

_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit: ; preds = %942, %947
  %.sroa.0.0.i.i.i.i61.1.1.i.i = phi i32 [ %946, %942 ], [ %950, %947 ], !dbg !452
  %951 = lshr i32 %.sroa.0.0.i.i.i.i61.1.1.i.i, 16, !dbg !453
  %952 = trunc nuw i32 %951 to i16, !dbg !454
  %953 = getelementptr inbounds %struct.__hip_bfloat16, ptr %729, i64 %721, !dbg !455
  store i16 %952, ptr %953, align 2, !dbg !456, !tbaa !393
  %954 = or disjoint i32 %673, %37, !dbg !353
  %955 = shl i32 %954, 5, !dbg !362
  %956 = or disjoint i32 %955, 128, !dbg !362
  %957 = sext i32 %956 to i64, !dbg !366
  %958 = getelementptr %struct.__hip_bfloat16, ptr %728, i64 %957, !dbg !367
  %bc1029 = bitcast <4 x float> %646 to <4 x i32>, !dbg !368
  %959 = extractelement <4 x i32> %bc1029, i64 0, !dbg !368
  %960 = and i32 %959, 2139095040, !dbg !369
  %.not.i.i.i.i.i.i.1 = icmp eq i32 %960, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.i.i.1, label %966, label %961, !dbg !379

961:                                              ; preds = %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit
  %962 = lshr i32 %959, 16, !dbg !380
  %963 = and i32 %962, 1, !dbg !381
  %964 = add nsw i32 %959, 32767, !dbg !382
  %965 = add nsw i32 %964, %963, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.1, !dbg !384

966:                                              ; preds = %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit
  %967 = and i32 %959, 65535, !dbg !385
  %.not6.i.i.i.i.i.i.1 = icmp eq i32 %967, 0, !dbg !386
  %968 = or i32 %959, 65536
  %969 = select i1 %.not6.i.i.i.i.i.i.1, i32 %959, i32 %968, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.1, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.1: ; preds = %966, %961
  %.sroa.0.0.i.i.i.i.i.i.1 = phi i32 [ %965, %961 ], [ %969, %966 ], !dbg !388
  %970 = lshr i32 %.sroa.0.0.i.i.i.i.i.i.1, 16, !dbg !389
  %971 = trunc nuw i32 %970 to i16, !dbg !390
  %972 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %683, !dbg !391
  store i16 %971, ptr %972, align 2, !dbg !392, !tbaa !393
  %bc1030 = bitcast <4 x float> %646 to <4 x i32>, !dbg !397
  %973 = extractelement <4 x i32> %bc1030, i64 1, !dbg !397
  %974 = and i32 %973, 2139095040, !dbg !398
  %.not.i.i.i.i52.i.i.1 = icmp eq i32 %974, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.i.i.1, label %980, label %975, !dbg !403

975:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.1
  %976 = lshr i32 %973, 16, !dbg !404
  %977 = and i32 %976, 1, !dbg !405
  %978 = add nsw i32 %973, 32767, !dbg !406
  %979 = add nsw i32 %978, %977, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.1, !dbg !408

980:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.1
  %981 = and i32 %973, 65535, !dbg !409
  %.not6.i.i.i.i54.i.i.1 = icmp eq i32 %981, 0, !dbg !410
  %982 = or i32 %973, 65536
  %983 = select i1 %.not6.i.i.i.i54.i.i.1, i32 %973, i32 %982, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.1, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.1: ; preds = %980, %975
  %.sroa.0.0.i.i.i.i53.i.i.1 = phi i32 [ %979, %975 ], [ %983, %980 ], !dbg !412
  %984 = lshr i32 %.sroa.0.0.i.i.i.i53.i.i.1, 16, !dbg !413
  %985 = trunc nuw i32 %984 to i16, !dbg !414
  %986 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %685, !dbg !415
  store i16 %985, ptr %986, align 2, !dbg !416, !tbaa !393
  %bc1031 = bitcast <4 x float> %646 to <4 x i32>, !dbg !417
  %987 = extractelement <4 x i32> %bc1031, i64 2, !dbg !417
  %988 = and i32 %987, 2139095040, !dbg !418
  %.not.i.i.i.i56.i.i.1 = icmp eq i32 %988, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.i.i.1, label %994, label %989, !dbg !423

989:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.1
  %990 = lshr i32 %987, 16, !dbg !424
  %991 = and i32 %990, 1, !dbg !425
  %992 = add nsw i32 %987, 32767, !dbg !426
  %993 = add nsw i32 %992, %991, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.1, !dbg !428

994:                                              ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.1
  %995 = and i32 %987, 65535, !dbg !429
  %.not6.i.i.i.i58.i.i.1 = icmp eq i32 %995, 0, !dbg !430
  %996 = or i32 %987, 65536
  %997 = select i1 %.not6.i.i.i.i58.i.i.1, i32 %987, i32 %996, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.1, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.1: ; preds = %994, %989
  %.sroa.0.0.i.i.i.i57.i.i.1 = phi i32 [ %993, %989 ], [ %997, %994 ], !dbg !432
  %998 = lshr i32 %.sroa.0.0.i.i.i.i57.i.i.1, 16, !dbg !433
  %999 = trunc nuw i32 %998 to i16, !dbg !434
  %1000 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %687, !dbg !435
  store i16 %999, ptr %1000, align 2, !dbg !436, !tbaa !393
  %bc1032 = bitcast <4 x float> %646 to <4 x i32>, !dbg !437
  %1001 = extractelement <4 x i32> %bc1032, i64 3, !dbg !437
  %1002 = and i32 %1001, 2139095040, !dbg !438
  %.not.i.i.i.i60.i.i.1 = icmp eq i32 %1002, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.i.i.1, label %1008, label %1003, !dbg !443

1003:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.1
  %1004 = lshr i32 %1001, 16, !dbg !444
  %1005 = and i32 %1004, 1, !dbg !445
  %1006 = add nsw i32 %1001, 32767, !dbg !446
  %1007 = add nsw i32 %1006, %1005, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.1, !dbg !448

1008:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.1
  %1009 = and i32 %1001, 65535, !dbg !449
  %.not6.i.i.i.i62.i.i.1 = icmp eq i32 %1009, 0, !dbg !450
  %1010 = or i32 %1001, 65536
  %1011 = select i1 %.not6.i.i.i.i62.i.i.1, i32 %1001, i32 %1010, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.1, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.1: ; preds = %1008, %1003
  %.sroa.0.0.i.i.i.i61.i.i.1 = phi i32 [ %1007, %1003 ], [ %1011, %1008 ], !dbg !452
  %1012 = lshr i32 %.sroa.0.0.i.i.i.i61.i.i.1, 16, !dbg !453
  %1013 = trunc nuw i32 %1012 to i16, !dbg !454
  %1014 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %689, !dbg !455
  store i16 %1013, ptr %1014, align 2, !dbg !456, !tbaa !393
  %bc1033 = bitcast <4 x float> %648 to <4 x i32>, !dbg !368
  %1015 = extractelement <4 x i32> %bc1033, i64 0, !dbg !368
  %1016 = and i32 %1015, 2139095040, !dbg !369
  %.not.i.i.i.i.1.i.i.1 = icmp eq i32 %1016, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.i.i.1, label %1022, label %1017, !dbg !379

1017:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.1
  %1018 = lshr i32 %1015, 16, !dbg !380
  %1019 = and i32 %1018, 1, !dbg !381
  %1020 = add nsw i32 %1015, 32767, !dbg !382
  %1021 = add nsw i32 %1020, %1019, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.1, !dbg !384

1022:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.1
  %1023 = and i32 %1015, 65535, !dbg !385
  %.not6.i.i.i.i.1.i.i.1 = icmp eq i32 %1023, 0, !dbg !386
  %1024 = or i32 %1015, 65536
  %1025 = select i1 %.not6.i.i.i.i.1.i.i.1, i32 %1015, i32 %1024, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.1, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.1: ; preds = %1022, %1017
  %.sroa.0.0.i.i.i.i.1.i.i.1 = phi i32 [ %1021, %1017 ], [ %1025, %1022 ], !dbg !388
  %1026 = lshr i32 %.sroa.0.0.i.i.i.i.1.i.i.1, 16, !dbg !389
  %1027 = trunc nuw i32 %1026 to i16, !dbg !390
  %1028 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %691, !dbg !391
  store i16 %1027, ptr %1028, align 2, !dbg !392, !tbaa !393
  %bc1034 = bitcast <4 x float> %648 to <4 x i32>, !dbg !397
  %1029 = extractelement <4 x i32> %bc1034, i64 1, !dbg !397
  %1030 = and i32 %1029, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.i.i.1 = icmp eq i32 %1030, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.i.i.1, label %1036, label %1031, !dbg !403

1031:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.1
  %1032 = lshr i32 %1029, 16, !dbg !404
  %1033 = and i32 %1032, 1, !dbg !405
  %1034 = add nsw i32 %1029, 32767, !dbg !406
  %1035 = add nsw i32 %1034, %1033, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.1, !dbg !408

1036:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.1
  %1037 = and i32 %1029, 65535, !dbg !409
  %.not6.i.i.i.i54.1.i.i.1 = icmp eq i32 %1037, 0, !dbg !410
  %1038 = or i32 %1029, 65536
  %1039 = select i1 %.not6.i.i.i.i54.1.i.i.1, i32 %1029, i32 %1038, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.1, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.1: ; preds = %1036, %1031
  %.sroa.0.0.i.i.i.i53.1.i.i.1 = phi i32 [ %1035, %1031 ], [ %1039, %1036 ], !dbg !412
  %1040 = lshr i32 %.sroa.0.0.i.i.i.i53.1.i.i.1, 16, !dbg !413
  %1041 = trunc nuw i32 %1040 to i16, !dbg !414
  %1042 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %693, !dbg !415
  store i16 %1041, ptr %1042, align 2, !dbg !416, !tbaa !393
  %bc1035 = bitcast <4 x float> %648 to <4 x i32>, !dbg !417
  %1043 = extractelement <4 x i32> %bc1035, i64 2, !dbg !417
  %1044 = and i32 %1043, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.i.i.1 = icmp eq i32 %1044, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.i.i.1, label %1050, label %1045, !dbg !423

1045:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.1
  %1046 = lshr i32 %1043, 16, !dbg !424
  %1047 = and i32 %1046, 1, !dbg !425
  %1048 = add nsw i32 %1043, 32767, !dbg !426
  %1049 = add nsw i32 %1048, %1047, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.1, !dbg !428

1050:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.1
  %1051 = and i32 %1043, 65535, !dbg !429
  %.not6.i.i.i.i58.1.i.i.1 = icmp eq i32 %1051, 0, !dbg !430
  %1052 = or i32 %1043, 65536
  %1053 = select i1 %.not6.i.i.i.i58.1.i.i.1, i32 %1043, i32 %1052, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.1, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.1: ; preds = %1050, %1045
  %.sroa.0.0.i.i.i.i57.1.i.i.1 = phi i32 [ %1049, %1045 ], [ %1053, %1050 ], !dbg !432
  %1054 = lshr i32 %.sroa.0.0.i.i.i.i57.1.i.i.1, 16, !dbg !433
  %1055 = trunc nuw i32 %1054 to i16, !dbg !434
  %1056 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %695, !dbg !435
  store i16 %1055, ptr %1056, align 2, !dbg !436, !tbaa !393
  %bc1036 = bitcast <4 x float> %648 to <4 x i32>, !dbg !437
  %1057 = extractelement <4 x i32> %bc1036, i64 3, !dbg !437
  %1058 = and i32 %1057, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.i.i.1 = icmp eq i32 %1058, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.i.i.1, label %1064, label %1059, !dbg !443

1059:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.1
  %1060 = lshr i32 %1057, 16, !dbg !444
  %1061 = and i32 %1060, 1, !dbg !445
  %1062 = add nsw i32 %1057, 32767, !dbg !446
  %1063 = add nsw i32 %1062, %1061, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.1, !dbg !448

1064:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.1
  %1065 = and i32 %1057, 65535, !dbg !449
  %.not6.i.i.i.i62.1.i.i.1 = icmp eq i32 %1065, 0, !dbg !450
  %1066 = or i32 %1057, 65536
  %1067 = select i1 %.not6.i.i.i.i62.1.i.i.1, i32 %1057, i32 %1066, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.1, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.1: ; preds = %1064, %1059
  %.sroa.0.0.i.i.i.i61.1.i.i.1 = phi i32 [ %1063, %1059 ], [ %1067, %1064 ], !dbg !452
  %1068 = lshr i32 %.sroa.0.0.i.i.i.i61.1.i.i.1, 16, !dbg !453
  %1069 = trunc nuw i32 %1068 to i16, !dbg !454
  %1070 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %697, !dbg !455
  store i16 %1069, ptr %1070, align 2, !dbg !456, !tbaa !393
  %bc1037 = bitcast <4 x float> %650 to <4 x i32>, !dbg !368
  %1071 = extractelement <4 x i32> %bc1037, i64 0, !dbg !368
  %1072 = and i32 %1071, 2139095040, !dbg !369
  %.not.i.i.i.i.171.i.i.1 = icmp eq i32 %1072, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.171.i.i.1, label %1078, label %1073, !dbg !379

1073:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.1
  %1074 = lshr i32 %1071, 16, !dbg !380
  %1075 = and i32 %1074, 1, !dbg !381
  %1076 = add nsw i32 %1071, 32767, !dbg !382
  %1077 = add nsw i32 %1076, %1075, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.1, !dbg !384

1078:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.1
  %1079 = and i32 %1071, 65535, !dbg !385
  %.not6.i.i.i.i.172.i.i.1 = icmp eq i32 %1079, 0, !dbg !386
  %1080 = or i32 %1071, 65536
  %1081 = select i1 %.not6.i.i.i.i.172.i.i.1, i32 %1071, i32 %1080, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.1, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.1: ; preds = %1078, %1073
  %.sroa.0.0.i.i.i.i.173.i.i.1 = phi i32 [ %1077, %1073 ], [ %1081, %1078 ], !dbg !388
  %1082 = lshr i32 %.sroa.0.0.i.i.i.i.173.i.i.1, 16, !dbg !389
  %1083 = trunc nuw i32 %1082 to i16, !dbg !390
  %1084 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %707, !dbg !391
  store i16 %1083, ptr %1084, align 2, !dbg !392, !tbaa !393
  %bc1038 = bitcast <4 x float> %650 to <4 x i32>, !dbg !397
  %1085 = extractelement <4 x i32> %bc1038, i64 1, !dbg !397
  %1086 = and i32 %1085, 2139095040, !dbg !398
  %.not.i.i.i.i52.175.i.i.1 = icmp eq i32 %1086, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.175.i.i.1, label %1092, label %1087, !dbg !403

1087:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.1
  %1088 = lshr i32 %1085, 16, !dbg !404
  %1089 = and i32 %1088, 1, !dbg !405
  %1090 = add nsw i32 %1085, 32767, !dbg !406
  %1091 = add nsw i32 %1090, %1089, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.1, !dbg !408

1092:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.1
  %1093 = and i32 %1085, 65535, !dbg !409
  %.not6.i.i.i.i54.177.i.i.1 = icmp eq i32 %1093, 0, !dbg !410
  %1094 = or i32 %1085, 65536
  %1095 = select i1 %.not6.i.i.i.i54.177.i.i.1, i32 %1085, i32 %1094, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.1, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.1: ; preds = %1092, %1087
  %.sroa.0.0.i.i.i.i53.178.i.i.1 = phi i32 [ %1091, %1087 ], [ %1095, %1092 ], !dbg !412
  %1096 = lshr i32 %.sroa.0.0.i.i.i.i53.178.i.i.1, 16, !dbg !413
  %1097 = trunc nuw i32 %1096 to i16, !dbg !414
  %1098 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %709, !dbg !415
  store i16 %1097, ptr %1098, align 2, !dbg !416, !tbaa !393
  %bc1039 = bitcast <4 x float> %650 to <4 x i32>, !dbg !417
  %1099 = extractelement <4 x i32> %bc1039, i64 2, !dbg !417
  %1100 = and i32 %1099, 2139095040, !dbg !418
  %.not.i.i.i.i56.180.i.i.1 = icmp eq i32 %1100, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.180.i.i.1, label %1106, label %1101, !dbg !423

1101:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.1
  %1102 = lshr i32 %1099, 16, !dbg !424
  %1103 = and i32 %1102, 1, !dbg !425
  %1104 = add nsw i32 %1099, 32767, !dbg !426
  %1105 = add nsw i32 %1104, %1103, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.1, !dbg !428

1106:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.1
  %1107 = and i32 %1099, 65535, !dbg !429
  %.not6.i.i.i.i58.182.i.i.1 = icmp eq i32 %1107, 0, !dbg !430
  %1108 = or i32 %1099, 65536
  %1109 = select i1 %.not6.i.i.i.i58.182.i.i.1, i32 %1099, i32 %1108, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.1, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.1: ; preds = %1106, %1101
  %.sroa.0.0.i.i.i.i57.183.i.i.1 = phi i32 [ %1105, %1101 ], [ %1109, %1106 ], !dbg !432
  %1110 = lshr i32 %.sroa.0.0.i.i.i.i57.183.i.i.1, 16, !dbg !433
  %1111 = trunc nuw i32 %1110 to i16, !dbg !434
  %1112 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %711, !dbg !435
  store i16 %1111, ptr %1112, align 2, !dbg !436, !tbaa !393
  %bc1040 = bitcast <4 x float> %650 to <4 x i32>, !dbg !437
  %1113 = extractelement <4 x i32> %bc1040, i64 3, !dbg !437
  %1114 = and i32 %1113, 2139095040, !dbg !438
  %.not.i.i.i.i60.185.i.i.1 = icmp eq i32 %1114, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.185.i.i.1, label %1120, label %1115, !dbg !443

1115:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.1
  %1116 = lshr i32 %1113, 16, !dbg !444
  %1117 = and i32 %1116, 1, !dbg !445
  %1118 = add nsw i32 %1113, 32767, !dbg !446
  %1119 = add nsw i32 %1118, %1117, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.1, !dbg !448

1120:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.1
  %1121 = and i32 %1113, 65535, !dbg !449
  %.not6.i.i.i.i62.187.i.i.1 = icmp eq i32 %1121, 0, !dbg !450
  %1122 = or i32 %1113, 65536
  %1123 = select i1 %.not6.i.i.i.i62.187.i.i.1, i32 %1113, i32 %1122, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.1, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.1: ; preds = %1120, %1115
  %.sroa.0.0.i.i.i.i61.188.i.i.1 = phi i32 [ %1119, %1115 ], [ %1123, %1120 ], !dbg !452
  %1124 = lshr i32 %.sroa.0.0.i.i.i.i61.188.i.i.1, 16, !dbg !453
  %1125 = trunc nuw i32 %1124 to i16, !dbg !454
  %1126 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %713, !dbg !455
  store i16 %1125, ptr %1126, align 2, !dbg !456, !tbaa !393
  %bc1041 = bitcast <4 x float> %652 to <4 x i32>, !dbg !368
  %1127 = extractelement <4 x i32> %bc1041, i64 0, !dbg !368
  %1128 = and i32 %1127, 2139095040, !dbg !369
  %.not.i.i.i.i.1.1.i.i.1 = icmp eq i32 %1128, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.1.i.i.1, label %1134, label %1129, !dbg !379

1129:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.1
  %1130 = lshr i32 %1127, 16, !dbg !380
  %1131 = and i32 %1130, 1, !dbg !381
  %1132 = add nsw i32 %1127, 32767, !dbg !382
  %1133 = add nsw i32 %1132, %1131, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.1, !dbg !384

1134:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.1
  %1135 = and i32 %1127, 65535, !dbg !385
  %.not6.i.i.i.i.1.1.i.i.1 = icmp eq i32 %1135, 0, !dbg !386
  %1136 = or i32 %1127, 65536
  %1137 = select i1 %.not6.i.i.i.i.1.1.i.i.1, i32 %1127, i32 %1136, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.1, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.1: ; preds = %1134, %1129
  %.sroa.0.0.i.i.i.i.1.1.i.i.1 = phi i32 [ %1133, %1129 ], [ %1137, %1134 ], !dbg !388
  %1138 = lshr i32 %.sroa.0.0.i.i.i.i.1.1.i.i.1, 16, !dbg !389
  %1139 = trunc nuw i32 %1138 to i16, !dbg !390
  %1140 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %715, !dbg !391
  store i16 %1139, ptr %1140, align 2, !dbg !392, !tbaa !393
  %bc1042 = bitcast <4 x float> %652 to <4 x i32>, !dbg !397
  %1141 = extractelement <4 x i32> %bc1042, i64 1, !dbg !397
  %1142 = and i32 %1141, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.1.i.i.1 = icmp eq i32 %1142, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.1.i.i.1, label %1148, label %1143, !dbg !403

1143:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.1
  %1144 = lshr i32 %1141, 16, !dbg !404
  %1145 = and i32 %1144, 1, !dbg !405
  %1146 = add nsw i32 %1141, 32767, !dbg !406
  %1147 = add nsw i32 %1146, %1145, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.1, !dbg !408

1148:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.1
  %1149 = and i32 %1141, 65535, !dbg !409
  %.not6.i.i.i.i54.1.1.i.i.1 = icmp eq i32 %1149, 0, !dbg !410
  %1150 = or i32 %1141, 65536
  %1151 = select i1 %.not6.i.i.i.i54.1.1.i.i.1, i32 %1141, i32 %1150, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.1, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.1: ; preds = %1148, %1143
  %.sroa.0.0.i.i.i.i53.1.1.i.i.1 = phi i32 [ %1147, %1143 ], [ %1151, %1148 ], !dbg !412
  %1152 = lshr i32 %.sroa.0.0.i.i.i.i53.1.1.i.i.1, 16, !dbg !413
  %1153 = trunc nuw i32 %1152 to i16, !dbg !414
  %1154 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %717, !dbg !415
  store i16 %1153, ptr %1154, align 2, !dbg !416, !tbaa !393
  %bc1043 = bitcast <4 x float> %652 to <4 x i32>, !dbg !417
  %1155 = extractelement <4 x i32> %bc1043, i64 2, !dbg !417
  %1156 = and i32 %1155, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.1.i.i.1 = icmp eq i32 %1156, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.1.i.i.1, label %1162, label %1157, !dbg !423

1157:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.1
  %1158 = lshr i32 %1155, 16, !dbg !424
  %1159 = and i32 %1158, 1, !dbg !425
  %1160 = add nsw i32 %1155, 32767, !dbg !426
  %1161 = add nsw i32 %1160, %1159, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.1, !dbg !428

1162:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.1
  %1163 = and i32 %1155, 65535, !dbg !429
  %.not6.i.i.i.i58.1.1.i.i.1 = icmp eq i32 %1163, 0, !dbg !430
  %1164 = or i32 %1155, 65536
  %1165 = select i1 %.not6.i.i.i.i58.1.1.i.i.1, i32 %1155, i32 %1164, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.1, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.1: ; preds = %1162, %1157
  %.sroa.0.0.i.i.i.i57.1.1.i.i.1 = phi i32 [ %1161, %1157 ], [ %1165, %1162 ], !dbg !432
  %1166 = lshr i32 %.sroa.0.0.i.i.i.i57.1.1.i.i.1, 16, !dbg !433
  %1167 = trunc nuw i32 %1166 to i16, !dbg !434
  %1168 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %719, !dbg !435
  store i16 %1167, ptr %1168, align 2, !dbg !436, !tbaa !393
  %bc1044 = bitcast <4 x float> %652 to <4 x i32>, !dbg !437
  %1169 = extractelement <4 x i32> %bc1044, i64 3, !dbg !437
  %1170 = and i32 %1169, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.1.i.i.1 = icmp eq i32 %1170, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.1.i.i.1, label %1176, label %1171, !dbg !443

1171:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.1
  %1172 = lshr i32 %1169, 16, !dbg !444
  %1173 = and i32 %1172, 1, !dbg !445
  %1174 = add nsw i32 %1169, 32767, !dbg !446
  %1175 = add nsw i32 %1174, %1173, !dbg !447
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.1, !dbg !448

1176:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.1
  %1177 = and i32 %1169, 65535, !dbg !449
  %.not6.i.i.i.i62.1.1.i.i.1 = icmp eq i32 %1177, 0, !dbg !450
  %1178 = or i32 %1169, 65536
  %1179 = select i1 %.not6.i.i.i.i62.1.1.i.i.1, i32 %1169, i32 %1178, !dbg !451
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.1, !dbg !450

_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.1: ; preds = %1176, %1171
  %.sroa.0.0.i.i.i.i61.1.1.i.i.1 = phi i32 [ %1175, %1171 ], [ %1179, %1176 ], !dbg !452
  %1180 = lshr i32 %.sroa.0.0.i.i.i.i61.1.1.i.i.1, 16, !dbg !453
  %1181 = trunc nuw i32 %1180 to i16, !dbg !454
  %1182 = getelementptr inbounds %struct.__hip_bfloat16, ptr %958, i64 %721, !dbg !455
  store i16 %1181, ptr %1182, align 2, !dbg !456, !tbaa !393
  %1183 = shl i32 %672, 5, !dbg !354
  %1184 = add i32 %1183, 64, !dbg !354
  %1185 = sext i32 %1184 to i64, !dbg !363
  %1186 = mul i64 %.sroa.20.0.copyload, %1185, !dbg !365
  %1187 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.17819.0.copyload, i64 %1186, !dbg !367
  %1188 = getelementptr %struct.__hip_bfloat16, ptr %1187, i64 %727, !dbg !367
  %bc1045 = bitcast <4 x float> %654 to <4 x i32>, !dbg !368
  %1189 = extractelement <4 x i32> %bc1045, i64 0, !dbg !368
  %1190 = and i32 %1189, 2139095040, !dbg !369
  %.not.i.i.i.i.i.i.2 = icmp eq i32 %1190, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.i.i.2, label %1196, label %1191, !dbg !379

1191:                                             ; preds = %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.1
  %1192 = lshr i32 %1189, 16, !dbg !380
  %1193 = and i32 %1192, 1, !dbg !381
  %1194 = add nsw i32 %1189, 32767, !dbg !382
  %1195 = add nsw i32 %1194, %1193, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.2, !dbg !384

1196:                                             ; preds = %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.1
  %1197 = and i32 %1189, 65535, !dbg !385
  %.not6.i.i.i.i.i.i.2 = icmp eq i32 %1197, 0, !dbg !386
  %1198 = or i32 %1189, 65536
  %1199 = select i1 %.not6.i.i.i.i.i.i.2, i32 %1189, i32 %1198, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.2, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.2: ; preds = %1196, %1191
  %.sroa.0.0.i.i.i.i.i.i.2 = phi i32 [ %1195, %1191 ], [ %1199, %1196 ], !dbg !388
  %1200 = lshr i32 %.sroa.0.0.i.i.i.i.i.i.2, 16, !dbg !389
  %1201 = trunc nuw i32 %1200 to i16, !dbg !390
  %1202 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %683, !dbg !391
  store i16 %1201, ptr %1202, align 2, !dbg !392, !tbaa !393
  %bc1046 = bitcast <4 x float> %654 to <4 x i32>, !dbg !397
  %1203 = extractelement <4 x i32> %bc1046, i64 1, !dbg !397
  %1204 = and i32 %1203, 2139095040, !dbg !398
  %.not.i.i.i.i52.i.i.2 = icmp eq i32 %1204, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.i.i.2, label %1210, label %1205, !dbg !403

1205:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.2
  %1206 = lshr i32 %1203, 16, !dbg !404
  %1207 = and i32 %1206, 1, !dbg !405
  %1208 = add nsw i32 %1203, 32767, !dbg !406
  %1209 = add nsw i32 %1208, %1207, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.2, !dbg !408

1210:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.2
  %1211 = and i32 %1203, 65535, !dbg !409
  %.not6.i.i.i.i54.i.i.2 = icmp eq i32 %1211, 0, !dbg !410
  %1212 = or i32 %1203, 65536
  %1213 = select i1 %.not6.i.i.i.i54.i.i.2, i32 %1203, i32 %1212, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.2, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.2: ; preds = %1210, %1205
  %.sroa.0.0.i.i.i.i53.i.i.2 = phi i32 [ %1209, %1205 ], [ %1213, %1210 ], !dbg !412
  %1214 = lshr i32 %.sroa.0.0.i.i.i.i53.i.i.2, 16, !dbg !413
  %1215 = trunc nuw i32 %1214 to i16, !dbg !414
  %1216 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %685, !dbg !415
  store i16 %1215, ptr %1216, align 2, !dbg !416, !tbaa !393
  %bc1047 = bitcast <4 x float> %654 to <4 x i32>, !dbg !417
  %1217 = extractelement <4 x i32> %bc1047, i64 2, !dbg !417
  %1218 = and i32 %1217, 2139095040, !dbg !418
  %.not.i.i.i.i56.i.i.2 = icmp eq i32 %1218, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.i.i.2, label %1224, label %1219, !dbg !423

1219:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.2
  %1220 = lshr i32 %1217, 16, !dbg !424
  %1221 = and i32 %1220, 1, !dbg !425
  %1222 = add nsw i32 %1217, 32767, !dbg !426
  %1223 = add nsw i32 %1222, %1221, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.2, !dbg !428

1224:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.2
  %1225 = and i32 %1217, 65535, !dbg !429
  %.not6.i.i.i.i58.i.i.2 = icmp eq i32 %1225, 0, !dbg !430
  %1226 = or i32 %1217, 65536
  %1227 = select i1 %.not6.i.i.i.i58.i.i.2, i32 %1217, i32 %1226, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.2, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.2: ; preds = %1224, %1219
  %.sroa.0.0.i.i.i.i57.i.i.2 = phi i32 [ %1223, %1219 ], [ %1227, %1224 ], !dbg !432
  %1228 = lshr i32 %.sroa.0.0.i.i.i.i57.i.i.2, 16, !dbg !433
  %1229 = trunc nuw i32 %1228 to i16, !dbg !434
  %1230 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %687, !dbg !435
  store i16 %1229, ptr %1230, align 2, !dbg !436, !tbaa !393
  %bc1048 = bitcast <4 x float> %654 to <4 x i32>, !dbg !437
  %1231 = extractelement <4 x i32> %bc1048, i64 3, !dbg !437
  %1232 = and i32 %1231, 2139095040, !dbg !438
  %.not.i.i.i.i60.i.i.2 = icmp eq i32 %1232, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.i.i.2, label %1238, label %1233, !dbg !443

1233:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.2
  %1234 = lshr i32 %1231, 16, !dbg !444
  %1235 = and i32 %1234, 1, !dbg !445
  %1236 = add nsw i32 %1231, 32767, !dbg !446
  %1237 = add nsw i32 %1236, %1235, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.2, !dbg !448

1238:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.2
  %1239 = and i32 %1231, 65535, !dbg !449
  %.not6.i.i.i.i62.i.i.2 = icmp eq i32 %1239, 0, !dbg !450
  %1240 = or i32 %1231, 65536
  %1241 = select i1 %.not6.i.i.i.i62.i.i.2, i32 %1231, i32 %1240, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.2, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.2: ; preds = %1238, %1233
  %.sroa.0.0.i.i.i.i61.i.i.2 = phi i32 [ %1237, %1233 ], [ %1241, %1238 ], !dbg !452
  %1242 = lshr i32 %.sroa.0.0.i.i.i.i61.i.i.2, 16, !dbg !453
  %1243 = trunc nuw i32 %1242 to i16, !dbg !454
  %1244 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %689, !dbg !455
  store i16 %1243, ptr %1244, align 2, !dbg !456, !tbaa !393
  %bc1049 = bitcast <4 x float> %656 to <4 x i32>, !dbg !368
  %1245 = extractelement <4 x i32> %bc1049, i64 0, !dbg !368
  %1246 = and i32 %1245, 2139095040, !dbg !369
  %.not.i.i.i.i.1.i.i.2 = icmp eq i32 %1246, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.i.i.2, label %1252, label %1247, !dbg !379

1247:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.2
  %1248 = lshr i32 %1245, 16, !dbg !380
  %1249 = and i32 %1248, 1, !dbg !381
  %1250 = add nsw i32 %1245, 32767, !dbg !382
  %1251 = add nsw i32 %1250, %1249, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.2, !dbg !384

1252:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.2
  %1253 = and i32 %1245, 65535, !dbg !385
  %.not6.i.i.i.i.1.i.i.2 = icmp eq i32 %1253, 0, !dbg !386
  %1254 = or i32 %1245, 65536
  %1255 = select i1 %.not6.i.i.i.i.1.i.i.2, i32 %1245, i32 %1254, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.2, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.2: ; preds = %1252, %1247
  %.sroa.0.0.i.i.i.i.1.i.i.2 = phi i32 [ %1251, %1247 ], [ %1255, %1252 ], !dbg !388
  %1256 = lshr i32 %.sroa.0.0.i.i.i.i.1.i.i.2, 16, !dbg !389
  %1257 = trunc nuw i32 %1256 to i16, !dbg !390
  %1258 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %691, !dbg !391
  store i16 %1257, ptr %1258, align 2, !dbg !392, !tbaa !393
  %bc1050 = bitcast <4 x float> %656 to <4 x i32>, !dbg !397
  %1259 = extractelement <4 x i32> %bc1050, i64 1, !dbg !397
  %1260 = and i32 %1259, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.i.i.2 = icmp eq i32 %1260, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.i.i.2, label %1266, label %1261, !dbg !403

1261:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.2
  %1262 = lshr i32 %1259, 16, !dbg !404
  %1263 = and i32 %1262, 1, !dbg !405
  %1264 = add nsw i32 %1259, 32767, !dbg !406
  %1265 = add nsw i32 %1264, %1263, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.2, !dbg !408

1266:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.2
  %1267 = and i32 %1259, 65535, !dbg !409
  %.not6.i.i.i.i54.1.i.i.2 = icmp eq i32 %1267, 0, !dbg !410
  %1268 = or i32 %1259, 65536
  %1269 = select i1 %.not6.i.i.i.i54.1.i.i.2, i32 %1259, i32 %1268, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.2, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.2: ; preds = %1266, %1261
  %.sroa.0.0.i.i.i.i53.1.i.i.2 = phi i32 [ %1265, %1261 ], [ %1269, %1266 ], !dbg !412
  %1270 = lshr i32 %.sroa.0.0.i.i.i.i53.1.i.i.2, 16, !dbg !413
  %1271 = trunc nuw i32 %1270 to i16, !dbg !414
  %1272 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %693, !dbg !415
  store i16 %1271, ptr %1272, align 2, !dbg !416, !tbaa !393
  %bc1051 = bitcast <4 x float> %656 to <4 x i32>, !dbg !417
  %1273 = extractelement <4 x i32> %bc1051, i64 2, !dbg !417
  %1274 = and i32 %1273, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.i.i.2 = icmp eq i32 %1274, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.i.i.2, label %1280, label %1275, !dbg !423

1275:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.2
  %1276 = lshr i32 %1273, 16, !dbg !424
  %1277 = and i32 %1276, 1, !dbg !425
  %1278 = add nsw i32 %1273, 32767, !dbg !426
  %1279 = add nsw i32 %1278, %1277, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.2, !dbg !428

1280:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.2
  %1281 = and i32 %1273, 65535, !dbg !429
  %.not6.i.i.i.i58.1.i.i.2 = icmp eq i32 %1281, 0, !dbg !430
  %1282 = or i32 %1273, 65536
  %1283 = select i1 %.not6.i.i.i.i58.1.i.i.2, i32 %1273, i32 %1282, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.2, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.2: ; preds = %1280, %1275
  %.sroa.0.0.i.i.i.i57.1.i.i.2 = phi i32 [ %1279, %1275 ], [ %1283, %1280 ], !dbg !432
  %1284 = lshr i32 %.sroa.0.0.i.i.i.i57.1.i.i.2, 16, !dbg !433
  %1285 = trunc nuw i32 %1284 to i16, !dbg !434
  %1286 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %695, !dbg !435
  store i16 %1285, ptr %1286, align 2, !dbg !436, !tbaa !393
  %bc1052 = bitcast <4 x float> %656 to <4 x i32>, !dbg !437
  %1287 = extractelement <4 x i32> %bc1052, i64 3, !dbg !437
  %1288 = and i32 %1287, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.i.i.2 = icmp eq i32 %1288, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.i.i.2, label %1294, label %1289, !dbg !443

1289:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.2
  %1290 = lshr i32 %1287, 16, !dbg !444
  %1291 = and i32 %1290, 1, !dbg !445
  %1292 = add nsw i32 %1287, 32767, !dbg !446
  %1293 = add nsw i32 %1292, %1291, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.2, !dbg !448

1294:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.2
  %1295 = and i32 %1287, 65535, !dbg !449
  %.not6.i.i.i.i62.1.i.i.2 = icmp eq i32 %1295, 0, !dbg !450
  %1296 = or i32 %1287, 65536
  %1297 = select i1 %.not6.i.i.i.i62.1.i.i.2, i32 %1287, i32 %1296, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.2, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.2: ; preds = %1294, %1289
  %.sroa.0.0.i.i.i.i61.1.i.i.2 = phi i32 [ %1293, %1289 ], [ %1297, %1294 ], !dbg !452
  %1298 = lshr i32 %.sroa.0.0.i.i.i.i61.1.i.i.2, 16, !dbg !453
  %1299 = trunc nuw i32 %1298 to i16, !dbg !454
  %1300 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %697, !dbg !455
  store i16 %1299, ptr %1300, align 2, !dbg !456, !tbaa !393
  %bc1053 = bitcast <4 x float> %658 to <4 x i32>, !dbg !368
  %1301 = extractelement <4 x i32> %bc1053, i64 0, !dbg !368
  %1302 = and i32 %1301, 2139095040, !dbg !369
  %.not.i.i.i.i.171.i.i.2 = icmp eq i32 %1302, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.171.i.i.2, label %1308, label %1303, !dbg !379

1303:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.2
  %1304 = lshr i32 %1301, 16, !dbg !380
  %1305 = and i32 %1304, 1, !dbg !381
  %1306 = add nsw i32 %1301, 32767, !dbg !382
  %1307 = add nsw i32 %1306, %1305, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.2, !dbg !384

1308:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.2
  %1309 = and i32 %1301, 65535, !dbg !385
  %.not6.i.i.i.i.172.i.i.2 = icmp eq i32 %1309, 0, !dbg !386
  %1310 = or i32 %1301, 65536
  %1311 = select i1 %.not6.i.i.i.i.172.i.i.2, i32 %1301, i32 %1310, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.2, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.2: ; preds = %1308, %1303
  %.sroa.0.0.i.i.i.i.173.i.i.2 = phi i32 [ %1307, %1303 ], [ %1311, %1308 ], !dbg !388
  %1312 = lshr i32 %.sroa.0.0.i.i.i.i.173.i.i.2, 16, !dbg !389
  %1313 = trunc nuw i32 %1312 to i16, !dbg !390
  %1314 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %707, !dbg !391
  store i16 %1313, ptr %1314, align 2, !dbg !392, !tbaa !393
  %bc1054 = bitcast <4 x float> %658 to <4 x i32>, !dbg !397
  %1315 = extractelement <4 x i32> %bc1054, i64 1, !dbg !397
  %1316 = and i32 %1315, 2139095040, !dbg !398
  %.not.i.i.i.i52.175.i.i.2 = icmp eq i32 %1316, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.175.i.i.2, label %1322, label %1317, !dbg !403

1317:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.2
  %1318 = lshr i32 %1315, 16, !dbg !404
  %1319 = and i32 %1318, 1, !dbg !405
  %1320 = add nsw i32 %1315, 32767, !dbg !406
  %1321 = add nsw i32 %1320, %1319, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.2, !dbg !408

1322:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.2
  %1323 = and i32 %1315, 65535, !dbg !409
  %.not6.i.i.i.i54.177.i.i.2 = icmp eq i32 %1323, 0, !dbg !410
  %1324 = or i32 %1315, 65536
  %1325 = select i1 %.not6.i.i.i.i54.177.i.i.2, i32 %1315, i32 %1324, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.2, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.2: ; preds = %1322, %1317
  %.sroa.0.0.i.i.i.i53.178.i.i.2 = phi i32 [ %1321, %1317 ], [ %1325, %1322 ], !dbg !412
  %1326 = lshr i32 %.sroa.0.0.i.i.i.i53.178.i.i.2, 16, !dbg !413
  %1327 = trunc nuw i32 %1326 to i16, !dbg !414
  %1328 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %709, !dbg !415
  store i16 %1327, ptr %1328, align 2, !dbg !416, !tbaa !393
  %bc1055 = bitcast <4 x float> %658 to <4 x i32>, !dbg !417
  %1329 = extractelement <4 x i32> %bc1055, i64 2, !dbg !417
  %1330 = and i32 %1329, 2139095040, !dbg !418
  %.not.i.i.i.i56.180.i.i.2 = icmp eq i32 %1330, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.180.i.i.2, label %1336, label %1331, !dbg !423

1331:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.2
  %1332 = lshr i32 %1329, 16, !dbg !424
  %1333 = and i32 %1332, 1, !dbg !425
  %1334 = add nsw i32 %1329, 32767, !dbg !426
  %1335 = add nsw i32 %1334, %1333, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.2, !dbg !428

1336:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.2
  %1337 = and i32 %1329, 65535, !dbg !429
  %.not6.i.i.i.i58.182.i.i.2 = icmp eq i32 %1337, 0, !dbg !430
  %1338 = or i32 %1329, 65536
  %1339 = select i1 %.not6.i.i.i.i58.182.i.i.2, i32 %1329, i32 %1338, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.2, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.2: ; preds = %1336, %1331
  %.sroa.0.0.i.i.i.i57.183.i.i.2 = phi i32 [ %1335, %1331 ], [ %1339, %1336 ], !dbg !432
  %1340 = lshr i32 %.sroa.0.0.i.i.i.i57.183.i.i.2, 16, !dbg !433
  %1341 = trunc nuw i32 %1340 to i16, !dbg !434
  %1342 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %711, !dbg !435
  store i16 %1341, ptr %1342, align 2, !dbg !436, !tbaa !393
  %bc1056 = bitcast <4 x float> %658 to <4 x i32>, !dbg !437
  %1343 = extractelement <4 x i32> %bc1056, i64 3, !dbg !437
  %1344 = and i32 %1343, 2139095040, !dbg !438
  %.not.i.i.i.i60.185.i.i.2 = icmp eq i32 %1344, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.185.i.i.2, label %1350, label %1345, !dbg !443

1345:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.2
  %1346 = lshr i32 %1343, 16, !dbg !444
  %1347 = and i32 %1346, 1, !dbg !445
  %1348 = add nsw i32 %1343, 32767, !dbg !446
  %1349 = add nsw i32 %1348, %1347, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.2, !dbg !448

1350:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.2
  %1351 = and i32 %1343, 65535, !dbg !449
  %.not6.i.i.i.i62.187.i.i.2 = icmp eq i32 %1351, 0, !dbg !450
  %1352 = or i32 %1343, 65536
  %1353 = select i1 %.not6.i.i.i.i62.187.i.i.2, i32 %1343, i32 %1352, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.2, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.2: ; preds = %1350, %1345
  %.sroa.0.0.i.i.i.i61.188.i.i.2 = phi i32 [ %1349, %1345 ], [ %1353, %1350 ], !dbg !452
  %1354 = lshr i32 %.sroa.0.0.i.i.i.i61.188.i.i.2, 16, !dbg !453
  %1355 = trunc nuw i32 %1354 to i16, !dbg !454
  %1356 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %713, !dbg !455
  store i16 %1355, ptr %1356, align 2, !dbg !456, !tbaa !393
  %bc1057 = bitcast <4 x float> %660 to <4 x i32>, !dbg !368
  %1357 = extractelement <4 x i32> %bc1057, i64 0, !dbg !368
  %1358 = and i32 %1357, 2139095040, !dbg !369
  %.not.i.i.i.i.1.1.i.i.2 = icmp eq i32 %1358, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.1.i.i.2, label %1364, label %1359, !dbg !379

1359:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.2
  %1360 = lshr i32 %1357, 16, !dbg !380
  %1361 = and i32 %1360, 1, !dbg !381
  %1362 = add nsw i32 %1357, 32767, !dbg !382
  %1363 = add nsw i32 %1362, %1361, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.2, !dbg !384

1364:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.2
  %1365 = and i32 %1357, 65535, !dbg !385
  %.not6.i.i.i.i.1.1.i.i.2 = icmp eq i32 %1365, 0, !dbg !386
  %1366 = or i32 %1357, 65536
  %1367 = select i1 %.not6.i.i.i.i.1.1.i.i.2, i32 %1357, i32 %1366, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.2, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.2: ; preds = %1364, %1359
  %.sroa.0.0.i.i.i.i.1.1.i.i.2 = phi i32 [ %1363, %1359 ], [ %1367, %1364 ], !dbg !388
  %1368 = lshr i32 %.sroa.0.0.i.i.i.i.1.1.i.i.2, 16, !dbg !389
  %1369 = trunc nuw i32 %1368 to i16, !dbg !390
  %1370 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %715, !dbg !391
  store i16 %1369, ptr %1370, align 2, !dbg !392, !tbaa !393
  %bc1058 = bitcast <4 x float> %660 to <4 x i32>, !dbg !397
  %1371 = extractelement <4 x i32> %bc1058, i64 1, !dbg !397
  %1372 = and i32 %1371, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.1.i.i.2 = icmp eq i32 %1372, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.1.i.i.2, label %1378, label %1373, !dbg !403

1373:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.2
  %1374 = lshr i32 %1371, 16, !dbg !404
  %1375 = and i32 %1374, 1, !dbg !405
  %1376 = add nsw i32 %1371, 32767, !dbg !406
  %1377 = add nsw i32 %1376, %1375, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.2, !dbg !408

1378:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.2
  %1379 = and i32 %1371, 65535, !dbg !409
  %.not6.i.i.i.i54.1.1.i.i.2 = icmp eq i32 %1379, 0, !dbg !410
  %1380 = or i32 %1371, 65536
  %1381 = select i1 %.not6.i.i.i.i54.1.1.i.i.2, i32 %1371, i32 %1380, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.2, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.2: ; preds = %1378, %1373
  %.sroa.0.0.i.i.i.i53.1.1.i.i.2 = phi i32 [ %1377, %1373 ], [ %1381, %1378 ], !dbg !412
  %1382 = lshr i32 %.sroa.0.0.i.i.i.i53.1.1.i.i.2, 16, !dbg !413
  %1383 = trunc nuw i32 %1382 to i16, !dbg !414
  %1384 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %717, !dbg !415
  store i16 %1383, ptr %1384, align 2, !dbg !416, !tbaa !393
  %bc1059 = bitcast <4 x float> %660 to <4 x i32>, !dbg !417
  %1385 = extractelement <4 x i32> %bc1059, i64 2, !dbg !417
  %1386 = and i32 %1385, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.1.i.i.2 = icmp eq i32 %1386, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.1.i.i.2, label %1392, label %1387, !dbg !423

1387:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.2
  %1388 = lshr i32 %1385, 16, !dbg !424
  %1389 = and i32 %1388, 1, !dbg !425
  %1390 = add nsw i32 %1385, 32767, !dbg !426
  %1391 = add nsw i32 %1390, %1389, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.2, !dbg !428

1392:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.2
  %1393 = and i32 %1385, 65535, !dbg !429
  %.not6.i.i.i.i58.1.1.i.i.2 = icmp eq i32 %1393, 0, !dbg !430
  %1394 = or i32 %1385, 65536
  %1395 = select i1 %.not6.i.i.i.i58.1.1.i.i.2, i32 %1385, i32 %1394, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.2, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.2: ; preds = %1392, %1387
  %.sroa.0.0.i.i.i.i57.1.1.i.i.2 = phi i32 [ %1391, %1387 ], [ %1395, %1392 ], !dbg !432
  %1396 = lshr i32 %.sroa.0.0.i.i.i.i57.1.1.i.i.2, 16, !dbg !433
  %1397 = trunc nuw i32 %1396 to i16, !dbg !434
  %1398 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %719, !dbg !435
  store i16 %1397, ptr %1398, align 2, !dbg !436, !tbaa !393
  %bc1060 = bitcast <4 x float> %660 to <4 x i32>, !dbg !437
  %1399 = extractelement <4 x i32> %bc1060, i64 3, !dbg !437
  %1400 = and i32 %1399, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.1.i.i.2 = icmp eq i32 %1400, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.1.i.i.2, label %1406, label %1401, !dbg !443

1401:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.2
  %1402 = lshr i32 %1399, 16, !dbg !444
  %1403 = and i32 %1402, 1, !dbg !445
  %1404 = add nsw i32 %1399, 32767, !dbg !446
  %1405 = add nsw i32 %1404, %1403, !dbg !447
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.2, !dbg !448

1406:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.2
  %1407 = and i32 %1399, 65535, !dbg !449
  %.not6.i.i.i.i62.1.1.i.i.2 = icmp eq i32 %1407, 0, !dbg !450
  %1408 = or i32 %1399, 65536
  %1409 = select i1 %.not6.i.i.i.i62.1.1.i.i.2, i32 %1399, i32 %1408, !dbg !451
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.2, !dbg !450

_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.2: ; preds = %1406, %1401
  %.sroa.0.0.i.i.i.i61.1.1.i.i.2 = phi i32 [ %1405, %1401 ], [ %1409, %1406 ], !dbg !452
  %1410 = lshr i32 %.sroa.0.0.i.i.i.i61.1.1.i.i.2, 16, !dbg !453
  %1411 = trunc nuw i32 %1410 to i16, !dbg !454
  %1412 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1188, i64 %721, !dbg !455
  store i16 %1411, ptr %1412, align 2, !dbg !456, !tbaa !393
  %1413 = getelementptr %struct.__hip_bfloat16, ptr %1187, i64 %957, !dbg !367
  %bc1061 = bitcast <4 x float> %662 to <4 x i32>, !dbg !368
  %1414 = extractelement <4 x i32> %bc1061, i64 0, !dbg !368
  %1415 = and i32 %1414, 2139095040, !dbg !369
  %.not.i.i.i.i.i.i.3 = icmp eq i32 %1415, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.i.i.3, label %1421, label %1416, !dbg !379

1416:                                             ; preds = %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.2
  %1417 = lshr i32 %1414, 16, !dbg !380
  %1418 = and i32 %1417, 1, !dbg !381
  %1419 = add nsw i32 %1414, 32767, !dbg !382
  %1420 = add nsw i32 %1419, %1418, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.3, !dbg !384

1421:                                             ; preds = %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.2
  %1422 = and i32 %1414, 65535, !dbg !385
  %.not6.i.i.i.i.i.i.3 = icmp eq i32 %1422, 0, !dbg !386
  %1423 = or i32 %1414, 65536
  %1424 = select i1 %.not6.i.i.i.i.i.i.3, i32 %1414, i32 %1423, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.3, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.3: ; preds = %1421, %1416
  %.sroa.0.0.i.i.i.i.i.i.3 = phi i32 [ %1420, %1416 ], [ %1424, %1421 ], !dbg !388
  %1425 = lshr i32 %.sroa.0.0.i.i.i.i.i.i.3, 16, !dbg !389
  %1426 = trunc nuw i32 %1425 to i16, !dbg !390
  %1427 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %683, !dbg !391
  store i16 %1426, ptr %1427, align 2, !dbg !392, !tbaa !393
  %bc1062 = bitcast <4 x float> %662 to <4 x i32>, !dbg !397
  %1428 = extractelement <4 x i32> %bc1062, i64 1, !dbg !397
  %1429 = and i32 %1428, 2139095040, !dbg !398
  %.not.i.i.i.i52.i.i.3 = icmp eq i32 %1429, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.i.i.3, label %1435, label %1430, !dbg !403

1430:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.3
  %1431 = lshr i32 %1428, 16, !dbg !404
  %1432 = and i32 %1431, 1, !dbg !405
  %1433 = add nsw i32 %1428, 32767, !dbg !406
  %1434 = add nsw i32 %1433, %1432, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.3, !dbg !408

1435:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.i.i.3
  %1436 = and i32 %1428, 65535, !dbg !409
  %.not6.i.i.i.i54.i.i.3 = icmp eq i32 %1436, 0, !dbg !410
  %1437 = or i32 %1428, 65536
  %1438 = select i1 %.not6.i.i.i.i54.i.i.3, i32 %1428, i32 %1437, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.3, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.3: ; preds = %1435, %1430
  %.sroa.0.0.i.i.i.i53.i.i.3 = phi i32 [ %1434, %1430 ], [ %1438, %1435 ], !dbg !412
  %1439 = lshr i32 %.sroa.0.0.i.i.i.i53.i.i.3, 16, !dbg !413
  %1440 = trunc nuw i32 %1439 to i16, !dbg !414
  %1441 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %685, !dbg !415
  store i16 %1440, ptr %1441, align 2, !dbg !416, !tbaa !393
  %bc1063 = bitcast <4 x float> %662 to <4 x i32>, !dbg !417
  %1442 = extractelement <4 x i32> %bc1063, i64 2, !dbg !417
  %1443 = and i32 %1442, 2139095040, !dbg !418
  %.not.i.i.i.i56.i.i.3 = icmp eq i32 %1443, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.i.i.3, label %1449, label %1444, !dbg !423

1444:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.3
  %1445 = lshr i32 %1442, 16, !dbg !424
  %1446 = and i32 %1445, 1, !dbg !425
  %1447 = add nsw i32 %1442, 32767, !dbg !426
  %1448 = add nsw i32 %1447, %1446, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.3, !dbg !428

1449:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.i.i.3
  %1450 = and i32 %1442, 65535, !dbg !429
  %.not6.i.i.i.i58.i.i.3 = icmp eq i32 %1450, 0, !dbg !430
  %1451 = or i32 %1442, 65536
  %1452 = select i1 %.not6.i.i.i.i58.i.i.3, i32 %1442, i32 %1451, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.3, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.3: ; preds = %1449, %1444
  %.sroa.0.0.i.i.i.i57.i.i.3 = phi i32 [ %1448, %1444 ], [ %1452, %1449 ], !dbg !432
  %1453 = lshr i32 %.sroa.0.0.i.i.i.i57.i.i.3, 16, !dbg !433
  %1454 = trunc nuw i32 %1453 to i16, !dbg !434
  %1455 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %687, !dbg !435
  store i16 %1454, ptr %1455, align 2, !dbg !436, !tbaa !393
  %bc1064 = bitcast <4 x float> %662 to <4 x i32>, !dbg !437
  %1456 = extractelement <4 x i32> %bc1064, i64 3, !dbg !437
  %1457 = and i32 %1456, 2139095040, !dbg !438
  %.not.i.i.i.i60.i.i.3 = icmp eq i32 %1457, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.i.i.3, label %1463, label %1458, !dbg !443

1458:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.3
  %1459 = lshr i32 %1456, 16, !dbg !444
  %1460 = and i32 %1459, 1, !dbg !445
  %1461 = add nsw i32 %1456, 32767, !dbg !446
  %1462 = add nsw i32 %1461, %1460, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.3, !dbg !448

1463:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.i.i.3
  %1464 = and i32 %1456, 65535, !dbg !449
  %.not6.i.i.i.i62.i.i.3 = icmp eq i32 %1464, 0, !dbg !450
  %1465 = or i32 %1456, 65536
  %1466 = select i1 %.not6.i.i.i.i62.i.i.3, i32 %1456, i32 %1465, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.3, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.3: ; preds = %1463, %1458
  %.sroa.0.0.i.i.i.i61.i.i.3 = phi i32 [ %1462, %1458 ], [ %1466, %1463 ], !dbg !452
  %1467 = lshr i32 %.sroa.0.0.i.i.i.i61.i.i.3, 16, !dbg !453
  %1468 = trunc nuw i32 %1467 to i16, !dbg !454
  %1469 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %689, !dbg !455
  store i16 %1468, ptr %1469, align 2, !dbg !456, !tbaa !393
  %bc1065 = bitcast <4 x float> %664 to <4 x i32>, !dbg !368
  %1470 = extractelement <4 x i32> %bc1065, i64 0, !dbg !368
  %1471 = and i32 %1470, 2139095040, !dbg !369
  %.not.i.i.i.i.1.i.i.3 = icmp eq i32 %1471, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.i.i.3, label %1477, label %1472, !dbg !379

1472:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.3
  %1473 = lshr i32 %1470, 16, !dbg !380
  %1474 = and i32 %1473, 1, !dbg !381
  %1475 = add nsw i32 %1470, 32767, !dbg !382
  %1476 = add nsw i32 %1475, %1474, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.3, !dbg !384

1477:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.i.i.3
  %1478 = and i32 %1470, 65535, !dbg !385
  %.not6.i.i.i.i.1.i.i.3 = icmp eq i32 %1478, 0, !dbg !386
  %1479 = or i32 %1470, 65536
  %1480 = select i1 %.not6.i.i.i.i.1.i.i.3, i32 %1470, i32 %1479, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.3, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.3: ; preds = %1477, %1472
  %.sroa.0.0.i.i.i.i.1.i.i.3 = phi i32 [ %1476, %1472 ], [ %1480, %1477 ], !dbg !388
  %1481 = lshr i32 %.sroa.0.0.i.i.i.i.1.i.i.3, 16, !dbg !389
  %1482 = trunc nuw i32 %1481 to i16, !dbg !390
  %1483 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %691, !dbg !391
  store i16 %1482, ptr %1483, align 2, !dbg !392, !tbaa !393
  %bc1066 = bitcast <4 x float> %664 to <4 x i32>, !dbg !397
  %1484 = extractelement <4 x i32> %bc1066, i64 1, !dbg !397
  %1485 = and i32 %1484, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.i.i.3 = icmp eq i32 %1485, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.i.i.3, label %1491, label %1486, !dbg !403

1486:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.3
  %1487 = lshr i32 %1484, 16, !dbg !404
  %1488 = and i32 %1487, 1, !dbg !405
  %1489 = add nsw i32 %1484, 32767, !dbg !406
  %1490 = add nsw i32 %1489, %1488, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.3, !dbg !408

1491:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.i.i.3
  %1492 = and i32 %1484, 65535, !dbg !409
  %.not6.i.i.i.i54.1.i.i.3 = icmp eq i32 %1492, 0, !dbg !410
  %1493 = or i32 %1484, 65536
  %1494 = select i1 %.not6.i.i.i.i54.1.i.i.3, i32 %1484, i32 %1493, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.3, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.3: ; preds = %1491, %1486
  %.sroa.0.0.i.i.i.i53.1.i.i.3 = phi i32 [ %1490, %1486 ], [ %1494, %1491 ], !dbg !412
  %1495 = lshr i32 %.sroa.0.0.i.i.i.i53.1.i.i.3, 16, !dbg !413
  %1496 = trunc nuw i32 %1495 to i16, !dbg !414
  %1497 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %693, !dbg !415
  store i16 %1496, ptr %1497, align 2, !dbg !416, !tbaa !393
  %bc1067 = bitcast <4 x float> %664 to <4 x i32>, !dbg !417
  %1498 = extractelement <4 x i32> %bc1067, i64 2, !dbg !417
  %1499 = and i32 %1498, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.i.i.3 = icmp eq i32 %1499, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.i.i.3, label %1505, label %1500, !dbg !423

1500:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.3
  %1501 = lshr i32 %1498, 16, !dbg !424
  %1502 = and i32 %1501, 1, !dbg !425
  %1503 = add nsw i32 %1498, 32767, !dbg !426
  %1504 = add nsw i32 %1503, %1502, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.3, !dbg !428

1505:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.i.i.3
  %1506 = and i32 %1498, 65535, !dbg !429
  %.not6.i.i.i.i58.1.i.i.3 = icmp eq i32 %1506, 0, !dbg !430
  %1507 = or i32 %1498, 65536
  %1508 = select i1 %.not6.i.i.i.i58.1.i.i.3, i32 %1498, i32 %1507, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.3, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.3: ; preds = %1505, %1500
  %.sroa.0.0.i.i.i.i57.1.i.i.3 = phi i32 [ %1504, %1500 ], [ %1508, %1505 ], !dbg !432
  %1509 = lshr i32 %.sroa.0.0.i.i.i.i57.1.i.i.3, 16, !dbg !433
  %1510 = trunc nuw i32 %1509 to i16, !dbg !434
  %1511 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %695, !dbg !435
  store i16 %1510, ptr %1511, align 2, !dbg !436, !tbaa !393
  %bc1068 = bitcast <4 x float> %664 to <4 x i32>, !dbg !437
  %1512 = extractelement <4 x i32> %bc1068, i64 3, !dbg !437
  %1513 = and i32 %1512, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.i.i.3 = icmp eq i32 %1513, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.i.i.3, label %1519, label %1514, !dbg !443

1514:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.3
  %1515 = lshr i32 %1512, 16, !dbg !444
  %1516 = and i32 %1515, 1, !dbg !445
  %1517 = add nsw i32 %1512, 32767, !dbg !446
  %1518 = add nsw i32 %1517, %1516, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.3, !dbg !448

1519:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.i.i.3
  %1520 = and i32 %1512, 65535, !dbg !449
  %.not6.i.i.i.i62.1.i.i.3 = icmp eq i32 %1520, 0, !dbg !450
  %1521 = or i32 %1512, 65536
  %1522 = select i1 %.not6.i.i.i.i62.1.i.i.3, i32 %1512, i32 %1521, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.3, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.3: ; preds = %1519, %1514
  %.sroa.0.0.i.i.i.i61.1.i.i.3 = phi i32 [ %1518, %1514 ], [ %1522, %1519 ], !dbg !452
  %1523 = lshr i32 %.sroa.0.0.i.i.i.i61.1.i.i.3, 16, !dbg !453
  %1524 = trunc nuw i32 %1523 to i16, !dbg !454
  %1525 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %697, !dbg !455
  store i16 %1524, ptr %1525, align 2, !dbg !456, !tbaa !393
  %bc1069 = bitcast <4 x float> %666 to <4 x i32>, !dbg !368
  %1526 = extractelement <4 x i32> %bc1069, i64 0, !dbg !368
  %1527 = and i32 %1526, 2139095040, !dbg !369
  %.not.i.i.i.i.171.i.i.3 = icmp eq i32 %1527, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.171.i.i.3, label %1533, label %1528, !dbg !379

1528:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.3
  %1529 = lshr i32 %1526, 16, !dbg !380
  %1530 = and i32 %1529, 1, !dbg !381
  %1531 = add nsw i32 %1526, 32767, !dbg !382
  %1532 = add nsw i32 %1531, %1530, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.3, !dbg !384

1533:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.1.i.i.3
  %1534 = and i32 %1526, 65535, !dbg !385
  %.not6.i.i.i.i.172.i.i.3 = icmp eq i32 %1534, 0, !dbg !386
  %1535 = or i32 %1526, 65536
  %1536 = select i1 %.not6.i.i.i.i.172.i.i.3, i32 %1526, i32 %1535, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.3, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.3: ; preds = %1533, %1528
  %.sroa.0.0.i.i.i.i.173.i.i.3 = phi i32 [ %1532, %1528 ], [ %1536, %1533 ], !dbg !388
  %1537 = lshr i32 %.sroa.0.0.i.i.i.i.173.i.i.3, 16, !dbg !389
  %1538 = trunc nuw i32 %1537 to i16, !dbg !390
  %1539 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %707, !dbg !391
  store i16 %1538, ptr %1539, align 2, !dbg !392, !tbaa !393
  %bc1070 = bitcast <4 x float> %666 to <4 x i32>, !dbg !397
  %1540 = extractelement <4 x i32> %bc1070, i64 1, !dbg !397
  %1541 = and i32 %1540, 2139095040, !dbg !398
  %.not.i.i.i.i52.175.i.i.3 = icmp eq i32 %1541, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.175.i.i.3, label %1547, label %1542, !dbg !403

1542:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.3
  %1543 = lshr i32 %1540, 16, !dbg !404
  %1544 = and i32 %1543, 1, !dbg !405
  %1545 = add nsw i32 %1540, 32767, !dbg !406
  %1546 = add nsw i32 %1545, %1544, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.3, !dbg !408

1547:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.176.i.i.3
  %1548 = and i32 %1540, 65535, !dbg !409
  %.not6.i.i.i.i54.177.i.i.3 = icmp eq i32 %1548, 0, !dbg !410
  %1549 = or i32 %1540, 65536
  %1550 = select i1 %.not6.i.i.i.i54.177.i.i.3, i32 %1540, i32 %1549, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.3, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.3: ; preds = %1547, %1542
  %.sroa.0.0.i.i.i.i53.178.i.i.3 = phi i32 [ %1546, %1542 ], [ %1550, %1547 ], !dbg !412
  %1551 = lshr i32 %.sroa.0.0.i.i.i.i53.178.i.i.3, 16, !dbg !413
  %1552 = trunc nuw i32 %1551 to i16, !dbg !414
  %1553 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %709, !dbg !415
  store i16 %1552, ptr %1553, align 2, !dbg !416, !tbaa !393
  %bc1071 = bitcast <4 x float> %666 to <4 x i32>, !dbg !417
  %1554 = extractelement <4 x i32> %bc1071, i64 2, !dbg !417
  %1555 = and i32 %1554, 2139095040, !dbg !418
  %.not.i.i.i.i56.180.i.i.3 = icmp eq i32 %1555, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.180.i.i.3, label %1561, label %1556, !dbg !423

1556:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.3
  %1557 = lshr i32 %1554, 16, !dbg !424
  %1558 = and i32 %1557, 1, !dbg !425
  %1559 = add nsw i32 %1554, 32767, !dbg !426
  %1560 = add nsw i32 %1559, %1558, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.3, !dbg !428

1561:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.181.i.i.3
  %1562 = and i32 %1554, 65535, !dbg !429
  %.not6.i.i.i.i58.182.i.i.3 = icmp eq i32 %1562, 0, !dbg !430
  %1563 = or i32 %1554, 65536
  %1564 = select i1 %.not6.i.i.i.i58.182.i.i.3, i32 %1554, i32 %1563, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.3, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.3: ; preds = %1561, %1556
  %.sroa.0.0.i.i.i.i57.183.i.i.3 = phi i32 [ %1560, %1556 ], [ %1564, %1561 ], !dbg !432
  %1565 = lshr i32 %.sroa.0.0.i.i.i.i57.183.i.i.3, 16, !dbg !433
  %1566 = trunc nuw i32 %1565 to i16, !dbg !434
  %1567 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %711, !dbg !435
  store i16 %1566, ptr %1567, align 2, !dbg !436, !tbaa !393
  %bc1072 = bitcast <4 x float> %666 to <4 x i32>, !dbg !437
  %1568 = extractelement <4 x i32> %bc1072, i64 3, !dbg !437
  %1569 = and i32 %1568, 2139095040, !dbg !438
  %.not.i.i.i.i60.185.i.i.3 = icmp eq i32 %1569, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.185.i.i.3, label %1575, label %1570, !dbg !443

1570:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.3
  %1571 = lshr i32 %1568, 16, !dbg !444
  %1572 = and i32 %1571, 1, !dbg !445
  %1573 = add nsw i32 %1568, 32767, !dbg !446
  %1574 = add nsw i32 %1573, %1572, !dbg !447
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.3, !dbg !448

1575:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.186.i.i.3
  %1576 = and i32 %1568, 65535, !dbg !449
  %.not6.i.i.i.i62.187.i.i.3 = icmp eq i32 %1576, 0, !dbg !450
  %1577 = or i32 %1568, 65536
  %1578 = select i1 %.not6.i.i.i.i62.187.i.i.3, i32 %1568, i32 %1577, !dbg !451
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.3, !dbg !450

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.3: ; preds = %1575, %1570
  %.sroa.0.0.i.i.i.i61.188.i.i.3 = phi i32 [ %1574, %1570 ], [ %1578, %1575 ], !dbg !452
  %1579 = lshr i32 %.sroa.0.0.i.i.i.i61.188.i.i.3, 16, !dbg !453
  %1580 = trunc nuw i32 %1579 to i16, !dbg !454
  %1581 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %713, !dbg !455
  store i16 %1580, ptr %1581, align 2, !dbg !456, !tbaa !393
  %bc1073 = bitcast <4 x float> %668 to <4 x i32>, !dbg !368
  %1582 = extractelement <4 x i32> %bc1073, i64 0, !dbg !368
  %1583 = and i32 %1582, 2139095040, !dbg !369
  %.not.i.i.i.i.1.1.i.i.3 = icmp eq i32 %1583, 2139095040, !dbg !379
  br i1 %.not.i.i.i.i.1.1.i.i.3, label %1589, label %1584, !dbg !379

1584:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.3
  %1585 = lshr i32 %1582, 16, !dbg !380
  %1586 = and i32 %1585, 1, !dbg !381
  %1587 = add nsw i32 %1582, 32767, !dbg !382
  %1588 = add nsw i32 %1587, %1586, !dbg !383
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.3, !dbg !384

1589:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit63.189.i.i.3
  %1590 = and i32 %1582, 65535, !dbg !385
  %.not6.i.i.i.i.1.1.i.i.3 = icmp eq i32 %1590, 0, !dbg !386
  %1591 = or i32 %1582, 65536
  %1592 = select i1 %.not6.i.i.i.i.1.1.i.i.3, i32 %1582, i32 %1591, !dbg !387
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.3, !dbg !386

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.3: ; preds = %1589, %1584
  %.sroa.0.0.i.i.i.i.1.1.i.i.3 = phi i32 [ %1588, %1584 ], [ %1592, %1589 ], !dbg !388
  %1593 = lshr i32 %.sroa.0.0.i.i.i.i.1.1.i.i.3, 16, !dbg !389
  %1594 = trunc nuw i32 %1593 to i16, !dbg !390
  %1595 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %715, !dbg !391
  store i16 %1594, ptr %1595, align 2, !dbg !392, !tbaa !393
  %bc1074 = bitcast <4 x float> %668 to <4 x i32>, !dbg !397
  %1596 = extractelement <4 x i32> %bc1074, i64 1, !dbg !397
  %1597 = and i32 %1596, 2139095040, !dbg !398
  %.not.i.i.i.i52.1.1.i.i.3 = icmp eq i32 %1597, 2139095040, !dbg !403
  br i1 %.not.i.i.i.i52.1.1.i.i.3, label %1603, label %1598, !dbg !403

1598:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.3
  %1599 = lshr i32 %1596, 16, !dbg !404
  %1600 = and i32 %1599, 1, !dbg !405
  %1601 = add nsw i32 %1596, 32767, !dbg !406
  %1602 = add nsw i32 %1601, %1600, !dbg !407
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.3, !dbg !408

1603:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit.1.1.i.i.3
  %1604 = and i32 %1596, 65535, !dbg !409
  %.not6.i.i.i.i54.1.1.i.i.3 = icmp eq i32 %1604, 0, !dbg !410
  %1605 = or i32 %1596, 65536
  %1606 = select i1 %.not6.i.i.i.i54.1.1.i.i.3, i32 %1596, i32 %1605, !dbg !411
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.3, !dbg !410

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.3: ; preds = %1603, %1598
  %.sroa.0.0.i.i.i.i53.1.1.i.i.3 = phi i32 [ %1602, %1598 ], [ %1606, %1603 ], !dbg !412
  %1607 = lshr i32 %.sroa.0.0.i.i.i.i53.1.1.i.i.3, 16, !dbg !413
  %1608 = trunc nuw i32 %1607 to i16, !dbg !414
  %1609 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %717, !dbg !415
  store i16 %1608, ptr %1609, align 2, !dbg !416, !tbaa !393
  %bc1075 = bitcast <4 x float> %668 to <4 x i32>, !dbg !417
  %1610 = extractelement <4 x i32> %bc1075, i64 2, !dbg !417
  %1611 = and i32 %1610, 2139095040, !dbg !418
  %.not.i.i.i.i56.1.1.i.i.3 = icmp eq i32 %1611, 2139095040, !dbg !423
  br i1 %.not.i.i.i.i56.1.1.i.i.3, label %1617, label %1612, !dbg !423

1612:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.3
  %1613 = lshr i32 %1610, 16, !dbg !424
  %1614 = and i32 %1613, 1, !dbg !425
  %1615 = add nsw i32 %1610, 32767, !dbg !426
  %1616 = add nsw i32 %1615, %1614, !dbg !427
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.3, !dbg !428

1617:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit55.1.1.i.i.3
  %1618 = and i32 %1610, 65535, !dbg !429
  %.not6.i.i.i.i58.1.1.i.i.3 = icmp eq i32 %1618, 0, !dbg !430
  %1619 = or i32 %1610, 65536
  %1620 = select i1 %.not6.i.i.i.i58.1.1.i.i.3, i32 %1610, i32 %1619, !dbg !431
  br label %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.3, !dbg !430

_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.3: ; preds = %1617, %1612
  %.sroa.0.0.i.i.i.i57.1.1.i.i.3 = phi i32 [ %1616, %1612 ], [ %1620, %1617 ], !dbg !432
  %1621 = lshr i32 %.sroa.0.0.i.i.i.i57.1.1.i.i.3, 16, !dbg !433
  %1622 = trunc nuw i32 %1621 to i16, !dbg !434
  %1623 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %719, !dbg !435
  store i16 %1622, ptr %1623, align 2, !dbg !436, !tbaa !393
  %bc1076 = bitcast <4 x float> %668 to <4 x i32>, !dbg !437
  %1624 = extractelement <4 x i32> %bc1076, i64 3, !dbg !437
  %1625 = and i32 %1624, 2139095040, !dbg !438
  %.not.i.i.i.i60.1.1.i.i.3 = icmp eq i32 %1625, 2139095040, !dbg !443
  br i1 %.not.i.i.i.i60.1.1.i.i.3, label %1631, label %1626, !dbg !443

1626:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.3
  %1627 = lshr i32 %1624, 16, !dbg !444
  %1628 = and i32 %1627, 1, !dbg !445
  %1629 = add nsw i32 %1624, 32767, !dbg !446
  %1630 = add nsw i32 %1629, %1628, !dbg !447
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.3, !dbg !448

1631:                                             ; preds = %_ZN7kittens10base_types9convertorI14__hip_bfloat16fE7convertERKf.exit59.1.1.i.i.3
  %1632 = and i32 %1624, 65535, !dbg !449
  %.not6.i.i.i.i62.1.1.i.i.3 = icmp eq i32 %1632, 0, !dbg !450
  %1633 = or i32 %1624, 65536
  %1634 = select i1 %.not6.i.i.i.i62.1.1.i.i.3, i32 %1624, i32 %1633, !dbg !451
  br label %_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.3, !dbg !450

_ZN7kittensL5storeITkNS_5ducks2rt3allENS_2rtIfLi32ELi32ENS1_9rt_layout3colEEETkNS1_2gl3allENS_2glI14__hip_bfloat16Lin1ELin1ELin1ELin1EJEEETkNS1_5coord4tileENS_5coordIS6_EEEEvRKT0_RKT_RKT1_.exit.3: ; preds = %1631, %1626
  %.sroa.0.0.i.i.i.i61.1.1.i.i.3 = phi i32 [ %1630, %1626 ], [ %1634, %1631 ], !dbg !452
  %1635 = lshr i32 %.sroa.0.0.i.i.i.i61.1.1.i.i.3, 16, !dbg !453
  %1636 = trunc nuw i32 %1635 to i16, !dbg !454
  %1637 = getelementptr inbounds %struct.__hip_bfloat16, ptr %1413, i64 %721, !dbg !455
  store i16 %1636, ptr %1637, align 2, !dbg !456, !tbaa !393
  ret void, !dbg !457
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

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,512" "amdgpu-waves-per-eu"="2" "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts" "uniform-work-group-size"="true" "unsafe-fp-math"="true" }
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
!1 = !DIFile(filename: "256_128_64_32.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_outbf16")
!2 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!3 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)"}
!9 = distinct !DISubprogram(name: "micro_tk", scope: !1, file: !1, line: 46, type: !10, scopeLine: 46, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !15)
!13 = distinct !DISubprogram(name: "align_ptr<16>", scope: !14, file: !14, line: 228, type: !10, scopeLine: 228, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!14 = !DIFile(filename: "ThunderKittens-HIP/include/common/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!15 = distinct !DILocation(line: 252, column: 13, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 128, 64> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DILocation(line: 49, column: 44, scope: !9)
!18 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !15)
!19 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !15)
!20 = !DILocation(line: 255, column: 17, scope: !16, inlinedAt: !17)
!21 = !DILocation(line: 230, column: 30, scope: !13, inlinedAt: !22)
!22 = distinct !DILocation(line: 252, column: 13, scope: !23, inlinedAt: !24)
!23 = distinct !DISubprogram(name: "allocate<kittens::st<__hip_bfloat16, 256, 64> >", scope: !14, file: !14, line: 250, type: !10, scopeLine: 250, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = distinct !DILocation(line: 50, column: 44, scope: !9)
!25 = !DILocation(line: 231, column: 22, scope: !13, inlinedAt: !22)
!26 = !DILocation(line: 231, column: 20, scope: !13, inlinedAt: !22)
!27 = !DILocation(line: 265, column: 117, scope: !28, inlinedAt: !30)
!28 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !29, file: !29, line: 265, type: !10, scopeLine: 265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!29 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h", directory: "")
!30 = distinct !DILocation(line: 291, column: 160, scope: !31, inlinedAt: !32)
!31 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 291, type: !10, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!32 = distinct !DILocation(line: 57, column: 85, scope: !33, inlinedAt: !34)
!33 = distinct !DISubprogram(name: "warpid", scope: !14, file: !14, line: 57, type: !10, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!34 = distinct !DILocation(line: 52, column: 25, scope: !9)
!35 = !DILocation(line: 53, column: 34, scope: !9)
!36 = !DILocation(line: 271, column: 116, scope: !37, inlinedAt: !38)
!37 = distinct !DISubprogram(name: "__hip_get_block_idx_y", scope: !29, file: !29, line: 271, type: !10, scopeLine: 271, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!38 = distinct !DILocation(line: 301, column: 160, scope: !39, inlinedAt: !40)
!39 = distinct !DISubprogram(name: "__get_y", scope: !29, file: !29, line: 301, type: !10, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!40 = distinct !DILocation(line: 58, column: 17, scope: !9)
!41 = !DILocation(line: 280, column: 115, scope: !42, inlinedAt: !43)
!42 = distinct !DISubprogram(name: "__hip_get_grid_dim_x", scope: !29, file: !29, line: 280, type: !10, scopeLine: 280, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!43 = distinct !DILocation(line: 318, column: 160, scope: !44, inlinedAt: !45)
!44 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 318, type: !10, scopeLine: 318, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!45 = distinct !DILocation(line: 58, column: 30, scope: !9)
!46 = !{!47, !47, i64 0}
!47 = !{!"int", !48, i64 0}
!48 = !{!"omnipotent char", !49, i64 0}
!49 = !{!"Simple C/C++ TBAA"}
!50 = !DILocation(line: 58, column: 28, scope: !9)
!51 = !DILocation(line: 270, column: 116, scope: !52, inlinedAt: !53)
!52 = distinct !DISubprogram(name: "__hip_get_block_idx_x", scope: !29, file: !29, line: 270, type: !10, scopeLine: 270, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!53 = distinct !DILocation(line: 300, column: 160, scope: !54, inlinedAt: !55)
!54 = distinct !DISubprogram(name: "__get_x", scope: !29, file: !29, line: 300, type: !10, scopeLine: 300, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!55 = distinct !DILocation(line: 58, column: 43, scope: !9)
!56 = !DILocation(line: 58, column: 41, scope: !9)
!57 = !DILocation(line: 281, column: 115, scope: !58, inlinedAt: !59)
!58 = distinct !DISubprogram(name: "__hip_get_grid_dim_y", scope: !29, file: !29, line: 281, type: !10, scopeLine: 281, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!59 = distinct !DILocation(line: 319, column: 160, scope: !60, inlinedAt: !61)
!60 = distinct !DISubprogram(name: "__get_y", scope: !29, file: !29, line: 319, type: !10, scopeLine: 319, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!61 = distinct !DILocation(line: 59, column: 37, scope: !9)
!62 = !DILocation(line: 59, column: 35, scope: !9)
!63 = !DILocation(line: 61, column: 47, scope: !9)
!64 = !DILocation(line: 61, column: 18, scope: !9)
!65 = !DILocation(line: 61, column: 34, scope: !9)
!66 = !DILocation(line: 61, column: 23, scope: !9)
!67 = !DILocation(line: 61, column: 39, scope: !9)
!68 = !DILocation(line: 66, column: 31, scope: !9)
!69 = !DILocation(line: 67, column: 38, scope: !9)
!70 = !DILocation(line: 68, column: 44, scope: !9)
!71 = !DILocation(line: 1321, column: 10, scope: !72, inlinedAt: !74)
!72 = distinct !DISubprogram(name: "min", scope: !73, file: !73, line: 1320, type: !10, scopeLine: 1320, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!73 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/lib/clang/19/include/__clang_hip_math.h", directory: "")
!74 = distinct !DILocation(line: 68, column: 30, scope: !9)
!75 = !DILocation(line: 69, column: 44, scope: !9)
!76 = !DILocation(line: 70, column: 50, scope: !9)
!77 = !DILocation(line: 69, column: 65, scope: !9)
!78 = !DILocation(line: 69, column: 35, scope: !9)
!79 = !DILocation(line: 231, column: 34, scope: !13, inlinedAt: !22)
!80 = !DILocation(line: 57, column: 97, scope: !33, inlinedAt: !34)
!81 = !DILocation(line: 54, column: 34, scope: !9)
!82 = !DILocation(line: 20, column: 28, scope: !83, inlinedAt: !85)
!83 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !84, file: !84, line: 17, type: !10, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!84 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!85 = distinct !DILocation(line: 12, column: 5, scope: !86, inlinedAt: !88)
!86 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> > >", scope: !87, file: !87, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!87 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!88 = distinct !DILocation(line: 87, column: 5, scope: !9)
!89 = !DILocation(line: 60, column: 34, scope: !90, inlinedAt: !92)
!90 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!91 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!92 = distinct !DILocation(line: 27, column: 39, scope: !83, inlinedAt: !85)
!93 = !DILocation(line: 63, column: 54, scope: !94, inlinedAt: !96)
!94 = distinct !DISubprogram(name: "operator[]", scope: !95, file: !95, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!95 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/gl.cuh", directory: "/workdir/AMD-benchmarking-harness")
!96 = distinct !DILocation(line: 28, column: 57, scope: !83, inlinedAt: !85)
!97 = !DILocation(line: 63, column: 60, scope: !94, inlinedAt: !96)
!98 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !96)
!99 = !DILocation(line: 30, column: 24, scope: !83, inlinedAt: !85)
!100 = !DILocation(line: 44, column: 32, scope: !83, inlinedAt: !85)
!101 = !DILocation(line: 48, column: 75, scope: !83, inlinedAt: !85)
!102 = !DILocation(line: 48, column: 88, scope: !83, inlinedAt: !85)
!103 = !DILocation(line: 48, column: 68, scope: !83, inlinedAt: !85)
!104 = !DILocation(line: 92, column: 5, scope: !105, inlinedAt: !107)
!105 = distinct !DISubprogram(name: "load_global_vec4_async", scope: !106, file: !106, line: 89, type: !10, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!106 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/util/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!107 = distinct !DILocation(line: 48, column: 26, scope: !83, inlinedAt: !85)
!108 = !{i64 3065050}
!109 = !DILocation(line: 52, column: 9, scope: !83, inlinedAt: !85)
!110 = !{i64 3083124}
!111 = !DILocation(line: 103, column: 35, scope: !112, inlinedAt: !114)
!112 = distinct !DISubprogram(name: "idx", scope: !113, file: !113, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!113 = !DIFile(filename: "ThunderKittens-HIP/include/types/shared/st.cuh", directory: "/workdir/AMD-benchmarking-harness")
!114 = distinct !DILocation(line: 61, column: 34, scope: !83, inlinedAt: !85)
!115 = !DILocation(line: 104, column: 60, scope: !112, inlinedAt: !114)
!116 = !DILocation(line: 105, column: 22, scope: !112, inlinedAt: !114)
!117 = !DILocation(line: 61, column: 72, scope: !83, inlinedAt: !85)
!118 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !120)
!119 = distinct !DISubprogram(name: "store_shared_vec", scope: !106, file: !106, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!120 = distinct !DILocation(line: 61, column: 17, scope: !83, inlinedAt: !85)
!121 = !{i64 3063385}
!122 = !DILocation(line: 103, column: 35, scope: !112, inlinedAt: !123)
!123 = distinct !DILocation(line: 62, column: 34, scope: !83, inlinedAt: !85)
!124 = !DILocation(line: 104, column: 60, scope: !112, inlinedAt: !123)
!125 = !DILocation(line: 105, column: 22, scope: !112, inlinedAt: !123)
!126 = !DILocation(line: 62, column: 95, scope: !83, inlinedAt: !85)
!127 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !128)
!128 = distinct !DILocation(line: 62, column: 17, scope: !83, inlinedAt: !85)
!129 = !DILocation(line: 103, column: 46, scope: !112, inlinedAt: !114)
!130 = !DILocation(line: 65, column: 9, scope: !83, inlinedAt: !85)
!131 = !{i64 3083665}
!132 = !DILocation(line: 20, column: 28, scope: !133, inlinedAt: !134)
!133 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !84, file: !84, line: 17, type: !10, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!134 = distinct !DILocation(line: 12, column: 5, scope: !135, inlinedAt: !136)
!135 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> > >", scope: !87, file: !87, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!136 = distinct !DILocation(line: 88, column: 5, scope: !9)
!137 = !DILocation(line: 60, column: 34, scope: !138, inlinedAt: !139)
!138 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!139 = distinct !DILocation(line: 27, column: 39, scope: !133, inlinedAt: !134)
!140 = !DILocation(line: 63, column: 54, scope: !94, inlinedAt: !141)
!141 = distinct !DILocation(line: 28, column: 57, scope: !133, inlinedAt: !134)
!142 = !DILocation(line: 63, column: 60, scope: !94, inlinedAt: !141)
!143 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !141)
!144 = !DILocation(line: 30, column: 24, scope: !133, inlinedAt: !134)
!145 = !DILocation(line: 48, column: 75, scope: !133, inlinedAt: !134)
!146 = !DILocation(line: 48, column: 88, scope: !133, inlinedAt: !134)
!147 = !DILocation(line: 48, column: 68, scope: !133, inlinedAt: !134)
!148 = !DILocation(line: 92, column: 5, scope: !105, inlinedAt: !149)
!149 = distinct !DILocation(line: 48, column: 26, scope: !133, inlinedAt: !134)
!150 = !DILocation(line: 44, column: 32, scope: !133, inlinedAt: !134)
!151 = !DILocation(line: 52, column: 9, scope: !133, inlinedAt: !134)
!152 = !DILocation(line: 103, column: 35, scope: !153, inlinedAt: !154)
!153 = distinct !DISubprogram(name: "idx", scope: !113, file: !113, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!154 = distinct !DILocation(line: 61, column: 34, scope: !133, inlinedAt: !134)
!155 = !DILocation(line: 104, column: 60, scope: !153, inlinedAt: !154)
!156 = !DILocation(line: 105, column: 22, scope: !153, inlinedAt: !154)
!157 = !DILocation(line: 61, column: 72, scope: !133, inlinedAt: !134)
!158 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !159)
!159 = distinct !DILocation(line: 61, column: 17, scope: !133, inlinedAt: !134)
!160 = !DILocation(line: 103, column: 35, scope: !153, inlinedAt: !161)
!161 = distinct !DILocation(line: 62, column: 34, scope: !133, inlinedAt: !134)
!162 = !DILocation(line: 104, column: 60, scope: !153, inlinedAt: !161)
!163 = !DILocation(line: 105, column: 22, scope: !153, inlinedAt: !161)
!164 = !DILocation(line: 62, column: 95, scope: !133, inlinedAt: !134)
!165 = !DILocation(line: 27, column: 5, scope: !119, inlinedAt: !166)
!166 = distinct !DILocation(line: 62, column: 17, scope: !133, inlinedAt: !134)
!167 = !DILocation(line: 103, column: 46, scope: !153, inlinedAt: !154)
!168 = !DILocation(line: 65, column: 9, scope: !133, inlinedAt: !134)
!169 = !DILocation(line: 89, column: 5, scope: !9)
!170 = !DILocation(line: 90, column: 18, scope: !9)
!171 = !DILocation(line: 90, column: 9, scope: !9)
!172 = !DILocation(line: 91, column: 9, scope: !9)
!173 = !DILocation(line: 92, column: 5, scope: !9)
!174 = !DILocation(line: 95, column: 5, scope: !9)
!175 = !DILocation(line: 155, column: 18, scope: !9)
!176 = !DILocation(line: 155, column: 9, scope: !9)
!177 = !DILocation(line: 97, column: 35, scope: !9)
!178 = !DILocation(line: 97, column: 39, scope: !9)
!179 = !DILocation(line: 99, column: 13, scope: !9)
!180 = !DILocation(line: 26, column: 23, scope: !181, inlinedAt: !183)
!181 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !182, file: !182, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!182 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_outbf16")
!183 = distinct !DILocation(line: 77, column: 39, scope: !184, inlinedAt: !185)
!184 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !182, file: !182, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!185 = distinct !DILocation(line: 100, column: 13, scope: !9)
!186 = !DILocation(line: 102, column: 13, scope: !9)
!187 = !DILocation(line: 103, column: 9, scope: !9)
!188 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !190)
!189 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !182, file: !182, line: 127, type: !10, scopeLine: 127, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!190 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !192)
!191 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 64>, 32, 32> >", scope: !182, file: !182, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!192 = distinct !DILocation(line: 106, column: 9, scope: !9)
!193 = !{i64 4274638}
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
!218 = !DILocation(line: 111, column: 13, scope: !9)
!219 = !DILocation(line: 61, column: 18, scope: !138, inlinedAt: !220)
!220 = distinct !DILocation(line: 54, column: 39, scope: !221, inlinedAt: !222)
!221 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !182, file: !182, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!222 = distinct !DILocation(line: 112, column: 13, scope: !9)
!223 = !DILocation(line: 63, column: 68, scope: !94, inlinedAt: !224)
!224 = distinct !DILocation(line: 55, column: 24, scope: !221, inlinedAt: !222)
!225 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !224)
!226 = !DILocation(line: 32, column: 28, scope: !227, inlinedAt: !228)
!227 = distinct !DISubprogram(name: "make_srsrc", scope: !182, file: !182, line: 31, type: !10, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!228 = distinct !DILocation(line: 60, column: 19, scope: !221, inlinedAt: !222)
!229 = !DILocation(line: 26, column: 23, scope: !181, inlinedAt: !230)
!230 = distinct !DILocation(line: 77, column: 39, scope: !221, inlinedAt: !222)
!231 = !DILocation(line: 114, column: 13, scope: !9)
!232 = !DILocation(line: 115, column: 9, scope: !9)
!233 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !234)
!234 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !235)
!235 = distinct !DILocation(line: 118, column: 9, scope: !9)
!236 = !DILocation(line: 187, column: 47, scope: !191, inlinedAt: !235)
!237 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !238)
!238 = distinct !DILocation(line: 188, column: 45, scope: !191, inlinedAt: !235)
!239 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !240)
!240 = distinct !DILocation(line: 185, column: 37, scope: !191, inlinedAt: !241)
!241 = distinct !DILocation(line: 119, column: 9, scope: !9)
!242 = !DILocation(line: 187, column: 47, scope: !191, inlinedAt: !241)
!243 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !244)
!244 = distinct !DILocation(line: 188, column: 45, scope: !191, inlinedAt: !241)
!245 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !246)
!246 = distinct !DILocation(line: 185, column: 37, scope: !207, inlinedAt: !247)
!247 = distinct !DILocation(line: 120, column: 9, scope: !9)
!248 = !DILocation(line: 187, column: 47, scope: !207, inlinedAt: !247)
!249 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !250)
!250 = distinct !DILocation(line: 188, column: 45, scope: !207, inlinedAt: !247)
!251 = !DILocation(line: 129, column: 5, scope: !189, inlinedAt: !252)
!252 = distinct !DILocation(line: 185, column: 37, scope: !207, inlinedAt: !253)
!253 = distinct !DILocation(line: 121, column: 9, scope: !9)
!254 = !DILocation(line: 187, column: 47, scope: !207, inlinedAt: !253)
!255 = !DILocation(line: 494, column: 7, scope: !196, inlinedAt: !256)
!256 = distinct !DILocation(line: 188, column: 45, scope: !207, inlinedAt: !253)
!257 = !DILocation(line: 122, column: 9, scope: !9)
!258 = !DILocation(line: 123, column: 9, scope: !9)
!259 = !DILocation(line: 125, column: 9, scope: !9)
!260 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !263)
!261 = distinct !DISubprogram(name: "mfma161616", scope: !262, file: !262, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!262 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!263 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !265)
!264 = distinct !DISubprogram(name: "mma_ABt_base", scope: !262, file: !262, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!265 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !267)
!266 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !262, file: !262, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!267 = distinct !DILocation(line: 126, column: 9, scope: !9)
!268 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !269)
!269 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !270)
!270 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !267)
!271 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !272)
!272 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !273)
!273 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !274)
!274 = distinct !DILocation(line: 127, column: 9, scope: !9)
!275 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !276)
!276 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !277)
!277 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !274)
!278 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !279)
!279 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !280)
!280 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !281)
!281 = distinct !DILocation(line: 128, column: 9, scope: !9)
!282 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !283)
!283 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !284)
!284 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !281)
!285 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !286)
!286 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !287)
!287 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !288)
!288 = distinct !DILocation(line: 129, column: 9, scope: !9)
!289 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !290)
!290 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !291)
!291 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !288)
!292 = !DILocation(line: 130, column: 9, scope: !9)
!293 = !DILocation(line: 131, column: 9, scope: !9)
!294 = !DILocation(line: 132, column: 9, scope: !9)
!295 = !DILocation(line: 135, column: 13, scope: !9)
!296 = !DILocation(line: 136, column: 13, scope: !9)
!297 = !{i64 4284430}
!298 = !DILocation(line: 17, column: 5, scope: !299, inlinedAt: !300)
!299 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !182, file: !182, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!300 = distinct !DILocation(line: 119, column: 17, scope: !301, inlinedAt: !302)
!301 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !182, file: !182, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!302 = distinct !DILocation(line: 137, column: 13, scope: !9)
!303 = !{i64 4270664}
!304 = !DILocation(line: 17, column: 5, scope: !299, inlinedAt: !305)
!305 = distinct !DILocation(line: 120, column: 17, scope: !301, inlinedAt: !302)
!306 = !DILocation(line: 17, column: 5, scope: !299, inlinedAt: !307)
!307 = distinct !DILocation(line: 119, column: 17, scope: !308, inlinedAt: !309)
!308 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 64>, 512>", scope: !182, file: !182, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!309 = distinct !DILocation(line: 139, column: 13, scope: !9)
!310 = !DILocation(line: 17, column: 5, scope: !299, inlinedAt: !311)
!311 = distinct !DILocation(line: 120, column: 17, scope: !308, inlinedAt: !309)
!312 = !DILocation(line: 141, column: 9, scope: !9)
!313 = !DILocation(line: 142, column: 9, scope: !9)
!314 = !DILocation(line: 143, column: 9, scope: !9)
!315 = !DILocation(line: 145, column: 9, scope: !9)
!316 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !317)
!317 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !318)
!318 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !319)
!319 = distinct !DILocation(line: 146, column: 9, scope: !9)
!320 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !321)
!321 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !322)
!322 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !319)
!323 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !324)
!324 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !325)
!325 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !326)
!326 = distinct !DILocation(line: 147, column: 9, scope: !9)
!327 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !328)
!328 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !329)
!329 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !326)
!330 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !331)
!331 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !332)
!332 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !333)
!333 = distinct !DILocation(line: 148, column: 9, scope: !9)
!334 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !335)
!335 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !336)
!336 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !333)
!337 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !338)
!338 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !339)
!339 = distinct !DILocation(line: 216, column: 13, scope: !266, inlinedAt: !340)
!340 = distinct !DILocation(line: 149, column: 9, scope: !9)
!341 = !DILocation(line: 29, column: 27, scope: !261, inlinedAt: !342)
!342 = distinct !DILocation(line: 81, column: 5, scope: !264, inlinedAt: !343)
!343 = distinct !DILocation(line: 224, column: 17, scope: !266, inlinedAt: !340)
!344 = !DILocation(line: 150, column: 9, scope: !9)
!345 = !DILocation(line: 151, column: 9, scope: !9)
!346 = !DILocation(line: 152, column: 9, scope: !9)
!347 = !DILocation(line: 95, column: 29, scope: !9)
!348 = distinct !{!348, !174, !349, !350}
!349 = !DILocation(line: 153, column: 5, scope: !9)
!350 = !{!"llvm.loop.mustprogress"}
!351 = !DILocation(line: 156, column: 9, scope: !9)
!352 = !DILocation(line: 157, column: 5, scope: !9)
!353 = !DILocation(line: 167, column: 32, scope: !9)
!354 = !DILocation(line: 60, column: 34, scope: !355, inlinedAt: !356)
!355 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!356 = distinct !DILocation(line: 192, column: 41, scope: !357, inlinedAt: !359)
!357 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !358, file: !358, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!358 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!359 = distinct !DILocation(line: 212, column: 5, scope: !360, inlinedAt: !361)
!360 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !358, file: !358, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!361 = distinct !DILocation(line: 164, column: 9, scope: !9)
!362 = !DILocation(line: 61, column: 18, scope: !355, inlinedAt: !356)
!363 = !DILocation(line: 63, column: 54, scope: !94, inlinedAt: !364)
!364 = distinct !DILocation(line: 192, column: 23, scope: !357, inlinedAt: !359)
!365 = !DILocation(line: 63, column: 60, scope: !94, inlinedAt: !364)
!366 = !DILocation(line: 63, column: 68, scope: !94, inlinedAt: !364)
!367 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !364)
!368 = !DILocation(line: 203, column: 49, scope: !357, inlinedAt: !359)
!369 = !DILocation(line: 218, column: 16, scope: !370, inlinedAt: !371)
!370 = distinct !DISubprogram(name: "float_2_bfloatraw", scope: !197, file: !197, line: 206, type: !10, scopeLine: 206, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!371 = distinct !DILocation(line: 300, column: 87, scope: !372, inlinedAt: !373)
!372 = distinct !DISubprogram(name: "__hip_bfloat16", scope: !197, file: !197, line: 300, type: !10, scopeLine: 300, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!373 = distinct !DILocation(line: 514, column: 18, scope: !374, inlinedAt: !375)
!374 = distinct !DISubprogram(name: "__float2bfloat16", scope: !197, file: !197, line: 513, type: !10, scopeLine: 513, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!375 = distinct !DILocation(line: 227, column: 16, scope: !376, inlinedAt: !378)
!376 = distinct !DISubprogram(name: "convert", scope: !377, file: !377, line: 226, type: !10, scopeLine: 226, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!377 = !DIFile(filename: "ThunderKittens-HIP/include/common/base_types.cuh", directory: "/workdir/AMD-benchmarking-harness")
!378 = distinct !DILocation(line: 203, column: 49, scope: !357, inlinedAt: !359)
!379 = !DILocation(line: 218, column: 9, scope: !370, inlinedAt: !371)
!380 = !DILocation(line: 235, column: 33, scope: !370, inlinedAt: !371)
!381 = !DILocation(line: 235, column: 40, scope: !370, inlinedAt: !371)
!382 = !DILocation(line: 235, column: 23, scope: !370, inlinedAt: !371)
!383 = !DILocation(line: 235, column: 13, scope: !370, inlinedAt: !371)
!384 = !DILocation(line: 236, column: 5, scope: !370, inlinedAt: !371)
!385 = !DILocation(line: 236, column: 22, scope: !370, inlinedAt: !371)
!386 = !DILocation(line: 236, column: 16, scope: !370, inlinedAt: !371)
!387 = !DILocation(line: 247, column: 42, scope: !370, inlinedAt: !371)
!388 = !DILocation(line: 0, scope: !370, inlinedAt: !371)
!389 = !DILocation(line: 247, column: 46, scope: !370, inlinedAt: !371)
!390 = !DILocation(line: 247, column: 40, scope: !370, inlinedAt: !371)
!391 = !DILocation(line: 203, column: 13, scope: !357, inlinedAt: !359)
!392 = !DILocation(line: 203, column: 47, scope: !357, inlinedAt: !359)
!393 = !{!394, !394, i64 0}
!394 = !{!"short", !395, i64 0}
!395 = !{!"omnipotent char", !396, i64 0}
!396 = !{!"Simple C++ TBAA"}
!397 = !DILocation(line: 204, column: 49, scope: !357, inlinedAt: !359)
!398 = !DILocation(line: 218, column: 16, scope: !370, inlinedAt: !399)
!399 = distinct !DILocation(line: 300, column: 87, scope: !372, inlinedAt: !400)
!400 = distinct !DILocation(line: 514, column: 18, scope: !374, inlinedAt: !401)
!401 = distinct !DILocation(line: 227, column: 16, scope: !376, inlinedAt: !402)
!402 = distinct !DILocation(line: 204, column: 49, scope: !357, inlinedAt: !359)
!403 = !DILocation(line: 218, column: 9, scope: !370, inlinedAt: !399)
!404 = !DILocation(line: 235, column: 33, scope: !370, inlinedAt: !399)
!405 = !DILocation(line: 235, column: 40, scope: !370, inlinedAt: !399)
!406 = !DILocation(line: 235, column: 23, scope: !370, inlinedAt: !399)
!407 = !DILocation(line: 235, column: 13, scope: !370, inlinedAt: !399)
!408 = !DILocation(line: 236, column: 5, scope: !370, inlinedAt: !399)
!409 = !DILocation(line: 236, column: 22, scope: !370, inlinedAt: !399)
!410 = !DILocation(line: 236, column: 16, scope: !370, inlinedAt: !399)
!411 = !DILocation(line: 247, column: 42, scope: !370, inlinedAt: !399)
!412 = !DILocation(line: 0, scope: !370, inlinedAt: !399)
!413 = !DILocation(line: 247, column: 46, scope: !370, inlinedAt: !399)
!414 = !DILocation(line: 247, column: 40, scope: !370, inlinedAt: !399)
!415 = !DILocation(line: 204, column: 13, scope: !357, inlinedAt: !359)
!416 = !DILocation(line: 204, column: 47, scope: !357, inlinedAt: !359)
!417 = !DILocation(line: 205, column: 49, scope: !357, inlinedAt: !359)
!418 = !DILocation(line: 218, column: 16, scope: !370, inlinedAt: !419)
!419 = distinct !DILocation(line: 300, column: 87, scope: !372, inlinedAt: !420)
!420 = distinct !DILocation(line: 514, column: 18, scope: !374, inlinedAt: !421)
!421 = distinct !DILocation(line: 227, column: 16, scope: !376, inlinedAt: !422)
!422 = distinct !DILocation(line: 205, column: 49, scope: !357, inlinedAt: !359)
!423 = !DILocation(line: 218, column: 9, scope: !370, inlinedAt: !419)
!424 = !DILocation(line: 235, column: 33, scope: !370, inlinedAt: !419)
!425 = !DILocation(line: 235, column: 40, scope: !370, inlinedAt: !419)
!426 = !DILocation(line: 235, column: 23, scope: !370, inlinedAt: !419)
!427 = !DILocation(line: 235, column: 13, scope: !370, inlinedAt: !419)
!428 = !DILocation(line: 236, column: 5, scope: !370, inlinedAt: !419)
!429 = !DILocation(line: 236, column: 22, scope: !370, inlinedAt: !419)
!430 = !DILocation(line: 236, column: 16, scope: !370, inlinedAt: !419)
!431 = !DILocation(line: 247, column: 42, scope: !370, inlinedAt: !419)
!432 = !DILocation(line: 0, scope: !370, inlinedAt: !419)
!433 = !DILocation(line: 247, column: 46, scope: !370, inlinedAt: !419)
!434 = !DILocation(line: 247, column: 40, scope: !370, inlinedAt: !419)
!435 = !DILocation(line: 205, column: 13, scope: !357, inlinedAt: !359)
!436 = !DILocation(line: 205, column: 47, scope: !357, inlinedAt: !359)
!437 = !DILocation(line: 206, column: 49, scope: !357, inlinedAt: !359)
!438 = !DILocation(line: 218, column: 16, scope: !370, inlinedAt: !439)
!439 = distinct !DILocation(line: 300, column: 87, scope: !372, inlinedAt: !440)
!440 = distinct !DILocation(line: 514, column: 18, scope: !374, inlinedAt: !441)
!441 = distinct !DILocation(line: 227, column: 16, scope: !376, inlinedAt: !442)
!442 = distinct !DILocation(line: 206, column: 49, scope: !357, inlinedAt: !359)
!443 = !DILocation(line: 218, column: 9, scope: !370, inlinedAt: !439)
!444 = !DILocation(line: 235, column: 33, scope: !370, inlinedAt: !439)
!445 = !DILocation(line: 235, column: 40, scope: !370, inlinedAt: !439)
!446 = !DILocation(line: 235, column: 23, scope: !370, inlinedAt: !439)
!447 = !DILocation(line: 235, column: 13, scope: !370, inlinedAt: !439)
!448 = !DILocation(line: 236, column: 5, scope: !370, inlinedAt: !439)
!449 = !DILocation(line: 236, column: 22, scope: !370, inlinedAt: !439)
!450 = !DILocation(line: 236, column: 16, scope: !370, inlinedAt: !439)
!451 = !DILocation(line: 247, column: 42, scope: !370, inlinedAt: !439)
!452 = !DILocation(line: 0, scope: !370, inlinedAt: !439)
!453 = !DILocation(line: 247, column: 46, scope: !370, inlinedAt: !439)
!454 = !DILocation(line: 247, column: 40, scope: !370, inlinedAt: !439)
!455 = !DILocation(line: 206, column: 13, scope: !357, inlinedAt: !359)
!456 = !DILocation(line: 206, column: 47, scope: !357, inlinedAt: !359)
!457 = !DILocation(line: 170, column: 1, scope: !9)
