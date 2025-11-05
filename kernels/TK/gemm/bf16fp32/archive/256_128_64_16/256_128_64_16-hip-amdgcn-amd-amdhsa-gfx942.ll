; ModuleID = '256_128_64_16-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_128_64_16.cpp"
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
@__hip_cuid_7eee1c634c0cf24c = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_7eee1c634c0cf24c to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.01082.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.71086.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.71086.0.copyload = load i64, ptr addrspace(4) %.sroa.71086.0..sroa_idx, align 8
  %.sroa.91087.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.91087.0.copyload = load ptr, ptr addrspace(4) %.sroa.91087.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.171091.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.171091.0.copyload = load ptr, ptr addrspace(4) %.sroa.171091.0..sroa_idx, align 8
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
  %reass.sub.i.i181 = and i64 %7, -16, !dbg !26
  %9 = add i64 %reass.sub.i.i181, 16, !dbg !26
  %10 = inttoptr i64 %9 to ptr, !dbg !26
  %.not.i.i180 = icmp eq i64 %8, 0, !dbg !27
  %11 = select i1 %.not.i.i180, ptr %6, ptr %10, !dbg !26
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
  %.frozen1378 = freeze i32 %25, !dbg !61
  %26 = sdiv i32 %.frozen1378, 512, !dbg !61
  %27 = shl nsw i32 %26, 4, !dbg !62
  %28 = sub nsw i32 64, %27, !dbg !63
  %29 = tail call noundef range(i32 -67108784, 67108929) i32 @llvm.smin.i32(i32 %28, i32 16), !dbg !64
  %30 = mul i32 %26, 512, !dbg !68
  %.decomposed1379 = sub i32 %.frozen1378, %30, !dbg !68
  %31 = sdiv i32 %.decomposed1379, %29, !dbg !69
  %32 = mul i32 %31, %29, !dbg !70
  %.decomposed1380 = sub i32 %.decomposed1379, %32, !dbg !70
  %33 = add nsw i32 %27, %.decomposed1380, !dbg !71
  %34 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !72
  %35 = lshr i32 %34, 6, !dbg !79
  %36 = lshr i32 %34, 8, !dbg !80
  %37 = and i32 %35, 3, !dbg !81
  %38 = trunc i64 %.sroa.71086.0.copyload to i32, !dbg !82
  %39 = shl nsw i32 %33, 7, !dbg !89
  %40 = sext i32 %39 to i64, !dbg !93
  %41 = mul i64 %.sroa.71086.0.copyload, %40, !dbg !97
  %42 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.01082.0.copyload, i64 %41, !dbg !98
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
  %86 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.91087.0.copyload, i64 %85, !dbg !143
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
  %.sroa.0.0.extract.trunc.i.i186 = trunc i128 %92 to i64, !dbg !157
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %116, i64 %.sroa.0.0.extract.trunc.i.i186) #7, !dbg !158, !srcloc !121
  %117 = add i32 %112, %.masked.i.i, !dbg !160
  %118 = lshr i32 %117, 4, !dbg !162
  %119 = and i32 %118, 120, !dbg !162
  %120 = xor i32 %119, %117, !dbg !163
  %.sroa.0.8.extract.shift.i.i187 = lshr i128 %92, 64, !dbg !164
  %.sroa.0.8.extract.trunc.i.i188 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i187 to i64, !dbg !164
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %120, i64 %.sroa.0.8.extract.trunc.i.i188) #7, !dbg !165, !srcloc !121
  %121 = add i32 %111, %73, !dbg !152
  %122 = lshr exact i32 %121, 4, !dbg !155
  %123 = and i32 %122, 120, !dbg !155
  %124 = xor i32 %123, %121, !dbg !156
  %.sroa.5.16.extract.trunc.i.i189 = trunc i128 %97 to i64, !dbg !157
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %124, i64 %.sroa.5.16.extract.trunc.i.i189) #7, !dbg !158, !srcloc !121
  %125 = add i32 %112, %73, !dbg !160
  %126 = lshr i32 %125, 4, !dbg !162
  %127 = and i32 %126, 120, !dbg !162
  %128 = xor i32 %127, %125, !dbg !163
  %.sroa.5.24.extract.shift.i.i190 = lshr i128 %97, 64, !dbg !164
  %.sroa.5.24.extract.trunc.i.i191 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i190 to i64, !dbg !164
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %128, i64 %.sroa.5.24.extract.trunc.i.i191) #7, !dbg !165, !srcloc !121
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
  %162 = or disjoint i32 %151, 16
  %163 = or disjoint i32 %150, %162
  %164 = shl nuw nsw i32 %163, 7
  %165 = add i32 %164, %110
  %166 = add i32 %165, %156
  %167 = lshr i32 %166, 4
  %168 = and i32 %167, 120
  %169 = xor i32 %168, %166
  %170 = or disjoint i32 %150, 128
  %171 = or disjoint i32 %170, %151
  %172 = shl nuw nsw i32 %171, 7
  %173 = add i32 %172, %110
  %174 = add i32 %173, %156
  %175 = lshr i32 %174, 4
  %176 = and i32 %175, 120
  %177 = xor i32 %176, %174
  %178 = or disjoint i32 %170, %162
  %179 = shl nuw nsw i32 %178, 7
  %180 = add i32 %179, %110
  %181 = add i32 %180, %156
  %182 = lshr i32 %181, 4
  %183 = and i32 %182, 120
  %184 = xor i32 %183, %181
  %185 = shl nuw nsw i32 %36, 5
  %186 = or disjoint i32 %185, %151
  %187 = shl nuw nsw i32 %186, 7
  %188 = add i32 %187, %59
  %189 = add i32 %188, %156
  %190 = lshr i32 %189, 4
  %191 = and i32 %190, 120
  %192 = xor i32 %191, %189
  %193 = or disjoint i32 %162, %185
  %194 = shl nuw nsw i32 %193, 7
  %195 = add i32 %194, %59
  %196 = add i32 %195, %156
  %197 = lshr i32 %196, 4
  %198 = and i32 %197, 120
  %199 = xor i32 %198, %196
  %200 = add nuw nsw i32 %185, 64
  %201 = or disjoint i32 %200, %151
  %202 = shl nuw nsw i32 %201, 7
  %203 = add i32 %202, %59
  %204 = add i32 %203, %156
  %205 = lshr i32 %204, 4
  %206 = and i32 %205, 120
  %207 = xor i32 %206, %204
  %208 = or disjoint i32 %200, %162
  %209 = shl nuw nsw i32 %208, 7
  %210 = add i32 %209, %59
  %211 = add i32 %210, %156
  %212 = lshr i32 %211, 4
  %213 = and i32 %212, 120
  %214 = xor i32 %213, %211
  %215 = shl i32 %55, 1
  %216 = shl i32 %49, 1
  %217 = shl i32 %38, 8
  %218 = or disjoint i32 %156, 32
  %219 = add i32 %157, %218
  %220 = lshr i32 %219, 4
  %221 = and i32 %220, 120
  %222 = xor i32 %221, %219
  %223 = add i32 %165, %218
  %224 = lshr i32 %223, 4
  %225 = and i32 %224, 120
  %226 = xor i32 %225, %223
  %227 = add i32 %173, %218
  %228 = lshr i32 %227, 4
  %229 = and i32 %228, 120
  %230 = xor i32 %229, %227
  %231 = add i32 %180, %218
  %232 = lshr i32 %231, 4
  %233 = and i32 %232, 120
  %234 = xor i32 %233, %231
  %235 = add i32 %188, %218
  %236 = lshr i32 %235, 4
  %237 = and i32 %236, 120
  %238 = xor i32 %237, %235
  %239 = add i32 %195, %218
  %240 = lshr i32 %239, 4
  %241 = and i32 %240, 120
  %242 = xor i32 %241, %239
  %243 = add i32 %203, %218
  %244 = lshr i32 %243, 4
  %245 = and i32 %244, 120
  %246 = xor i32 %245, %243
  %247 = add i32 %210, %218
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
  %261 = add i32 %165, %256
  %262 = lshr i32 %261, 4
  %263 = and i32 %262, 120
  %264 = xor i32 %263, %261
  %265 = add i32 %173, %256
  %266 = lshr i32 %265, 4
  %267 = and i32 %266, 120
  %268 = xor i32 %267, %265
  %269 = add i32 %180, %256
  %270 = lshr i32 %269, 4
  %271 = and i32 %270, 120
  %272 = xor i32 %271, %269
  %273 = add i32 %188, %256
  %274 = lshr i32 %273, 4
  %275 = and i32 %274, 120
  %276 = xor i32 %275, %273
  %277 = add i32 %195, %256
  %278 = lshr i32 %277, 4
  %279 = and i32 %278, 120
  %280 = xor i32 %279, %277
  %281 = add i32 %203, %256
  %282 = lshr i32 %281, 4
  %283 = and i32 %282, 120
  %284 = xor i32 %283, %281
  %285 = add i32 %210, %256
  %286 = lshr i32 %285, 4
  %287 = and i32 %286, 120
  %288 = xor i32 %287, %285
  %289 = or disjoint i32 %156, 96
  %290 = add i32 %157, %289
  %291 = lshr i32 %290, 4
  %292 = and i32 %291, 120
  %293 = xor i32 %292, %290
  %294 = add i32 %165, %289
  %295 = lshr i32 %294, 4
  %296 = and i32 %295, 120
  %297 = xor i32 %296, %294
  %298 = add i32 %173, %289
  %299 = lshr i32 %298, 4
  %300 = and i32 %299, 120
  %301 = xor i32 %300, %298
  %302 = add i32 %180, %289
  %303 = lshr i32 %302, 4
  %304 = and i32 %303, 120
  %305 = xor i32 %304, %302
  %306 = add i32 %188, %289
  %307 = lshr i32 %306, 4
  %308 = and i32 %307, 120
  %309 = xor i32 %308, %306
  %310 = add i32 %195, %289
  %311 = lshr i32 %310, 4
  %312 = and i32 %311, 120
  %313 = xor i32 %312, %310
  %314 = add i32 %203, %289
  %315 = lshr i32 %314, 4
  %316 = and i32 %315, 120
  %317 = xor i32 %316, %314
  %318 = add i32 %210, %289
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

376:                                              ; preds = %652
  %377 = icmp ult i32 %34, 256, !dbg !175
  br i1 %377, label %669, label %670, !dbg !176

378:                                              ; preds = %149, %652
  %379 = phi <4 x float> [ zeroinitializer, %149 ], [ %668, %652 ]
  %380 = phi <4 x float> [ zeroinitializer, %149 ], [ %667, %652 ]
  %381 = phi <4 x float> [ zeroinitializer, %149 ], [ %666, %652 ]
  %.val30.i2821181 = phi <4 x float> [ zeroinitializer, %149 ], [ %665, %652 ]
  %382 = phi <4 x float> [ zeroinitializer, %149 ], [ %664, %652 ]
  %383 = phi <4 x float> [ zeroinitializer, %149 ], [ %663, %652 ]
  %384 = phi <4 x float> [ zeroinitializer, %149 ], [ %662, %652 ]
  %.val30.i2701172 = phi <4 x float> [ zeroinitializer, %149 ], [ %661, %652 ]
  %385 = phi <4 x float> [ zeroinitializer, %149 ], [ %660, %652 ]
  %386 = phi <4 x float> [ zeroinitializer, %149 ], [ %659, %652 ]
  %387 = phi <4 x float> [ zeroinitializer, %149 ], [ %658, %652 ]
  %.val30.i2581163 = phi <4 x float> [ zeroinitializer, %149 ], [ %657, %652 ]
  %.val30.1.1.i6561160 = phi <4 x float> [ zeroinitializer, %149 ], [ %656, %652 ]
  %.val30.135.i6531157 = phi <4 x float> [ zeroinitializer, %149 ], [ %655, %652 ]
  %.val30.1.i6501154 = phi <4 x float> [ zeroinitializer, %149 ], [ %654, %652 ]
  %.01731151 = phi i32 [ 0, %149 ], [ %388, %652 ]
  %.sroa.5809.0.off641150 = phi i64 [ 0, %149 ], [ %.sroa.5809.1.off64, %652 ]
  %.sroa.5809.0.off01149 = phi i64 [ undef, %149 ], [ %.sroa.5809.1.off0, %652 ]
  %.sroa.0808.0.off641148 = phi i64 [ 0, %149 ], [ %.sroa.0808.1.off64, %652 ]
  %.sroa.0808.0.off01147 = phi i64 [ undef, %149 ], [ %.sroa.0808.1.off0, %652 ]
  %.sroa.11.0.off641146 = phi i64 [ 0, %149 ], [ %.sroa.11.1.off64, %652 ]
  %.sroa.11.0.off01145 = phi i64 [ undef, %149 ], [ %.sroa.11.1.off0, %652 ]
  %.sroa.8807.0.off641144 = phi i64 [ 0, %149 ], [ %.sroa.8807.1.off64, %652 ]
  %.sroa.8807.0.off01143 = phi i64 [ undef, %149 ], [ %.sroa.8807.1.off0, %652 ]
  %.sroa.5.0.off641142 = phi i64 [ 0, %149 ], [ %.sroa.5.1.off64, %652 ]
  %.sroa.5.0.off01141 = phi i64 [ undef, %149 ], [ %.sroa.5.1.off0, %652 ]
  %.sroa.0806.0.off641140 = phi i64 [ 0, %149 ], [ %.sroa.0806.1.off64, %652 ]
  %.sroa.0806.0.off01139 = phi i64 [ undef, %149 ], [ %.sroa.0806.1.off0, %652 ]
  %.val30.i11371138 = phi <4 x float> [ zeroinitializer, %149 ], [ %653, %652 ]
  %388 = add nuw nsw i32 %.01731151, 1, !dbg !177
  %.not = icmp eq i32 %.01731151, 127, !dbg !178
  %389 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %161) #7, !dbg !179, !srcloc !185
  %.sroa.036.2.extract.shift37.i = lshr i64 %389, 16, !dbg !186
  %390 = insertelement <2 x i64> poison, i64 %389, i64 0, !dbg !186
  %391 = insertelement <2 x i64> %390, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !186
  %392 = shufflevector <2 x i64> %390, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !179
  %393 = lshr <2 x i64> %392, <i64 32, i64 48>, !dbg !179
  %394 = shufflevector <2 x i64> %391, <2 x i64> %393, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !187
  %.sroa.0876.6.vec.insert8921347 = trunc <4 x i64> %394 to <4 x i16>, !dbg !187
  %395 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %169) #7, !dbg !179, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %395, 16, !dbg !186
  %396 = insertelement <2 x i64> poison, i64 %395, i64 0, !dbg !186
  %397 = insertelement <2 x i64> %396, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !186
  %398 = shufflevector <2 x i64> %396, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !179
  %399 = lshr <2 x i64> %398, <i64 32, i64 48>, !dbg !179
  %400 = shufflevector <2 x i64> %397, <2 x i64> %399, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !187
  %.sroa.18893.14.vec.insert9091346 = trunc <4 x i64> %400 to <4 x i16>, !dbg !187
  %401 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %177) #7, !dbg !191, !srcloc !185
  %.sroa.036.2.extract.shift37.i199 = lshr i64 %401, 16, !dbg !194
  %402 = insertelement <2 x i64> poison, i64 %401, i64 0, !dbg !194
  %403 = insertelement <2 x i64> %402, i64 %.sroa.036.2.extract.shift37.i199, i64 1, !dbg !194
  %404 = shufflevector <2 x i64> %402, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !191
  %405 = lshr <2 x i64> %404, <i64 32, i64 48>, !dbg !191
  %406 = shufflevector <2 x i64> %403, <2 x i64> %405, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.0811.6.vec.insert8271345 = trunc <4 x i64> %406 to <4 x i16>, !dbg !195
  %407 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %184) #7, !dbg !191, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i208 = lshr i64 %407, 16, !dbg !194
  %408 = insertelement <2 x i64> poison, i64 %407, i64 0, !dbg !194
  %409 = insertelement <2 x i64> %408, i64 %.sroa.036.2.extract.shift37.1.i208, i64 1, !dbg !194
  %410 = shufflevector <2 x i64> %408, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !191
  %411 = lshr <2 x i64> %410, <i64 32, i64 48>, !dbg !191
  %412 = shufflevector <2 x i64> %409, <2 x i64> %411, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !195
  %.sroa.18.14.vec.insert8431344 = trunc <4 x i64> %412 to <4 x i16>, !dbg !195
  %413 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %192) #7, !dbg !197, !srcloc !185
  %.sroa.036.2.extract.shift37.i218 = lshr i64 %413, 16, !dbg !201
  %414 = insertelement <2 x i64> poison, i64 %413, i64 0, !dbg !201
  %415 = insertelement <2 x i64> %414, i64 %.sroa.036.2.extract.shift37.i218, i64 1, !dbg !201
  %416 = shufflevector <2 x i64> %414, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !197
  %417 = lshr <2 x i64> %416, <i64 32, i64 48>, !dbg !197
  %418 = shufflevector <2 x i64> %415, <2 x i64> %417, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !202
  %.sroa.01012.6.vec.insert10281343 = trunc <4 x i64> %418 to <4 x i16>, !dbg !202
  %419 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %199) #7, !dbg !197, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i225 = lshr i64 %419, 16, !dbg !201
  %420 = insertelement <2 x i64> poison, i64 %419, i64 0, !dbg !201
  %421 = insertelement <2 x i64> %420, i64 %.sroa.036.2.extract.shift37.1.i225, i64 1, !dbg !201
  %422 = shufflevector <2 x i64> %420, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !197
  %423 = lshr <2 x i64> %422, <i64 32, i64 48>, !dbg !197
  %424 = shufflevector <2 x i64> %421, <2 x i64> %423, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !202
  %.sroa.181029.14.vec.insert10451342 = trunc <4 x i64> %424 to <4 x i16>, !dbg !202
  %425 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %207) #7, !dbg !204, !srcloc !185
  %.sroa.036.2.extract.shift37.i237 = lshr i64 %425, 16, !dbg !207
  %426 = insertelement <2 x i64> poison, i64 %425, i64 0, !dbg !207
  %427 = insertelement <2 x i64> %426, i64 %.sroa.036.2.extract.shift37.i237, i64 1, !dbg !207
  %428 = shufflevector <2 x i64> %426, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !204
  %429 = lshr <2 x i64> %428, <i64 32, i64 48>, !dbg !204
  %430 = shufflevector <2 x i64> %427, <2 x i64> %429, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !208
  %.sroa.0944.6.vec.insert9601341 = trunc <4 x i64> %430 to <4 x i16>, !dbg !208
  %431 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %214) #7, !dbg !204, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i246 = lshr i64 %431, 16, !dbg !207
  %432 = insertelement <2 x i64> poison, i64 %431, i64 0, !dbg !207
  %433 = insertelement <2 x i64> %432, i64 %.sroa.036.2.extract.shift37.1.i246, i64 1, !dbg !207
  %434 = shufflevector <2 x i64> %432, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !204
  %435 = lshr <2 x i64> %434, <i64 32, i64 48>, !dbg !204
  %436 = shufflevector <2 x i64> %433, <2 x i64> %435, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !208
  %.sroa.18961.14.vec.insert9771340 = trunc <4 x i64> %436 to <4 x i16>, !dbg !208
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !210
  br i1 %.not, label %446, label %437, !dbg !211

437:                                              ; preds = %378
  %438 = shl nuw nsw i32 %388, 6
  %439 = zext nneg i32 %438 to i64
  %440 = getelementptr %struct.__hip_bfloat16, ptr %42, i64 %439
  %441 = ptrtoint ptr %440 to i64
  %442 = bitcast i64 %441 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %442, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %443 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %443, i32 %217, i64 2
  %444 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %216, i32 noundef 0, i32 noundef 0) #7, !dbg !212
  %445 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %215, i32 noundef 0, i32 noundef 0) #7, !dbg !212
  %extract.t1122 = trunc i128 %444 to i64, !dbg !217
  %extract1124 = lshr i128 %444, 64, !dbg !217
  %extract.t1125 = trunc nuw i128 %extract1124 to i64, !dbg !217
  %extract.t1129 = trunc i128 %445 to i64, !dbg !217
  %extract1131 = lshr i128 %445, 64, !dbg !217
  %extract.t1132 = trunc nuw i128 %extract1131 to i64, !dbg !217
  br label %446, !dbg !217

446:                                              ; preds = %437, %378
  %.sroa.0808.1.off0 = phi i64 [ %.sroa.0808.0.off01147, %378 ], [ %extract.t1122, %437 ]
  %.sroa.0808.1.off64 = phi i64 [ %.sroa.0808.0.off641148, %378 ], [ %extract.t1125, %437 ]
  %.sroa.5809.1.off0 = phi i64 [ %.sroa.5809.0.off01149, %378 ], [ %extract.t1129, %437 ]
  %.sroa.5809.1.off64 = phi i64 [ %.sroa.5809.0.off641150, %378 ], [ %extract.t1132, %437 ]
  tail call void @llvm.amdgcn.s.barrier(), !dbg !218
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !219
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !220, !srcloc !221
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !222
  %447 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert10281343, <4 x i16> %.sroa.0876.6.vec.insert8921347, <4 x float> %.val30.i11371138, i32 0, i32 0, i32 0), !dbg !223
  %448 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert10281343, <4 x i16> %.sroa.18893.14.vec.insert9091346, <4 x float> %.val30.1.i6501154, i32 0, i32 0, i32 0), !dbg !223
  %449 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert10451342, <4 x i16> %.sroa.0876.6.vec.insert8921347, <4 x float> %.val30.135.i6531157, i32 0, i32 0, i32 0), !dbg !223
  %450 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert10451342, <4 x i16> %.sroa.18893.14.vec.insert9091346, <4 x float> %.val30.1.1.i6561160, i32 0, i32 0, i32 0), !dbg !223
  %451 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert10281343, <4 x i16> %.sroa.0811.6.vec.insert8271345, <4 x float> %.val30.i2581163, i32 0, i32 0, i32 0), !dbg !231
  %452 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert10281343, <4 x i16> %.sroa.18.14.vec.insert8431344, <4 x float> %387, i32 0, i32 0, i32 0), !dbg !231
  %453 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert10451342, <4 x i16> %.sroa.0811.6.vec.insert8271345, <4 x float> %386, i32 0, i32 0, i32 0), !dbg !231
  %454 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert10451342, <4 x i16> %.sroa.18.14.vec.insert8431344, <4 x float> %385, i32 0, i32 0, i32 0), !dbg !231
  %455 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert9601341, <4 x i16> %.sroa.0876.6.vec.insert8921347, <4 x float> %.val30.i2701172, i32 0, i32 0, i32 0), !dbg !235
  %456 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert9601341, <4 x i16> %.sroa.18893.14.vec.insert9091346, <4 x float> %384, i32 0, i32 0, i32 0), !dbg !235
  %457 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert9771340, <4 x i16> %.sroa.0876.6.vec.insert8921347, <4 x float> %383, i32 0, i32 0, i32 0), !dbg !235
  %458 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert9771340, <4 x i16> %.sroa.18893.14.vec.insert9091346, <4 x float> %382, i32 0, i32 0, i32 0), !dbg !235
  %459 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert9601341, <4 x i16> %.sroa.0811.6.vec.insert8271345, <4 x float> %.val30.i2821181, i32 0, i32 0, i32 0), !dbg !239
  %460 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert9601341, <4 x i16> %.sroa.18.14.vec.insert8431344, <4 x float> %381, i32 0, i32 0, i32 0), !dbg !239
  %461 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert9771340, <4 x i16> %.sroa.0811.6.vec.insert8271345, <4 x float> %380, i32 0, i32 0, i32 0), !dbg !239
  %462 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert9771340, <4 x i16> %.sroa.18.14.vec.insert8431344, <4 x float> %379, i32 0, i32 0, i32 0), !dbg !239
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !243
  tail call void @llvm.amdgcn.s.barrier(), !dbg !244
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !245
  %463 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %222) #7, !dbg !246, !srcloc !185
  %.sroa.036.2.extract.shift37.i299 = lshr i64 %463, 16, !dbg !249
  %464 = insertelement <2 x i64> poison, i64 %463, i64 0, !dbg !249
  %465 = insertelement <2 x i64> %464, i64 %.sroa.036.2.extract.shift37.i299, i64 1, !dbg !249
  %466 = shufflevector <2 x i64> %464, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !246
  %467 = lshr <2 x i64> %466, <i64 32, i64 48>, !dbg !246
  %468 = shufflevector <2 x i64> %465, <2 x i64> %467, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !250
  %.sroa.34910.22.vec.insert9261339 = trunc <4 x i64> %468 to <4 x i16>, !dbg !250
  %469 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %226) #7, !dbg !246, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i308 = lshr i64 %469, 16, !dbg !249
  %470 = insertelement <2 x i64> poison, i64 %469, i64 0, !dbg !249
  %471 = insertelement <2 x i64> %470, i64 %.sroa.036.2.extract.shift37.1.i308, i64 1, !dbg !249
  %472 = shufflevector <2 x i64> %470, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !246
  %473 = lshr <2 x i64> %472, <i64 32, i64 48>, !dbg !246
  %474 = shufflevector <2 x i64> %471, <2 x i64> %473, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !250
  %.sroa.50927.30.vec.insert9431338 = trunc <4 x i64> %474 to <4 x i16>, !dbg !250
  %475 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %230) #7, !dbg !252, !srcloc !185
  %.sroa.036.2.extract.shift37.i320 = lshr i64 %475, 16, !dbg !255
  %476 = insertelement <2 x i64> poison, i64 %475, i64 0, !dbg !255
  %477 = insertelement <2 x i64> %476, i64 %.sroa.036.2.extract.shift37.i320, i64 1, !dbg !255
  %478 = shufflevector <2 x i64> %476, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !252
  %479 = lshr <2 x i64> %478, <i64 32, i64 48>, !dbg !252
  %480 = shufflevector <2 x i64> %477, <2 x i64> %479, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !256
  %.sroa.34.22.vec.insert8591337 = trunc <4 x i64> %480 to <4 x i16>, !dbg !256
  %481 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %234) #7, !dbg !252, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i329 = lshr i64 %481, 16, !dbg !255
  %482 = insertelement <2 x i64> poison, i64 %481, i64 0, !dbg !255
  %483 = insertelement <2 x i64> %482, i64 %.sroa.036.2.extract.shift37.1.i329, i64 1, !dbg !255
  %484 = shufflevector <2 x i64> %482, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !252
  %485 = lshr <2 x i64> %484, <i64 32, i64 48>, !dbg !252
  %486 = shufflevector <2 x i64> %483, <2 x i64> %485, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !256
  %.sroa.50.30.vec.insert8751336 = trunc <4 x i64> %486 to <4 x i16>, !dbg !256
  %487 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %238) #7, !dbg !258, !srcloc !185
  %.sroa.036.2.extract.shift37.i341 = lshr i64 %487, 16, !dbg !261
  %488 = insertelement <2 x i64> poison, i64 %487, i64 0, !dbg !261
  %489 = insertelement <2 x i64> %488, i64 %.sroa.036.2.extract.shift37.i341, i64 1, !dbg !261
  %490 = shufflevector <2 x i64> %488, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !258
  %491 = lshr <2 x i64> %490, <i64 32, i64 48>, !dbg !258
  %492 = shufflevector <2 x i64> %489, <2 x i64> %491, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !262
  %.sroa.341046.22.vec.insert10621335 = trunc <4 x i64> %492 to <4 x i16>, !dbg !262
  %493 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %242) #7, !dbg !258, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i350 = lshr i64 %493, 16, !dbg !261
  %494 = insertelement <2 x i64> poison, i64 %493, i64 0, !dbg !261
  %495 = insertelement <2 x i64> %494, i64 %.sroa.036.2.extract.shift37.1.i350, i64 1, !dbg !261
  %496 = shufflevector <2 x i64> %494, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !258
  %497 = lshr <2 x i64> %496, <i64 32, i64 48>, !dbg !258
  %498 = shufflevector <2 x i64> %495, <2 x i64> %497, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !262
  %.sroa.501063.30.vec.insert10791334 = trunc <4 x i64> %498 to <4 x i16>, !dbg !262
  %499 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %246) #7, !dbg !264, !srcloc !185
  %.sroa.036.2.extract.shift37.i362 = lshr i64 %499, 16, !dbg !267
  %500 = insertelement <2 x i64> poison, i64 %499, i64 0, !dbg !267
  %501 = insertelement <2 x i64> %500, i64 %.sroa.036.2.extract.shift37.i362, i64 1, !dbg !267
  %502 = shufflevector <2 x i64> %500, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !264
  %503 = lshr <2 x i64> %502, <i64 32, i64 48>, !dbg !264
  %504 = shufflevector <2 x i64> %501, <2 x i64> %503, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !268
  %.sroa.34978.22.vec.insert9941333 = trunc <4 x i64> %504 to <4 x i16>, !dbg !268
  %505 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %250) #7, !dbg !264, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i371 = lshr i64 %505, 16, !dbg !267
  %506 = insertelement <2 x i64> poison, i64 %505, i64 0, !dbg !267
  %507 = insertelement <2 x i64> %506, i64 %.sroa.036.2.extract.shift37.1.i371, i64 1, !dbg !267
  %508 = shufflevector <2 x i64> %506, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !264
  %509 = lshr <2 x i64> %508, <i64 32, i64 48>, !dbg !264
  %510 = shufflevector <2 x i64> %507, <2 x i64> %509, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !268
  %.sroa.50995.30.vec.insert10111332 = trunc <4 x i64> %510 to <4 x i16>, !dbg !268
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !270
  br i1 %.not, label %522, label %511, !dbg !271

511:                                              ; preds = %446
  %512 = shl nuw nsw i32 %388, 6, !dbg !272
  %513 = zext nneg i32 %512 to i64, !dbg !276
  %514 = getelementptr %struct.__hip_bfloat16, ptr %86, i64 %513, !dbg !278
  %515 = ptrtoint ptr %514 to i64, !dbg !279
  %516 = bitcast i64 %515 to <2 x i32>, !dbg !279
  %.sroa.0.0.vec.expand.i.i379 = shufflevector <2 x i32> %516, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>, !dbg !279
  %517 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i379, i32 1114112, i64 3, !dbg !279
  %.sroa.0.12.vec.insert.i.i380 = insertelement <4 x i32> %517, i32 %251, i64 2, !dbg !279
  %518 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i380, i32 noundef %252, i32 noundef 0, i32 noundef 0) #7, !dbg !282
  %519 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i380, i32 noundef %253, i32 noundef 0, i32 noundef 0) #7, !dbg !282
  %520 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i380, i32 noundef %254, i32 noundef 0, i32 noundef 0) #7, !dbg !282
  %521 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i380, i32 noundef %255, i32 noundef 0, i32 noundef 0) #7, !dbg !282
  %extract.t1094 = trunc i128 %518 to i64, !dbg !284
  %extract1096 = lshr i128 %518, 64, !dbg !284
  %extract.t1097 = trunc nuw i128 %extract1096 to i64, !dbg !284
  %extract.t1101 = trunc i128 %519 to i64, !dbg !284
  %extract1103 = lshr i128 %519, 64, !dbg !284
  %extract.t1104 = trunc nuw i128 %extract1103 to i64, !dbg !284
  %extract.t1108 = trunc i128 %520 to i64, !dbg !284
  %extract1110 = lshr i128 %520, 64, !dbg !284
  %extract.t1111 = trunc nuw i128 %extract1110 to i64, !dbg !284
  %extract.t1115 = trunc i128 %521 to i64, !dbg !284
  %extract1117 = lshr i128 %521, 64, !dbg !284
  %extract.t1118 = trunc nuw i128 %extract1117 to i64, !dbg !284
  br label %522, !dbg !284

522:                                              ; preds = %511, %446
  %.sroa.0806.1.off0 = phi i64 [ %.sroa.0806.0.off01139, %446 ], [ %extract.t1094, %511 ]
  %.sroa.0806.1.off64 = phi i64 [ %.sroa.0806.0.off641140, %446 ], [ %extract.t1097, %511 ]
  %.sroa.5.1.off0 = phi i64 [ %.sroa.5.0.off01141, %446 ], [ %extract.t1101, %511 ]
  %.sroa.5.1.off64 = phi i64 [ %.sroa.5.0.off641142, %446 ], [ %extract.t1104, %511 ]
  %.sroa.8807.1.off0 = phi i64 [ %.sroa.8807.0.off01143, %446 ], [ %extract.t1108, %511 ]
  %.sroa.8807.1.off64 = phi i64 [ %.sroa.8807.0.off641144, %446 ], [ %extract.t1111, %511 ]
  %.sroa.11.1.off0 = phi i64 [ %.sroa.11.0.off01145, %446 ], [ %extract.t1115, %511 ]
  %.sroa.11.1.off64 = phi i64 [ %.sroa.11.0.off641146, %446 ], [ %extract.t1118, %511 ]
  tail call void @llvm.amdgcn.s.barrier(), !dbg !285
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !286
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #7, !dbg !287, !srcloc !288
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !289
  %523 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert10621335, <4 x i16> %.sroa.34910.22.vec.insert9261339, <4 x float> %447, i32 0, i32 0, i32 0), !dbg !290
  %524 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert10621335, <4 x i16> %.sroa.50927.30.vec.insert9431338, <4 x float> %448, i32 0, i32 0, i32 0), !dbg !290
  %525 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert10791334, <4 x i16> %.sroa.34910.22.vec.insert9261339, <4 x float> %449, i32 0, i32 0, i32 0), !dbg !290
  %526 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert10791334, <4 x i16> %.sroa.50927.30.vec.insert9431338, <4 x float> %450, i32 0, i32 0, i32 0), !dbg !290
  %527 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert10621335, <4 x i16> %.sroa.34.22.vec.insert8591337, <4 x float> %451, i32 0, i32 0, i32 0), !dbg !294
  %528 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert10621335, <4 x i16> %.sroa.50.30.vec.insert8751336, <4 x float> %452, i32 0, i32 0, i32 0), !dbg !294
  %529 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert10791334, <4 x i16> %.sroa.34.22.vec.insert8591337, <4 x float> %453, i32 0, i32 0, i32 0), !dbg !294
  %530 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert10791334, <4 x i16> %.sroa.50.30.vec.insert8751336, <4 x float> %454, i32 0, i32 0, i32 0), !dbg !294
  %531 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert9941333, <4 x i16> %.sroa.34910.22.vec.insert9261339, <4 x float> %455, i32 0, i32 0, i32 0), !dbg !298
  %532 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert9941333, <4 x i16> %.sroa.50927.30.vec.insert9431338, <4 x float> %456, i32 0, i32 0, i32 0), !dbg !298
  %533 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert10111332, <4 x i16> %.sroa.34910.22.vec.insert9261339, <4 x float> %457, i32 0, i32 0, i32 0), !dbg !298
  %534 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert10111332, <4 x i16> %.sroa.50927.30.vec.insert9431338, <4 x float> %458, i32 0, i32 0, i32 0), !dbg !298
  %535 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert9941333, <4 x i16> %.sroa.34.22.vec.insert8591337, <4 x float> %459, i32 0, i32 0, i32 0), !dbg !302
  %536 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert9941333, <4 x i16> %.sroa.50.30.vec.insert8751336, <4 x float> %460, i32 0, i32 0, i32 0), !dbg !302
  %537 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert10111332, <4 x i16> %.sroa.34.22.vec.insert8591337, <4 x float> %461, i32 0, i32 0, i32 0), !dbg !302
  %538 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert10111332, <4 x i16> %.sroa.50.30.vec.insert8751336, <4 x float> %462, i32 0, i32 0, i32 0), !dbg !302
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !306
  tail call void @llvm.amdgcn.s.barrier(), !dbg !307
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !308
  %539 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %260) #7, !dbg !309, !srcloc !185
  %.sroa.036.2.extract.shift37.i436 = lshr i64 %539, 16, !dbg !312
  %540 = insertelement <2 x i64> poison, i64 %539, i64 0, !dbg !312
  %541 = insertelement <2 x i64> %540, i64 %.sroa.036.2.extract.shift37.i436, i64 1, !dbg !312
  %542 = shufflevector <2 x i64> %540, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !309
  %543 = lshr <2 x i64> %542, <i64 32, i64 48>, !dbg !309
  %544 = shufflevector <2 x i64> %541, <2 x i64> %543, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !313
  %.sroa.0876.6.vec.insert1331 = trunc <4 x i64> %544 to <4 x i16>, !dbg !313
  %545 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %264) #7, !dbg !309, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i445 = lshr i64 %545, 16, !dbg !312
  %546 = insertelement <2 x i64> poison, i64 %545, i64 0, !dbg !312
  %547 = insertelement <2 x i64> %546, i64 %.sroa.036.2.extract.shift37.1.i445, i64 1, !dbg !312
  %548 = shufflevector <2 x i64> %546, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !309
  %549 = lshr <2 x i64> %548, <i64 32, i64 48>, !dbg !309
  %550 = shufflevector <2 x i64> %547, <2 x i64> %549, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !313
  %.sroa.18893.14.vec.insert1330 = trunc <4 x i64> %550 to <4 x i16>, !dbg !313
  %551 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %268) #7, !dbg !315, !srcloc !185
  %.sroa.036.2.extract.shift37.i457 = lshr i64 %551, 16, !dbg !318
  %552 = insertelement <2 x i64> poison, i64 %551, i64 0, !dbg !318
  %553 = insertelement <2 x i64> %552, i64 %.sroa.036.2.extract.shift37.i457, i64 1, !dbg !318
  %554 = shufflevector <2 x i64> %552, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !315
  %555 = lshr <2 x i64> %554, <i64 32, i64 48>, !dbg !315
  %556 = shufflevector <2 x i64> %553, <2 x i64> %555, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !319
  %.sroa.0811.6.vec.insert1329 = trunc <4 x i64> %556 to <4 x i16>, !dbg !319
  %557 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %272) #7, !dbg !315, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i466 = lshr i64 %557, 16, !dbg !318
  %558 = insertelement <2 x i64> poison, i64 %557, i64 0, !dbg !318
  %559 = insertelement <2 x i64> %558, i64 %.sroa.036.2.extract.shift37.1.i466, i64 1, !dbg !318
  %560 = shufflevector <2 x i64> %558, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !315
  %561 = lshr <2 x i64> %560, <i64 32, i64 48>, !dbg !315
  %562 = shufflevector <2 x i64> %559, <2 x i64> %561, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !319
  %.sroa.18.14.vec.insert1328 = trunc <4 x i64> %562 to <4 x i16>, !dbg !319
  %563 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %276) #7, !dbg !321, !srcloc !185
  %.sroa.036.2.extract.shift37.i478 = lshr i64 %563, 16, !dbg !324
  %564 = insertelement <2 x i64> poison, i64 %563, i64 0, !dbg !324
  %565 = insertelement <2 x i64> %564, i64 %.sroa.036.2.extract.shift37.i478, i64 1, !dbg !324
  %566 = shufflevector <2 x i64> %564, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !321
  %567 = lshr <2 x i64> %566, <i64 32, i64 48>, !dbg !321
  %568 = shufflevector <2 x i64> %565, <2 x i64> %567, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !325
  %.sroa.01012.6.vec.insert1327 = trunc <4 x i64> %568 to <4 x i16>, !dbg !325
  %569 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %280) #7, !dbg !321, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i487 = lshr i64 %569, 16, !dbg !324
  %570 = insertelement <2 x i64> poison, i64 %569, i64 0, !dbg !324
  %571 = insertelement <2 x i64> %570, i64 %.sroa.036.2.extract.shift37.1.i487, i64 1, !dbg !324
  %572 = shufflevector <2 x i64> %570, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !321
  %573 = lshr <2 x i64> %572, <i64 32, i64 48>, !dbg !321
  %574 = shufflevector <2 x i64> %571, <2 x i64> %573, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !325
  %.sroa.181029.14.vec.insert1326 = trunc <4 x i64> %574 to <4 x i16>, !dbg !325
  %575 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %284) #7, !dbg !327, !srcloc !185
  %.sroa.036.2.extract.shift37.i499 = lshr i64 %575, 16, !dbg !330
  %576 = insertelement <2 x i64> poison, i64 %575, i64 0, !dbg !330
  %577 = insertelement <2 x i64> %576, i64 %.sroa.036.2.extract.shift37.i499, i64 1, !dbg !330
  %578 = shufflevector <2 x i64> %576, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !327
  %579 = lshr <2 x i64> %578, <i64 32, i64 48>, !dbg !327
  %580 = shufflevector <2 x i64> %577, <2 x i64> %579, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !331
  %.sroa.0944.6.vec.insert1325 = trunc <4 x i64> %580 to <4 x i16>, !dbg !331
  %581 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %288) #7, !dbg !327, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i508 = lshr i64 %581, 16, !dbg !330
  %582 = insertelement <2 x i64> poison, i64 %581, i64 0, !dbg !330
  %583 = insertelement <2 x i64> %582, i64 %.sroa.036.2.extract.shift37.1.i508, i64 1, !dbg !330
  %584 = shufflevector <2 x i64> %582, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !327
  %585 = lshr <2 x i64> %584, <i64 32, i64 48>, !dbg !327
  %586 = shufflevector <2 x i64> %583, <2 x i64> %585, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !331
  %.sroa.18961.14.vec.insert1324 = trunc <4 x i64> %586 to <4 x i16>, !dbg !331
  %587 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %293) #7, !dbg !333, !srcloc !185
  %.sroa.036.2.extract.shift37.i520 = lshr i64 %587, 16, !dbg !336
  %588 = insertelement <2 x i64> poison, i64 %587, i64 0, !dbg !336
  %589 = insertelement <2 x i64> %588, i64 %.sroa.036.2.extract.shift37.i520, i64 1, !dbg !336
  %590 = shufflevector <2 x i64> %588, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !333
  %591 = lshr <2 x i64> %590, <i64 32, i64 48>, !dbg !333
  %592 = shufflevector <2 x i64> %589, <2 x i64> %591, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !337
  %.sroa.34910.22.vec.insert1323 = trunc <4 x i64> %592 to <4 x i16>, !dbg !337
  %593 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %297) #7, !dbg !333, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i529 = lshr i64 %593, 16, !dbg !336
  %594 = insertelement <2 x i64> poison, i64 %593, i64 0, !dbg !336
  %595 = insertelement <2 x i64> %594, i64 %.sroa.036.2.extract.shift37.1.i529, i64 1, !dbg !336
  %596 = shufflevector <2 x i64> %594, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !333
  %597 = lshr <2 x i64> %596, <i64 32, i64 48>, !dbg !333
  %598 = shufflevector <2 x i64> %595, <2 x i64> %597, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !337
  %.sroa.50927.30.vec.insert1322 = trunc <4 x i64> %598 to <4 x i16>, !dbg !337
  %599 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %301) #7, !dbg !339, !srcloc !185
  %.sroa.036.2.extract.shift37.i541 = lshr i64 %599, 16, !dbg !342
  %600 = insertelement <2 x i64> poison, i64 %599, i64 0, !dbg !342
  %601 = insertelement <2 x i64> %600, i64 %.sroa.036.2.extract.shift37.i541, i64 1, !dbg !342
  %602 = shufflevector <2 x i64> %600, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !339
  %603 = lshr <2 x i64> %602, <i64 32, i64 48>, !dbg !339
  %604 = shufflevector <2 x i64> %601, <2 x i64> %603, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !343
  %.sroa.34.22.vec.insert1321 = trunc <4 x i64> %604 to <4 x i16>, !dbg !343
  %605 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %305) #7, !dbg !339, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i550 = lshr i64 %605, 16, !dbg !342
  %606 = insertelement <2 x i64> poison, i64 %605, i64 0, !dbg !342
  %607 = insertelement <2 x i64> %606, i64 %.sroa.036.2.extract.shift37.1.i550, i64 1, !dbg !342
  %608 = shufflevector <2 x i64> %606, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !339
  %609 = lshr <2 x i64> %608, <i64 32, i64 48>, !dbg !339
  %610 = shufflevector <2 x i64> %607, <2 x i64> %609, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !343
  %.sroa.50.30.vec.insert1320 = trunc <4 x i64> %610 to <4 x i16>, !dbg !343
  %611 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %309) #7, !dbg !345, !srcloc !185
  %.sroa.036.2.extract.shift37.i562 = lshr i64 %611, 16, !dbg !348
  %612 = insertelement <2 x i64> poison, i64 %611, i64 0, !dbg !348
  %613 = insertelement <2 x i64> %612, i64 %.sroa.036.2.extract.shift37.i562, i64 1, !dbg !348
  %614 = shufflevector <2 x i64> %612, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !345
  %615 = lshr <2 x i64> %614, <i64 32, i64 48>, !dbg !345
  %616 = shufflevector <2 x i64> %613, <2 x i64> %615, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !349
  %.sroa.341046.22.vec.insert1319 = trunc <4 x i64> %616 to <4 x i16>, !dbg !349
  %617 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %313) #7, !dbg !345, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i571 = lshr i64 %617, 16, !dbg !348
  %618 = insertelement <2 x i64> poison, i64 %617, i64 0, !dbg !348
  %619 = insertelement <2 x i64> %618, i64 %.sroa.036.2.extract.shift37.1.i571, i64 1, !dbg !348
  %620 = shufflevector <2 x i64> %618, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !345
  %621 = lshr <2 x i64> %620, <i64 32, i64 48>, !dbg !345
  %622 = shufflevector <2 x i64> %619, <2 x i64> %621, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !349
  %.sroa.501063.30.vec.insert1318 = trunc <4 x i64> %622 to <4 x i16>, !dbg !349
  %623 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %317) #7, !dbg !351, !srcloc !185
  %.sroa.036.2.extract.shift37.i583 = lshr i64 %623, 16, !dbg !354
  %624 = insertelement <2 x i64> poison, i64 %623, i64 0, !dbg !354
  %625 = insertelement <2 x i64> %624, i64 %.sroa.036.2.extract.shift37.i583, i64 1, !dbg !354
  %626 = shufflevector <2 x i64> %624, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !351
  %627 = lshr <2 x i64> %626, <i64 32, i64 48>, !dbg !351
  %628 = shufflevector <2 x i64> %625, <2 x i64> %627, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !355
  %.sroa.34978.22.vec.insert1317 = trunc <4 x i64> %628 to <4 x i16>, !dbg !355
  %629 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %321) #7, !dbg !351, !srcloc !185
  %.sroa.036.2.extract.shift37.1.i592 = lshr i64 %629, 16, !dbg !354
  %630 = insertelement <2 x i64> poison, i64 %629, i64 0, !dbg !354
  %631 = insertelement <2 x i64> %630, i64 %.sroa.036.2.extract.shift37.1.i592, i64 1, !dbg !354
  %632 = shufflevector <2 x i64> %630, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !351
  %633 = lshr <2 x i64> %632, <i64 32, i64 48>, !dbg !351
  %634 = shufflevector <2 x i64> %631, <2 x i64> %633, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !355
  %.sroa.50995.30.vec.insert1316 = trunc <4 x i64> %634 to <4 x i16>, !dbg !355
  tail call void @llvm.amdgcn.s.barrier(), !dbg !357
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !358
  tail call void asm sideeffect "s_waitcnt lgkmcnt(8)", ""() #7, !dbg !359, !srcloc !360
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !361
  %635 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert1327, <4 x i16> %.sroa.0876.6.vec.insert1331, <4 x float> %523, i32 0, i32 0, i32 0), !dbg !362
  %636 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert1327, <4 x i16> %.sroa.18893.14.vec.insert1330, <4 x float> %524, i32 0, i32 0, i32 0), !dbg !362
  %637 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert1326, <4 x i16> %.sroa.0876.6.vec.insert1331, <4 x float> %525, i32 0, i32 0, i32 0), !dbg !362
  %638 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert1326, <4 x i16> %.sroa.18893.14.vec.insert1330, <4 x float> %526, i32 0, i32 0, i32 0), !dbg !362
  %639 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert1327, <4 x i16> %.sroa.0811.6.vec.insert1329, <4 x float> %527, i32 0, i32 0, i32 0), !dbg !366
  %640 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.01012.6.vec.insert1327, <4 x i16> %.sroa.18.14.vec.insert1328, <4 x float> %528, i32 0, i32 0, i32 0), !dbg !366
  %641 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert1326, <4 x i16> %.sroa.0811.6.vec.insert1329, <4 x float> %529, i32 0, i32 0, i32 0), !dbg !366
  %642 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.181029.14.vec.insert1326, <4 x i16> %.sroa.18.14.vec.insert1328, <4 x float> %530, i32 0, i32 0, i32 0), !dbg !366
  %643 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert1325, <4 x i16> %.sroa.0876.6.vec.insert1331, <4 x float> %531, i32 0, i32 0, i32 0), !dbg !370
  %644 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert1325, <4 x i16> %.sroa.18893.14.vec.insert1330, <4 x float> %532, i32 0, i32 0, i32 0), !dbg !370
  %645 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert1324, <4 x i16> %.sroa.0876.6.vec.insert1331, <4 x float> %533, i32 0, i32 0, i32 0), !dbg !370
  %646 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert1324, <4 x i16> %.sroa.18893.14.vec.insert1330, <4 x float> %534, i32 0, i32 0, i32 0), !dbg !370
  %647 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert1325, <4 x i16> %.sroa.0811.6.vec.insert1329, <4 x float> %535, i32 0, i32 0, i32 0), !dbg !374
  %648 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0944.6.vec.insert1325, <4 x i16> %.sroa.18.14.vec.insert1328, <4 x float> %536, i32 0, i32 0, i32 0), !dbg !374
  %649 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert1324, <4 x i16> %.sroa.0811.6.vec.insert1329, <4 x float> %537, i32 0, i32 0, i32 0), !dbg !374
  %650 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18961.14.vec.insert1324, <4 x i16> %.sroa.18.14.vec.insert1328, <4 x float> %538, i32 0, i32 0, i32 0), !dbg !374
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !378
  tail call void @llvm.amdgcn.s.barrier(), !dbg !379
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !380
  br i1 %.not, label %652, label %651, !dbg !381

651:                                              ; preds = %522
  tail call void asm sideeffect "s_waitcnt vmcnt(4)", ""() #7, !dbg !382, !srcloc !383
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %329, i64 %.sroa.0808.1.off0) #7, !dbg !384, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %333, i64 %.sroa.0808.1.off64) #7, !dbg !390, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %337, i64 %.sroa.5809.1.off0) #7, !dbg !384, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %341, i64 %.sroa.5809.1.off64) #7, !dbg !390, !srcloc !389
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #7, !dbg !392, !srcloc !393
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %347, i64 %.sroa.0806.1.off0) #7, !dbg !394, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %351, i64 %.sroa.0806.1.off64) #7, !dbg !398, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %355, i64 %.sroa.5.1.off0) #7, !dbg !394, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %359, i64 %.sroa.5.1.off64) #7, !dbg !398, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %363, i64 %.sroa.8807.1.off0) #7, !dbg !394, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %367, i64 %.sroa.8807.1.off64) #7, !dbg !398, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %371, i64 %.sroa.11.1.off0) #7, !dbg !394, !srcloc !389
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %375, i64 %.sroa.11.1.off64) #7, !dbg !398, !srcloc !389
  br label %652, !dbg !400

652:                                              ; preds = %651, %522
  tail call void @llvm.amdgcn.s.barrier(), !dbg !401
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !402
  tail call void asm sideeffect "s_waitcnt lgkmcnt(12)", ""() #7, !dbg !403, !srcloc !404
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !405
  %653 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert1319, <4 x i16> %.sroa.34910.22.vec.insert1323, <4 x float> %635, i32 0, i32 0, i32 0), !dbg !406
  %654 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert1319, <4 x i16> %.sroa.50927.30.vec.insert1322, <4 x float> %636, i32 0, i32 0, i32 0), !dbg !406
  %655 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert1318, <4 x i16> %.sroa.34910.22.vec.insert1323, <4 x float> %637, i32 0, i32 0, i32 0), !dbg !406
  %656 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert1318, <4 x i16> %.sroa.50927.30.vec.insert1322, <4 x float> %638, i32 0, i32 0, i32 0), !dbg !406
  %657 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert1319, <4 x i16> %.sroa.34.22.vec.insert1321, <4 x float> %639, i32 0, i32 0, i32 0), !dbg !410
  %658 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.341046.22.vec.insert1319, <4 x i16> %.sroa.50.30.vec.insert1320, <4 x float> %640, i32 0, i32 0, i32 0), !dbg !410
  %659 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert1318, <4 x i16> %.sroa.34.22.vec.insert1321, <4 x float> %641, i32 0, i32 0, i32 0), !dbg !410
  %660 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.501063.30.vec.insert1318, <4 x i16> %.sroa.50.30.vec.insert1320, <4 x float> %642, i32 0, i32 0, i32 0), !dbg !410
  %661 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert1317, <4 x i16> %.sroa.34910.22.vec.insert1323, <4 x float> %643, i32 0, i32 0, i32 0), !dbg !414
  %662 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert1317, <4 x i16> %.sroa.50927.30.vec.insert1322, <4 x float> %644, i32 0, i32 0, i32 0), !dbg !414
  %663 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert1316, <4 x i16> %.sroa.34910.22.vec.insert1323, <4 x float> %645, i32 0, i32 0, i32 0), !dbg !414
  %664 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert1316, <4 x i16> %.sroa.50927.30.vec.insert1322, <4 x float> %646, i32 0, i32 0, i32 0), !dbg !414
  %665 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert1317, <4 x i16> %.sroa.34.22.vec.insert1321, <4 x float> %647, i32 0, i32 0, i32 0), !dbg !418
  %666 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34978.22.vec.insert1317, <4 x i16> %.sroa.50.30.vec.insert1320, <4 x float> %648, i32 0, i32 0, i32 0), !dbg !418
  %667 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert1316, <4 x i16> %.sroa.34.22.vec.insert1321, <4 x float> %649, i32 0, i32 0, i32 0), !dbg !418
  %668 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50995.30.vec.insert1316, <4 x i16> %.sroa.50.30.vec.insert1320, <4 x float> %650, i32 0, i32 0, i32 0), !dbg !418
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !422
  tail call void @llvm.amdgcn.s.barrier(), !dbg !423
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !424
  %exitcond.not = icmp eq i32 %388, 128, !dbg !425
  br i1 %exitcond.not, label %376, label %378, !dbg !174, !llvm.loop !426

669:                                              ; preds = %376
  tail call void @llvm.amdgcn.s.barrier(), !dbg !429
  br label %670, !dbg !430

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
  %690 = add nsw i32 %675, %162
  %691 = sext i32 %690 to i64
  %692 = add nsw i32 %677, %162
  %693 = sext i32 %692 to i64
  %694 = add nsw i32 %679, %162
  %695 = sext i32 %694 to i64
  %696 = add nsw i32 %681, %162
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
  %714 = add nsw i32 %699, %162
  %715 = sext i32 %714 to i64
  %716 = add nsw i32 %701, %162
  %717 = sext i32 %716 to i64
  %718 = add nsw i32 %703, %162
  %719 = sext i32 %718 to i64
  %720 = add nsw i32 %705, %162
  %721 = sext i32 %720 to i64
  %722 = or disjoint i32 %673, %37, !dbg !431
  %723 = shl nsw i32 %672, 5, !dbg !432
  %724 = shl nsw i32 %722, 5, !dbg !440
  %725 = sext i32 %723 to i64, !dbg !441
  %726 = mul i64 %.sroa.20.0.copyload, %725, !dbg !444
  %727 = sext i32 %724 to i64, !dbg !445
  %728 = getelementptr float, ptr %.sroa.171091.0.copyload, i64 %726, !dbg !446
  %729 = getelementptr float, ptr %728, i64 %727, !dbg !446
  %.sroa.0.0.vec.extract = extractelement <4 x float> %653, i64 0, !dbg !447
  %730 = getelementptr inbounds float, ptr %729, i64 %683, !dbg !448
  store float %.sroa.0.0.vec.extract, ptr %730, align 4, !dbg !449, !tbaa !450
  %.sroa.0.4.vec.extract = extractelement <4 x float> %653, i64 1, !dbg !454
  %731 = getelementptr inbounds float, ptr %729, i64 %685, !dbg !455
  store float %.sroa.0.4.vec.extract, ptr %731, align 4, !dbg !456, !tbaa !450
  %.sroa.0.8.vec.extract = extractelement <4 x float> %653, i64 2, !dbg !457
  %732 = getelementptr inbounds float, ptr %729, i64 %687, !dbg !458
  store float %.sroa.0.8.vec.extract, ptr %732, align 4, !dbg !459, !tbaa !450
  %.sroa.0.12.vec.extract = extractelement <4 x float> %653, i64 3, !dbg !460
  %733 = getelementptr inbounds float, ptr %729, i64 %689, !dbg !461
  store float %.sroa.0.12.vec.extract, ptr %733, align 4, !dbg !462, !tbaa !450
  %.sroa.12.16.vec.extract = extractelement <4 x float> %654, i64 0, !dbg !447
  %734 = getelementptr inbounds float, ptr %729, i64 %691, !dbg !448
  store float %.sroa.12.16.vec.extract, ptr %734, align 4, !dbg !449, !tbaa !450
  %.sroa.12.20.vec.extract = extractelement <4 x float> %654, i64 1, !dbg !454
  %735 = getelementptr inbounds float, ptr %729, i64 %693, !dbg !455
  store float %.sroa.12.20.vec.extract, ptr %735, align 4, !dbg !456, !tbaa !450
  %.sroa.12.24.vec.extract = extractelement <4 x float> %654, i64 2, !dbg !457
  %736 = getelementptr inbounds float, ptr %729, i64 %695, !dbg !458
  store float %.sroa.12.24.vec.extract, ptr %736, align 4, !dbg !459, !tbaa !450
  %.sroa.12.28.vec.extract = extractelement <4 x float> %654, i64 3, !dbg !460
  %737 = getelementptr inbounds float, ptr %729, i64 %697, !dbg !461
  store float %.sroa.12.28.vec.extract, ptr %737, align 4, !dbg !462, !tbaa !450
  %.sroa.22.32.vec.extract = extractelement <4 x float> %655, i64 0, !dbg !447
  %738 = getelementptr inbounds float, ptr %729, i64 %707, !dbg !448
  store float %.sroa.22.32.vec.extract, ptr %738, align 4, !dbg !449, !tbaa !450
  %.sroa.22.36.vec.extract = extractelement <4 x float> %655, i64 1, !dbg !454
  %739 = getelementptr inbounds float, ptr %729, i64 %709, !dbg !455
  store float %.sroa.22.36.vec.extract, ptr %739, align 4, !dbg !456, !tbaa !450
  %.sroa.22.40.vec.extract = extractelement <4 x float> %655, i64 2, !dbg !457
  %740 = getelementptr inbounds float, ptr %729, i64 %711, !dbg !458
  store float %.sroa.22.40.vec.extract, ptr %740, align 4, !dbg !459, !tbaa !450
  %.sroa.22.44.vec.extract = extractelement <4 x float> %655, i64 3, !dbg !460
  %741 = getelementptr inbounds float, ptr %729, i64 %713, !dbg !461
  store float %.sroa.22.44.vec.extract, ptr %741, align 4, !dbg !462, !tbaa !450
  %.sroa.32.48.vec.extract = extractelement <4 x float> %656, i64 0, !dbg !447
  %742 = getelementptr inbounds float, ptr %729, i64 %715, !dbg !448
  store float %.sroa.32.48.vec.extract, ptr %742, align 4, !dbg !449, !tbaa !450
  %.sroa.32.52.vec.extract = extractelement <4 x float> %656, i64 1, !dbg !454
  %743 = getelementptr inbounds float, ptr %729, i64 %717, !dbg !455
  store float %.sroa.32.52.vec.extract, ptr %743, align 4, !dbg !456, !tbaa !450
  %.sroa.32.56.vec.extract = extractelement <4 x float> %656, i64 2, !dbg !457
  %744 = getelementptr inbounds float, ptr %729, i64 %719, !dbg !458
  store float %.sroa.32.56.vec.extract, ptr %744, align 4, !dbg !459, !tbaa !450
  %.sroa.32.60.vec.extract = extractelement <4 x float> %656, i64 3, !dbg !460
  %745 = getelementptr inbounds float, ptr %729, i64 %721, !dbg !461
  store float %.sroa.32.60.vec.extract, ptr %745, align 4, !dbg !462, !tbaa !450
  %746 = or disjoint i32 %673, %37, !dbg !431
  %747 = shl i32 %746, 5, !dbg !440
  %748 = or disjoint i32 %747, 128, !dbg !440
  %749 = sext i32 %748 to i64, !dbg !445
  %750 = getelementptr float, ptr %728, i64 %749, !dbg !446
  %.sroa.42.64.vec.extract = extractelement <4 x float> %657, i64 0, !dbg !447
  %751 = getelementptr inbounds float, ptr %750, i64 %683, !dbg !448
  store float %.sroa.42.64.vec.extract, ptr %751, align 4, !dbg !449, !tbaa !450
  %.sroa.42.68.vec.extract = extractelement <4 x float> %657, i64 1, !dbg !454
  %752 = getelementptr inbounds float, ptr %750, i64 %685, !dbg !455
  store float %.sroa.42.68.vec.extract, ptr %752, align 4, !dbg !456, !tbaa !450
  %.sroa.42.72.vec.extract = extractelement <4 x float> %657, i64 2, !dbg !457
  %753 = getelementptr inbounds float, ptr %750, i64 %687, !dbg !458
  store float %.sroa.42.72.vec.extract, ptr %753, align 4, !dbg !459, !tbaa !450
  %.sroa.42.76.vec.extract = extractelement <4 x float> %657, i64 3, !dbg !460
  %754 = getelementptr inbounds float, ptr %750, i64 %689, !dbg !461
  store float %.sroa.42.76.vec.extract, ptr %754, align 4, !dbg !462, !tbaa !450
  %.sroa.52.80.vec.extract = extractelement <4 x float> %658, i64 0, !dbg !447
  %755 = getelementptr inbounds float, ptr %750, i64 %691, !dbg !448
  store float %.sroa.52.80.vec.extract, ptr %755, align 4, !dbg !449, !tbaa !450
  %.sroa.52.84.vec.extract = extractelement <4 x float> %658, i64 1, !dbg !454
  %756 = getelementptr inbounds float, ptr %750, i64 %693, !dbg !455
  store float %.sroa.52.84.vec.extract, ptr %756, align 4, !dbg !456, !tbaa !450
  %.sroa.52.88.vec.extract = extractelement <4 x float> %658, i64 2, !dbg !457
  %757 = getelementptr inbounds float, ptr %750, i64 %695, !dbg !458
  store float %.sroa.52.88.vec.extract, ptr %757, align 4, !dbg !459, !tbaa !450
  %.sroa.52.92.vec.extract = extractelement <4 x float> %658, i64 3, !dbg !460
  %758 = getelementptr inbounds float, ptr %750, i64 %697, !dbg !461
  store float %.sroa.52.92.vec.extract, ptr %758, align 4, !dbg !462, !tbaa !450
  %.sroa.62.96.vec.extract = extractelement <4 x float> %659, i64 0, !dbg !447
  %759 = getelementptr inbounds float, ptr %750, i64 %707, !dbg !448
  store float %.sroa.62.96.vec.extract, ptr %759, align 4, !dbg !449, !tbaa !450
  %.sroa.62.100.vec.extract = extractelement <4 x float> %659, i64 1, !dbg !454
  %760 = getelementptr inbounds float, ptr %750, i64 %709, !dbg !455
  store float %.sroa.62.100.vec.extract, ptr %760, align 4, !dbg !456, !tbaa !450
  %.sroa.62.104.vec.extract = extractelement <4 x float> %659, i64 2, !dbg !457
  %761 = getelementptr inbounds float, ptr %750, i64 %711, !dbg !458
  store float %.sroa.62.104.vec.extract, ptr %761, align 4, !dbg !459, !tbaa !450
  %.sroa.62.108.vec.extract = extractelement <4 x float> %659, i64 3, !dbg !460
  %762 = getelementptr inbounds float, ptr %750, i64 %713, !dbg !461
  store float %.sroa.62.108.vec.extract, ptr %762, align 4, !dbg !462, !tbaa !450
  %.sroa.72.112.vec.extract = extractelement <4 x float> %660, i64 0, !dbg !447
  %763 = getelementptr inbounds float, ptr %750, i64 %715, !dbg !448
  store float %.sroa.72.112.vec.extract, ptr %763, align 4, !dbg !449, !tbaa !450
  %.sroa.72.116.vec.extract = extractelement <4 x float> %660, i64 1, !dbg !454
  %764 = getelementptr inbounds float, ptr %750, i64 %717, !dbg !455
  store float %.sroa.72.116.vec.extract, ptr %764, align 4, !dbg !456, !tbaa !450
  %.sroa.72.120.vec.extract = extractelement <4 x float> %660, i64 2, !dbg !457
  %765 = getelementptr inbounds float, ptr %750, i64 %719, !dbg !458
  store float %.sroa.72.120.vec.extract, ptr %765, align 4, !dbg !459, !tbaa !450
  %.sroa.72.124.vec.extract = extractelement <4 x float> %660, i64 3, !dbg !460
  %766 = getelementptr inbounds float, ptr %750, i64 %721, !dbg !461
  store float %.sroa.72.124.vec.extract, ptr %766, align 4, !dbg !462, !tbaa !450
  %767 = shl i32 %672, 5, !dbg !432
  %768 = add i32 %767, 64, !dbg !432
  %769 = sext i32 %768 to i64, !dbg !441
  %770 = mul i64 %.sroa.20.0.copyload, %769, !dbg !444
  %771 = getelementptr float, ptr %.sroa.171091.0.copyload, i64 %770, !dbg !446
  %772 = getelementptr float, ptr %771, i64 %727, !dbg !446
  %.sroa.82.128.vec.extract = extractelement <4 x float> %661, i64 0, !dbg !447
  %773 = getelementptr inbounds float, ptr %772, i64 %683, !dbg !448
  store float %.sroa.82.128.vec.extract, ptr %773, align 4, !dbg !449, !tbaa !450
  %.sroa.82.132.vec.extract = extractelement <4 x float> %661, i64 1, !dbg !454
  %774 = getelementptr inbounds float, ptr %772, i64 %685, !dbg !455
  store float %.sroa.82.132.vec.extract, ptr %774, align 4, !dbg !456, !tbaa !450
  %.sroa.82.136.vec.extract = extractelement <4 x float> %661, i64 2, !dbg !457
  %775 = getelementptr inbounds float, ptr %772, i64 %687, !dbg !458
  store float %.sroa.82.136.vec.extract, ptr %775, align 4, !dbg !459, !tbaa !450
  %.sroa.82.140.vec.extract = extractelement <4 x float> %661, i64 3, !dbg !460
  %776 = getelementptr inbounds float, ptr %772, i64 %689, !dbg !461
  store float %.sroa.82.140.vec.extract, ptr %776, align 4, !dbg !462, !tbaa !450
  %.sroa.92.144.vec.extract = extractelement <4 x float> %662, i64 0, !dbg !447
  %777 = getelementptr inbounds float, ptr %772, i64 %691, !dbg !448
  store float %.sroa.92.144.vec.extract, ptr %777, align 4, !dbg !449, !tbaa !450
  %.sroa.92.148.vec.extract = extractelement <4 x float> %662, i64 1, !dbg !454
  %778 = getelementptr inbounds float, ptr %772, i64 %693, !dbg !455
  store float %.sroa.92.148.vec.extract, ptr %778, align 4, !dbg !456, !tbaa !450
  %.sroa.92.152.vec.extract = extractelement <4 x float> %662, i64 2, !dbg !457
  %779 = getelementptr inbounds float, ptr %772, i64 %695, !dbg !458
  store float %.sroa.92.152.vec.extract, ptr %779, align 4, !dbg !459, !tbaa !450
  %.sroa.92.156.vec.extract = extractelement <4 x float> %662, i64 3, !dbg !460
  %780 = getelementptr inbounds float, ptr %772, i64 %697, !dbg !461
  store float %.sroa.92.156.vec.extract, ptr %780, align 4, !dbg !462, !tbaa !450
  %.sroa.102.160.vec.extract = extractelement <4 x float> %663, i64 0, !dbg !447
  %781 = getelementptr inbounds float, ptr %772, i64 %707, !dbg !448
  store float %.sroa.102.160.vec.extract, ptr %781, align 4, !dbg !449, !tbaa !450
  %.sroa.102.164.vec.extract = extractelement <4 x float> %663, i64 1, !dbg !454
  %782 = getelementptr inbounds float, ptr %772, i64 %709, !dbg !455
  store float %.sroa.102.164.vec.extract, ptr %782, align 4, !dbg !456, !tbaa !450
  %.sroa.102.168.vec.extract = extractelement <4 x float> %663, i64 2, !dbg !457
  %783 = getelementptr inbounds float, ptr %772, i64 %711, !dbg !458
  store float %.sroa.102.168.vec.extract, ptr %783, align 4, !dbg !459, !tbaa !450
  %.sroa.102.172.vec.extract = extractelement <4 x float> %663, i64 3, !dbg !460
  %784 = getelementptr inbounds float, ptr %772, i64 %713, !dbg !461
  store float %.sroa.102.172.vec.extract, ptr %784, align 4, !dbg !462, !tbaa !450
  %.sroa.112.176.vec.extract = extractelement <4 x float> %664, i64 0, !dbg !447
  %785 = getelementptr inbounds float, ptr %772, i64 %715, !dbg !448
  store float %.sroa.112.176.vec.extract, ptr %785, align 4, !dbg !449, !tbaa !450
  %.sroa.112.180.vec.extract = extractelement <4 x float> %664, i64 1, !dbg !454
  %786 = getelementptr inbounds float, ptr %772, i64 %717, !dbg !455
  store float %.sroa.112.180.vec.extract, ptr %786, align 4, !dbg !456, !tbaa !450
  %.sroa.112.184.vec.extract = extractelement <4 x float> %664, i64 2, !dbg !457
  %787 = getelementptr inbounds float, ptr %772, i64 %719, !dbg !458
  store float %.sroa.112.184.vec.extract, ptr %787, align 4, !dbg !459, !tbaa !450
  %.sroa.112.188.vec.extract = extractelement <4 x float> %664, i64 3, !dbg !460
  %788 = getelementptr inbounds float, ptr %772, i64 %721, !dbg !461
  store float %.sroa.112.188.vec.extract, ptr %788, align 4, !dbg !462, !tbaa !450
  %789 = getelementptr float, ptr %771, i64 %749, !dbg !446
  %.sroa.122.192.vec.extract = extractelement <4 x float> %665, i64 0, !dbg !447
  %790 = getelementptr inbounds float, ptr %789, i64 %683, !dbg !448
  store float %.sroa.122.192.vec.extract, ptr %790, align 4, !dbg !449, !tbaa !450
  %.sroa.122.196.vec.extract = extractelement <4 x float> %665, i64 1, !dbg !454
  %791 = getelementptr inbounds float, ptr %789, i64 %685, !dbg !455
  store float %.sroa.122.196.vec.extract, ptr %791, align 4, !dbg !456, !tbaa !450
  %.sroa.122.200.vec.extract = extractelement <4 x float> %665, i64 2, !dbg !457
  %792 = getelementptr inbounds float, ptr %789, i64 %687, !dbg !458
  store float %.sroa.122.200.vec.extract, ptr %792, align 4, !dbg !459, !tbaa !450
  %.sroa.122.204.vec.extract = extractelement <4 x float> %665, i64 3, !dbg !460
  %793 = getelementptr inbounds float, ptr %789, i64 %689, !dbg !461
  store float %.sroa.122.204.vec.extract, ptr %793, align 4, !dbg !462, !tbaa !450
  %.sroa.132.208.vec.extract = extractelement <4 x float> %666, i64 0, !dbg !447
  %794 = getelementptr inbounds float, ptr %789, i64 %691, !dbg !448
  store float %.sroa.132.208.vec.extract, ptr %794, align 4, !dbg !449, !tbaa !450
  %.sroa.132.212.vec.extract = extractelement <4 x float> %666, i64 1, !dbg !454
  %795 = getelementptr inbounds float, ptr %789, i64 %693, !dbg !455
  store float %.sroa.132.212.vec.extract, ptr %795, align 4, !dbg !456, !tbaa !450
  %.sroa.132.216.vec.extract = extractelement <4 x float> %666, i64 2, !dbg !457
  %796 = getelementptr inbounds float, ptr %789, i64 %695, !dbg !458
  store float %.sroa.132.216.vec.extract, ptr %796, align 4, !dbg !459, !tbaa !450
  %.sroa.132.220.vec.extract = extractelement <4 x float> %666, i64 3, !dbg !460
  %797 = getelementptr inbounds float, ptr %789, i64 %697, !dbg !461
  store float %.sroa.132.220.vec.extract, ptr %797, align 4, !dbg !462, !tbaa !450
  %.sroa.142.224.vec.extract = extractelement <4 x float> %667, i64 0, !dbg !447
  %798 = getelementptr inbounds float, ptr %789, i64 %707, !dbg !448
  store float %.sroa.142.224.vec.extract, ptr %798, align 4, !dbg !449, !tbaa !450
  %.sroa.142.228.vec.extract = extractelement <4 x float> %667, i64 1, !dbg !454
  %799 = getelementptr inbounds float, ptr %789, i64 %709, !dbg !455
  store float %.sroa.142.228.vec.extract, ptr %799, align 4, !dbg !456, !tbaa !450
  %.sroa.142.232.vec.extract = extractelement <4 x float> %667, i64 2, !dbg !457
  %800 = getelementptr inbounds float, ptr %789, i64 %711, !dbg !458
  store float %.sroa.142.232.vec.extract, ptr %800, align 4, !dbg !459, !tbaa !450
  %.sroa.142.236.vec.extract = extractelement <4 x float> %667, i64 3, !dbg !460
  %801 = getelementptr inbounds float, ptr %789, i64 %713, !dbg !461
  store float %.sroa.142.236.vec.extract, ptr %801, align 4, !dbg !462, !tbaa !450
  %.sroa.152.240.vec.extract = extractelement <4 x float> %668, i64 0, !dbg !447
  %802 = getelementptr inbounds float, ptr %789, i64 %715, !dbg !448
  store float %.sroa.152.240.vec.extract, ptr %802, align 4, !dbg !449, !tbaa !450
  %.sroa.152.244.vec.extract = extractelement <4 x float> %668, i64 1, !dbg !454
  %803 = getelementptr inbounds float, ptr %789, i64 %717, !dbg !455
  store float %.sroa.152.244.vec.extract, ptr %803, align 4, !dbg !456, !tbaa !450
  %.sroa.152.248.vec.extract = extractelement <4 x float> %668, i64 2, !dbg !457
  %804 = getelementptr inbounds float, ptr %789, i64 %719, !dbg !458
  store float %.sroa.152.248.vec.extract, ptr %804, align 4, !dbg !459, !tbaa !450
  %.sroa.152.252.vec.extract = extractelement <4 x float> %668, i64 3, !dbg !460
  %805 = getelementptr inbounds float, ptr %789, i64 %721, !dbg !461
  store float %.sroa.152.252.vec.extract, ptr %805, align 4, !dbg !462, !tbaa !450
  ret void, !dbg !463
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
!1 = !DIFile(filename: "256_128_64_16.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_16")
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
!78 = distinct !DILocation(line: 85, column: 25, scope: !9)
!79 = !DILocation(line: 57, column: 97, scope: !77, inlinedAt: !78)
!80 = !DILocation(line: 86, column: 34, scope: !9)
!81 = !DILocation(line: 87, column: 34, scope: !9)
!82 = !DILocation(line: 30, column: 28, scope: !83, inlinedAt: !85)
!83 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !84, file: !84, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!84 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!85 = distinct !DILocation(line: 12, column: 5, scope: !86, inlinedAt: !88)
!86 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> > >", scope: !87, file: !87, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!87 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!88 = distinct !DILocation(line: 92, column: 5, scope: !9)
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
!136 = distinct !DILocation(line: 93, column: 5, scope: !9)
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
!169 = !DILocation(line: 94, column: 5, scope: !9)
!170 = !DILocation(line: 96, column: 18, scope: !9)
!171 = !DILocation(line: 96, column: 9, scope: !9)
!172 = !DILocation(line: 97, column: 9, scope: !9)
!173 = !DILocation(line: 98, column: 5, scope: !9)
!174 = !DILocation(line: 100, column: 5, scope: !9)
!175 = !DILocation(line: 202, column: 18, scope: !9)
!176 = !DILocation(line: 202, column: 9, scope: !9)
!177 = !DILocation(line: 101, column: 35, scope: !9)
!178 = !DILocation(line: 101, column: 39, scope: !9)
!179 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !182)
!180 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !181, file: !181, line: 127, type: !10, scopeLine: 127, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!181 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_16")
!182 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !184)
!183 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 64>, 32, 16> >", scope: !181, file: !181, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!184 = distinct !DILocation(line: 104, column: 9, scope: !9)
!185 = !{i64 4294467}
!186 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !184)
!187 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !190)
!188 = distinct !DISubprogram(name: "operator=", scope: !189, file: !189, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!189 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!190 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !184)
!191 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !192)
!192 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !193)
!193 = distinct !DILocation(line: 105, column: 9, scope: !9)
!194 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !193)
!195 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !196)
!196 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !193)
!197 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !198)
!198 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !200)
!199 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 32, 16> >", scope: !181, file: !181, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!200 = distinct !DILocation(line: 106, column: 9, scope: !9)
!201 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !200)
!202 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !203)
!203 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !200)
!204 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !205)
!205 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !206)
!206 = distinct !DILocation(line: 107, column: 9, scope: !9)
!207 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !206)
!208 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !209)
!209 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !206)
!210 = !DILocation(line: 108, column: 9, scope: !9)
!211 = !DILocation(line: 110, column: 13, scope: !9)
!212 = !DILocation(line: 26, column: 23, scope: !213, inlinedAt: !214)
!213 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !181, file: !181, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!214 = distinct !DILocation(line: 77, column: 39, scope: !215, inlinedAt: !216)
!215 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !181, file: !181, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!216 = distinct !DILocation(line: 111, column: 13, scope: !9)
!217 = !DILocation(line: 113, column: 9, scope: !9)
!218 = !DILocation(line: 114, column: 9, scope: !9)
!219 = !DILocation(line: 115, column: 9, scope: !9)
!220 = !DILocation(line: 118, column: 9, scope: !9)
!221 = !{i64 4303163}
!222 = !DILocation(line: 119, column: 9, scope: !9)
!223 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !226)
!224 = distinct !DISubprogram(name: "mfma161616", scope: !225, file: !225, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!225 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!226 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !228)
!227 = distinct !DISubprogram(name: "mma_ABt_base", scope: !225, file: !225, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!228 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !230)
!229 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 16, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !225, file: !225, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!230 = distinct !DILocation(line: 120, column: 9, scope: !9)
!231 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !232)
!232 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !233)
!233 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !234)
!234 = distinct !DILocation(line: 121, column: 9, scope: !9)
!235 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !236)
!236 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !237)
!237 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !238)
!238 = distinct !DILocation(line: 122, column: 9, scope: !9)
!239 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !240)
!240 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !241)
!241 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !242)
!242 = distinct !DILocation(line: 123, column: 9, scope: !9)
!243 = !DILocation(line: 124, column: 9, scope: !9)
!244 = !DILocation(line: 125, column: 9, scope: !9)
!245 = !DILocation(line: 126, column: 9, scope: !9)
!246 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !247)
!247 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !248)
!248 = distinct !DILocation(line: 129, column: 9, scope: !9)
!249 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !248)
!250 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !251)
!251 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !248)
!252 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !253)
!253 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !254)
!254 = distinct !DILocation(line: 130, column: 9, scope: !9)
!255 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !254)
!256 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !257)
!257 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !254)
!258 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !259)
!259 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !260)
!260 = distinct !DILocation(line: 131, column: 9, scope: !9)
!261 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !260)
!262 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !263)
!263 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !260)
!264 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !265)
!265 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !266)
!266 = distinct !DILocation(line: 132, column: 9, scope: !9)
!267 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !266)
!268 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !269)
!269 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !266)
!270 = !DILocation(line: 133, column: 9, scope: !9)
!271 = !DILocation(line: 135, column: 13, scope: !9)
!272 = !DILocation(line: 61, column: 18, scope: !138, inlinedAt: !273)
!273 = distinct !DILocation(line: 54, column: 39, scope: !274, inlinedAt: !275)
!274 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !181, file: !181, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!275 = distinct !DILocation(line: 136, column: 13, scope: !9)
!276 = !DILocation(line: 63, column: 68, scope: !94, inlinedAt: !277)
!277 = distinct !DILocation(line: 55, column: 24, scope: !274, inlinedAt: !275)
!278 = !DILocation(line: 63, column: 16, scope: !94, inlinedAt: !277)
!279 = !DILocation(line: 32, column: 28, scope: !280, inlinedAt: !281)
!280 = distinct !DISubprogram(name: "make_srsrc", scope: !181, file: !181, line: 31, type: !10, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!281 = distinct !DILocation(line: 60, column: 19, scope: !274, inlinedAt: !275)
!282 = !DILocation(line: 26, column: 23, scope: !213, inlinedAt: !283)
!283 = distinct !DILocation(line: 77, column: 39, scope: !274, inlinedAt: !275)
!284 = !DILocation(line: 138, column: 9, scope: !9)
!285 = !DILocation(line: 139, column: 9, scope: !9)
!286 = !DILocation(line: 140, column: 9, scope: !9)
!287 = !DILocation(line: 143, column: 9, scope: !9)
!288 = !{i64 4304460}
!289 = !DILocation(line: 144, column: 9, scope: !9)
!290 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !291)
!291 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !292)
!292 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !293)
!293 = distinct !DILocation(line: 145, column: 9, scope: !9)
!294 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !295)
!295 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !296)
!296 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !297)
!297 = distinct !DILocation(line: 146, column: 9, scope: !9)
!298 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !299)
!299 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !300)
!300 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !301)
!301 = distinct !DILocation(line: 147, column: 9, scope: !9)
!302 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !303)
!303 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !304)
!304 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !305)
!305 = distinct !DILocation(line: 148, column: 9, scope: !9)
!306 = !DILocation(line: 149, column: 9, scope: !9)
!307 = !DILocation(line: 150, column: 9, scope: !9)
!308 = !DILocation(line: 151, column: 9, scope: !9)
!309 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !310)
!310 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !311)
!311 = distinct !DILocation(line: 155, column: 9, scope: !9)
!312 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !311)
!313 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !314)
!314 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !311)
!315 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !316)
!316 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !317)
!317 = distinct !DILocation(line: 156, column: 9, scope: !9)
!318 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !317)
!319 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !320)
!320 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !317)
!321 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !322)
!322 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !323)
!323 = distinct !DILocation(line: 157, column: 9, scope: !9)
!324 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !323)
!325 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !326)
!326 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !323)
!327 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !328)
!328 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !329)
!329 = distinct !DILocation(line: 158, column: 9, scope: !9)
!330 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !329)
!331 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !332)
!332 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !329)
!333 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !334)
!334 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !335)
!335 = distinct !DILocation(line: 159, column: 9, scope: !9)
!336 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !335)
!337 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !338)
!338 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !335)
!339 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !340)
!340 = distinct !DILocation(line: 185, column: 37, scope: !183, inlinedAt: !341)
!341 = distinct !DILocation(line: 160, column: 9, scope: !9)
!342 = !DILocation(line: 187, column: 47, scope: !183, inlinedAt: !341)
!343 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !344)
!344 = distinct !DILocation(line: 188, column: 45, scope: !183, inlinedAt: !341)
!345 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !346)
!346 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !347)
!347 = distinct !DILocation(line: 161, column: 9, scope: !9)
!348 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !347)
!349 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !350)
!350 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !347)
!351 = !DILocation(line: 129, column: 5, scope: !180, inlinedAt: !352)
!352 = distinct !DILocation(line: 185, column: 37, scope: !199, inlinedAt: !353)
!353 = distinct !DILocation(line: 162, column: 9, scope: !9)
!354 = !DILocation(line: 187, column: 47, scope: !199, inlinedAt: !353)
!355 = !DILocation(line: 494, column: 7, scope: !188, inlinedAt: !356)
!356 = distinct !DILocation(line: 188, column: 45, scope: !199, inlinedAt: !353)
!357 = !DILocation(line: 164, column: 9, scope: !9)
!358 = !DILocation(line: 165, column: 9, scope: !9)
!359 = !DILocation(line: 168, column: 9, scope: !9)
!360 = !{i64 4305900}
!361 = !DILocation(line: 169, column: 9, scope: !9)
!362 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !363)
!363 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !364)
!364 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !365)
!365 = distinct !DILocation(line: 170, column: 9, scope: !9)
!366 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !367)
!367 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !368)
!368 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !369)
!369 = distinct !DILocation(line: 171, column: 9, scope: !9)
!370 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !371)
!371 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !372)
!372 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !373)
!373 = distinct !DILocation(line: 172, column: 9, scope: !9)
!374 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !375)
!375 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !376)
!376 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !377)
!377 = distinct !DILocation(line: 173, column: 9, scope: !9)
!378 = !DILocation(line: 174, column: 9, scope: !9)
!379 = !DILocation(line: 175, column: 9, scope: !9)
!380 = !DILocation(line: 176, column: 9, scope: !9)
!381 = !DILocation(line: 179, column: 13, scope: !9)
!382 = !DILocation(line: 180, column: 13, scope: !9)
!383 = !{i64 4306394}
!384 = !DILocation(line: 17, column: 5, scope: !385, inlinedAt: !386)
!385 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !181, file: !181, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!386 = distinct !DILocation(line: 119, column: 17, scope: !387, inlinedAt: !388)
!387 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !181, file: !181, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!388 = distinct !DILocation(line: 181, column: 13, scope: !9)
!389 = !{i64 4290493}
!390 = !DILocation(line: 17, column: 5, scope: !385, inlinedAt: !391)
!391 = distinct !DILocation(line: 120, column: 17, scope: !387, inlinedAt: !388)
!392 = !DILocation(line: 183, column: 13, scope: !9)
!393 = !{i64 4306575}
!394 = !DILocation(line: 17, column: 5, scope: !385, inlinedAt: !395)
!395 = distinct !DILocation(line: 119, column: 17, scope: !396, inlinedAt: !397)
!396 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 64>, 512>", scope: !181, file: !181, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!397 = distinct !DILocation(line: 184, column: 13, scope: !9)
!398 = !DILocation(line: 17, column: 5, scope: !385, inlinedAt: !399)
!399 = distinct !DILocation(line: 120, column: 17, scope: !396, inlinedAt: !397)
!400 = !DILocation(line: 186, column: 9, scope: !9)
!401 = !DILocation(line: 187, column: 9, scope: !9)
!402 = !DILocation(line: 188, column: 9, scope: !9)
!403 = !DILocation(line: 191, column: 9, scope: !9)
!404 = !{i64 4306845}
!405 = !DILocation(line: 192, column: 9, scope: !9)
!406 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !407)
!407 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !408)
!408 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !409)
!409 = distinct !DILocation(line: 193, column: 9, scope: !9)
!410 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !411)
!411 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !412)
!412 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !413)
!413 = distinct !DILocation(line: 194, column: 9, scope: !9)
!414 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !415)
!415 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !416)
!416 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !417)
!417 = distinct !DILocation(line: 195, column: 9, scope: !9)
!418 = !DILocation(line: 29, column: 27, scope: !224, inlinedAt: !419)
!419 = distinct !DILocation(line: 81, column: 5, scope: !227, inlinedAt: !420)
!420 = distinct !DILocation(line: 216, column: 13, scope: !229, inlinedAt: !421)
!421 = distinct !DILocation(line: 196, column: 9, scope: !9)
!422 = !DILocation(line: 197, column: 9, scope: !9)
!423 = !DILocation(line: 198, column: 9, scope: !9)
!424 = !DILocation(line: 199, column: 9, scope: !9)
!425 = !DILocation(line: 100, column: 29, scope: !9)
!426 = distinct !{!426, !174, !427, !428}
!427 = !DILocation(line: 200, column: 5, scope: !9)
!428 = !{!"llvm.loop.mustprogress"}
!429 = !DILocation(line: 203, column: 9, scope: !9)
!430 = !DILocation(line: 204, column: 5, scope: !9)
!431 = !DILocation(line: 213, column: 32, scope: !9)
!432 = !DILocation(line: 60, column: 34, scope: !433, inlinedAt: !434)
!433 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !91, file: !91, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!434 = distinct !DILocation(line: 192, column: 41, scope: !435, inlinedAt: !437)
!435 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !436, file: !436, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!436 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!437 = distinct !DILocation(line: 212, column: 5, scope: !438, inlinedAt: !439)
!438 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !436, file: !436, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!439 = distinct !DILocation(line: 210, column: 9, scope: !9)
!440 = !DILocation(line: 61, column: 18, scope: !433, inlinedAt: !434)
!441 = !DILocation(line: 63, column: 54, scope: !442, inlinedAt: !443)
!442 = distinct !DISubprogram(name: "operator[]", scope: !95, file: !95, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!443 = distinct !DILocation(line: 192, column: 23, scope: !435, inlinedAt: !437)
!444 = !DILocation(line: 63, column: 60, scope: !442, inlinedAt: !443)
!445 = !DILocation(line: 63, column: 68, scope: !442, inlinedAt: !443)
!446 = !DILocation(line: 63, column: 16, scope: !442, inlinedAt: !443)
!447 = !DILocation(line: 203, column: 49, scope: !435, inlinedAt: !437)
!448 = !DILocation(line: 203, column: 13, scope: !435, inlinedAt: !437)
!449 = !DILocation(line: 203, column: 47, scope: !435, inlinedAt: !437)
!450 = !{!451, !451, i64 0}
!451 = !{!"float", !452, i64 0}
!452 = !{!"omnipotent char", !453, i64 0}
!453 = !{!"Simple C++ TBAA"}
!454 = !DILocation(line: 204, column: 49, scope: !435, inlinedAt: !437)
!455 = !DILocation(line: 204, column: 13, scope: !435, inlinedAt: !437)
!456 = !DILocation(line: 204, column: 47, scope: !435, inlinedAt: !437)
!457 = !DILocation(line: 205, column: 49, scope: !435, inlinedAt: !437)
!458 = !DILocation(line: 205, column: 13, scope: !435, inlinedAt: !437)
!459 = !DILocation(line: 205, column: 47, scope: !435, inlinedAt: !437)
!460 = !DILocation(line: 206, column: 49, scope: !435, inlinedAt: !437)
!461 = !DILocation(line: 206, column: 13, scope: !435, inlinedAt: !437)
!462 = !DILocation(line: 206, column: 47, scope: !435, inlinedAt: !437)
!463 = !DILocation(line: 216, column: 1, scope: !9)
