; ModuleID = '256_128_64_32_no_swizzle-hip-amdgcn-amd-amdhsa-gfx942.tmp.bc'
source_filename = "256_128_64_32_no_swizzle.cpp"
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
@__hip_cuid_40cce7fbd9d48eb7 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_40cce7fbd9d48eb7 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z8micro_tk13micro_globals(ptr addrspace(4) nocapture noundef readonly byref(%struct.micro_globals) align 8 %0) local_unnamed_addr #0 !dbg !9 {
  %.sroa.0788.0.copyload = load ptr, ptr addrspace(4) %0, align 8
  %.sroa.7.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 32
  %.sroa.7.0.copyload = load i64, ptr addrspace(4) %.sroa.7.0..sroa_idx, align 8
  %.sroa.9793.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 48
  %.sroa.9793.0.copyload = load ptr, ptr addrspace(4) %.sroa.9793.0..sroa_idx, align 8
  %.sroa.15.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 80
  %.sroa.15.0.copyload = load i64, ptr addrspace(4) %.sroa.15.0..sroa_idx, align 8
  %.sroa.17797.0..sroa_idx = getelementptr inbounds i8, ptr addrspace(4) %0, i64 96
  %.sroa.17797.0.copyload = load ptr, ptr addrspace(4) %.sroa.17797.0..sroa_idx, align 8
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
  %reass.sub.i.i97 = and i64 %7, -16, !dbg !26
  %9 = add i64 %reass.sub.i.i97, 16, !dbg !26
  %10 = inttoptr i64 %9 to ptr, !dbg !26
  %.not.i.i96 = icmp eq i64 %8, 0, !dbg !27
  %11 = select i1 %.not.i.i96, ptr %6, ptr %10, !dbg !26
  %12 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y(), !dbg !28
  %13 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x(), !dbg !34
  %14 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x(), !dbg !39
  %15 = lshr i32 %14, 6, !dbg !46
  %16 = lshr i32 %14, 8, !dbg !47
  %17 = and i32 %15, 3, !dbg !48
  %18 = trunc i64 %.sroa.7.0.copyload to i32, !dbg !49
  %19 = shl nsw i32 %12, 7, !dbg !56
  %20 = sext i32 %19 to i64, !dbg !60
  %21 = mul i64 %.sroa.7.0.copyload, %20, !dbg !64
  %22 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.0788.0.copyload, i64 %21, !dbg !65
  %23 = ptrtoint ptr %5 to i64, !dbg !66
  %24 = shl nuw nsw i32 %14, 3
  %25 = and i32 %24, 56
  %26 = lshr i32 %14, 3, !dbg !67
  %27 = and i32 %26, 63, !dbg !67
  %28 = mul i32 %27, %18, !dbg !68
  %29 = add i32 %28, %25, !dbg !69
  %30 = sext i32 %29 to i64, !dbg !70
  %31 = getelementptr inbounds %struct.__hip_bfloat16, ptr %22, i64 %30, !dbg !70
  %32 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %31) #6, !dbg !71, !srcloc !75
  %33 = or i32 %26, 64, !dbg !67
  %34 = mul i32 %33, %18, !dbg !68
  %35 = add i32 %34, %25, !dbg !69
  %36 = sext i32 %35 to i64, !dbg !70
  %37 = getelementptr inbounds %struct.__hip_bfloat16, ptr %22, i64 %36, !dbg !70
  %38 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %37) #6, !dbg !71, !srcloc !75
  %39 = trunc i64 %23 to i32, !dbg !66
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #6, !dbg !76, !srcloc !77
  %40 = shl nuw nsw i32 %14, 4
  %.masked.i.i = and i32 %40, 8064
  %41 = shl nuw nsw i32 %25, 1
  %42 = add i32 %41, %39
  %43 = or disjoint i32 %41, 8
  %44 = add i32 %43, %39
  %45 = add i32 %42, %.masked.i.i, !dbg !78
  %46 = lshr exact i32 %45, 4, !dbg !82
  %47 = and i32 %46, 120, !dbg !82
  %48 = xor i32 %47, %45, !dbg !83
  %.sroa.0.0.extract.trunc.i.i = trunc i128 %32 to i64, !dbg !84
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %48, i64 %.sroa.0.0.extract.trunc.i.i) #6, !dbg !85, !srcloc !88
  %49 = add i32 %44, %.masked.i.i, !dbg !89
  %50 = lshr i32 %49, 4, !dbg !91
  %51 = and i32 %50, 120, !dbg !91
  %52 = xor i32 %51, %49, !dbg !92
  %.sroa.0.8.extract.shift.i.i = lshr i128 %32, 64, !dbg !93
  %.sroa.0.8.extract.trunc.i.i = trunc nuw i128 %.sroa.0.8.extract.shift.i.i to i64, !dbg !93
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %52, i64 %.sroa.0.8.extract.trunc.i.i) #6, !dbg !94, !srcloc !88
  %53 = or disjoint i32 %.masked.i.i, 8192, !dbg !96
  %54 = add i32 %42, %53, !dbg !78
  %55 = lshr exact i32 %54, 4, !dbg !82
  %56 = and i32 %55, 120, !dbg !82
  %57 = xor i32 %56, %54, !dbg !83
  %.sroa.5.16.extract.trunc.i.i = trunc i128 %38 to i64, !dbg !84
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %57, i64 %.sroa.5.16.extract.trunc.i.i) #6, !dbg !85, !srcloc !88
  %58 = add i32 %44, %53, !dbg !89
  %59 = lshr i32 %58, 4, !dbg !91
  %60 = and i32 %59, 120, !dbg !91
  %61 = xor i32 %60, %58, !dbg !92
  %.sroa.5.24.extract.shift.i.i = lshr i128 %38, 64, !dbg !93
  %.sroa.5.24.extract.trunc.i.i = trunc nuw i128 %.sroa.5.24.extract.shift.i.i to i64, !dbg !93
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %61, i64 %.sroa.5.24.extract.trunc.i.i) #6, !dbg !94, !srcloc !88
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #6, !dbg !97, !srcloc !98
  %62 = trunc i64 %.sroa.15.0.copyload to i32, !dbg !99
  %63 = shl nsw i32 %13, 8, !dbg !104
  %64 = sext i32 %63 to i64, !dbg !107
  %65 = mul i64 %.sroa.15.0.copyload, %64, !dbg !109
  %66 = getelementptr %struct.__hip_bfloat16, ptr %.sroa.9793.0.copyload, i64 %65, !dbg !110
  %67 = ptrtoint ptr %11 to i64, !dbg !111
  %68 = mul i32 %27, %62, !dbg !112
  %69 = add i32 %68, %25, !dbg !113
  %70 = sext i32 %69 to i64, !dbg !114
  %71 = getelementptr inbounds %struct.__hip_bfloat16, ptr %66, i64 %70, !dbg !114
  %72 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %71) #6, !dbg !115, !srcloc !75
  %73 = mul i32 %33, %62, !dbg !112
  %74 = add i32 %73, %25, !dbg !113
  %75 = sext i32 %74 to i64, !dbg !114
  %76 = getelementptr inbounds %struct.__hip_bfloat16, ptr %66, i64 %75, !dbg !114
  %77 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %76) #6, !dbg !115, !srcloc !75
  %78 = or disjoint i32 %27, 128, !dbg !117
  %79 = mul i32 %78, %62, !dbg !112
  %80 = add i32 %79, %25, !dbg !113
  %81 = sext i32 %80 to i64, !dbg !114
  %82 = getelementptr inbounds %struct.__hip_bfloat16, ptr %66, i64 %81, !dbg !114
  %83 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %82) #6, !dbg !115, !srcloc !75
  %84 = or i32 %26, 192, !dbg !117
  %85 = mul i32 %84, %62, !dbg !112
  %86 = add i32 %85, %25, !dbg !113
  %87 = sext i32 %86 to i64, !dbg !114
  %88 = getelementptr inbounds %struct.__hip_bfloat16, ptr %66, i64 %87, !dbg !114
  %89 = tail call i128 asm sideeffect "global_load_dwordx4 $0, $1, off\0A", "=v,v,~{memory}"(ptr %88) #6, !dbg !115, !srcloc !75
  %90 = trunc i64 %67 to i32, !dbg !111
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #6, !dbg !118, !srcloc !77
  %91 = add i32 %41, %90
  %92 = add i32 %43, %90
  %93 = add i32 %91, %.masked.i.i, !dbg !119
  %94 = lshr exact i32 %93, 4, !dbg !122
  %95 = and i32 %94, 120, !dbg !122
  %96 = xor i32 %95, %93, !dbg !123
  %.sroa.0.0.extract.trunc.i.i102 = trunc i128 %72 to i64, !dbg !124
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %96, i64 %.sroa.0.0.extract.trunc.i.i102) #6, !dbg !125, !srcloc !88
  %97 = add i32 %92, %.masked.i.i, !dbg !127
  %98 = lshr i32 %97, 4, !dbg !129
  %99 = and i32 %98, 120, !dbg !129
  %100 = xor i32 %99, %97, !dbg !130
  %.sroa.0.8.extract.shift.i.i103 = lshr i128 %72, 64, !dbg !131
  %.sroa.0.8.extract.trunc.i.i104 = trunc nuw i128 %.sroa.0.8.extract.shift.i.i103 to i64, !dbg !131
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %100, i64 %.sroa.0.8.extract.trunc.i.i104) #6, !dbg !132, !srcloc !88
  %101 = add i32 %91, %53, !dbg !119
  %102 = lshr exact i32 %101, 4, !dbg !122
  %103 = and i32 %102, 120, !dbg !122
  %104 = xor i32 %103, %101, !dbg !123
  %.sroa.5.16.extract.trunc.i.i105 = trunc i128 %77 to i64, !dbg !124
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %104, i64 %.sroa.5.16.extract.trunc.i.i105) #6, !dbg !125, !srcloc !88
  %105 = add i32 %92, %53, !dbg !127
  %106 = lshr i32 %105, 4, !dbg !129
  %107 = and i32 %106, 120, !dbg !129
  %108 = xor i32 %107, %105, !dbg !130
  %.sroa.5.24.extract.shift.i.i106 = lshr i128 %77, 64, !dbg !131
  %.sroa.5.24.extract.trunc.i.i107 = trunc nuw i128 %.sroa.5.24.extract.shift.i.i106 to i64, !dbg !131
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %108, i64 %.sroa.5.24.extract.trunc.i.i107) #6, !dbg !132, !srcloc !88
  %109 = or disjoint i32 %.masked.i.i, 16384, !dbg !134
  %110 = add i32 %91, %109, !dbg !119
  %111 = lshr exact i32 %110, 4, !dbg !122
  %112 = and i32 %111, 120, !dbg !122
  %113 = xor i32 %112, %110, !dbg !123
  %.sroa.8.32.extract.trunc.i.i = trunc i128 %83 to i64, !dbg !124
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %113, i64 %.sroa.8.32.extract.trunc.i.i) #6, !dbg !125, !srcloc !88
  %114 = add i32 %92, %109, !dbg !127
  %115 = lshr i32 %114, 4, !dbg !129
  %116 = and i32 %115, 120, !dbg !129
  %117 = xor i32 %116, %114, !dbg !130
  %.sroa.8.40.extract.shift.i.i = lshr i128 %83, 64, !dbg !131
  %.sroa.8.40.extract.trunc.i.i = trunc nuw i128 %.sroa.8.40.extract.shift.i.i to i64, !dbg !131
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %117, i64 %.sroa.8.40.extract.trunc.i.i) #6, !dbg !132, !srcloc !88
  %118 = or disjoint i32 %.masked.i.i, 24576, !dbg !134
  %119 = add i32 %91, %118, !dbg !119
  %120 = lshr exact i32 %119, 4, !dbg !122
  %121 = and i32 %120, 120, !dbg !122
  %122 = xor i32 %121, %119, !dbg !123
  %.sroa.11.48.extract.trunc.i.i = trunc i128 %89 to i64, !dbg !124
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %122, i64 %.sroa.11.48.extract.trunc.i.i) #6, !dbg !125, !srcloc !88
  %123 = add i32 %92, %118, !dbg !127
  %124 = lshr i32 %123, 4, !dbg !129
  %125 = and i32 %124, 120, !dbg !129
  %126 = xor i32 %125, %123, !dbg !130
  %.sroa.11.56.extract.shift.i.i = lshr i128 %89, 64, !dbg !131
  %.sroa.11.56.extract.trunc.i.i = trunc nuw i128 %.sroa.11.56.extract.shift.i.i to i64, !dbg !131
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %126, i64 %.sroa.11.56.extract.trunc.i.i) #6, !dbg !132, !srcloc !88
  tail call void asm sideeffect "s_waitcnt lgkmcnt(0)", ""() #6, !dbg !135, !srcloc !98
  tail call void @llvm.amdgcn.s.barrier(), !dbg !136
  %127 = icmp eq i32 %16, 1, !dbg !137
  br i1 %127, label %128, label %129, !dbg !138

128:                                              ; preds = %1
  tail call void @llvm.amdgcn.s.barrier(), !dbg !139
  br label %129, !dbg !140

129:                                              ; preds = %128, %1
  %130 = shl nuw nsw i32 %17, 5
  %131 = and i32 %14, 15
  %132 = lshr i32 %14, 2
  %133 = and i32 %132, 12
  %134 = or disjoint i32 %130, %131
  %135 = shl nuw nsw i32 %134, 7
  %136 = shl nuw nsw i32 %133, 1
  %137 = add i32 %135, %90
  %138 = add i32 %137, %136
  %139 = lshr i32 %138, 4
  %140 = and i32 %139, 120
  %141 = xor i32 %140, %138
  %142 = shl i32 %35, 1
  %143 = shl i32 %29, 1
  %144 = shl i32 %18, 8
  %145 = or disjoint i32 %136, 32
  %146 = add i32 %137, %145
  %147 = lshr i32 %146, 4
  %148 = and i32 %147, 120
  %149 = xor i32 %148, %146
  %150 = or disjoint i32 %131, 16
  %151 = or disjoint i32 %130, %150
  %152 = shl nuw nsw i32 %151, 7
  %153 = add i32 %152, %90
  %154 = add i32 %153, %136
  %155 = lshr i32 %154, 4
  %156 = and i32 %155, 120
  %157 = xor i32 %156, %154
  %158 = add i32 %153, %145
  %159 = lshr i32 %158, 4
  %160 = and i32 %159, 120
  %161 = xor i32 %160, %158
  %162 = or disjoint i32 %130, 128
  %163 = or disjoint i32 %162, %131
  %164 = shl nuw nsw i32 %163, 7
  %165 = add i32 %164, %90
  %166 = add i32 %165, %136
  %167 = lshr i32 %166, 4
  %168 = and i32 %167, 120
  %169 = xor i32 %168, %166
  %170 = add i32 %165, %145
  %171 = lshr i32 %170, 4
  %172 = and i32 %171, 120
  %173 = xor i32 %172, %170
  %174 = or disjoint i32 %162, %150
  %175 = shl nuw nsw i32 %174, 7
  %176 = add i32 %175, %90
  %177 = add i32 %176, %136
  %178 = lshr i32 %177, 4
  %179 = and i32 %178, 120
  %180 = xor i32 %179, %177
  %181 = add i32 %176, %145
  %182 = lshr i32 %181, 4
  %183 = and i32 %182, 120
  %184 = xor i32 %183, %181
  %185 = shl nuw nsw i32 %16, 5
  %186 = or disjoint i32 %185, %131
  %187 = shl nuw nsw i32 %186, 7
  %188 = add i32 %187, %39
  %189 = add i32 %188, %136
  %190 = lshr i32 %189, 4
  %191 = and i32 %190, 120
  %192 = xor i32 %191, %189
  %193 = add i32 %188, %145
  %194 = lshr i32 %193, 4
  %195 = and i32 %194, 120
  %196 = xor i32 %195, %193
  %197 = or disjoint i32 %150, %185
  %198 = shl nuw nsw i32 %197, 7
  %199 = add i32 %198, %39
  %200 = add i32 %199, %136
  %201 = lshr i32 %200, 4
  %202 = and i32 %201, 120
  %203 = xor i32 %202, %200
  %204 = add i32 %199, %145
  %205 = lshr i32 %204, 4
  %206 = and i32 %205, 120
  %207 = xor i32 %206, %204
  %208 = add nuw nsw i32 %185, 64
  %209 = or disjoint i32 %208, %131
  %210 = shl nuw nsw i32 %209, 7
  %211 = add i32 %210, %39
  %212 = add i32 %211, %136
  %213 = lshr i32 %212, 4
  %214 = and i32 %213, 120
  %215 = xor i32 %214, %212
  %216 = add i32 %211, %145
  %217 = lshr i32 %216, 4
  %218 = and i32 %217, 120
  %219 = xor i32 %218, %216
  %220 = or disjoint i32 %208, %150
  %221 = shl nuw nsw i32 %220, 7
  %222 = add i32 %221, %39
  %223 = add i32 %222, %136
  %224 = lshr i32 %223, 4
  %225 = and i32 %224, 120
  %226 = xor i32 %225, %223
  %227 = add i32 %222, %145
  %228 = lshr i32 %227, 4
  %229 = and i32 %228, 120
  %230 = xor i32 %229, %227
  %231 = shl i32 %62, 9
  %232 = shl i32 %69, 1
  %233 = shl i32 %74, 1
  %234 = shl i32 %80, 1
  %235 = shl i32 %86, 1
  %236 = or disjoint i32 %136, 64
  %237 = add i32 %137, %236
  %238 = lshr i32 %237, 4
  %239 = and i32 %238, 120
  %240 = xor i32 %239, %237
  %241 = or disjoint i32 %136, 96
  %242 = add i32 %137, %241
  %243 = lshr i32 %242, 4
  %244 = and i32 %243, 120
  %245 = xor i32 %244, %242
  %246 = add i32 %153, %236
  %247 = lshr i32 %246, 4
  %248 = and i32 %247, 120
  %249 = xor i32 %248, %246
  %250 = add i32 %153, %241
  %251 = lshr i32 %250, 4
  %252 = and i32 %251, 120
  %253 = xor i32 %252, %250
  %254 = add i32 %165, %236
  %255 = lshr i32 %254, 4
  %256 = and i32 %255, 120
  %257 = xor i32 %256, %254
  %258 = add i32 %165, %241
  %259 = lshr i32 %258, 4
  %260 = and i32 %259, 120
  %261 = xor i32 %260, %258
  %262 = add i32 %176, %236
  %263 = lshr i32 %262, 4
  %264 = and i32 %263, 120
  %265 = xor i32 %264, %262
  %266 = add i32 %176, %241
  %267 = lshr i32 %266, 4
  %268 = and i32 %267, 120
  %269 = xor i32 %268, %266
  %270 = add i32 %188, %236
  %271 = lshr i32 %270, 4
  %272 = and i32 %271, 120
  %273 = xor i32 %272, %270
  %274 = add i32 %188, %241
  %275 = lshr i32 %274, 4
  %276 = and i32 %275, 120
  %277 = xor i32 %276, %274
  %278 = add i32 %199, %236
  %279 = lshr i32 %278, 4
  %280 = and i32 %279, 120
  %281 = xor i32 %280, %278
  %282 = add i32 %199, %241
  %283 = lshr i32 %282, 4
  %284 = and i32 %283, 120
  %285 = xor i32 %284, %282
  %286 = add i32 %211, %236
  %287 = lshr i32 %286, 4
  %288 = and i32 %287, 120
  %289 = xor i32 %288, %286
  %290 = add i32 %211, %241
  %291 = lshr i32 %290, 4
  %292 = and i32 %291, 120
  %293 = xor i32 %292, %290
  %294 = add i32 %222, %236
  %295 = lshr i32 %294, 4
  %296 = and i32 %295, 120
  %297 = xor i32 %296, %294
  %298 = add i32 %222, %241
  %299 = lshr i32 %298, 4
  %300 = and i32 %299, 120
  %301 = xor i32 %300, %298
  %302 = and i32 %40, 112
  %303 = add i32 %302, %39
  %304 = or disjoint i32 %302, 8
  %305 = add i32 %304, %39
  %306 = add i32 %303, %.masked.i.i
  %307 = lshr exact i32 %306, 4
  %308 = and i32 %307, 120
  %309 = xor i32 %308, %306
  %310 = add i32 %305, %.masked.i.i
  %311 = lshr i32 %310, 4
  %312 = and i32 %311, 120
  %313 = xor i32 %312, %310
  %314 = add i32 %303, %53
  %315 = lshr exact i32 %314, 4
  %316 = and i32 %315, 120
  %317 = xor i32 %316, %314
  %318 = add i32 %305, %53
  %319 = lshr i32 %318, 4
  %320 = and i32 %319, 120
  %321 = xor i32 %320, %318
  %322 = add i32 %302, %90
  %323 = add i32 %304, %90
  %324 = add i32 %322, %.masked.i.i
  %325 = lshr exact i32 %324, 4
  %326 = and i32 %325, 120
  %327 = xor i32 %326, %324
  %328 = add i32 %323, %.masked.i.i
  %329 = lshr i32 %328, 4
  %330 = and i32 %329, 120
  %331 = xor i32 %330, %328
  %332 = add i32 %322, %53
  %333 = lshr exact i32 %332, 4
  %334 = and i32 %333, 120
  %335 = xor i32 %334, %332
  %336 = add i32 %323, %53
  %337 = lshr i32 %336, 4
  %338 = and i32 %337, 120
  %339 = xor i32 %338, %336
  %340 = add i32 %322, %109
  %341 = lshr exact i32 %340, 4
  %342 = and i32 %341, 120
  %343 = xor i32 %342, %340
  %344 = add i32 %323, %109
  %345 = lshr i32 %344, 4
  %346 = and i32 %345, 120
  %347 = xor i32 %346, %344
  %348 = add i32 %322, %118
  %349 = lshr exact i32 %348, 4
  %350 = and i32 %349, 120
  %351 = xor i32 %350, %348
  %352 = add i32 %323, %118
  %353 = lshr i32 %352, 4
  %354 = and i32 %353, 120
  %355 = xor i32 %354, %352
  br label %358, !dbg !141

356:                                              ; preds = %616
  %357 = icmp ult i32 %14, 256, !dbg !142
  br i1 %357, label %649, label %650, !dbg !143

358:                                              ; preds = %129, %616
  %359 = phi <4 x float> [ zeroinitializer, %129 ], [ %648, %616 ]
  %360 = phi <4 x float> [ zeroinitializer, %129 ], [ %646, %616 ]
  %361 = phi <4 x float> [ zeroinitializer, %129 ], [ %644, %616 ]
  %.val33.i433887 = phi <4 x float> [ zeroinitializer, %129 ], [ %642, %616 ]
  %362 = phi <4 x float> [ zeroinitializer, %129 ], [ %640, %616 ]
  %363 = phi <4 x float> [ zeroinitializer, %129 ], [ %638, %616 ]
  %364 = phi <4 x float> [ zeroinitializer, %129 ], [ %636, %616 ]
  %.val33.i413878 = phi <4 x float> [ zeroinitializer, %129 ], [ %634, %616 ]
  %365 = phi <4 x float> [ zeroinitializer, %129 ], [ %632, %616 ]
  %366 = phi <4 x float> [ zeroinitializer, %129 ], [ %630, %616 ]
  %367 = phi <4 x float> [ zeroinitializer, %129 ], [ %628, %616 ]
  %.val33.i393869 = phi <4 x float> [ zeroinitializer, %129 ], [ %626, %616 ]
  %.val33.1.1.i468866 = phi <4 x float> [ zeroinitializer, %129 ], [ %624, %616 ]
  %.val33.138.i463863 = phi <4 x float> [ zeroinitializer, %129 ], [ %622, %616 ]
  %.val33.1.i458860 = phi <4 x float> [ zeroinitializer, %129 ], [ %620, %616 ]
  %.093857 = phi i32 [ 0, %129 ], [ %368, %616 ]
  %.sroa.5631.0.off64856 = phi i64 [ 0, %129 ], [ %.sroa.5631.1.off64, %616 ]
  %.sroa.5631.0.off0855 = phi i64 [ undef, %129 ], [ %.sroa.5631.1.off0, %616 ]
  %.sroa.0630.0.off64854 = phi i64 [ 0, %129 ], [ %.sroa.0630.1.off64, %616 ]
  %.sroa.0630.0.off0853 = phi i64 [ undef, %129 ], [ %.sroa.0630.1.off0, %616 ]
  %.sroa.11.0.off64852 = phi i64 [ 0, %129 ], [ %.sroa.11.1.off64, %616 ]
  %.sroa.11.0.off0851 = phi i64 [ undef, %129 ], [ %.sroa.11.1.off0, %616 ]
  %.sroa.8629.0.off64850 = phi i64 [ 0, %129 ], [ %.sroa.8629.1.off64, %616 ]
  %.sroa.8629.0.off0849 = phi i64 [ undef, %129 ], [ %.sroa.8629.1.off0, %616 ]
  %.sroa.5.0.off64848 = phi i64 [ 0, %129 ], [ %.sroa.5.1.off64, %616 ]
  %.sroa.5.0.off0847 = phi i64 [ undef, %129 ], [ %.sroa.5.1.off0, %616 ]
  %.sroa.0628.0.off64846 = phi i64 [ 0, %129 ], [ %.sroa.0628.1.off64, %616 ]
  %.sroa.0628.0.off0845 = phi i64 [ undef, %129 ], [ %.sroa.0628.1.off0, %616 ]
  %.val33.i843844 = phi <4 x float> [ zeroinitializer, %129 ], [ %618, %616 ]
  %368 = add nuw nsw i32 %.093857, 1, !dbg !144
  %.not = icmp eq i32 %.093857, 127, !dbg !145
  br i1 %.not, label %378, label %369, !dbg !146

369:                                              ; preds = %358
  %370 = shl nuw nsw i32 %368, 6
  %371 = zext nneg i32 %370 to i64
  %372 = getelementptr %struct.__hip_bfloat16, ptr %22, i64 %371
  %373 = ptrtoint ptr %372 to i64
  %374 = bitcast i64 %373 to <2 x i32>
  %.sroa.0.0.vec.expand.i.i = shufflevector <2 x i32> %374, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %375 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i, i32 1114112, i64 3
  %.sroa.0.12.vec.insert.i.i = insertelement <4 x i32> %375, i32 %144, i64 2
  %376 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %143, i32 noundef 0, i32 noundef 0) #6, !dbg !147
  %377 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i, i32 noundef %142, i32 noundef 0, i32 noundef 0) #6, !dbg !147
  %extract.t828 = trunc i128 %376 to i64, !dbg !153
  %extract830 = lshr i128 %376, 64, !dbg !153
  %extract.t831 = trunc nuw i128 %extract830 to i64, !dbg !153
  %extract.t835 = trunc i128 %377 to i64, !dbg !153
  %extract837 = lshr i128 %377, 64, !dbg !153
  %extract.t838 = trunc nuw i128 %extract837 to i64, !dbg !153
  br label %378, !dbg !153

378:                                              ; preds = %369, %358
  %.sroa.0630.1.off0 = phi i64 [ %.sroa.0630.0.off0853, %358 ], [ %extract.t828, %369 ]
  %.sroa.0630.1.off64 = phi i64 [ %.sroa.0630.0.off64854, %358 ], [ %extract.t831, %369 ]
  %.sroa.5631.1.off0 = phi i64 [ %.sroa.5631.0.off0855, %358 ], [ %extract.t835, %369 ]
  %.sroa.5631.1.off64 = phi i64 [ %.sroa.5631.0.off64856, %358 ], [ %extract.t838, %369 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !154
  %379 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %141) #6, !dbg !155, !srcloc !160
  %.sroa.036.2.extract.shift37.i = lshr i64 %379, 16, !dbg !161
  %380 = insertelement <2 x i64> poison, i64 %379, i64 0, !dbg !161
  %381 = insertelement <2 x i64> %380, i64 %.sroa.036.2.extract.shift37.i, i64 1, !dbg !161
  %382 = shufflevector <2 x i64> %380, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !155
  %383 = lshr <2 x i64> %382, <i64 32, i64 48>, !dbg !155
  %384 = shufflevector <2 x i64> %381, <2 x i64> %383, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !162
  %.sroa.0666.6.vec.insert1053 = trunc <4 x i64> %384 to <4 x i16>, !dbg !162
  %385 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %149) #6, !dbg !155, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i = lshr i64 %385, 16, !dbg !161
  %386 = insertelement <2 x i64> poison, i64 %385, i64 0, !dbg !161
  %387 = insertelement <2 x i64> %386, i64 %.sroa.036.2.extract.shift37.1.i, i64 1, !dbg !161
  %388 = shufflevector <2 x i64> %386, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !155
  %389 = lshr <2 x i64> %388, <i64 32, i64 48>, !dbg !155
  %390 = shufflevector <2 x i64> %387, <2 x i64> %389, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !162
  %.sroa.10671.14.vec.insert1052 = trunc <4 x i64> %390 to <4 x i16>, !dbg !162
  %391 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %157) #6, !dbg !155, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i = lshr i64 %391, 16, !dbg !161
  %392 = insertelement <2 x i64> poison, i64 %391, i64 0, !dbg !161
  %393 = insertelement <2 x i64> %392, i64 %.sroa.036.2.extract.shift37.145.i, i64 1, !dbg !161
  %394 = shufflevector <2 x i64> %392, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !155
  %395 = lshr <2 x i64> %394, <i64 32, i64 48>, !dbg !155
  %396 = shufflevector <2 x i64> %393, <2 x i64> %395, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !162
  %.sroa.18676.22.vec.insert1051 = trunc <4 x i64> %396 to <4 x i16>, !dbg !162
  %397 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %161) #6, !dbg !155, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i = lshr i64 %397, 16, !dbg !161
  %398 = insertelement <2 x i64> poison, i64 %397, i64 0, !dbg !161
  %399 = insertelement <2 x i64> %398, i64 %.sroa.036.2.extract.shift37.1.1.i, i64 1, !dbg !161
  %400 = shufflevector <2 x i64> %398, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !155
  %401 = lshr <2 x i64> %400, <i64 32, i64 48>, !dbg !155
  %402 = shufflevector <2 x i64> %399, <2 x i64> %401, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !162
  %.sroa.26681.30.vec.insert1050 = trunc <4 x i64> %402 to <4 x i16>, !dbg !162
  %403 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %169) #6, !dbg !166, !srcloc !160
  %.sroa.036.2.extract.shift37.i116 = lshr i64 %403, 16, !dbg !169
  %404 = insertelement <2 x i64> poison, i64 %403, i64 0, !dbg !169
  %405 = insertelement <2 x i64> %404, i64 %.sroa.036.2.extract.shift37.i116, i64 1, !dbg !169
  %406 = shufflevector <2 x i64> %404, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %407 = lshr <2 x i64> %406, <i64 32, i64 48>, !dbg !166
  %408 = shufflevector <2 x i64> %405, <2 x i64> %407, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !170
  %.sroa.0633.6.vec.insert1049 = trunc <4 x i64> %408 to <4 x i16>, !dbg !170
  %409 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %173) #6, !dbg !166, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i125 = lshr i64 %409, 16, !dbg !169
  %410 = insertelement <2 x i64> poison, i64 %409, i64 0, !dbg !169
  %411 = insertelement <2 x i64> %410, i64 %.sroa.036.2.extract.shift37.1.i125, i64 1, !dbg !169
  %412 = shufflevector <2 x i64> %410, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %413 = lshr <2 x i64> %412, <i64 32, i64 48>, !dbg !166
  %414 = shufflevector <2 x i64> %411, <2 x i64> %413, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !170
  %.sroa.10.14.vec.insert1048 = trunc <4 x i64> %414 to <4 x i16>, !dbg !170
  %415 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %180) #6, !dbg !166, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i134 = lshr i64 %415, 16, !dbg !169
  %416 = insertelement <2 x i64> poison, i64 %415, i64 0, !dbg !169
  %417 = insertelement <2 x i64> %416, i64 %.sroa.036.2.extract.shift37.145.i134, i64 1, !dbg !169
  %418 = shufflevector <2 x i64> %416, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %419 = lshr <2 x i64> %418, <i64 32, i64 48>, !dbg !166
  %420 = shufflevector <2 x i64> %417, <2 x i64> %419, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !170
  %.sroa.18.22.vec.insert1047 = trunc <4 x i64> %420 to <4 x i16>, !dbg !170
  %421 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %184) #6, !dbg !166, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i143 = lshr i64 %421, 16, !dbg !169
  %422 = insertelement <2 x i64> poison, i64 %421, i64 0, !dbg !169
  %423 = insertelement <2 x i64> %422, i64 %.sroa.036.2.extract.shift37.1.1.i143, i64 1, !dbg !169
  %424 = shufflevector <2 x i64> %422, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !166
  %425 = lshr <2 x i64> %424, <i64 32, i64 48>, !dbg !166
  %426 = shufflevector <2 x i64> %423, <2 x i64> %425, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !170
  %.sroa.26.30.vec.insert1046 = trunc <4 x i64> %426 to <4 x i16>, !dbg !170
  %427 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %192) #6, !dbg !172, !srcloc !160
  %.sroa.036.2.extract.shift37.i153 = lshr i64 %427, 16, !dbg !176
  %428 = insertelement <2 x i64> poison, i64 %427, i64 0, !dbg !176
  %429 = insertelement <2 x i64> %428, i64 %.sroa.036.2.extract.shift37.i153, i64 1, !dbg !176
  %430 = shufflevector <2 x i64> %428, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !172
  %431 = lshr <2 x i64> %430, <i64 32, i64 48>, !dbg !172
  %432 = shufflevector <2 x i64> %429, <2 x i64> %431, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.0746.6.vec.insert1045 = trunc <4 x i64> %432 to <4 x i16>, !dbg !177
  %433 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %196) #6, !dbg !172, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i160 = lshr i64 %433, 16, !dbg !176
  %434 = insertelement <2 x i64> poison, i64 %433, i64 0, !dbg !176
  %435 = insertelement <2 x i64> %434, i64 %.sroa.036.2.extract.shift37.1.i160, i64 1, !dbg !176
  %436 = shufflevector <2 x i64> %434, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !172
  %437 = lshr <2 x i64> %436, <i64 32, i64 48>, !dbg !172
  %438 = shufflevector <2 x i64> %435, <2 x i64> %437, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.10751.14.vec.insert1044 = trunc <4 x i64> %438 to <4 x i16>, !dbg !177
  %439 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %203) #6, !dbg !172, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i167 = lshr i64 %439, 16, !dbg !176
  %440 = insertelement <2 x i64> poison, i64 %439, i64 0, !dbg !176
  %441 = insertelement <2 x i64> %440, i64 %.sroa.036.2.extract.shift37.145.i167, i64 1, !dbg !176
  %442 = shufflevector <2 x i64> %440, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !172
  %443 = lshr <2 x i64> %442, <i64 32, i64 48>, !dbg !172
  %444 = shufflevector <2 x i64> %441, <2 x i64> %443, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.18756.22.vec.insert1043 = trunc <4 x i64> %444 to <4 x i16>, !dbg !177
  %445 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %207) #6, !dbg !172, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i174 = lshr i64 %445, 16, !dbg !176
  %446 = insertelement <2 x i64> poison, i64 %445, i64 0, !dbg !176
  %447 = insertelement <2 x i64> %446, i64 %.sroa.036.2.extract.shift37.1.1.i174, i64 1, !dbg !176
  %448 = shufflevector <2 x i64> %446, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !172
  %449 = lshr <2 x i64> %448, <i64 32, i64 48>, !dbg !172
  %450 = shufflevector <2 x i64> %447, <2 x i64> %449, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !177
  %.sroa.26761.30.vec.insert1042 = trunc <4 x i64> %450 to <4 x i16>, !dbg !177
  %451 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %215) #6, !dbg !179, !srcloc !160
  %.sroa.036.2.extract.shift37.i186 = lshr i64 %451, 16, !dbg !182
  %452 = insertelement <2 x i64> poison, i64 %451, i64 0, !dbg !182
  %453 = insertelement <2 x i64> %452, i64 %.sroa.036.2.extract.shift37.i186, i64 1, !dbg !182
  %454 = shufflevector <2 x i64> %452, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !179
  %455 = lshr <2 x i64> %454, <i64 32, i64 48>, !dbg !179
  %456 = shufflevector <2 x i64> %453, <2 x i64> %455, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !183
  %.sroa.0706.6.vec.insert1041 = trunc <4 x i64> %456 to <4 x i16>, !dbg !183
  %457 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %219) #6, !dbg !179, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i195 = lshr i64 %457, 16, !dbg !182
  %458 = insertelement <2 x i64> poison, i64 %457, i64 0, !dbg !182
  %459 = insertelement <2 x i64> %458, i64 %.sroa.036.2.extract.shift37.1.i195, i64 1, !dbg !182
  %460 = shufflevector <2 x i64> %458, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !179
  %461 = lshr <2 x i64> %460, <i64 32, i64 48>, !dbg !179
  %462 = shufflevector <2 x i64> %459, <2 x i64> %461, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !183
  %.sroa.10711.14.vec.insert1040 = trunc <4 x i64> %462 to <4 x i16>, !dbg !183
  %463 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %226) #6, !dbg !179, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i204 = lshr i64 %463, 16, !dbg !182
  %464 = insertelement <2 x i64> poison, i64 %463, i64 0, !dbg !182
  %465 = insertelement <2 x i64> %464, i64 %.sroa.036.2.extract.shift37.145.i204, i64 1, !dbg !182
  %466 = shufflevector <2 x i64> %464, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !179
  %467 = lshr <2 x i64> %466, <i64 32, i64 48>, !dbg !179
  %468 = shufflevector <2 x i64> %465, <2 x i64> %467, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !183
  %.sroa.18716.22.vec.insert1039 = trunc <4 x i64> %468 to <4 x i16>, !dbg !183
  %469 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %230) #6, !dbg !179, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i213 = lshr i64 %469, 16, !dbg !182
  %470 = insertelement <2 x i64> poison, i64 %469, i64 0, !dbg !182
  %471 = insertelement <2 x i64> %470, i64 %.sroa.036.2.extract.shift37.1.1.i213, i64 1, !dbg !182
  %472 = shufflevector <2 x i64> %470, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !179
  %473 = lshr <2 x i64> %472, <i64 32, i64 48>, !dbg !179
  %474 = shufflevector <2 x i64> %471, <2 x i64> %473, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !183
  %.sroa.26721.30.vec.insert1038 = trunc <4 x i64> %474 to <4 x i16>, !dbg !183
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !185
  br i1 %.not, label %486, label %475, !dbg !186

475:                                              ; preds = %378
  %476 = shl nuw nsw i32 %368, 6, !dbg !187
  %477 = zext nneg i32 %476 to i64, !dbg !191
  %478 = getelementptr %struct.__hip_bfloat16, ptr %66, i64 %477, !dbg !193
  %479 = ptrtoint ptr %478 to i64, !dbg !194
  %480 = bitcast i64 %479 to <2 x i32>, !dbg !194
  %.sroa.0.0.vec.expand.i.i221 = shufflevector <2 x i32> %480, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>, !dbg !194
  %481 = insertelement <4 x i32> %.sroa.0.0.vec.expand.i.i221, i32 1114112, i64 3, !dbg !194
  %.sroa.0.12.vec.insert.i.i222 = insertelement <4 x i32> %481, i32 %231, i64 2, !dbg !194
  %482 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i222, i32 noundef %232, i32 noundef 0, i32 noundef 0) #6, !dbg !197
  %483 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i222, i32 noundef %233, i32 noundef 0, i32 noundef 0) #6, !dbg !197
  %484 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i222, i32 noundef %234, i32 noundef 0, i32 noundef 0) #6, !dbg !197
  %485 = tail call noundef i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> noundef %.sroa.0.12.vec.insert.i.i222, i32 noundef %235, i32 noundef 0, i32 noundef 0) #6, !dbg !197
  %extract.t800 = trunc i128 %482 to i64, !dbg !199
  %extract802 = lshr i128 %482, 64, !dbg !199
  %extract.t803 = trunc nuw i128 %extract802 to i64, !dbg !199
  %extract.t807 = trunc i128 %483 to i64, !dbg !199
  %extract809 = lshr i128 %483, 64, !dbg !199
  %extract.t810 = trunc nuw i128 %extract809 to i64, !dbg !199
  %extract.t814 = trunc i128 %484 to i64, !dbg !199
  %extract816 = lshr i128 %484, 64, !dbg !199
  %extract.t817 = trunc nuw i128 %extract816 to i64, !dbg !199
  %extract.t821 = trunc i128 %485 to i64, !dbg !199
  %extract823 = lshr i128 %485, 64, !dbg !199
  %extract.t824 = trunc nuw i128 %extract823 to i64, !dbg !199
  br label %486, !dbg !199

486:                                              ; preds = %475, %378
  %.sroa.0628.1.off0 = phi i64 [ %.sroa.0628.0.off0845, %378 ], [ %extract.t800, %475 ]
  %.sroa.0628.1.off64 = phi i64 [ %.sroa.0628.0.off64846, %378 ], [ %extract.t803, %475 ]
  %.sroa.5.1.off0 = phi i64 [ %.sroa.5.0.off0847, %378 ], [ %extract.t807, %475 ]
  %.sroa.5.1.off64 = phi i64 [ %.sroa.5.0.off64848, %378 ], [ %extract.t810, %475 ]
  %.sroa.8629.1.off0 = phi i64 [ %.sroa.8629.0.off0849, %378 ], [ %extract.t814, %475 ]
  %.sroa.8629.1.off64 = phi i64 [ %.sroa.8629.0.off64850, %378 ], [ %extract.t817, %475 ]
  %.sroa.11.1.off0 = phi i64 [ %.sroa.11.0.off0851, %378 ], [ %extract.t821, %475 ]
  %.sroa.11.1.off64 = phi i64 [ %.sroa.11.0.off64852, %378 ], [ %extract.t824, %475 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !200
  %487 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %240) #6, !dbg !201, !srcloc !160
  %.sroa.036.2.extract.shift37.i230 = lshr i64 %487, 16, !dbg !204
  %488 = insertelement <2 x i64> poison, i64 %487, i64 0, !dbg !204
  %489 = insertelement <2 x i64> %488, i64 %.sroa.036.2.extract.shift37.i230, i64 1, !dbg !204
  %490 = shufflevector <2 x i64> %488, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %491 = lshr <2 x i64> %490, <i64 32, i64 48>, !dbg !201
  %492 = shufflevector <2 x i64> %489, <2 x i64> %491, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.34686.38.vec.insert1037 = trunc <4 x i64> %492 to <4 x i16>, !dbg !205
  %493 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %245) #6, !dbg !201, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i239 = lshr i64 %493, 16, !dbg !204
  %494 = insertelement <2 x i64> poison, i64 %493, i64 0, !dbg !204
  %495 = insertelement <2 x i64> %494, i64 %.sroa.036.2.extract.shift37.1.i239, i64 1, !dbg !204
  %496 = shufflevector <2 x i64> %494, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %497 = lshr <2 x i64> %496, <i64 32, i64 48>, !dbg !201
  %498 = shufflevector <2 x i64> %495, <2 x i64> %497, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.42691.46.vec.insert1036 = trunc <4 x i64> %498 to <4 x i16>, !dbg !205
  %499 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %249) #6, !dbg !201, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i248 = lshr i64 %499, 16, !dbg !204
  %500 = insertelement <2 x i64> poison, i64 %499, i64 0, !dbg !204
  %501 = insertelement <2 x i64> %500, i64 %.sroa.036.2.extract.shift37.145.i248, i64 1, !dbg !204
  %502 = shufflevector <2 x i64> %500, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %503 = lshr <2 x i64> %502, <i64 32, i64 48>, !dbg !201
  %504 = shufflevector <2 x i64> %501, <2 x i64> %503, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.50696.54.vec.insert1035 = trunc <4 x i64> %504 to <4 x i16>, !dbg !205
  %505 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %253) #6, !dbg !201, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i257 = lshr i64 %505, 16, !dbg !204
  %506 = insertelement <2 x i64> poison, i64 %505, i64 0, !dbg !204
  %507 = insertelement <2 x i64> %506, i64 %.sroa.036.2.extract.shift37.1.1.i257, i64 1, !dbg !204
  %508 = shufflevector <2 x i64> %506, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !201
  %509 = lshr <2 x i64> %508, <i64 32, i64 48>, !dbg !201
  %510 = shufflevector <2 x i64> %507, <2 x i64> %509, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !205
  %.sroa.58701.62.vec.insert1034 = trunc <4 x i64> %510 to <4 x i16>, !dbg !205
  %511 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %257) #6, !dbg !207, !srcloc !160
  %.sroa.036.2.extract.shift37.i269 = lshr i64 %511, 16, !dbg !210
  %512 = insertelement <2 x i64> poison, i64 %511, i64 0, !dbg !210
  %513 = insertelement <2 x i64> %512, i64 %.sroa.036.2.extract.shift37.i269, i64 1, !dbg !210
  %514 = shufflevector <2 x i64> %512, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %515 = lshr <2 x i64> %514, <i64 32, i64 48>, !dbg !207
  %516 = shufflevector <2 x i64> %513, <2 x i64> %515, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.34.38.vec.insert1033 = trunc <4 x i64> %516 to <4 x i16>, !dbg !211
  %517 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %261) #6, !dbg !207, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i278 = lshr i64 %517, 16, !dbg !210
  %518 = insertelement <2 x i64> poison, i64 %517, i64 0, !dbg !210
  %519 = insertelement <2 x i64> %518, i64 %.sroa.036.2.extract.shift37.1.i278, i64 1, !dbg !210
  %520 = shufflevector <2 x i64> %518, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %521 = lshr <2 x i64> %520, <i64 32, i64 48>, !dbg !207
  %522 = shufflevector <2 x i64> %519, <2 x i64> %521, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.42.46.vec.insert1032 = trunc <4 x i64> %522 to <4 x i16>, !dbg !211
  %523 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %265) #6, !dbg !207, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i287 = lshr i64 %523, 16, !dbg !210
  %524 = insertelement <2 x i64> poison, i64 %523, i64 0, !dbg !210
  %525 = insertelement <2 x i64> %524, i64 %.sroa.036.2.extract.shift37.145.i287, i64 1, !dbg !210
  %526 = shufflevector <2 x i64> %524, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %527 = lshr <2 x i64> %526, <i64 32, i64 48>, !dbg !207
  %528 = shufflevector <2 x i64> %525, <2 x i64> %527, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.50.54.vec.insert1031 = trunc <4 x i64> %528 to <4 x i16>, !dbg !211
  %529 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %269) #6, !dbg !207, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i296 = lshr i64 %529, 16, !dbg !210
  %530 = insertelement <2 x i64> poison, i64 %529, i64 0, !dbg !210
  %531 = insertelement <2 x i64> %530, i64 %.sroa.036.2.extract.shift37.1.1.i296, i64 1, !dbg !210
  %532 = shufflevector <2 x i64> %530, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !207
  %533 = lshr <2 x i64> %532, <i64 32, i64 48>, !dbg !207
  %534 = shufflevector <2 x i64> %531, <2 x i64> %533, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !211
  %.sroa.58.62.vec.insert1030 = trunc <4 x i64> %534 to <4 x i16>, !dbg !211
  %535 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %273) #6, !dbg !213, !srcloc !160
  %.sroa.036.2.extract.shift37.i308 = lshr i64 %535, 16, !dbg !216
  %536 = insertelement <2 x i64> poison, i64 %535, i64 0, !dbg !216
  %537 = insertelement <2 x i64> %536, i64 %.sroa.036.2.extract.shift37.i308, i64 1, !dbg !216
  %538 = shufflevector <2 x i64> %536, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %539 = lshr <2 x i64> %538, <i64 32, i64 48>, !dbg !213
  %540 = shufflevector <2 x i64> %537, <2 x i64> %539, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.34766.38.vec.insert1029 = trunc <4 x i64> %540 to <4 x i16>, !dbg !217
  %541 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %277) #6, !dbg !213, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i317 = lshr i64 %541, 16, !dbg !216
  %542 = insertelement <2 x i64> poison, i64 %541, i64 0, !dbg !216
  %543 = insertelement <2 x i64> %542, i64 %.sroa.036.2.extract.shift37.1.i317, i64 1, !dbg !216
  %544 = shufflevector <2 x i64> %542, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %545 = lshr <2 x i64> %544, <i64 32, i64 48>, !dbg !213
  %546 = shufflevector <2 x i64> %543, <2 x i64> %545, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.42771.46.vec.insert1028 = trunc <4 x i64> %546 to <4 x i16>, !dbg !217
  %547 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %281) #6, !dbg !213, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i326 = lshr i64 %547, 16, !dbg !216
  %548 = insertelement <2 x i64> poison, i64 %547, i64 0, !dbg !216
  %549 = insertelement <2 x i64> %548, i64 %.sroa.036.2.extract.shift37.145.i326, i64 1, !dbg !216
  %550 = shufflevector <2 x i64> %548, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %551 = lshr <2 x i64> %550, <i64 32, i64 48>, !dbg !213
  %552 = shufflevector <2 x i64> %549, <2 x i64> %551, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.50776.54.vec.insert1027 = trunc <4 x i64> %552 to <4 x i16>, !dbg !217
  %553 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %285) #6, !dbg !213, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i335 = lshr i64 %553, 16, !dbg !216
  %554 = insertelement <2 x i64> poison, i64 %553, i64 0, !dbg !216
  %555 = insertelement <2 x i64> %554, i64 %.sroa.036.2.extract.shift37.1.1.i335, i64 1, !dbg !216
  %556 = shufflevector <2 x i64> %554, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !213
  %557 = lshr <2 x i64> %556, <i64 32, i64 48>, !dbg !213
  %558 = shufflevector <2 x i64> %555, <2 x i64> %557, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !217
  %.sroa.58781.62.vec.insert1026 = trunc <4 x i64> %558 to <4 x i16>, !dbg !217
  %559 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %289) #6, !dbg !219, !srcloc !160
  %.sroa.036.2.extract.shift37.i347 = lshr i64 %559, 16, !dbg !222
  %560 = insertelement <2 x i64> poison, i64 %559, i64 0, !dbg !222
  %561 = insertelement <2 x i64> %560, i64 %.sroa.036.2.extract.shift37.i347, i64 1, !dbg !222
  %562 = shufflevector <2 x i64> %560, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !219
  %563 = lshr <2 x i64> %562, <i64 32, i64 48>, !dbg !219
  %564 = shufflevector <2 x i64> %561, <2 x i64> %563, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !223
  %.sroa.34726.38.vec.insert1025 = trunc <4 x i64> %564 to <4 x i16>, !dbg !223
  %565 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %293) #6, !dbg !219, !srcloc !160
  %.sroa.036.2.extract.shift37.1.i356 = lshr i64 %565, 16, !dbg !222
  %566 = insertelement <2 x i64> poison, i64 %565, i64 0, !dbg !222
  %567 = insertelement <2 x i64> %566, i64 %.sroa.036.2.extract.shift37.1.i356, i64 1, !dbg !222
  %568 = shufflevector <2 x i64> %566, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !219
  %569 = lshr <2 x i64> %568, <i64 32, i64 48>, !dbg !219
  %570 = shufflevector <2 x i64> %567, <2 x i64> %569, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !223
  %.sroa.42731.46.vec.insert1024 = trunc <4 x i64> %570 to <4 x i16>, !dbg !223
  %571 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %297) #6, !dbg !219, !srcloc !160
  %.sroa.036.2.extract.shift37.145.i365 = lshr i64 %571, 16, !dbg !222
  %572 = insertelement <2 x i64> poison, i64 %571, i64 0, !dbg !222
  %573 = insertelement <2 x i64> %572, i64 %.sroa.036.2.extract.shift37.145.i365, i64 1, !dbg !222
  %574 = shufflevector <2 x i64> %572, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !219
  %575 = lshr <2 x i64> %574, <i64 32, i64 48>, !dbg !219
  %576 = shufflevector <2 x i64> %573, <2 x i64> %575, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !223
  %.sroa.50736.54.vec.insert1023 = trunc <4 x i64> %576 to <4 x i16>, !dbg !223
  %577 = tail call i64 asm sideeffect "ds_read_b64 $0, $1\0A", "=v,v,~{memory}"(i32 %301) #6, !dbg !219, !srcloc !160
  %.sroa.036.2.extract.shift37.1.1.i374 = lshr i64 %577, 16, !dbg !222
  %578 = insertelement <2 x i64> poison, i64 %577, i64 0, !dbg !222
  %579 = insertelement <2 x i64> %578, i64 %.sroa.036.2.extract.shift37.1.1.i374, i64 1, !dbg !222
  %580 = shufflevector <2 x i64> %578, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !219
  %581 = lshr <2 x i64> %580, <i64 32, i64 48>, !dbg !219
  %582 = shufflevector <2 x i64> %579, <2 x i64> %581, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !223
  %.sroa.58741.62.vec.insert1022 = trunc <4 x i64> %582 to <4 x i16>, !dbg !223
  tail call void @llvm.amdgcn.s.barrier(), !dbg !225
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !226
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !227
  %583 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0746.6.vec.insert1045, <4 x i16> %.sroa.0666.6.vec.insert1053, <4 x float> %.val33.i843844, i32 0, i32 0, i32 0), !dbg !228
  %584 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10751.14.vec.insert1044, <4 x i16> %.sroa.10671.14.vec.insert1052, <4 x float> %583, i32 0, i32 0, i32 0), !dbg !236
  %585 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0746.6.vec.insert1045, <4 x i16> %.sroa.18676.22.vec.insert1051, <4 x float> %.val33.1.i458860, i32 0, i32 0, i32 0), !dbg !228
  %586 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10751.14.vec.insert1044, <4 x i16> %.sroa.26681.30.vec.insert1050, <4 x float> %585, i32 0, i32 0, i32 0), !dbg !236
  %587 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18756.22.vec.insert1043, <4 x i16> %.sroa.0666.6.vec.insert1053, <4 x float> %.val33.138.i463863, i32 0, i32 0, i32 0), !dbg !228
  %588 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26761.30.vec.insert1042, <4 x i16> %.sroa.10671.14.vec.insert1052, <4 x float> %587, i32 0, i32 0, i32 0), !dbg !236
  %589 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18756.22.vec.insert1043, <4 x i16> %.sroa.18676.22.vec.insert1051, <4 x float> %.val33.1.1.i468866, i32 0, i32 0, i32 0), !dbg !228
  %590 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26761.30.vec.insert1042, <4 x i16> %.sroa.26681.30.vec.insert1050, <4 x float> %589, i32 0, i32 0, i32 0), !dbg !236
  %591 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0746.6.vec.insert1045, <4 x i16> %.sroa.0633.6.vec.insert1049, <4 x float> %.val33.i393869, i32 0, i32 0, i32 0), !dbg !239
  %592 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10751.14.vec.insert1044, <4 x i16> %.sroa.10.14.vec.insert1048, <4 x float> %591, i32 0, i32 0, i32 0), !dbg !243
  %593 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0746.6.vec.insert1045, <4 x i16> %.sroa.18.22.vec.insert1047, <4 x float> %367, i32 0, i32 0, i32 0), !dbg !239
  %594 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10751.14.vec.insert1044, <4 x i16> %.sroa.26.30.vec.insert1046, <4 x float> %593, i32 0, i32 0, i32 0), !dbg !243
  %595 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18756.22.vec.insert1043, <4 x i16> %.sroa.0633.6.vec.insert1049, <4 x float> %366, i32 0, i32 0, i32 0), !dbg !239
  %596 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26761.30.vec.insert1042, <4 x i16> %.sroa.10.14.vec.insert1048, <4 x float> %595, i32 0, i32 0, i32 0), !dbg !243
  %597 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18756.22.vec.insert1043, <4 x i16> %.sroa.18.22.vec.insert1047, <4 x float> %365, i32 0, i32 0, i32 0), !dbg !239
  %598 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26761.30.vec.insert1042, <4 x i16> %.sroa.26.30.vec.insert1046, <4 x float> %597, i32 0, i32 0, i32 0), !dbg !243
  %599 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0706.6.vec.insert1041, <4 x i16> %.sroa.0666.6.vec.insert1053, <4 x float> %.val33.i413878, i32 0, i32 0, i32 0), !dbg !246
  %600 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10711.14.vec.insert1040, <4 x i16> %.sroa.10671.14.vec.insert1052, <4 x float> %599, i32 0, i32 0, i32 0), !dbg !250
  %601 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0706.6.vec.insert1041, <4 x i16> %.sroa.18676.22.vec.insert1051, <4 x float> %364, i32 0, i32 0, i32 0), !dbg !246
  %602 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10711.14.vec.insert1040, <4 x i16> %.sroa.26681.30.vec.insert1050, <4 x float> %601, i32 0, i32 0, i32 0), !dbg !250
  %603 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18716.22.vec.insert1039, <4 x i16> %.sroa.0666.6.vec.insert1053, <4 x float> %363, i32 0, i32 0, i32 0), !dbg !246
  %604 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26721.30.vec.insert1038, <4 x i16> %.sroa.10671.14.vec.insert1052, <4 x float> %603, i32 0, i32 0, i32 0), !dbg !250
  %605 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18716.22.vec.insert1039, <4 x i16> %.sroa.18676.22.vec.insert1051, <4 x float> %362, i32 0, i32 0, i32 0), !dbg !246
  %606 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26721.30.vec.insert1038, <4 x i16> %.sroa.26681.30.vec.insert1050, <4 x float> %605, i32 0, i32 0, i32 0), !dbg !250
  %607 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0706.6.vec.insert1041, <4 x i16> %.sroa.0633.6.vec.insert1049, <4 x float> %.val33.i433887, i32 0, i32 0, i32 0), !dbg !253
  %608 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10711.14.vec.insert1040, <4 x i16> %.sroa.10.14.vec.insert1048, <4 x float> %607, i32 0, i32 0, i32 0), !dbg !257
  %609 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.0706.6.vec.insert1041, <4 x i16> %.sroa.18.22.vec.insert1047, <4 x float> %361, i32 0, i32 0, i32 0), !dbg !253
  %610 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.10711.14.vec.insert1040, <4 x i16> %.sroa.26.30.vec.insert1046, <4 x float> %609, i32 0, i32 0, i32 0), !dbg !257
  %611 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18716.22.vec.insert1039, <4 x i16> %.sroa.0633.6.vec.insert1049, <4 x float> %360, i32 0, i32 0, i32 0), !dbg !253
  %612 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26721.30.vec.insert1038, <4 x i16> %.sroa.10.14.vec.insert1048, <4 x float> %611, i32 0, i32 0, i32 0), !dbg !257
  %613 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.18716.22.vec.insert1039, <4 x i16> %.sroa.18.22.vec.insert1047, <4 x float> %359, i32 0, i32 0, i32 0), !dbg !253
  %614 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.26721.30.vec.insert1038, <4 x i16> %.sroa.26.30.vec.insert1046, <4 x float> %613, i32 0, i32 0, i32 0), !dbg !257
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !260
  tail call void @llvm.amdgcn.s.barrier(), !dbg !261
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !262
  br i1 %.not, label %616, label %615, !dbg !263

615:                                              ; preds = %486
  tail call void asm sideeffect "s_waitcnt vmcnt(2)", ""() #6, !dbg !264, !srcloc !265
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %309, i64 %.sroa.0630.1.off0) #6, !dbg !266, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %313, i64 %.sroa.0630.1.off64) #6, !dbg !272, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %317, i64 %.sroa.5631.1.off0) #6, !dbg !266, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %321, i64 %.sroa.5631.1.off64) #6, !dbg !272, !srcloc !271
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", ""() #6, !dbg !274, !srcloc !275
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %327, i64 %.sroa.0628.1.off0) #6, !dbg !276, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %331, i64 %.sroa.0628.1.off64) #6, !dbg !280, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %335, i64 %.sroa.5.1.off0) #6, !dbg !276, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %339, i64 %.sroa.5.1.off64) #6, !dbg !280, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %343, i64 %.sroa.8629.1.off0) #6, !dbg !276, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %347, i64 %.sroa.8629.1.off64) #6, !dbg !280, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %351, i64 %.sroa.11.1.off0) #6, !dbg !276, !srcloc !271
  tail call void asm sideeffect "ds_write_b64 $0, $1\0A", "v,v,~{memory}"(i32 %355, i64 %.sroa.11.1.off64) #6, !dbg !280, !srcloc !271
  br label %616, !dbg !282

616:                                              ; preds = %615, %486
  tail call void @llvm.amdgcn.s.barrier(), !dbg !283
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !284
  tail call void @llvm.amdgcn.s.setprio(i16 1), !dbg !285
  %617 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34766.38.vec.insert1029, <4 x i16> %.sroa.34686.38.vec.insert1037, <4 x float> %584, i32 0, i32 0, i32 0), !dbg !286
  %618 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42771.46.vec.insert1028, <4 x i16> %.sroa.42691.46.vec.insert1036, <4 x float> %617, i32 0, i32 0, i32 0), !dbg !290
  %619 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34766.38.vec.insert1029, <4 x i16> %.sroa.50696.54.vec.insert1035, <4 x float> %586, i32 0, i32 0, i32 0), !dbg !286
  %620 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42771.46.vec.insert1028, <4 x i16> %.sroa.58701.62.vec.insert1034, <4 x float> %619, i32 0, i32 0, i32 0), !dbg !290
  %621 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50776.54.vec.insert1027, <4 x i16> %.sroa.34686.38.vec.insert1037, <4 x float> %588, i32 0, i32 0, i32 0), !dbg !286
  %622 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58781.62.vec.insert1026, <4 x i16> %.sroa.42691.46.vec.insert1036, <4 x float> %621, i32 0, i32 0, i32 0), !dbg !290
  %623 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50776.54.vec.insert1027, <4 x i16> %.sroa.50696.54.vec.insert1035, <4 x float> %590, i32 0, i32 0, i32 0), !dbg !286
  %624 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58781.62.vec.insert1026, <4 x i16> %.sroa.58701.62.vec.insert1034, <4 x float> %623, i32 0, i32 0, i32 0), !dbg !290
  %625 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34766.38.vec.insert1029, <4 x i16> %.sroa.34.38.vec.insert1033, <4 x float> %592, i32 0, i32 0, i32 0), !dbg !293
  %626 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42771.46.vec.insert1028, <4 x i16> %.sroa.42.46.vec.insert1032, <4 x float> %625, i32 0, i32 0, i32 0), !dbg !297
  %627 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34766.38.vec.insert1029, <4 x i16> %.sroa.50.54.vec.insert1031, <4 x float> %594, i32 0, i32 0, i32 0), !dbg !293
  %628 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42771.46.vec.insert1028, <4 x i16> %.sroa.58.62.vec.insert1030, <4 x float> %627, i32 0, i32 0, i32 0), !dbg !297
  %629 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50776.54.vec.insert1027, <4 x i16> %.sroa.34.38.vec.insert1033, <4 x float> %596, i32 0, i32 0, i32 0), !dbg !293
  %630 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58781.62.vec.insert1026, <4 x i16> %.sroa.42.46.vec.insert1032, <4 x float> %629, i32 0, i32 0, i32 0), !dbg !297
  %631 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50776.54.vec.insert1027, <4 x i16> %.sroa.50.54.vec.insert1031, <4 x float> %598, i32 0, i32 0, i32 0), !dbg !293
  %632 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58781.62.vec.insert1026, <4 x i16> %.sroa.58.62.vec.insert1030, <4 x float> %631, i32 0, i32 0, i32 0), !dbg !297
  %633 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34726.38.vec.insert1025, <4 x i16> %.sroa.34686.38.vec.insert1037, <4 x float> %600, i32 0, i32 0, i32 0), !dbg !300
  %634 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42731.46.vec.insert1024, <4 x i16> %.sroa.42691.46.vec.insert1036, <4 x float> %633, i32 0, i32 0, i32 0), !dbg !304
  %635 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34726.38.vec.insert1025, <4 x i16> %.sroa.50696.54.vec.insert1035, <4 x float> %602, i32 0, i32 0, i32 0), !dbg !300
  %636 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42731.46.vec.insert1024, <4 x i16> %.sroa.58701.62.vec.insert1034, <4 x float> %635, i32 0, i32 0, i32 0), !dbg !304
  %637 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50736.54.vec.insert1023, <4 x i16> %.sroa.34686.38.vec.insert1037, <4 x float> %604, i32 0, i32 0, i32 0), !dbg !300
  %638 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58741.62.vec.insert1022, <4 x i16> %.sroa.42691.46.vec.insert1036, <4 x float> %637, i32 0, i32 0, i32 0), !dbg !304
  %639 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50736.54.vec.insert1023, <4 x i16> %.sroa.50696.54.vec.insert1035, <4 x float> %606, i32 0, i32 0, i32 0), !dbg !300
  %640 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58741.62.vec.insert1022, <4 x i16> %.sroa.58701.62.vec.insert1034, <4 x float> %639, i32 0, i32 0, i32 0), !dbg !304
  %641 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34726.38.vec.insert1025, <4 x i16> %.sroa.34.38.vec.insert1033, <4 x float> %608, i32 0, i32 0, i32 0), !dbg !307
  %642 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42731.46.vec.insert1024, <4 x i16> %.sroa.42.46.vec.insert1032, <4 x float> %641, i32 0, i32 0, i32 0), !dbg !311
  %643 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.34726.38.vec.insert1025, <4 x i16> %.sroa.50.54.vec.insert1031, <4 x float> %610, i32 0, i32 0, i32 0), !dbg !307
  %644 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.42731.46.vec.insert1024, <4 x i16> %.sroa.58.62.vec.insert1030, <4 x float> %643, i32 0, i32 0, i32 0), !dbg !311
  %645 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50736.54.vec.insert1023, <4 x i16> %.sroa.34.38.vec.insert1033, <4 x float> %612, i32 0, i32 0, i32 0), !dbg !307
  %646 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58741.62.vec.insert1022, <4 x i16> %.sroa.42.46.vec.insert1032, <4 x float> %645, i32 0, i32 0, i32 0), !dbg !311
  %647 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.50736.54.vec.insert1023, <4 x i16> %.sroa.50.54.vec.insert1031, <4 x float> %614, i32 0, i32 0, i32 0), !dbg !307
  %648 = tail call contract <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %.sroa.58741.62.vec.insert1022, <4 x i16> %.sroa.58.62.vec.insert1030, <4 x float> %647, i32 0, i32 0, i32 0), !dbg !311
  tail call void @llvm.amdgcn.s.setprio(i16 0), !dbg !314
  tail call void @llvm.amdgcn.s.barrier(), !dbg !315
  tail call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !316
  %exitcond.not = icmp eq i32 %368, 128, !dbg !317
  br i1 %exitcond.not, label %356, label %358, !dbg !141, !llvm.loop !318

649:                                              ; preds = %356
  tail call void @llvm.amdgcn.s.barrier(), !dbg !321
  br label %650, !dbg !322

650:                                              ; preds = %649, %356
  %651 = shl nsw i32 %12, 2
  %652 = or disjoint i32 %651, %16
  %653 = shl nsw i32 %13, 3
  %654 = trunc i64 %.sroa.20.0.copyload to i32
  %655 = mul nsw i32 %133, %654
  %656 = or disjoint i32 %133, 1
  %657 = mul nsw i32 %656, %654
  %658 = or disjoint i32 %133, 2
  %659 = mul nsw i32 %658, %654
  %660 = or disjoint i32 %133, 3
  %661 = mul nsw i32 %660, %654
  %662 = add nsw i32 %655, %131
  %663 = sext i32 %662 to i64
  %664 = add nsw i32 %657, %131
  %665 = sext i32 %664 to i64
  %666 = add nsw i32 %659, %131
  %667 = sext i32 %666 to i64
  %668 = add nsw i32 %661, %131
  %669 = sext i32 %668 to i64
  %670 = add nsw i32 %655, %150
  %671 = sext i32 %670 to i64
  %672 = add nsw i32 %657, %150
  %673 = sext i32 %672 to i64
  %674 = add nsw i32 %659, %150
  %675 = sext i32 %674 to i64
  %676 = add nsw i32 %661, %150
  %677 = sext i32 %676 to i64
  %678 = or disjoint i32 %133, 16
  %679 = mul nsw i32 %678, %654
  %680 = or disjoint i32 %133, 17
  %681 = mul nsw i32 %680, %654
  %682 = or disjoint i32 %133, 18
  %683 = mul nsw i32 %682, %654
  %684 = or disjoint i32 %133, 19
  %685 = mul nsw i32 %684, %654
  %686 = add nsw i32 %679, %131
  %687 = sext i32 %686 to i64
  %688 = add nsw i32 %681, %131
  %689 = sext i32 %688 to i64
  %690 = add nsw i32 %683, %131
  %691 = sext i32 %690 to i64
  %692 = add nsw i32 %685, %131
  %693 = sext i32 %692 to i64
  %694 = add nsw i32 %679, %150
  %695 = sext i32 %694 to i64
  %696 = add nsw i32 %681, %150
  %697 = sext i32 %696 to i64
  %698 = add nsw i32 %683, %150
  %699 = sext i32 %698 to i64
  %700 = add nsw i32 %685, %150
  %701 = sext i32 %700 to i64
  %702 = or disjoint i32 %653, %17, !dbg !323
  %703 = shl nsw i32 %652, 5, !dbg !324
  %704 = shl nsw i32 %702, 5, !dbg !332
  %705 = sext i32 %703 to i64, !dbg !333
  %706 = mul i64 %.sroa.20.0.copyload, %705, !dbg !336
  %707 = sext i32 %704 to i64, !dbg !337
  %708 = getelementptr float, ptr %.sroa.17797.0.copyload, i64 %706, !dbg !338
  %709 = getelementptr float, ptr %708, i64 %707, !dbg !338
  %.sroa.0.0.vec.extract = extractelement <4 x float> %618, i64 0, !dbg !339
  %710 = getelementptr inbounds float, ptr %709, i64 %663, !dbg !340
  store float %.sroa.0.0.vec.extract, ptr %710, align 4, !dbg !341, !tbaa !342
  %.sroa.0.4.vec.extract = extractelement <4 x float> %618, i64 1, !dbg !346
  %711 = getelementptr inbounds float, ptr %709, i64 %665, !dbg !347
  store float %.sroa.0.4.vec.extract, ptr %711, align 4, !dbg !348, !tbaa !342
  %.sroa.0.8.vec.extract = extractelement <4 x float> %618, i64 2, !dbg !349
  %712 = getelementptr inbounds float, ptr %709, i64 %667, !dbg !350
  store float %.sroa.0.8.vec.extract, ptr %712, align 4, !dbg !351, !tbaa !342
  %.sroa.0.12.vec.extract = extractelement <4 x float> %618, i64 3, !dbg !352
  %713 = getelementptr inbounds float, ptr %709, i64 %669, !dbg !353
  store float %.sroa.0.12.vec.extract, ptr %713, align 4, !dbg !354, !tbaa !342
  %.sroa.12.16.vec.extract = extractelement <4 x float> %620, i64 0, !dbg !339
  %714 = getelementptr inbounds float, ptr %709, i64 %671, !dbg !340
  store float %.sroa.12.16.vec.extract, ptr %714, align 4, !dbg !341, !tbaa !342
  %.sroa.12.20.vec.extract = extractelement <4 x float> %620, i64 1, !dbg !346
  %715 = getelementptr inbounds float, ptr %709, i64 %673, !dbg !347
  store float %.sroa.12.20.vec.extract, ptr %715, align 4, !dbg !348, !tbaa !342
  %.sroa.12.24.vec.extract = extractelement <4 x float> %620, i64 2, !dbg !349
  %716 = getelementptr inbounds float, ptr %709, i64 %675, !dbg !350
  store float %.sroa.12.24.vec.extract, ptr %716, align 4, !dbg !351, !tbaa !342
  %.sroa.12.28.vec.extract = extractelement <4 x float> %620, i64 3, !dbg !352
  %717 = getelementptr inbounds float, ptr %709, i64 %677, !dbg !353
  store float %.sroa.12.28.vec.extract, ptr %717, align 4, !dbg !354, !tbaa !342
  %.sroa.22.32.vec.extract = extractelement <4 x float> %622, i64 0, !dbg !339
  %718 = getelementptr inbounds float, ptr %709, i64 %687, !dbg !340
  store float %.sroa.22.32.vec.extract, ptr %718, align 4, !dbg !341, !tbaa !342
  %.sroa.22.36.vec.extract = extractelement <4 x float> %622, i64 1, !dbg !346
  %719 = getelementptr inbounds float, ptr %709, i64 %689, !dbg !347
  store float %.sroa.22.36.vec.extract, ptr %719, align 4, !dbg !348, !tbaa !342
  %.sroa.22.40.vec.extract = extractelement <4 x float> %622, i64 2, !dbg !349
  %720 = getelementptr inbounds float, ptr %709, i64 %691, !dbg !350
  store float %.sroa.22.40.vec.extract, ptr %720, align 4, !dbg !351, !tbaa !342
  %.sroa.22.44.vec.extract = extractelement <4 x float> %622, i64 3, !dbg !352
  %721 = getelementptr inbounds float, ptr %709, i64 %693, !dbg !353
  store float %.sroa.22.44.vec.extract, ptr %721, align 4, !dbg !354, !tbaa !342
  %.sroa.32.48.vec.extract = extractelement <4 x float> %624, i64 0, !dbg !339
  %722 = getelementptr inbounds float, ptr %709, i64 %695, !dbg !340
  store float %.sroa.32.48.vec.extract, ptr %722, align 4, !dbg !341, !tbaa !342
  %.sroa.32.52.vec.extract = extractelement <4 x float> %624, i64 1, !dbg !346
  %723 = getelementptr inbounds float, ptr %709, i64 %697, !dbg !347
  store float %.sroa.32.52.vec.extract, ptr %723, align 4, !dbg !348, !tbaa !342
  %.sroa.32.56.vec.extract = extractelement <4 x float> %624, i64 2, !dbg !349
  %724 = getelementptr inbounds float, ptr %709, i64 %699, !dbg !350
  store float %.sroa.32.56.vec.extract, ptr %724, align 4, !dbg !351, !tbaa !342
  %.sroa.32.60.vec.extract = extractelement <4 x float> %624, i64 3, !dbg !352
  %725 = getelementptr inbounds float, ptr %709, i64 %701, !dbg !353
  store float %.sroa.32.60.vec.extract, ptr %725, align 4, !dbg !354, !tbaa !342
  %726 = or disjoint i32 %653, %17, !dbg !323
  %727 = shl i32 %726, 5, !dbg !332
  %728 = or disjoint i32 %727, 128, !dbg !332
  %729 = sext i32 %728 to i64, !dbg !337
  %730 = getelementptr float, ptr %708, i64 %729, !dbg !338
  %.sroa.42.64.vec.extract = extractelement <4 x float> %626, i64 0, !dbg !339
  %731 = getelementptr inbounds float, ptr %730, i64 %663, !dbg !340
  store float %.sroa.42.64.vec.extract, ptr %731, align 4, !dbg !341, !tbaa !342
  %.sroa.42.68.vec.extract = extractelement <4 x float> %626, i64 1, !dbg !346
  %732 = getelementptr inbounds float, ptr %730, i64 %665, !dbg !347
  store float %.sroa.42.68.vec.extract, ptr %732, align 4, !dbg !348, !tbaa !342
  %.sroa.42.72.vec.extract = extractelement <4 x float> %626, i64 2, !dbg !349
  %733 = getelementptr inbounds float, ptr %730, i64 %667, !dbg !350
  store float %.sroa.42.72.vec.extract, ptr %733, align 4, !dbg !351, !tbaa !342
  %.sroa.42.76.vec.extract = extractelement <4 x float> %626, i64 3, !dbg !352
  %734 = getelementptr inbounds float, ptr %730, i64 %669, !dbg !353
  store float %.sroa.42.76.vec.extract, ptr %734, align 4, !dbg !354, !tbaa !342
  %.sroa.52.80.vec.extract = extractelement <4 x float> %628, i64 0, !dbg !339
  %735 = getelementptr inbounds float, ptr %730, i64 %671, !dbg !340
  store float %.sroa.52.80.vec.extract, ptr %735, align 4, !dbg !341, !tbaa !342
  %.sroa.52.84.vec.extract = extractelement <4 x float> %628, i64 1, !dbg !346
  %736 = getelementptr inbounds float, ptr %730, i64 %673, !dbg !347
  store float %.sroa.52.84.vec.extract, ptr %736, align 4, !dbg !348, !tbaa !342
  %.sroa.52.88.vec.extract = extractelement <4 x float> %628, i64 2, !dbg !349
  %737 = getelementptr inbounds float, ptr %730, i64 %675, !dbg !350
  store float %.sroa.52.88.vec.extract, ptr %737, align 4, !dbg !351, !tbaa !342
  %.sroa.52.92.vec.extract = extractelement <4 x float> %628, i64 3, !dbg !352
  %738 = getelementptr inbounds float, ptr %730, i64 %677, !dbg !353
  store float %.sroa.52.92.vec.extract, ptr %738, align 4, !dbg !354, !tbaa !342
  %.sroa.62.96.vec.extract = extractelement <4 x float> %630, i64 0, !dbg !339
  %739 = getelementptr inbounds float, ptr %730, i64 %687, !dbg !340
  store float %.sroa.62.96.vec.extract, ptr %739, align 4, !dbg !341, !tbaa !342
  %.sroa.62.100.vec.extract = extractelement <4 x float> %630, i64 1, !dbg !346
  %740 = getelementptr inbounds float, ptr %730, i64 %689, !dbg !347
  store float %.sroa.62.100.vec.extract, ptr %740, align 4, !dbg !348, !tbaa !342
  %.sroa.62.104.vec.extract = extractelement <4 x float> %630, i64 2, !dbg !349
  %741 = getelementptr inbounds float, ptr %730, i64 %691, !dbg !350
  store float %.sroa.62.104.vec.extract, ptr %741, align 4, !dbg !351, !tbaa !342
  %.sroa.62.108.vec.extract = extractelement <4 x float> %630, i64 3, !dbg !352
  %742 = getelementptr inbounds float, ptr %730, i64 %693, !dbg !353
  store float %.sroa.62.108.vec.extract, ptr %742, align 4, !dbg !354, !tbaa !342
  %.sroa.72.112.vec.extract = extractelement <4 x float> %632, i64 0, !dbg !339
  %743 = getelementptr inbounds float, ptr %730, i64 %695, !dbg !340
  store float %.sroa.72.112.vec.extract, ptr %743, align 4, !dbg !341, !tbaa !342
  %.sroa.72.116.vec.extract = extractelement <4 x float> %632, i64 1, !dbg !346
  %744 = getelementptr inbounds float, ptr %730, i64 %697, !dbg !347
  store float %.sroa.72.116.vec.extract, ptr %744, align 4, !dbg !348, !tbaa !342
  %.sroa.72.120.vec.extract = extractelement <4 x float> %632, i64 2, !dbg !349
  %745 = getelementptr inbounds float, ptr %730, i64 %699, !dbg !350
  store float %.sroa.72.120.vec.extract, ptr %745, align 4, !dbg !351, !tbaa !342
  %.sroa.72.124.vec.extract = extractelement <4 x float> %632, i64 3, !dbg !352
  %746 = getelementptr inbounds float, ptr %730, i64 %701, !dbg !353
  store float %.sroa.72.124.vec.extract, ptr %746, align 4, !dbg !354, !tbaa !342
  %747 = shl i32 %652, 5, !dbg !324
  %748 = add i32 %747, 64, !dbg !324
  %749 = sext i32 %748 to i64, !dbg !333
  %750 = mul i64 %.sroa.20.0.copyload, %749, !dbg !336
  %751 = getelementptr float, ptr %.sroa.17797.0.copyload, i64 %750, !dbg !338
  %752 = getelementptr float, ptr %751, i64 %707, !dbg !338
  %.sroa.82.128.vec.extract = extractelement <4 x float> %634, i64 0, !dbg !339
  %753 = getelementptr inbounds float, ptr %752, i64 %663, !dbg !340
  store float %.sroa.82.128.vec.extract, ptr %753, align 4, !dbg !341, !tbaa !342
  %.sroa.82.132.vec.extract = extractelement <4 x float> %634, i64 1, !dbg !346
  %754 = getelementptr inbounds float, ptr %752, i64 %665, !dbg !347
  store float %.sroa.82.132.vec.extract, ptr %754, align 4, !dbg !348, !tbaa !342
  %.sroa.82.136.vec.extract = extractelement <4 x float> %634, i64 2, !dbg !349
  %755 = getelementptr inbounds float, ptr %752, i64 %667, !dbg !350
  store float %.sroa.82.136.vec.extract, ptr %755, align 4, !dbg !351, !tbaa !342
  %.sroa.82.140.vec.extract = extractelement <4 x float> %634, i64 3, !dbg !352
  %756 = getelementptr inbounds float, ptr %752, i64 %669, !dbg !353
  store float %.sroa.82.140.vec.extract, ptr %756, align 4, !dbg !354, !tbaa !342
  %.sroa.92.144.vec.extract = extractelement <4 x float> %636, i64 0, !dbg !339
  %757 = getelementptr inbounds float, ptr %752, i64 %671, !dbg !340
  store float %.sroa.92.144.vec.extract, ptr %757, align 4, !dbg !341, !tbaa !342
  %.sroa.92.148.vec.extract = extractelement <4 x float> %636, i64 1, !dbg !346
  %758 = getelementptr inbounds float, ptr %752, i64 %673, !dbg !347
  store float %.sroa.92.148.vec.extract, ptr %758, align 4, !dbg !348, !tbaa !342
  %.sroa.92.152.vec.extract = extractelement <4 x float> %636, i64 2, !dbg !349
  %759 = getelementptr inbounds float, ptr %752, i64 %675, !dbg !350
  store float %.sroa.92.152.vec.extract, ptr %759, align 4, !dbg !351, !tbaa !342
  %.sroa.92.156.vec.extract = extractelement <4 x float> %636, i64 3, !dbg !352
  %760 = getelementptr inbounds float, ptr %752, i64 %677, !dbg !353
  store float %.sroa.92.156.vec.extract, ptr %760, align 4, !dbg !354, !tbaa !342
  %.sroa.102.160.vec.extract = extractelement <4 x float> %638, i64 0, !dbg !339
  %761 = getelementptr inbounds float, ptr %752, i64 %687, !dbg !340
  store float %.sroa.102.160.vec.extract, ptr %761, align 4, !dbg !341, !tbaa !342
  %.sroa.102.164.vec.extract = extractelement <4 x float> %638, i64 1, !dbg !346
  %762 = getelementptr inbounds float, ptr %752, i64 %689, !dbg !347
  store float %.sroa.102.164.vec.extract, ptr %762, align 4, !dbg !348, !tbaa !342
  %.sroa.102.168.vec.extract = extractelement <4 x float> %638, i64 2, !dbg !349
  %763 = getelementptr inbounds float, ptr %752, i64 %691, !dbg !350
  store float %.sroa.102.168.vec.extract, ptr %763, align 4, !dbg !351, !tbaa !342
  %.sroa.102.172.vec.extract = extractelement <4 x float> %638, i64 3, !dbg !352
  %764 = getelementptr inbounds float, ptr %752, i64 %693, !dbg !353
  store float %.sroa.102.172.vec.extract, ptr %764, align 4, !dbg !354, !tbaa !342
  %.sroa.112.176.vec.extract = extractelement <4 x float> %640, i64 0, !dbg !339
  %765 = getelementptr inbounds float, ptr %752, i64 %695, !dbg !340
  store float %.sroa.112.176.vec.extract, ptr %765, align 4, !dbg !341, !tbaa !342
  %.sroa.112.180.vec.extract = extractelement <4 x float> %640, i64 1, !dbg !346
  %766 = getelementptr inbounds float, ptr %752, i64 %697, !dbg !347
  store float %.sroa.112.180.vec.extract, ptr %766, align 4, !dbg !348, !tbaa !342
  %.sroa.112.184.vec.extract = extractelement <4 x float> %640, i64 2, !dbg !349
  %767 = getelementptr inbounds float, ptr %752, i64 %699, !dbg !350
  store float %.sroa.112.184.vec.extract, ptr %767, align 4, !dbg !351, !tbaa !342
  %.sroa.112.188.vec.extract = extractelement <4 x float> %640, i64 3, !dbg !352
  %768 = getelementptr inbounds float, ptr %752, i64 %701, !dbg !353
  store float %.sroa.112.188.vec.extract, ptr %768, align 4, !dbg !354, !tbaa !342
  %769 = getelementptr float, ptr %751, i64 %729, !dbg !338
  %.sroa.122.192.vec.extract = extractelement <4 x float> %642, i64 0, !dbg !339
  %770 = getelementptr inbounds float, ptr %769, i64 %663, !dbg !340
  store float %.sroa.122.192.vec.extract, ptr %770, align 4, !dbg !341, !tbaa !342
  %.sroa.122.196.vec.extract = extractelement <4 x float> %642, i64 1, !dbg !346
  %771 = getelementptr inbounds float, ptr %769, i64 %665, !dbg !347
  store float %.sroa.122.196.vec.extract, ptr %771, align 4, !dbg !348, !tbaa !342
  %.sroa.122.200.vec.extract = extractelement <4 x float> %642, i64 2, !dbg !349
  %772 = getelementptr inbounds float, ptr %769, i64 %667, !dbg !350
  store float %.sroa.122.200.vec.extract, ptr %772, align 4, !dbg !351, !tbaa !342
  %.sroa.122.204.vec.extract = extractelement <4 x float> %642, i64 3, !dbg !352
  %773 = getelementptr inbounds float, ptr %769, i64 %669, !dbg !353
  store float %.sroa.122.204.vec.extract, ptr %773, align 4, !dbg !354, !tbaa !342
  %.sroa.132.208.vec.extract = extractelement <4 x float> %644, i64 0, !dbg !339
  %774 = getelementptr inbounds float, ptr %769, i64 %671, !dbg !340
  store float %.sroa.132.208.vec.extract, ptr %774, align 4, !dbg !341, !tbaa !342
  %.sroa.132.212.vec.extract = extractelement <4 x float> %644, i64 1, !dbg !346
  %775 = getelementptr inbounds float, ptr %769, i64 %673, !dbg !347
  store float %.sroa.132.212.vec.extract, ptr %775, align 4, !dbg !348, !tbaa !342
  %.sroa.132.216.vec.extract = extractelement <4 x float> %644, i64 2, !dbg !349
  %776 = getelementptr inbounds float, ptr %769, i64 %675, !dbg !350
  store float %.sroa.132.216.vec.extract, ptr %776, align 4, !dbg !351, !tbaa !342
  %.sroa.132.220.vec.extract = extractelement <4 x float> %644, i64 3, !dbg !352
  %777 = getelementptr inbounds float, ptr %769, i64 %677, !dbg !353
  store float %.sroa.132.220.vec.extract, ptr %777, align 4, !dbg !354, !tbaa !342
  %.sroa.142.224.vec.extract = extractelement <4 x float> %646, i64 0, !dbg !339
  %778 = getelementptr inbounds float, ptr %769, i64 %687, !dbg !340
  store float %.sroa.142.224.vec.extract, ptr %778, align 4, !dbg !341, !tbaa !342
  %.sroa.142.228.vec.extract = extractelement <4 x float> %646, i64 1, !dbg !346
  %779 = getelementptr inbounds float, ptr %769, i64 %689, !dbg !347
  store float %.sroa.142.228.vec.extract, ptr %779, align 4, !dbg !348, !tbaa !342
  %.sroa.142.232.vec.extract = extractelement <4 x float> %646, i64 2, !dbg !349
  %780 = getelementptr inbounds float, ptr %769, i64 %691, !dbg !350
  store float %.sroa.142.232.vec.extract, ptr %780, align 4, !dbg !351, !tbaa !342
  %.sroa.142.236.vec.extract = extractelement <4 x float> %646, i64 3, !dbg !352
  %781 = getelementptr inbounds float, ptr %769, i64 %693, !dbg !353
  store float %.sroa.142.236.vec.extract, ptr %781, align 4, !dbg !354, !tbaa !342
  %.sroa.152.240.vec.extract = extractelement <4 x float> %648, i64 0, !dbg !339
  %782 = getelementptr inbounds float, ptr %769, i64 %695, !dbg !340
  store float %.sroa.152.240.vec.extract, ptr %782, align 4, !dbg !341, !tbaa !342
  %.sroa.152.244.vec.extract = extractelement <4 x float> %648, i64 1, !dbg !346
  %783 = getelementptr inbounds float, ptr %769, i64 %697, !dbg !347
  store float %.sroa.152.244.vec.extract, ptr %783, align 4, !dbg !348, !tbaa !342
  %.sroa.152.248.vec.extract = extractelement <4 x float> %648, i64 2, !dbg !349
  %784 = getelementptr inbounds float, ptr %769, i64 %699, !dbg !350
  store float %.sroa.152.248.vec.extract, ptr %784, align 4, !dbg !351, !tbaa !342
  %.sroa.152.252.vec.extract = extractelement <4 x float> %648, i64 3, !dbg !352
  %785 = getelementptr inbounds float, ptr %769, i64 %701, !dbg !353
  store float %.sroa.152.252.vec.extract, ptr %785, align 4, !dbg !354, !tbaa !342
  ret void, !dbg !355
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
declare noundef i32 @llvm.amdgcn.workitem.id.x() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #5

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,512" "amdgpu-waves-per-eu"="2" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(read) }
attributes #4 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!opencl.ocl.version = !{!7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "AMD clang version 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "256_128_64_32_no_swizzle.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_no_swizzle")
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
!33 = distinct !DILocation(line: 56, column: 21, scope: !9)
!34 = !DILocation(line: 270, column: 116, scope: !35, inlinedAt: !36)
!35 = distinct !DISubprogram(name: "__hip_get_block_idx_x", scope: !30, file: !30, line: 270, type: !10, scopeLine: 270, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!36 = distinct !DILocation(line: 300, column: 160, scope: !37, inlinedAt: !38)
!37 = distinct !DISubprogram(name: "__get_x", scope: !30, file: !30, line: 300, type: !10, scopeLine: 300, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!38 = distinct !DILocation(line: 57, column: 21, scope: !9)
!39 = !DILocation(line: 265, column: 117, scope: !40, inlinedAt: !41)
!40 = distinct !DISubprogram(name: "__hip_get_thread_idx_x", scope: !30, file: !30, line: 265, type: !10, scopeLine: 265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!41 = distinct !DILocation(line: 291, column: 160, scope: !42, inlinedAt: !43)
!42 = distinct !DISubprogram(name: "__get_x", scope: !30, file: !30, line: 291, type: !10, scopeLine: 291, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!43 = distinct !DILocation(line: 57, column: 85, scope: !44, inlinedAt: !45)
!44 = distinct !DISubprogram(name: "warpid", scope: !14, file: !14, line: 57, type: !10, scopeLine: 57, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!45 = distinct !DILocation(line: 59, column: 25, scope: !9)
!46 = !DILocation(line: 57, column: 97, scope: !44, inlinedAt: !45)
!47 = !DILocation(line: 60, column: 34, scope: !9)
!48 = !DILocation(line: 61, column: 34, scope: !9)
!49 = !DILocation(line: 30, column: 28, scope: !50, inlinedAt: !52)
!50 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !51, file: !51, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!52 = distinct !DILocation(line: 12, column: 5, scope: !53, inlinedAt: !55)
!53 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> > >", scope: !54, file: !54, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!54 = !DIFile(filename: "ThunderKittens-HIP/include/ops/group/memory/tile/global_to_shared.cuh", directory: "/workdir/AMD-benchmarking-harness")
!55 = distinct !DILocation(line: 66, column: 5, scope: !9)
!56 = !DILocation(line: 60, column: 34, scope: !57, inlinedAt: !59)
!57 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !58, file: !58, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!58 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!59 = distinct !DILocation(line: 37, column: 39, scope: !50, inlinedAt: !52)
!60 = !DILocation(line: 63, column: 54, scope: !61, inlinedAt: !63)
!61 = distinct !DISubprogram(name: "operator[]", scope: !62, file: !62, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!62 = !DIFile(filename: "ThunderKittens-HIP/include/types/global/gl.cuh", directory: "/workdir/AMD-benchmarking-harness")
!63 = distinct !DILocation(line: 38, column: 57, scope: !50, inlinedAt: !52)
!64 = !DILocation(line: 63, column: 60, scope: !61, inlinedAt: !63)
!65 = !DILocation(line: 63, column: 16, scope: !61, inlinedAt: !63)
!66 = !DILocation(line: 40, column: 24, scope: !50, inlinedAt: !52)
!67 = !DILocation(line: 54, column: 32, scope: !50, inlinedAt: !52)
!68 = !DILocation(line: 58, column: 75, scope: !50, inlinedAt: !52)
!69 = !DILocation(line: 58, column: 88, scope: !50, inlinedAt: !52)
!70 = !DILocation(line: 58, column: 68, scope: !50, inlinedAt: !52)
!71 = !DILocation(line: 92, column: 5, scope: !72, inlinedAt: !74)
!72 = distinct !DISubprogram(name: "load_global_vec4_async", scope: !73, file: !73, line: 89, type: !10, scopeLine: 89, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!73 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/util/util.cuh", directory: "/workdir/AMD-benchmarking-harness")
!74 = distinct !DILocation(line: 58, column: 26, scope: !50, inlinedAt: !52)
!75 = !{i64 3065642}
!76 = !DILocation(line: 62, column: 9, scope: !50, inlinedAt: !52)
!77 = !{i64 3083825}
!78 = !DILocation(line: 103, column: 35, scope: !79, inlinedAt: !81)
!79 = distinct !DISubprogram(name: "idx", scope: !80, file: !80, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!80 = !DIFile(filename: "ThunderKittens-HIP/include/types/shared/st.cuh", directory: "/workdir/AMD-benchmarking-harness")
!81 = distinct !DILocation(line: 71, column: 34, scope: !50, inlinedAt: !52)
!82 = !DILocation(line: 104, column: 60, scope: !79, inlinedAt: !81)
!83 = !DILocation(line: 105, column: 22, scope: !79, inlinedAt: !81)
!84 = !DILocation(line: 71, column: 72, scope: !50, inlinedAt: !52)
!85 = !DILocation(line: 27, column: 5, scope: !86, inlinedAt: !87)
!86 = distinct !DISubprogram(name: "store_shared_vec", scope: !73, file: !73, line: 26, type: !10, scopeLine: 26, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!87 = distinct !DILocation(line: 71, column: 17, scope: !50, inlinedAt: !52)
!88 = !{i64 3063977}
!89 = !DILocation(line: 103, column: 35, scope: !79, inlinedAt: !90)
!90 = distinct !DILocation(line: 72, column: 34, scope: !50, inlinedAt: !52)
!91 = !DILocation(line: 104, column: 60, scope: !79, inlinedAt: !90)
!92 = !DILocation(line: 105, column: 22, scope: !79, inlinedAt: !90)
!93 = !DILocation(line: 72, column: 95, scope: !50, inlinedAt: !52)
!94 = !DILocation(line: 27, column: 5, scope: !86, inlinedAt: !95)
!95 = distinct !DILocation(line: 72, column: 17, scope: !50, inlinedAt: !52)
!96 = !DILocation(line: 103, column: 46, scope: !79, inlinedAt: !81)
!97 = !DILocation(line: 75, column: 9, scope: !50, inlinedAt: !52)
!98 = !{i64 3084366}
!99 = !DILocation(line: 30, column: 28, scope: !100, inlinedAt: !101)
!100 = distinct !DISubprogram(name: "load<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !51, file: !51, line: 27, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!101 = distinct !DILocation(line: 12, column: 5, scope: !102, inlinedAt: !103)
!102 = distinct !DISubprogram(name: "load<kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> > >", scope: !54, file: !54, line: 11, type: !10, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!103 = distinct !DILocation(line: 67, column: 5, scope: !9)
!104 = !DILocation(line: 60, column: 34, scope: !105, inlinedAt: !106)
!105 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !58, file: !58, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!106 = distinct !DILocation(line: 37, column: 39, scope: !100, inlinedAt: !101)
!107 = !DILocation(line: 63, column: 54, scope: !61, inlinedAt: !108)
!108 = distinct !DILocation(line: 38, column: 57, scope: !100, inlinedAt: !101)
!109 = !DILocation(line: 63, column: 60, scope: !61, inlinedAt: !108)
!110 = !DILocation(line: 63, column: 16, scope: !61, inlinedAt: !108)
!111 = !DILocation(line: 40, column: 24, scope: !100, inlinedAt: !101)
!112 = !DILocation(line: 58, column: 75, scope: !100, inlinedAt: !101)
!113 = !DILocation(line: 58, column: 88, scope: !100, inlinedAt: !101)
!114 = !DILocation(line: 58, column: 68, scope: !100, inlinedAt: !101)
!115 = !DILocation(line: 92, column: 5, scope: !72, inlinedAt: !116)
!116 = distinct !DILocation(line: 58, column: 26, scope: !100, inlinedAt: !101)
!117 = !DILocation(line: 54, column: 32, scope: !100, inlinedAt: !101)
!118 = !DILocation(line: 62, column: 9, scope: !100, inlinedAt: !101)
!119 = !DILocation(line: 103, column: 35, scope: !120, inlinedAt: !121)
!120 = distinct !DISubprogram(name: "idx", scope: !80, file: !80, line: 98, type: !10, scopeLine: 98, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!121 = distinct !DILocation(line: 71, column: 34, scope: !100, inlinedAt: !101)
!122 = !DILocation(line: 104, column: 60, scope: !120, inlinedAt: !121)
!123 = !DILocation(line: 105, column: 22, scope: !120, inlinedAt: !121)
!124 = !DILocation(line: 71, column: 72, scope: !100, inlinedAt: !101)
!125 = !DILocation(line: 27, column: 5, scope: !86, inlinedAt: !126)
!126 = distinct !DILocation(line: 71, column: 17, scope: !100, inlinedAt: !101)
!127 = !DILocation(line: 103, column: 35, scope: !120, inlinedAt: !128)
!128 = distinct !DILocation(line: 72, column: 34, scope: !100, inlinedAt: !101)
!129 = !DILocation(line: 104, column: 60, scope: !120, inlinedAt: !128)
!130 = !DILocation(line: 105, column: 22, scope: !120, inlinedAt: !128)
!131 = !DILocation(line: 72, column: 95, scope: !100, inlinedAt: !101)
!132 = !DILocation(line: 27, column: 5, scope: !86, inlinedAt: !133)
!133 = distinct !DILocation(line: 72, column: 17, scope: !100, inlinedAt: !101)
!134 = !DILocation(line: 103, column: 46, scope: !120, inlinedAt: !121)
!135 = !DILocation(line: 75, column: 9, scope: !100, inlinedAt: !101)
!136 = !DILocation(line: 68, column: 5, scope: !9)
!137 = !DILocation(line: 70, column: 18, scope: !9)
!138 = !DILocation(line: 70, column: 9, scope: !9)
!139 = !DILocation(line: 71, column: 9, scope: !9)
!140 = !DILocation(line: 72, column: 5, scope: !9)
!141 = !DILocation(line: 74, column: 5, scope: !9)
!142 = !DILocation(line: 133, column: 18, scope: !9)
!143 = !DILocation(line: 133, column: 9, scope: !9)
!144 = !DILocation(line: 75, column: 35, scope: !9)
!145 = !DILocation(line: 75, column: 39, scope: !9)
!146 = !DILocation(line: 77, column: 13, scope: !9)
!147 = !DILocation(line: 26, column: 23, scope: !148, inlinedAt: !150)
!148 = distinct !DISubprogram(name: "buffer_load_vec4", scope: !149, file: !149, line: 25, type: !10, scopeLine: 25, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!149 = !DIFile(filename: "./../utils.cpp", directory: "/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_no_swizzle")
!150 = distinct !DILocation(line: 77, column: 39, scope: !151, inlinedAt: !152)
!151 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 128, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 128, 64> >, 512>", scope: !149, file: !149, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!152 = distinct !DILocation(line: 78, column: 13, scope: !9)
!153 = !DILocation(line: 80, column: 9, scope: !9)
!154 = !DILocation(line: 81, column: 9, scope: !9)
!155 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !157)
!156 = distinct !DISubprogram(name: "load_shared_vec_async", scope: !149, file: !149, line: 127, type: !10, scopeLine: 127, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!157 = distinct !DILocation(line: 185, column: 37, scope: !158, inlinedAt: !159)
!158 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 256, 64>, 32, 32> >", scope: !149, file: !149, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!159 = distinct !DILocation(line: 83, column: 9, scope: !9)
!160 = !{i64 4294511}
!161 = !DILocation(line: 187, column: 47, scope: !158, inlinedAt: !159)
!162 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !165)
!163 = distinct !DISubprogram(name: "operator=", scope: !164, file: !164, line: 492, type: !10, scopeLine: 492, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!164 = !DIFile(filename: "/opt/rocm-6.4.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h", directory: "")
!165 = distinct !DILocation(line: 188, column: 45, scope: !158, inlinedAt: !159)
!166 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !167)
!167 = distinct !DILocation(line: 185, column: 37, scope: !158, inlinedAt: !168)
!168 = distinct !DILocation(line: 84, column: 9, scope: !9)
!169 = !DILocation(line: 187, column: 47, scope: !158, inlinedAt: !168)
!170 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !171)
!171 = distinct !DILocation(line: 188, column: 45, scope: !158, inlinedAt: !168)
!172 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !173)
!173 = distinct !DILocation(line: 185, column: 37, scope: !174, inlinedAt: !175)
!174 = distinct !DISubprogram(name: "load_async_shared_to_register<kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::st_subtile<kittens::st<__hip_bfloat16, 128, 64>, 32, 32> >", scope: !149, file: !149, line: 147, type: !10, scopeLine: 147, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!175 = distinct !DILocation(line: 85, column: 9, scope: !9)
!176 = !DILocation(line: 187, column: 47, scope: !174, inlinedAt: !175)
!177 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !178)
!178 = distinct !DILocation(line: 188, column: 45, scope: !174, inlinedAt: !175)
!179 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !180)
!180 = distinct !DILocation(line: 185, column: 37, scope: !174, inlinedAt: !181)
!181 = distinct !DILocation(line: 86, column: 9, scope: !9)
!182 = !DILocation(line: 187, column: 47, scope: !174, inlinedAt: !181)
!183 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !184)
!184 = distinct !DILocation(line: 188, column: 45, scope: !174, inlinedAt: !181)
!185 = !DILocation(line: 87, column: 9, scope: !9)
!186 = !DILocation(line: 89, column: 13, scope: !9)
!187 = !DILocation(line: 61, column: 18, scope: !105, inlinedAt: !188)
!188 = distinct !DILocation(line: 54, column: 39, scope: !189, inlinedAt: !190)
!189 = distinct !DISubprogram(name: "load_global_to_registers<2, false, kittens::st<__hip_bfloat16, 256, 64>, kittens::gl<__hip_bfloat16, -1, -1, -1, -1>, kittens::coord<kittens::st<__hip_bfloat16, 256, 64> >, 512>", scope: !149, file: !149, line: 41, type: !10, scopeLine: 44, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!190 = distinct !DILocation(line: 90, column: 13, scope: !9)
!191 = !DILocation(line: 63, column: 68, scope: !61, inlinedAt: !192)
!192 = distinct !DILocation(line: 55, column: 24, scope: !189, inlinedAt: !190)
!193 = !DILocation(line: 63, column: 16, scope: !61, inlinedAt: !192)
!194 = !DILocation(line: 32, column: 28, scope: !195, inlinedAt: !196)
!195 = distinct !DISubprogram(name: "make_srsrc", scope: !149, file: !149, line: 31, type: !10, scopeLine: 31, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!196 = distinct !DILocation(line: 60, column: 19, scope: !189, inlinedAt: !190)
!197 = !DILocation(line: 26, column: 23, scope: !148, inlinedAt: !198)
!198 = distinct !DILocation(line: 77, column: 39, scope: !189, inlinedAt: !190)
!199 = !DILocation(line: 92, column: 9, scope: !9)
!200 = !DILocation(line: 93, column: 9, scope: !9)
!201 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !202)
!202 = distinct !DILocation(line: 185, column: 37, scope: !158, inlinedAt: !203)
!203 = distinct !DILocation(line: 95, column: 9, scope: !9)
!204 = !DILocation(line: 187, column: 47, scope: !158, inlinedAt: !203)
!205 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !206)
!206 = distinct !DILocation(line: 188, column: 45, scope: !158, inlinedAt: !203)
!207 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !208)
!208 = distinct !DILocation(line: 185, column: 37, scope: !158, inlinedAt: !209)
!209 = distinct !DILocation(line: 96, column: 9, scope: !9)
!210 = !DILocation(line: 187, column: 47, scope: !158, inlinedAt: !209)
!211 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !212)
!212 = distinct !DILocation(line: 188, column: 45, scope: !158, inlinedAt: !209)
!213 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !214)
!214 = distinct !DILocation(line: 185, column: 37, scope: !174, inlinedAt: !215)
!215 = distinct !DILocation(line: 97, column: 9, scope: !9)
!216 = !DILocation(line: 187, column: 47, scope: !174, inlinedAt: !215)
!217 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !218)
!218 = distinct !DILocation(line: 188, column: 45, scope: !174, inlinedAt: !215)
!219 = !DILocation(line: 129, column: 5, scope: !156, inlinedAt: !220)
!220 = distinct !DILocation(line: 185, column: 37, scope: !174, inlinedAt: !221)
!221 = distinct !DILocation(line: 98, column: 9, scope: !9)
!222 = !DILocation(line: 187, column: 47, scope: !174, inlinedAt: !221)
!223 = !DILocation(line: 494, column: 7, scope: !163, inlinedAt: !224)
!224 = distinct !DILocation(line: 188, column: 45, scope: !174, inlinedAt: !221)
!225 = !DILocation(line: 99, column: 9, scope: !9)
!226 = !DILocation(line: 100, column: 9, scope: !9)
!227 = !DILocation(line: 102, column: 9, scope: !9)
!228 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !231)
!229 = distinct !DISubprogram(name: "mfma161616", scope: !230, file: !230, line: 25, type: !10, scopeLine: 28, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!230 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/register/tile/mma.cuh", directory: "/workdir/AMD-benchmarking-harness")
!231 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !233)
!232 = distinct !DISubprogram(name: "mma_ABt_base", scope: !230, file: !230, line: 77, type: !10, scopeLine: 80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!233 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !235)
!234 = distinct !DISubprogram(name: "mma_ABt<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<__hip_bfloat16, 32, 32, kittens::ducks::rt_layout::row>, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> >", scope: !230, file: !230, line: 197, type: !10, scopeLine: 200, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!235 = distinct !DILocation(line: 103, column: 9, scope: !9)
!236 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !237)
!237 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !238)
!238 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !235)
!239 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !240)
!240 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !241)
!241 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !242)
!242 = distinct !DILocation(line: 104, column: 9, scope: !9)
!243 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !244)
!244 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !245)
!245 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !242)
!246 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !247)
!247 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !248)
!248 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !249)
!249 = distinct !DILocation(line: 105, column: 9, scope: !9)
!250 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !251)
!251 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !252)
!252 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !249)
!253 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !254)
!254 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !255)
!255 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !256)
!256 = distinct !DILocation(line: 106, column: 9, scope: !9)
!257 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !258)
!258 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !259)
!259 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !256)
!260 = !DILocation(line: 107, column: 9, scope: !9)
!261 = !DILocation(line: 108, column: 9, scope: !9)
!262 = !DILocation(line: 109, column: 9, scope: !9)
!263 = !DILocation(line: 112, column: 13, scope: !9)
!264 = !DILocation(line: 113, column: 13, scope: !9)
!265 = !{i64 4303716}
!266 = !DILocation(line: 17, column: 5, scope: !267, inlinedAt: !268)
!267 = distinct !DISubprogram(name: "store_shared_vec_new", scope: !149, file: !149, line: 16, type: !10, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!268 = distinct !DILocation(line: 119, column: 17, scope: !269, inlinedAt: !270)
!269 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 128, 64>, 512>", scope: !149, file: !149, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!270 = distinct !DILocation(line: 114, column: 13, scope: !9)
!271 = !{i64 4290537}
!272 = !DILocation(line: 17, column: 5, scope: !267, inlinedAt: !273)
!273 = distinct !DILocation(line: 120, column: 17, scope: !269, inlinedAt: !270)
!274 = !DILocation(line: 116, column: 13, scope: !9)
!275 = !{i64 4303897}
!276 = !DILocation(line: 17, column: 5, scope: !267, inlinedAt: !277)
!277 = distinct !DILocation(line: 119, column: 17, scope: !278, inlinedAt: !279)
!278 = distinct !DISubprogram(name: "store_registers_to_shared<kittens::st<__hip_bfloat16, 256, 64>, 512>", scope: !149, file: !149, line: 88, type: !10, scopeLine: 90, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!279 = distinct !DILocation(line: 117, column: 13, scope: !9)
!280 = !DILocation(line: 17, column: 5, scope: !267, inlinedAt: !281)
!281 = distinct !DILocation(line: 120, column: 17, scope: !278, inlinedAt: !279)
!282 = !DILocation(line: 119, column: 9, scope: !9)
!283 = !DILocation(line: 120, column: 9, scope: !9)
!284 = !DILocation(line: 121, column: 9, scope: !9)
!285 = !DILocation(line: 123, column: 9, scope: !9)
!286 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !287)
!287 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !288)
!288 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !289)
!289 = distinct !DILocation(line: 124, column: 9, scope: !9)
!290 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !291)
!291 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !292)
!292 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !289)
!293 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !294)
!294 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !295)
!295 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !296)
!296 = distinct !DILocation(line: 125, column: 9, scope: !9)
!297 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !298)
!298 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !299)
!299 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !296)
!300 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !301)
!301 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !302)
!302 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !303)
!303 = distinct !DILocation(line: 126, column: 9, scope: !9)
!304 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !305)
!305 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !306)
!306 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !303)
!307 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !308)
!308 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !309)
!309 = distinct !DILocation(line: 216, column: 13, scope: !234, inlinedAt: !310)
!310 = distinct !DILocation(line: 127, column: 9, scope: !9)
!311 = !DILocation(line: 29, column: 27, scope: !229, inlinedAt: !312)
!312 = distinct !DILocation(line: 81, column: 5, scope: !232, inlinedAt: !313)
!313 = distinct !DILocation(line: 224, column: 17, scope: !234, inlinedAt: !310)
!314 = !DILocation(line: 128, column: 9, scope: !9)
!315 = !DILocation(line: 129, column: 9, scope: !9)
!316 = !DILocation(line: 130, column: 9, scope: !9)
!317 = !DILocation(line: 74, column: 29, scope: !9)
!318 = distinct !{!318, !141, !319, !320}
!319 = !DILocation(line: 131, column: 5, scope: !9)
!320 = !{!"llvm.loop.mustprogress"}
!321 = !DILocation(line: 134, column: 9, scope: !9)
!322 = !DILocation(line: 135, column: 5, scope: !9)
!323 = !DILocation(line: 144, column: 32, scope: !9)
!324 = !DILocation(line: 60, column: 34, scope: !325, inlinedAt: !326)
!325 = distinct !DISubprogram(name: "unit_coord<2, 3>", scope: !58, file: !58, line: 51, type: !10, scopeLine: 51, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!326 = distinct !DILocation(line: 192, column: 41, scope: !327, inlinedAt: !329)
!327 = distinct !DISubprogram(name: "store<2, kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !328, file: !328, line: 188, type: !10, scopeLine: 188, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!328 = !DIFile(filename: "ThunderKittens-HIP/include/ops/warp/memory/tile/global_to_register.cuh", directory: "/workdir/AMD-benchmarking-harness")
!329 = distinct !DILocation(line: 212, column: 5, scope: !330, inlinedAt: !331)
!330 = distinct !DISubprogram(name: "store<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col>, kittens::gl<float, -1, -1, -1, -1>, kittens::coord<kittens::rt<float, 32, 32, kittens::ducks::rt_layout::col> > >", scope: !328, file: !328, line: 211, type: !10, scopeLine: 211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!331 = distinct !DILocation(line: 141, column: 9, scope: !9)
!332 = !DILocation(line: 61, column: 18, scope: !325, inlinedAt: !326)
!333 = !DILocation(line: 63, column: 54, scope: !334, inlinedAt: !335)
!334 = distinct !DISubprogram(name: "operator[]", scope: !62, file: !62, line: 62, type: !10, scopeLine: 62, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!335 = distinct !DILocation(line: 192, column: 23, scope: !327, inlinedAt: !329)
!336 = !DILocation(line: 63, column: 60, scope: !334, inlinedAt: !335)
!337 = !DILocation(line: 63, column: 68, scope: !334, inlinedAt: !335)
!338 = !DILocation(line: 63, column: 16, scope: !334, inlinedAt: !335)
!339 = !DILocation(line: 203, column: 49, scope: !327, inlinedAt: !329)
!340 = !DILocation(line: 203, column: 13, scope: !327, inlinedAt: !329)
!341 = !DILocation(line: 203, column: 47, scope: !327, inlinedAt: !329)
!342 = !{!343, !343, i64 0}
!343 = !{!"float", !344, i64 0}
!344 = !{!"omnipotent char", !345, i64 0}
!345 = !{!"Simple C++ TBAA"}
!346 = !DILocation(line: 204, column: 49, scope: !327, inlinedAt: !329)
!347 = !DILocation(line: 204, column: 13, scope: !327, inlinedAt: !329)
!348 = !DILocation(line: 204, column: 47, scope: !327, inlinedAt: !329)
!349 = !DILocation(line: 205, column: 49, scope: !327, inlinedAt: !329)
!350 = !DILocation(line: 205, column: 13, scope: !327, inlinedAt: !329)
!351 = !DILocation(line: 205, column: 47, scope: !327, inlinedAt: !329)
!352 = !DILocation(line: 206, column: 49, scope: !327, inlinedAt: !329)
!353 = !DILocation(line: 206, column: 13, scope: !327, inlinedAt: !329)
!354 = !DILocation(line: 206, column: 47, scope: !327, inlinedAt: !329)
!355 = !DILocation(line: 147, column: 1, scope: !9)
