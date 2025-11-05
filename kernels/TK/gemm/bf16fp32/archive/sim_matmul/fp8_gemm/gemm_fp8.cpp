#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define HIP_CHECK(call)                                                        \
  {                                                                            \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at line "      \
                << __LINE__ << std::endl;                                      \
      exit(err);                                                               \
    }                                                                          \
  }
  
// SUBMIT CODE START

#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

typedef __hip_fp8_e4m3_fnuz fp8;
using vec4f = __attribute__((__vector_size__(4 * sizeof(float)))) float;

#define WARP_SIZE 64
#define N_WARP_Y_DEF 2
#define N_WARP_X_DEF 4
// Output Tile config
#define TILE_SIZE_Y_DEF 128
#define TILE_SIZE_X_DEF 128
#define TILE_K_DIM_DEF 128

constexpr int TILE_SIZE_Y = TILE_SIZE_Y_DEF;
constexpr int TILE_SIZE_X = TILE_SIZE_X_DEF;
constexpr int K_DIM = TILE_K_DIM_DEF;
constexpr int NBWARPS_Y = N_WARP_Y_DEF;
constexpr int NBWARPS_X = N_WARP_X_DEF;
// intrinsic config
constexpr int MATMUL_DIM_K = 32;
constexpr int MATMUL_DIM_NM = 16;
constexpr int BLOCK_DIM = NBWARPS_X * NBWARPS_Y * WARP_SIZE;

using vec4u = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec2u = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;


__device__ void matmul16x16(const __hip_fp8_e4m3_fnuz *a,
                            const __hip_fp8_e4m3_fnuz *b, int k, int a_col,
                            int b_col, vec4f &c, int m, int n) {
  int tid = threadIdx.x % WARP_SIZE;
  int row = k + tid % 8 + (tid / 16) * 8;
  int col = ((tid % 16) / 8);
  int a_col_s = (a_col / 8 + col);
  int b_col_s = (b_col / 8 + col);
  int col_a_swizzled = ((a_col_s / 2) ^ (row % 8)) * 2 + a_col_s % 2;
  int col_b_swizzled = ((b_col_s / 2) ^ (row % 8)) * 2 + b_col_s % 2;

  auto packedA = reinterpret_cast<const uint64_t *>(a + row * m)[col_a_swizzled];
  auto packedB = reinterpret_cast<const uint64_t *>(b + row * n)[col_b_swizzled];

  c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(packedA, packedB, c, 0, 0, 0);
}

template<bool boundCheck>
__device__ void loadLDS(int warpId, int laneId, int kId, fp8 *s_a, fp8 *s_b,
                        const fp8 *a, const fp8 *b, int M, int N) {
  
  constexpr int nbReadPerThread = 8;
  constexpr int nbWarpPerMatrix = 4;
  const fp8 *ptr = a;
  fp8 *dst = s_a;
  int warpIdLocal = warpId%nbWarpPerMatrix;
  int stride = M;
  
  uint32_t col_offset = TILE_SIZE_X * blockIdx.x;
  if (warpId>=nbWarpPerMatrix){
    ptr=b;
    col_offset = TILE_SIZE_Y * blockIdx.y;
    dst=s_b;
    stride= N;
  }
  bool b_check = true;
  if constexpr (boundCheck){
    b_check = (warpId>=nbWarpPerMatrix) || (col_offset < stride);
  }
  
  constexpr int DIMY = TILE_SIZE_Y / nbReadPerThread;
  int nbRowsPerWarp = WARP_SIZE/DIMY*nbReadPerThread;
  vec2u b_value[nbReadPerThread] = {0};
  if (b_check){
    // Read 8x8 tiles
    uint32_t row_offset = kId;
    for (int r = 0; r < nbReadPerThread; r++) {
      int col = laneId % DIMY;
      int row = r + nbRowsPerWarp * warpIdLocal + nbReadPerThread * (laneId / DIMY);
      b_value[r] = (reinterpret_cast<const vec2u *>(ptr + (row_offset + row) * stride + (col_offset + col * nbReadPerThread)))[0];
    }
  // Transpose tiles
    vec2u result_b[nbReadPerThread];
    for (int i = 0; i < nbReadPerThread; ++i)
      for (int j = 0; j < nbReadPerThread; ++j) {
        reinterpret_cast<uint8_t *>(&result_b[i])[j] =
            reinterpret_cast<uint8_t *>(&b_value[j])[i];
      }
// Write tiles to LDS
    for (int r = 0; r < nbReadPerThread; r++) {
      int col = laneId % DIMY;
      int row = r + nbRowsPerWarp * warpIdLocal + nbReadPerThread * (laneId / DIMY);
      int col_swizzled = 2 * ((col / 2) ^ (row % 8)) + col % 2;
      (reinterpret_cast<vec2u *>(dst))[(DIMY)*row + col_swizzled] = result_b[r];
    }
  }
}

__global__ void cast_kernel(__hip_bfloat16 *c_bf16, const float *c_fp32, int M, int N) {

    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid<M*N){
      c_bf16[tid] = (__hip_bfloat16)c_fp32[tid];
    }
}


template <int K_SPLIT,bool boundCheck, bool useAtomic>
__global__ void __launch_bounds__(N_WARP_X_DEF *N_WARP_Y_DEF *WARP_SIZE)
    custom_kernel(const fp8 *a, const fp8 *b, const float *as, const float *bs,
                  __hip_bfloat16 *c,float *c_atomic, int M, int N, int K) {
  int tid = threadIdx.x;
  int laneId = tid % WARP_SIZE;

  int splitK = K / K_SPLIT;
  int kIdStart = blockIdx.z * splitK;
  int kIdStop = splitK + blockIdx.z * splitK;

  int SN = (N + 128 - 1) / 128;

  int warpId = threadIdx.x / WARP_SIZE;
  int warpIdx = warpId / NBWARPS_Y; 
  int warpIdy = warpId % NBWARPS_Y;

  __shared__ fp8 s_a[K_DIM][TILE_SIZE_X];
  __shared__ fp8 s_b[K_DIM][TILE_SIZE_Y];

  constexpr int WARP_TILE_SIZE_Y = TILE_SIZE_Y / NBWARPS_Y / MATMUL_DIM_NM;
  constexpr int WARP_TILE_SIZE_X = TILE_SIZE_X / NBWARPS_X / MATMUL_DIM_NM;
  vec4f c_s[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y] = {0};

  for (int kId = kIdStart; kId < kIdStop; kId += K_DIM) {

    loadLDS<boundCheck>(warpId, laneId, kId, &s_a[0][0], &s_b[0][0], a, b, M, N);
    __syncthreads();

    vec4f c_tmp[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y] = {0};
#pragma unroll
    for (int t_wIdx = 0; t_wIdx < WARP_TILE_SIZE_X; t_wIdx++) {
      for (int t_wIdy = 0; t_wIdy < WARP_TILE_SIZE_Y; t_wIdy++) {
        constexpr int stepX = TILE_SIZE_X / WARP_TILE_SIZE_X;
        constexpr int stepY = TILE_SIZE_Y / WARP_TILE_SIZE_Y;

        vec4f c_tmp = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < K_DIM;k += MATMUL_DIM_K) // MATMUL_DIM_K must be a multiple of K_DIM
        {
          int a_col = (t_wIdx * stepX) + warpIdx * MATMUL_DIM_NM;
          int b_col = (t_wIdy * stepY) + warpIdy * MATMUL_DIM_NM;
          matmul16x16(&s_a[0][0], &s_b[0][0], k, a_col, b_col, c_tmp,
                           TILE_SIZE_X, TILE_SIZE_Y);
        }

        int a_row = (t_wIdx * stepX) + warpIdx * MATMUL_DIM_NM;
        // Apply scaling
        int as_col = kId / 128;
        int as_row = a_row + blockIdx.x * TILE_SIZE_X;
        int bs_col = blockIdx.y;
        int bs_row = kId / 128;
        float bs_val = bs[bs_row * SN + bs_col];
        for (int i = 0; i < 4; i++) {
          int row = as_row + i + (laneId / 16) * 4;
          float as_val = as[as_col * M + row];
          c_s[t_wIdx][t_wIdy][i] += c_tmp[i] * as_val * bs_val;
        }
      }
    }

    __syncthreads();
  }
  

  for (int t_wIdx = 0; t_wIdx < WARP_TILE_SIZE_X; t_wIdx++) {
    for (int t_wIdy = 0; t_wIdy < WARP_TILE_SIZE_Y; t_wIdy++) {
      // offset pointing to the accumulated 16x16 block
      int col_offset =
          blockIdx.y * TILE_SIZE_Y +                 // TILE offset
          (warpIdy * MATMUL_DIM_NM) +                // Warp offset
          t_wIdy * (TILE_SIZE_Y / WARP_TILE_SIZE_Y); // Warp tile offset

      int row_offset =
          blockIdx.x * TILE_SIZE_X +                 // TILE offset
          (warpIdx * MATMUL_DIM_NM) +                // Warp offset
          t_wIdx * (TILE_SIZE_X / WARP_TILE_SIZE_X); // Warp tile offset

      auto res = c_s[t_wIdx][t_wIdy];

      bool b_check = true;
      if constexpr (boundCheck){
        b_check = (row_offset) < M && (col_offset) < N;
      }
      if (b_check) {
          for (int i = 0; i < 4; i++)
          {
              int col = laneId % 16;
              int row = i + (laneId / 16) * 4;
              if constexpr (useAtomic){
                atomicAdd( &c_atomic[(row_offset + row) * N + (col_offset + col)],res[i]);
              }else
               {
                c[(row_offset + row) * N + (col_offset + col)] = (__hip_bfloat16)(res[i]);
              }
          }
        
      }
    }
  }
}


struct LaunchConfig {
  int K_SPLIT;
  bool useAtomic;
  bool boundCheck;
};

template <int K_SPLIT, bool boundCheck, bool useAtomic>
inline void launch_kernel_variant(const dim3& blocks, const dim3& threads,
                                  const fp8 *d_A, const fp8 *d_B,
                                  const float *d_As, const float *d_Bs, __hip_bfloat16 *d_C,float*c_atomic,
                                  int M, int N, int K) {

  dim3 threadsCast(256);
  dim3 blockCast((M*N + (N*M-1))/256);

  hipLaunchKernelGGL((custom_kernel<K_SPLIT, boundCheck, useAtomic>), blocks, threads, 0, 0,
                     d_A, d_B, d_As, d_Bs, d_C, c_atomic,M, N, K);
  if constexpr (useAtomic){
    hipLaunchKernelGGL(cast_kernel, blockCast,threadsCast,0,0,d_C,c_atomic,M,N);
  }
}


template <int K_SPLIT>
inline void dispatch_bound_atomic(bool boundCheck, bool useAtomic,
                                  const dim3& blocks, const dim3& threads,
                                  const fp8 *d_A, const fp8 *d_B,
                                  const float *d_As, const float *d_Bs, __hip_bfloat16 *d_C,float*c_atomic,
                                  int M, int N, int K) {
  if (boundCheck) {
    if (useAtomic){
      HIP_CHECK(hipMemsetAsync(c_atomic, 0, N*M * sizeof(float)));
      launch_kernel_variant<K_SPLIT, true, true>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
    } 
    else{
      launch_kernel_variant<K_SPLIT, true, false>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
    }
  } else {
    if (useAtomic){
      HIP_CHECK(hipMemsetAsync(c_atomic, 0, N*M * sizeof(float)));
      launch_kernel_variant<K_SPLIT, false, true>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
    }
    else{
      launch_kernel_variant<K_SPLIT, false, false>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
    }
  }
}

void launch_kernel(const LaunchConfig &config, const fp8 *d_A, const fp8 *d_B,
                   const float *d_As, const float *d_Bs, __hip_bfloat16 *d_C,float*c_atomic,
                   int M, int N, int K) {

  dim3 threadsPerBlock(N_WARP_X_DEF * N_WARP_Y_DEF * WARP_SIZE);
  dim3 blocksPerGrid((M + TILE_SIZE_X_DEF - 1) / TILE_SIZE_X_DEF,
                     (N + TILE_SIZE_Y_DEF - 1) / TILE_SIZE_Y_DEF,
                     config.K_SPLIT);

  switch (config.K_SPLIT) {
    case 1: dispatch_bound_atomic<1>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
    case 2: dispatch_bound_atomic<2>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
    case 4: dispatch_bound_atomic<4>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
    case 8: dispatch_bound_atomic<8>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
  
    default:
      throw std::invalid_argument("Unsupported K_SPLIT value");
  }
}


void launch_kernel(const fp8 *d_A, const fp8 *d_B, const float *d_As,
                   const float *d_Bs, __hip_bfloat16 *d_C, float*c_atomic,int M, int N,
                   int K) {
  LaunchConfig config;
  config.useAtomic=false;
  config.K_SPLIT=1;
  if (M % 128 ==0 && N %128 ==0)
    config.boundCheck = false;
  else
    config.boundCheck = true;

  if ((M == 1024 &&  N==512 && K ==7168 ) ||
      (M == 1024 &&  N==576 && K ==7168 )){
        config.useAtomic = true;
        config.K_SPLIT = 8;
      
      }else if (M == 1024 &&  N==1536 && K ==7168 ){
        config.useAtomic = true;
        config.K_SPLIT = 4;
      }
      else if (M == 6144 &&  N==512 && K ==7168 ){
        config.useAtomic = true;
        config.K_SPLIT = 2;
      }
  launch_kernel(config, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
}


/// SUBMIT CODE- END

//Reference kernel taken from template
constexpr const int BLOCK = 128;
__global__ void ref_kernel(const __hip_fp8_e4m3_fnuz *a,
                           const __hip_fp8_e4m3_fnuz *b, const float *as,
                           const float *bs, __hip_bfloat16 *c, int m, int n,
                           int k) {

  // Your implementation here
  int cx = threadIdx.x + blockDim.x * blockIdx.x;
  int cy = threadIdx.y + blockDim.y * blockIdx.y;
  if (cx >= m || cy >= n)
    return;

  int sn = (n + BLOCK - 1) / BLOCK;

  float result = 0;
  // split loop into an outer loop over different blocks, and an inner loop
  // within one block. we can assume k % BLOCK == 0.
  for (int i = 0; i < k; i += BLOCK) {
    // block results accumulates the inner product across a single block.
    // within each block, scales are constant, so we can lift the scaling
    // outside of the inner loop.
    float block_result = 0;
    for (int ii = 0; ii < BLOCK; ++ii) {
      // load input matrix elements and convert to float for computations
      float av = (float)a[cx + (i + ii) * m];
      float bv = (float)b[cy + (i + ii) * n];
      block_result += av * bv;
    }

    // before we can go to the next block, scale the result of the current block
    // and accumulate to final result
    // note the different indexing into as and bs

    result +=
        block_result * as[cx + i / BLOCK * m] * bs[cy / BLOCK + i / BLOCK * sn];

  }

  // finally, write the result as bf16
  c[cx * n + cy] = (__hip_bfloat16)result;
}
/// SUBMIT CODE- END


std::vector<float> initRandomVec(int size, int seed = 42) {
  std::vector<float> v(size);
  std::random_device rd;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < size; ++i) {
    v[i] = dist(gen);
  }
  return v;
}

std::vector<__hip_fp8_e4m3_fnuz> convertToFp8(const std::vector<float> &vec) {
  std::vector<__hip_fp8_e4m3_fnuz> fp8Vec(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    fp8Vec[i] = __hip_fp8_e4m3_fnuz(vec[i]);
  }
  return fp8Vec;
}

template <typename T> T *toGPU(const std::vector<T> &h_p) {
  auto size = h_p.size() * sizeof(T);
  T *d_p;
  HIP_CHECK(hipMalloc(&d_p, size));
  HIP_CHECK(hipMemcpy(d_p, h_p.data(), size, hipMemcpyHostToDevice));
  return d_p;
}

template <typename T> std::vector<T> toCPU(const T *d_p, size_t count) {
  std::vector<T> h_p(count);
  HIP_CHECK(
      hipMemcpy(h_p.data(), d_p, count * sizeof(T), hipMemcpyDeviceToHost));
  return h_p;
}

//Port of Python comparison function
template<typename T>
std::tuple<bool, std::vector<std::string>> verbose_allclose(
  const std::vector<T>& received,
  const std::vector<T>& expected,
  float rtol = 1e-5f,
  float atol = 1e-8f,
  int max_print = 5
) {
  if (received.size() != expected.size()) {
      return {false, {"SIZE MISMATCH"}};
  }

  std::vector<std::string> mismatch_details;
  int num_mismatched = 0;
  float max_error = 0.0f;

  for (size_t i = 0; i < received.size(); ++i) {
      float r = float(received[i]);
      float e = float(expected[i]);

      bool is_nan_mismatch = std::isnan(r) != std::isnan(e);
      bool is_posinf_mismatch = (std::isinf(r) && r > 0) != (std::isinf(e) && e > 0);
      bool is_neginf_mismatch = (std::isinf(r) && r < 0) != (std::isinf(e) && e < 0);

      float abs_diff = std::abs(r - e);
      float tolerance = atol + rtol * std::abs(e);

      bool is_tol_mismatch = !std::isnan(r) && !std::isnan(e) && abs_diff > tolerance;

      if (is_nan_mismatch || is_posinf_mismatch || is_neginf_mismatch || is_tol_mismatch) {
          ++num_mismatched;
          if (mismatch_details.size() < static_cast<size_t>(max_print)) {
              std::ostringstream oss;
              oss << "ERROR at [" << i << "]: " << r << " vs " << e;
              mismatch_details.push_back(oss.str());
          }
      }

      if (!std::isnan(abs_diff) && abs_diff > max_error) {
          max_error = abs_diff;
      }
  }

  if (num_mismatched > 0) {
      mismatch_details.insert(mismatch_details.begin(),
                              "Number of mismatched elements: " + std::to_string(num_mismatched));
      if (num_mismatched > max_print) {
          mismatch_details.push_back("... and " + std::to_string(num_mismatched - max_print) + " more mismatched elements.");
      }
      return {false, mismatch_details};
  }

  return {true, {"Maximum error: " + std::to_string(max_error)}};
}


struct TestCase {
  int M;
  int N;
  int K;
};

void printTestCase(const TestCase &test, int idx) {
  std::cout << "\n####### TEST " << idx << " | M: " << test.M
            << " | N: " << test.N << " | K: " << test.K << " | #######"
            << std::endl;
}

int main() {
  std::cout << "FP8 GEMM test" << std::endl;
  const TestCase tests[] = {
    // {1024,	1536,	7168},
    // {1024,	3072,	1536},
    // {1024,	576	,7168},
    // {1024,	7168,	256},
    // {1024,	7168,	2048},
    // {1024,	4608,	7168},
    // {1024,	7168,	2304},
    // {1024,	512	,7168},
    // {1024,	4096,	512},
    // {6144,	1536,	7168},
    // {6144,	3072,	1536},
    // {6144,	576	,7168},
    // {6144,	7168,	256},
    // {6144,	7168,	2048},
    // {6144,	4608,	7168},
    // {6144,	7168,	2304},
    // {6144,	512	,7168},
    // {6144,	4096,	512}
    {8192,	8192,	8192}
};
  constexpr int nbRun = 16;
  hipEvent_t start[nbRun + 1], stop[nbRun + 1];

  for (int i = 0; i < nbRun + 1; i++) {
    HIP_CHECK(hipEventCreate(&start[i]));
    HIP_CHECK(hipEventCreate(&stop[i]));
  }

  const int testsCount = sizeof(tests) / sizeof(TestCase);
  for (int i = 0; i < testsCount; i++) {
    const TestCase &test = tests[i];
    printTestCase(tests[i], i);

    const int M = test.M, N = test.N, K = test.K;

    // Host buffers
    auto h_A = initRandomVec(M * K);
    auto h_B = initRandomVec(N * K);

    auto h_A_fp8 = convertToFp8(h_A);
    auto h_B_fp8 = convertToFp8(h_B);

    auto h_As = initRandomVec(K * M / 128); 
    auto h_Bs = initRandomVec(K * N / 128 / 128);

    // Device buffers
    auto d_A = toGPU(h_A_fp8);
    auto d_B = toGPU(h_B_fp8);
    auto d_As = toGPU(h_As);
    auto d_Bs = toGPU(h_Bs);
    auto h_C_init = std::vector<__hip_bfloat16>(M * N, 0.0);
    auto h_C_atomic = std::vector<float>(M * N, 0.0);
    auto d_C = toGPU(h_C_init);
    auto d_C_ref = toGPU(std::vector<__hip_bfloat16>(M * N, 2.0));
    auto d_C_atomic = toGPU(h_C_atomic);
    
    float milliseconds = 0;
    for (int i = 0; i < nbRun; i++) {
      HIP_CHECK(hipFree(d_A));
      HIP_CHECK(hipFree(d_B));
      HIP_CHECK(hipFree(d_As));
      HIP_CHECK(hipFree(d_Bs));

      d_A = toGPU(h_A_fp8);
      d_B = toGPU(h_B_fp8);
      d_As = toGPU(h_As);
      d_Bs = toGPU(h_Bs);

      HIP_CHECK(hipEventRecord(start[i], 0));
      launch_kernel(d_A, d_B, d_As, d_Bs, d_C,d_C_atomic, M, N, K);
      HIP_CHECK(hipEventRecord(stop[i], 0));
      HIP_CHECK(hipEventSynchronize(stop[i]));
    }

    double flops = 2.0 * M * N * K;
    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
    for (int i = 0; i < nbRun; i++) {
      float milliseconds = 0;
      HIP_CHECK(hipEventElapsedTime(&milliseconds, start[i], stop[i]));
      std::cout << "Time : " << milliseconds * 1000.0 << " us" << std::endl;
      std::cout << "FLOPs: " << flops << std::endl;
      std::cout << "TFLOPS: " << flops / (milliseconds) / 1e9 << std::endl;
    }

    // execute reference kernel
    dim3 threadsPerBlock_ref(16, 16);
    dim3 blocksPerGrid_ref((M + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(ref_kernel, blocksPerGrid_ref, threadsPerBlock_ref, 0, 0,
                       d_A, d_B, d_As, d_Bs, d_C_ref, M, N, K);
    HIP_CHECK(hipEventRecord(stop[nbRun], 0));
    HIP_CHECK(hipEventSynchronize(stop[nbRun]));


    // Copy result back to host
    auto h_C = toCPU(d_C, M * N);
    auto h_C_ref = toCPU(d_C_ref, M * N);

    auto [ok, details] = verbose_allclose(h_C,h_C_ref,2e-2, 1e-3);
    if (!ok) {
        for (const auto& msg : details) {
            std::cout << msg << std::endl;
        }
    } else {
        std::cout << details[0] << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_As));
    HIP_CHECK(hipFree(d_Bs));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_C_ref));

    std::cout << "\n#####################" << std::endl;
  }

  return 0;
}