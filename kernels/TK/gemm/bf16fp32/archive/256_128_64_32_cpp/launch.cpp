#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>

#include <hip_bf16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

void rocblas_gemm(__hip_bfloat16* A, __hip_bfloat16* B, __hip_bfloat16* C, int M_size, int N_size, int K_size) {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // We want: C = A * B^T where:
    // A is M×K stored row-major as A[i*K + j] 
    // B is N×K stored row-major as B[i*K + j]
    // C is M×N stored row-major as C[i*N + j]
    // Result: C[i][j] = sum_k A[i][k] * B[j][k]
    
    // The key insight for row-major to column-major conversion:
    // Row-major C = A * B^T becomes column-major C^T = B * A^T
    // Since rocBLAS expects column-major, we compute: C^T = B * A^T
    // This means we swap the order of operands and transpose the result
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Use rocblas_gemm_ex to compute C^T = B * A^T 
    // where B is transposed to match A*B^T semantics
    rocblas_gemm_ex(handle,
                    rocblas_operation_transpose,  // op(B) = T (transpose B to get B^T)
                    rocblas_operation_none,       // op(A) = N (A is not transposed)
                    N_size, M_size, K_size,       // N, M, K (dimensions for C^T)
                    &alpha,                       // alpha
                    B, rocblas_datatype_bf16_r, K_size,  // B matrix with leading dim K
                    A, rocblas_datatype_bf16_r, K_size,  // A matrix with leading dim K  
                    &beta,                        // beta
                    C, rocblas_datatype_bf16_r, N_size,  // C matrix with leading dim N
                    C, rocblas_datatype_bf16_r, N_size,  // C matrix with leading dim N
                    rocblas_datatype_f32_r,       // compute type (fp32 accumulation)
                    rocblas_gemm_algo_standard,   // algorithm
                    0, 0);                        // solution index, flags
    
    rocblas_destroy_handle(handle);
}

void cpu_gemm(float* a, float* b, float* c, int M_size_size, int N_size_size, int K_size_size) {
    #pragma omp parallel for collapse(2) // otherwise the CPU version takes for everrrrrr
    for (int i = 0; i < M_size_size; i++) {
        for (int j = 0; j < N_size_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K_size_size; k++) {
                // sum += a[i * K_size_size + k] * b[k * N_size_size + j]; // mma_AB (standard)
                sum += a[i * K_size_size + k] * b[j * K_size_size + k]; // mma_ABt (A * B^T)
            }
            c[i * N_size_size + j] = sum;
        }
    }
}

int run_benchmark(size_t M_size, size_t N_size, size_t K_size) {
    hipError_t hipStatus;
    std::cout << "--------------------  M_size=" << M_size << " N_size=" << N_size << " K_size=" << K_size << "  --------------------\n";

    // Allocate host memory for A*B^T
    float *h_A = new float[M_size * K_size];     // A: M x K
    float *h_B = new float[N_size * K_size];     // B: N x K (for B^T operation)
    float *h_C = new float[M_size * N_size];
    float *h_C_ref = new float[M_size * N_size];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M_size * K_size; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < N_size * K_size; ++i) h_B[i] = dis(gen);  // B: N x K
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if(true) cpu_gemm(h_A, h_B, h_C_ref, M_size, N_size, K_size);
    std::cout << "Performed CPU matrix multiplication" << std::endl;

    // Allocate device memory for A*B^T
    __hip_bfloat16 *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M_size*K_size*sizeof(__hip_bfloat16));   // A: M x K
    hipMalloc(&d_B, N_size*K_size*sizeof(__hip_bfloat16));   // B: N x K (for B^T)
    hipMalloc(&d_C, M_size*N_size*sizeof(__hip_bfloat16));
    // Check for HIP errors
    hipStatus = hipGetLastError();
    if (hipStatus != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(hipStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __hip_bfloat16 and copy to device
    __hip_bfloat16 *h_A_bf16 = new __hip_bfloat16[M_size * K_size];
    __hip_bfloat16 *h_B_bf16 = new __hip_bfloat16[N_size * K_size];  // B: N x K
    for (int i = 0; i < M_size * K_size; ++i) h_A_bf16[i] = __hip_bfloat16(h_A[i]);
    for (int i = 0; i < N_size * K_size; ++i) h_B_bf16[i] = __hip_bfloat16(h_B[i]);
    hipMemcpy(d_A, h_A_bf16, M_size*K_size*2, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B_bf16, N_size*K_size*2, hipMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;
    printf("\n");

    // Allocate result arrays for comparison
    __hip_bfloat16 *d_C_hipblas;
    hipMalloc(&d_C_hipblas, M_size*N_size*sizeof(__hip_bfloat16));

    // ========== ThunderKittens Kernel Timing ==========
    std::cout << "\n=== ThunderKittens Kernel ===" << std::endl;
    for(int i = 0; i < 20; i++) { // warmup
        matmul(d_A, d_B, d_C);
    }
    // Start timing
    constexpr int ITERS = 20;
    hipDeviceSynchronize();
    auto start_tk = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERS; i++) {
        matmul(d_A, d_B, d_C);
    }
    auto end_tk = std::chrono::high_resolution_clock::now();
    hipDeviceSynchronize();

    // Calculate duration
    std::chrono::duration<double> diff_tk = end_tk - start_tk;
    double useconds_tk = diff_tk.count() * 1e6 / ITERS;

    // Calculate TFLOPs
    double flops = double(2.0) * M_size * N_size * K_size; // 2 FLOPs per multiply-add
    double tflops_tk = (flops / useconds_tk) / 1e6;
    std::cout << "ThunderKittens execution time: " << useconds_tk << " us" << std::endl;
    std::cout << "ThunderKittens performance: " << tflops_tk << " TFLOPs" << std::endl;

    // ========== rocBLAS Timing ==========
    std::cout << "\n=== rocBLAS Baseline ===" << std::endl;
    for(int i = 0; i < 20; i++) { // warmup
        rocblas_gemm(d_A, d_B, d_C_hipblas, M_size, N_size, K_size);
    }
    hipDeviceSynchronize();
    auto start_blas = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < ITERS; i++) {
        rocblas_gemm(d_A, d_B, d_C_hipblas, M_size, N_size, K_size);
    }
    auto end_blas = std::chrono::high_resolution_clock::now();
    hipDeviceSynchronize();
    std::chrono::duration<double> diff_blas = end_blas - start_blas;
    double useconds_blas = diff_blas.count() * 1e6 / ITERS;
    double tflops_blas = (flops / useconds_blas) / 1e6;
    std::cout << "rocBLAS execution time: " << useconds_blas << " us" << std::endl;
    std::cout << "rocBLAS performance: " << tflops_blas << " TFLOPs" << std::endl;

    // ========== Performance Comparison ==========
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "ThunderKittens: " << tflops_tk << " TFLOPs" << std::endl;
    std::cout << "rocBLAS:        " << tflops_blas << " TFLOPs" << std::endl;
    std::cout << "Speedup:        " << (tflops_tk / tflops_blas) << "x" << std::endl;
    
    // Check for HIP errors
    hipStatus = hipGetLastError();
    if (hipStatus != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(hipStatus) << std::endl;
        // Optionally, you might want to exit the program or handle the error in some way
        return -1;
    }

    // Copy results back to host
    __hip_bfloat16 *h_C_tk = new __hip_bfloat16[M_size * N_size];     // ThunderKittens result
    __hip_bfloat16 *h_C_blas = new __hip_bfloat16[M_size * N_size];   // hipBLASLt result
    
    hipMemcpy(h_C_tk, d_C, M_size*N_size*2, hipMemcpyDeviceToHost);
    hipMemcpy(h_C_blas, d_C_hipblas, M_size*N_size*2, hipMemcpyDeviceToHost);
    std::cout << "Copied results back to host" << std::endl;

    // Convert results to float for comparison
    float *h_C_tk_f = new float[M_size * N_size];
    float *h_C_blas_f = new float[M_size * N_size];
    
    for (int i = 0; i < M_size * N_size; ++i) {
        h_C_tk_f[i] = float(h_C_tk[i]);
        h_C_blas_f[i] = float(h_C_blas[i]);
    }
    std::cout << "Converted results to float" << std::endl;

    // ========== Correctness Comparison: ThunderKittens vs rocBLAS ==========
    std::cout << "\n=== Correctness Analysis: ThunderKittens vs rocBLAS ===" << std::endl;
    float max_error = 0.0f;
    float sum_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < M_size * N_size; ++i) {
        float error = std::abs(h_C_tk_f[i] - h_C_blas_f[i]);
        sum_error += error;
        
        if (error > 0.01f) { // Much tighter tolerance since both use bf16/fp32
            if(error_count < 20) {
                std::cout << "Difference at [" << i / N_size << "," << i % N_size << "]: " 
                         << "TK=" << h_C_tk_f[i] << " vs BLAS=" << h_C_blas_f[i] 
                         << " (diff=" << error << ")" << std::endl;
            } else if(error_count == 20) {
                std::cout << "... (showing only first 20 differences)" << std::endl;
            }
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    float mean_error = sum_error / (M_size * N_size);
    std::cout << "\nCorrectness Summary:" << std::endl;
    std::cout << "Max error:     " << max_error << std::endl;
    std::cout << "Mean error:    " << mean_error << std::endl;
    std::cout << "Large errors:  " << error_count << " / " << (M_size * N_size) << std::endl;
    std::cout << "Accuracy:      " << (100.0 * (1.0 - float(error_count) / (M_size * N_size))) << "%" << std::endl;

    // Also compare against CPU reference for completeness
    std::cout << "\n=== CPU Reference Comparison ===" << std::endl;
    int cpu_errors_tk = 0, cpu_errors_blas = 0;
    for (int i = 0; i < M_size * N_size; ++i) {
        if (std::abs(h_C_tk_f[i] - h_C_ref[i]) > 0.1f) cpu_errors_tk++;
        if (std::abs(h_C_blas_f[i] - h_C_ref[i]) > 0.1f) cpu_errors_blas++;
    }
    std::cout << "TK vs CPU errors:     " << cpu_errors_tk << std::endl;
    std::cout << "BLAS vs CPU errors:   " << cpu_errors_blas << std::endl;

    // Cleanup
    delete[] h_C_tk_f;
    delete[] h_C_blas_f;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_tk;
    delete[] h_C_blas;
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_C_hipblas);

    return 0;
}

int main() {
    int N_size;
    N_size = 8192;  // Required size
    run_benchmark(N_size, N_size, N_size);
    return 0;
}