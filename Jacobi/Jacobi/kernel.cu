
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

using REAL = float;

// ------------------------------------------------------------
// CUDA Jacobi Kernel
// ------------------------------------------------------------
__global__ void JacobiKernel(
    const REAL* __restrict__ A,
    const REAL* __restrict__ b,
    const REAL* __restrict__ x_prev,
    REAL* x_curr,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    REAL sum = 0.0f;

    for (int j = 0; j < n; j++)
    {
        if (j == i) continue;
        sum += A[i * n + j] * x_prev[j];
    }

    x_curr[i] = (b[i] - sum) / A[i * n + i];
}

// ------------------------------------------------------------
// CPU Jacobi (reference)
// ------------------------------------------------------------
void JacobiCPU(const REAL* A, const REAL* b, const REAL* x_prev, REAL* x_curr, int n)
{
    for (int i = 0; i < n; i++)
    {
        REAL sum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            if (j == i) continue;
            sum += A[i * n + j] * x_prev[j];
        }
        x_curr[i] = (b[i] - sum) / A[i * n + i];
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main()
{
    const int n = 4096;                 // adjust this to see GPU advantage
    const int N = n * n;

    printf("Matrix size: %d x %d\n", n, n);

    // Host memory
    REAL* h_A = (REAL*)malloc(N * sizeof(REAL));
    REAL* h_b = (REAL*)malloc(n * sizeof(REAL));
    REAL* h_x_prev = (REAL*)malloc(n * sizeof(REAL));
    REAL* h_x_curr = (REAL*)malloc(n * sizeof(REAL));

    // Initialize random SPD-ish matrix
    for (int i = 0; i < n; i++)
    {
        REAL rowsum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            REAL v = (i == j) ? 2.0f : (rand() / (REAL)RAND_MAX) * 0.01f;
            h_A[i * n + j] = v;
            if (i != j) rowsum += fabs(v);
        }
        h_A[i * n + i] = rowsum + 1.0f; // make diagonally dominant
        h_b[i] = 1.0f;
        h_x_prev[i] = 0.0f;
    }

    // Device memory
    REAL* d_A, * d_b, * d_x_prev, * d_x_curr;
    cudaMalloc(&d_A, N * sizeof(REAL));
    cudaMalloc(&d_b, n * sizeof(REAL));
    cudaMalloc(&d_x_prev, n * sizeof(REAL));
    cudaMalloc(&d_x_curr, n * sizeof(REAL));

    cudaMemcpy(d_A, h_A, N * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_prev, h_x_prev, n * sizeof(REAL), cudaMemcpyHostToDevice);

    // ------------------------------------------------------------
    // GPU timing
    // ------------------------------------------------------------
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    JacobiKernel <<<grid, block >>> ( d_A, d_b, d_x_prev, d_x_curr, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("GPU Jacobi time: %.3f ms\n", ms);

    cudaMemcpy(h_x_curr, d_x_curr, n * sizeof(REAL), cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------
    // CPU timing
    // ------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    JacobiCPU(h_A, h_b, h_x_prev, h_x_curr, n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU Jacobi time: %.3f ms\n", cpu_ms);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x_prev);
    cudaFree(d_x_curr);

    free(h_A);
    free(h_b);
    free(h_x_prev);
    free(h_x_curr);

    return 0;
}
