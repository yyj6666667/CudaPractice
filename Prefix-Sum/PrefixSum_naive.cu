/**
 * @file Prefix-Sum_naive.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief a naive and ugly version with dozens of bug
          I put it here for anti-teaching example
          排名倒数10%
 * @version 0.1
 * @date 2026-01-12
 * 
 * 
 */
#include <cuda_runtime.h>

__global__ void prefix_kernel(float* in,  float* out, int n) {
    extern __shared__ double s_sum[];
    int   size_per_thread = (n + blockDim.x - 1) / blockDim.x;
    int   in_offset_begin = threadIdx.x * size_per_thread;
    float* in_begin = in + in_offset_begin;
    int   last_idx_per_thread = -1;

    double local_sum = 0.0f;
    for (int i = 0; i < size_per_thread; i++) {
        int idx = i + threadIdx.x * size_per_thread;
        if (idx < n) {
            local_sum += in[idx];
            in[idx]    = local_sum;
        }
    }

    s_sum[threadIdx.x] = (threadIdx.x * size_per_thread < n) ? local_sum : 0.0f;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x - 1; i++) {
            s_sum[i+1] += s_sum[i];
        }
    }
    __syncthreads();

    float pre_sum = (threadIdx.x > 0) ? s_sum[threadIdx.x - 1] : 0.0f;

    float* cursor;
    for (int i = 0; i < size_per_thread; i++) {
        if (i + threadIdx.x * size_per_thread < n) {
            cursor = in_begin + i;
            *cursor += pre_sum;
            out[i + threadIdx.x * size_per_thread] = *cursor;
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    //solution: div into x grids, each 做梯度加， 最后统一聚合， 每个线程负责一个小批次
    //其实可以在grid 维度上在做文章， 先写个简陋的
    int n_threads = 256;
    int s_size    = n_threads * sizeof(double);
    float* in_mut = NULL;
    cudaMalloc(&in_mut, N * sizeof(float));
    cudaMemcpy(in_mut, input, N * sizeof(float), cudaMemcpyDeviceToDevice);

    prefix_kernel<<<1, n_threads, s_size>>>(in_mut, output, N);
}
