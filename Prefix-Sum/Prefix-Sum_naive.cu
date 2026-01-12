/**
 * @file Prefix-Sum_naive.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief a naive and ugly version with dozens of bug
          I put it here for anti-teaching example
 * @version 0.1
 * @date 2026-01-12
 * 
 * 
 */
#include <cuda_runtime.h>
//这是一个最优子问题

__global__ void prefix_kernel(float* in,  float* out, int n) {
    extern __shared__ float s_sum[];
    __shared__ float  s_second_sum[1024];
    float sum_thread = 0.0f;
    int   size_per_thread = (n + blockDim.x - 1) / blockDim.x;
    int   in_offset_begin = threadIdx.x * size_per_thread;
    float* in_begin = in + in_offset_begin;
    for (int i = 0; i < size_per_thread - 1; i++) {
        //特判最后一个
        if (threadIdx.x == blockDim.x - 1 && i >= n % size_per_thread - 1) {
            //do nothing
        } else {
        in_begin[i + 1] += in_begin[i];
        }
        //加载进入s_mem
        if (i == size_per_thread - 2 || (threadIdx.x == blockDim.x - 1) && i == n % size_per_thread - 1) {
            s_sum[threadIdx.x] = in_begin[i + 1];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x - 1; i++) {
            s_sum[i+1] += s_sum[i];
        }
    }
    __syncthreads();

    float* cursor = in_begin;
    for (int i = 0; i < size_per_thread; i++) {
        if (threadIdx.x == blockDim.x - 1 && i == n % size_per_thread) {
            return;
        }
        if (threadIdx.x != 0)
            out[i + threadIdx.x * size_per_thread] = in[i + threadIdx.x * size_per_thread] + s_sum[threadIdx.x - 1];
                                                    //直接命名为变量吧， 太丑了
    }


}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    //solution: div into x grids, each 做梯度加， 最后统一聚合， 每个线程负责一个小批次
    //其实可以在grid 维度上在做文章， 先写个简陋的
    int n_threads = 256;
    int s_size    = n_threads * sizeof(float);
    float* in_mut = NULL;
    cudaMalloc(&in_mut, N * sizeof(float));
    cudaMemcpy(in_mut, input, N * sizeof(float), cudaMemcpyDeviceToDevice);

    prefix_kernel<<<1, n_threads, s_size>>>(in_mut, output, N);
}
