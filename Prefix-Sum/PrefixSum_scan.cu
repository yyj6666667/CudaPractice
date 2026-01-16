/**
 * @file PrefixSum_scan.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief  using scan algorithm to reduce prefixSum time cost to O(log(n)), ranging 60% in leetgpu
 * @date 2026-01-16
 * 
 * 
 */
#include <cuda_runtime.h>

__global__ void scan_kernel(const float* input, float* block_sum, float* output, int N) {
    __shared__ float s_input[256];
    int  gid = threadIdx.x + blockIdx.x * blockDim.x;
    s_input[threadIdx.x] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    //naive "scan" for parallel prefix sum
    //refer Mark Harris's paper "Parallel Prefix Sum (Scan) with CUDA"
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0.0f;
        if (threadIdx.x >= stride) {
            val = s_input[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            s_input[threadIdx.x] += val;
        }
        __syncthreads();
    }

    if (gid < N) {
        output[gid] = s_input[threadIdx.x];
    }
    if (threadIdx.x == 255) {
        block_sum[blockIdx.x] = s_input[255];
    }
}

__global__ void add_block_sum_kernel(float* block_sum, float* output, int N) {
    int gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid < N) {
        float sum = 0.0f;
        if (threadIdx == 0)
            for (int i = 0; i < blockIdx.x; i++) {
                sum += block_sum[i];
            }
        output[gid] += sum;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int n_threads = 256;
    int n_blocks = (N + 255) / n_threads;
    float* block_sum = NULL;
    cudaMalloc(&block_sum, n_blocks * sizeof(float));

    scan_kernel<<<n_blocks, n_threads>>>(input, block_sum, output, N);
    add_block_sum_kernel<<<n_blocks, n_threads>>>(block_sum, output, N);
}

