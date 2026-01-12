/**
 * @file DotProduct.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief  ranging top 20% in leetGPU in the world
 * @version 0.1
 * @date 2026-01-12
 * 
 * @copyright Copyright (c) 2026
 * 
 */

__global__ void kernel(const float* A, const float* B, float* result, int n) {
    extern __shared__ float s_sum[];
    float sum = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += A[i] * B[i];
    }
    s_sum[threadIdx.x] = sum;
    __syncthreads();

    //reduce
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        result[0] = s_sum[0];
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int n_threads = 1024;
    int n_grids = 1;
    int shared_size = n_threads * sizeof(float);

    kernel<<<n_grids, n_threads, shared_size>>>(A, B, result, N);
}
