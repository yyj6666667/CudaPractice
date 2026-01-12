/**
* 打榜在 倒数1% 左右的位置， 不可以作为模板
*/
#include <cuda_runtime.h>

__global__ void reduce_kernel(const float* input, float *output, int N) {
    extern __shared__ float shared_sum[];
    float single_sum = 0;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        single_sum += input[i];
    }
    shared_sum[threadIdx.x] = single_sum;
    __syncthreads();

    // reduce sum_up
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
                        // only permit one to write
        float sum_final = shared_sum[0];
        *output = sum_final;
    }
}


// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int n_threads = 1024;
    reduce_kernel<<<(N + n_threads - 1) / n_threads, n_threads, n_threads * sizeof(float)>>>
    (input, output, N);
}
