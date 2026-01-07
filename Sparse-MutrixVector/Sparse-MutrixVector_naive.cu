/*
* 排在60%, 一定还可以在grid层做文章
*/
#include <cuda_runtime.h>


__global__ void single_kernel(const float* A, const float* x, float* out, int M, int N) {
    extern __shared__ float shared_mem[];

    const float* A_each_grid = A + blockIdx.x * N; // A in row
    float*     out_each_grid = out + blockIdx.x;

    // do element-wise multi
    float muti_res = 0.0f;
    for (int i = threadIdx.x ; i < N; i += blockDim.x) {
        muti_res += A_each_grid[i] * x[i];
    }
    shared_mem[threadIdx.x] = muti_res;
                                      // haha, most anti-logic moment, 
                                      // this is cuda
                                      // we are in a block!
    __syncthreads(); 
                    //这个必须加
                    //一个block里面， sharedmem由多个线程写的
                    //后面会交叉用到， 就必须同步!!

    // do reduce over each row
    for (int stride = blockDim.x / 2 ; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum_each_grid = shared_mem[0];
    *out_each_grid = sum_each_grid;

}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int n_threads = 1024;

    int size_shared_mem = n_threads * sizeof(float);
    single_kernel<<<M, n_threads, size_shared_mem>>>(A, x, y, M, N);
}

//debug: 你把blockIdx 写成 gridIdx, hh， 不存在啊
