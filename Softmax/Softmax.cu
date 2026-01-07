#include <cuda_runtime.h>

//#define DEBUG

#define CEIL(x, y)  (((x) + (y) - 1) / (y))

__device__ float sum;


__global__ void split_sum_kernel(const float* input, float* exp_input, float* output, int N, float* sum_split) {
    extern __shared__ float shared_exp[];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N) 
        return ;
    shared_exp[threadIdx.x] = __expf(input[idx]);
    exp_input[idx] = shared_exp[threadIdx.x];
    __syncthreads();


    if (threadIdx.x == 0) {
        // sum to sum_split
            //自毁长城哈哈哈哈
            //memcpy(&exp_input[blockDim.x * blockIdx.x], sharedexp, blockDim.x * sizeof(float));
        for (int i = 0; i < blockDim.x; i++){
            if (i + blockDim.x * blockIdx.x < N)
                sum_split[blockIdx.x] += shared_exp[i];
        }
    }
    __syncthreads();
    #ifdef DEBUG
    // 下面对于sum_split 的访问是垮块的， cuda在一个kernel内只能做到块内sync
    // sum_split is used before it's rightly wriiten in other blocks;
    // 修改建议： 放到下一个kernel中， kernel是顺序执行， 完了一个才另一个， 才能解决块间同步的问题
    if (idx == 0) {
        sum = 0;
        for(int i = 0; i < gridDim.x; i++) {
            sum += sum_split[i];
        }
    }
    #endif
}

__global__ void softmax_kernel(float *exp_input, float *output, int N, float* sum_split,
                                int origin_threads_per_blk) {
    #ifndef DEBUG
    if (threadIdx.x == 0) {
        sum = 0;
        for (int i = 0; i < (int)ceilf((float)N / origin_threads_per_blk); i++) {
            sum += sum_split[i];
        }
    }
    #endif
    if (threadIdx.x == 0) {
        for (int i = 0; i < N; i++) {
            output[i] = exp_input[i] / sum;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* sum_split, *exp_input;
    cudaMalloc(&sum_split, blocksPerGrid * sizeof(float));
    cudaMalloc(&exp_input, N * sizeof(float));
    cudaMemset(sum_split, 0,  blocksPerGrid * sizeof(float));
    split_sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
    (input, exp_input, output, N, sum_split);
    softmax_kernel<<<1, 1>>>(exp_input, output, N, sum_split, threadsPerBlock);
    cudaFree(sum_split);
    cudaFree(exp_input);

}
