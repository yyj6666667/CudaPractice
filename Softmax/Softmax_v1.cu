#include <cuda_runtime.h>

//#define DEBUG
//#define WRONG

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
    #ifdef WRONG
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

__global__ void softmax_kernel(float *exp_input, float *output, int N, float* sum_split, int len_sum_split) {
    extern __shared__ sum_list_for_reduction = [];
    float sum_single_thread = 0.0f; 
                                 // 这是用来之后统一给sum_list_for_reduction 赋值的， 这样可以初始化0的麻烦
                                 // 有多个sum_single_thread, 因为这是cuda， 避免了初始化0的麻烦, 再重复一遍
    for (int i = threadIdx.x; i < len_sum_split; i += blockDim.x)  {
                                                                    //Grid-Stride loop
        sum_single_thread += sum_split[i];
    }
    sum_list_for_reduction[threadIdx.x] = sum_single_thread;

    __syncthreads();

    // 神奇的归约， from O(n) to O(log(n))
    // a fixed pattern ， 太妙了
    // 然而这是一个简陋版的归约， 只能处理blockDim.x 是 2 的幂的情况
                                        //没错， 可能偶数都不行
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sum_list_for_reduction[threadIdx.x] += 
            sum_list_for_reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float total = sum_list_for_reduction[0];

    //Grid-Stride 计算输出
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = exp_input[i] / total;
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
    softmax_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof (float)>>>(exp_input, output, N, sum_split, blocksPerGrid);
    cudaFree(sum_split);
    cudaFree(exp_input);

}
