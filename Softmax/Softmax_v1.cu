#include <cuda_runtime.h>
#include <cmath>

//#define DEBUG
//#define WRONG

#define CEIL(x, y)  (((x) + (y) - 1) / (y))

__device__ float sum;


__global__ void stable_kernel(const float* input, float* input_stable, int N) {
    extern __shared__ float max_battle_stage[];

    // 选出 blockDim.x 个 最大的来打擂台
    float max = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max = fmaxf(max, input[i]);
    }
    if (threadIdx.x < blockDim.x)
        max_battle_stage[threadIdx.x] = max; 

    for (int stride = blockDim.x / 2; stride > 0 ; stride /= 2) {
        float* t = max_battle_stage;
        if (threadIdx.x < stride) {
            t[threadIdx.x] = fmaxf(t[threadIdx.x], t[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float max_res = max_battle_stage[0];

    // each element - max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        input_stable[i] = input[i] - max_res;
    }
}

__global__ void split_sum_kernel(const float* input, float* exp_input, float* output, int N, float* sum_split) {
    extern __shared__ float shared_exp[];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
                //之前的版本是 >=N return, 在syncthreads时可能发生未定义行为
        shared_exp[threadIdx.x] = __expf(input[idx]);
        exp_input[idx] = shared_exp[threadIdx.x];
    } else {
        shared_exp[threadIdx.x] = 0.0f;
    }
    __syncthreads();


    if (threadIdx.x == 0) {
        //threadIdx.x == 0 , 单选一个线程来做sum是必要的！ 反复写不仅没意义， 而且危险 
                        //之前自毁长城哈哈哈哈
                        //memcpy(&exp_input[blockDim.x * blockIdx.x], sharedexp, blockDim.x * sizeof(float));
        // sum to sum_split
        float single_result = 0.0f;
        for (int i = 0; i < blockDim.x; i++){
            if (i + blockDim.x * blockIdx.x < N)
                                            //这个条件判断可以省略， 因为前面已经安全写0了， 加了也没关系
                single_result += shared_exp[i];
        }

        sum_split[blockIdx.x] = single_result;
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
    extern __shared__ float sum_list_for_reduction[];
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

    float* sum_split, *exp_input, *input_stable;
    cudaMalloc(&sum_split, blocksPerGrid * sizeof(float));
    cudaMalloc(&exp_input, N * sizeof(float));
    cudaMalloc(&input_stable, N * sizeof(float));
    stable_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
    (input, input_stable, N);
    split_sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>
    (input_stable, exp_input, output, N, sum_split);
    softmax_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof (float)>>>
    (exp_input, output, N, sum_split, blocksPerGrid);
                                                    //聪明的你有没有发现
                                                    //这里分配动态共享内存的大小
                                                    //都是（或者不是？）blockDim.x 的大小
                                                    //但是cuda居然不允许kernel内分配
                                                    //还要求编译期常量
                                                    //常量我能理解
                                                    //内部随手分配是不是可以考虑通融一下？                                                    
    cudaFree(sum_split);
    cudaFree(exp_input);
    cudaFree(input_stable);

}
