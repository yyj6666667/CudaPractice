
#include <cuda_runtime.h>

__device__ uint floatToOrder (float a) {
    uint a_u = *(uint*)&a; //指针强转， 哈哈
    // 根据正负生成掩码
    uint mask = (a_u & 0x80000000) ? 0xffffffff : 0x80000000;
    // 翻转, 异或bit = 1 就是翻转
    return a_u ^ mask;
}

__device__ float OrderTofloat(uint order) {
    uint mask = (order & 0x80000000) ? 0x80000000 : 0xffffffff;
    uint float_in_uint = order ^ mask;
    float res = *(float*)&float_in_uint;
    return res;                                
}

__global__ void pick(float* in, float* out, int N, int k) {
    extern __shared__ uint shared_topk[];
    __shared__ int lock;

    //init
    if (threadIdx.x < k) {
        shared_topk[threadIdx.x] = 0;
    }
    if (threadIdx.x == k) {
        lock = 0;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
                                                     //非常常用： Grid-Stride Loop
                                                     //尤其在blockDim无法一次性涵盖所有元素的时候
        float cur = in[j];
        uint order = floatToOrder(cur);
        if (order > shared_topk[k - 1]) {
            while (atomicCAS(&lock, 0, 1) != 0); 
                        // CAS: compare and swap
                        // other threads compare wtih 0 fails, traped in while
                        // cuda 中锁的简单实现
            __threadfence_block();
                        /*Cuda有多级缓存， 需要threadfence来确保当前线程看得到-
                        -前面线程的所有写入的sharedData， 不是为了看到lock的！！！
                          类似的：
                             __threadfence_block()
                             __threadfence()   for whole device(GPU)
                             __threadfence_system()  for all GPUs and CPU

                             该系列函数确保 shared + global memory 可见 
                             (第三个还要加上host)

                             shared mem On-chip SRAM, 单个block共享， life circle 同 block
                             global mem 存在显存(Video Memory), 所有block可见
                        */
            for (int cursor = k - 1; cursor >= 0; cursor--) {
                if (shared_topk[cursor] >= order) {
                    break;
                }
                //swap
                uint old = shared_topk[cursor];
                shared_topk[cursor] = order;

                if (cursor < k - 1) {
                                    // 后面的已经被挤掉了
                                    // 需要挪位置
                    shared_topk[cursor + 1] = old;
                }
            }

            __threadfence_block();
            atomicExch(&lock, 0); 
                                //atomic
                                //防止编译器成重排列
                                // lock 本身的写入对其他线程是可见的
        }

    }
    __syncthreads();

    if (threadIdx.x < k) {
        out[threadIdx.x] = OrderTofloat(shared_topk[threadIdx.x]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int n_threads = 1024;
    float* input_mut = const_cast<float*>(input);
    int shared_mem_size = k * sizeof(uint);
    pick<<<1, n_threads, shared_mem_size>>>(input_mut, output, N, k);
    cudaDeviceSynchronize(); // 确保之前cpu提交的kernel执行完成
}