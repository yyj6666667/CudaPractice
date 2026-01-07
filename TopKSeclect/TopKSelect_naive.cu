
#include <cuda_runtime.h>

//  dealing with the minus case of flaot 
//  using ordered encoding
// 你的疑问： 必须是unsigned int , unsigned 强调一种位模式
// unsigned int 跟 int的比较逻辑是不一样的， eg：
// 0x80000000 ? 0x00000001, 结果不一样
// 而且右移>>逻辑也不同， int是算术右移， 要补符号位， 才能满足运算， 想想为什么0x80000000是最大的负数？ 0xffffffff是最小的负数（-1）？ 就是要满足算术右移
// 综上， 必须转换成unsigned int !
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

__global__ void pick(float* in, float* out, int N, int i) {
    __shared__ unsigned long long packedmax ;
    //shared memory block内可见， 不能直接初始化， 
    //单独用一个thread来做初始化
    if (threadIdx.x == 0) {
        packedmax = 0ULL;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float cur = in[j];
        uint order_cur = floatToOrder(cur);
        //pack to 64bit for easier comparison
        unsigned long long pack = ((unsigned long long)order_cur << 32)
                                  | j ;
        //cuda's atomic ops
        // compare packedmax and pack, choose the bigger one write into packedmax by ptr  
        // atomic 返回值是执行ops之前， 该ptr位置指向的旧值                       
        atomicMax(&packedmax, pack);

    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // 置位， 避免下一次再找到
        uint idx = (uint) packedmax;
        in[idx] = -INFINITY;// float, double 都用 INFINITY 表示极端值
                            // int 用 INT_MAX, INT_MIN
                            // uint 用 UINT_MAX                          
        //write res
        *(out + i) = OrderTofloat((uint)(packedmax >> 32));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    int n_threads = 1024;
    float* input_mut = const_cast<float*>(input);
    for (int i = 0; i < k; i++) {
        pick<<<1, n_threads>>>(input_mut, output, N, i);
                                                        /// <<< >>> kerner执行是异步的， cpu只是提交任务
    }
    cudaDeviceSynchronize(); // 确保cpu提交的kernel执行完成
    //printf("%f", output[k-1]);
}