/**
 * @file   SoftmaxAttention_improved.cu
 * @author Yujie-Yang 
 * @brief  leetcode 排名前6%
 * @date   2026-01-14
 */

#include <cuda_runtime.h>
#include <cmath>
#define TILE 16
#define CEIL(x, y) (((x) + (y) - 1) / (y))

__global__ void trans_kernel(const float* K, float* K_T, int N, int d){
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int r = threadIdx.y + blockDim.y * blockIdx.y;

    if (c < d && r < N) {
        K_T[r + c * N] = K[c + r * d];
    }
}

__global__ void matmul_kernel(const float* Q, const float* K_T, float* scores, int M, int d, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int r = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.0f;
    for (int i = 0; i < CEIL(d, TILE); i++) {
        if (r < M && (threadIdx.x + i * TILE) < d) {
            sA[threadIdx.y][threadIdx.x] = Q[threadIdx.x + i * TILE + r * d];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (c < N && (threadIdx.y + i * TILE) < d) {
            sB[threadIdx.y][threadIdx.x] = K_T[c + (threadIdx.y + i * TILE) * N]; 
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
    
        for (int i = 0; i < TILE; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (c < N && r < M) {
        scores[r * N + c] = sum; 
    }
}

__global__ void softmax_kernel(float* scores, int M, int N, float scalar) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s_max[256];
    __shared__ float s_sum[256];

    if (row < M) {

        float max_val = -INFINITY;
        for (int i = tid; i < N; i += blockDim.x) {
            max_val = fmaxf(max_val, scores[row * N + i]);
        }
        s_max[tid] = max_val;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
            }
            __syncthreads();
        }

        float global_max = s_max[0];
        float sum_exp = 0.0f;
        for (int i = tid ; i < N; i += blockDim.x) {
            float tem = expf((scores[i + row * N] - global_max) * scalar) ;
            sum_exp += tem;
            scores[i + row * N] = tem;
        }
        s_sum[tid] = sum_exp;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tid < stride) {
                s_sum[tid] +=  s_sum[tid + stride];
            }
            __syncthreads();
        } 

        float global_sum = s_sum[0];
        for (int i = tid ; i < N; i += blockDim.x) {
            scores[i + row * N] /= global_sum;
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
    //tiled trans:  K -> K^T     
    float* K_T = NULL;
    cudaMalloc(&K_T, N * d * sizeof(float));
    dim3 block_trans(TILE, TILE);  
    dim3 grid_trans(CEIL(N, TILE), CEIL(d, TILE));
    trans_kernel<<<grid_trans, block_trans>>>(K, K_T, N, d);

    //tiled matmul: (M, d) * (d, N) -> (M, N)
    float* scores = NULL;
    cudaMalloc(&scores, M * N * sizeof(float));
    dim3 block_matmul(TILE, TILE);
    dim3 grid_matmul(CEIL(N, TILE), CEIL(M, TILE));
    matmul_kernel<<<grid_matmul, block_matmul>>>(Q, (const float*)K_T, scores, M, d, N);

    //行级softmax-stable
    float scalar = 1 / sqrtf(d);
    softmax_kernel<<<M, 256>>>(scores, M, N, scalar);

    //tiled matmul: (M, N) * (N, d) -> (M, d)
    dim3 block_res(TILE, TILE);
    dim3 grid_res(CEIL(d, TILE), CEIL(M, TILE));
    matmul_kernel<<<grid_res, block_res>>>((const float*)scores, V, output, M, N, d);

    cudaFree(K_T);
    cudaFree(scores);
}
