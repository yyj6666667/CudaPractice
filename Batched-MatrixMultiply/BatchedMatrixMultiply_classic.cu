/**
 * @file BatchedMatrixMultiply_classic.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief classic tiled matmul, add dimension BATCH, one element one thread, ranging top 30% in leetgpu
 * @version classic
 * @date 2026-01-21
 * 
 * 
 */
#include <cuda_runtime.h>
#define TILE 32

__global__ void kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int batch_offset_A = blockIdx.z * M * K;
    int batch_offset_B = blockIdx.z * K * N;
    int batch_offset_C = blockIdx.z * M * N;

    const float* A_curr = A + batch_offset_A;
    const float* B_curr = B + batch_offset_B;
    float* C_curr = C + batch_offset_C;

    float value = 0.0f;

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    for (int i = 0; i < K ; i += TILE) {
        if(row < M && i + tx < K) {
            sA[ty][tx] = A_curr[row * K + i + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        if(col < N && i + ty < K) {
            sB[ty][tx] = B_curr[(i + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; j < TILE; j++) {
            value += sA[ty][j] * sB[j][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_curr[row * N + col] = value;
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1)/ TILE, (M + TILE - 1)/ TILE, BATCH);
    kernel<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
}
