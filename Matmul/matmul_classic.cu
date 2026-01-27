/**
 * @file matmul_classic.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief classical Tile splited matmul, ranging 50% in leetgpu
 * 
 * 
 */
#include <cuda_runtime.h>
#define TILE 16
#define CEIL(x, y) (((x) + (y) - 1) / (y))

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int r = threadIdx.y + blockDim.y * blockIdx.y;

    float sum = 0.0f;
    for (int i = 0; i < CEIL(N, TILE); i++) {
        if (r < M && (i * TILE + threadIdx.x < N)) {
            sA[threadIdx.y][threadIdx.x] = A[r * N + i * TILE + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (c < K && (i * TILE + threadIdx.y < N)) {
            sB[threadIdx.y][threadIdx.x] = B[c + (i * TILE + threadIdx.y) * K];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (r < M && c < K) {
        C[r * K + c] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    dim3 block(TILE, TILE);
    dim3 grid (CEIL(K, TILE), CEIL(M, TILE));
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
                
}

