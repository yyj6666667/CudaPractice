/**
 * @file BatchedMatrixMultiply_improved.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief   ranging top 20% of leetgpu
 * @version 1
 * @date 2026-01-27
 * 
 * 
 */
#include <cuda_runtime.h>
#define BM  64
#define BN  64
#define BK  8
#define TM  8

__global__ void kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    int threadcol = (threadIdx.x % BN);
    int threadrow = (threadIdx.x / BN) * TM;
    int Segmentx =  BN * blockIdx.x;
    int Segmenty =  BM * blockIdx.y;
    int        z =  blockIdx.z;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int innerColA = threadIdx.x % BK;
    int innerRowA = threadIdx.x / BK;
    int innerColB = threadIdx.x % BN;
    int innerRowB = threadIdx.x / BN;

    float threadRes[TM] = {0.0};
    
    for (int stride = 0; stride < K; stride += BK) {
        int globalRowA = Segmenty + innerRowA;
        int globalColA = innerColA + stride;
        int globalRowB = innerRowB + stride;
        int globalColB = Segmentx + innerColB;

        As[innerRowA][innerColA] = (globalRowA < M && globalColA < K) ? A[z * M * K + globalRowA * K + globalColA] : 0.0f;
        Bs[innerRowB][innerColB] = (globalRowB < K && globalColB < N) ? B[z * K * N + globalRowB * N + globalColB] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BK; j++) {
            float tempB = Bs[j][threadcol];
            for (int i = 0; i < TM; i++) {
                threadRes[i] += As[threadrow + i][j] * tempB;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int globalRow = Segmenty + threadrow + i;
        int globalCol = Segmentx + threadcol;
        if (globalRow < M && globalCol < N) {
            C[z * M * N + globalRow * N + globalCol] = threadRes[i];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 blockDim((BM * BN) / TM);
    dim3 gridDim((N + BN - 1)/BN, (M + BM - 1)/BM, BATCH);
    kernel<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
}
