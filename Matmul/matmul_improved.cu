/**
 * @brief out of bound writing remains to be debug
 * 
 */
#define BM  64
#define BN  64
#define BK  8
#define TM  8

__global__ void kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;

    int t_row = tx / BN;
    int t_col = tx % BN;

    float reg_C[TM] = {0.0f};

    int rowA = tx / BK;
    int colA = tx % BK;
    int rowB = tx / BN;
    int colB = tx % BN;
    for (int k = 0; k < K; k += BK) {
        if (by * BM + rowA < M && k + colA < K) {
            sA[rowA][colA] = A[(by * BM + rowA) * K + k + colA];
        } else{
            sA[rowA][colA] = 0.0f;
        }
        if (bx * BN + colB < N && k + rowB < K) {
            sB[rowB][colB]  = B[(k + rowB) * N + bx * BN + colB];
        } else {
            sB[rowB][colB] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int fix = 0; fix < BK; fix ++) {
            float reg_tem = sB[fix][t_col];
            for (int i = 0; i < TM; i++) {
                reg_C[i] += sA[t_row * TM + i][fix] * reg_tem;
            }
        }
        __syncthreads();

    }

    for (int i = 0; i < TM; i++) {
        int cur_row = by * BM + t_row * TM + i;
        int cur_col = bx * BN + t_col;
        if (cur_row < M && cur_col < N) {
            C[cur_row * N + cur_col] = reg_C[i];
        }
    }

}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim((BM * BN) / TM);
    dim3 gridDim((N + BN - 1)/BN, (M + BM - 1)/BM);
    kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}