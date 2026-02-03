/**
 * @brief matmul_8elem_per_thread, ranging top 12% of leetgpu
 * @author yyj
 */
#define BM  64 // tiled size for row
#define BN  64 // tiled size for col
#define BK  16 //K-loop 前进步长， 也可设置为8
#define TM  4  //element num per thread
               //"TM" indicates specific to one thread on the axis row (which has length"M")

__global__ void kernel(const float* A, const float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    unsigned int bx = blockIdx.x, by = blockIdx.y;
    unsigned int tx = threadIdx.x;

    //declared for copy
    int n_threads = (BM * BN) / TM;
    int elements_sA  = BM * BK;
    int elements_sB  = BK * BN;

    //declared for compute
    int tRow = tx / BN * TM;
    int tCol = tx % BN;
    float reg_C[TM] = {0.f};
    
    //classic K-loop
    for(int k = 0; k < N; k += BK) {
        //copy 1:
        for (int stride = 0; stride < elements_sA; stride += n_threads) {
            int idx = tx + stride;
            if (idx < elements_sA) {
                int row_sA = idx / BK;
                int col_sA = idx % BK;
                int g_row = by * BM + row_sA;
                int g_col = k + col_sA;

                sA[row_sA][col_sA] = (g_row < M && g_col < N) ? A[g_row * N + g_col] : 0.f;
            }
        }
                                                                        //在外面套一层stride循环， 
                                                                        //是因为n_threads不能保证一次就把一个tile的sA成功写入
        //copy 2:
        for (int stride = 0; stride < elements_sB; stride += n_threads) {
            int idx = tx + stride;
            if (idx < elements_sB) {
                int row = idx / BN;
                int col = idx % BN;
                int g_row = k + row;
                int g_col = bx * BN + col;

                sB[row][col] = (g_row < N && g_col < K) ? B[g_row * K + g_col] : 0.f;
            }
        }

        __syncthreads();

        //compute:
        //(BM, BK) & (BK, BN)
        //using single register to accelarate
        //视图： 把线程按逻辑平铺到res矩阵里面
        //注意！ tRow = (tid / BN) * TM, 因为一个线程负责了TM个结果（列方向）
        for (int i = 0; i < BK; i++) {
            float reg_B = sB[i][tCol];
            for (int aj = 0; aj < TM; aj++) {
                float reg_A = sA[tRow + aj][i];
                reg_C[aj] += reg_A * reg_B;
            }
        }
        __syncthreads();
                        // 开始忘写了， 旧计算使用的shared mem可能被新数据覆盖！
    }
    
    //write back
    for (int i = 0; i < TM; i++) {
        int g_row = by * BM + tRow + i;
        int g_col      = bx * BN + tCol;
        if (g_row < M && g_col < K) {
            C[g_row * K + g_col] = reg_C[i];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim((BM / TM) * BN);
    dim3 gridDim((K + BN - 1)/BN, (M + BM - 1)/BM);
    kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}