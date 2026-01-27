
#define BM  64
#define BK  64
#define TM  8

__global__ void kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockDim((BM * BK) / TM);
    dim3 gridDim((K + BK - 1)/BK, (M + BM - 1)/BM);
    kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}