/**
 * @file SoftmaxAttention_naive.cu
 * @author Yujie-Yang (yyj6666667@gmail.com)
 * @brief  adapted for large N, but still naive, 由于N的限制， 暂时没有用shared_memory
 * @version 0.1
 * @date 2026-01-14
 * 
 * 
 */
#include <cuda_runtime.h>
#include <cmath>

__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* hidden_buf, float* output, int M, int N, int d) {
    // one thread one row in M
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1 / sqrtf(d);
    if (row < M) {
        float* row_store = hidden_buf + row * N;

        for (int i = 0; i < N; i++) {
            float sum_tem = 0.0f;
            for (int j = 0; j < d; j++) {
                sum_tem += Q[row * d + j] * K[i * d + j];
            }
            row_store[i] = sum_tem * scale;
        }
        //apply softmax to each row
        float max = -INFINITY;
        float sum_row = 0.0f;
        for (int i = 0; i < N; i++) {
            max = fmaxf(max, row_store[i]);
        }

        for (int i = 0; i < N; i++) {
            row_store[i] -= max;
            row_store[i] = expf(row_store[i]);
            sum_row += row_store[i];
        }

        for (int i = 0; i < N; i++) {
            row_store[i] /= sum_row;
        }
        //multiply with V
        for (int i = 0; i < d; i++) {
            float res_row = 0.0f;
            for (int j = 0; j < N; j++) {
                res_row += row_store[j] * V[j * d + i]; // 跨行访问了
            }
            output[row * d + i] = res_row;
        }
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d) {
                        int n_threads = 256;
                        int n_blocks  = (M + n_threads - 1) / n_threads;
                        int smem_sz   = N * sizeof(float);

                        float* hidden_buf;
                        cudaMalloc(&hidden_buf, M * N * sizeof(float));
                        attention_kernel<<<n_blocks, n_threads, smem_sz>>>(Q, K, V, hidden_buf, output, M, N, d);
                        cudaFree(hidden_buf);
                      }
