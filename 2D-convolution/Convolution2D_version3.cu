/**
 * @file Convolution2D_version2.cu
 * @author yyj (you@domain.com)
 * @brief  每个矩阵加载它所在需要的tile， ``better shared-mem used, range 70% in geetgpu, 奇怪的是排名变化不大
 * @version 3
 * @date 2026-01-10
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include <cuda_runtime.h>

#define TILE 16
#define CEIL(x, y)  (((x) + (y) - 1) / (y))

__global__ void conv_kernel(const float* input, const float* kernel, float* output, 
                            int in_rows, int in_cols, int k_rows, int k_cols, int res_rows, int res_cols) {
                                extern __shared__ float s_mem[];
                                float* s_kernel = s_mem;
                                float* s_input  = s_mem + k_rows * k_cols;
                                                                  // 已知k_max 也可以用宏定义声明

                                int tid = threadIdx.x + threadIdx.y * blockDim.x;
                                for (int i = tid; i < k_rows * k_cols; i += blockDim.x * blockDim.y) {
                                    s_kernel[i] = kernel[i];
                                }
                                __syncthreads();

                                //version3 improvement:
                                int   iter_offset;

                                for (int i = tid; i < (TILE + k_cols -1) * (TILE + k_rows -1); 
                                     i += blockDim.x * blockDim.y) {
                                        int local_row = i / (TILE + k_cols - 1);
                                        int local_col = i % (TILE + k_cols - 1);
                                        int global_row = blockDim.y * blockIdx.y;
                                        int global_col = blockDim.x * blockIdx.x;
                                        iter_offset = local_col + global_col + in_cols * (local_row + global_row);//offset must be updated each epoch!!! 
                                    if (iter_offset < in_rows * in_cols) {
                                        s_input[i] = *(input + iter_offset);
                                    }
                                }
                                __syncthreads();

                                int cur_col = threadIdx.x + blockIdx.x * blockDim.x;
                                int cur_row = threadIdx.y + blockIdx.y * blockDim.y;

                                if (cur_col < res_cols && cur_row < res_rows) {
                                                        //res_cols既是大小， 也是坐标
                                    float mul_res = 0.0f;
                                    int   out_loc = cur_row * res_cols + cur_col;
                                    const float *cursor = s_input + threadIdx.y * (TILE + k_cols -1) + threadIdx.x;
                                    for (int i = 0; i < k_rows ; i++) {
                                        for(int j = 0; j < k_cols; j++) {
                                            mul_res += *(cursor) * s_kernel[i * k_cols + j];
                                            cursor += 1;
                                        }
                                        cursor += (TILE + k_cols - 1) - k_cols ;   //易错
                                    }
                                    output[out_loc] = mul_res;
                                }
                                __syncthreads();
                            }

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {

                        int res_rows = input_rows - kernel_rows + 1;
                        int res_cols = input_cols - kernel_cols + 1;
                        int res_len  = res_rows * res_cols;
                        dim3 block2D_yyj(TILE, TILE);
                        dim3 grid2D_yyj (CEIL(res_cols, block2D_yyj.x), CEIL(res_rows, block2D_yyj.y));

                        int shared_size = kernel_rows * kernel_cols * sizeof(float) //kernel
                                        + (TILE + kernel_rows - 1) * (TILE + kernel_cols - 1) * sizeof(float);                  //input
                        conv_kernel<<<grid2D_yyj, block2D_yyj, shared_size>>>
                        (input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, res_rows, res_cols);
                      }
