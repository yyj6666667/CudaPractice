/**
* version1, 针对kernel = 32 * 32 
* 这题是经典的一个大位置里面找小位置， 通过小位置再反推大位置
* grid 用结果的个数就可以了， 他们是并行且不相干的
* 
*/

#include <cuda_runtime.h>

__global__ void cov_kernel(const float* input, const float* kernel, float* output,
                          int in_rows, int in_cols, int k_rows, int k_cols ) {
                            extern __shared__ float shared_mem[];
                            float* shared_sum = shared_mem;
                            float* shared_kernel = shared_mem + blockDim.x;

                            int res_idx = blockIdx.x;
                            int res_rows = in_rows - k_rows + 1;
                            int res_cols = in_cols - k_cols + 1;
                            int size_kernel = k_rows * k_cols;
                            // in each block
                            int res_row_idx = res_idx / res_cols;
                            int res_col_idx = res_idx % res_cols;
                                                                //haha , here both is res_cols, this is axis = 1 

                            const float* res_begin = input + res_row_idx * in_cols + res_col_idx;
                                                            // where a single cov begins in input
    
                            //do conv
                            float sum_per_conv = 0.0f;
                            for (int i = threadIdx.x; i < size_kernel; i += blockDim.x) {
                                int offset_row = i / k_cols; 
                                int offset_col = i % k_cols;

                                shared_kernel[i] = kernel[i];
                                //seek loc by 2 offset
                                float target = *(res_begin + offset_row * in_cols + offset_col);
                                sum_per_conv += target * shared_kernel[i];
                            }
                            shared_sum[threadIdx.x] = sum_per_conv;
                            __syncthreads();

                            //reduce sum_split
                            for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
                                if (threadIdx.x < stride) {
                                    shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
                                }
                                __syncthreads();
                                               // reduce sum need to sync each epoch , cause later we use it across;
                            }

                            if (threadIdx.x == 0) {
                                output[res_idx] = shared_sum[0];
                            }
                            // game over
                            
                          }


// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
                        int n_grids = (input_rows - kernel_rows + 1) *
                                     (input_cols - kernel_cols + 1);
                        int n_threads = 256;
                        int shared_size = (n_threads + kernel_rows * kernel_cols) * sizeof(float);

                        cov_kernel<<<n_grids, n_threads, shared_size>>>
                        (input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols); 
                      
                      }
