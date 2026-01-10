/**
 * @file Convolution2D_version2.cu
 * @author yyj (you@domain.com)
 * @brief  range 70% in geetgpu
 * @version 2
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
                                extern __shared__ float s_kernel[];
                                                                  // 已知k_max 也可以用宏定义声明

                                int tid = threadIdx.x + threadIdx.y * blockDim.x;
                                                                                // 如果有threadIdx.z , 是不是还要加上
                                                                                // .. + threadIdx.z * blockDim.x * blockDim.y
                                //多线程协作加载到共享内存， 管它kernel大小， 巨大巨小都可以用grid-stride 赋值
                                for (int i = tid; i < k_rows * k_cols; i += blockDim.x * blockDim.y) {
                                                    // tid 是逻辑延续的传递， 哈哈哈哈哈
                                    s_kernel[i] = kernel[i];
                                }
                                __syncthreads();

                                //寻找原始input位置，注意沿着内存的连续方向
                                int in_loc = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * in_cols;
                                                                                                        // 非常简单抽象粗暴的一句， 有助于加深理解
                                                                                                        //xxxDim, 是xxx 的shape信息
                                                                                                        //xxxIdx, block独有， 是在grid 中的位置 

                                //debug: 写入output的线程筛选条件错了
                                //这就是计算res_col, res_row 的必要性！
                                int cur_col = threadIdx.x + blockIdx.x * blockDim.x;
                                int cur_row = threadIdx.y + blockIdx.y * blockDim.y; 
                                if (cur_col < res_cols && cur_row < res_rows) {
                                                        //res_cols既是大小， 也是坐标
                                    float mul_res = 0.0f;
                                    int   out_loc = cur_row * res_cols + cur_col;
                                                                                //debug: loc = a * cols + b, not a * b!
                                    const float *cursor = input + in_loc;
                                    for (int i = 0; i < k_rows ; i++) {
                                        for(int j = 0; j < k_cols; j++) {
                                            mul_res += (*cursor) * s_kernel[i * k_cols + j];
                                            cursor += 1;
                                        }
                                        cursor += in_cols - k_cols ;   //易错
                                    }
                                    output[out_loc] = mul_res;
                                }
                                __syncthreads();
                            }

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {
                        //突发奇想， 让threadIdx.x 对应 res_row, 大概会慢32倍吧哈哈哈
                        //访问的地址是由threadIdx.x 算出来的， 逻辑上是对应的，
                        //应该尽量的让同一个warp里面的threadIdx.x 访问的地址连续
                        //触发warp的内存合并机制

                        //set grid
                        int res_rows = input_rows - kernel_rows + 1;
                        int res_cols = input_cols - kernel_cols + 1;
                        int res_len  = res_rows * res_cols;
                                    // dim3就是一个自带构造函数的类！ 如果不传参数就默认是1, 传的顺序是: x, y, z
                                    // 初学者很容易搞混的一个点是：
                                    // dim3 的 x 隐含内存连续方向  ,一般数组最右索引才是内存连续方向
                                    // 因此grid声明时， grid.x 计算时分母用res_cols, 内存连续方向
                                    // kernel 内部调用时也是反的， 这样更符合cuda规范
                                    // 在kernel外部用自己的名字就可以访问了， 在kernel内部用cuda内置的名字访问
                        dim3 block2D_yyj(TILE, TILE);
                        dim3 grid2D_yyj (CEIL(res_cols, block2D_yyj.x), CEIL(res_rows, block2D_yyj.y));

                        int shared_size = kernel_rows * kernel_cols * sizeof(float);
                        conv_kernel<<<grid2D_yyj, block2D_yyj, shared_size>>>
                        (input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols, res_rows, res_cols);
                      }
