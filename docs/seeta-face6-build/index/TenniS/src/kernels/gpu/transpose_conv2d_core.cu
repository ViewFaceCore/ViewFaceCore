#include <kernels/gpu/transpose_conv2d_core.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>

#include "backend/common_structure.h"



#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/gpu/cuda_context.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"
#include "kernels/gpu/math_cublas.h"

#include <kernels/gpu/operator_on_gpu.h>
#include "kernels/gpu/gpu_kernel.h"

/////////////////////////////////////////////////
namespace ts {
    namespace gpu {

        template <typename T>
        static __global__ void gpu_col2im_kernel(const int n, const T* data_col,
            const int height, const int width, 
            const int kernel_h, const int kernel_w,
            const int pad_top, const int pad_left,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            const int height_col, const int width_col, T* data_im) {

            for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
                T val = T(0.f); 
                const int w_im = index % width + pad_left;
                const int h_im = (index / width) % height + pad_top;
                const int c_im = index / (width * height);
                int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
                int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
                const int w_col_start =
                           (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
                const int w_col_end = min(w_im / stride_w + 1, width_col);
                const int h_col_start =
                           (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
                const int h_col_end = min(h_im / stride_h + 1, height_col);
                for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
                    for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                        int h_k = (h_im - h_col * stride_h);
                        int w_k = (w_im - w_col * stride_w);
                        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                            h_k /= dilation_h;
                            w_k /= dilation_w; 
                            int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
                            val += data_col[data_col_index];
                        }
                     }
                 }
              data_im[index] = val;
           }

        }



        template<typename T>
        __global__ static void gpu_transpose_conv2d_run_kernel(const T *pfA, const T *pfB, T *pfC, int dwN, int dwM, int dwP) {
            __shared__ T pfTmpA[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
            __shared__ T pfTmpB[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
            int dwGlobalIdxN = blockDim.y * blockIdx.y + threadIdx.y;
            int dwGlobalIdxM = blockDim.x * blockIdx.x + threadIdx.x;
            int dwLocalIdxN = threadIdx.y;
            int dwLocalIdxM = threadIdx.x;
            T fResults = T(0.f);
            T fComp = T(0.f);
            for (int j = 0; j < dwP; j += TRANS_BLOCK_DIM) {
                if (dwGlobalIdxN < dwN && dwLocalIdxM + j < dwP) {
                    pfTmpA[dwLocalIdxN][dwLocalIdxM] = pfA[(dwLocalIdxM + j) * dwN + dwGlobalIdxN]; 
                }
                else {
                    pfTmpA[dwLocalIdxN][dwLocalIdxM] = 0;
                }

                if (dwGlobalIdxM < dwM && dwLocalIdxN + j < dwP) {
                    pfTmpB[dwLocalIdxN][dwLocalIdxM] = pfB[(dwLocalIdxN + j) * dwM + dwGlobalIdxM];
                }
                else {
                    pfTmpB[dwLocalIdxN][dwLocalIdxM] = 0;
                }
                __syncthreads();
                for (int i = 0; i < TRANS_BLOCK_DIM; i++) {
                    T fTmp;
                    fComp -= pfTmpA[dwLocalIdxN][i] * pfTmpB[i][dwLocalIdxM];
                    fTmp = fResults - fComp;
                    fComp = (fTmp - fResults) + fComp;
                    fResults = fTmp;
                }
                __syncthreads();
            }

            if (dwGlobalIdxM < dwM && dwGlobalIdxN < dwN) {
                pfC[dwGlobalIdxN * dwM + dwGlobalIdxM] = fResults;
            }

        }


        template <typename T>
        static void gpu_transpose_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                           const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                           Tensor &out, Stack &stack) {
            auto weight_shape = w.sizes();
            auto output_shape = out.sizes();
            auto x_shape = x.sizes();
            int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
            int conv_out_spatial_dim = x_shape[2] * x_shape[3];
            int output_number_offset = output_shape[1] * output_shape[2] * output_shape[3];
            int input_number_offset = x_shape[1] * conv_out_spatial_dim;
            int col_buffer_size = weight_shape[1] * weight_shape[2] * weight_shape[3] * x_shape[2] * x_shape[3];

            auto number = x_shape[0];
            auto input_channels = weight_shape[1];
            Size2D ksize(weight_shape[2], weight_shape[3]);
            Size2D input(output_shape[2], output_shape[3]);

            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();

            Tensor col_tensor;
            T *col_buffer = nullptr;

            bool is_1x1_conv = stride.height == 1 && stride.width == 1 &&
                               ksize.height == 1 && ksize.width == 1 &&
                               padding.top == 0 && padding.bottom == 0 &&
                               padding.left == 0 && padding.right == 0;

            Shape col_shape;
            col_shape.resize(1);
            col_shape[0] = col_buffer_size;
            col_tensor = stack.make(out.dtype(), col_shape, MemoryDevice(GPU));
            col_buffer = col_tensor.data<T>();
          
            int put_param = input_channels * output_shape[2]  * output_shape[3];

            auto &context = ctx::ref<DeviceContext>();
            auto* handle = reinterpret_cast<CUDAContextHandle*>(context.handle);

#ifndef TS_USE_CUBLAS
            int N = conv_out_spatial_dim;
            int M = kernel_dims;
            int K = x_shape[1];
            dim3 blocksize(CUDA_BLOCK(N, TRANS_BLOCK_DIM),CUDA_BLOCK(M, TRANS_BLOCK_DIM), 1);
            dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM,1);
#endif

            for (int i = 0; i < number; i++) {
            #ifdef TS_USE_CUBLAS
                auto cublas_handle = handle->cublas_handle();

                cublas::math<T>::gemm(cublas_handle, cublas::Trans, cublas::NoTrans,
                    kernel_dims, conv_out_spatial_dim, weight_shape[0], T(1.f), pweight, pinput, T(0.f), col_buffer);

            #else
                RUN_KERNEL(gpu_transpose_conv2d_run_kernel<T>, blocksize, threadsize,
                           pweight, pinput, col_buffer, M, N, K);
            #endif
 
               
                if (is_1x1_conv) {
                   memcpy((void*)poutput, out.device(), out.sizes().size() * sizeof(T),
                          (void*)col_buffer, col_tensor.device(), col_buffer_size * sizeof(T));

                } else {
                    RUN_KERNEL(gpu_col2im_kernel<T>, CUDA_BLOCK(put_param, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                               put_param, col_buffer, input.height, input.width,
                               ksize.height, ksize.width, padding.top, padding.left,
                               stride.height, stride.width, dilation.height, dilation.width,
                               x_shape[2], x_shape[3], poutput);
                }

                pinput += input_number_offset;
                poutput += output_number_offset;

            }

       }

       void Conv2DTransposeCore::conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value, 
                                             const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                             Conv2DFormat format, Tensor &out, Stack & stack) {

           if (format != FORMAT_NCHW) {
               TS_LOG_ERROR << "Conv2DTransposeCore only support NCHW" << eject;
           }
           DTYPE dtype = out.dtype();
           switch (dtype) {
           #define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
           case DTYPE: { gpu_transpose_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack);; break; }
#ifdef TS_USE_CUDA_FP16
           DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
           DECLARE_COMPUTE_RUN(FLOAT32, float);
           DECLARE_COMPUTE_RUN(FLOAT64, double);
           #undef DECLARE_COMPUTE_RUN
               default: {
                   TS_LOG_ERROR << "Conv2dTransposeCore not support data type(" << dtype << "): " << type_str(dtype) << eject;
                   break;
               }
          }
          return;
      }

   }

}
/////////////////////////////////////////////////

//using namespace ts;
//using namespace gpu;

//TS_REGISTER_OPERATOR(Transpose_Conv2D, GPU, std::string("transpose_conv2d"))

