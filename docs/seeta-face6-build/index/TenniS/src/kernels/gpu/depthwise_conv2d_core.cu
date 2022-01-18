#include <kernels/gpu/depthwise_conv2d_core.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>

#include <kernels/gpu/operator_on_gpu.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {

        template <typename T>
        static __global__ void gpu_depthwise_conv2d_nchw_kernel(
                                int nthreads, const T* bottom_data, const T* weight_data, 
                                int num, int channels,
                                int top_height, int top_width, int bottom_height, int bottom_width,
                                int kernel_h,  int kernel_w,  int stride_h, int stride_w,
                                int pad_top, int pad_bottom, int pad_left, int pad_right, 
                                int dilation_h, int dilation_w, T* top_data) {

            for(int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
                const int n = index / channels / top_height / top_width;
                const int c = (index / top_height / top_width) % channels;
                const int h = (index / top_width) % top_height;
                const int w = index % top_width;
                const T* weight = weight_data + c * kernel_h * kernel_w;
                T value = 0;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int h_in = -pad_top + h * stride_h + kh * dilation_h;
                        const int w_in = -pad_left + w * stride_w + kw * dilation_w;
                        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
                            const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
                            value += (*weight) * bottom_data[offset];
                        }
                        ++weight;
                    }
                }
                top_data[index] = value;
            }
        }

#ifdef TS_USE_CUDA_FP16
        template <>
        __global__ void gpu_depthwise_conv2d_nchw_kernel<half>(
            int nthreads, const half* bottom_data, const half* weight_data,
            int num, int channels,
            int top_height, int top_width, int bottom_height, int bottom_width,
            int kernel_h, int kernel_w, int stride_h, int stride_w,
            int pad_top, int pad_bottom, int pad_left, int pad_right,
            int dilation_h, int dilation_w, half* top_data) {

            for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
                const int n = index / channels / top_height / top_width;
                const int c = (index / top_height / top_width) % channels;
                const int h = (index / top_width) % top_height;
                const int w = index % top_width;
                const half* weight = weight_data + c * kernel_h * kernel_w;
                half value = half(0.f);
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        const int h_in = -pad_top + h * stride_h + kh * dilation_h;
                        const int w_in = -pad_left + w * stride_w + kw * dilation_w;
                        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
                            const int offset = ((n * channels + c) * bottom_height + h_in) * bottom_width + w_in;
                            value = value + (*weight) * bottom_data[offset];
                        }
                        ++weight;
                    }
                }
                top_data[index] = value;
            }
        }
#endif

        template<typename T>
        static void gpu_depthwise_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                              const Tensor &weight, const Stride2D &stride, const Dilation2D &dilation,
                                              Tensor &out, Stack &stack) {
            auto weight_shape = weight.sizes();
            auto output_shape = out.sizes();
            auto input_shape = x.sizes();

            const T *pinput = x.data<T>();
            const T *pweight_base = weight.data<T>();
            T *poutput = out.data<T>();

            RUN_KERNEL(gpu_depthwise_conv2d_nchw_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       out.count(), pinput, pweight_base, output_shape[0], output_shape[1],
                       output_shape[2], output_shape[3], input_shape[2], input_shape[3],
                       weight_shape[2], weight_shape[3], stride.height, stride.width,
                       padding.top, padding.bottom, padding.left, padding.right,
                       dilation.height, dilation.width, poutput);
        }

        void
        DepthwiseConv2DCore::conv2d(const Tensor &x, const Padding2D &padding, float padding_value, const Tensor &w,
                                    const Stride2D &stride, const Dilation2D &dilation, Conv2DFormat format,
                                    Tensor &out, Stack &stack) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "DepthwiseConv2DCore Conv2D only support NCHW" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_depthwise_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack);; break; }
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "DepthwiseConv2DCore not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}
