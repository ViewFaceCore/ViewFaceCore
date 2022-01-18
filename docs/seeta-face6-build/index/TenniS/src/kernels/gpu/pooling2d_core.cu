#include <kernels/gpu/pooling2d_core.h>

#include "kernels/gpu/pooling2d_core.h"

#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {

        template<typename T>
        __global__ static void max_pooling_kernel(
            const T* input_data, T* output_data, int count, int out_channal,
            int out_height, int out_width, int input_channal, int input_height,
            int input_width, int kernel_height, int kernel_width, int stride_height,
            int stride_width, int pad_top, int pad_left) {
            
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            int batch_offset = out_channal * out_height * out_width;
            int channal_offset = out_height * out_width;
                
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                int n = index / batch_offset;
                int c = index % batch_offset / channal_offset;
                int out_row_index = index % channal_offset / out_width;
                int out_col_index = index % out_width;

                int input_row_start = out_row_index * stride_height - pad_top;
                int input_row_end = min(input_height, input_row_start + kernel_height);
                //int input_row_end = input_row_start + ksize.height < input_height ? input_row_start + ksize.height : input_height;
                int input_col_start = out_col_index * stride_width - pad_left;
                int input_col_end = min(input_col_start + kernel_width, input_width);
                //int input_col_end = input_col_start + ksize.width < input_width ? input_col_start + ksize.width : input_width;

                int index_row_start = max(input_row_start, 0);
                int index_col_start = max(input_col_start, 0);
                //int index_row_start = input_row_start > 0 ? input_row_start : 0;
                //int index_col_start = input_col_start > 0 ? input_col_start : 0;

                T max_val = input_data[((n * input_channal + c) * input_height + index_row_start) * input_width + index_col_start];
                for (int h = index_row_start; h < input_row_end; h++)
                {
                    for (int w = index_col_start; w < input_col_end; w++)
                    {
                        //max_val = max(max_val, input_data[((n * input_channal + c) * input_height + h) * input_width + w]);
                        T input_cur = input_data[((n * input_channal + c) * input_height + h) * input_width + w];
                        max_val = max_val > input_cur ? max_val : input_cur;
                    }
                }
                output_data[index] = max_val;
            }
        }

        template<typename T>
        __global__ static void average_pooling_kernel(
            const T* input_data, T* output_data, int count, int out_channal,
            int out_height, int out_width, int input_channal, int input_height,
            int input_width, int kernel_height,int kernel_width,int stride_height,
            int stride_width, int pad_top,int pad_left) {
            
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            int batch_offset = out_channal * out_height * out_width;
            int channal_offset = out_height * out_width;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                int n = index / batch_offset;
                int c = (index % batch_offset) / channal_offset;
                int out_row_index = (index % channal_offset) / out_width;
                int out_col_index = index % out_width;

                int input_row_start = out_row_index * stride_height - pad_top;
                int input_row_end = min(input_height, input_row_start + kernel_height);
                int input_col_start = out_col_index * stride_width - pad_left;
                int input_col_end = min(input_col_start + kernel_width, input_width);

                int index_row_start = max(input_row_start, 0);
                int index_col_start = max(input_col_start, 0);

                T sumValue = T(0.f);
                int count = 0;
                for (int h = index_row_start; h < input_row_end; h++)
                {
                    for (int w = index_col_start; w < input_col_end; w++)
                    {
                        sumValue += input_data[((n * input_channal + c) * input_height + h) * input_width + w];
                        count++;
                    }
                }
                if (count == 0)
                    output_data[index] = T(0.f);
                else
                    output_data[index] = sumValue / T((float)count);
            }
        }

        template<typename T>
        __global__ static void average_pooling_white_kernel(
                const T* input_data, T* output_data, int count, int out_channal,
                int out_height, int out_width, int input_channal, int input_height,
                int input_width, int kernel_height,int kernel_width,int stride_height,
                int stride_width, int pad_top,int pad_left) {

            int index = blockDim.x * blockIdx.x + threadIdx.x;

            int batch_offset = out_channal * out_height * out_width;
            int channal_offset = out_height * out_width;
            const int kernel_count = kernel_height * kernel_width;
            for (; index < count; index += blockDim.x * gridDim.x)
            {
                int n = index / batch_offset;
                int c = (index % batch_offset) / channal_offset;
                int out_row_index = (index % channal_offset) / out_width;
                int out_col_index = index % out_width;

                int input_row_start = out_row_index * stride_height - pad_top;
                int input_row_end = min(input_height, input_row_start + kernel_height);
                int input_col_start = out_col_index * stride_width - pad_left;
                int input_col_end = min(input_col_start + kernel_width, input_width);

                int index_row_start = max(input_row_start, 0);
                int index_col_start = max(input_col_start, 0);

                T sumValue = T(0.f);
                for (int h = index_row_start; h < input_row_end; h++)
                {
                    for (int w = index_col_start; w < input_col_end; w++)
                    {
                        sumValue += input_data[((n * input_channal + c) * input_height + h) * input_width + w];
                    }
                }
                output_data[index] = sumValue / T((float)kernel_count);
            }
        }

        template<typename T>
        static bool gpu_max_pooling(
            const T *input_data, T *output_data,
            const Shape &input_shape, const Shape &output_shape,
            const KSize2D &ksize, const Stride2D &stride,
            const Padding2D &m_padding) {
            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channal = input_shape[1];
            int output_h = output_shape[2];
            int output_w = output_shape[3];
            int batch_num = output_shape[0];
            int out_channal = output_shape[1];

            int count = batch_num * out_channal * output_h * output_w;
            
            dim3 block_size(512);
            dim3 grid_size((count + block_size.x - 1) / block_size.x);

            RUN_KERNEL(max_pooling_kernel<T>, grid_size, block_size,
                       input_data, output_data, count, out_channal, output_h, output_w, input_channal,
                       input_h, input_w, ksize.height, ksize.width, stride.height, stride.width,
                       m_padding.top, m_padding.left);
            return true;
        }

        template<typename T>
        static bool gpu_average_pooling(
            const T *input_data, T *output_data,
            const Shape &input_shape, const Shape &output_shape,
            const KSize2D &ksize, const Stride2D &stride,
            const Padding2D &m_padding) {
            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channal = input_shape[1];
            int output_h = output_shape[2];
            int output_w = output_shape[3];
            int batch_num = output_shape[0];
            int out_channal = output_shape[1];
            int count = batch_num * out_channal * output_h * output_w;

            dim3 block_size(512);
            dim3 grid_size((count + block_size.x - 1) / block_size.x);

            RUN_KERNEL(average_pooling_kernel<T>, grid_size, block_size,
                       input_data, output_data, count, out_channal, output_h, output_w, input_channal,
                       input_h, input_w, ksize.height, ksize.width, stride.height, stride.width,
                       m_padding.top, m_padding.left);
            return true;
        }

        template<typename T>
        static bool gpu_average_pooling_white(
                const T *input_data, T *output_data,
                const Shape &input_shape, const Shape &output_shape,
                const KSize2D &ksize, const Stride2D &stride,
                const Padding2D &m_padding) {
            int input_h = input_shape[2];
            int input_w = input_shape[3];
            int input_channal = input_shape[1];
            int output_h = output_shape[2];
            int output_w = output_shape[3];
            int batch_num = output_shape[0];
            int out_channal = output_shape[1];
            int count = batch_num * out_channal * output_h * output_w;

            dim3 block_size(512);
            dim3 grid_size((count + block_size.x - 1) / block_size.x);

            RUN_KERNEL(average_pooling_white_kernel<T>, grid_size, block_size,
                       input_data, output_data, count, out_channal, output_h, output_w, input_channal,
                       input_h, input_w, ksize.height, ksize.width, stride.height, stride.width,
                       m_padding.top, m_padding.left);
            return true;
        }

        template<typename T>
        static void cpu_pooling2d_compute_run(
            const Tensor &x, Pooling2DType type, const Padding2D &padding,
            Padding2DType padding_type, const Size2D &ksize, const Stride2D &stride,
            Conv2DFormat format, Tensor &out) {
#define ENCODE(a, b) ((int(a) << 8) | int(b))
            auto code = ENCODE(padding_type, type);
            static const auto BLACK_MAX = ENCODE(Padding2DType::BLACK, Pooling2DType::MAX);
            static const auto BLACK_AVG = ENCODE(Padding2DType::BLACK, Pooling2DType::AVG);
            static const auto WHITE_MAX = ENCODE(Padding2DType::WHITE, Pooling2DType::MAX);
            static const auto WHITE_AVG = ENCODE(Padding2DType::WHITE, Pooling2DType::AVG);
            if (code == BLACK_MAX) {
                gpu_max_pooling(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                    ksize, stride, padding);
            }
            else if (code == BLACK_AVG) {
                gpu_average_pooling(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                    ksize, stride, padding);
            } else if (code == WHITE_MAX) {
                gpu_max_pooling(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                    ksize, stride, padding);
            } else if (code == WHITE_AVG) {
                gpu_average_pooling_white(x.data<T>(), out.data<T>(), x.sizes(), out.sizes(),
                                    ksize, stride, padding);
            } else {
                TS_LOG_ERROR << "Pooling type only support MAX and AVG" << eject;
            }
#undef ENCODE
        }

        void Pooling2DCore::pooling2d(const Tensor &x, Pooling2DType type, const Padding2D &padding,
            Padding2DType padding_type, const Size2D &ksize, const Stride2D &stride,
            Conv2DFormat format, Tensor &out) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "Pooling2D only support NCHW" << eject;
            }
            if (padding_type != Padding2DType::BLACK && padding_type != Padding2DType::WHITE) {
                TS_LOG_ERROR << "Pooling2D only support black padding or white padding" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_pooling2d_compute_run<TYPE>(x, type, padding, padding_type, ksize, stride, format, out); break; }
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << "Pooling2D not support data type(" << dtype << "): " << type_str(dtype) << eject;
                break;
            }
            }
        }
    }
}